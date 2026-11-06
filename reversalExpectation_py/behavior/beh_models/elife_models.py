"""
elife_models.py
===============
Python port of the eLife model family (Murphy et al. 2024), from the MATLAB
sources in sim/ (funWSLS.m, funQ_RPE.m, funFQ_RPE.m, funDFQ_RPE.m,
funFQ_RPE_CK.m, funDFQ_RPE_CK.m, funbelief.m, funbelief_CK.m,
funbelief_CK_bilateral.m), with numba kernels so models can be fitted and
cross-validated at scale.

Every model returns a PER-TRIAL log-likelihood (NaN on trials that do not
enter the likelihood), so the same code serves in-sample fitting and held-out
scoring of a subset of sessions.

Two data layouts
----------------
  "concat" : replicates bandit_fittingPerAnimal.m exactly. merge_sessions.m
             concatenates an animal's sessions with 20 NaN (missed) trials
             between consecutive sessions, so latent variables (values, belief,
             choice kernel) carry over from one session to the next. Use
             build_concat() to construct the sequence.
  "reset"  : every session starts from the initial latent state (as a
             per-session MATLAB fit would), and the log-likelihood is summed
             over sessions. Use build_reset(); session starts are passed to the
             kernels, which re-initialise there.

MATLAB behaviours that are reproduced on purpose (literal=True)
---------------------------------------------------------------
  * softmax probabilities of exactly 0 are replaced by realmin.
  * belief models: the trial after a missed trial is excluded from the
    likelihood (mu(k+1) is never set on a miss), and in belief_CK /
    belief_CK_bilateral the first trial is excluded too (mu(1) = NaN), while in
    belief it is included (mu(1) = 0.4).
  * funbelief.m and funbelief_CK.m do NOT renormalise the right-side
    likelihood before converting it back (p = 1 - pR on an unnormalised pR),
    so their Bayes update after a RIGHT choice is not the exact posterior.
    funbelief_CK_bilateral.m renormalises (exact). With literal=False the exact
    update is used for all belief models.
  * WSLS: first trial and trials after a miss get p(right) = 0.5.

Choice coding: c = +1 right, -1 left, NaN miss; r = 1 / 0, NaN on miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

REALMIN = 2.2250738585072014e-308
N_GAP = 20                                   # NaN trials inserted by merge_sessions.m


# ---------------------------------------------------------------------------
# Model registry (bounds and initial values from bandit_fittingPerAnimal.m)
# ---------------------------------------------------------------------------
@dataclass
class Model:
    name: str
    labels: list
    lb: list
    ub: list
    x0: list
    kind: str                                # wsls | q | belief
    opts: dict = field(default_factory=dict)

    @property
    def n_params(self):
        return len(self.labels)


MODELS = {
    "WSLS": Model("WSLS", ["p"], [0], [1], [0.8], "wsls"),
    "Q_RPE": Model("Q_RPE", ["alpha", "beta"], [0, 0], [1, 100], [0.3, 5], "q",
                   {"forget": 0, "ck": False}),
    "DFQ_RPE": Model("DFQ_RPE", ["alpha", "lambda", "beta"], [0, 0, 0], [1, 1, 100],
                     [0.3, 0.3, 5], "q", {"forget": 2, "ck": False}),
    "FQ_RPE": Model("FQ_RPE", ["alpha", "beta"], [0, 0], [1, 100], [0.3, 5], "q",
                    {"forget": 1, "ck": False}),
    "FQ_RPE_CK": Model("FQ_RPE_CK", ["alpha", "beta", "alpha_ck", "beta_ck"],
                       [0, 0, 0, 0], [1, 100, 1, 100], [0.3, 5, 0.2, 3], "q",
                       {"forget": 1, "ck": True}),
    "DFQ_RPE_CK": Model("DFQ_RPE_CK", ["alpha", "lambda", "beta", "alpha_ck", "beta_ck"],
                        [0, 0, 0, 0, 0], [1, 1, 100, 1, 100], [0.3, 0.3, 5, 0.2, 3], "q",
                        {"forget": 2, "ck": True}),
    "belief": Model("belief", ["H", "beta"], [0, 0], [1, 100], [0.1, 3], "belief",
                    {"ck": False, "bilateral": False, "mu1_defined": True,
                     "exact_right_update": False}),
    "belief_CK": Model("belief_CK", ["H", "beta", "alpha_ck", "beta_ck"],
                       [0, 0, 0, 0], [1, 100, 1, 100], [0.3, 5, 0.2, 3], "belief",
                       {"ck": True, "bilateral": False, "mu1_defined": False,
                        "exact_right_update": False}),
    "belief_CK_bilateral": Model(
        "belief_CK_bilateral", ["H_left", "H_right", "alpha_ck_left", "alpha_ck_right",
                                "beta", "beta_ck"],
        [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 100, 100], [0.3, 0.3, 0.2, 0.2, 5, 3], "belief",
        {"ck": True, "bilateral": True, "mu1_defined": False, "exact_right_update": True}),
    # --- variable hazard: H at the transition into trial k+1 = sigmoid(a + b*tau[k+1]),
    # tau = trials since criterion (0 during the criterion phase). Same kernel
    # and trial conventions as the eLife belief models, exact Bayes update;
    # b = 0 gives the constant-hazard model with H = sigmoid(a).
    "belief_vhr": Model("belief_vhr", ["a", "b", "beta"], [-12, 0, 0], [12, 5, 100],
                        [-2.2, 0.1, 5], "belief",
                        {"ck": False, "bilateral": False, "mu1_defined": True,
                         "exact_right_update": True, "vhr": True}),
    "belief_CK_vhr": Model("belief_CK_vhr", ["a", "b", "beta", "alpha_ck", "beta_ck"],
                           [-12, 0, 0, 0, 0], [12, 5, 100, 1, 100], [-2.2, 0.1, 5, 0.2, 3],
                           "belief",
                           {"ck": True, "bilateral": False, "mu1_defined": False,
                            "exact_right_update": True, "vhr": True}),
    # --- reviewer 3 non-anticipating choice-history agent: repeat the previous
    # choice with p_repeat; otherwise win-stay with p_ws after a reward /
    # lose-shift with p_ls after an omission; otherwise choose at random.
    # P(stay | reward)   = p_rep + (1 - p_rep) * (0.5 + 0.5 * p_ws)
    # P(stay | omission) = p_rep + (1 - p_rep) * (0.5 - 0.5 * p_ls)
    # (3 parameters for 2 observable probabilities: not identifiable, by construction)
    "R3_agent": Model("R3_agent", ["p_repeat", "p_ws", "p_ls"], [0, 0, 0], [1, 1, 1],
                      [0.85, 0.85, 0.85], "r3"),
}
# eLife belief models with the EXACT Bayes update (the MATLAB funbelief.m /
# funbelief_CK.m update after right choices is not exact; see module docstring)
MODELS["belief_exact"] = Model("belief_exact", ["H", "beta"], [0, 0], [1, 100], [0.1, 3],
                               "belief", {**MODELS["belief"].opts, "exact_right_update": True})
MODELS["belief_CK_exact"] = Model("belief_CK_exact", ["H", "beta", "alpha_ck", "beta_ck"],
                                  [0, 0, 0, 0], [1, 100, 1, 100], [0.3, 5, 0.2, 3], "belief",
                                  {**MODELS["belief_CK"].opts, "exact_right_update": True})
ELIFE8 = ["WSLS", "Q_RPE", "DFQ_RPE", "FQ_RPE", "FQ_RPE_CK", "DFQ_RPE_CK",
          "belief", "belief_CK"]


# ---------------------------------------------------------------------------
# Data layouts
# ---------------------------------------------------------------------------
def build_concat(sessions):
    """sessions: list of (c, r) arrays in temporal order. Returns (c, r, starts)
    with N_GAP NaN trials between sessions (merge_sessions.m) and the index of
    the first trial of each session."""
    cs, rs, starts, n = [], [], [], 0
    for i, (c, r) in enumerate(sessions):
        if i > 0:
            cs.append(np.full(N_GAP, np.nan))
            rs.append(np.full(N_GAP, np.nan))
            n += N_GAP
        starts.append(n)
        cs.append(np.asarray(c, float))
        rs.append(np.asarray(r, float))
        n += len(c)
    return np.concatenate(cs), np.concatenate(rs), np.asarray(starts, np.int64)


def build_reset(sessions):
    """Plain concatenation; latents re-initialise at every session start."""
    cs = [np.asarray(c, float) for c, _ in sessions]
    rs = [np.asarray(r, float) for _, r in sessions]
    starts = np.cumsum([0] + [len(c) for c in cs[:-1]]).astype(np.int64)
    return np.concatenate(cs), np.concatenate(rs), starts


def _reset_mask(n, starts, reset):
    m = np.zeros(n, np.bool_)
    if reset:
        m[np.asarray(starts, np.int64)] = True
    m[0] = True
    return m


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def _clamp(p):
    return REALMIN if p == 0.0 else p


@njit(cache=True)
def _softmax_right(v_right, v_left):
    """exp(V)/sum(exp(V)) for [right, left], as MATLAB computes it."""
    a, b = np.exp(v_right), np.exp(v_left)
    s = a + b
    return a / s, b / s


@njit(cache=True)
def _wsls_kernel(p, c, r, is_start):
    n = len(c)
    ll = np.full(n, np.nan)
    for k in range(n):
        if is_start[k]:
            pr = 0.5
        else:
            cp, rp = c[k - 1], r[k - 1]
            if rp > 0 and cp == 1:
                pr = p
            elif rp > 0 and cp == -1:
                pr = 1 - p
            elif rp == 0 and cp == 1:
                pr = 1 - p
            elif rp == 0 and cp == -1:
                pr = p
            else:
                pr = 0.5
        pl = 1 - pr
        pr, pl = _clamp(pr), _clamp(pl)
        if c[k] == 1:
            ll[k] = np.log(pr)
        elif c[k] == -1:
            ll[k] = np.log(pl)
    return ll


@njit(cache=True)
def _r3_kernel(p_rep, p_ws, p_ls, c, r, is_start):
    n = len(c)
    ll = np.full(n, np.nan)
    for k in range(n):
        if is_start[k] or np.isnan(c[k - 1]) or np.isnan(r[k - 1]):
            p_right = 0.5
        else:
            if r[k - 1] > 0:
                p_stay = p_rep + (1 - p_rep) * (0.5 + 0.5 * p_ws)
            else:
                p_stay = p_rep + (1 - p_rep) * (0.5 - 0.5 * p_ls)
            p_right = p_stay if c[k - 1] == 1 else 1 - p_stay
        pr, pl = _clamp(p_right), _clamp(1 - p_right)
        if c[k] == 1:
            ll[k] = np.log(pr)
        elif c[k] == -1:
            ll[k] = np.log(pl)
    return ll


@njit(cache=True)
def _q_kernel(alpha, lam, beta, a_ck, b_ck, forget, use_ck, c, r, is_start):
    """forget: 0 = Q_RPE (unchosen untouched), 1 = FQ (unchosen *(1-alpha)),
    2 = DFQ (unchosen *(1-lambda))."""
    n = len(c)
    ll = np.full(n, np.nan)
    vr = vl = 0.5
    ckr = ckl = 0.0
    for k in range(n):
        if is_start[k]:
            vr = vl = 0.5
            ckr = ckl = 0.0
        Vr = beta * vr + (b_ck * ckr if use_ck else 0.0)
        Vl = beta * vl + (b_ck * ckl if use_ck else 0.0)
        pr, pl = _softmax_right(Vr, Vl)
        pr, pl = _clamp(pr), _clamp(pl)
        ck = c[k]
        if ck == 1:
            ll[k] = np.log(pr)
        elif ck == -1:
            ll[k] = np.log(pl)
        if use_ck:
            if ck == 1:
                ckr = (1 - a_ck) * ckr + a_ck
                ckl = (1 - a_ck) * ckl
            elif ck == -1:
                ckr = (1 - a_ck) * ckr
                ckl = (1 - a_ck) * ckl + a_ck
        decay = 0.0 if forget == 0 else (alpha if forget == 1 else lam)
        if ck == 1:
            vr = vr + alpha * (r[k] - vr)
            if forget > 0:
                vl = (1 - decay) * vl
        elif ck == -1:
            vl = vl + alpha * (r[k] - vl)
            if forget > 0:
                vr = (1 - decay) * vr
    return ll


@njit(cache=True)
def _belief_kernel(HLs, HRs, aL, aR, beta, b_ck, use_ck, mu1_defined, exact_right,
                   c, r, is_start):
    """Generic belief model. p = [P(left = 0.1), P(left = 0.7)] over the two
    states ps = [0.1, 0.7]. HLs[k] / HRs[k] = hazard of the transition at the end
    of trial k after a left / right choice (constant for the eLife models, equal
    for the unilateral ones, H(tau[k+1]) for the variable-hazard models); the
    choice kernel learning rate may also depend on the side chosen."""
    n = len(c)
    ll = np.full(n, np.nan)
    ps0, ps1 = 0.1, 0.7
    pr0 = pr1 = 0.5                                   # cpPrior
    p0, p1 = 0.5, 0.5
    mu_l = np.full(n + 1, np.nan)
    mu_r = np.full(n + 1, np.nan)
    ckr = ckl = 0.0
    for k in range(n):
        if is_start[k]:
            p0, p1 = pr0, pr1
            ckr = ckl = 0.0
            if mu1_defined:
                mu_l[k] = ps0 * pr0 + ps1 * pr1
                mu_r[k] = ps0 * pr0 + ps1 * pr1
            else:
                mu_l[k] = np.nan
                mu_r[k] = np.nan
        Vr = beta * mu_r[k] + (b_ck * ckr if use_ck else 0.0)
        Vl = beta * mu_l[k] + (b_ck * ckl if use_ck else 0.0)
        prr, pll = _softmax_right(Vr, Vl)
        if prr == 0.0:
            prr = REALMIN
        if pll == 0.0:
            pll = REALMIN
        ch, out = c[k], r[k]
        if ch == 1:
            ll[k] = np.log(prr)                       # NaN if mu is NaN
        elif ch == -1:
            ll[k] = np.log(pll)
        if not np.isnan(ch) and not np.isnan(out):
            # ---- transition p(mu_{t+1} | mu_t, H)
            if ch == 1:
                H = HRs[k]
                q0, q1 = 1 - p0, 1 - p1
                q0 = q0 * (1 - H) + pr0 * H
                q1 = q1 * (1 - H) + pr1 * H
                s = q0 + q1
                q0, q1 = q0 / s, q1 / s
                p0, p1 = 1 - q0, 1 - q1
            else:
                H = HLs[k]
                p0 = p0 * (1 - H) + pr0 * H
                p1 = p1 * (1 - H) + pr1 * H
            s = p0 + p1
            p0, p1 = p0 / s, p1 / s
            # ---- likelihood of the outcome
            if ch == -1:
                if out == 1:
                    p0, p1 = p0 * ps0, p1 * ps1
                elif out == 0:
                    p0, p1 = p0 * (1 - ps0), p1 * (1 - ps1)
                s = p0 + p1
                p0, p1 = p0 / s, p1 / s
            else:
                q0, q1 = 1 - p0, 1 - p1
                if out == 1:
                    q0, q1 = q0 * ps0, q1 * ps1
                elif out == 0:
                    q0, q1 = q0 * (1 - ps0), q1 * (1 - ps1)
                if exact_right:
                    s = q0 + q1
                    q0, q1 = q0 / s, q1 / s
                p0, p1 = 1 - q0, 1 - q1
                s = p0 + p1
                p0, p1 = p0 / s, p1 / s
            mu_l[k + 1] = ps0 * p0 + ps1 * p1
            mu_r[k + 1] = ps0 * (1 - p0) + ps1 * (1 - p1)
        else:
            if k > 0 and not is_start[k]:
                mu_l[k] = mu_l[k - 1]
                mu_r[k] = mu_r[k - 1]
        # ---- choice kernel
        if use_ck:
            if ch == 1:
                ckr = (1 - aR) * ckr + aR
                ckl = (1 - aR) * ckl
            elif ch == -1:
                ckr = (1 - aL) * ckr
                ckl = (1 - aL) * ckl + aL
    return ll


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def trial_loglik(model, params, c, r, starts=None, reset=False, literal=True, tau=None):
    """Per-trial log-likelihood (NaN = not in the likelihood).

    model   : name in MODELS (or a Model)
    params  : parameter vector in the MATLAB order (Model.labels)
    starts  : session start indices; only used when reset=True
    tau     : trials since criterion per trial (variable-hazard models only)
    literal : reproduce the MATLAB belief update after right choices exactly
              (only affects belief / belief_CK)
    """
    m = MODELS[model] if isinstance(model, str) else model
    x = np.asarray(params, float)
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    is_start = _reset_mask(len(c), starts if starts is not None else [0], reset)
    if m.kind == "wsls":
        return _wsls_kernel(x[0], c, r, is_start)
    if m.kind == "r3":
        return _r3_kernel(x[0], x[1], x[2], c, r, is_start)
    if m.kind == "q":
        f = m.opts["forget"]
        if f == 2:
            alpha, lam, beta = x[0], x[1], x[2]
            rest = x[3:]
        else:
            alpha, lam, beta = x[0], 0.0, x[1]
            rest = x[2:]
        a_ck, b_ck = (rest[0], rest[1]) if m.opts["ck"] else (0.0, 0.0)
        return _q_kernel(alpha, lam, beta, a_ck, b_ck, f, m.opts["ck"], c, r, is_start)
    o = m.opts
    exact = o["exact_right_update"] or not literal
    n = len(c)
    if o.get("vhr"):
        if tau is None:
            raise ValueError(f"{m.name} needs tau")
        tau = np.asarray(tau, float)
        tau_next = np.r_[tau[1:], tau[-1:]]          # hazard of the step into k+1
        H = 1.0 / (1.0 + np.exp(-(x[0] + x[1] * tau_next)))
        beta = x[2]
        aL = aR = x[3] if o["ck"] else 0.0
        b_ck = x[4] if o["ck"] else 0.0
        return _belief_kernel(H, H, aL, aR, beta, b_ck, o["ck"], o["mu1_defined"], True,
                              c, r, is_start)
    if o["bilateral"]:
        HL, HR, aL, aR, beta, b_ck = x
    elif o["ck"]:
        HL = HR = x[0]
        beta, aL, b_ck = x[1], x[2], x[3]
        aR = aL
    else:
        HL = HR = x[0]
        beta, aL, aR, b_ck = x[1], 0.0, 0.0, 0.0
    return _belief_kernel(np.full(n, HL), np.full(n, HR), aL, aR, beta, b_ck, o["ck"],
                          o["mu1_defined"], exact, c, r, is_start)


def negloglike(model, params, c, r, starts=None, reset=False, literal=True, mask=None,
               tau=None):
    """Negative log-likelihood, optionally restricted to trials in `mask`."""
    ll = trial_loglik(model, params, c, r, starts, reset, literal, tau)
    if mask is not None:
        ll = np.where(mask, ll, np.nan)
    return float(-np.nansum(ll))


def n_scored(model, params, c, r, starts=None, reset=False, literal=True, mask=None,
             tau=None):
    ll = trial_loglik(model, params, c, r, starts, reset, literal, tau)
    ok = np.isfinite(ll)
    if mask is not None:
        ok &= mask
    return int(ok.sum())


# ---------------------------------------------------------------------------
# Fitting (MAP-free maximum likelihood, bounded, multi-start)
# ---------------------------------------------------------------------------
def fit(model, c, r, starts=None, reset=False, literal=True, mask=None,
        n_starts=10, seed=0, extra_x0=None, tau=None):
    """Maximum-likelihood fit with L-BFGS-B from the MATLAB initial values plus
    (n_starts - 1) random starts uniform within the bounds (beta-type
    parameters drawn on [0, 20] so random starts stay in a sensible range).

    Returns dict(params, negloglike, n_scored, bic, starts_agree, all_nll).
    BIC follows fit_fun.m: k*log(N) + 2*negloglike, N = responded trials.
    """
    from scipy.optimize import minimize
    m = MODELS[model] if isinstance(model, str) else model
    lb, ub = np.asarray(m.lb, float), np.asarray(m.ub, float)
    rng = np.random.default_rng(seed)
    ub_draw = np.where(ub > 1, np.minimum(ub, 20.0), ub)
    lb_draw = np.where(lb < 0, np.maximum(lb, -6.0), lb)      # vhr intercept a
    x0s = [np.asarray(m.x0, float)]
    if extra_x0 is not None:
        x0s += [np.clip(np.asarray(x, float), lb, ub) for x in extra_x0]
    while len(x0s) < n_starts:
        x0s.append(rng.uniform(lb_draw, ub_draw))

    def f(x):
        v = negloglike(m, x, c, r, starts, reset, literal, mask, tau)
        return v if np.isfinite(v) else 1e12

    res = []
    for x0 in x0s:
        o = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(lb, ub)),
                     options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        res.append((o.fun, o.x))
    res.sort(key=lambda t: t[0])
    best_nll, best_x = res[0]
    all_nll = np.array([v for v, _ in res])
    n_resp = int(np.sum(~np.isnan(c) if mask is None else (~np.isnan(c) & mask)))
    return {"params": best_x, "negloglike": float(best_nll),
            "n_scored": n_scored(m, best_x, c, r, starts, reset, literal, mask, tau),
            "bic": float(m.n_params * np.log(n_resp) + 2 * best_nll),
            "starts_agree": int(np.sum(all_nll - best_nll < 1e-3)),
            "n_starts": len(res), "all_nll": all_nll}
