"""
bayesian_models.py
==================
Bayesian belief models for two-armed bandit reversal task.
Port of funbelief.m / funbelief_CK.m (AC Kwan / H Atilgan lab).

Reference: Murphy et al. 2024 eLife, Figure 3, Methods pp. 21-22.

The belief transition (switch) step follows Murphy Eq. (7): the belief leaks
toward the uniform prior at rate H (reset-to-prior), NOT a symmetric two-state
flip. See `_transition` for details. The Bayes outcome update (Eq. 8), expected
reward (Eq. 9), softmax (Eqs. 3/5) and BIC (Eq. 10) match the paper as well.

Trial loop order also matches the paper: the choice is made on the belief (and
choice kernel) carried from the end of the previous trial; then, at the end of
the trial, the belief is transitioned (Eq. 7) and Bayes-updated with the
outcome (Eq. 8), and the choice kernel is updated. Fitting uses BADS (Acerbi &
Ma 2017) by default, as in Murphy et al.

Models
------
belief      (2 params): H, beta
belief_CK   (4 params): H, beta, alpha_k, beta_k

Encoding convention (same as the rest of the pipeline)
-------------------------------------------------------
choice  : -1 = left,  1 = right,  NaN = miss
reward  :  1 = rewarded,  0 = not rewarded,  NaN = miss
rule    :  integer index mapping into REWARD_PROBS

Likelihood layout
-----------------
The per-trial log-likelihood is computed by `belief_trial_loglikes` /
`belief_ck_trial_loglikes`, which propagate the latent state (belief, and the
choice kernel for CK) through *every* trial and return one log-likelihood per
trial (NaN on miss). `belief_negloglike` / `belief_ck_negloglike` are thin
wrappers that sum those values, optionally restricted to a subset of trials via
`score_mask`. This keeps in-sample fitting identical to before while letting the
cross-validation code score a held-out subset without breaking the sequential
state propagation.
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid

from preprocessing.presentation_codes import REWARD_PROBS

# BADS (Bayesian Adaptive Direct Search; Acerbi & Ma 2017) is the optimiser
# Murphy et al. used. PyBADS is its Python port. Imported lazily so the module
# still loads in environments without it (use optimizer="lbfgs" there).
try:
    from pybads import BADS as _BADS
    _HAVE_BADS = True
except Exception:                       # pragma: no cover
    _BADS = None
    _HAVE_BADS = False


import os
import contextlib

def _run_bads(fun, x0, lb, ub, plb, pub):
    """
    Minimise `fun` with PyBADS from start point x0 (Murphy et al.'s optimiser).

    lb/ub are hard bounds; plb/pub are plausible bounds. x0 is clipped strictly
    inside the plausible box to avoid pybads start-point corrections. PyBADS'
    console messages are silenced (display is off and stdout/stderr are
    redirected during the run); exceptions still propagate. Returns (x_best, fval).
    """
    if not _HAVE_BADS:
        raise ImportError(
            "pybads is not installed. Install it with `pip install pybads`, "
            "or pass optimizer='lbfgs' to use scipy L-BFGS-B instead."
        )
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    plb, pub = np.asarray(plb, float), np.asarray(pub, float)
    margin = 1e-3 * (pub - plb)
    x0 = np.clip(np.asarray(x0, float), plb + margin, pub - margin)
    options = {"display": "off", "uncertainty_handling": False}
    with open(os.devnull, "w") as _dev, \
            contextlib.redirect_stdout(_dev), contextlib.redirect_stderr(_dev), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bads = _BADS(fun, x0, lb, ub, plb, pub, options=options)
        res = bads.optimize()
    return np.asarray(res["x"], float).ravel(), float(res["fval"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_reward_probs(n_rules: int):
    """Return {rule_idx: (p_left, p_right)} for this task variant."""
    return REWARD_PROBS.get(n_rules, REWARD_PROBS[2])


def _transition(b: float, H: float, prior: float = 0.5) -> float:
    """
    Belief transition (switch) step. Matches Murphy et al. 2024, Eq. (7):

        rho' propto rho * (1 - H) + prior * H

    i.e. the belief leaks toward the uniform prior (prior = 0.5) at rate H.
    The two states rho_L70 and rho_L10 each get a `prior * H` term, so the
    result is already normalised (retained terms sum to 1 - H, leak terms sum
    to H). This is a "reset-to-prior" jump process: with probability H the
    hidden state is redrawn from the prior; with probability 1 - H it persists.

    NOTE: this is NOT the symmetric two-state flip `b*(1-H) + (1-b)*H`. Under
    the flip form the same H decays toward 0.5 about twice as fast, which biases
    the fitted H downward relative to the values reported in Murphy et al.
    (their best-fit H ~= 0.32). `b = prior` is the fixed point of both forms.
    """
    return b * (1.0 - H) + prior * H


def _belief_update(b_hat: float, choice: float, reward: float,
                   p_high: float = 0.70, p_low: float = 0.10) -> float:
    """
    Bayes update of belief b_hat = P(high-left state) after observing
    (choice, reward).  Returns unchanged b_hat on miss.
    """
    if np.isnan(choice) or np.isnan(reward):
        return b_hat

    if choice == -1:                     # chose left
        L1 = p_high if reward else (1.0 - p_high)   # high-left state
        L2 = p_low  if reward else (1.0 - p_low)    # high-right state
    else:                                # chose right
        L1 = p_low  if reward else (1.0 - p_low)
        L2 = p_high if reward else (1.0 - p_high)

    denom = b_hat * L1 + (1.0 - b_hat) * L2
    if denom < 1e-300:
        return b_hat
    return b_hat * L1 / denom


def _q_diff(b_hat: float, p_high: float = 0.70, p_low: float = 0.10) -> float:
    """Expected-reward difference: Q(left) - Q(right) given belief b_hat."""
    q_left  = p_high * b_hat + p_low  * (1.0 - b_hat)
    q_right = p_low  * b_hat + p_high * (1.0 - b_hat)
    return q_left - q_right


def _reward_probs_for_n_rules(n_rules: int):
    """Return (p_high, p_low) from the first entry in REWARD_PROBS[n_rules]."""
    probs = REWARD_PROBS.get(n_rules, REWARD_PROBS[2])
    # p_high is the larger of the two reward probs in rule 1
    lp, rp = probs[1]
    return max(lp, rp), min(lp, rp)


# ---------------------------------------------------------------------------
# Per-trial log-likelihoods (latent state propagates through ALL trials)
# ---------------------------------------------------------------------------

def belief_trial_loglikes(params, c, r, n_rules: int = 2) -> np.ndarray:
    """
    Per-trial log-likelihood of the observed choice under the belief model.

    The belief propagates through every trial (including misses, which leave it
    unchanged). The returned array has one entry per trial: log P(observed
    choice) on choice trials, NaN on misses. Scoring a subset of trials is the
    caller's job (e.g. via np.nansum over a mask) — this function never resets
    the latent state, so a warm-up segment can precede the scored trials.

    Parameters
    ----------
    params   : [H, beta]
    c        : (n_trials,) choices  (-1/1/NaN)
    r        : (n_trials,) rewards  (0/1/NaN)
    n_rules  : number of rules / hidden states (default 2)
    """
    H, beta = params
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    c = np.asarray(c, float)
    r = np.asarray(r, float)

    b  = 0.5
    ll = np.full(len(c), np.nan)
    for t in range(len(c)):
        # Decision uses the belief carried from the end of the previous trial
        # (Murphy et al.: action selection on the current belief, then the
        # end-of-trial update). For t = 0 this is the uniform prior.
        p_l = expit(beta * _q_diff(b, p_high, p_low))

        ct = c[t]
        if not np.isnan(ct):
            p_chosen = p_l if ct == -1 else (1.0 - p_l)
            ll[t] = np.log(np.clip(p_chosen, 1e-15, 1.0 - 1e-15))

        # End-of-trial belief update, Murphy Eq. (7) then Eq. (8): first the
        # switch transition, then the Bayesian outcome inference. On a miss
        # _belief_update returns the transitioned belief unchanged.
        b_pred = _transition(b, H)
        b = _belief_update(b_pred, ct, r[t], p_high, p_low)

    return ll


def belief_ck_trial_loglikes(params, c, r, n_rules: int = 2) -> np.ndarray:
    """
    Per-trial log-likelihood under the belief + choice-kernel model.

    Both the belief and the choice kernel propagate through every trial (the
    kernel only updates on choice trials, matching the original loop). Returns
    one log-likelihood per trial (NaN on miss).

    Parameters
    ----------
    params   : [H, beta, alpha_k, beta_k]
    c        : (n_trials,) choices
    r        : (n_trials,) rewards
    n_rules  : number of rules (default 2)
    """
    H, beta, alpha_k, beta_k = params
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    c = np.asarray(c, float)
    r = np.asarray(r, float)

    b  = 0.5
    ck = np.zeros(2)   # [CK_left, CK_right]
    ll = np.full(len(c), np.nan)
    for t in range(len(c)):
        # Decision uses the belief and choice kernel carried from the end of the
        # previous trial (Murphy order).
        qdiff   = _q_diff(b, p_high, p_low)
        ck_diff = ck[0] - ck[1]
        p_l     = expit(beta * qdiff + beta_k * ck_diff)

        ct = c[t]
        if not np.isnan(ct):
            p_chosen = p_l if ct == -1 else (1.0 - p_l)
            ll[t] = np.log(np.clip(p_chosen, 1e-15, 1.0 - 1e-15))

        # End-of-trial belief update: transition (Eq. 7) then Bayes (Eq. 8).
        b_pred = _transition(b, H)
        b = _belief_update(b_pred, ct, r[t], p_high, p_low)

        # End-of-trial choice-kernel update (only on choice trials).
        if not np.isnan(ct):
            if ct == -1:                       # chose left
                ck[0] = ck[0] + alpha_k * (1.0 - ck[0])
                ck[1] = ck[1] * (1.0 - alpha_k)
            else:                              # chose right
                ck[1] = ck[1] + alpha_k * (1.0 - ck[1])
                ck[0] = ck[0] * (1.0 - alpha_k)

    return ll


# ---------------------------------------------------------------------------
# Negative log-likelihoods (thin wrappers; optional held-out scoring)
# ---------------------------------------------------------------------------

def belief_negloglike(params, c, r, n_rules: int = 2, score_mask=None) -> float:
    """
    Negative log-likelihood for the belief model.

    With score_mask=None this is identical to summing the loss over all choice
    trials (the original behaviour). When a boolean score_mask is given, only
    the trials where the mask is True contribute to the sum, while the latent
    state still propagates through every trial.

    Parameters
    ----------
    params     : [H, beta]
    c, r       : choices / rewards
    n_rules    : number of rules (default 2)
    score_mask : optional (n_trials,) bool array selecting scored trials
    """
    H, beta = params
    if H <= 0 or H >= 1 or beta <= 0:
        return 1e9

    ll = belief_trial_loglikes(params, c, r, n_rules)
    if score_mask is not None:
        ll = ll[np.asarray(score_mask, bool)]
    return float(-np.nansum(ll))


def belief_ck_negloglike(params, c, r, n_rules: int = 2, score_mask=None) -> float:
    """
    Negative log-likelihood for the belief + choice-kernel model.

    See `belief_negloglike` for the score_mask semantics.

    Parameters
    ----------
    params     : [H, beta, alpha_k, beta_k]
    c, r       : choices / rewards
    n_rules    : number of rules (default 2)
    score_mask : optional (n_trials,) bool array selecting scored trials
    """
    H, beta, alpha_k, beta_k = params
    if H <= 0 or H >= 1 or beta <= 0 or alpha_k < 0 or alpha_k > 1 or beta_k <= 0:
        return 1e9

    ll = belief_ck_trial_loglikes(params, c, r, n_rules)
    if score_mask is not None:
        ll = ll[np.asarray(score_mask, bool)]
    return float(-np.nansum(ll))


# ---------------------------------------------------------------------------
# Model fitting (BADS, as in Murphy et al.; L-BFGS-B available as fallback)
# ---------------------------------------------------------------------------

def fit_belief(c, r, n_rules: int = 2, n_restarts: int = 5,
               rng=None, score_mask=None, optimizer: str = "bads") -> dict:
    """
    Fit the belief model to observed choices and rewards.

    optimizer : "bads" (default; PyBADS, as in Murphy et al. 2024) or "lbfgs"
    (scipy L-BFGS-B fallback for environments without pybads). The first
    restart starts from Murphy's published initial values (H=0.1, beta=5); any
    further restarts start from random points and the best fit is kept.

    With a boolean score_mask the negative log-likelihood is minimised over the
    scored trials only, while the latent state still propagates through every
    trial (cross-validation). The returned BIC / nlike use the scored trials.

    Returns
    -------
    dict with keys: model, fitpar, negloglike, bic, nlike
        fitpar = [H, beta]
    """
    if rng is None:
        rng = np.random.default_rng()
    c, r = np.asarray(c, float), np.asarray(r, float)
    if score_mask is None:
        n_obs = int(np.sum(~np.isnan(c)))
    else:
        n_obs = int(np.sum(~np.isnan(c) & np.asarray(score_mask, bool)))

    # Murphy bounds: 0..1 for H, 0..100 for the inverse temperature.
    lb, ub   = [0.0, 0.0],   [1.0, 100.0]
    plb, pub = [0.02, 1.0],  [0.7, 20.0]
    fun = lambda x: belief_negloglike(x, c, r, n_rules, score_mask)
    murphy_x0 = [0.1, 5.0]   # Murphy et al. 2024 published initial values

    best_nll, best_par = np.inf, None
    for i in range(n_restarts):
        x0 = murphy_x0 if i == 0 else [rng.uniform(0.01, 0.49), rng.uniform(0.5, 15.0)]
        if optimizer == "bads":
            par, nll = _run_bads(fun, x0, lb, ub, plb, pub)
        else:
            res = minimize(belief_negloglike, x0, args=(c, r, n_rules, score_mask),
                           method="L-BFGS-B", bounds=[(1e-6, 1.0 - 1e-6), (1e-3, 100.0)])
            par, nll = res.x, res.fun
        if nll < best_nll:
            best_nll, best_par = nll, par

    bic = 2.0 * best_nll + 2.0 * np.log(max(n_obs, 1))
    return {
        "model":      "belief",
        "fitpar":     best_par,
        "negloglike": best_nll,
        "bic":        bic,
        "nlike":      np.exp(-best_nll / max(n_obs, 1)),
    }


def fit_belief_ck(c, r, n_rules: int = 2, n_restarts: int = 5,
                  rng=None, score_mask=None, optimizer: str = "bads") -> dict:
    """
    Fit the belief-CK model to observed choices and rewards.

    See `fit_belief` for the optimizer / score_mask semantics. The first restart
    starts from Murphy's published initial values (H=0.1, beta=5, alpha_k=0.2,
    beta_k=5).

    Returns
    -------
    dict with keys: model, fitpar, negloglike, bic, nlike
        fitpar = [H, beta, alpha_k, beta_k]
    """
    if rng is None:
        rng = np.random.default_rng()
    c, r = np.asarray(c, float), np.asarray(r, float)
    if score_mask is None:
        n_obs = int(np.sum(~np.isnan(c)))
    else:
        n_obs = int(np.sum(~np.isnan(c) & np.asarray(score_mask, bool)))

    # Murphy bounds: 0..1 for H and alpha_k, 0..100 for the inverse temperatures.
    lb  = [0.0, 0.0, 0.0, 0.0]
    ub  = [1.0, 100.0, 1.0, 100.0]
    plb = [0.02, 1.0, 0.02, 1.0]
    pub = [0.7, 20.0, 0.9, 20.0]
    fun = lambda x: belief_ck_negloglike(x, c, r, n_rules, score_mask)
    murphy_x0 = [0.1, 5.0, 0.2, 5.0]   # Murphy et al. 2024 published initial values

    best_nll, best_par = np.inf, None
    for i in range(n_restarts):
        x0 = murphy_x0 if i == 0 else [
            rng.uniform(0.01, 0.49),
            rng.uniform(0.5,  15.0),
            rng.uniform(0.01, 0.49),
            rng.uniform(0.5,  10.0),
        ]
        if optimizer == "bads":
            par, nll = _run_bads(fun, x0, lb, ub, plb, pub)
        else:
            res = minimize(belief_ck_negloglike, x0, args=(c, r, n_rules, score_mask),
                           method="L-BFGS-B",
                           bounds=[(1e-6, 1.0 - 1e-6), (1e-3, 100.0),
                                   (1e-6, 1.0 - 1e-6), (1e-3, 100.0)])
            par, nll = res.x, res.fun
        if nll < best_nll:
            best_nll, best_par = nll, par

    bic = 2.0 * best_nll + 4.0 * np.log(max(n_obs, 1))
    return {
        "model":      "belief_ck",
        "fitpar":     best_par,
        "negloglike": best_nll,
        "bic":        bic,
        "nlike":      np.exp(-best_nll / max(n_obs, 1)),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_belief(c_real, r_real, rule, H: float, beta: float,
                    n_sims: int = 100, rng=None) -> dict:
    """
    Simulate choices under the belief model.

    On each simulation the model uses:
    - the same miss-trial mask as the real session
    - the true block rules to draw synthetic rewards

    Parameters
    ----------
    c_real  : (n_trials,) real choices — only used for miss masking
    r_real  : (n_trials,) real rewards — not used in simulation
    rule    : (n_trials,) true rule index per trial
    H, beta : model parameters
    n_sims  : number of independent repeats
    rng     : np.random.Generator

    Returns
    -------
    dict with keys:
        c_sim   : (n_trials, n_sims)  simulated choices (-1 / 1)
        belief  : (n_trials, n_sims)  P(high-left state) after transition, before update
        p_left  : (n_trials, n_sims)  P(choose left) on each trial
    """
    if rng is None:
        rng = np.random.default_rng()

    c_real = np.asarray(c_real, float)
    rule   = np.asarray(rule,   float)
    n      = len(c_real)
    miss   = np.isnan(c_real)

    # Determine p_high/p_low from the rules present
    unique_rules = rule[~np.isnan(rule)]
    n_rules = 2 if unique_rules.max() <= 2 else int(unique_rules.max())
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    prob_map = _get_reward_probs(n_rules)

    c_sim  = np.full((n, n_sims), np.nan)
    belief = np.full((n, n_sims), np.nan)
    p_left = np.full((n, n_sims), np.nan)

    for s in range(n_sims):
        b = 0.5
        for t in range(n):
            # Decision uses the belief carried into the trial (Murphy order);
            # belief[t,s] is the decision-time belief, not a post-transition one.
            belief[t, s] = b

            if miss[t]:
                b = _transition(b, H)   # no observation → transition only
                continue

            qdiff      = _q_diff(b, p_high, p_low)
            pl         = expit(beta * qdiff)
            p_left[t, s] = pl

            ct = -1.0 if rng.random() < pl else 1.0
            c_sim[t, s] = ct

            # Synthetic reward from true block rule
            r_idx = rule[t]
            if not np.isnan(r_idx):
                lp, rp = prob_map.get(int(r_idx), (0.5, 0.5))
                prew   = lp if ct == -1 else rp
                rt     = 1.0 if rng.random() < prew else 0.0
            else:
                rt = np.nan

            # End-of-trial update: transition (Eq. 7) then Bayes (Eq. 8).
            b = _belief_update(_transition(b, H), ct, rt, p_high, p_low)

    return {"c_sim": c_sim, "belief": belief, "p_left": p_left}


def simulate_belief_ck(c_real, r_real, rule,
                       H: float, beta: float,
                       alpha_k: float, beta_k: float,
                       n_sims: int = 100, rng=None) -> dict:
    """
    Simulate choices under the belief + choice-kernel model.

    Parameters
    ----------
    c_real, r_real, rule : see simulate_belief
    H, beta, alpha_k, beta_k : model parameters
    n_sims, rng : see simulate_belief

    Returns
    -------
    dict with keys: c_sim, belief, p_left, ck_diff
        ck_diff : (n_trials, n_sims)  CK_left - CK_right at decision time
    """
    if rng is None:
        rng = np.random.default_rng()

    c_real = np.asarray(c_real, float)
    rule   = np.asarray(rule,   float)
    n      = len(c_real)
    miss   = np.isnan(c_real)

    unique_rules = rule[~np.isnan(rule)]
    n_rules = 2 if (len(unique_rules) == 0 or unique_rules.max() <= 2) else int(unique_rules.max())
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    prob_map = _get_reward_probs(n_rules)

    c_sim   = np.full((n, n_sims), np.nan)
    belief  = np.full((n, n_sims), np.nan)
    p_left  = np.full((n, n_sims), np.nan)
    ck_diff = np.full((n, n_sims), np.nan)

    for s in range(n_sims):
        b  = 0.5
        ck = np.zeros(2)   # [CK_left, CK_right]

        for t in range(n):
            # Decision uses belief and choice kernel carried into the trial.
            belief[t, s] = b

            if miss[t]:
                b = _transition(b, H)   # transition only on misses
                continue

            qdiff       = _q_diff(b, p_high, p_low)
            ck_d        = ck[0] - ck[1]
            logit       = beta * qdiff + beta_k * ck_d
            pl          = expit(logit)

            p_left[t, s]  = pl
            ck_diff[t, s] = ck_d

            ct = -1.0 if rng.random() < pl else 1.0
            c_sim[t, s] = ct

            r_idx = rule[t]
            if not np.isnan(r_idx):
                lp, rp = prob_map.get(int(r_idx), (0.5, 0.5))
                prew   = lp if ct == -1 else rp
                rt     = 1.0 if rng.random() < prew else 0.0
            else:
                rt = np.nan

            # End-of-trial updates: belief transition+Bayes, then choice kernel.
            b = _belief_update(_transition(b, H), ct, rt, p_high, p_low)

            if ct == -1:
                ck[0] = ck[0] + alpha_k * (1.0 - ck[0])
                ck[1] = ck[1] * (1.0 - alpha_k)
            else:
                ck[1] = ck[1] + alpha_k * (1.0 - ck[1])
                ck[0] = ck[0] * (1.0 - alpha_k)

    return {"c_sim": c_sim, "belief": belief, "p_left": p_left, "ck_diff": ck_diff}