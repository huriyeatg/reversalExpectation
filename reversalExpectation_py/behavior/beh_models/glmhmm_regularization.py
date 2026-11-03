"""
glmhmm_regularization.py
========================
Sensitivity of the K=2 `ashwood_wsls` GLM-HMM to the strength of the Gaussian
(MAP) prior on the GLM weights.

Why: the exploit state's weights are huge (prev_choice ~10.4, wsls ~7.1),
which makes it near-deterministic (P(stay | reward) = 1 - 3e-8). With
prior_sigma = 100 the prior does practically nothing. If a tighter prior
(sigma = 5, 2, 1, ...) shrinks the weights WITHOUT losing held-out
log-likelihood, the large weights are a quasi-separation artefact, not
something the data require. We also track how the state interpretation moves:
P(stay) after reward / omission per state, dwell times, occupancy, and which
state absorbs the switch trials.

Implementation
--------------
A self-contained numpy/scipy EM for the input-driven Bernoulli HMM, so it does
NOT need the `ssm` package (runs in revExp, ssm, or any env with numpy, scipy,
pandas). It mirrors the model fitted with zashwood/ssm in hmmGlm.py:

  * observations : P(right | z=k, x_t) = sigmoid(w_k . x_t),
                   x_t = [1, prev_choice (+-1), wsls (+-1)]   (hmmGlm design)
  * prior        : w_k ~ N(0, prior_sigma^2 I)  -> MAP M-step for the weights
  * transitions  : stationary K x K Markov matrix; Dirichlet alpha = 1,
                   kappa = 0 -> plain MLE (as TRANSITION_ALPHA/KAPPA in hmmGlm)
  * pi0          : one initial-state distribution shared by all sessions
  * sequences    : every SESSION is its own sequence starting from pi0
  * misses       : masked (likelihood 1), never dropped
  * init         : Ashwood hierarchical recipe -- fit a 1-state GLM, then
                   each restart starts the K states at GLM weights + N(0, 0.2)
                   noise and a sticky transition matrix (0.95 self)
  * convergence  : stop when the per-trial log-posterior improves by < 1e-7
                   (about 0.03 nats in total) or after MAX_ITERS iterations
States are sorted so state 0 = exploit (largest prev_choice weight).

Sessions are padded to a common length and the forward-backward runs
vectorised over all sessions at once (padded steps have likelihood 1 and are
excluded from every sufficient statistic), which makes one EM iteration take
well under a second on the full dataset.

Validation: with prior_sigma = 100 the fit must reproduce the saved ssm model
(analysis/glmhmm_model_ashwood_wsls_K2.pkl: exploit 10.38 / 7.12). The script
prints that comparison first.

Cross-validation: 5 folds, sessions split by hmmGlm._session_folds (stratified
by animal, seed 0) -- the same split used by cross_validate_glmhmm -- and the
held-out log-likelihood per observed trial is reported per sigma.

Inputs  (relative to reversalExpectation_py/):
    analysis/bandit_R71_lesion.csv
Outputs (analysis/glmhmm_regularization/):
    fits_full_data.csv        weights, transitions, dwell, occupancy, train LL per sigma
    pstay_by_sigma.csv        model P(stay) after reward / omission per state and sigma
    switch_attribution.csv    share of switch trials assigned to explore, per sigma
    cv_heldout.csv            held-out LL per trial per (sigma, fold)
    cv_summary.csv            mean +- SEM over folds, and paired difference vs sigma=100
    checkpoint.csv            per-(sigma, fold) progress, so an interrupted run resumes
    report.txt                human-readable summary

Usage (from reversalExpectation_py/):
    python behavior/beh_models/glmhmm_regularization.py
    python behavior/beh_models/glmhmm_regularization.py --sigmas 100 5 2 1 0.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent              # .../behavior/beh_models
REPO_ROOT = HERE.parents[1]                         # .../reversalExpectation_py
CSV_PATH = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_DIR = REPO_ROOT / "analysis" / "glmhmm_regularization"

K = 2
SIGMAS = [100.0, 5.0, 2.0, 1.0, 0.5]
N_RESTARTS_FULL = 5           # restarts for the full-data fit of each sigma
N_RESTARTS_CV = 3             # restarts for each CV training fit
N_FOLDS = 5
FOLD_SEED = 0                 # same as cross_validate_glmhmm(seed=0)
INIT_NOISE = 0.2              # hmmGlm.INIT_NOISE
STICKY_SELF = 0.95            # hmmGlm.STICKY_SELF
MAX_ITERS = 300
TOL_PER_TRIAL = 1e-7
MIN_TRIALS = 20               # hmmGlm.build_glmhmm_inputs(min_trials=20)
REGRESSORS = ["bias", "prev_choice", "wsls"]
LABELS = ["exploit", "explore"]

# Weights of the saved ssm fit (prior_sigma = 100), for the validation line.
SSM_REFERENCE = np.array([[-0.0463, 10.3782, 7.1192],
                          [-0.0108, 0.1842, 0.2881]])


# ---------------------------------------------------------------------------
# 1. Data -> padded arrays (design identical to hmmGlm.build_session_arrays)
# ---------------------------------------------------------------------------
def session_design(df_ses: pd.DataFrame):
    """Return (y, observed, X) for one session (ashwood_wsls design)."""
    c = df_ses["choice"].to_numpy(dtype=float)                 # -1 / +1 / NaN
    r = np.nan_to_num(df_ses["rewarded"].to_numpy(dtype=float), nan=0.0)
    observed = ~np.isnan(c)
    signed = np.where(observed, c, 0.0)
    T = len(c)
    filled_c, filled_r = signed.copy(), r.copy()
    last_c, last_r = 0.0, 0.0
    for t in range(T):                                         # forward-fill misses
        if observed[t]:
            last_c, last_r = signed[t], r[t]
        else:
            filled_c[t], filled_r[t] = last_c, last_r
    prev_c = np.zeros(T)
    prev_r = np.zeros(T)
    prev_c[1:], prev_r[1:] = filled_c[:-1], filled_r[:-1]
    rs = 2.0 * prev_r - 1.0
    rs[prev_c == 0] = 0.0                                      # no history -> 0
    X = np.column_stack([np.ones(T), prev_c, rs * prev_c])
    y = np.zeros(T)
    y[observed] = (c[observed] > 0).astype(float)              # 1 = right
    return y, observed, X


def load_sessions(csv_path: Path):
    """Select naive + meets_criteria sessions and build per-session arrays."""
    cols = ["animal", "session_file", "choice", "rewarded", "lesioned", "meets_criteria"]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df[df["lesioned"].isna() & (df["meets_criteria"] == True)]   # noqa: E712
    sessions, tags = [], []
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        y, obs, X = session_design(d)
        if obs.sum() < MIN_TRIALS:
            continue
        sessions.append((y, obs, X))
        tags.append((animal, ses))
    n_obs = sum(int(s[1].sum()) for s in sessions)
    print(f"[data] {len(sessions)} sessions, {len({t[0] for t in tags})} animals, "
          f"{n_obs} observed trials")
    return sessions, tags


class Batch:
    """Sessions padded to a common length T_max (S x T_max arrays)."""

    def __init__(self, sessions):
        S = len(sessions)
        T = max(len(s[0]) for s in sessions)
        self.y = np.zeros((S, T))
        self.obs = np.zeros((S, T), dtype=bool)                # observed choice
        self.valid = np.zeros((S, T), dtype=bool)              # real (non-padded) trial
        self.X = np.zeros((S, T, 3))
        for i, (y, obs, X) in enumerate(sessions):
            n = len(y)
            self.y[i, :n], self.obs[i, :n], self.valid[i, :n], self.X[i, :n] = y, obs, True, X
        self.n_obs = int(self.obs.sum())
        # flattened observed trials, used by the weight M-step
        self.Xo = self.X[self.obs]                             # (N, 3)
        self.yo = self.y[self.obs]                             # (N,)


def split_sessions(sessions, idx):
    return [sessions[i] for i in idx]


def session_folds(animals, n_folds, seed):
    """Copy of hmmGlm._session_folds: CV folds stratified by animal."""
    animals = np.asarray(animals, dtype=object)
    rng = np.random.default_rng(seed)
    fold_assign = np.full(len(animals), -1, dtype=int)
    for a in pd.unique(animals):
        idx = rng.permutation(np.where(animals == a)[0])
        for f, chunk in enumerate(np.array_split(idx, n_folds)):
            fold_assign[chunk] = f
    return [np.where(fold_assign == f)[0] for f in range(n_folds)]


# ---------------------------------------------------------------------------
# 2. Forward-backward, vectorised over sessions
# ---------------------------------------------------------------------------
def emission_lik(b: Batch, W: np.ndarray) -> np.ndarray:
    """(S, T, K) likelihood of the observed choice under each state (1 if masked)."""
    logits = b.X @ W.T                                         # (S, T, K)
    p_right = expit(logits)
    lik = np.where(b.y[..., None] == 1, p_right, 1.0 - p_right)
    lik[~b.obs] = 1.0                                          # misses and padding
    return lik


def forward_backward(b: Batch, W, P, pi0, need_posteriors=True):
    """Scaled forward-backward over all sessions at once.

    Returns loglik (total), and if need_posteriors: gamma (S,T,K) smoothed
    state probabilities and xi_sum (K,K) expected transition counts.
    Padded steps have likelihood 1, scale 1 and are excluded from xi.
    """
    lik = emission_lik(b, W)
    S, T, K = lik.shape                                       # K = number of states
    alpha = np.empty((S, T, K))
    scale = np.empty((S, T))
    prior = np.broadcast_to(pi0, (S, K)).copy()
    for t in range(T):
        a = prior * lik[:, t]
        scale[:, t] = a.sum(1)
        alpha[:, t] = a / scale[:, t, None]
        prior = alpha[:, t] @ P
    loglik = float(np.log(scale[b.valid]).sum())
    if not need_posteriors:
        return loglik, None, None

    beta = np.ones((S, T, K))
    xi_sum = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        lb = lik[:, t + 1] * beta[:, t + 1]                    # (S, K)
        beta[:, t] = (lb @ P.T) / scale[:, t + 1, None]
        # xi_t(i,j) = alpha_t(i) P(i,j) lik_{t+1}(j) beta_{t+1}(j) / scale_{t+1}
        m = b.valid[:, t + 1]                                  # only real transitions
        if m.any():
            xi = (alpha[m, t][:, :, None] * P[None] * lb[m][:, None, :]
                  / scale[m, t + 1][:, None, None])
            xi_sum += xi.sum(0)
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    return loglik, gamma, xi_sum


# ---------------------------------------------------------------------------
# 3. M-step
# ---------------------------------------------------------------------------
def weighted_logistic_map(Xo, yo, wts, sigma, w0, free=None):
    """argmax_w  sum_n wts_n log P(y_n | x_n, w)  -  ||w||^2 / (2 sigma^2).

    free : optional boolean mask over the coefficients; coefficients where it is
           False stay fixed at their value in w0 (used to pin a state's weights,
           e.g. a 'lapse' state with prev_choice = wsls = 0).
    """
    free = np.ones(len(w0), dtype=bool) if free is None else np.asarray(free, bool)
    if not free.any():
        return np.asarray(w0, float).copy()
    w_fix = np.asarray(w0, float).copy()
    offset = Xo[:, ~free] @ w_fix[~free]                       # fixed part of the logit
    Xf = Xo[:, free]

    def negobj(wf):
        z = Xf @ wf + offset
        ll = wts * (yo * log_expit(z) + (1 - yo) * log_expit(-z))
        grad = Xf.T @ (wts * (yo - expit(z)))
        return -(ll.sum() - 0.5 * wf @ wf / sigma ** 2), -(grad - wf / sigma ** 2)
    res = minimize(negobj, w_fix[free], jac=True, method="L-BFGS-B")
    w_fix[free] = res.x
    return w_fix


def log_prior(W, sigma):
    return float(-0.5 * (W ** 2).sum() / sigma ** 2)


def m_step(b: Batch, gamma, xi_sum, W, sigma, free=None):
    """free : optional (K, 3) boolean mask of the weights that are estimated."""
    pi0 = gamma[:, 0].sum(0)
    pi0 /= pi0.sum()
    P = xi_sum / xi_sum.sum(1, keepdims=True)                  # alpha=1, kappa=0: MLE
    g_obs = gamma[b.obs]                                       # (N, K)
    W_new = np.vstack([weighted_logistic_map(b.Xo, b.yo, g_obs[:, k], sigma, W[k],
                                             None if free is None else free[k])
                       for k in range(W.shape[0])])
    return W_new, P, pi0


# ---------------------------------------------------------------------------
# 4. Fitting
# ---------------------------------------------------------------------------
def fit_glm(b: Batch, sigma):
    """1-state GLM (MAP), used to initialise the GLM-HMM (Ashwood recipe)."""
    return weighted_logistic_map(b.Xo, b.yo, np.ones(len(b.yo)), sigma, np.zeros(3))


def fit_em(b: Batch, sigma, W_init, P_init, pi0_init, free=None):
    W, P, pi0 = W_init.copy(), P_init.copy(), pi0_init.copy()
    prev = -np.inf
    for it in range(1, MAX_ITERS + 1):
        ll, gamma, xi = forward_backward(b, W, P, pi0)
        lp = (ll + log_prior(W, sigma)) / b.n_obs              # log-posterior / trial
        if lp - prev < TOL_PER_TRIAL:
            break
        prev = lp
        W, P, pi0 = m_step(b, gamma, xi, W, sigma, free)
    ll, _, _ = forward_backward(b, W, P, pi0, need_posteriors=False)
    # sort states: 0 = exploit = largest prev_choice weight
    order = np.argsort(-W[:, 1])
    W, P, pi0 = W[order], P[np.ix_(order, order)], pi0[order]
    return {"W": W, "P": P, "pi0": pi0, "ll": ll,
            "logpost": ll + log_prior(W, sigma), "n_iters": it}


def fit_glmhmm(b: Batch, sigma, n_restarts, seed, free=None, K=K):
    """Hierarchical init + random restarts; keep the best log-posterior.

    free : optional (K, 3) boolean mask; weights where it is False are pinned
           at 0 (e.g. free[1] = [True, False, False] makes state 1 a 'lapse'
           state that ignores history and keeps only a side bias).
    """
    rng = np.random.default_rng(seed)
    w_glm = fit_glm(b, sigma)
    P0 = np.full((K, K), (1 - STICKY_SELF) / (K - 1))
    np.fill_diagonal(P0, STICKY_SELF)
    fits = []
    for _ in range(n_restarts):
        W0 = w_glm[None] + rng.normal(0, INIT_NOISE, size=(K, 3))
        if free is not None:
            W0 = np.where(free, W0, 0.0)                       # pinned weights = 0
        fits.append(fit_em(b, sigma, W0, P0, np.full(K, 1.0 / K), free))
    best = max(fits, key=lambda f: f["logpost"])
    # how many restarts reached the same optimum (within 1e-4 nats/trial)
    best["restarts_agree"] = int(sum(abs(f["logpost"] - best["logpost"]) / b.n_obs < 1e-4
                                     for f in fits))
    best["n_restarts"] = n_restarts
    return best


# ---------------------------------------------------------------------------
# 5. Summaries of a fitted model
# ---------------------------------------------------------------------------
def pstay_rows(W, sigma):
    rows = []
    for k in range(K):
        b0, wc, ww = W[k]
        for outcome, rs in (("reward", 1.0), ("omission", -1.0)):
            # average over previous side (the bias makes the two sides differ slightly)
            p = np.mean([expit((b0 + wc * pc + ww * rs * pc) * pc) if pc > 0
                         else 1 - expit(b0 + wc * pc + ww * rs * pc)
                         for pc in (1.0, -1.0)])
            rows.append({"sigma": sigma, "state": LABELS[k], "prev_outcome": outcome,
                         "p_stay": p})
    return rows


def switch_rows(b: Batch, fit, sigma):
    """Share of switch trials that the smoothed posterior assigns to explore."""
    _, gamma, _ = forward_backward(b, fit["W"], fit["P"], fit["pi0"])
    has_prev = b.obs & (b.X[..., 1] != 0)
    choice_pm = 2 * b.y - 1
    stay = (choice_pm == b.X[..., 1])
    rewarded_prev = (b.X[..., 2] * b.X[..., 1]) > 0            # wsls*prev_choice = +1
    rows = []
    for outcome, sel in (("all", has_prev), ("reward", has_prev & rewarded_prev),
                         ("omission", has_prev & ~rewarded_prev)):
        sw = sel & ~stay
        rows.append({"sigma": sigma, "prev_outcome": outcome, "n_switches": int(sw.sum()),
                     "frac_switches_in_explore": float((gamma[..., 1][sw] > 0.5).mean()),
                     "mean_P_explore_on_stays": float(gamma[..., 1][sel & stay].mean()),
                     "explore_occupancy": float((gamma[..., 1][b.obs] > 0.5).mean())})
    return rows


def fit_row(fit, sigma, n_obs):
    W, P = fit["W"], fit["P"]
    dwell = 1.0 / (1.0 - np.diag(P))
    evals, evecs = np.linalg.eig(P.T)
    stat = np.real(evecs[:, np.argmin(abs(evals - 1))])
    stat /= stat.sum()
    row = {"sigma": sigma, "train_ll_per_trial": fit["ll"] / n_obs,
           "n_iters": fit["n_iters"],
           "restarts_agree": f"{fit['restarts_agree']}/{fit['n_restarts']}"}
    for k in range(K):
        for j, name in enumerate(REGRESSORS):
            row[f"{LABELS[k]}_{name}"] = W[k, j]
        row[f"{LABELS[k]}_dwell"] = dwell[k]
        row[f"{LABELS[k]}_stationary"] = stat[k]
    row["P_exploit_to_explore"] = P[0, 1]
    row["P_explore_to_exploit"] = P[1, 0]
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sessions, tags = load_sessions(args.csv)
    full = Batch(sessions)

    # ---- full-data fits ------------------------------------------------------
    fit_rows, pstay, switches = [], [], []
    for sigma in args.sigmas:
        t0 = time.time()
        fit = fit_glmhmm(full, sigma, N_RESTARTS_FULL, seed=0)
        fit_rows.append(fit_row(fit, sigma, full.n_obs))
        pstay += pstay_rows(fit["W"], sigma)
        switches += switch_rows(full, fit, sigma)
        W = fit["W"]
        print(f"[full] sigma={sigma:g}: exploit {np.round(W[0], 3)}  explore "
              f"{np.round(W[1], 3)}  LL/trial={fit['ll'] / full.n_obs:.4f}  "
              f"iters={fit['n_iters']}  agree={fit['restarts_agree']}/"
              f"{fit['n_restarts']}  ({time.time() - t0:.0f}s)", flush=True)
        if sigma == 100.0:
            print(f"[validate] ssm saved fit: exploit {SSM_REFERENCE[0]}  explore "
                  f"{SSM_REFERENCE[1]}  -> max |diff| = "
                  f"{np.abs(W - SSM_REFERENCE).max():.3f}", flush=True)
        pd.DataFrame(fit_rows).to_csv(out / "fits_full_data.csv", index=False)
        pd.DataFrame(pstay).to_csv(out / "pstay_by_sigma.csv", index=False)
        pd.DataFrame(switches).to_csv(out / "switch_attribution.csv", index=False)

    # ---- cross-validation -----------------------------------------------------
    cv = pd.DataFrame()
    if not args.skip_cv:
        ckpt = out / "checkpoint.csv"
        rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
        done = {(float(r["sigma"]), int(r["fold"])) for r in rows}
        folds = session_folds([t[0] for t in tags], N_FOLDS, FOLD_SEED)
        for f, te in enumerate(folds):
            te_set = set(te.tolist())
            tr = [i for i in range(len(sessions)) if i not in te_set]
            b_tr = Batch(split_sessions(sessions, tr))
            b_te = Batch(split_sessions(sessions, list(te)))
            for sigma in args.sigmas:
                if (sigma, f) in done:
                    continue
                t0 = time.time()
                fit = fit_glmhmm(b_tr, sigma, N_RESTARTS_CV, seed=100 + f)
                ll_te, _, _ = forward_backward(b_te, fit["W"], fit["P"], fit["pi0"],
                                               need_posteriors=False)
                rows.append({"sigma": sigma, "fold": f,
                             "test_ll_per_trial": ll_te / b_te.n_obs,
                             "exploit_prev_choice": fit["W"][0, 1],
                             "exploit_wsls": fit["W"][0, 2]})
                pd.DataFrame(rows).to_csv(ckpt, index=False)
                print(f"[cv] fold {f} sigma={sigma:g}: held-out LL/trial = "
                      f"{ll_te / b_te.n_obs:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        cv = pd.DataFrame(rows)
        cv.to_csv(out / "cv_heldout.csv", index=False)

    # ---- summary ----------------------------------------------------------------
    fits = pd.DataFrame(fit_rows)
    ps = pd.DataFrame(pstay).pivot_table(index=["sigma", "state"], columns="prev_outcome",
                                         values="p_stay").reset_index()
    sw = pd.DataFrame(switches)
    lines = ["GLM-HMM K=2 (ashwood_wsls): sensitivity to the prior on the weights",
             "=" * 66, "",
             "Full-data fits (state 0 = exploit):",
             fits.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
             "Model P(stay) by state and previous outcome:",
             ps.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
             "Switch attribution (smoothed posterior > 0.5):",
             sw.to_string(index=False, float_format=lambda v: f"{v:.4f}"), ""]
    if len(cv):
        piv = cv.pivot(index="fold", columns="sigma", values="test_ll_per_trial")
        ref = piv[100.0] if 100.0 in piv.columns else piv[piv.columns.max()]
        summ = pd.DataFrame({
            "sigma": piv.columns,
            "heldout_ll_mean": piv.mean().values,
            "heldout_ll_sem": (piv.std(ddof=1) / np.sqrt(len(piv))).values,
            "diff_vs_sigma100_mean": (piv.sub(ref, axis=0)).mean().values,
            "diff_vs_sigma100_sem": (piv.sub(ref, axis=0).std(ddof=1)
                                     / np.sqrt(len(piv))).values,
            "folds_better_than_sigma100": (piv.sub(ref, axis=0) > 0).sum().values,
        })
        summ.to_csv(out / "cv_summary.csv", index=False)
        lines += ["Held-out LL per trial (5-fold CV, sessions stratified by animal):",
                  summ.to_string(index=False, float_format=lambda v: f"{v:.5f}")]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
