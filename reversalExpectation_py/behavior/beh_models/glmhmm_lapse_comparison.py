"""
glmhmm_lapse_comparison.py
==========================
Is the GLM-HMM 'explore' state a behavioural strategy, or just lapses?

The GLM-HMM explains choices with two discrete states: exploit (near-
deterministic win-stay / perseveration) and explore (weak history dependence).
An alternative account is a SINGLE strategy plus lapses (random choices) whose
rate drifts over the session (fatigue, satiety) or after reward droughts. If a
lapse model predicts held-out choices as well as the GLM-HMM, the second state
need not be a distinct strategy.

Models (same design matrix x_t = [1, prev_choice, wsls], same Gaussian prior
on the GLM weights, sigma = SIGMA_W):

  M1  GLM                     P(R) = s(w.x)
  M2  GLM + constant lapse    P(R) = lam*q + (1 - lam) s(w.x)
  M3  GLM + lapse(session)    lam_t = s(c . B(trial number in session))
                              B = intercept + Gaussian bumps over trial number
  M4  GLM + lapse(session,    lam_t = s(c . B(trial) + c_r * rate_t),
      reward rate)            rate_t = causal exponentially-weighted reward
                              rate (half-life ~3 trials) before trial t
  M5  GLM-HMM, Markov lapse   2 states; state 2 has prev_choice = wsls = 0
                              (only a side bias): sticky, history-blind lapses
  M6  GLM-HMM (free)          2 states, all weights free (the current model)

  (q = side split of lapses, a free parameter.)

Comparison: 5-fold cross-validation over sessions, stratified by animal
(same split as hmmGlm._session_folds, seed 0). Reported: held-out
log-likelihood per observed trial per fold, and per animal (Wilcoxon signed-
rank test of each model against M6).

Behavioural signature: one-step-ahead predicted P(switch) of each model
(fitted on all data) against the observed P(switch), as a function of
(a) trials since the last block reversal and (b) position in the session.
A lapse model can only raise switching through the lapse rate; if the data's
switching around reversals follows the GLM-HMM but not the lapse models, the
second state carries structure that lapses do not.

Needs numpy, scipy, pandas (no ssm). Imports the EM from
glmhmm_regularization.py (same folder).

Inputs  (relative to reversalExpectation_py/):
    analysis/bandit_R71_lesion.csv
Outputs (analysis/glmhmm_lapse_comparison/):
    cv_per_fold.csv          held-out LL / trial, per model and fold
    cv_per_animal.csv        held-out LL / trial, per model and animal
    cv_summary.csv           means, SEMs, differences vs M6, Wilcoxon per animal
    full_fit_params.csv      parameters of each model fitted on all data
    signature_reversal.csv   predicted vs observed P(switch) by trials since reversal
    signature_session.csv    predicted vs observed P(switch) by session position
    lapse_curves.csv         fitted lapse rate vs trial number (M3, M4)
    report.txt
    signature.png

Usage (from reversalExpectation_py/):
    python behavior/beh_models/glmhmm_lapse_comparison.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glmhmm_regularization as gr          # noqa: E402  (EM + design matrix)

REPO_ROOT = HERE.parents[1]
CSV_PATH = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_DIR = REPO_ROOT / "analysis" / "glmhmm_lapse_comparison"

SIGMA_W = 1.0               # prior sd on GLM weights (see glmhmm_regularization)
SIGMA_LAPSE = 10.0          # weak prior sd on lapse parameters
N_BUMPS = 6                 # Gaussian bumps over trial number (M3, M4)
RATE_HALFLIFE = 3.0         # trials, for the reward-rate covariate (M4)
N_RESTARTS_FULL = 5
N_RESTARTS_CV = 3
MAX_TAU = 30                # trials since reversal shown in the signature
MODELS = ["M1_GLM", "M2_lapse_const", "M3_lapse_session", "M4_lapse_session_reward",
          "M5_HMM_markov_lapse", "M6_GLMHMM"]


# ---------------------------------------------------------------------------
# 1. Data: design matrix (shared with gr) + covariates, padded per session
# ---------------------------------------------------------------------------
def reward_rate(rewarded, observed, halflife):
    """Causal EWMA of reward over observed trials, value BEFORE each trial."""
    a = 1.0 - 0.5 ** (1.0 / halflife)
    rate = np.zeros(len(rewarded))
    r = 0.5                                                    # neutral start
    for t in range(len(rewarded)):
        rate[t] = r
        if observed[t]:
            r = (1 - a) * r + a * rewarded[t]
    return rate


def load(csv_path: Path):
    cols = ["animal", "session_file", "block_idx", "choice", "rewarded",
            "lesioned", "meets_criteria"]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df[df["lesioned"].isna() & (df["meets_criteria"] == True)]   # noqa: E712
    sessions, covs, tags = [], [], []
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        y, obs, X = gr.session_design(d)
        if obs.sum() < gr.MIN_TRIALS:
            continue
        T = len(y)
        blk = d["block_idx"].to_numpy()
        new_blk = np.r_[False, blk[1:] != blk[:-1]]            # first trial of a new block
        tau = np.full(T, -1)                                   # trials since reversal
        last = None
        for t in range(T):
            if new_blk[t]:
                last = t
            if last is not None:
                tau[t] = t - last
        r = np.nan_to_num(d["rewarded"].to_numpy(float), nan=0.0)
        sessions.append((y, obs, X))
        covs.append({"trial": np.arange(T, dtype=float), "tau": tau,
                     "rate": reward_rate(r, obs, RATE_HALFLIFE)})
        tags.append((animal, ses))
    print(f"[data] {len(sessions)} sessions, {len({t[0] for t in tags})} animals")
    return sessions, covs, tags


class Data:
    """gr.Batch plus padded covariates and bookkeeping for one subset of sessions."""

    def __init__(self, sessions, covs, tags, bump_centres, bump_width):
        self.b = gr.Batch(sessions)
        S, T = self.b.y.shape
        self.trial = np.zeros((S, T))
        self.tau = np.full((S, T), -1)
        self.rate = np.zeros((S, T))
        for i, c in enumerate(covs):
            n = len(c["trial"])
            self.trial[i, :n], self.tau[i, :n], self.rate[i, :n] = c["trial"], c["tau"], c["rate"]
        self.animals = np.array([t[0] for t in tags], dtype=object)
        # lapse design: intercept + Gaussian bumps over trial number
        bumps = np.exp(-0.5 * ((self.trial[..., None] - bump_centres) / bump_width) ** 2)
        self.B = np.concatenate([np.ones((S, T, 1)), bumps], axis=2)
        o = self.b.obs
        self.Xo, self.yo = self.b.X[o], self.b.y[o]
        self.Bo, self.rate_o = self.B[o], self.rate[o]
        self.sess_of_obs = np.broadcast_to(np.arange(S)[:, None], (S, T))[o]


# ---------------------------------------------------------------------------
# 2. Lapse models (M1-M4): direct maximum a posteriori
# ---------------------------------------------------------------------------
def lapse_unpack(theta, kind, nB):
    w = theta[:3]
    if kind == "M1_GLM":
        return w, None, None, None
    q = theta[3]
    if kind == "M2_lapse_const":
        return w, q, theta[4:5], None
    if kind == "M3_lapse_session":
        return w, q, theta[4:4 + nB], None
    return w, q, theta[4:4 + nB], theta[4 + nB]                # M4


def lapse_p_right(theta, kind, X, B, rate):
    """P(right) for every trial in the arrays (any leading shape)."""
    w, q, c, cr = lapse_unpack(theta, kind, B.shape[-1])
    p = expit(X @ w)
    if kind == "M1_GLM":
        return p
    if kind == "M2_lapse_const":
        lam_logit = np.full(X.shape[:-1], c[0])
    else:
        lam_logit = B @ c
        if cr is not None:
            lam_logit = lam_logit + cr * (rate - 0.5)
    lam = expit(lam_logit)
    return lam * expit(q) + (1 - lam) * p


def lapse_n_params(kind, nB):
    return {"M1_GLM": 3, "M2_lapse_const": 5, "M3_lapse_session": 4 + nB,
            "M4_lapse_session_reward": 5 + nB}[kind]


def fit_lapse(d: Data, kind):
    nB = d.B.shape[-1]
    n = lapse_n_params(kind, nB)

    def negpost(theta):
        p = np.clip(lapse_p_right(theta, kind, d.Xo, d.Bo, d.rate_o), 1e-12, 1 - 1e-12)
        ll = np.sum(d.yo * np.log(p) + (1 - d.yo) * np.log(1 - p))
        lp = -0.5 * (theta[:3] @ theta[:3]) / SIGMA_W ** 2 \
             - 0.5 * (theta[3:] @ theta[3:]) / SIGMA_LAPSE ** 2
        return -(ll + lp)

    w_glm = gr.fit_glm(d.b, SIGMA_W)
    best = None
    for lam0 in (-4.0, -2.0):                                  # low / moderate lapse start
        theta0 = np.zeros(n)
        theta0[:3] = w_glm
        if n > 3:
            theta0[4] = lam0
        res = minimize(negpost, theta0, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def lapse_loglik_per_session(theta, kind, d: Data):
    p = np.clip(lapse_p_right(theta, kind, d.Xo, d.Bo, d.rate_o), 1e-12, 1 - 1e-12)
    ll = d.yo * np.log(p) + (1 - d.yo) * np.log(1 - p)
    return np.bincount(d.sess_of_obs, weights=ll, minlength=d.b.y.shape[0])


# ---------------------------------------------------------------------------
# 3. HMM models (M5, M6): EM from gr; forward pass for per-session LL and
#    one-step-ahead predictions
# ---------------------------------------------------------------------------
FREE_MARKOV_LAPSE = np.array([[True, True, True], [True, False, False]])


def fit_hmm(d: Data, kind, n_restarts, seed):
    free = FREE_MARKOV_LAPSE if kind == "M5_HMM_markov_lapse" else None
    return gr.fit_glmhmm(d.b, SIGMA_W, n_restarts, seed, free=free)


def hmm_forward(d: Data, fit):
    """Per-session log-likelihood and one-step-ahead P(right) for every trial."""
    b = d.b
    lik = gr.emission_lik(b, fit["W"])
    p_right_k = expit(b.X @ fit["W"].T)                       # (S, T, K)
    S, T, K = lik.shape
    prior = np.broadcast_to(fit["pi0"], (S, K)).copy()
    logscale = np.zeros((S, T))
    p_pred = np.zeros((S, T))
    for t in range(T):
        p_pred[:, t] = (prior * p_right_k[:, t]).sum(1)
        a = prior * lik[:, t]
        s = a.sum(1)
        logscale[:, t] = np.log(s)
        prior = (a / s[:, None]) @ fit["P"]
    ll_sess = np.where(b.valid, logscale, 0.0).sum(1)
    return ll_sess, p_pred


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def bump_basis(sessions):
    """Bump centres at quantiles of the trial number; width = centre spacing."""
    trials = np.concatenate([np.arange(len(s[0])) for s in sessions])
    centres = np.quantile(trials, np.linspace(0.02, 0.98, N_BUMPS))
    width = np.median(np.diff(centres))
    return centres, width


def subset(lst, idx):
    return [lst[i] for i in idx]


def predicted_p_right(kind, model, d: Data):
    if kind in ("M5_HMM_markov_lapse", "M6_GLMHMM"):
        return hmm_forward(d, model)[1]
    return lapse_p_right(model, kind, d.b.X, d.B, d.rate)


def signature_tables(d: Data, preds: dict):
    """Observed vs predicted P(switch) by trials since reversal and session position."""
    b = d.b
    has_prev = b.obs & (b.X[..., 1] != 0)
    choice_pm = 2 * b.y - 1
    switch = (choice_pm != b.X[..., 1]).astype(float)
    prev_right = b.X[..., 1] > 0
    rows_tau, rows_ses = [], []
    p_sw = {k: np.where(prev_right, 1 - p, p) for k, p in preds.items()}  # P(switch)
    for tau in range(MAX_TAU + 1):
        m = has_prev & (d.tau == tau)
        row = {"tau": tau, "n": int(m.sum()), "data": switch[m].mean()}
        row.update({k: v[m].mean() for k, v in p_sw.items()})
        rows_tau.append(row)
    edges = np.arange(0, 1000, 50)
    for lo in edges:
        m = has_prev & (d.trial >= lo) & (d.trial < lo + 50)
        if m.sum() < 500:
            continue
        row = {"trial_bin_start": int(lo), "n": int(m.sum()), "data": switch[m].mean()}
        row.update({k: v[m].mean() for k, v in p_sw.items()})
        rows_ses.append(row)
    return pd.DataFrame(rows_tau), pd.DataFrame(rows_ses)


def make_figure(sig_tau, sig_ses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"M1_GLM": "#9A9A94", "M2_lapse_const": "#C9B27C",
              "M3_lapse_session": "#E8A33D", "M4_lapse_session_reward": "#C9472B",
              "M5_HMM_markov_lapse": "#6F9BD1", "M6_GLMHMM": "#2C5F2D"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sig, x, xl in ((axes[0], sig_tau, "tau", "trials since reversal"),
                           (axes[1], sig_ses, "trial_bin_start", "trial in session (bin start)")):
        ax.plot(sig[x], sig["data"], "o", color="#1A2E2A", ms=4, label="data")
        for k, c in colors.items():
            ax.plot(sig[x], sig[k], "-", color=c, lw=1.5, label=k)
        ax.set_xlabel(xl)
        ax.set_ylabel("P(switch)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sessions, covs, tags = load(args.csv)
    centres, width = bump_basis(sessions)
    animals = np.array([t[0] for t in tags], dtype=object)

    # ---- cross-validation ------------------------------------------------------
    ckpt = out / "checkpoint.csv"
    rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
    done = {(r["model"], int(r["fold"])) for r in rows}
    folds = gr.session_folds(animals, gr.N_FOLDS, gr.FOLD_SEED)
    for f, te in enumerate(folds):
        te = list(te)
        tr = [i for i in range(len(sessions)) if i not in set(te)]
        d_tr = Data(subset(sessions, tr), subset(covs, tr), subset(tags, tr), centres, width)
        d_te = Data(subset(sessions, te), subset(covs, te), subset(tags, te), centres, width)
        for kind in MODELS:
            if (kind, f) in done:
                continue
            t0 = time.time()
            if kind.startswith("M5") or kind.startswith("M6"):
                model = fit_hmm(d_tr, kind, N_RESTARTS_CV, seed=100 + f)
                ll_s = hmm_forward(d_te, model)[0]
            else:
                model = fit_lapse(d_tr, kind)
                ll_s = lapse_loglik_per_session(model, kind, d_te)
            n_s = d_te.b.obs.sum(1)
            for a, ll, n in zip(d_te.animals, ll_s, n_s):
                rows.append({"model": kind, "fold": f, "animal": a, "ll": ll, "n_obs": int(n)})
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            print(f"[cv] fold {f} {kind}: held-out LL/trial = {ll_s.sum() / n_s.sum():.5f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    cv = pd.DataFrame(rows)

    per_fold = (cv.groupby(["model", "fold"])[["ll", "n_obs"]].sum()
                  .assign(ll_per_trial=lambda x: x["ll"] / x["n_obs"]).reset_index())
    per_animal = (cv.groupby(["model", "animal"])[["ll", "n_obs"]].sum()
                    .assign(ll_per_trial=lambda x: x["ll"] / x["n_obs"]).reset_index())
    per_fold.to_csv(out / "cv_per_fold.csv", index=False)
    per_animal.to_csv(out / "cv_per_animal.csv", index=False)

    pf = per_fold.pivot(index="fold", columns="model", values="ll_per_trial")
    pa = per_animal.pivot(index="animal", columns="model", values="ll_per_trial")
    summ = []
    for kind in MODELS:
        dfold = pf[kind] - pf["M6_GLMHMM"]
        dan = pa[kind] - pa["M6_GLMHMM"]
        p = wilcoxon(dan).pvalue if kind != "M6_GLMHMM" else np.nan
        summ.append({"model": kind, "heldout_ll_mean": pf[kind].mean(),
                     "heldout_ll_sem": pf[kind].std(ddof=1) / np.sqrt(len(pf)),
                     "diff_vs_M6_mean": dfold.mean(),
                     "diff_vs_M6_sem": dfold.std(ddof=1) / np.sqrt(len(dfold)),
                     "folds_M6_better": int((dfold < 0).sum()),
                     "animals_M6_better": f"{int((dan < 0).sum())}/{len(dan)}",
                     "median_animal_diff_vs_M6": dan.median(),
                     "wilcoxon_p_animals": p})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "cv_summary.csv", index=False)

    # ---- full-data fits + behavioural signature ---------------------------------
    d_all = Data(sessions, covs, tags, centres, width)
    models, preds, par_rows = {}, {}, []
    for kind in MODELS:
        if kind.startswith("M5") or kind.startswith("M6"):
            m = fit_hmm(d_all, kind, N_RESTARTS_FULL, seed=0)
            W, P = m["W"], m["P"]
            par_rows.append({"model": kind, "params":
                             f"W_exploit={np.round(W[0], 3).tolist()} "
                             f"W_state2={np.round(W[1], 3).tolist()} "
                             f"P01={P[0, 1]:.4f} P10={P[1, 0]:.4f} "
                             f"dwell=({1 / (1 - P[0, 0]):.1f}, {1 / (1 - P[1, 1]):.1f})"})
        else:
            m = fit_lapse(d_all, kind)
            par_rows.append({"model": kind,
                             "params": f"theta={np.round(m, 3).tolist()}"})
        models[kind] = m
        preds[kind] = predicted_p_right(kind, m, d_all)
        print(f"[full] {kind} done", flush=True)
    pd.DataFrame(par_rows).to_csv(out / "full_fit_params.csv", index=False)

    # lapse rate vs trial number for M3 / M4 (M4 at a neutral reward rate)
    grid = np.arange(0, int(d_all.trial.max()) + 1, 25, dtype=float)
    Bg = np.concatenate([np.ones((len(grid), 1)),
                         np.exp(-0.5 * ((grid[:, None] - centres) / width) ** 2)], axis=1)
    lc = pd.DataFrame({"trial": grid})
    for kind in ("M3_lapse_session", "M4_lapse_session_reward"):
        _, _, c, _ = lapse_unpack(models[kind], kind, Bg.shape[1])
        lc[kind] = expit(Bg @ c)
    lc["M2_lapse_const"] = expit(models["M2_lapse_const"][4])
    lc.to_csv(out / "lapse_curves.csv", index=False)

    sig_tau, sig_ses = signature_tables(d_all, preds)
    sig_tau.to_csv(out / "signature_reversal.csv", index=False)
    sig_ses.to_csv(out / "signature_session.csv", index=False)
    make_figure(sig_tau, sig_ses, out / "signature.png")

    # signature error: mean absolute error of predicted P(switch) vs data
    def mae(sig):
        w = sig["n"] / sig["n"].sum()
        return {k: float((w * (sig[k] - sig["data"]).abs()).sum()) for k in MODELS}
    err = pd.DataFrame({"reversal_MAE": mae(sig_tau), "session_MAE": mae(sig_ses)})

    fmt = lambda x: x.to_string(float_format=lambda v: f"{v:.5f}")  # noqa: E731
    report = "\n".join([
        "Lapse vs strategy: held-out comparison (5-fold CV, sessions stratified by animal)",
        "=" * 78,
        fmt(summ.set_index("model")), "",
        "Behavioural signature: weighted mean |predicted - observed| P(switch)",
        fmt(err), "",
        "P(switch) by trials since reversal:",
        fmt(sig_tau.set_index("tau")), "",
        "Fitted lapse rate vs trial number:",
        fmt(lc.set_index("trial").iloc[::4]), "",
        "Full-data parameters:",
        pd.DataFrame(par_rows).to_string(index=False),
    ])
    (out / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
