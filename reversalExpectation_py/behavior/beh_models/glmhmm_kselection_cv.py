"""
glmhmm_kselection_cv.py
=======================
Why K = 2? Cross-validated held-out log-likelihood of the `ashwood_wsls`
GLM-HMM for K = 1, 2, 3, 4 states, all fitted with the SAME implementation
and the SAME Gaussian prior on the weights (sigma = SIGMA_W), so the curve is
comparable across K.

  * K = 1 is the plain GLM (MAP).
  * K >= 2 uses the numpy EM in glmhmm_regularization.py (Ashwood hierarchical
    init from the GLM, N_RESTARTS random restarts, best log-posterior kept).
  * Folds: 5, sessions stratified by animal (hmmGlm._session_folds, seed 0),
    i.e. the same split as every other CV in the project.

Reported per K: held-out LL per observed trial per fold (mean +- SEM), the
paired gain over K-1 (folds improving, per-animal Wilcoxon), the fraction of
restarts that reach the best solution (fit stability), and the full-data
weights / dwell times of each K so the extra states can be inspected.

Resumable: every (K, fold) result is written to checkpoint.csv immediately.
Needs numpy, scipy, pandas (no ssm).

Inputs  (relative to reversalExpectation_py/):
    analysis/bandit_R71_lesion.csv
Outputs (analysis/glmhmm_kselection_cv/):
    checkpoint.csv       per (K, fold, animal) held-out LL
    cv_per_fold.csv
    cv_summary.csv
    full_fits.csv        weights, dwell and occupancy of each state, per K
    report.txt

Usage (from reversalExpectation_py/):
    python behavior/beh_models/glmhmm_kselection_cv.py
    python behavior/beh_models/glmhmm_kselection_cv.py --Ks 1 2 3 4 --restarts 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glmhmm_regularization as gr          # noqa: E402

REPO_ROOT = HERE.parents[1]
CSV_PATH = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_DIR = REPO_ROOT / "analysis" / "glmhmm_kselection_cv"
SIGMA_W = 1.0


# ---------------------------------------------------------------------------
# Per-session held-out log-likelihood
# ---------------------------------------------------------------------------
def glm_ll_per_session(b: gr.Batch, w):
    z = b.X @ w                                                # (S, T)
    ll = b.y * log_expit(z) + (1 - b.y) * log_expit(-z)
    return np.where(b.obs, ll, 0.0).sum(1)


def hmm_ll_per_session(b: gr.Batch, fit):
    lik = gr.emission_lik(b, fit["W"])
    S, T, K = lik.shape
    prior = np.broadcast_to(fit["pi0"], (S, K)).copy()
    ll = np.zeros(S)
    for t in range(T):
        a = prior * lik[:, t]
        s = a.sum(1)
        ll += np.where(b.valid[:, t], np.log(s), 0.0)
        prior = (a / s[:, None]) @ fit["P"]
    return ll


def fit_and_score(b_tr, b_te, K, restarts, seed):
    """Fit on the training batch; return per-session test LL and the fit."""
    if K == 1:
        w = gr.fit_glm(b_tr, SIGMA_W)
        return glm_ll_per_session(b_te, w), {"W": w[None], "restarts_agree": 1,
                                             "n_restarts": 1}
    fit = gr.fit_glmhmm(b_tr, SIGMA_W, restarts, seed, K=K)
    return hmm_ll_per_session(b_te, fit), fit


def state_table(fit, K):
    W = fit["W"]
    rows = []
    if K == 1:
        rows.append({"K": 1, "state": 0, "bias": W[0, 0], "prev_choice": W[0, 1],
                     "wsls": W[0, 2], "dwell": np.inf, "stationary": 1.0})
        return rows
    P = fit["P"]
    evals, evecs = np.linalg.eig(P.T)
    stat = np.real(evecs[:, np.argmin(abs(evals - 1))])
    stat /= stat.sum()
    for k in range(K):
        rows.append({"K": K, "state": k, "bias": W[k, 0], "prev_choice": W[k, 1],
                     "wsls": W[k, 2], "dwell": 1 / (1 - P[k, k]), "stationary": stat[k],
                     "restarts_agree": f"{fit['restarts_agree']}/{fit['n_restarts']}"})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sessions, tags = gr.load_sessions(args.csv)
    animals = np.array([t[0] for t in tags], dtype=object)
    folds = gr.session_folds(animals, gr.N_FOLDS, gr.FOLD_SEED)

    # ---- cross-validation ------------------------------------------------------
    ckpt = out / "checkpoint.csv"
    rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
    done = {(int(r["K"]), int(r["fold"])) for r in rows}
    agree_rows = []
    for f, te in enumerate(folds):
        te = list(te)
        tr = [i for i in range(len(sessions)) if i not in set(te)]
        b_tr = gr.Batch([sessions[i] for i in tr])
        b_te = gr.Batch([sessions[i] for i in te])
        for K in args.Ks:
            if (K, f) in done:
                continue
            t0 = time.time()
            ll_s, fit = fit_and_score(b_tr, b_te, K, args.restarts, seed=100 + f)
            n_s = b_te.obs.sum(1)
            for a, ll, n in zip(animals[te], ll_s, n_s):
                rows.append({"K": K, "fold": f, "animal": a, "ll": ll, "n_obs": int(n),
                             "restarts_agree": fit["restarts_agree"],
                             "n_restarts": fit["n_restarts"]})
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            print(f"[cv] fold {f} K={K}: held-out LL/trial = {ll_s.sum() / n_s.sum():.5f}"
                  f"  restarts agree {fit['restarts_agree']}/{fit['n_restarts']}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    cv = pd.DataFrame(rows)

    per_fold = (cv.groupby(["K", "fold"]).agg(ll=("ll", "sum"), n_obs=("n_obs", "sum"),
                                              agree=("restarts_agree", "first"),
                                              n_restarts=("n_restarts", "first"))
                  .assign(ll_per_trial=lambda x: x["ll"] / x["n_obs"]).reset_index())
    per_fold.to_csv(out / "cv_per_fold.csv", index=False)
    per_animal = (cv.groupby(["K", "animal"])[["ll", "n_obs"]].sum()
                    .assign(ll_per_trial=lambda x: x["ll"] / x["n_obs"]).reset_index())
    pf = per_fold.pivot(index="fold", columns="K", values="ll_per_trial")
    pa = per_animal.pivot(index="animal", columns="K", values="ll_per_trial")

    summ = []
    Ks = sorted(pf.columns)
    for i, K in enumerate(Ks):
        row = {"K": K, "heldout_ll_mean": pf[K].mean(),
               "heldout_ll_sem": pf[K].std(ddof=1) / np.sqrt(len(pf)),
               "restarts_agree_mean": (per_fold[per_fold["K"] == K]["agree"]
                                       / per_fold[per_fold["K"] == K]["n_restarts"]).mean()}
        if i > 0:
            Kp = Ks[i - 1]
            dfold, dan = pf[K] - pf[Kp], pa[K] - pa[Kp]
            row.update({"gain_vs_prev_mean": dfold.mean(),
                        "gain_vs_prev_sem": dfold.std(ddof=1) / np.sqrt(len(dfold)),
                        "folds_improving": f"{int((dfold > 0).sum())}/{len(dfold)}",
                        "animals_improving": f"{int((dan > 0).sum())}/{len(dan)}",
                        "median_animal_gain": dan.median(),
                        "wilcoxon_p_animals": wilcoxon(dan).pvalue})
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "cv_summary.csv", index=False)

    # ---- full-data fits, to look at the states of each K -----------------------
    st_rows = []
    if not args.skip_full:
        b_all = gr.Batch(sessions)
        for K in args.Ks:
            t0 = time.time()
            if K == 1:
                fit = {"W": gr.fit_glm(b_all, SIGMA_W)[None]}
            else:
                fit = gr.fit_glmhmm(b_all, SIGMA_W, max(args.restarts, 5), seed=0, K=K)
            st_rows += state_table(fit, K)
            pd.DataFrame(st_rows).to_csv(out / "full_fits.csv", index=False)
            print(f"[full] K={K} done ({time.time() - t0:.0f}s)", flush=True)

    fmt = lambda x: x.to_string(index=False, float_format=lambda v: f"{v:.5f}")  # noqa: E731
    report = "\n".join([
        f"GLM-HMM (ashwood_wsls) K selection, 5-fold CV, prior sigma = {SIGMA_W}",
        "=" * 70, fmt(summ), "",
        "Full-data fits (states sorted by prev_choice weight, descending):",
        fmt(pd.DataFrame(st_rows)) if st_rows else "(skipped)",
    ])
    (out / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
