"""
glmhmm_cv_fast.py
=================
Fast version of the GLM-HMM cross-validation over K AND the model comparison
(GLM vs lapse vs GLM-HMM), in a single pass. Same model, same folds, same
Ashwood recipe (GLM -> global GLM-HMM initialised from the GLM weights) as
hmmGlm.cross_validate_glmhmm + hmmGlm.model_comparison; only the computing
strategy changes.

Why the original takes most of a day
  1. EM never stops early. ssm stops when |LP_t - LP_{t-1}| < tolerance, with LP
     the TOTAL log probability of the training set (ssm/hmm.py, _fit_em). With
     ~270,000 training trials, an absolute tolerance of 1e-4 is essentially
     never reached, so every restart runs all N_ITERS = 200 EM iterations.
     Here the tolerance is per trial: TOL_PER_TRIAL x (number of training
     trials), i.e. stop when an EM step improves the fit by less than 1e-6
     nats per trial, far below the differences between models we compare
     (~1e-2 nats per trial).
  2. Folds run one after another, and only the restarts of one fit run in
     parallel. Here every (fold, restart) fit of a given K is an independent
     job, so all cores are busy at once.
  3. 15 restarts per fit. With the Ashwood initialisation, restarts differ
     only by small noise on the GLM weights and land on the same optimum;
     N_RESTARTS = 3 is enough to compare K.
  4. model_comparison refits the GLM and the K=2 GLM-HMM on the same folds the
     CV already used. Here the GLM of each fold is fitted once and serves as
     both the K = 1 point of the CV and the GLM of the comparison, and the
     K = 2 GLM-HMM is shared too; only the (cheap) lapse model is added.

Outputs (analysis/)
  glmhmm_cv.csv                [K, fold, test_ll_per_trial, n_em_iters]
  glmhmm_model_comparison.csv  [model, fold, test_ll_per_trial]
  glmhmm_cv_fast_checkpoint.csv  written after each K; an interrupted run
      resumes from it (delete it to start over).

Run from the `ssm` conda environment, in behavior/beh_models:
    python -X utf8 glmhmm_cv_fast.py > ..\\..\\analysis\\glmhmm_cv_fast.log 2>&1
Progress lines from joblib ("Done 3 out of 15 | elapsed ...") go to the log.
"""

import os
import sys
import time
import warnings
from pathlib import Path

# one BLAS thread per worker process, BEFORE numpy is imported (same rule as
# run_glmhmm.py: avoids oversubscription / deadlocks on Windows)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore", category=UserWarning, module="ssm.optimizers")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent            # .../behavior/beh_models
REPO_ROOT = HERE.parents[1]                        # .../reversalExpectation_py
sys.path.insert(0, str(HERE))
import hmmGlm as g                                 # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUTPUT_DIR = REPO_ROOT / "analysis"
CHECKPOINT = OUTPUT_DIR / "glmhmm_cv_fast_checkpoint.csv"

PARAMETRIZATION = "ashwood_wsls"
K_RANGE = (1, 2, 3)          # K = 4 adds hours and is not needed to choose between 2 and 3
K_COMPARISON = 2             # GLM-HMM used in the GLM / lapse / GLM-HMM comparison
N_FOLDS = 5                  # same folds as run_glmhmm.py (same seed -> same split)
SEED = 0
N_RESTARTS = 3               # restarts per (fold, K) global fit
N_GLM_RESTARTS = 2           # restarts of the 1-state GLM (near-convex problem)
N_ITERS = 200                # EM iteration cap (rarely reached with the per-trial tolerance)
TOL_PER_TRIAL = 1e-6         # stop EM when the gain is < this many nats per training trial
N_JOBS = -1                  # all cores


# ---------------------------------------------------------------------------
# Worker jobs (top-level functions so Windows can pickle them)
# ---------------------------------------------------------------------------
def _n_obs(masks) -> int:
    return max(int(sum(int(m.sum()) for m in masks)), 1)


def _glm_and_lapse_job(f, train, test, seed):
    """Fold f: 1-state GLM (= the K = 1 model and the comparison GLM) + lapse model."""
    tr_ch, tr_in, tr_mk = train
    te_ch, te_in, te_mk = test
    tol = TOL_PER_TRIAL * _n_obs(tr_mk)
    glm, glm_w = g.fit_glm(tr_ch, tr_in, tr_mk, n_restarts=N_GLM_RESTARTS,
                           n_iters=N_ITERS, tolerance=tol, seed=seed, n_jobs=1)
    lapse = g.fit_lapse_model(tr_ch, tr_in, tr_mk, seed=seed)
    n_te = _n_obs(te_mk)
    return {"fold": f, "glm_w": glm_w,
            "ll_glm": glm.log_likelihood(te_ch, inputs=te_in, masks=te_mk) / n_te,
            "ll_lapse": g.lapse_log_likelihood(lapse, te_ch, te_in, te_mk) / n_te}


def _global_job(f, K, restart_seed, train, test, glm_w):
    """One restart of the global K-state GLM-HMM on fold f (Ashwood initialisation,
    identical to hmmGlm._fit_global_once except for the per-trial tolerance)."""
    tr_ch, tr_in, tr_mk = train
    te_ch, te_in, te_mk = test
    M = glm_w.shape[0]
    rng = np.random.default_rng(restart_seed)
    model = g._new_hmm(K, M)
    Wk = np.tile(glm_w, (K, 1)).reshape(K, 1, M)
    model.observations.Wk = Wk + g.INIT_NOISE * rng.standard_normal((K, 1, M))
    model.transitions.log_Ps = g._sticky_log_Ps(K, g.STICKY_SELF)
    lps = model.fit(tr_ch, inputs=tr_in, masks=tr_mk, method="em", num_iters=N_ITERS,
                    tolerance=TOL_PER_TRIAL * _n_obs(tr_mk), verbose=0, initialize=False)
    return {"fold": f, "K": K, "train_lp": float(lps[-1]), "n_em_iters": len(lps) - 1,
            "test_ll": model.log_likelihood(te_ch, inputs=te_in, masks=te_mk) / _n_obs(te_mk)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_all = time.time()
    print(f"[cv-fast] {os.cpu_count()} CPU cores; K = {K_RANGE}, {N_FOLDS} folds, "
          f"{N_RESTARTS} restarts, tolerance {TOL_PER_TRIAL:g} nats/trial", flush=True)

    df = g.select_glmhmm_sessions(pd.read_csv(CSV_PATH))
    choices, inputs, masks, tags = g.build_glmhmm_inputs(df, parametrization=PARAMETRIZATION)
    n_ses = len(choices)
    folds = g._session_folds([t[0] for t in tags], N_FOLDS, SEED)   # same split as hmmGlm
    data = []
    for te in folds:
        te = set(int(i) for i in te)
        tr = [i for i in range(n_ses) if i not in te]
        te = sorted(te)
        data.append(((g._split(choices, tr), g._split(inputs, tr), g._split(masks, tr)),
                     (g._split(choices, te), g._split(inputs, te), g._split(masks, te))))

    done = pd.read_csv(CHECKPOINT) if CHECKPOINT.exists() else pd.DataFrame()
    if len(done):
        print(f"[cv-fast] resuming: K already done = {sorted(done['K'].unique())}", flush=True)

    # ---- stage 1: GLM (K = 1) + lapse, all folds in parallel -------------
    t0 = time.time()
    stage1 = Parallel(n_jobs=min(N_FOLDS, os.cpu_count() or 1), verbose=10)(
        delayed(_glm_and_lapse_job)(f, data[f][0], data[f][1], SEED + f)
        for f in range(N_FOLDS))
    stage1 = sorted(stage1, key=lambda r: r["fold"])
    print(f"[cv-fast] GLM + lapse done in {(time.time() - t0) / 60:.1f} min", flush=True)

    cv_rows = [{"K": 1, "fold": r["fold"], "test_ll_per_trial": r["ll_glm"],
                "n_em_iters": np.nan} for r in stage1]
    cmp_rows = ([{"model": "GLM", "fold": r["fold"], "test_ll_per_trial": r["ll_glm"]}
                 for r in stage1]
                + [{"model": "lapse", "fold": r["fold"], "test_ll_per_trial": r["ll_lapse"]}
                   for r in stage1])
    if len(done):
        cv_rows += done.to_dict("records")

    # ---- stage 2: every (fold, restart) of each K is a parallel job ------
    for K in [k for k in K_RANGE if k > 1]:
        if len(done) and K in set(done["K"]):
            continue
        t0 = time.time()
        jobs = []
        for f in range(N_FOLDS):
            seeds = np.random.SeedSequence(SEED + f).generate_state(N_RESTARTS)
            jobs += [(f, int(s)) for s in seeds]
        res = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(_global_job)(f, K, s, data[f][0], data[f][1], stage1[f]["glm_w"])
            for f, s in jobs)
        res = pd.DataFrame(res)
        # keep, per fold, the restart with the best TRAINING log probability
        best = res.loc[res.groupby("fold")["train_lp"].idxmax()]
        rows = [{"K": K, "fold": int(r.fold), "test_ll_per_trial": r.test_ll,
                 "n_em_iters": int(r.n_em_iters)} for r in best.itertuples()]
        cv_rows += rows
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(CHECKPOINT, index=False)
        print(f"[cv-fast] K={K} done in {(time.time() - t0) / 60:.1f} min; EM iterations "
              f"(all restarts) median {res['n_em_iters'].median():.0f}, "
              f"max {res['n_em_iters'].max():.0f}; restarts agreeing with the best "
              f"(|test LL diff| < 1e-4): "
              f"{(res.groupby('fold')['test_ll'].transform(lambda x: (x - x.max()).abs() < 1e-4)).mean():.0%}",
              flush=True)

    cv = pd.DataFrame(cv_rows).drop_duplicates(["K", "fold"]).sort_values(["K", "fold"])
    cv.to_csv(OUTPUT_DIR / "glmhmm_cv.csv", index=False)
    k2 = cv[cv["K"] == K_COMPARISON]
    cmp_rows += [{"model": f"GLM-HMM (K={K_COMPARISON})", "fold": int(r.fold),
                  "test_ll_per_trial": r.test_ll_per_trial} for r in k2.itertuples()]
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUTPUT_DIR / "glmhmm_model_comparison.csv", index=False)

    # ---- report ------------------------------------------------------------
    print("\n=== Held-out LL per trial by K (mean +- SEM over folds) ===")
    s = cv.groupby("K")["test_ll_per_trial"].agg(["mean", "sem"])
    print(s.round(4).to_string())
    print("\nPaired gain vs the previous K (same folds):")
    piv = cv.pivot(index="fold", columns="K", values="test_ll_per_trial")
    for a, b in zip(piv.columns[:-1], piv.columns[1:]):
        d = piv[b] - piv[a]
        print(f"  K={a} -> K={b}: {d.mean():+.4f} +- {d.sem():.4f} nats/trial "
              f"({int((d > 0).sum())}/{len(d)} folds improve)")
    print("\n=== Model comparison (mean held-out LL per trial) ===")
    print(cmp.groupby("model")["test_ll_per_trial"].agg(["mean", "sem"]).round(4).to_string())
    print(f"\n[cv-fast] total {(time.time() - t_all) / 60:.1f} min. Wrote glmhmm_cv.csv, "
          f"glmhmm_model_comparison.csv in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
