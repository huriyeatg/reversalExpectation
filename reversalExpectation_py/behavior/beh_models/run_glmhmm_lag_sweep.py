"""
run_glmhmm_lag_sweep.py
========================
Option (b) from the parametrization discussion: instead of the faithful
Ashwood design (`ashwood_wsls`, fixed lag-1 history: bias, prev_choice,
wsls), try `reward_perseveration` (separate reward-seeking and perseveration
kernels, extendable to multiple trials back) and pick the number of lags by
held-out log-likelihood, the same way K was chosen for the state count.

This is a SEPARATE script from run_glmhmm.py on purpose — it doesn't touch
your working pipeline file. Once you're happy with a lag choice here, port
PARAMETRIZATION / N_LAGS into run_glmhmm.py and re-run the rest of the
pipeline (anticipation, occupancy, etc.) as usual.

Gotcha fixed here that would silently break if you just flipped
PARAMETRIZATION in run_glmhmm.py: the "engaged" state is picked as the state
with the strongest PERSEVERATION weight. In ashwood_wsls that column is
literally called "prev_choice"; in reward_perseveration the equivalent
(lag-1 perseveration kernel) is called "choice_1". `_engaged_column()` below
picks the right one so this doesn't KeyError.

Requires the `ssm` conda env (same as run_glmhmm.py). Run from
behavior/beh_models/:

    conda activate ssm
    python run_glmhmm_lag_sweep.py

Cost note: this is N_LAGS_CANDIDATES x N_FOLDS x N_RESTARTS fits, i.e. several
times the cost of the K-selection CV in run_glmhmm.py. Start with
SUBSET_N_ANIMALS set small and/or fewer restarts to gauge timing before the
full run.
"""

import sys
import time
import os
from pathlib import Path

# Windows + joblib + ssm/BLAS: cap each process to 1 BLAS thread so that
# parallel workers do not oversubscribe cores and deadlock. MUST run before
# numpy is imported to take effect. (This was missing from the original
# version of this script -- run_glmhmm.py has it, this didn't. Very likely
# culprit for a run that takes hours with N_JOBS=-1.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import hmmGlm as g                 # noqa: E402
import glmhmm_occupancy as occ     # noqa: E402

# ===========================================================================
# Config
# ===========================================================================
CSV_PATH   = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUTPUT_DIR = REPO_ROOT / "analysis"

PARAMETRIZATION   = "reward_perseveration"
LAG_CANDIDATES    = [1, 2, 3, 4, 5]     # trials of history in the design matrix
CHOSEN_K          = 2                    # keep K fixed -- this run is only about lags

N_FOLDS     = 5
N_RESTARTS  = 5     # drop this for a faster first pass (e.g. 5) if timing is a concern
N_ITERS     = 200
N_JOBS      = -1

SUBSET_N_ANIMALS = 5   # set to e.g. 5 for a quick timing check before the full run

# Baseline to compare against, from choice_quality_by_state_lrandom on the
# current ashwood_wsls / K=2 fit (see corroborate_choice_quality_lrandom.py).
BASELINE_ASHWOOD_WSLS = {"p_better_exploit": 0.921, "p_better_explore": 0.555}


def _engaged_column(parametrization: str) -> str:
    """Name of the lag-1 perseveration regressor, which differs by
    parametrization -- this is the fix for the KeyError you'd get by just
    flipping PARAMETRIZATION in run_glmhmm.py without changing this."""
    if parametrization == "ashwood_wsls":
        return "prev_choice"
    elif parametrization == "reward_perseveration":
        return "choice_1"     # lag-1 perseveration kernel; see regressor_names()
    raise ValueError(f"unknown parametrization: {parametrization!r}")


def attach_glmhmm_states(df, post, tags, engaged_idx):
    """Same as run_glmhmm.py's version -- map per-session posteriors back onto
    df rows by (animal, session_file) tag."""
    state = pd.Series(np.nan, index=df.index)
    p_eng = pd.Series(np.nan, index=df.index)
    for (animal, ses), P in zip(tags, post):
        rows = df.index[(df["animal"] == animal) & (df["session_file"] == ses)]
        if len(rows) != P.shape[0]:
            raise ValueError(f"length mismatch for {(animal, ses)}: "
                             f"{len(rows)} df rows vs {P.shape[0]} posterior rows")
        state.loc[rows] = P.argmax(1)
        p_eng.loc[rows] = P[:, engaged_idx]
    return df.assign(glmhmm_state=state, p_engaged=p_eng)


def main():
    df = pd.read_csv(CSV_PATH)
    df = g.select_glmhmm_sessions(df)

    if SUBSET_N_ANIMALS is not None:
        top = (df.groupby("animal")["session_file"].nunique()
                 .sort_values(ascending=False).head(SUBSET_N_ANIMALS).index.tolist())
        df = df[df["animal"].isin(top)].copy()
        print(f"[subset] {SUBSET_N_ANIMALS} animals -> "
              f"{df.groupby(['animal','session_file']).ngroups} sessions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. lag selection by held-out LL (K fixed at CHOSEN_K) ------------
    print(f"\n=== Lag selection: {PARAMETRIZATION}, K={CHOSEN_K}, "
          f"lags={LAG_CANDIDATES} ===")
    rows = []
    for L in LAG_CANDIDATES:
        t0 = time.time()
        cv = g.cross_validate_glmhmm(
            df, K_range=(CHOSEN_K,), n_lags=L, parametrization=PARAMETRIZATION,
            n_folds=N_FOLDS, n_restarts=N_RESTARTS, n_iters=N_ITERS, n_jobs=N_JOBS)
        cv["n_lags"] = L
        rows.append(cv)
        mean_ll = cv["test_ll_per_trial"].mean()
        print(f"  n_lags={L}: held-out LL/trial = {mean_ll:.4f}  "
              f"({time.time()-t0:.0f}s)")

    lag_cv = pd.concat(rows, ignore_index=True)
    lag_cv.to_csv(OUTPUT_DIR / f"glmhmm_lag_selection_K{CHOSEN_K}.csv", index=False)
    summary = lag_cv.groupby("n_lags")["test_ll_per_trial"].mean()
    print("\nMean held-out LL/trial by n_lags:")
    print(summary.round(4).to_string())
    best_lag = int(summary.idxmax())
    print(f"best n_lags (by held-out LL) = {best_lag}  "
          "-- eyeball the table above for a plateau before trusting the argmax "
          "the same way K=2 was chosen over K=3/4.")

    # ---- 2. final fit at the best lag --------------------------------------
    print(f"\n=== Final fit: {PARAMETRIZATION}, K={CHOSEN_K}, n_lags={best_lag} ===")
    ch, inp, mk, tags = g.build_glmhmm_inputs(df, n_lags=best_lag,
                                              parametrization=PARAMETRIZATION)
    _, glm_w = g.fit_glm(ch, inp, mk, n_jobs=N_JOBS)
    model = g.fit_global_glmhmm(ch, inp, mk, K=CHOSEN_K, glm_weights=glm_w,
                                n_restarts=N_RESTARTS, n_iters=N_ITERS, n_jobs=N_JOBS)
    weights_df = g.glmhmm_weights(model, n_lags=best_lag, parametrization=PARAMETRIZATION)
    print("per-state weights (log-odds of choosing right):")
    print(weights_df.round(3).to_string(index=False))

    engaged_col = _engaged_column(PARAMETRIZATION)
    engaged_idx = int(weights_df.set_index("state")[engaged_col].idxmax())
    print(f"engaged/exploit state index = {engaged_idx}  "
          f"(by max '{engaged_col}' weight)")

    # ---- 3. derive states, write cache -------------------------------------
    post = g.glmhmm_posteriors(model, ch, inp, mk)
    dfo = attach_glmhmm_states(df, post, tags, engaged_idx)
    dfo = dfo[dfo["glmhmm_state"].notna()].copy()
    states_csv = OUTPUT_DIR / f"glmhmm_states_{PARAMETRIZATION}_lag{best_lag}_K{CHOSEN_K}.csv"
    keep = [c for c in ["animal", "session_file", "block_idx", "trial_idx",
                        "glmhmm_state", "p_engaged"] if c in dfo.columns]
    dfo[keep].to_csv(states_csv, index=False)
    print(f"wrote {states_csv}")

    # ---- 4. corroborate against the ashwood_wsls baseline, same L_Random   -
    #         convention as everything else (see modeling-principles note:    -
    #         evaluate in the pre-switch L_Random window by default).         -
    if engaged_idx != 0:
        # choice_quality_by_state_lrandom's default exploit/explore state
        # indices assume 0=exploit; remap if the fit put engaged at 1.
        exploit_state, explore_state = engaged_idx, 1 - engaged_idx
    else:
        exploit_state, explore_state = 0, 1

    res = occ.choice_quality_by_state_lrandom(
        dfo, exploit_state=exploit_state, explore_state=explore_state)

    print(f"\n=== choice_quality_by_state_lrandom "
          f"({PARAMETRIZATION}, n_lags={best_lag}) vs ashwood_wsls baseline ===")
    print(f"{'':22s}{'ashwood_wsls (n_lags=1)':>25s}{'reward_perseveration':>25s}")
    print(f"{'P(better|exploit)':22s}"
          f"{BASELINE_ASHWOOD_WSLS['p_better_exploit']:>25.3f}"
          f"{res['p_better_exploit']:>25.3f}")
    print(f"{'P(better|explore)':22s}"
          f"{BASELINE_ASHWOOD_WSLS['p_better_explore']:>25.3f}"
          f"{res['p_better_explore']:>25.3f}")
    print(f"\nWilcoxon p = {res['wilcoxon_p']:.3g}  (n animals = {res['n']}), "
          f"fraction exploit>explore = {res['frac_pos']:.3f}")


if __name__ == "__main__":
    main()
