"""
belief_vhr_tau_switch_check.py
==============================
Sensitivity check for the tau indexing at the real block switch.

belief_vhr applies the end-of-trial transition with H = sigmoid(a + b * tau[t+1]).
On the last trial of a block, tau[t+1] belongs to the NEW block, which is in its
criterion phase (tau clipped to 0). So exactly at the true switch the model uses
H(0) instead of the elevated H(L_Random) of the block that is ending.

This script refits the pooled L_Random LR test with a modified clock and compares
it with the saved main fit (analysis/belief_vhr_lrandom_global.csv):
  original : tau as built by prepare_sessions (read from the saved global fit);
  carried  : on the first trial of every block after the first, tau is replaced
             by (tau of the previous trial + 1), so the transition into the new
             block uses the old block's clock. tau inside each block, and the
             L_Random scoring mask, are unchanged.
If b, LR and dBIC barely move, the reset is irrelevant for the anticipation result.

Run from the repo root in the revExp environment:
    python -u -X utf8 behavior\\beh_models\\belief_vhr_tau_switch_check.py > analysis\\vhr_tau_switch_check.log 2>&1
(-u: unbuffered, so progress appears in the log while it runs). Restarts run in
parallel, one per core.
Output: analysis/belief_vhr_tau_switch_check.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root importable

import numpy as np
import pandas as pd

from functools import partial
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

from behavior.beh_models.belief_vhr_lrandom import (
    select_naive_meets_criteria, _session_inputs, _pooled_nll)

print = partial(print, flush=True)       # progress visible in the log while running

CSV = "analysis/bandit_R71_lesion.csv"
N_RESTARTS = 8          # same as the main global fit
OUT = Path("analysis/belief_vhr_tau_switch_check.csv")
GLOBAL = Path("analysis/belief_vhr_lrandom_global.csv")   # saved main fit (original tau)


def _one_restart(sessions, b_free, x0):
    """One L-BFGS-B run from x0; same bounds as belief_vhr_lrandom._fit_pooled."""
    bounds = [(-12, 12), (0, 5), (1e-3, 100)] if b_free else [(-12, 12), (1e-3, 100)]
    res = minimize(_pooled_nll, x0, args=(sessions, b_free), method="L-BFGS-B", bounds=bounds)
    return res.fun, res.x


def lr_test_parallel(sessions, n_restarts=N_RESTARTS, seed=0) -> dict:
    """Same test as belief_vhr_lrandom.lr_test, with all restarts of the full
    (b free) and reduced (b = 0) fits run in parallel."""
    rng = np.random.default_rng(seed)
    x_full = [[rng.uniform(-4, 0), rng.uniform(0, 0.5), rng.uniform(0.5, 15)] for _ in range(n_restarts)]
    x_red = [[rng.uniform(-4, 0), rng.uniform(0.5, 15)] for _ in range(n_restarts)]
    jobs = [(True, x) for x in x_full] + [(False, x) for x in x_red]
    out = Parallel(n_jobs=-1, verbose=5)(delayed(_one_restart)(sessions, bf, x) for bf, x in jobs)
    nll_full, par_full = min(out[:n_restarts], key=lambda o: o[0])
    nll_red, par_red = min(out[n_restarts:], key=lambda o: o[0])
    n_scored = sum(int(np.sum(m & ~np.isnan(c))) for (c, r, nr, tau, m) in sessions)
    LR = max(2 * (nll_red - nll_full), 0.0)
    a, b, beta = par_full
    return dict(a=float(a), b=float(b), beta=float(beta), a0=float(par_red[0]), LR=float(LR),
                p_boundary=0.5 * float(chi2.sf(LR, 1)), dBIC=float(LR - np.log(n_scored)),
                dLL_per_trial=float((nll_red - nll_full) / n_scored), n_scored=int(n_scored),
                base_hazard=float(expit(a)))


def carry_tau_over_switch(df_ses: pd.DataFrame, tau: np.ndarray) -> np.ndarray:
    """Return a copy of tau where the first trial of each new block carries the
    previous block's clock forward (previous tau + 1)."""
    tau2 = tau.copy()
    block = df_ses["block_idx"].to_numpy()
    first_of_block = np.r_[False, block[1:] != block[:-1]]     # skip the session's first block
    for t in np.flatnonzero(first_of_block):
        tau2[t] = tau[t - 1] + 1.0
    return tau2


def main():
    t0 = time.time()
    df = pd.read_csv(CSV)
    df = select_naive_meets_criteria(df)

    # build both versions from the same loop as prepare_sessions (same sessions, same order)
    sessions, carried = [], []           # sessions (original tau) kept only to count changed trials
    for (_animal, _ses), df_ses in df.groupby(["animal", "session_file"], sort=False):
        c, r, n_rules, tau, mask = _session_inputs(df_ses)
        if mask.sum() == 0:                       # no L_Random window: dropped, as in the main fit
            continue
        sessions.append((c, r, n_rules, tau, mask))
        carried.append((c, r, n_rules, carry_tau_over_switch(df_ses, tau), mask))
    n_changed = sum(int(np.sum(s1[3] != s0[3])) for s0, s1 in zip(sessions, carried))
    print(f"[tau_switch] {len(sessions)} sessions; tau changed on {n_changed} trials "
          f"(first trial of each new block)")

    orig = pd.read_csv(GLOBAL).iloc[0]
    rows = [dict(version="original", **{k: float(orig[k]) for k in
                                        ["a", "b", "beta", "a0", "LR", "p_boundary", "dBIC", "dLL_per_trial"]})]
    res = lr_test_parallel(carried)
    res["version"] = "carried"
    rows.append(res)
    for r in rows:
        print(f"[tau_switch] {r['version']:>8}: a={r['a']:.4f}  b={r['b']:.4f}  beta={r['beta']:.4f}  "
              f"LR={r['LR']:.1f}  p={r['p_boundary']:.3g}  dBIC={r['dBIC']:+.1f}")
    print(f"[tau_switch] total {(time.time() - t0) / 60:.1f} min")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[tau_switch] wrote {OUT}")


if __name__ == "__main__":
    main()
