"""
belief_vhr_tau_switch_check.py
==============================
Sensitivity check for the tau indexing at the real block switch.

belief_vhr applies the end-of-trial transition with H = sigmoid(a + b * tau[t+1]).
On the last trial of a block, tau[t+1] belongs to the NEW block, which is in its
criterion phase (tau clipped to 0). So exactly at the true switch the model uses
H(0) instead of the elevated H(L_Random) of the block that is ending.

This script refits the pooled L_Random LR test twice on the same sessions:
  original : tau as built by prepare_sessions (current analysis);
  carried  : on the first trial of every block after the first, tau is replaced
             by (tau of the previous trial + 1), so the transition into the new
             block uses the old block's clock. tau inside each block, and the
             L_Random scoring mask, are unchanged.
If b, LR and dBIC barely move, the reset is irrelevant for the anticipation result.

Run from the repo root in the revExp environment:
    python -X utf8 behavior\\beh_models\\belief_vhr_tau_switch_check.py > analysis\\vhr_tau_switch_check.log 2>&1
Output: analysis/belief_vhr_tau_switch_check.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root importable

import numpy as np
import pandas as pd

from behavior.beh_models.belief_vhr_lrandom import (
    select_naive_meets_criteria, _session_inputs, lr_test)

CSV = "analysis/bandit_R71_lesion.csv"
N_RESTARTS = 8          # same as the main global fit
OUT = Path("analysis/belief_vhr_tau_switch_check.csv")


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
    sessions, carried = [], []
    for (_animal, _ses), df_ses in df.groupby(["animal", "session_file"], sort=False):
        c, r, n_rules, tau, mask = _session_inputs(df_ses)
        if mask.sum() == 0:                       # no L_Random window: dropped, as in the main fit
            continue
        sessions.append((c, r, n_rules, tau, mask))
        carried.append((c, r, n_rules, carry_tau_over_switch(df_ses, tau), mask))
    n_changed = sum(int(np.sum(s1[3] != s0[3])) for s0, s1 in zip(sessions, carried))
    print(f"[tau_switch] {len(sessions)} sessions; tau changed on {n_changed} trials "
          f"(first trial of each new block)")

    rows = []
    for name, sess in [("original", sessions), ("carried", carried)]:
        res = lr_test(sess, n_restarts=N_RESTARTS, rng=np.random.default_rng(0))
        res["version"] = name
        rows.append(res)
        print(f"[tau_switch] {name:>8}: a={res['a']:.4f}  b={res['b']:.4f}  beta={res['beta']:.4f}  "
              f"LR={res['LR']:.1f}  p={res['p_boundary']:.3g}  dBIC={res['dBIC']:+.1f}  "
              f"({time.time() - t0:.0f} s)")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[tau_switch] wrote {OUT}")


if __name__ == "__main__":
    main()
