"""
belief_vhr_splithalf.py
=======================
Split-half reliability of the per-animal hazard slope b of belief_vhr.

Question: is an animal's anticipation (b > 0) a stable trait, or does the
per-animal b mostly reflect fitting noise? Each animal's sessions are sorted
chronologically and split into two interleaved halves (odd / even sessions,
so slow learning drift lands in BOTH halves). The same L_Random-restricted
likelihood-ratio fit used in belief_vhr_lrandom.py is run on each half, and
the two b estimates are correlated across animals.

Reported:
  - Spearman rho between half-A and half-B b (all animals);
  - the same excluding "boundary" animals, i.e. those whose full-data fit sits
    at b = 0 (their b is not estimable, only bounded; they were shown as grey
    points in the forest plot);
  - Spearman-Brown corrected reliability of the full-data b, 2r / (1 + r);
  - a figure (half A vs half B) and a CSV with every per-half fit.

Inputs (from reversalExpectation_py):
  - analysis/bandit_R71_lesion.csv
  - analysis/belief_vhr_lrandom_per_animal.csv  (full-data per-animal fits;
    produced by belief_vhr_lrandom.py, used only to flag boundary animals)

Run from reversalExpectation_py in the revExp env:
    python -X utf8 behavior\\beh_models\\belief_vhr_splithalf.py > analysis\\vhr_splithalf.log 2>&1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# make `behavior.beh_models...` importable when run by path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from behavior.beh_models.belief_vhr_lrandom import (
    select_naive_meets_criteria, _session_inputs, lr_test)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = Path("analysis/bandit_R71_lesion.csv")
FULL_FITS = Path("analysis/belief_vhr_lrandom_per_animal.csv")
OUT_CSV = Path("analysis/belief_vhr_splithalf.csv")
OUT_FIG = Path("figs/belief_vhr_splithalf.png")
N_RESTARTS = 6          # same as the per-animal fits in belief_vhr_lrandom.py
B_BOUNDARY = 1e-3       # b below this in the FULL fit = boundary animal
MIN_SESSIONS_HALF = 3   # an animal needs >= this many sessions in EACH half
N_JOBS = -1


def _session_time(session_file: str) -> int:
    """Chronological key from the file name: the last run of >= 10 digits
    (yymmddHHMM in '1806_phase3_R71_NoCue_1805101101.log'). Falls back to -1,
    in which case the original CSV order is kept (stable sort)."""
    m = re.findall(r"(\d{10,})", str(session_file))
    return int(m[-1]) if m else -1


def build_halves(df: pd.DataFrame) -> dict:
    """{animal: {"A": [session inputs...], "B": [...], "n": n_sessions}}.
    Sessions sorted chronologically, then alternated A, B, A, B..."""
    rows = []
    for (animal, ses), df_ses in df.groupby(["animal", "session_file"], sort=False):
        ci = _session_inputs(df_ses)
        if ci[4].sum() > 0:                       # >= 1 L_Random trial, as in the main test
            rows.append((animal, ses, _session_time(ses), ci))
    by_animal = {}
    for animal, ses, t, ci in rows:
        by_animal.setdefault(animal, []).append((t, ses, ci))
    halves = {}
    for animal, lst in by_animal.items():
        lst.sort(key=lambda x: x[0])              # stable: ties keep CSV order
        A = [ci for i, (_, _, ci) in enumerate(lst) if i % 2 == 0]
        B = [ci for i, (_, _, ci) in enumerate(lst) if i % 2 == 1]
        halves[animal] = {"A": A, "B": B, "n": len(lst)}
    return halves


def _fit_half(animal, half, sessions, seed):
    res = lr_test(sessions, n_restarts=N_RESTARTS, rng=np.random.default_rng(seed))
    res.update(animal=animal, half=half, n_sessions=len(sessions))
    return res


def _rho(x, y):
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    r, p = spearmanr(x, y)
    return float(r), float(p), len(x)


def main():
    df = pd.read_csv(CSV_PATH)
    df = select_naive_meets_criteria(df)
    halves = build_halves(df)

    # animals with enough sessions in both halves
    keep = {a: h for a, h in halves.items()
            if len(h["A"]) >= MIN_SESSIONS_HALF and len(h["B"]) >= MIN_SESSIONS_HALF}
    dropped = sorted(set(halves) - set(keep))
    print(f"[splithalf] {len(keep)} animals with >= {MIN_SESSIONS_HALF} sessions per half"
          + (f"; dropped {dropped}" if dropped else ""))

    jobs = [(a, hname, h[hname], 1000 * i + j)
            for i, (a, h) in enumerate(sorted(keep.items(), key=lambda kv: str(kv[0])))
            for j, hname in enumerate(("A", "B"))]
    rows = Parallel(n_jobs=N_JOBS)(delayed(_fit_half)(*jb) for jb in jobs)
    fits = pd.DataFrame(rows)

    wide = fits.pivot(index="animal", columns="half", values="b")
    wide.columns = ["b_A", "b_B"]
    wide["n_sessions"] = fits.groupby("animal")["n_sessions"].sum()

    # flag boundary animals from the FULL-data per-animal fit
    if FULL_FITS.exists():
        full = pd.read_csv(FULL_FITS)[["animal", "b"]].rename(columns={"b": "b_full"})
        full["animal"] = full["animal"].astype(wide.index.dtype)
        wide = wide.join(full.set_index("animal"))
        wide["boundary"] = wide["b_full"] < B_BOUNDARY
    else:
        print(f"NOTE: {FULL_FITS} not found; boundary animals cannot be flagged")
        wide["b_full"] = np.nan
        wide["boundary"] = False

    # also flag animals that hit the boundary in either half (b not estimable there)
    wide["half_at_boundary"] = (wide["b_A"] < B_BOUNDARY) | (wide["b_B"] < B_BOUNDARY)

    r_all = _rho(wide["b_A"], wide["b_B"])
    nb = wide[~wide["boundary"]]
    r_nb = _rho(nb["b_A"], nb["b_B"])
    clean = wide[~wide["boundary"] & ~wide["half_at_boundary"]]
    r_cl = _rho(clean["b_A"], clean["b_B"])

    def sb(r):                                   # Spearman-Brown: reliability of full-length b
        return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan

    print("\n=== Split-half reliability of per-animal b (odd vs even sessions) ===")
    for lab, (r, p, n) in [("all animals", r_all),
                           ("excl. full-fit boundary (b=0)", r_nb),
                           ("excl. boundary in full fit OR either half", r_cl)]:
        print(f"  {lab:44s} rho={r:+.3f}  p={p:.3g}  n={n}  "
              f"Spearman-Brown={sb(r):+.3f}")
    print(f"\n  boundary animals (full fit): {int(wide['boundary'].sum())} of {len(wide)}")
    print(f"  median b: half A {wide['b_A'].median():.3f}, half B {wide['b_B'].median():.3f}, "
          f"full {wide['b_full'].median():.3f}")
    print(f"  % b > 0: half A {100*(wide['b_A'] > B_BOUNDARY).mean():.0f}%, "
          f"half B {100*(wide['b_B'] > B_BOUNDARY).mean():.0f}%")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.reset_index().to_csv(OUT_CSV, index=False)
    fits.to_csv(OUT_CSV.with_name("belief_vhr_splithalf_fits.csv"), index=False)

    # figure: half A vs half B, boundary animals in grey
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lim = [0, max(0.05, float(np.nanmax(wide[["b_A", "b_B"]].to_numpy())) * 1.08)]
    ax.plot(lim, lim, ls=":", color="k", lw=1)
    est = ~wide["boundary"]
    ax.scatter(wide.loc[~est, "b_A"], wide.loc[~est, "b_B"], color="0.65",
               s=30, label="boundary in full fit", zorder=2)
    ax.scatter(wide.loc[est, "b_A"], wide.loc[est, "b_B"], color="#2A9D8F",
               edgecolor="k", lw=0.4, s=34, label="estimable", zorder=3)
    ax.set(xlim=lim, ylim=lim, xlabel="b, half A (odd sessions)",
           ylabel="b, half B (even sessions)",
           title=f"Split-half b  (ρ={r_all[0]:+.2f} all; {r_nb[0]:+.2f} excl. boundary)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT_CSV}, {OUT_FIG}")


if __name__ == "__main__":
    main()
