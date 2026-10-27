"""
belief_vhr_neuromod.py
======================
Independent-cohort replication of the belief_vhr anticipation result on the
BEHAVIOUR of the neuromodulator (imaging) animals.

These 12 animals (NE: 1930x, 2033x; ACh: 1940x) were never used to fit any
behavioural model, so this is an out-of-sample test of the main finding
(b > 0: the internal hazard of a block switch grows with time since
criterion, beyond what the outcomes justify).

What it does
  1. Builds the trial tables exactly as glmhmm_heldout_validation.py does
     (phase 31, > 100 responded trials, >= 4 switches; same sessions as the
     GLM-HMM held-out validation).
  2. Pooled L_Random-restricted likelihood-ratio test, b free vs b = 0, with
     the same functions as belief_vhr_lrandom.py:
        - whole imaging cohort  (primary test)
        - NE and ACh cohorts separately (descriptive; fewer animals)
  3. Per-animal fits, DESCRIPTIVE only: the split-half analysis showed that
     per-animal b is not reliable (rho = 0.21), so no per-animal claims.
  4. Saves the cohort-level parameters (a, b, beta) to
     analysis/belief_vhr_neuromod_global.csv, the natural parameters for the
     latents used in the neural regression (objective 5) of these animals.

  5. With --loao: leave-one-animal-out robustness of the pooled test. The pooled
     LR treats trials as independent, so a few strong animals could carry it;
     refitting without each animal in turn shows whether b > 0 survives.

Run from reversalExpectation_py in the revExp env:
    python -X utf8 behavior\\beh_models\\belief_vhr_neuromod.py > analysis\\vhr_neuromod.log 2>&1
    python -X utf8 behavior\\beh_models\\belief_vhr_neuromod.py --loao > analysis\\vhr_neuromod_loao.log 2>&1
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit

from behavior.beh_models.belief_vhr_lrandom import _session_inputs, lr_test

# ---------------------------------------------------------------------------
# Config (selection identical to glmhmm_heldout_validation.py)
# ---------------------------------------------------------------------------
PHASES = [31]
MIN_RESPONDED, MIN_SWITCHES = 100, 3      # > 100 responded, block_idx.max() > 3 (>= 4 switches)
COHORT = {1.0: "NE", 2.0: "ACh"}
N_RESTARTS_POOLED = 8                     # as the global fit of the main cohort
N_RESTARTS_ANIMAL = 6                     # as the per-animal fits of the main cohort
N_RULES = 2                               # two-armed bandit
MAIN_GLOBAL = ROOT / "analysis" / "belief_vhr_lrandom_global.csv"
OUT_DIR = ROOT / "analysis"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tables() -> pd.DataFrame:
    """All passing imaging sessions, one row per trial, with a cohort column."""
    nm = ROOT / "neuromodulator"
    ntr = _load(nm / "neuromodulator_trials.py", "neuromodulator_trials")
    master = _load(nm / "master_neuromodulator.py", "master_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(PHASES)]
    tables, n_fail, n_err = [], 0, 0
    for _, r in idx.iterrows():
        ses = Path(str(r["session_file"])).stem
        try:
            d = ntr.build_trial_table(r["beh_path"], str(r["animal"]), ses)
        except Exception:
            n_err += 1
            continue
        n_resp, n_sw = int(d["responded"].sum()), int(d["block_idx"].max())
        if not (n_resp > MIN_RESPONDED and n_sw > MIN_SWITCHES):
            n_fail += 1
            continue
        d["cohort"] = COHORT.get(r.get("experiment"), "?")
        d["n_rules"] = N_RULES
        tables.append(d)
    df = pd.concat(tables, ignore_index=True)
    print(f"[neuromod] {df['session_file'].nunique()} sessions, {df['animal'].nunique()} animals "
          f"({n_fail} fail criteria, {n_err} unreadable)")
    return df


def sessions_of(df: pd.DataFrame):
    """[(animal, cohort, session inputs)] for sessions with >= 1 L_Random trial."""
    out = []
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        ci = _session_inputs(d)
        if ci[4].sum() > 0:
            out.append((animal, d["cohort"].iloc[0], ci))
    return out


def _report(lab, res):
    sig = f"p={res['p_boundary']:.3g}"
    print(f"  {lab:14s} b={res['b']:.3f}  LR={res['LR']:.1f}  {sig}  dBIC={res['dBIC']:+.1f}  "
          f"dLL/trial={res['dLL_per_trial']:+.5f}  n_scored={res['n_scored']}  "
          f"H(0)={expit(res['a']):.3f} H(10)={expit(res['a'] + 10 * res['b']):.3f} "
          f"H(20)={expit(res['a'] + 20 * res['b']):.3f}")


def _animal_fit(animal, cohort, cis, seed):
    res = lr_test(cis, n_restarts=N_RESTARTS_ANIMAL, rng=np.random.default_rng(seed))
    res.update(animal=animal, cohort=cohort, n_sessions=len(cis))
    return res


def _loao_fit(left_out, cis, seed):
    res = lr_test(cis, n_restarts=N_RESTARTS_POOLED, rng=np.random.default_rng(seed))
    res.update(left_out=left_out, n_sessions=len(cis))
    return res


def run_loao():
    """Pooled LR test refit 12 times, each time without one animal."""
    df = load_tables()
    sess = sessions_of(df)
    animals = sorted({a for (a, _, _) in sess})
    rows = Parallel(n_jobs=-1)(
        delayed(_loao_fit)(a, [ci for (b_, _, ci) in sess if b_ != a], i)
        for i, a in enumerate(animals))
    lo = pd.DataFrame(rows)
    cohort = {a: c for (a, c, _) in sess}
    lo["cohort_left_out"] = lo["left_out"].map(cohort)
    print("\n=== Leave-one-animal-out: pooled LR test without each animal ===")
    print(lo[["left_out", "cohort_left_out", "n_sessions", "b", "LR", "p_boundary",
              "dLL_per_trial", "n_scored"]].round(5).to_string(index=False))
    print(f"\n  b range {lo['b'].min():.3f} - {lo['b'].max():.3f} (median {lo['b'].median():.3f}); "
          f"p_boundary < 0.05 in {int((lo['p_boundary'] < 0.05).sum())}/{len(lo)} refits; "
          f"max p = {lo['p_boundary'].max():.3g} (without {lo.loc[lo['p_boundary'].idxmax(), 'left_out']})")
    lo.to_csv(OUT_DIR / "belief_vhr_neuromod_loao.csv", index=False)
    print(f"wrote {OUT_DIR / 'belief_vhr_neuromod_loao.csv'}")


def main():
    if "--loao" in sys.argv:
        run_loao()
        return
    df = load_tables()
    sess = sessions_of(df)

    print("\n=== Pooled L_Random LR test: belief_vhr (b free) vs belief (b = 0) ===")
    pooled = []
    groups = [("imaging cohort", [s for s in sess])] + \
             [(c, [s for s in sess if s[1] == c]) for c in ("NE", "ACh")]
    for lab, ss in groups:
        if not ss:
            continue
        res = lr_test([ci for (_, _, ci) in ss], n_restarts=N_RESTARTS_POOLED,
                      rng=np.random.default_rng(0))
        res.update(group=lab, n_sessions=len(ss), n_animals=len({a for (a, _, _) in ss}))
        _report(lab, res)
        pooled.append(res)

    # reference: the main (training) cohort
    if MAIN_GLOBAL.exists():
        g = pd.read_csv(MAIN_GLOBAL).iloc[0]
        print(f"  {'main cohort':14s} b={g['b']:.3f}  LR={g['LR']:.1f}  p={g['p_boundary']:.3g}  "
              f"dLL/trial={g['dLL_per_trial']:+.5f}  n_scored={int(g['n_scored'])}  (reference)")
        # expected LR here if the per-trial effect were the same as in the main cohort
        exp_lr = 2 * g["dLL_per_trial"] * pooled[0]["n_scored"]
        print(f"  -> LR expected with the main-cohort effect size: ~{exp_lr:.1f} "
              f"(power check: LR > 2.7 is p < 0.05 at the boundary)")

    print("\n=== Per animal (DESCRIPTIVE: per-animal b is unreliable, split-half rho = 0.21) ===")
    by_animal = {}
    for a, c, ci in sess:
        by_animal.setdefault((a, c), []).append(ci)
    rows = Parallel(n_jobs=-1)(delayed(_animal_fit)(a, c, cis, i)
                               for i, ((a, c), cis) in enumerate(sorted(by_animal.items())))
    pa = pd.DataFrame(rows).sort_values(["cohort", "animal"])
    print(pa[["cohort", "animal", "n_sessions", "b", "LR", "p_boundary", "n_scored"]]
          .round(4).to_string(index=False))
    b = pa["b"].to_numpy()
    print(f"  median b={np.median(b):.3f}, {100 * np.mean(b > 1e-3):.0f}% > 0, "
          f"{int(np.sum(b <= 1e-3))} at the b = 0 boundary (n={len(b)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pooled).to_csv(OUT_DIR / "belief_vhr_neuromod_pooled.csv", index=False)
    pa.to_csv(OUT_DIR / "belief_vhr_neuromod_per_animal.csv", index=False)
    # cohort-level parameters, same layout as belief_vhr_lrandom_global.csv
    pd.DataFrame([pooled[0]]).to_csv(OUT_DIR / "belief_vhr_neuromod_global.csv", index=False)
    print(f"\nwrote belief_vhr_neuromod_pooled.csv, _per_animal.csv, _global.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
