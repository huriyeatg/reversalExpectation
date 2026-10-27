"""
neuromodulator_state_hierarchical.py
====================================
Animal-aware statistics for the objective-5 result (neuromodulator_state_regression.py).

The session-level test printed by neuromodulator_state_regression.py treats
sessions as independent, but sessions are nested in animals (ACh: 4 animals,
NE: 7). This script re-analyses session_summary.csv WITHOUT refitting anything,
and answers two different questions:

  1. Fixed-effects test, "is there an effect in THESE animals?"
     Statistic: the excess fraction of modulated sessions over the circular-
     shift null, averaged within animal first and then across animals, so every
     animal weighs the same regardless of its number of sessions.
     Null distribution: each session's modulation is redrawn as
     Bernoulli(null fraction of that session), which is exactly the false-
     positive rate of the same criterion under the circular-shift null, and the
     statistic is recomputed (parametric bootstrap, N_SIM draws).
     Valid inference about the recorded animals; does not generalize to the
     population of mice.

  2. Hierarchical bootstrap, "would it generalize to new animals?"
     Animals are resampled with replacement, then sessions within each resampled
     animal; the 95% CI of the same statistic is reported. With 4 ACh animals
     this CI is necessarily wide: it is the honest statement of how much the
     population-level claim is limited by n.

  3. Sensitivity of the fixed-effects test:
     - session-weighted version (every session weighs the same), and
     - leave-one-animal-out (the animal-weighted test without each animal),
     because animals with very few sessions weigh as much as the others in (1).

  4. Benjamini-Hochberg FDR across all cohort x predictor tests of (1).

Only numpy and pandas are needed.

Run from reversalExpectation_py (revExp env), after neuromodulator_state_regression.py:
    python -X utf8 neuromodulator\\neuromodulator_state_hierarchical.py > analysis\\obj5_hierarchical.log 2>&1
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "data" / "figs" / "neuromodulator-state" / "session_summary.csv"
OUT_CSV = ROOT / "data" / "figs" / "neuromodulator-state" / "hierarchical_stats.csv"

TESTS = ["state_F", "belief_F", "p_exploit", "confidence", "v_chosen", "abs_rpe", "tau"]
N_SIM = 20000          # null draws for the fixed-effects test
N_BOOT = 20000         # hierarchical bootstrap resamples
SEED = 20261025        # fixed seed: identical output on every run


def animal_weighted_excess(mod: np.ndarray, null: np.ndarray, groups: list) -> float:
    """Mean over animals of (mean over that animal's sessions of mod - null)."""
    d = mod - null
    return float(np.mean([d[idx].mean() for idx in groups]))


def fixed_effects_test(mod, null, groups, rng):
    """One-sided p: P(statistic under the null >= observed)."""
    obs = animal_weighted_excess(mod, null, groups)
    # redraw every session's modulation from its own null false-positive rate
    sims = rng.random((N_SIM, len(null))) < null[None, :]
    d = sims.astype(float) - null[None, :]
    stat = np.mean(np.stack([d[:, idx].mean(axis=1) for idx in groups], axis=1), axis=1)
    p = (np.sum(stat >= obs) + 1) / (N_SIM + 1)   # +1: never report p = 0
    return obs, float(p)


def hierarchical_bootstrap(mod, null, groups, rng):
    """95% CI of the animal-weighted excess, resampling animals then sessions."""
    d = mod - null
    n_an = len(groups)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, n_an, n_an)                    # animals with replacement
        means = []
        for a in pick:
            idx = groups[a]
            means.append(d[rng.choice(idx, size=len(idx), replace=True)].mean())
        out[b] = np.mean(means)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(out <= 0))


def session_weighted_test(mod, null, rng):
    """One-sided p for the total number of modulated sessions, sessions weighted equally."""
    obs = mod.sum()
    sims = (rng.random((N_SIM, len(null))) < null[None, :]).sum(axis=1)
    return float((np.sum(sims >= obs) + 1) / (N_SIM + 1))


def leave_one_animal_out(mod, null, groups, animals, rng):
    """Animal-weighted fixed-effects test repeated without each animal."""
    out = []
    for j, a in enumerate(animals):
        keep = [g for i, g in enumerate(groups) if i != j]
        if not keep:
            continue
        obs, p = fixed_effects_test(mod, null, keep, rng)
        out.append(f"-{a}:{obs:+.2f}/p={p:.3f}")
    return " ".join(out)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def main():
    s = pd.read_csv(SUMMARY)
    s = s[s["status"] == "ok"].copy()
    s["animal"] = s["animal"].astype(str)
    rng = np.random.default_rng(SEED)

    rows = []
    for coh, g in s.groupby("cohort"):
        g = g.reset_index(drop=True)
        animals = sorted(g["animal"].unique())
        groups = [np.flatnonzero(g["animal"].to_numpy() == a) for a in animals]
        for k in TESTS:
            mod = g[f"mod_{k}"].astype(float).to_numpy()
            null = g[f"null_{k}"].astype(float).to_numpy()
            obs, p_fixed = fixed_effects_test(mod, null, groups, rng)
            lo, hi, p_boot = hierarchical_bootstrap(mod, null, groups, rng)
            per_animal = [float((mod[i] - null[i]).mean()) for i in groups]
            p_sess = session_weighted_test(mod, null, rng)
            loao = leave_one_animal_out(mod, null, groups, animals, rng)
            rows.append({"cohort": coh, "test": k, "n_animals": len(animals),
                         "n_sessions": len(g), "excess": obs,
                         "p_fixed_effects": p_fixed, "boot_ci_low": lo,
                         "boot_ci_high": hi, "boot_frac_le_0": p_boot,
                         "p_session_weighted": p_sess, "leave_one_animal_out": loao,
                         "animals_positive": int(np.sum(np.array(per_animal) > 0)),
                         "per_animal_excess": " ".join(f"{a}:{v:+.2f}" for a, v in
                                                       zip(animals, per_animal))})

    res = pd.DataFrame(rows)
    res["p_fixed_fdr_bh"] = bh_fdr(res["p_fixed_effects"].to_numpy())
    res.to_csv(OUT_CSV, index=False)

    print("Excess = fraction of modulated sessions minus the circular-shift null,")
    print("averaged within animal first (every animal weighs the same).")
    print("p_fixed: these animals (parametric draw from each session's null).")
    print("CI: hierarchical bootstrap, animals then sessions (generalization).\n")
    for coh, g in res.groupby("cohort"):
        print(f"=== {coh}: {g['n_animals'].iloc[0]} animals, {g['n_sessions'].iloc[0]} sessions ===")
        for _, r in g.iterrows():
            print(f"  {r['test']:11s} excess {r['excess']:+.3f}  p_fixed={r['p_fixed_effects']:.4f}  "
                  f"CI95 [{r['boot_ci_low']:+.3f}, {r['boot_ci_high']:+.3f}]  "
                  f"animals>0 {r['animals_positive']}/{r['n_animals']}")
            print(f"  {'':11s} per animal: {r['per_animal_excess']}")
            print(f"  {'':11s} p_session_weighted={r['p_session_weighted']:.4f}  "
                  f"p_fdr_bh={r['p_fixed_fdr_bh']:.3f} ({len(res)} tests)")
            print(f"  {'':11s} leave-one-animal-out: {r['leave_one_animal_out']}")
        print()
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
