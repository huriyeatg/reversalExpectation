"""
neuromodulator_state_baseline.py
================================
ALTERNATIVE analysis to the per-bin regression of neuromodulator_state_regression.py
(objective 5). It does not replace it: the per-bin analysis stays the primary,
Wang-style result; this one asks the same question with a different summary of
the signal.

Why: in the per-bin analysis the ACh coefficients of the GLM-HMM state and of
the value of the chosen option are small, of constant sign, and present over
the whole -3..5 s window, including before the cue. A sustained effect is
spread over ~80 weak per-bin tests there. Summarising the signal as ONE number
per trial and window gives a single, better-powered test per session. The
pre-cue window has an interpretive advantage too: it precedes the choice and
the outcome of the same trial, so neither can cause it.

Per session (same sessions, alignment and predictors as the per-bin analysis,
via build_session_design):
  y_w(trial) = mean dF/F over the bins of window w, z-scored within session
               (so coefficients are in SD of the signal and comparable across
               sessions and animals with different dF/F scales)
  M1 base (Wang) / M2 + state / M3 + belief, one OLS each; recorded:
    - coefficient of every added predictor (from M3)
    - partial F of M2 vs M1 and of M3 vs M2
  Null: N_SHIFT circular shifts of the ADDED regressors (same construction and
  stable seeds as the per-bin analysis), giving a null value of every statistic.

Across sessions (per cohort):
  statistic = mean over animals of the mean over that animal's sessions
              (every animal weighs the same); for coefficients the signed beta,
              for F tests log(F).
  p (two-sided for betas, one-sided for F) against the same statistic computed
  on the j-th null draw of every session, j = 1..N_SHIFT.
  Also: session-weighted p, leave-one-animal-out, hierarchical bootstrap CI
  (animals, then sessions) of the animal-weighted mean beta, and
  Benjamini-Hochberg FDR across all tests.

Run from reversalExpectation_py (revExp env), after glmhmm_decode.py:
    python -X utf8 neuromodulator\\neuromodulator_state_baseline.py > analysis\\obj5_baseline.log 2>&1
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

WINDOWS = {"pre_cue": (-3.0, 0.0),      # before the cue: precedes choice and outcome
           "post_cue": (0.0, 5.0)}      # rest of the trial, for comparison
N_SHIFT = 1000                          # circular shifts per session (cheap: 1 OLS each)
N_BOOT = 10000                          # hierarchical bootstrap resamples
SEED = 20261025
OUT_DIR = ROOT / "data" / "figs" / "neuromodulator-state-baseline"


# ---------------------------------------------------------------------------
# Regression core (numpy only: significance comes from the shift null, so no
# parametric distributions are needed)
# ---------------------------------------------------------------------------
def _ols_rss(y, X):
    """OLS with intercept. Returns (beta, rss, n, k)."""
    XX = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(XX, y, rcond=None)
    r = y - XX @ beta
    return beta, float(r @ r), len(y), XX.shape[1]


def _partial_F(rss_s, k_s, rss_b, k_b, n):
    df1, df2 = k_b - k_s, n - k_b
    if df1 <= 0 or df2 <= 0 or rss_b <= 0:
        return np.nan
    return ((rss_s - rss_b) / df1) / (rss_b / df2)


def fit_nested(y, X1, X2, X3):
    """M1 / M2 / M3 on the same rows. Returns (betas of X2+X3 from M3, F21, F32)."""
    XA = np.column_stack([X1, X2, X3])
    keep = np.isfinite(y) & np.all(np.isfinite(XA), axis=1)
    y, X1, X2, X3 = y[keep], X1[keep], X2[keep], X3[keep]
    _, r1, n, k1 = _ols_rss(y, X1)
    _, r2, _, k2 = _ols_rss(y, np.column_stack([X1, X2]))
    b3, r3, _, k3 = _ols_rss(y, np.column_stack([X1, X2, X3]))
    added = b3[1 + X1.shape[1]:]
    return added, _partial_F(r1, k1, r2, k2, n), _partial_F(r2, k2, r3, k3, n)


def window_signal(Y, centers, lo, hi):
    """Per-trial mean over the bins whose centre falls in [lo, hi), z-scored."""
    cols = (centers >= lo) & (centers < hi)
    y = np.nanmean(Y[:, cols], axis=1)
    sd = np.nanstd(y)
    return (y - np.nanmean(y)) / sd if sd > 0 else y * np.nan


def session_stats(design, names, rng):
    """Real and null statistics of one session for every window.

    Returns {window: {"real": (n_stats,), "null": (N_SHIFT, n_stats)}} with the
    statistics ordered as names + ["F_state", "F_belief"]."""
    Y, centers = design["Y"], design["centers"]
    X1, X2, X3, n_tr = design["X1"], design["X2"], design["X3"], design["n_tr"]
    min_shift = 30
    shifts = rng.integers(min_shift, max(n_tr - min_shift, min_shift + 1), N_SHIFT)
    out = {}
    for w, (lo, hi) in WINDOWS.items():
        y = window_signal(Y, centers, lo, hi)
        b, F21, F32 = fit_nested(y, X1, X2, X3)
        real = np.r_[b, F21, F32]
        null = np.empty((N_SHIFT, len(real)))
        for j, k in enumerate(shifts):            # same shifts for every window
            bj, F21j, F32j = fit_nested(y, X1, np.roll(X2, k, axis=0), np.roll(X3, k, axis=0))
            null[j] = np.r_[bj, F21j, F32j]
        out[w] = {"real": real, "null": null}
    return out


# ---------------------------------------------------------------------------
# Animal-aware aggregation
# ---------------------------------------------------------------------------
def _transform(stat_name, v):
    """F statistics are compared on the log scale (right-skewed)."""
    return np.log(np.maximum(v, 1e-12)) if stat_name.startswith("F_") else v


def animal_weighted(values, groups):
    """values (..., n_sessions) -> mean over animals of within-animal means."""
    return np.mean(np.stack([values[..., g].mean(axis=-1) for g in groups], axis=-1), axis=-1)


def test_statistic(real, null, groups, two_sided):
    """real (n_sessions,), null (N_SHIFT, n_sessions). Returns (obs, p)."""
    obs = float(animal_weighted(real, groups))
    nul = animal_weighted(null, groups)
    if two_sided:
        center = np.mean(nul)
        p = (np.sum(np.abs(nul - center) >= abs(obs - center)) + 1) / (len(nul) + 1)
    else:
        p = (np.sum(nul >= obs) + 1) / (len(nul) + 1)
    return obs, float(p), float(np.mean(nul))


def hierarchical_ci(real, groups, rng):
    """95% CI of the animal-weighted mean: resample animals, then sessions."""
    n_an = len(groups)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, n_an, n_an)
        boot[b] = np.mean([real[rng.choice(groups[a], len(groups[a]), replace=True)].mean()
                           for a in pick])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def aggregate(records, stat_names, rng):
    """records: list of dicts with animal, cohort, window, real, null."""
    rows = []
    for (coh, w), recs in pd.DataFrame(records).groupby(["cohort", "window"]):
        recs = recs.reset_index(drop=True)
        animals = sorted(recs["animal"].unique())
        groups = [np.flatnonzero(recs["animal"].to_numpy() == a) for a in animals]
        R = np.stack(recs["real"].to_numpy())                    # (n_sessions, n_stats)
        N = np.stack(recs["null"].to_numpy(), axis=-1)           # (N_SHIFT, n_stats, n_sessions)
        for s_i, name in enumerate(stat_names):
            two = not name.startswith("F_")
            real = _transform(name, R[:, s_i])
            null = _transform(name, N[:, s_i, :])
            ok = np.isfinite(real) & np.all(np.isfinite(null), axis=0)
            real, null = real[ok], null[:, ok]
            an = recs["animal"].to_numpy()[ok]
            anim = sorted(set(an))
            g = [np.flatnonzero(an == a) for a in anim]
            obs, p, null_mean = test_statistic(real, null, g, two)
            # session-weighted version
            sw_groups = [np.arange(len(real))]
            _, p_sw, _ = test_statistic(real, null, sw_groups, two)
            # leave-one-animal-out
            loao = []
            for j, a in enumerate(anim):
                keep = [gg for i, gg in enumerate(g) if i != j]
                if keep:
                    o_j, p_j, _ = test_statistic(real, null, keep, two)
                    loao.append(f"-{a}:{o_j:+.3f}/p={p_j:.3f}")
            per_animal = [real[gg].mean() for gg in g]
            null_animal = [null[:, gg].mean() for gg in g]
            row = {"cohort": coh, "window": w, "stat": name, "n_animals": len(anim),
                   "n_sessions": len(real), "obs": obs, "null_mean": null_mean,
                   "p": p, "p_session_weighted": p_sw,
                   # animals on the same side of their null as the pooled effect
                   # (betas: either sign; F: above the null)
                   "animals_beyond_null": int(np.sum(
                       (np.array(per_animal) - np.array(null_animal))
                       * (np.sign(obs - null_mean) or 1.0 if two else 1.0) > 0)),
                   "per_animal": " ".join(f"{a}:{v:+.3f}" for a, v in zip(anim, per_animal)),
                   "leave_one_animal_out": " ".join(loao)}
            if two:
                row["ci_low"], row["ci_high"] = hierarchical_ci(real, g, rng)
            rows.append(row)
    res = pd.DataFrame(rows)
    res["p_fdr_bh"] = bh_fdr(res["p"].to_numpy())
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    reg = _load("neuromodulator_state_regression")          # shared session builder
    names = reg.STATE_PRED + reg.BELIEF_PRED
    stat_names = names + ["F_state", "F_belief"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not reg.STATE_CSV.exists():
        sys.exit(f"{reg.STATE_CSV} not found: run glmhmm_decode.py in the ssm env first")
    states = pd.read_csv(reg.STATE_CSV)
    states["session_file"] = states["session_file"].astype(str)
    vhr_par = reg.load_vhr_params()
    master = reg._load("master_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(reg.PHASES)]
    idx = reg.ntr.drop_excluded_animals(idx)

    records, status = [], []
    for k, (_, r) in enumerate(idx.iterrows(), 1):
        try:
            out, design = reg.build_session_design(r, states, vhr_par)
        except Exception as e:
            out, design = {"session": Path(str(r["session_file"])).stem,
                           "animal": str(r["animal"]),
                           "cohort": reg.COHORT.get(r.get("experiment"), "?"),
                           "status": f"error: {e!r}"}, None
        status.append(out)
        if design is not None:
            # stable seed per session, same rule as the per-bin analysis
            rng = np.random.default_rng(zlib.crc32(design["ses"].encode("utf-8")))
            st = session_stats(design, names, rng)
            for w, d in st.items():
                records.append({"session": design["ses"], "animal": out["animal"],
                                "cohort": out["cohort"], "window": w,
                                "real": d["real"], "null": d["null"]})
        print(f"[{k}/{len(idx)}] {out['session']}: {out['status']}", flush=True)

    status = pd.DataFrame(status)
    status.to_csv(OUT_DIR / "session_status.csv", index=False)
    print("\n=== Session status ===")
    print(status.groupby(["cohort", "status"]).size().to_string())
    if not records:
        return

    # per-session real values, for inspection
    pd.DataFrame([{"session": r["session"], "animal": r["animal"], "cohort": r["cohort"],
                   "window": r["window"], **dict(zip(stat_names, r["real"]))}
                  for r in records]).to_csv(OUT_DIR / "session_values.csv", index=False)

    res = aggregate(records, stat_names, np.random.default_rng(SEED))
    res.to_csv(OUT_DIR / "baseline_stats.csv", index=False)

    print("\n=== Per-trial window analysis (alternative to the per-bin analysis) ===")
    print("betas: signal SD per SD of the predictor (from M3); two-sided p vs the shift null")
    print("F_*  : mean log partial F; one-sided p vs the shift null")
    print("every animal weighs the same; CI = hierarchical bootstrap (animals, sessions)\n")
    for (coh, w), g in res.groupby(["cohort", "window"]):
        print(f"--- {coh} | {w} {WINDOWS[w]} s | {g['n_animals'].iloc[0]} animals, "
              f"{g['n_sessions'].iloc[0]} sessions")
        for _, r in g.iterrows():
            ci = (f"CI95 [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                  if pd.notna(r.get("ci_low")) else " " * 22)
            print(f"  {r['stat']:11s} obs {r['obs']:+.4f} (null {r['null_mean']:+.4f})  "
                  f"p={r['p']:.4f}  p_sess={r['p_session_weighted']:.4f}  "
                  f"p_fdr={r['p_fdr_bh']:.3f}  {ci}  "
                  f"animals {r['animals_beyond_null']}/{r['n_animals']}")
            print(f"  {'':11s} per animal: {r['per_animal']}")
            print(f"  {'':11s} leave-one-animal-out: {r['leave_one_animal_out']}")
        print()
    print(f"Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
