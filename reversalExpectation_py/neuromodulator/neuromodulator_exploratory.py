"""
neuromodulator_exploratory.py
=============================
General exploratory analysis of the neuromodulator (imaging) sessions that is
also a NEURAL test of the session / trial matching.

Per session (phase 31, with trace + timeEvents):
  1. QC: sampling rate, duration, raw F level, bleaching, n trial stamps vs
     n log trials, whether the trace looks like raw fluorescence.
  2. Wang et al. 2025 dF/F, aligned to cue onset (IO1 trial stamps), binned
     in 100-ms bins from -2 to 5 s.
  3. Outcome response: rewarded vs unrewarded trials (responded only), with
     trial response = mean(0..2 s) - mean(-1..0 s).
  4. Trial matching: each log trial is matched to its imaging stamp with
     neuromodulator_trials.match_stamps_to_trials (timing-based; handles
     trials before imaging started, early stops and lost stamps).
  5. LAG TEST (neural check of that matching): pair the neural response of log
     trial j with the behavior of trial j+k, k = -5..+5, and recompute the
     rewarded-vs-unrewarded t statistic. If the timing-based matching is right,
     the outcome effect peaks at k = 0 in every session. This is independent
     evidence: it uses the signal, not the timestamps.

Outputs (data/figs/neuromodulator-exploratory/):
  session_qc.csv, lag_test.csv, fig_qc.png, fig_psth_outcome.png, fig_lag_test.png

Run from reversalExpectation_py (revExp env):
    python neuromodulator\\neuromodulator_exploratory.py
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

wp = _load("wang2025_pipeline")
runner = _load("run_wang2025_neuromodulator")      # load_trace / load_trial_stamps
ntr = _load("neuromodulator_trials")

PHASES = [31]
EDGES = np.round(np.arange(-2.0, 5.0 + 1e-9, 0.1), 10)   # 100-ms bins
RESP_WIN = (0.0, 2.0)          # post-cue response window (s)
BASE_WIN = (-1.0, 0.0)         # pre-cue baseline (s)
LAGS = np.arange(-5, 6)
MIN_TRIALS_PER_GROUP = 15
OUT_DIR = ROOT / "data" / "figs" / "neuromodulator-exploratory"
COLORS = {"NE": "#2A9D8F", "ACh": "#E9A93A"}
COHORT = {1.0: "NE", 2.0: "ACh"}


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------
def trial_matrix(t, dff, stamps):
    """dF/F per trial on the 100-ms grid: (n_trials, n_bins) + bin centers."""
    it, isig = wp.interpolate_signal(t, dff)
    sig, tb = wp.align_signal(it, isig, stamps, (EDGES[0] - 1, EDGES[-1] + 1))
    binned, centers = wp.bin_aligned(sig, tb, EDGES)
    return binned.T, centers


def trial_response(mat, centers):
    r = (centers > RESP_WIN[0]) & (centers < RESP_WIN[1])
    b = (centers > BASE_WIN[0]) & (centers < BASE_WIN[1])
    with np.errstate(all="ignore"):
        return np.nanmean(mat[:, r], axis=1) - np.nanmean(mat[:, b], axis=1)


def outcome_t(resp, rewarded, responded):
    """Welch t: rewarded minus unrewarded (responded trials only)."""
    ok = responded & np.isfinite(resp) & np.isfinite(rewarded)
    a, b = resp[ok & (rewarded == 1)], resp[ok & (rewarded == 0)]
    if len(a) < MIN_TRIALS_PER_GROUP or len(b) < MIN_TRIALS_PER_GROUP:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).statistic)


def lag_curve(resp, beh):
    """t(rewarded - unrewarded) pairing neural trial j with behavior trial j+k."""
    rew = beh["rewarded"].to_numpy(float)
    rsp = beh["responded"].to_numpy(bool)
    n = min(len(resp), len(rew))
    out = []
    for k in LAGS:
        if k >= 0:
            r_, w_, s_ = resp[:n - k], rew[k:n], rsp[k:n]
        else:
            r_, w_, s_ = resp[-k:n], rew[:n + k], rsp[:n + k]
        out.append(outcome_t(r_, w_, s_))
    return np.array(out)


def process_session(row):
    out = {"session": Path(str(row["session_file"])).stem, "animal": str(row["animal"]),
           "cohort": COHORT.get(row.get("experiment"), "?"), "status": "ok"}
    try:
        if pd.isna(row.get("cell_created")) or pd.isna(row.get("time_events_created")):
            out["status"] = "no_neural_data"
            return out, None
        beh = ntr.build_trial_table(row["beh_path"], out["animal"], out["session"])
        t, f = runner.load_trace(Path(row["neural_data_path"]) / row["cell_filename"])
        stamps = runner.load_trial_stamps(Path(row["neural_data_path"]) / row["time_events_filename"])
        match = ntr.match_stamps_to_trials(stamps, beh["cue_time_log"].to_numpy())
        out.update(stamp_lag=match["lag"], frac_trials_imaged=match["frac_trials_matched"],
                   match_resid_ms=match["fit_resid_ms"])

        fs = 1.0 / np.median(np.diff(t))
        two_min = int(120 * fs)
        qc = wp.check_raw_trace(f)
        out.update(fs=fs, duration_min=(t[-1] - t[0]) / 60, n_trials_log=len(beh),
                   n_stamps=len(stamps), raw_median=qc["median"], looks_raw=qc["looks_raw"],
                   bleach_pct=100 * (np.median(f[-two_min:]) / np.median(f[:two_min]) - 1)
                   if len(f) > 2 * two_min else np.nan)
        out.update(ntr.session_summary(beh))
        if not qc["looks_raw"]:
            out["status"] = "trace_not_raw"
            return out, None
        if not match["ok"]:
            out["status"] = "bad_alignment"
            return out, None

        dff, _ = wp.calc_dff(f, fs)
        # one row per LOG trial (NaN rows for trials without an imaging stamp)
        mat, centers = trial_matrix(t, dff, match["cue_time_imaging"])
        resp = trial_response(mat, centers)
        lags = lag_curve(resp, beh)
        out.update({f"t_lag{k:+d}": v for k, v in zip(LAGS, lags)})
        finite = np.isfinite(lags)
        if finite.any():
            out["best_lag"] = int(LAGS[np.nanargmax(np.abs(lags))])
            out["t_lag0"] = float(lags[LAGS == 0][0])
            # how much the effect at lag 0 stands out from the neighbors
            others = np.abs(lags[(LAGS != 0) & finite])
            out["lag0_contrast"] = float(abs(out["t_lag0"]) - np.nanmax(others)) if others.size else np.nan

        n = min(len(mat), len(beh))
        m = mat[:n]
        b = beh.iloc[:n]
        rsp = b["responded"].to_numpy(bool)
        rew = b["rewarded"].to_numpy(float)
        psth = {"rewarded": np.nanmean(m[rsp & (rew == 1)], axis=0),
                "unrewarded": np.nanmean(m[rsp & (rew == 0)], axis=0),
                "miss": np.nanmean(m[~rsp], axis=0) if (~rsp).sum() >= 5 else np.full(m.shape[1], np.nan),
                "centers": centers}
        return out, psth
    except Exception as e:
        out["status"] = f"error: {e!r}"
        return out, None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(qc: pd.DataFrame, psths: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = qc[qc["status"] == "ok"]

    # --- QC panel ---
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for coh, g in ok.groupby("cohort"):
        c = COLORS.get(coh, "gray")
        ax[0].scatter(g["n_trials_log"], g["n_stamps"], s=14, color=c, label=coh)
        ax[1].hist(g["bleach_pct"].dropna(), bins=20, color=c, alpha=0.6, label=coh)
        ax[2].scatter(g["reward_rate"], g["p_better"], s=14, color=c, label=coh)
        ax[3].hist(g["median_lrandom"].dropna(), bins=15, color=c, alpha=0.6, label=coh)
    lim = [0, max(ok["n_trials_log"].max(), ok["n_stamps"].max()) * 1.05] if len(ok) else [0, 1]
    ax[0].plot(lim, lim, "k:", lw=1)
    ax[0].set(xlabel="trials in log", ylabel="trial stamps (IO1)", title="Trial counts")
    ax[1].set(xlabel="F change, last vs first 2 min (%)", ylabel="sessions", title="Bleaching")
    ax[2].set(xlabel="reward rate", ylabel="P(better)", title="Behavior per session")
    ax[3].set(xlabel="median L_Random per session", ylabel="sessions", title="Block structure")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "fig_qc.png", dpi=150); plt.close(fig)

    # --- outcome PSTH: sessions -> animal -> cohort ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax_, coh in zip(axes, ("NE", "ACh")):
        sess = ok[ok["cohort"] == coh]
        for cond, ls in (("rewarded", "-"), ("unrewarded", "--"), ("miss", ":")):
            per_animal = []
            for a, g in sess.groupby("animal"):
                arr = np.array([psths[s][cond] for s in g["session"] if s in psths])
                if len(arr):
                    per_animal.append(np.nanmean(arr, axis=0))
            if not per_animal:
                continue
            arr = np.array(per_animal)
            t = next(iter(psths.values()))["centers"]
            mu = np.nanmean(arr, axis=0)
            se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * mu
            ax_.plot(t, mu, ls, color=COLORS[coh], label=f"{cond} ({len(arr)} animals)")
            ax_.fill_between(t, mu - se, mu + se, color=COLORS[coh], alpha=0.15, lw=0)
        ax_.axvline(0, color="k", lw=0.8)
        ax_.set(title=f"{coh}: dF/F by outcome", xlabel="time from cue (s)")
        ax_.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("dF/F")
    fig.tight_layout(); fig.savefig(out_dir / "fig_psth_outcome.png", dpi=150); plt.close(fig)

    # --- lag test ---
    cols = [f"t_lag{k:+d}" for k in LAGS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax_, coh in zip(axes, ("NE", "ACh")):
        g = ok[ok["cohort"] == coh]
        if not len(g) or cols[0] not in g:
            continue
        M = g[cols].to_numpy(float)
        for row in M:
            ax_.plot(LAGS, row, color=COLORS[coh], alpha=0.25, lw=0.8)
        ax_.plot(LAGS, np.nanmedian(M, axis=0), color="k", lw=2, label="median")
        ax_.axvline(0, color="gray", ls=":")
        ax_.axhline(0, color="gray", lw=0.6)
        ax_.set(title=f"{coh}: outcome effect vs trial lag ({len(g)} sessions)",
                xlabel="behavior trial offset k (neural j <-> behavior j+k)")
        ax_.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("t (rewarded - unrewarded)")
    fig.tight_layout(); fig.savefig(out_dir / "fig_lag_test.png", dpi=150); plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = _load("master_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(PHASES)]
    idx = ntr.drop_excluded_animals(idx)

    rows, psths = [], {}
    for k, (_, row) in enumerate(idx.iterrows(), 1):
        out, psth = process_session(row)
        rows.append(out)
        if psth is not None:
            psths[out["session"]] = psth
        extra = (f"  t(lag0)={out.get('t_lag0', np.nan):+.1f}  best lag={out.get('best_lag')}"
                 if out["status"] == "ok" else "")
        print(f"[{k}/{len(idx)}] {out['session']}: {out['status']}{extra}")

    qc = pd.DataFrame(rows)
    qc.to_csv(OUT_DIR / "session_qc.csv", index=False)
    ok = qc[qc["status"] == "ok"]

    print("\n=== Session status ===")
    print(qc.groupby(["cohort", "status"]).size().to_string())
    if len(ok):
        # A session can only CONFIRM the matching neurally if it has an outcome
        # signal: with |t(lag0)| small, the best lag is the argmax of noise.
        ok = ok.copy()
        ok["neural_check"] = np.select(
            [(ok["t_lag0"].abs() >= 3) & (ok["best_lag"] == 0),
             (ok["t_lag0"].abs() >= 3) & (ok["best_lag"] != 0)],
            ["confirmed", "CONTRADICTS"], default="no_signal")
        qc = qc.merge(ok[["session", "neural_check"]], on="session", how="left")
        qc.to_csv(OUT_DIR / "session_qc.csv", index=False)
        print("\n=== Neural check of the matching (needs an outcome signal: |t(lag0)| >= 3) ===")
        print(pd.crosstab(ok["stamp_lag"], ok["neural_check"],
                          rownames=["stamp lag (timing)"], colnames=["neural"]).to_string())
        print("\nPer animal: sessions with an outcome signal / total, median t(lag0)")
        pa = ok.groupby(["cohort", "animal"]).agg(
            n=("session", "size"),
            with_signal=("neural_check", lambda x: int((x != "no_signal").sum())),
            median_t_lag0=("t_lag0", "median"))
        print(pa.round(2).to_string())
        print("\n=== Lag test (neural matching check) ===")
        print("Best lag per session (should be 0):")
        print(ok["best_lag"].value_counts().sort_index().to_string())
        bad = ok[(ok["best_lag"] != 0) | (ok["t_lag0"].abs() < 2)]
        print(f"\nSessions with best lag != 0 or |t(lag0)| < 2: {len(bad)}")
        if len(bad):
            print(bad[["session", "cohort", "t_lag0", "best_lag", "lag0_contrast"]]
                  .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
        print("\n=== Timing-based matching ===")
        print("Stamp lag (0 = imaging started before trial 1):")
        print(ok["stamp_lag"].value_counts().sort_index().to_string())
        print(f"Median fraction of log trials with an imaging stamp: "
              f"{ok['frac_trials_imaged'].median():.3f}")
        print("\n=== Behavior of the imaging cohort (median across sessions) ===")
        cols = ["reward_rate", "p_better", "miss_rate", "median_trials_to_crit",
                "median_lrandom", "median_rt", "n_switches", "n_cues_estimated"]
        print(ok.groupby("cohort")[cols].median().round(3).to_string())
        make_figures(qc, psths, OUT_DIR)
    print(f"\nResults in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
