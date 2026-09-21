"""
run_wang2025_neuromodulator.py
==============================
Applies the Wang et al. 2025 pipeline (wang2025_pipeline.py) to every session
of the miniscope dataset and summarizes NE vs ACh.

Per session:
  1. Behavior with the same loader as per_session.py
     (parse_logfile -> get_session_data -> get_trial_masks).
  2. RAW trace from *_cell.csv (the old _dff.npz files are not used) + QC:
     if the trace is not positive fluorescence, the session is excluded.
  3. Cue onsets = trial stamps from the IO1 channel of timeEvents (imaging clock),
     matched to LOG trials one by one with
     neuromodulator_trials.match_stamps_to_trials (timing-based). Trials that
     happened before imaging started or after it stopped get no stamp and are
     left out of the regression; sessions whose stamps do not match the log's
     inter-trial intervals are excluded (status "bad_alignment").
  4. Wang dF/F (F0 = 10th percentile, centered 2-min window).
  5. Per-bin regression over 100-ms bins (-3 to 5 s), 14 predictors.
  6. Unit "modulated" per predictor (>=3 consecutive or >=10 total bins, p<0.01).

Summary (here the unit is the SESSION: a single ROI per session):
  - % of sessions with p<0.01 per bin and predictor, per cohort, + binomial vs 1%.
  - Mean coefficients (sessions averaged within animal, then across animals).
  - Temporal dynamics of R(n) and C(n): group by sign of the 0-2 s AUC,
    time-to-peak and peak value per session; NE vs ACh with rank-sum at the
    session level (as in Wang) and at the animal level (avoids pseudo-replication).

Run from reversalExpectation_py:
    python neuromodulator\\run_wang2025_neuromodulator.py
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

# np.trapezoid exists from numpy 2.0 and np.trapz was removed in numpy 2.4;
# the ssm env needs numpy < 2, so support both
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.log_parser import parse_logfile, get_session_data, detect_phase
from behavior.trial_processing import get_trial_masks


def _load_by_path(name: str):
    """Load a sibling module by file path (avoids package-resolution issues)."""
    p = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wp = _load_by_path("wang2025_pipeline")
ntr = _load_by_path("neuromodulator_trials")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASES = [31]                 # phase 31 = main task (as in the master)
# Camera clock drift. Trace times in *_cell.csv are NOMINAL (i / fs) while the
# real frame period is slightly longer, so the trace slowly falls behind the
# trial stamps (timeEvents clock). Measured on 140 well-aligned sessions as
# (timeEvents end - trace end) / duration: median 257 ppm, IQR 251-262 ppm
# (verify_neuromodulator_alignment.py, section 4b).
#   "measured": stretch trace time by CAMERA_DRIFT_PPM (default)
#   "matlab"  : creatDffMatFiles_miniscope.m's "shutter time delay",
#               t + (1:N)*1e-5 = 200 ppm at 20 Hz (leaves ~57 ppm uncorrected)
#   "none"    : no correction
SHUTTER_DELAY_MODE = "measured"
CAMERA_DRIFT_PPM = 257.0
MIN_TRIALS = 100              # shorter sessions are excluded
# Recorded hemisphere per animal ('left'/'right'). Needed for ipsi/contra.
# While empty, C is coded right=1 / left=0 (a warning is issued).
RECORDING_SITE: dict[str, str] = {}
DEFAULT_SITE = "left"         # with 'left': right = contra = 1
COHORT = {1.0: "NE", 2.0: "ACh"}
COLORS = {"NE": "#2A9D8F", "ACh": "#E9A93A"}      # green / yellow as in Wang
OUT_DIR = ROOT / "data" / "figs" / "wang2025"

KEY_PREDICTORS = ["C(n)", "R(n)", "X(n)"]


# ---------------------------------------------------------------------------
# Loading one session's data
# ---------------------------------------------------------------------------
def load_behavior(log_path: Path) -> pd.DataFrame:
    """Per-trial table: choice_lr (-1/+1/NaN), rewarded (0/1), responded (bool)."""
    log_data = parse_logfile(log_path)
    phase = detect_phase(log_data["scenario"])
    session_data, trial_data = get_session_data(log_data, phase)
    trial_data["n_rules"] = session_data["nRules"]
    masks = get_trial_masks(trial_data)
    n = len(trial_data["cue"])

    def m(key):
        v = masks.get(key)
        if v is None:
            return np.zeros(n, dtype=bool)
        v = np.asarray(v)
        # masks from merged sessions may contain NaN -> False
        return np.nan_to_num(v[:n].astype(float), nan=0.0).astype(bool)

    left, right, reward = m("left"), m("right"), m("reward")
    choice = np.full(n, np.nan)
    choice[left], choice[right] = -1.0, 1.0
    return pd.DataFrame({"choice_lr": choice,
                         "rewarded": (reward & (left | right)).astype(float),
                         "responded": left | right})


def load_trace(cell_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Time and raw fluorescence (cols 0 and 1), as in create_dff_files.py."""
    raw = pd.read_csv(cell_path, skiprows=2, header=None, low_memory=False)
    t = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy()
    f = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    # camera clock drift correction (see SHUTTER_DELAY_MODE above)
    if SHUTTER_DELAY_MODE == "measured":
        t = t[0] + (t - t[0]) * (1.0 + CAMERA_DRIFT_PPM * 1e-6)
    elif SHUTTER_DELAY_MODE == "matlab":
        t = t + np.arange(1, len(t) + 1) * 1e-5
    t = t + np.arange(len(t)) * 1e-9              # break ties between duplicates
    if np.any(np.diff(t) <= 0):
        raise ValueError("trace times are not increasing (beyond duplicates)")
    return t, f


# Trial TTLs are normally on IO1, but in some sessions they were recorded on the
# "trigger" input instead (IO1 flat). timeEvents stores CONTINUOUS samples of
# every channel (~1 kHz), so a channel "has pulses" when its value changes.
TTL_CHANNELS = ("IO1", "trigger")
MIN_TTL_TRANSITIONS = 20


def load_trial_stamps(te_path: Path, return_channel: bool = False):
    """Cue onsets in the imaging clock: transitions of the TTL channel, every
    other one from the 2nd (creatDffMatFiles_miniscope.m: trialStamps(2:2:end)).
    Uses IO1 as the .m does; falls back to 'trigger' when IO1 has no pulses."""
    te = pd.read_csv(te_path, low_memory=False)
    if te.shape[1] != 3:
        raise ValueError(f"{te_path.name}: expected 3 columns, got {te.shape[1]}")
    te.columns = ["Time_s_", "ChannelName", "Value"]
    names = te["ChannelName"].astype(str).str.strip()
    for ch in TTL_CHANNELS:
        sub = te[names == ch]
        times = pd.to_numeric(sub["Time_s_"], errors="coerce").to_numpy()
        vals = pd.to_numeric(sub["Value"], errors="coerce").to_numpy()
        trans = np.where(vals[:-1] != vals[1:])[0]
        if len(trans) >= MIN_TTL_TRANSITIONS:
            stamps = times[trans][1::2]
            return (stamps, ch) if return_channel else stamps
    empty = np.array([])
    return (empty, "none") if return_channel else empty


# ---------------------------------------------------------------------------
# Processing one session
# ---------------------------------------------------------------------------
def process_session(row) -> dict:
    """Returns a metadata dict; if the session is valid, also coeff/pval."""
    out = {"session": Path(str(row["session_file"])).stem,
           "animal": str(row["animal"]), "phase": row.get("phase"),
           "cohort": COHORT.get(row.get("experiment"), "?"), "status": "ok"}
    try:
        if pd.isna(row.get("cell_created")) or pd.isna(row.get("time_events_created")):
            out["status"] = "no_neural_data"
            return out

        beh = ntr.build_trial_table(Path(row["beh_path"]), out["animal"], out["session"])
        t, f = load_trace(Path(row["neural_data_path"]) / row["cell_filename"])
        stamps, ttl_ch = load_trial_stamps(Path(row["neural_data_path"]) / row["time_events_filename"],
                                           return_channel=True)
        match = ntr.match_stamps_to_trials(stamps, beh["cue_time_log"].to_numpy())
        out.update(n_trials_log=len(beh), n_stamps=len(stamps), ttl_channel=ttl_ch, stamp_lag=match["lag"],
                   n_trials_imaged=match["n_matched"], match_resid_ms=match["fit_resid_ms"])

        qc = wp.check_raw_trace(f)
        out.update(raw_min=qc["min"], raw_median=qc["median"])
        if not qc["looks_raw"]:
            out["status"] = "trace_not_raw"
            return out
        if not match["ok"]:
            out["status"] = "bad_alignment"
            return out
        if match["n_matched"] < MIN_TRIALS:
            out["status"] = "too_few_trials"
            return out
        # cue onset in the imaging clock for every log trial (NaN = not imaged);
        # the design matrix uses the full behavioral history
        cue_img = match["cue_time_imaging"]

        fs = 1.0 / np.median(np.diff(t))
        dff, _ = wp.calc_dff(f, fs)

        site = RECORDING_SITE.get(out["animal"], DEFAULT_SITE)
        c_ic = wp.encode_choice_ipsi_contra(beh["choice"].to_numpy(), site)
        # misses: rewarded NaN in the trial table, 0 in trials.reward (MP_GRAB_MLR.m)
        X = wp.build_design_matrix(c_ic, np.nan_to_num(beh["rewarded"].to_numpy(), nan=0.0),
                                   beh["responded"].to_numpy())
        res = wp.linear_regr(t, dff, cue_img, X)

        out.update(fs=fs, n_used=int(match["n_matched"]), frac_responded=float(beh["responded"].mean()),
                   reward_rate=float(beh["rewarded"][beh["responded"]].mean()),
                   coeff=res["coeff"], pval=res["pval"], regr_time=res["regr_time"])
        for k, name in enumerate(wp.PREDICTOR_NAMES):
            out[f"mod_{name}"] = wp.is_modulated(res["pval"][:, k + 1])
    except Exception as e:
        out["status"] = f"error: {e!r}"
    return out


# ---------------------------------------------------------------------------
# Summary and figures
# ---------------------------------------------------------------------------
def summarize(results: list, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("coeff", "pval", "regr_time")} for r in results])
    meta.to_csv(out_dir / "sessions_status.csv", index=False)
    print("\n=== Session status ===")
    print(meta.groupby(["cohort", "status"]).size().to_string())

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("No valid session: no summary.")
        return
    t = ok[0]["regr_time"]
    cols = {nm: 1 + wp.PREDICTOR_NAMES.index(nm) for nm in wp.PREDICTOR_NAMES}

    # arrays per cohort: (n_sessions, n_bins, n_coef)
    by = {}
    for coh in ("NE", "ACh"):
        rs = [r for r in ok if r["cohort"] == coh]
        if rs:
            by[coh] = {"coeff": np.stack([r["coeff"] for r in rs]),
                       "pval": np.stack([r["pval"] for r in rs]),
                       "animal": np.array([r["animal"] for r in rs]),
                       "session": [r["session"] for r in rs]}
    np.savez(out_dir / "regression_results.npz", regr_time=t,
             predictors=np.array(wp.PREDICTOR_NAMES),
             **{f"{c}_{k}": v[k] for c, v in by.items() for k in ("coeff", "pval", "animal")})

    # --- table: sessions modulated per predictor ---
    mod_cols = [f"mod_{p}" for p in wp.PREDICTOR_NAMES]
    okm = meta[meta["status"] == "ok"]
    tab = okm.groupby("cohort")[mod_cols].mean().T.round(3)
    tab.index = wp.PREDICTOR_NAMES
    tab["n_ses"] = ""
    for coh in tab.columns[:-1]:
        tab.loc[wp.PREDICTOR_NAMES[0], "n_ses"] += f"{coh}={int((okm['cohort'] == coh).sum())} "
    print("\n=== Fraction of sessions modulated per predictor ===")
    print(tab.to_string())
    tab.to_csv(out_dir / "fraction_sessions_modulated.csv")

    # --- figure 1: % sessions with p<0.01 per bin (analogous to Fig. 3C) ---
    fig, axes = plt.subplots(1, len(KEY_PREDICTORS), figsize=(13, 3.8), sharey=True)
    for ax, nm in zip(axes, KEY_PREDICTORS):
        for coh, d in by.items():
            fr = wp.fraction_significant_per_bin(d["pval"], cols[nm])
            ax.plot(t, 100 * fr["frac_sig"], color=COLORS[coh],
                    label=f"{coh} (n={d['pval'].shape[0]} sessions)")
            sig = fr["p_binom"].to_numpy() < 0.01
            ax.plot(t[sig], np.full(sig.sum(), 100.0 + 3 * (coh == "ACh")), "s",
                    color=COLORS[coh], ms=2.5)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_title(nm); ax.set_xlabel("time from cue (s)")
    axes[0].set_ylabel("% sessions with p<0.01"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_fraction_significant.png", dpi=150)
    plt.close(fig)

    # --- figure 2: mean coefficient (sessions -> animal -> cohort) ---
    fig, axes = plt.subplots(1, len(KEY_PREDICTORS), figsize=(13, 3.8))
    for ax, nm in zip(axes, KEY_PREDICTORS):
        for coh, d in by.items():
            per_animal = np.stack([np.nanmean(d["coeff"][d["animal"] == a, :, cols[nm]], axis=0)
                                   for a in np.unique(d["animal"])])
            mu = np.nanmean(per_animal, axis=0)
            se = np.nanstd(per_animal, axis=0, ddof=1) / np.sqrt(per_animal.shape[0]) \
                if per_animal.shape[0] > 1 else np.zeros_like(mu)
            ax.plot(t, mu, color=COLORS[coh], label=f"{coh} (n={per_animal.shape[0]} animals)")
            ax.fill_between(t, mu - se, mu + se, color=COLORS[coh], alpha=0.25, lw=0)
        ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{nm} coefficient"); ax.set_xlabel("time from cue (s)")
    axes[0].set_ylabel("coefficient (dF/F)"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_mean_coefficients.png", dpi=150)
    plt.close(fig)

    # --- temporal dynamics: R(n) and C(n) in modulated sessions ---
    rows = []
    for nm in ("R(n)", "C(n)"):
        for coh, d in by.items():
            for s in range(d["coeff"].shape[0]):
                if not wp.is_modulated(d["pval"][s, :, cols[nm]]):
                    continue
                curve = d["coeff"][s, :, cols[nm]]
                w = (t > wp.AUC_WINDOW[0]) & (t < wp.AUC_WINDOW[1])
                sign = 1 if _trapz(curve[w], t[w]) > 0 else -1
                ttp, pk = wp.time_to_peak(curve, t, sign)
                rows.append({"predictor": nm, "cohort": coh, "animal": d["animal"][s],
                             "session": d["session"][s], "group": 1 if sign > 0 else 2,
                             "time_to_peak": ttp, "peak_value": pk})
    tp = pd.DataFrame(rows)
    tp.to_csv(out_dir / "temporal_dynamics_sessions.csv", index=False)

    print("\n=== Temporal dynamics (modulated sessions; group 1 = increase, 2 = decrease) ===")
    for (nm, g), sub in tp.groupby(["predictor", "group"]):
        ne, ach = sub[sub.cohort == "NE"], sub[sub.cohort == "ACh"]
        line = (f"{nm} group {g}: NE n={len(ne)} sessions, median ttp={ne.time_to_peak.median():.2f} s | "
                f"ACh n={len(ach)} sessions, median ttp={ach.time_to_peak.median():.2f} s")
        if len(ne) >= 3 and len(ach) >= 3:
            # session level (as in Wang) and animal level (median per animal)
            ps = wp.compare_groups(ne.time_to_peak, ach.time_to_peak)["p_ranksum"]
            an = sub.groupby(["cohort", "animal"]).time_to_peak.median().reset_index()
            a_ne, a_ach = an[an.cohort == "NE"].time_to_peak, an[an.cohort == "ACh"].time_to_peak
            pa = (wp.compare_groups(a_ne, a_ach)["p_ranksum"]
                  if len(a_ne) >= 2 and len(a_ach) >= 2 else np.nan)
            line += (f"\n    rank-sum ttp: session p={ps:.3g} | animal p={pa:.3g} "
                     f"({len(a_ne)} vs {len(a_ach)} animals)")
        print(line)

    print(f"\nResults in {out_dir.resolve()}")


def main():
    master = _load_by_path("master_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(PHASES)]
    idx = ntr.drop_excluded_animals(idx)
    if not RECORDING_SITE:
        warnings.warn("RECORDING_SITE is empty: C(n) is coded right=1 / left=0, "
                      "not ipsi/contra. The sign of the choice coefficients cannot be "
                      "interpreted as in Wang until the hemisphere per animal is set.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for k, (_, row) in enumerate(idx.iterrows(), 1):
        r = process_session(row)
        print(f"[{k}/{len(idx)}] {r['session']}: {r['status']}")
        results.append(r)
    summarize(results, OUT_DIR)


if __name__ == "__main__":
    main()
