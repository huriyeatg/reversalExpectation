"""
verify_neuromodulator_alignment.py
==================================
Checks, session by session:

(1) MATCHING: the new rule (timestamp, one-to-one) vs the rule in
    addIndexNeuromodulator.m: dir('M{Animal}_Phase{Phase}_{yymmdd}*_cell.csv').
      match          -> the .m finds exactly 1 file and it is the same one
      differs        -> the .m finds 1 file, different from ours
      m_no_file      -> the .m finds nothing (e.g. off-pattern name)
      m_ambiguous    -> the .m finds >1 file (in MATLAB, {filepath.name}
                        with >1 element makes the assignment fail)
      new_no_file    -> we assign nothing

(2) TRIAL-BY-TRIAL ALIGNMENT (equivalent to check_imageTriggerTimes in the .m,
    which the Python port lacked): intervals between trial stamps in the
    timeEvents file (Inscopix clock) are compared with intervals between
    cueTimes in the log (behavior PC clock), trying lags of -3..+3 trials.
    If the best lag is not 0, trial k of the trace is NOT trial k of the log.

(3) CLOCK DRIFT: slope of stamps vs cueTimes (in ppm).

(4) TRACE TIMING: sampling period and whether steps are exactly regular
    (nominal times) or irregular (real timestamps). This decides whether the
    cumulative "shutter time delay" in the .m (t + (1:N)*1e-5) makes sense as
    a correction or is a spurious shift.

Run from reversalExpectation_py:
    python neuromodulator\\verify_neuromodulator_alignment.py
"""

import importlib.util
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.log_parser import parse_logfile, get_session_data, detect_phase


def _load(name):
    p = Path(__file__).resolve().with_name(f"{name}.py")
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


master = _load("master_neuromodulator")
ai = _load("add_index_neuromodulator")

PHASES = [31, 32]
MAX_LAG = 3
OUT = ROOT / "data" / "figs" / "neuromodulator-qc" / "alignment_verification.csv"


# ---------------------------------------------------------------------------
def build_behavior_index() -> pd.DataFrame:
    """Same steps as the master (animal, date_number, phase), without add_index."""
    from behavior.master_behavior import make_data_index
    idx = make_data_index(master.DATA_ROOT, master.SUBFOLDER)
    idx = master.fix_animal_column(idx)
    idx = master.fix_date_number_column(idx)
    idx = master.add_phase_column(idx)
    return idx[idx["phase"].isin(PHASES)].copy()


def matlab_rule_candidates(animal: str, phase: int, date_number, folder: Path) -> list:
    """dir(fullfile(path, ['M', Animal, '_Phase', Phase, '_', yymmdd, '*_cell.csv']))
    (Windows: case-insensitive)."""
    if pd.isna(date_number):
        return []
    prefix = f"m{animal}_phase{int(phase)}_{str(int(date_number))[:6]}".lower()
    return sorted(p.name for p in folder.glob("*_cell.csv")
                  if p.name.lower().startswith(prefix))


def cue_times_from_log(beh_path: Path) -> tuple[np.ndarray, str]:
    """Find cue times in trial_data (the .m uses trialData.cueTimes)."""
    log = parse_logfile(beh_path)
    # count the missing-cue fix instead of printing one warning per session
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, td = get_session_data(log, detect_phase(log["scenario"]))
    n_est = 0
    for w in caught:
        m = re.search(r"rule=(\d+), cue=(\d+)", str(w.message))
        if m:
            n_est = int(m.group(1)) - int(m.group(2))
    for key in ("cueTimes", "cue_times", "cueTime", "cue"):
        if key in td:
            v = np.asarray(td[key], dtype=float).ravel()
            # skip 0/1 masks: it must be an increasing series of times
            if v.size > 3 and np.nanmax(v) > 10 and np.all(np.diff(v[np.isfinite(v)]) > 0):
                return v, key, n_est
    return np.array([]), "not_found", n_est


def trial_stamps(te_path: Path):
    """Returns (stamps, ttl_channel, last_time). IO1 as in the .m; falls back to
    'trigger' when IO1 has no pulses (same rule as
    run_wang2025_neuromodulator.load_trial_stamps). The file is read ONCE: it
    holds continuous ~1 kHz samples of 4 channels (millions of rows)."""
    te = pd.read_csv(te_path, low_memory=False)
    te.columns = ["Time_s_", "ChannelName", "Value"]
    t_all = pd.to_numeric(te["Time_s_"], errors="coerce")
    last_time = float(t_all.max())
    names = te["ChannelName"].astype(str).str.strip()
    for ch in ("IO1", "trigger"):
        sel = (names == ch).to_numpy()
        t = t_all.to_numpy()[sel]
        v = pd.to_numeric(te["Value"], errors="coerce").to_numpy()[sel]
        tr = np.where(v[:-1] != v[1:])[0]
        if len(tr) >= 20:
            return t[tr][1::2], ch, last_time
    return np.array([]), "none", last_time


def best_lag(stamps: np.ndarray, cues: np.ndarray, max_lag: int = MAX_LAG) -> dict:
    """Compares inter-trial intervals (independent of the absolute offset).
    best_lag = 0  -> stamp k <-> log trial k (correct alignment)
    best_lag = +L -> stamp k+L <-> trial k  (L extra stamps at the start)
    best_lag = -L -> stamp k <-> trial k+L  (the first L stamps are missing)"""
    a, b = np.diff(stamps), np.diff(cues)
    res = []
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        n = min(len(aa), len(bb))
        if n < 20:
            continue
        err = np.abs(aa[:n] - bb[:n])
        res.append((np.median(err), lag, np.corrcoef(aa[:n], bb[:n])[0, 1], np.percentile(err, 95)))
    if not res:
        return {}
    med, lag, r, p95 = min(res)
    return {"best_lag": lag, "iti_median_abs_err_s": med, "iti_p95_abs_err_s": p95, "iti_corr": r}


def clock_ppm(stamps, cues, lag) -> float:
    s, c = (stamps[lag:], cues) if lag >= 0 else (stamps, cues[-lag:])
    n = min(len(s), len(c))
    if n < 20:
        return np.nan
    slope = np.polyfit(c[:n], s[:n], 1)[0]
    return (slope - 1) * 1e6


def trace_timing(cell_path: Path, te_last_time: float) -> dict:
    t = pd.to_numeric(pd.read_csv(cell_path, skiprows=2, header=None, usecols=[0],
                                  low_memory=False).iloc[:, 0], errors="coerce").dropna().to_numpy()
    d = np.diff(t)
    med = np.median(d)
    # end of the trace (nominal frame times) vs last timeEvents entry: if frame
    # times are nominal and the camera runs slow, the gap grows with duration
    gap = float(te_last_time - t[-1])
    return {"frame_period_ms": 1e3 * med, "fs_hz": 1 / med,
            "trace_dur_s": float(t[-1] - t[0]), "te_minus_trace_end_s": gap,
            "end_gap_ppm": 1e6 * gap / float(t[-1] - t[0]),
            "step_sd_us": 1e6 * np.std(d),
            "frac_steps_exact": float(np.mean(np.abs(d - med) < 1e-7)),
            "n_dup_times": int(np.sum(d <= 0)),
            "matlab_jitter_end_s": len(t) * 1e-5}


# ---------------------------------------------------------------------------
def main():
    beh = build_behavior_index()
    new = ai.add_index_neuromodulator(beh, master.MINISCOPE_PATH)
    folder = Path(master.MINISCOPE_PATH)

    rows = []
    for k, (i, r) in enumerate(new.iterrows(), 1):
        print(f"  [{k}/{len(new)}] {Path(str(r['session_file'])).stem}", flush=True)
        out = {"session": Path(str(r["session_file"])).stem, "animal": r["animal"],
               "phase": r["phase"], "experiment": r["experiment"],
               "new_cell": r["cell_filename"], "cell_dt_min": r["cell_dt_min"]}
        cands = matlab_rule_candidates(str(r["animal"]), r["phase"], r["date_number"], folder)
        out["m_n_candidates"] = len(cands)
        if not r["cell_filename"]:
            out["match_vs_m"] = "new_no_file"
        elif len(cands) == 0:
            out["match_vs_m"] = "m_no_file"
        elif len(cands) > 1:
            out["match_vs_m"] = "m_ambiguous"
        else:
            out["match_vs_m"] = "match" if cands[0] == r["cell_filename"] else "differs"
        out["m_candidates"] = "|".join(cands)

        if r["cell_filename"] and r["time_events_filename"]:
            try:
                st, ttl_ch, te_last = trial_stamps(folder / r["time_events_filename"])
                out["ttl_channel"] = ttl_ch
                cues, key, n_est = cue_times_from_log(Path(r["beh_path"]))
                out.update(n_stamps=len(st), n_cues_log=len(cues), cue_key=key,
                           n_cues_estimated=n_est)
                if len(cues):
                    bl = best_lag(st, cues)
                    out.update(bl)
                    if bl:
                        out["clock_ppm"] = clock_ppm(st, cues, bl["best_lag"])
                out.update(trace_timing(folder / r["cell_filename"], te_last))
            except Exception as e:
                out["error"] = repr(e)
        rows.append(out)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)

    print("\n=== (1) New matching vs the .m rule ===")
    print(df.groupby(["phase", "match_vs_m"]).size().to_string())
    odd = df[df["match_vs_m"].isin(["differs", "m_ambiguous"])]
    if len(odd):
        print("\nCases to inspect:")
        print(odd[["session", "match_vs_m", "new_cell", "m_candidates", "cell_dt_min"]].to_string(index=False))

    if "best_lag" in df:
        ok = df["best_lag"].notna()
        print(f"\n=== (2) Trial-by-trial alignment ({int(ok.sum())} checkable sessions; "
              f"cue key in log: {df['cue_key'].dropna().unique().tolist()}) ===")
        print("Best lag (trials):")
        print(df.loc[ok, "best_lag"].value_counts().sort_index().to_string())
        print("Median ITI error (s) at the best lag:")
        print(df.loc[ok, "iti_median_abs_err_s"].describe().round(4).to_string())
        bad = df[ok & ((df["best_lag"] != 0) | (df["iti_median_abs_err_s"] > 0.05))]
        print(f"\nSessions with lag != 0 or ITI error > 50 ms: {len(bad)}")
        if len(bad):
            print(bad[["session", "n_stamps", "n_cues_log", "best_lag",
                       "iti_median_abs_err_s", "iti_corr"]].to_string(index=False))
        print("\n=== (3) Clock drift, Inscopix vs behavior PC (ppm) ===")
        print(df.loc[ok, "clock_ppm"].describe().round(1).to_string())
        print("(the .m 'shutter delay' equals 1e-5 s / 0.05 s = 200 ppm)")

    if "n_cues_estimated" in df:
        ne = df["n_cues_estimated"].dropna()
        print(f"\nMissing-cue fix (value_getSessionData.m) applied in {int((ne > 0).sum())}/{len(ne)} "
              f"sessions; cues estimated per session: median {ne[ne > 0].median():.0f}, "
              f"max {ne.max():.0f}")

    if "end_gap_ppm" in df:
        ok = df["best_lag"].notna() & (df["iti_median_abs_err_s"] <= 0.05)
        g = df.loc[ok & df["trace_dur_s"].notna()]
        if len(g) > 2:
            slope, icpt = np.polyfit(g["trace_dur_s"], g["te_minus_trace_end_s"], 1)
            r = np.corrcoef(g["trace_dur_s"], g["te_minus_trace_end_s"])[0, 1]
            print(f"\n=== (4b) Camera clock drift (well-aligned sessions, n={len(g)}) ===")
            print(f"timeEvents end - trace end = {icpt:+.3f} s + {1e6*slope:.0f} ppm x duration "
                  f"(r = {r:.2f}); per-session median {g['end_gap_ppm'].median():.0f} ppm")
            print("-> if ~200 ppm and r ~ 1: the .m 'shutter delay' is a valid drift "
                  "correction (keep SHUTTER_DELAY_MODE = 'matlab')")

    if "fs_hz" in df:
        odd = df[(df["fs_hz"] - 20).abs() > 0.5]
        if len(odd):
            print(f"\nSessions NOT at 20 Hz ({len(odd)}): create_dff_files.py hard-codes FS = 20 "
                  "(the .m too), so their windows are wrong there; the Wang runner reads fs from the data")
            print(odd[["session", "fs_hz"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print("\n=== (4) Trace timing ===")
        print(df[["fs_hz", "frame_period_ms", "step_sd_us", "frac_steps_exact",
                  "n_dup_times"]].describe().round(4).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
