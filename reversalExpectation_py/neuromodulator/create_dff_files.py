"""
create_dff_files.py
===================
Port of creatDffMatFiles_miniscope.m (H Atilgan & AC Kwan, 200210).

For each session that has miniscope cell + timeEvents CSV files, this
loads the fluorescence signal, detrends it, aligns to trial trigger
times, and saves a per-trial dF/F array as a .npz file alongside the
log file.

Saved file: <log_stem>_dff.npz  with arrays:
    dff   : (n_trials, tWindow) raw detrended dF/F per trial
    dffN  : (n_trials, tWindow) dF/F normalized to pre-cue baseline
    t     : (tWindow,)          time axis in seconds relative to cue onset
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


FS      = 20        # miniscope sample rate (Hz)
PRE_S   = 2         # seconds before cue to include
POST_S  = 4         # seconds after cue to include
T_WINDOW = int((PRE_S + POST_S) * FS)   # total samples per trial

# The real timeEvents.csv header is "Time (s), Channel Name, Value" (note the
# space after each comma, so pandas' default read_csv gives column names like
# "Time (s)" and " Channel Name" -- not the ChannelName / Time_s_ names this
# module expects). Those expected names come from how MATLAB's readtable
# auto-sanitizes headers (strips spaces/parens into valid identifiers);
# pandas doesn't do that automatically. The column ORDER is fixed, so rename
# positionally instead of relying on exact header text -- robust to stray
# whitespace/casing differences across files.
_TIME_EVENTS_COLS = ["Time_s_", "ChannelName", "Value"]

# "add shutter time delay" in the .m: data.signal_t = t + (1:N)'*0.00001.
# It is CUMULATIVE (0.36 s by the end of 30 min at 20 Hz). It only makes sense if
# the trace times are nominal and the real clock runs ~200 ppm slower.
# Kept for fidelity to the .m; verify_neuromodulator_alignment.py (section 4)
# tells whether it applies. "matlab" = as in the .m | "none" = no shift.
SHUTTER_DELAY_MODE = "matlab"


def movmean_matlab(x: np.ndarray, k: int) -> np.ndarray:
    """MATLAB movmean(x, k): centered window; for even k it covers [i-k/2, i+k/2-1];
    at the ends the window SHRINKS ('shrink' endpoints) instead of padding.
    (uniform_filter1d(mode='nearest') matches in the interior but differs in the
    first and last minute of the session.)"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    before = k // 2 if k % 2 == 0 else (k - 1) // 2
    after = k // 2 - 1 if k % 2 == 0 else (k - 1) // 2
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    lo, hi = np.maximum(0, i - before), np.minimum(n, i + after + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def create_dff_files(data_index: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_index : output of add_index_neuromodulator()

    Returns
    -------
    data_index with added column:
        dff_created : 1.0 if _dff.npz was created/found, else NaN
    """
    dff_created = np.full(len(data_index), float("nan"))

    for idx, (_, row) in enumerate(data_index.iterrows()):
        log_stem = Path(row["session_file"]).stem
        beh_dir  = Path(row["beh_path"]).parent
        out_path = beh_dir / f"{log_stem}_dff.npz"

        if out_path.exists():
            dff_created[idx] = 1.0
            continue

        cell_ok = not (isinstance(row.get("cell_created"), float) and math.isnan(row["cell_created"]))
        te_ok   = not (isinstance(row.get("time_events_created"), float) and math.isnan(row["time_events_created"]))
        if not (cell_ok and te_ok):
            continue

        try:
            dff, dffN = _process_session(row)
            np.savez(out_path, dff=dff, dffN=dffN,
                     t=np.arange(-PRE_S, POST_S, 1.0 / FS)[:T_WINDOW])
            dff_created[idx] = 1.0
            print(f"  Created {out_path.name}")
        except Exception as e:
            warnings.warn(f"  Failed {log_stem}: {e}")

    result = data_index.copy()
    result["dff_created"] = dff_created
    return result


def _process_session(row) -> tuple:
    """Load, detrend, and epoch the fluorescence signal for one session."""
    neural_path = Path(row["neural_data_path"])

    # --- Load fluorescence signal (cell CSV) ---
    cell_path = neural_path / row["cell_filename"]
    raw = pd.read_csv(cell_path, skiprows=2, header=None)
    signal_t   = raw.iloc[:, 0].values.astype(float)
    raw_signal = raw.iloc[:, 1].values.astype(float)
    # "add shutter time delay" (see SHUTTER_DELAY_MODE above)
    if SHUTTER_DELAY_MODE == "matlab":
        signal_t = signal_t + np.arange(1, len(signal_t) + 1) * 1e-5

    # --- Load trial trigger times (timeEvents CSV) ---
    te_path = neural_path / row["time_events_filename"]
    te = pd.read_csv(te_path)
    if len(te.columns) != len(_TIME_EVENTS_COLS):
        raise ValueError(f"{te_path.name}: expected {len(_TIME_EVENTS_COLS)} "
                         f"columns, got {len(te.columns)} ({list(te.columns)})")
    te.columns = _TIME_EVENTS_COLS
    io1 = te[te["ChannelName"].astype(str).str.strip() == "IO1"]
    times  = io1["Time_s_"].values.astype(float)
    values = io1["Value"].values.astype(float)

    # Transitions (value changes) → take every other one starting from 2nd
    transitions = np.where(values[:-1] != values[1:])[0]
    trial_stamps = times[transitions][1::2]   # mirrors MATLAB trialStamps(2:2:end)

    # --- Detrend: 2-minute moving average (movmean as in the .m, 'shrink' endpoints) ---
    window = int(FS * 60 * 2)
    trend  = movmean_matlab(raw_signal, window)
    signal = (raw_signal - trend) / np.nanmean(raw_signal)

    # --- Epoch into per-trial windows ---
    n_trials = len(trial_stamps) - 1
    pre_samp = round(PRE_S * FS)

    dff  = np.full((n_trials, T_WINDOW), np.nan)
    dffN = np.full((n_trials, T_WINDOW), np.nan)

    for k in range(n_trials):
        st_idx  = int(np.argmin(np.abs(signal_t - trial_stamps[k])))
        end_idx = int(np.argmin(np.abs(signal_t - trial_stamps[k + 1])))
        start   = st_idx - pre_samp

        # .m: ind(1) <= 0 with 1-based indices  <=>  start < 0 with 0-based ones.
        # (the previous "start <= 0" wrongly treated start == 0 as the first-trial case)
        if start < 0:
            # Not enough pre-cue signal for first trial
            seg = signal[st_idx:end_idx]
            n   = min(len(seg), T_WINDOW - pre_samp)   # cap to what actually fits
            dff[k,  pre_samp:pre_samp + n] = seg[:n]
            dffN[k, pre_samp:pre_samp + n] = seg[:n]
        else:
            seg = signal[start:end_idx]
            n   = min(len(seg), T_WINDOW)
            dff[k, :n]  = seg[:n]
            # Note: the .m does not preallocate data.dff, so MATLAB fills unassigned
            # positions of short trials with ZEROS; here they stay NaN
            # (deliberate divergence: a zero would be a fake data point).
            baseline     = np.nanmean(signal[start:start + pre_samp])
            dffN[k, :n] = seg[:n] - baseline

    return dff, dffN
