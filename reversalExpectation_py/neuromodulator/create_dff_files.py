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
from scipy.ndimage import uniform_filter1d


FS      = 20        # miniscope sample rate (Hz)
PRE_S   = 2         # seconds before cue to include
POST_S  = 4         # seconds after cue to include
T_WINDOW = int((PRE_S + POST_S) * FS)   # total samples per trial


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
    # Add tiny shutter-time jitter (mirrors MATLAB original)
    signal_t = signal_t + np.arange(1, len(signal_t) + 1) * 1e-5

    # --- Load trial trigger times (timeEvents CSV) ---
    te_path = neural_path / row["time_events_filename"]
    te = pd.read_csv(te_path)
    io1 = te[te["ChannelName"] == "IO1"]
    times  = io1["Time_s_"].values.astype(float)
    values = io1["Value"].values.astype(float)

    # Transitions (value changes) → take every other one starting from 2nd
    transitions = np.where(values[:-1] != values[1:])[0]
    trial_stamps = times[transitions][1::2]   # mirrors MATLAB trialStamps(2:2:end)

    # --- Detrend: 2-minute moving average ---
    window = int(FS * 60 * 2)
    trend  = uniform_filter1d(raw_signal, size=window, mode="nearest")
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

        if start <= 0:
            # Not enough pre-cue signal for first trial
            seg = signal[st_idx:end_idx]
            n   = len(seg)
            dff[k,  pre_samp:pre_samp + n] = seg
            dffN[k, pre_samp:pre_samp + n] = seg
        else:
            seg = signal[start:end_idx]
            n   = min(len(seg), T_WINDOW)
            dff[k, :n]  = seg[:n]
            baseline     = np.nanmean(signal[start:start + pre_samp])
            dffN[k, :n] = seg[:n] - baseline

    return dff, dffN
