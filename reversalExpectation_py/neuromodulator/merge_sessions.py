"""
merge_sessions.py (neuromodulator)
===================================
Port of merge_sessions_neuromodulator.m (H Atilgan & AC Kwan, 201002).

Like behavior/merge_sessions.py but also concatenates the dF/F arrays
(dff, dffN) across sessions, inserting NaN rows between sessions.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.log_parser    import parse_logfile, get_session_data, detect_phase
from behavior.trial_processing   import get_trial_masks


N_NAN    = 20     # NaN trials inserted between sessions
T_WINDOW = 120    # samples per trial (matches create_dff_files.py: 6 s × 20 Hz)

# Session-level metadata fields in trial_data that are scalars, not
# per-trial arrays -- must not be sliced/concatenated like the rest.
# ("n_rules" is added by hand a few lines below with trial_data["n_rules"]
# = n_rules; without excluding it here, np.asarray(31)[:n_trials] blows up
# with "too many indices for array: array is 0-dimensional".)
_SCALAR_FIELDS = {"presCodeSet", "n_rules"}

# Fields that are lists of ragged per-trial arrays (each trial can have a
# different number of licks) -- np.asarray() on these fails with
# "inhomogeneous shape" in recent numpy. Concatenate as Python lists with
# NaN-placeholder "trials" for the gap, same convention as
# behavior/merge_sessions.py's _CELL_FIELDS.
_CELL_FIELDS = {"leftlickTimes", "rightlickTimes"}


def merge_sessions_neuromodulator(data_index: pd.DataFrame) -> dict:
    """
    Merge sessions from data_index into one long session with NaN gaps.

    Parameters
    ----------
    data_index : rows from the neuromodulator data index (one animal,
                 sorted by date)

    Returns
    -------
    dict with keys:
        trial_data : merged trial_data dict
        trials     : merged trials dict (includes dff, dffN, dffN_zscore)
        n_rules    : int
    """
    from scipy.stats import zscore as _zscore

    trial_data_combined = {"presCodeSet": 31}
    trials_combined     = {}
    n_rules_global      = None

    nan_row  = np.full(N_NAN, np.nan)
    nan_dff  = np.full((N_NAN, T_WINDOW), np.nan)

    for i, (_, row) in enumerate(data_index.iterrows()):
        log_path = Path(row["beh_path"])
        dff_path = log_path.parent / f"{log_path.stem}_dff.npz"

        # --- Load behavioral data ---
        try:
            log_data = parse_logfile(log_path)
            phase    = detect_phase(log_data["scenario"])
            session_data, trial_data = get_session_data(log_data, phase)
        except Exception as e:
            warnings.warn(f"Cannot load {log_path.name}: {e}")
            continue

        n_rules = session_data["nRules"]
        trial_data["n_rules"] = n_rules

        trials = get_trial_masks(trial_data)

        # --- Load neural data ---
        if dff_path.exists():
            npz = np.load(dff_path)
            dff  = npz["dff"]
            dffN = npz["dffN"]
        else:
            dff  = np.full((len(trial_data["cue"]), T_WINDOW), np.nan)
            dffN = dff.copy()

        # .m: data.dffN(1:end-1, 1:tWindow) -> the last row is dropped
        dff, dffN = dff[:-1], dffN[:-1]

        # Align trial counts
        n_trials = min(len(trial_data["cue"]), len(dff))

        # .m: zscore(data.dffN(1:end-1,1:tWindow)) -> z-score PER COLUMN (each
        # time sample across trials), not over the whole matrix.
        dffN_zscore = _zscore(dffN[:n_trials, :T_WINDOW], axis=0, nan_policy="omit")
        trials["dff"]         = dff[:n_trials, :T_WINDOW]
        trials["dffN"]        = dffN[:n_trials, :T_WINDOW]
        trials["dffN_zscore"] = dffN_zscore

        if n_rules_global is None:
            n_rules_global = n_rules
            for field in trial_data:
                if field in _SCALAR_FIELDS:
                    continue
                val = trial_data[field]
                if isinstance(val, list):
                    trial_data_combined[field] = val[:n_trials]
                else:
                    arr = np.asarray(val)
                    if arr.ndim == 0:
                        # Defensive: any other scalar field we haven't
                        # explicitly named above -- skip rather than crash.
                        continue
                    trial_data_combined[field] = arr[:n_trials]

            for field in trials:
                trial_data_combined[field] = np.asarray(trials[field])[:n_trials]
                trials_combined[field]     = np.asarray(trials[field])[:n_trials]
        else:
            if n_rules != n_rules_global:
                warnings.warn(f"n_rules mismatch in {log_path.name}, skipping")
                continue

            for field in trial_data:
                if field in _SCALAR_FIELDS:
                    continue
                raw = trial_data[field]
                if field in _CELL_FIELDS or isinstance(raw, list):
                    nan_gap = [np.array([np.nan])] * N_NAN
                    trial_data_combined[field] = (
                        trial_data_combined[field] + nan_gap + list(raw[:n_trials])
                    )
                    continue
                val = np.asarray(raw)
                if val.ndim == 1:
                    trial_data_combined[field] = np.concatenate(
                        [trial_data_combined[field], nan_row, val[:n_trials]]
                    )

            dff_fields  = {"dff", "dffN", "dffN_zscore"}
            scalar_nan  = nan_row

            for field in trials:
                arr = np.asarray(trials[field])[:n_trials]
                if field in dff_fields:
                    trials_combined[field] = np.vstack(
                        [trials_combined[field], nan_dff, arr]
                    )
                else:
                    trials_combined[field] = np.concatenate(
                        [trials_combined[field], scalar_nan, arr]
                    )

    return {
        "trial_data": trial_data_combined,
        "trials":     trials_combined,
        "n_rules":    n_rules_global or 2,
    }