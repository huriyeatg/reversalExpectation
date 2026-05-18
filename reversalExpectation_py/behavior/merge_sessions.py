"""
merge_sessions.py
=================
Translation of merge_sessions.m
(H Atilgan & AC Kwan 191204).

Concatenates multiple sessions into one long session,
inserting NaN gaps between sessions so block-boundary
detection in value_getTrialStatsMore still works correctly.
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import scipy.io as sio

from .trial_processing import get_trial_masks

N_NAN = 20   # NaN trials inserted between sessions (mirrors MATLAB)

# Fields that contain cell arrays of timestamps in MATLAB
_CELL_FIELDS = {"leftlickTimes", "rightlickTimes"}


def merge_sessions(session_paths: List[Path]) -> tuple:
    """
    Translation of merge_sessions.m.

    Loads each *_beh.mat file in session_paths, concatenates all
    trialData fields and trial masks, inserting N_NAN NaN rows
    between consecutive sessions.

    Parameters
    ----------
    session_paths : list of Path
        Paths to *_beh.mat files (output of createBehMatFiles / the
        MATLAB preprocessing pipeline).

    Returns
    -------
    trial_data_combined : dict
        Concatenated trialData with NaN gaps.
    trials_combined : dict
        Concatenated boolean/float trial masks with NaN gaps.
    n_rules : int
        Number of reward-probability sets (must be the same for all
        sessions; raises ValueError otherwise).
    """
    trial_data_combined = None
    trials_combined     = None
    n_rules_ref         = None

    for i, path in enumerate(session_paths):
        path = Path(path)
        mat = sio.loadmat(
            str(path), struct_as_record=False, squeeze_me=True
        )
        td  = mat["trialData"]
        sd  = mat["sessionData"]

        # Build Python trial_data dict from MATLAB struct
        trial_data = {
            "presCodeSet": int(td.presCodeSet),
            "cue":         td.cue.astype(float),
            "response":    td.response.astype(float),
            "outcome":     td.outcome.astype(float),
            "rule":        td.rule.astype(float),
            "cueTimes":    td.cueTimes.astype(float),
            "rt":          td.rt.astype(float),
            "iti":         td.iti.astype(float),
            # lick times: list of 1-D arrays (one per trial)
            "leftlickTimes":  list(td.leftlickTimes),
            "rightlickTimes": list(td.rightlickTimes),
        }

        n_rules_session = int(sd.nRules)

        trials = get_trial_masks(trial_data)
        # Convert boolean masks to float so NaN gaps can be inserted
        trials = {k: v.astype(float) for k, v in trials.items()}

        if i == 0:
            trial_data_combined = trial_data
            trials_combined     = trials
            n_rules_ref         = n_rules_session
        else:
            if n_rules_session != n_rules_ref:
                raise ValueError(
                    f"merge_sessions: nRules mismatch — session {i} has "
                    f"{n_rules_session}, expected {n_rules_ref}."
                )

            n = N_NAN
            # --- Concatenate trial_data fields ---
            for field in trial_data:
                if field in _CELL_FIELDS:
                    nan_gap = [np.array([np.nan])] * n
                    trial_data_combined[field] = (
                        trial_data_combined[field] + nan_gap + trial_data[field]
                    )
                else:
                    nan_gap = np.full(n, np.nan)
                    trial_data_combined[field] = np.concatenate(
                        [trial_data_combined[field], nan_gap, trial_data[field]]
                    )

            # --- Concatenate trial masks ---
            for field in trials:
                nan_gap = np.full(n, np.nan)
                trials_combined[field] = np.concatenate(
                    [trials_combined[field], nan_gap, trials[field]]
                )

    return trial_data_combined, trials_combined, n_rules_ref
