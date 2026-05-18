"""
flip_trial_data.py
==================
Translation of fliptrialData.m
(H Atilgan & AC Kwan 191208).

Flips the choice direction in trial_data so that lesion experiments
can be analyzed with ipsilateral/contralateral as the reference frame
rather than left/right.
"""

import numpy as np

from preprocessing.presentation_codes import get_presentation_codes


def flip_trial_data(trial_data: dict) -> dict:
    """
    Translation of fliptrialData.m.

    Returns a copy of trial_data with left↔right swapped for
    response, outcome, rule, and lick-time arrays.

    Currently supported phases: 3 and 8 (two-rule task).
    Raises ValueError for other phases.

    Parameters
    ----------
    trial_data : dict
        Output of get_session_data() or merge_sessions().

    Returns
    -------
    new_trial_data : dict
        Copy with flipped laterality.
    """
    phase = trial_data["presCodeSet"]
    if phase not in (3, 8):
        raise ValueError(
            f"flip_trial_data: phase {phase} is not supported. "
            "Currently only phases 3 and 8 (two-rule task) are implemented."
        )

    stim, resp, outcome, rule_codes, event = get_presentation_codes(phase)

    new = {k: (v.copy() if isinstance(v, np.ndarray) else list(v))
           for k, v in trial_data.items()}

    # --- Swap lick times ---
    new["leftlickTimes"]  = list(trial_data["rightlickTimes"])
    new["rightlickTimes"] = list(trial_data["leftlickTimes"])

    # --- Flip responses ---
    resp_arr = trial_data["response"].copy()
    new_resp = resp_arr.copy()
    new_resp[resp_arr == resp.LEFT]  = resp.RIGHT
    new_resp[resp_arr == resp.RIGHT] = resp.LEFT
    new["response"] = new_resp

    # --- Flip outcomes ---
    out_arr = trial_data["outcome"].copy()
    new_out = out_arr.copy()
    new_out[out_arr == outcome.REWARDLEFT]    = outcome.REWARDRIGHT
    new_out[out_arr == outcome.REWARDRIGHT]   = outcome.REWARDLEFT
    new_out[out_arr == outcome.NOREWARDLEFT]  = outcome.NOREWARDRIGHT
    new_out[out_arr == outcome.NOREWARDRIGHT] = outcome.NOREWARDLEFT
    new["outcome"] = new_out

    # --- Flip rules ---
    rule_arr = trial_data["rule"].copy()
    new_rule = rule_arr.copy()
    new_rule[rule_arr == rule_codes.L70R10] = rule_codes.L10R70
    new_rule[rule_arr == rule_codes.L10R70] = rule_codes.L70R10
    new["rule"] = new_rule

    return new
