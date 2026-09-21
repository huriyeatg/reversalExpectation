"""
trial_processing.py
===================
Exact translation of value_getTrialMasks.m + value_getTrialStats.m
(H Atilgan & AC Kwan, 191202).

Converts trial_data dicts into flat trial-level arrays for analysis.
"""

import warnings
import numpy as np

from preprocessing.presentation_codes import get_presentation_codes, REWARD_PROBS, RULE_LABELS


def _bool_mask(mask: np.ndarray) -> np.ndarray:
    """
    Safely coerce a trial mask to boolean for use as a numpy index.

    get_trial_masks() returns pure booleans for a single session, but
    merge_sessions (both behavior/ and neuromodulator/ versions) insert NaN
    "gap" rows between concatenated sessions, which promotes the whole mask
    array to float64 -- and float arrays can't be used as boolean indices
    ("arrays used as indices must be of integer (or boolean) type").
    mask.astype(bool) directly would be WRONG here: NaN casts to True in
    numpy (NaN != 0), so every gap trial would be marked True for every
    mask. Route NaN -> 0.0 -> False first instead.
    """
    if mask.dtype == bool:
        return mask
    return np.nan_to_num(np.asarray(mask, dtype=float), nan=0.0).astype(bool)


# ---------------------------------------------------------------------------
# value_getTrialMasks.m
# ---------------------------------------------------------------------------

def get_trial_masks(trial_data: dict) -> dict:
    """
    Translation of value_getTrialMasks.m.

    Parameters
    ----------
    trial_data : dict   Output of get_session_data()

    Returns
    -------
    trials : dict
        Boolean arrays for each trial category, matching MATLAB field names:
        go, left, right, reward, noreward, miss,
        L70R10, L10R70  (or 6-rule equivalents)
    """
    phase = trial_data["presCodeSet"]
    stim, resp, outcome, rule, event = get_presentation_codes(phase)

    cue      = trial_data["cue"]
    response = trial_data["response"]
    out      = trial_data["outcome"]
    rule_arr = trial_data["rule"]

    trials = {}

    # --- Cue ---
    trials["go"] = cue == stim.GO

    # --- Response ---
    trials["left"]  = response == resp.LEFT
    trials["right"] = response == resp.RIGHT

    # --- Outcome ---
    trials["reward"]   = np.isin(out, [outcome.REWARDLEFT, outcome.REWARDRIGHT])
    trials["noreward"] = np.isin(out, [outcome.NOREWARDLEFT, outcome.NOREWARDRIGHT])
    trials["miss"]     = np.isin(out, [outcome.MISS, outcome.REWARDMANUAL])

    # --- Rule ---
    if phase in (3, 8, 1, 2, 21, 22, 31, 32):
        from preprocessing.presentation_codes import RuleCodes2
        trials["L70R10"] = rule_arr == rule.L70R10
        trials["L10R70"] = rule_arr == rule.L10R70
    elif phase == 6:
        from preprocessing.presentation_codes import RuleCodes6
        trials["L70R30"] = rule_arr == rule.L70R30
        trials["L70R10"] = rule_arr == rule.L70R10
        trials["L30R10"] = rule_arr == rule.L30R10
        trials["L30R70"] = rule_arr == rule.L30R70
        trials["L10R70"] = rule_arr == rule.L10R70
        trials["L10R30"] = rule_arr == rule.L10R30

    # --- Consistency checks (mirrors MATLAB) ---
    # .m: nTrials = numel(trialData.cueTimes); check #2 is in an ELSEIF, so it
    # only runs (and only raises) when check #1 passed.
    n = len(trial_data["cueTimes"]) if "cueTimes" in trial_data else len(cue)
    n_outcome = trials["reward"].sum() + trials["noreward"].sum() + trials["miss"].sum()
    n_response = trials["left"].sum() + trials["right"].sum() + trials["miss"].sum()
    if n_outcome != n:
        warnings.warn(f"check #1 failed: reward+noreward+miss={n_outcome} vs nTrials={n}")
    elif n_response != n:
        raise ValueError(
            f"check #2 failed: left+right+miss={n_response} vs nTrials={n}"
        )

    if phase == 6:
        n_rule = sum(trials[k].sum() for k in
                     ["L70R30","L70R10","L30R10","L30R70","L10R70","L10R30"])
        if n_rule != n:
            warnings.warn(f"check #3 failed: rule counts={n_rule} vs nTrials={n}")
    elif "L70R10" in trials:
        n_rule = trials["L70R10"].sum() + trials["L10R70"].sum()
        if n_rule != n:
            warnings.warn(f"check #4 failed: rule counts={n_rule} vs nTrials={n}")

    return trials


# ---------------------------------------------------------------------------
# value_getTrialStats.m
# ---------------------------------------------------------------------------

def get_trial_stats(trials: dict, n_rules: int) -> dict:
    """
    Translation of value_getTrialStats.m.

    Parameters
    ----------
    trials  : dict   Output of get_trial_masks()
    n_rules : int    Number of reward probability sets (2 or 6)

    Returns
    -------
    stats : dict with keys:
        c            : float array, choice (-1=left, 1=right, NaN=miss)
        r            : float array, outcome (1=reward, 0=noreward, NaN=miss)
        rule         : float array, rule index (1..n_rules, NaN=miss)
        rewardprob   : (n_trials, 2) array, [left_prob, right_prob]
        rule_labels  : list[str]
    """
    n = len(trials["go"])

    # choice: left=-1, right=1, miss=NaN
    c = np.full(n, np.nan)
    c[_bool_mask(trials["left"])]  = -1.0
    c[_bool_mask(trials["right"])] =  1.0

    # outcome: reward=1, noreward=0, miss=NaN
    r = np.full(n, np.nan)
    r[_bool_mask(trials["reward"])]   = 1.0
    r[_bool_mask(trials["noreward"])] = 0.0

    # rule index
    rule = np.full(n, np.nan)
    prob_list = REWARD_PROBS.get(n_rules, {})
    rule_labels_map = RULE_LABELS.get(n_rules, {})

    if n_rules == 2:
        rule[_bool_mask(trials["L70R10"])] = 1
        rule[_bool_mask(trials["L10R70"])] = 2
    elif n_rules == 6:
        for idx, key in enumerate(
            ["L70R30","L70R10","L30R10","L30R70","L10R70","L10R30"], start=1
        ):
            if key in trials:
                rule[_bool_mask(trials[key])] = idx

    # reward probabilities per trial
    rewardprob = np.full((n, 2), np.nan)
    for rule_idx, (lp, rp) in prob_list.items():
        mask = rule == rule_idx
        rewardprob[mask, 0] = lp
        rewardprob[mask, 1] = rp

    return {
        "c":           c,
        "r":           r,
        "rule":        rule,
        "rewardprob":  rewardprob,
        "rule_labels": list(rule_labels_map.values()),
        "n_rules":     n_rules,
    }