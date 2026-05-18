"""
log_parser.py
=============
Exact translation of parseLogfile.m + value_getSessionData.m
(H Atilgan & AC Kwan, 191210).

Parses a single NBS Presentation .log file into trial-level arrays,
replicating the exact logic of the MATLAB originals.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .presentation_codes import get_presentation_codes, REWARD_PROBS, RULE_LABELS


# ---------------------------------------------------------------------------
# Phase detection from scenario name (from createBehMatFiles.m)
# ---------------------------------------------------------------------------

SCENARIO_TO_PHASE = {
    "Phase3R_71_NoCue":          3,
    "Phase3_R71_NoCue":          3,
    "Phase3_R71NoCue":           3,
    "Phase6_Value":              6,
    "Phase8_R71NoCueWithPupil":  8,
    "Phase21_R71NoCueOpto":      21,
    "Phase22_R71NoCueOpto":      22,
    "phase31_R71NoCueNM":        31,
    "Phase31_R71NoCueWithPupil": 31,
    "Phase32_R71NoCueNM":        32,
}


def detect_phase(scenario: str) -> int:
    """Map scenario name to phase integer. Raises ValueError if unknown."""
    phase = SCENARIO_TO_PHASE.get(scenario.strip())
    if phase is None:
        raise ValueError(f"Unknown scenario: '{scenario}'. "
                         f"Known: {list(SCENARIO_TO_PHASE.keys())}")
    return phase


# ---------------------------------------------------------------------------
# parseLogfile.m — read raw events from disk
# ---------------------------------------------------------------------------

def parse_logfile(log_path: Path) -> Dict:
    """
    Translation of parseLogfile.m.

    Reads header + all data lines from a .log file.

    Returns
    -------
    dict with keys:
        subject    : str
        dateTime   : list[str, str]  (date string, time string)
        scenario   : str
        TYPE       : np.ndarray[str]
        CODE       : np.ndarray[float]
        TIME       : np.ndarray[float]   (raw Presentation time units)
    """
    log_path = Path(log_path)

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # ---- Parse header (first 3 lines) ----
    # Line 0: "Scenario - <name>"
    scenario_line = raw_lines[0].strip()
    m = re.search(r"Scenario\s*-\s*(.+)", scenario_line)
    scenario = m.group(1).strip() if m else scenario_line

    # Line 1: "Logfile written - MM/DD/YYYY HH:MM:SS"
    datetime_line = raw_lines[1].strip()
    m = re.search(r"written\s*-\s*(\S+)\s+(\S+)", datetime_line)
    if m:
        date_str = m.group(1)   # e.g. "05/02/2019"
        time_str = m.group(2)   # e.g. "17:28:56"
    else:
        date_str, time_str = "", ""

    # Lines 2-3: column headers (skip)

    # ---- Find first data line (Trial == 1) ----
    # MATLAB logic: skip until it finds a line whose second column == '1'
    # We skip lines that start with non-numeric first column
    data_lines = []
    found_first = False
    subject = ""

    for line in raw_lines[4:]:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 5:
            continue
        try:
            trial_num = int(parts[1])
        except ValueError:
            continue
        # First valid data line gives us the subject
        if not found_first:
            subject = parts[0].strip()
            found_first = True
        data_lines.append(parts)

    if not data_lines:
        raise ValueError(f"No data lines found in {log_path}")

    # ---- Build arrays ----
    TYPE = np.array([p[2].strip() for p in data_lines])
    TIME_raw = np.array([float(p[4]) if p[4].strip() else np.nan
                         for p in data_lines])

    # CODE: mostly numeric but sometimes strings like "BlockLen_13"
    CODE_str = [p[3].strip() for p in data_lines]
    CODE = np.full(len(CODE_str), np.nan)
    for i, c in enumerate(CODE_str):
        try:
            CODE[i] = float(c)
        except ValueError:
            pass  # non-numeric codes stay NaN

    return {
        "subject":  subject,
        "dateTime": [date_str, time_str],
        "scenario": scenario,
        "TYPE":     TYPE,
        "CODE":     CODE,
        "TIME":     TIME_raw,
    }


# ---------------------------------------------------------------------------
# value_getSessionData.m — extract trial-level data
# ---------------------------------------------------------------------------

def get_session_data(log_data: Dict, phase: int) -> Tuple[Dict, Dict]:
    """
    Translation of value_getSessionData.m.

    Parameters
    ----------
    log_data : dict   Output of parse_logfile()
    phase    : int    Task phase

    Returns
    -------
    session_data : dict
    trial_data   : dict
        Both mirror the MATLAB struct fields exactly.
    """
    stim, resp, outcome, rule, event = get_presentation_codes(phase)

    TYPE = log_data["TYPE"]
    CODE = log_data["CODE"]
    TIME = log_data["TIME"]

    # Remove Port events (shouldn't exist, but replicate the MATLAB check)
    port_mask = TYPE == "Port"
    if port_mask.any():
        warnings.warn("Port events found — removing them (check Presentation settings)")
        keep = ~port_mask
        TYPE = TYPE[keep]
        CODE = CODE[keep]
        TIME = TIME[keep]

    # Get all rule codes as array
    rule_codes_all = _rule_codes_as_array(rule)

    # Trim: keep only from first rule event onward
    first_rule = np.where(np.isin(CODE, rule_codes_all))[0]
    if len(first_rule) == 0:
        raise ValueError("No rule events found in log file")
    first_rule = first_rule[0]
    TYPE = TYPE[first_rule:]
    CODE = CODE[first_rule:]
    TIME = TIME[first_rule:]

    # Trim: keep only up to last WAITCUE event
    last_waitcue = np.where(CODE == event.WAITCUE)[0]
    if len(last_waitcue) == 0:
        raise ValueError("No WAITCUE events found in log file")
    last_waitcue = last_waitcue[-1]
    TYPE = TYPE[:last_waitcue + 1]
    CODE = CODE[:last_waitcue + 1]
    TIME = TIME[:last_waitcue + 1]

    # Time axis: start at zero, convert to seconds
    # MATLAB uses /10000 for the original Presentation time units
    t = (TIME - TIME[0]) / 10000.0

    # Lick times (all session)
    lick_times_left  = t[(TYPE == "Response") & (CODE == resp.LEFT)]
    lick_times_right = t[(TYPE == "Response") & (CODE == resp.RIGHT)]

    # --- Cue times ---
    cue_codes = np.array([stim.GO])
    cue_mask = (TYPE == "Sound") & np.isin(CODE, cue_codes)
    cue_times = t[cue_mask]
    cue_codes_out = CODE[cue_mask]

    # --- Outcome times ---
    outcome_codes_all = _outcome_codes_as_array(outcome)
    outcome_mask = ((TYPE == "Nothing") | (TYPE == "Sound")) & \
                   np.isin(CODE, outcome_codes_all)
    outcome_times = t[outcome_mask]
    outcome_codes_out = CODE[outcome_mask]

    # --- Rule times ---
    rule_mask = (TYPE == "Nothing") & np.isin(CODE, rule_codes_all)
    rule_times = t[rule_mask]
    rule_codes_out = CODE[rule_mask]

    n_trials = len(rule_times)

    # --- Consistency check (from value_getSessionData.m) ---
    n_cue     = len(cue_times)
    n_outcome = len(outcome_times)
    n_rule    = len(rule_times)

    if n_outcome != n_cue or n_outcome != n_rule or n_cue != n_rule:
        msg = (f"Inconsistent counts: rule={n_rule}, cue={n_cue}, "
               f"outcome={n_outcome}")
        if n_rule > n_cue:
            # Apply missing-cue fix (from value_getSessionData.m)
            warnings.warn(f"{msg} — applying missing-cue fix")
            cue_times, cue_codes_out = _fix_missing_cues(
                cue_times, cue_codes_out, rule_times, cue_codes_out[0]
            )
            n_cue = len(cue_times)
            if n_outcome != n_cue or n_cue != n_rule:
                raise ValueError(
                    f"Still inconsistent after missing-cue fix: "
                    f"rule={n_rule}, cue={n_cue}, outcome={n_outcome}"
                )
        else:
            raise ValueError(f"Inconsistent event counts: {msg}")

    # --- Responses ---
    resp_mask = (TYPE == "Response") & \
                ((CODE == resp.LEFT) | (CODE == resp.RIGHT))
    resp_times = t[resp_mask]
    resp_codes = CODE[resp_mask]

    response = np.zeros(n_trials, dtype=np.uint32)
    rt = np.full(n_trials, np.nan)

    miss_codes = np.array([outcome.MISS, outcome.REWARDMANUAL])
    non_miss_idx = np.where(~np.isin(outcome_codes_out, miss_codes))[0]

    for i in non_miss_idx:
        after_cue = resp_times > cue_times[i]
        if after_cue.any():
            first = np.where(after_cue)[0][0]
            response[i] = int(resp_codes[first])
            rt[i] = resp_times[first] - cue_times[i]

    # --- Per-trial lick times ---
    left_lick_times  = []
    right_lick_times = []
    for i in range(n_trials):
        time1 = cue_times[i - 1] if i > 0 else 0.0
        time2 = cue_times[i + 1] if i < n_trials - 1 else t[-1]

        ll = lick_times_left
        ll = ll[(ll >= time1) & (ll <= time2)] - cue_times[i]
        left_lick_times.append(ll)

        rl = lick_times_right
        rl = rl[(rl >= time1) & (rl <= time2)] - cue_times[i]
        right_lick_times.append(rl)

    # ITI: from outcome to next cue
    iti = np.full(n_trials, np.nan)
    iti[:-1] = cue_times[1:] - outcome_times[:-1]

    # --- Assemble structs ---
    n_rules = len(_rule_codes_as_array(rule))

    session_data = {
        "subject":    log_data["subject"],
        "dateTime":   log_data["dateTime"],
        "nTrials":    n_trials,
        "nRules":     n_rules,
        "lickTimes":  [lick_times_left, lick_times_right],
        "rule_labels": list(RULE_LABELS.get(n_rules, {}).values()),
    }

    trial_data = {
        "presCodeSet":    phase,
        "cue":            cue_codes_out.astype(int),
        "cueTimes":       cue_times,
        "outcome":        outcome_codes_out.astype(int),
        "outcomeTimes":   outcome_times,
        "rule":           rule_codes_out.astype(int),
        "ruleTimes":      rule_times,
        "response":       response,
        "rt":             rt,
        "iti":            iti,
        "leftlickTimes":  left_lick_times,
        "rightlickTimes": right_lick_times,
    }

    return session_data, trial_data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _rule_codes_as_array(rule) -> np.ndarray:
    """Extract all rule codes from a RuleCodes dataclass."""
    from dataclasses import fields
    return np.array([getattr(rule, f.name) for f in fields(rule)
                     if getattr(rule, f.name) is not None], dtype=float)


def _outcome_codes_as_array(outcome) -> np.ndarray:
    from dataclasses import fields
    return np.array([getattr(outcome, f.name) for f in fields(outcome)
                     if getattr(outcome, f.name) is not None], dtype=float)


def _fix_missing_cues(cue_times, cue_codes, rule_times, default_cue_code):
    """
    Translation of the missing-cue fix in value_getSessionData.m.

    When n_rule > n_cue, estimate missing cue times from rule times.
    """
    # Compute typical rule-to-cue delay
    diff_rulecue = []
    for j in range(len(cue_times)):
        diffs = cue_times[j] - rule_times
        pos = diffs[diffs > 0]
        if len(pos) > 0:
            diff_rulecue.append(pos.min())
    min_diff = np.min(diff_rulecue) if diff_rulecue else 0.01

    # Find rules with no cue between them and the next rule
    n_rule = len(rule_times)
    missing_idx = []
    for j in range(n_rule - 1):
        n_cue_in_window = np.sum(
            (cue_times > rule_times[j]) & (cue_times < rule_times[j + 1])
        )
        if n_cue_in_window == 0:
            missing_idx.append(j)

    # Insert estimated cue times
    new_cue_times = np.sort(np.concatenate([
        cue_times,
        rule_times[missing_idx] + min_diff
    ]))
    new_cue_codes = np.concatenate([
        cue_codes,
        np.full(len(missing_idx), default_cue_code, dtype=cue_codes.dtype)
    ])
    sort_idx = np.argsort(new_cue_times)
    return new_cue_times[sort_idx], new_cue_codes[sort_idx]
