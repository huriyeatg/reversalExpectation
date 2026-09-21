"""
neuromodulator_trials.py
========================
Per-trial behavioral table for the neuromodulator (imaging) sessions, built
with exactly the same chain as the behavioral dataset:

    parse_logfile -> get_session_data -> get_trial_masks
                  -> get_trial_stats -> get_trial_stats_more

Columns match bandit_R71_lesion.csv where they overlap (choice, rewarded,
hr_side, block_idx, block_trial_to_crit, block_trial_random_added), plus the
fields the neuromodulator analyses need (rt, cue time in the log clock, tau).

Conventions (same as the behavioral analyses):
    choice   : -1 left / +1 right / NaN miss
    rewarded : 1 / 0 / NaN miss
    tau      : pos_in_block - block_trial_to_crit (0 = first trial after the
               criterion trial); pre-switch L_Random window = 0 <= tau < L_Random
    in_lrandom_window : tau window AND complete block (last block excluded,
               its L_Random is right-censored)
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.log_parser import parse_logfile, get_session_data, detect_phase
from behavior.trial_processing import get_trial_masks, get_trial_stats
from behavior.trial_stats_more import get_trial_stats_more


# ---------------------------------------------------------------------------
# Animals excluded from ALL neural analyses (behavior logs are unaffected)
# ---------------------------------------------------------------------------
# 19303: recorded / exported with a protocol that cannot be reconstructed.
#   - 3 of 5 sessions have an already-processed trace (values around -0.05,
#     not raw fluorescence), so dF/F cannot be computed;
#   - those 3 sessions are not at 20 Hz (15.6, 17.5, 21.7 Hz), unlike every
#     other session in the dataset;
#   - their timeEvents / trace end gaps (-230 to +103 s) have no physical
#     meaning (all other sessions: camera drift 251-262 ppm, IQR);
#   - the 2 remaining sessions show an outcome response of opposite sign.
# This is a data-quality exclusion, not a performance criterion. It is applied
# in run_wang2025_neuromodulator.py and neuromodulator_exploratory.py; the
# behavioral GLM-HMM validation keeps 19303 (its logs are fine).
EXCLUDED_ANIMALS = {"19303"}


def drop_excluded_animals(idx: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove EXCLUDED_ANIMALS from a session index and report it."""
    bad = idx["animal"].astype(str).isin(EXCLUDED_ANIMALS)
    if verbose and bad.any():
        print(f"[exclusion] dropping {int(bad.sum())} sessions of animals "
              f"{sorted(EXCLUDED_ANIMALS)} (data quality, see neuromodulator_trials.py)")
    return idx[~bad]


def build_trial_table(beh_path, animal=None, session=None) -> pd.DataFrame:
    """One row per trial for one session log."""
    beh_path = Path(beh_path)
    log = parse_logfile(beh_path)
    phase = detect_phase(log["scenario"])
    # value_getSessionData's missing-cue fix warns once per session; capture it
    # and keep the count instead of flooding the console
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sd, td = get_session_data(log, phase)
    n_est = 0
    for w in caught:
        m = re.search(r"rule=(\d+), cue=(\d+)", str(w.message))
        if m and "missing-cue" in str(w.message):
            n_est = int(m.group(1)) - int(m.group(2))
    trials = get_trial_masks(td)
    st = get_trial_stats_more(get_trial_stats(trials, sd["nRules"]))

    n = len(st["c"])
    # expand block-level fields to trials
    block_idx = np.repeat(np.arange(len(st["blockLength"])), st["blockLength"].astype(int))
    pos = np.concatenate([np.arange(int(L)) for L in st["blockLength"]])
    ttc = st["blockTrialtoCrit"][block_idx]
    lrand = st["blockTrialRandomAdded"][block_idx]
    tau = pos - ttc
    last = block_idx == block_idx.max()

    df = pd.DataFrame({
        "animal": animal if animal is not None else sd["subject"],
        "session_file": session if session is not None else beh_path.stem,
        "phase": phase,
        "trial_idx": np.arange(n),
        "choice": st["c"],
        "rewarded": st["r"],
        "responded": np.isfinite(st["c"]),
        "rt": np.asarray(td["rt"], float)[:n],
        "cue_time_log": np.asarray(td["cueTimes"], float)[:n],
        "rule": st["rule"],
        "hr_side": st["hr_side"],
        "block_idx": block_idx,
        "pos_in_block": pos,
        "block_trial_to_crit": ttc,
        "block_trial_random_added": lrand,
        "is_last_block": last,
        "tau": tau,
    })
    df["chose_better"] = np.where(df["responded"], (df["choice"] == df["hr_side"]).astype(float), np.nan)
    df["in_lrandom_window"] = ((df["tau"] >= 0) & (df["tau"] < df["block_trial_random_added"])
                               & ~df["is_last_block"])
    # number of cue times estimated by the missing-cue fix (not logged by Presentation)
    df.attrs["n_cues_estimated"] = n_est
    return df


def session_summary(df: pd.DataFrame) -> dict:
    """Session-level stats used to compare the imaging cohort with the
    behavioral training dataset (same task => similar numbers expected)."""
    blocks = df[~df["is_last_block"]].groupby("block_idx").first()
    return {
        "n_trials": len(df),
        "n_responded": int(df["responded"].sum()),
        "miss_rate": float(1 - df["responded"].mean()),
        "reward_rate": float(df.loc[df["responded"], "rewarded"].mean()),
        "p_better": float(df["chose_better"].mean()),
        "n_switches": int(blocks["block_trial_to_crit"].notna().sum()),
        "median_trials_to_crit": float(blocks["block_trial_to_crit"].median()),
        "median_lrandom": float(blocks["block_trial_random_added"].median()),
        "median_rt": float(df["rt"].median()),
        "n_cues_estimated": int(df.attrs.get("n_cues_estimated", 0)),
    }


# ---------------------------------------------------------------------------
# Trial <-> imaging-stamp matching
# ---------------------------------------------------------------------------
# The original pipeline (creatDffMatFiles_miniscope.m and its port) assumed
# that trial stamp k (IO1 channel, imaging clock) belongs to log trial k. The
# verification showed that in ~18% of sessions the first 1-3 trials happened
# before imaging started, so stamp k belongs to trial k+1 (or k+2, k+3), and in
# some sessions imaging stopped early. This function matches each LOG TRIAL to
# its stamp explicitly:
#   1. best integer lag from inter-trial intervals (offset-free);
#   2. linear fit stamp = a + b * cueTime over the lag-matched pairs (absorbs
#      the clock offset and the small PC/Inscopix drift);
#   3. each log trial gets the nearest stamp within `tol` s of its predicted
#      time, one-to-one; trials without a stamp get NaN.
# Session-level quality is returned so bad pairings (wrong file) can be excluded.

def match_stamps_to_trials(stamps, cue_times_log, max_lag: int = 5, tol: float = 0.25,
                           max_median_err: float = 0.05) -> dict:
    stamps = np.asarray(stamps, float)
    cues = np.asarray(cue_times_log, float)
    a_int, b_int = np.diff(stamps), np.diff(cues)
    best = None
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a_int[lag:], b_int) if lag >= 0 else (a_int, b_int[-lag:])
        n = min(len(aa), len(bb))
        if n < 20:
            continue
        err = np.median(np.abs(aa[:n] - bb[:n]))
        if best is None or err < best[0]:
            best = (err, lag)
    out = {"lag": np.nan, "iti_median_err": np.nan, "fit_resid_ms": np.nan,
           "n_matched": 0, "frac_trials_matched": 0.0, "ok": False,
           "cue_time_imaging": np.full(len(cues), np.nan)}
    if best is None:
        return out
    err, lag = best
    # lag-matched pairs: lag >= 0 -> stamp k+lag <-> trial k ; lag < 0 -> stamp k <-> trial k-lag
    k = np.arange(min(len(stamps) - max(lag, 0), len(cues) - max(-lag, 0)))
    s_idx, t_idx = k + max(lag, 0), k + max(-lag, 0)
    # keep only pairs whose intervals on BOTH sides agree (< 20 ms): a stamp
    # lost mid-session shifts every later pair, and those must not enter the fit
    d = np.abs(np.diff(stamps[s_idx]) - np.diff(cues[t_idx])) < 0.02
    good = np.zeros(len(k), bool)
    good[1:-1] = d[:-1] & d[1:]
    if good.sum() < 10:
        return out
    slope, icpt = np.polyfit(cues[t_idx][good], stamps[s_idx][good], 1)

    def assign(pred):
        """One-to-one nearest-stamp assignment within tol."""
        cue_img = np.full(len(cues), np.nan)
        used = np.zeros(len(stamps), bool)
        for i in np.argsort(cues):
            j = int(np.argmin(np.abs(stamps - pred[i])))
            if abs(stamps[j] - pred[i]) <= tol and not used[j]:
                cue_img[i], used[j] = stamps[j], True
        return cue_img

    cue_img = assign(icpt + slope * cues)
    m = np.isfinite(cue_img)
    if m.sum() >= 10:                                   # refine with all matched trials
        slope, icpt = np.polyfit(cues[m], cue_img[m], 1)
        cue_img = assign(icpt + slope * cues)
    pred = icpt + slope * cues
    m = np.isfinite(cue_img)
    resid = np.abs(cue_img[m] - pred[m])
    out.update(lag=int(lag), iti_median_err=float(err), clock_ppm=float((slope - 1) * 1e6),
               fit_resid_ms=float(1e3 * np.median(resid)) if m.any() else np.nan,
               n_matched=int(m.sum()), frac_trials_matched=float(m.mean()),
               ok=bool(err <= max_median_err and m.sum() >= 20),
               cue_time_imaging=cue_img)
    return out
