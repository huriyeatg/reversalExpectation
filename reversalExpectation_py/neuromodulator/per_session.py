"""
per_session.py
==============
Port of the analysis section of bandit_neuromodulatorPerSession.m
(H Atilgan & AC Kwan, 200210).

Computes trial-type-averaged dF/F (PSTH) for each session.
No plotting — returns data structures only.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from preprocessing.log_parser  import parse_logfile, get_session_data, detect_phase
from behavior.trial_processing import get_trial_masks, get_trial_stats
from behavior.trial_stats_more import get_trial_stats_more
from .plot_snake import plot_snake


FS       = 20
T_WINDOW = 120    # 6 s × 20 Hz


def per_session_neuromodulator(data_index: pd.DataFrame,
                                save_path: str = None) -> list:
    """
    Run per-session dF/F analysis for all sessions in data_index.

    Parameters
    ----------
    data_index : output of add_index_neuromodulator() / create_dff_files()
    save_path  : root directory for per-session figures.
                 Sub-folder per session is created automatically.
                 None → no figures saved.

    Returns
    -------
    List of dicts, one per session, with keys:
        session_file : str
        animal       : str
        t            : (T_WINDOW,) time axis (s, relative to cue onset)
        psth         : dict mapping trial-type label → mean dF/F (T_WINDOW,)
        psth_sem     : dict mapping trial-type label → SEM dF/F (T_WINDOW,)
        n_trials     : dict mapping trial-type label → n trials used
    """
    import matplotlib.pyplot as plt
    results = []

    for _, row in data_index.iterrows():
        log_path = Path(row["beh_path"])
        dff_path = log_path.parent / f"{log_path.stem}_dff.npz"

        if not dff_path.exists():
            continue

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
        stats  = get_trial_stats(trials, n_rules=n_rules)

        npz  = np.load(dff_path)
        dff  = npz["dff"]
        dffN = npz["dffN"]

        # Align trial counts
        n = min(len(trial_data["cue"]), len(dff))
        dff  = dff[:n, :T_WINDOW]
        dffN = dffN[:n, :T_WINDOW]

        # Trim boolean masks
        bool_masks = {k: np.asarray(v, dtype=bool)[:n]
                      for k, v in trials.items() if np.asarray(v).ndim == 1}

        # Trial type conditions: (label, mask_fields to AND together)
        conditions = {
            "HR reward":    _combine(bool_masks, ["left", "reward", "L70R10"],
                                     bool_masks, ["right", "reward", "L10R70"]),
            "HR no-reward": _combine(bool_masks, ["left", "noreward", "L70R10"],
                                     bool_masks, ["right", "noreward", "L10R70"]),
            "LR reward":    _combine(bool_masks, ["left", "reward", "L10R70"],
                                     bool_masks, ["right", "reward", "L70R10"]),
            "LR no-reward": _combine(bool_masks, ["left", "noreward", "L10R70"],
                                     bool_masks, ["right", "noreward", "L70R10"]),
        }

        t = np.arange(-2, 4, 1.0 / FS)[:T_WINDOW]

        psth, psth_sem, n_trials = {}, {}, {}
        for label, mask in conditions.items():
            subset = dff[mask, :]
            psth[label]     = np.nanmean(subset, axis=0) if len(subset) else np.full(T_WINDOW, np.nan)
            psth_sem[label] = (np.nanstd(subset, axis=0) / np.sqrt(np.sum(mask))
                               if np.sum(mask) > 0 else np.full(T_WINDOW, np.nan))
            n_trials[label] = int(np.sum(mask))

        result = {
            "session_file": row["session_file"],
            "animal":       row["animal"],
            "t":            t,
            "psth":         psth,
            "psth_sem":     psth_sem,
            "n_trials":     n_trials,
        }
        results.append(result)

        # ---- Snake-plot figure: one subplot per trial type ----
        if save_path:
            _plot_session_snake(result, trials, bool_masks, dff, t,
                                save_path=save_path)

    return results


def _plot_session_snake(result: dict, trials: dict, bool_masks: dict,
                        dff: np.ndarray, t: np.ndarray,
                        save_path: str) -> None:
    """
    Port of the snake-plot section in bandit_neuromodulatorPerSession.m.
    Creates a 4-panel figure (one per trial type) and saves it.
    """
    import matplotlib.pyplot as plt

    tlabel = result["session_file"]
    n      = len(dff)

    conditions = [
        ("HR reward",    ["left","reward","L70R10"], ["right","reward","L10R70"]),
        ("HR no-reward", ["left","noreward","L70R10"], ["right","noreward","L10R70"]),
        ("LR reward",    ["left","reward","L10R70"], ["right","reward","L70R10"]),
        ("LR no-reward", ["left","noreward","L10R70"], ["right","noreward","L70R10"]),
    ]

    def _and(fields):
        out = np.ones(n, dtype=bool)
        for f in fields:
            if f in bool_masks:
                out &= bool_masks[f]
        return out

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (lbl, f1, f2) in zip(axes, conditions):
        mask   = _and(f1) | _and(f2)
        subset = dff[mask, :] if np.any(mask) else np.full((1, len(t)), np.nan)
        plot_snake(subset, t, label=lbl, ax=ax)

    fig.suptitle(tlabel, fontsize=10)
    fig.tight_layout()

    from pathlib import Path
    out = Path(save_path) / f"{Path(tlabel).stem}_neuralSignal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


def _combine(masks, fields1, masks2, fields2):
    """AND two sets of trial masks then OR the results (mirrors getMask + OR)."""
    def _and(m, fields):
        result = np.ones(len(next(iter(m.values()))), dtype=bool)
        for f in fields:
            if f in m:
                result &= m[f]
        return result
    return _and(masks, fields1) | _and(masks2, fields2)
