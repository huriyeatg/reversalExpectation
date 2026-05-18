"""
per_animal.py
=============
Port of bandit_neuromodulatorPerAnimal.m (H Atilgan & AC Kwan, 200210).

Groups sessions by animal, merges them chronologically, and returns
the merged behavioral + neural data for each animal.
No plotting — returns data structures only.
"""

import pandas as pd

from behavior.trial_processing import get_trial_stats
from behavior.trial_stats_more import get_trial_stats_more
from .merge_sessions                    import merge_sessions_neuromodulator
from .plot_session_neuromodulator       import plot_session_neuromodulator


def per_animal_neuromodulator(data_index: pd.DataFrame,
                               save_path: str = None) -> list:
    """
    Parameters
    ----------
    data_index : filtered output of add_index_neuromodulator() /
                 create_dff_files() (e.g. phase-31 sessions only)
    save_path  : root directory for per-animal figures.
                 Sub-folder per animal is created automatically.
                 None → no figures saved.

    Returns
    -------
    List of dicts, one per animal, with keys:
        animal     : str
        trial_data : merged trial_data dict
        trials     : merged trials dict (includes dff, dffN, dffN_zscore)
        stats      : output of get_trial_stats_more()
        n_rules    : int
    """
    import matplotlib.pyplot as plt

    results = []

    for animal, group in data_index.groupby("animal"):
        group_sorted = group.sort_values("date_number")

        merged = merge_sessions_neuromodulator(group_sorted)
        if merged["trial_data"] is None or not merged["trial_data"]:
            continue

        stats = get_trial_stats(merged["trials"], n_rules=merged["n_rules"])
        stats = get_trial_stats_more(stats)

        result = {
            "animal":     animal,
            "trial_data": merged["trial_data"],
            "trials":     merged["trials"],
            "stats":      stats,
            "n_rules":    merged["n_rules"],
        }
        results.append(result)

        # ---- Plot session overview + snake (mirrors plot_session_neuromodulator.m) ----
        animal_save = None
        if save_path:
            from pathlib import Path
            animal_save = str(Path(save_path) / animal)

        try:
            figs = plot_session_neuromodulator(
                stats=stats,
                trials=merged["trials"],
                tlabel=animal,
                save_path=animal_save,
            )
            plt.close("all")
        except Exception as e:
            import warnings
            warnings.warn(f"Plotting failed for {animal}: {e}")

    return results
