"""
trial_type_stats.py
===================
Translations of:
    get_lickrate_byTrialType.m  (AC Kwan 170518)
    get_val_byTrialType.m       (AC Kwan 170518)

Compute histograms of lick rates and scalar values
stratified by trial type.
"""

import numpy as np
from typing import List, Union

from .trial_processing import _bool_mask


TrialTypeSpec = Union[str, List[str]]


def get_lickrate_by_trial_type(
    trial_data: dict,
    trials: dict,
    trial_types: List[TrialTypeSpec],
    edges: np.ndarray,
) -> dict:
    """
    Translation of get_lickrate_byTrialType.m.

    Computes peri-event lick-rate histograms (in Hz) for left and right
    lick ports, separately for each trial type.

    Parameters
    ----------
    trial_data  : dict
        Must contain 'leftlickTimes' and 'rightlickTimes',
        each a list of 1-D arrays (one array per trial).
    trials      : dict
        Output of get_trial_masks() — boolean/float mask arrays.
    trial_types : list
        Each element is either a single string (e.g. 'reward') or a
        two-element list specifying a conjunction (e.g. ['left', 'reward']).
        Maximum conjunction depth: 2 (matches MATLAB).
    edges       : 1-D array
        Bin edges for the histogram (seconds relative to cue onset).

    Returns
    -------
    output : dict with keys
        trial_types  : list   (as supplied)
        trial_labels : list[str]
        edges        : 1-D array  (bin left edges, len = len(edges)-1)
        left_times   : list of 1-D arrays  (Hz per bin)
        right_times  : list of 1-D arrays  (Hz per bin)
    """
    edges     = np.asarray(edges, dtype=float)
    edge_width = float(np.nanmean(np.diff(edges)))
    n_bins     = len(edges) - 1

    left_times   = []
    right_times  = []
    trial_labels = []

    for tt in trial_types:
        # _bool_mask, not .astype(bool): masks from merged sessions carry NaN gap
        # rows, and NaN.astype(bool) is True -> every gap trial would be counted
        # as selected (inflating n_sel and deflating the lick rate in Hz).
        if isinstance(tt, str):
            mask  = _bool_mask(np.asarray(trials[tt]))
            label = tt
        elif len(tt) == 2:
            mask  = _bool_mask(np.asarray(trials[tt[0]])) & _bool_mask(np.asarray(trials[tt[1]]))
            label = f"{tt[0]} + {tt[1]}"
        else:
            raise ValueError(
                "get_lickrate_by_trial_type: conjunction of more than "
                "two trial types is not supported."
            )
        trial_labels.append(label)

        n_sel = int(mask.sum())

        def _hist(lick_list):
            # No selected trials -> rate undefined (NaN). Selected trials but no
            # licks on this side -> rate is 0 Hz, not NaN. (np.concatenate on an
            # empty list raised "need at least one array to concatenate".)
            if n_sel == 0:
                return np.full(n_bins, np.nan)
            arrays = [np.asarray(t, dtype=float).ravel()
                      for t in lick_list if t is not None]
            arrays = [a for a in arrays if a.size > 0]
            if not arrays:
                return np.zeros(n_bins)
            all_times = np.concatenate(arrays)
            all_times = all_times[np.isfinite(all_times)]   # NaN gap placeholders
            counts, _ = np.histogram(all_times, bins=edges)
            return counts.astype(float) / n_sel / edge_width  # Hz

        sel_indices = np.where(mask)[0]

        left_list  = [trial_data["leftlickTimes"][i]  for i in sel_indices]
        right_list = [trial_data["rightlickTimes"][i] for i in sel_indices]

        left_times.append(_hist(left_list))
        right_times.append(_hist(right_list))

    return {
        "trial_types":  trial_types,
        "trial_labels": trial_labels,
        "edges":        edges[:-1],
        "left_times":   left_times,
        "right_times":  right_times,
    }


def get_val_by_trial_type(
    val: np.ndarray,
    trials: dict,
    trial_types: List[str],
    edges: np.ndarray,
    val_label: str = "",
) -> dict:
    """
    Translation of get_val_byTrialType.m.

    Computes histograms and medians of a scalar per-trial value
    (e.g., reaction time, ITI) for each trial type.

    Parameters
    ----------
    val         : 1-D float array  (one value per trial)
    trials      : dict             Output of get_trial_masks()
    trial_types : list[str]        Single mask keys (no conjunction)
    edges       : 1-D array        Histogram bin edges
    val_label   : str              Human-readable axis label

    Returns
    -------
    output : dict with keys
        trial_types : list[str]
        edges       : 1-D array  (left edges, len = len(edges)-1)
        val         : list of 1-D int arrays  (histogram counts)
        val_median  : list of float  (median per trial type)
        val_label   : str
    """
    edges     = np.asarray(edges, dtype=float)
    val       = np.asarray(val,   dtype=float)
    n_bins    = len(edges) - 1

    val_hist   = []
    val_median = []

    for tt in trial_types:
        mask = trials[tt].astype(float)
        mask[np.isnan(mask)] = 0
        mask = mask.astype(bool)

        v_sel = val[mask]

        if len(v_sel) == 0:
            val_hist.append(np.zeros(n_bins, dtype=int))
            val_median.append(np.nan)
        else:
            counts, _ = np.histogram(v_sel, bins=edges)
            val_hist.append(counts)
            val_median.append(float(np.nanmedian(v_sel)))

    return {
        "trial_types": trial_types,
        "edges":       edges[:-1],
        "val":         val_hist,
        "val_median":  val_median,
        "val_label":   val_label,
    }
