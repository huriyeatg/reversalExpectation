"""
beh_performance.py
==================
Translation of beh_performance.m
(AC Kwan 191210).

Computes basic performance metrics for a single session.
"""

import numpy as np


def beh_performance(stats: dict) -> dict:
    """
    Translation of beh_performance.m.

    Parameters
    ----------
    stats : dict
        Output of get_trial_stats_more().

    Returns
    -------
    output : dict with keys:
        nTrial          : int   — total go trials (left + right)
        nLeftResp       : int   — left choices
        nRightResp      : int   — right choices
        nSwitch         : int   — number of block transitions
        rule_labels     : list[str]
        meanTrialtoCrit : float array (one per rule) — mean trials-to-criterion,
                          computed over all blocks *except* the last
        meanRewardRate  : float array (one per rule)
        meanHitRate     : float array (one per rule)
    """
    c           = stats["c"]
    block_rule  = stats["blockRule"]
    block_tri   = stats["blockTrialtoCrit"]
    rewardrates = stats["rewardrates"]
    hitrates    = stats["hitrates"]
    rule_labels = stats["rule_labels"]

    n_rules  = len(rule_labels)
    # exclude last block (matches MATLAB: blockRule(1:end-1))
    blk_mask = np.arange(len(block_rule) - 1)

    mean_ttc  = np.full(n_rules, np.nan)
    mean_rr   = np.full(n_rules, np.nan)
    mean_hr   = np.full(n_rules, np.nan)

    for k in range(1, n_rules + 1):
        sel = blk_mask[block_rule[blk_mask] == k]
        if len(sel) > 0:
            mean_ttc[k - 1] = np.nanmean(block_tri[sel])
            mean_rr[k - 1]  = np.nanmean(rewardrates[sel])
            mean_hr[k - 1]  = np.nanmean(hitrates[sel])

    return {
        "nTrial":          int(np.sum((c == -1) | (c == 1))),
        "nLeftResp":       int(np.sum(c == -1)),
        "nRightResp":      int(np.sum(c == 1)),
        "nSwitch":         int(len(block_rule) - 1),
        "rule_labels":     rule_labels,
        "meanTrialtoCrit": mean_ttc,
        "meanRewardRate":  mean_rr,
        "meanHitRate":     mean_hr,
    }
