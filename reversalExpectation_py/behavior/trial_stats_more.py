"""
trial_stats_more.py
===================
Faithful translation of value_getTrialStatsMore.m (H Atilgan & AC Kwan 191202).

Extends the stats dict produced by get_trial_stats() with block-level fields.

Faithfulness notes (what the .m does, and where this port deliberately differs)
------------------------------------------------------------------------------
Same as the .m:
  - hr_side is computed PER TRIAL from rewardprob (NaN where rule is NaN).
  - Blocks come from run-length encoding of `rule` with `~=`. Because
    NaN ~= NaN is true in MATLAB (and in numpy), every NaN gap trial of a
    merged session is its own 1-trial block. A block followed by a NaN block
    has no valid transition, so it gets no block-level stats (it is excluded),
    exactly as in the .m.
  - Block-level stats are computed only for blocks 1..nBlocks-1: the LAST block
    of a session never switched, so its L_Random is right-censored and the .m
    leaves it undefined. (The previous port computed ttc / L_Random for the
    last block too; any analysis that did not drop it treated a censored
    L_Random as a real switch -- e.g. belief_vhr.empirical_hazard.)
  - Trials-to-criterion = trial of the 10th CUMULATIVE choice of the block's
    high-reward side; L_Random = blockLength - trialsToCrit.
  - pWinStay / pLooseSwitch use only the last 6 trials of the block (5 pairs),
    with "stay" = same better/worse status on consecutive trials.
  - hitrates / rewardrates are PERCENT, over ALL trials of the block (misses
    count in the denominator) AND the first trial of the next block
    (the .m indexes blockStart:blockEnd+1 -- an off-by-one in the original,
    replicated here for fidelity).
  - blockPreSwitch*ChoiceRate: last 5 trials; *AtSwitch: the last trial
    (a miss counts as neither better nor worse -> 0).
  - ruletransList = all possible (i, j), i != j, in the .m order.

Deliberate differences:
  - Block-level arrays have length nBlocks (last entry NaN) instead of
    nBlocks-1, so callers can index them by block (master_behavior does).
  - blockTrans stores the NEXT RULE NUMBER, not the index into ruletransList
    (choice_switch.py builds its (from, to) lookup from blockRule + blockTrans).
  - A block that never reaches criterion gets NaN for trials-to-criterion and
    L_Random (the .m uses inf and -inf).
"""

import warnings

import numpy as np

N_CRIT = 10            # criterion: 10 cumulative choices of the high-reward side
N_PRESWITCH = 5        # trials before the switch for *ChoiceRate
N_WSLS = 5             # trial pairs before the switch for pWinStay / pLooseSwitch


def _eq(a, b):
    """MATLAB-style == on floats: NaN == anything is False."""
    return np.asarray(a, float) == b


def get_trial_stats_more(stats: dict) -> dict:
    """
    Translation of value_getTrialStatsMore.m.

    stats : output of get_trial_stats() with c, r, rule, rewardprob, rule_labels.
    Returns the same dict with the block-level fields added (see module docstring).
    """
    c = np.asarray(stats["c"], float)
    r = np.asarray(stats["r"], float)
    rule = np.asarray(stats["rule"], float)
    rewardprob = np.asarray(stats["rewardprob"], float)
    n_trials = len(c)
    n_rules = len(stats["rule_labels"])

    # ------------------------------------------------------------------
    # hr_side per trial (.m: from rewardprob; NaN where rewardprob is NaN)
    # ------------------------------------------------------------------
    hr_side = np.full(n_trials, np.nan)
    hr_side[rewardprob[:, 0] > rewardprob[:, 1]] = -1.0
    hr_side[rewardprob[:, 0] < rewardprob[:, 1]] = 1.0

    # ------------------------------------------------------------------
    # Run-length encoding: x0(1:end-1) ~= x0(2:end)  (NaN ~= NaN is True)
    # ------------------------------------------------------------------
    change = np.where(rule[:-1] != rule[1:])[0] + 1          # 0-based starts of new blocks
    ends = np.concatenate([change, [n_trials]])               # exclusive ends
    starts = np.concatenate([[0], change])
    block_length = (ends - starts).astype(float)
    block_rule = rule[ends - 1]                                # rule at the end of each block
    n_blocks = len(block_length)

    # all possible transitions i != j, in the .m order
    rule_trans_list = np.array([[i, j] for i in range(1, n_rules + 1)
                                for j in range(1, n_rules + 1) if i != j], dtype=float)
    valid_trans = {(a, b) for a, b in rule_trans_list}

    def nan_block():
        return np.full(n_blocks, np.nan)

    block_trans = nan_block()
    block_ttc = nan_block()
    block_random = nan_block()
    pre_better_rate = nan_block()
    pre_worse_rate = nan_block()
    pre_better_at = nan_block()
    pre_worse_at = nan_block()
    p_win_stay = nan_block()
    p_lose_switch = nan_block()
    hitrates = nan_block()
    rewardrates = nan_block()

    for i in range(n_blocks - 1):                              # .m: i = 1:numel(blockLength)-1
        if (block_rule[i], block_rule[i + 1]) not in valid_trans:
            continue                                           # e.g. transition into a NaN gap
        s = int(starts[i])                                     # first trial of block (0-based)
        e = int(ends[i]) - 1                                   # last trial of block (0-based)
        hr = hr_side[s]
        block_trans[i] = block_rule[i + 1]

        # trials to criterion: 10th cumulative choice of the high-reward side
        hits = np.cumsum(_eq(c[s:e + 1], hr))
        th = np.where(hits == N_CRIT)[0]
        if len(th):
            block_ttc[i] = th[0] + 1                            # 1-based position in block
            block_random[i] = block_length[i] - block_ttc[i]

        # choice rates in the last 5 trials (.m: c(E-4:E); miss counts as 0)
        w = slice(max(e - (N_PRESWITCH - 1), 0), e + 1)
        pre_better_rate[i] = np.mean(_eq(c[w], hr))
        pre_worse_rate[i] = np.mean(_eq(c[w], -hr))
        # choice at the trial before the switch (.m: c(E))
        pre_better_at[i] = float(_eq(c[e], hr))
        pre_worse_at[i] = float(_eq(c[e], -hr))

        # win-stay / lose-switch on the last 6 trials (.m: r(E-5:E-1), diff(c(E-5:E)==hr))
        lo = max(e - N_WSLS, 0)
        rew = r[lo:e]
        stay = np.diff(_eq(c[lo:e + 1], hr).astype(int)) == 0
        n_win, n_lose = np.nansum(rew == 1), np.nansum(rew == 0)
        p_win_stay[i] = np.sum((rew == 1) & stay) / n_win if n_win else np.nan
        p_lose_switch[i] = np.sum((rew == 0) & ~stay) / n_lose if n_lose else np.nan

        # hit / reward rates in PERCENT over blockStart : blockEnd+1 (off-by-one of the .m)
        rng = slice(s, min(e + 2, n_trials))
        n_rng = (min(e + 2, n_trials)) - s
        hitrates[i] = 100.0 * np.sum(_eq(c[rng], hr)) / n_rng
        rewardrates[i] = 100.0 * np.nansum(r[rng]) / n_rng

    # consistency checks of the .m (it only prints)
    if np.any(block_ttc[np.isfinite(block_ttc)] < N_CRIT):
        warnings.warn("value_getTrialStatsMore: fewer than 10 trials to reach criterion?!")
    if np.any(block_random[np.isfinite(block_random)] < 0):
        warnings.warn("value_getTrialStatsMore: random number added for block length < 0?!")

    stats["hr_side"] = hr_side
    stats["blockLength"] = block_length
    stats["blockRule"] = block_rule
    stats["ruletransList"] = rule_trans_list
    stats["blockTrans"] = block_trans
    stats["blockTrialtoCrit"] = block_ttc
    stats["blockTrialRandomAdded"] = block_random
    stats["blockPreSwitchBetterChoiceRate"] = pre_better_rate
    stats["blockPreSwitchWorseChoiceRate"] = pre_worse_rate
    stats["blockPreSwitchBetterChoiceAtSwitch"] = pre_better_at
    stats["blockPreSwitchWorseChoiceAtSwitch"] = pre_worse_at
    stats["pWinStay"] = p_win_stay
    stats["pLooseSwitch"] = p_lose_switch
    stats["hitrates"] = hitrates
    stats["rewardrates"] = rewardrates
    return stats
