"""
trial_stats_more.py
===================
Exact translation of value_getTrialStatsMore.m
(H Atilgan & AC Kwan).

Extends the stats dict produced by get_trial_stats() with block-level fields.
"""

import numpy as np


def get_trial_stats_more(stats: dict) -> dict:
    """
    Translation of value_getTrialStatsMore.m.

    Parameters
    ----------
    stats : dict
        Output of get_trial_stats(). Must contain:
            c           : float array  (-1=left, 1=right, NaN=miss)
            r           : float array  (1=reward, 0=noreward, NaN=miss)
            rule        : float array  (rule index, NaN=miss)
            rewardprob  : (nTrials, 2) array  [left_prob, right_prob]
            rule_labels : list[str]

    Returns
    -------
    stats : same dict, with added fields:

    Block-level (length = nBlocks):
        blockLength
        blockRule
        blockTrans                           (NaN for last block)
        blockTrialtoCrit
        blockTrialRandomAdded
        blockPreSwitchBetterChoiceAtSwitch
        blockPreSwitchWorseChoiceAtSwitch
        rewardrates
        hitrates
        pWinStay
        pLooseSwitch

    Trial-level:
        hr_side   : float array  (-1 = left is high-reward side, 1 = right)

    Other:
        ruletransList : (nTrans, 2) float array of unique [from, to] rule pairs
    """
    c          = stats["c"]
    r          = stats["r"]
    rule       = stats["rule"]
    rewardprob = stats["rewardprob"]
    n_trials   = len(c)

    # ------------------------------------------------------------------
    # 1. Block boundaries  (mirrors MATLAB loop exactly)
    # ------------------------------------------------------------------
    block_start = [0]   # 0-indexed (MATLAB: 1-indexed blockStart=[1])

    for t in range(1, n_trials):
        prev_nan = np.isnan(rule[t - 1])
        curr_nan = np.isnan(rule[t])

        if not curr_nan and not prev_nan and rule[t] != rule[t - 1]:
            block_start.append(t)
        elif not curr_nan and prev_nan:
            block_start.append(t)
        # entering a NaN gap: do not mark yet

    block_start = sorted(set(block_start))
    n_blocks    = len(block_start)
    block_end   = block_start[1:] + [n_trials]   # exclusive end

    # ------------------------------------------------------------------
    # 2. Block-level fields
    # ------------------------------------------------------------------
    block_length             = np.full(n_blocks, np.nan)
    block_rule               = np.full(n_blocks, np.nan)
    block_trans              = np.full(n_blocks, np.nan)
    block_trial_to_crit      = np.full(n_blocks, np.nan)
    block_trial_random_added = np.full(n_blocks, np.nan)
    reward_rates             = np.full(n_blocks, np.nan)
    hit_rates                = np.full(n_blocks, np.nan)
    p_win_stay               = np.full(n_blocks, np.nan)
    p_lose_switch            = np.full(n_blocks, np.nan)
    block_pre_switch_better  = np.full(n_blocks, np.nan)
    block_pre_switch_worse   = np.full(n_blocks, np.nan)

    trans_list = []

    for b in range(n_blocks):
        t1 = block_start[b]
        t2 = block_end[b]
        idx = np.arange(t1, t2)

        rule_blk  = rule[idx]
        rule_vals = rule_blk[~np.isnan(rule_blk)]

        if len(rule_vals) == 0:
            continue

        block_length[b] = len(idx)

        # Rule = mode of non-NaN rule values in block
        block_rule[b] = float(
            np.bincount(rule_vals.astype(int) - 1).argmax() + 1
        )

        # High-reward side from first valid trial in block
        first_valid_abs = t1 + int(np.where(~np.isnan(rule_blk))[0][0])
        lp = rewardprob[first_valid_abs, 0]
        rp = rewardprob[first_valid_abs, 1]
        hr_side_b = -1.0 if lp > rp else 1.0

        # Transition to next block
        if b < n_blocks - 1:
            next_t1   = block_start[b + 1]
            next_t2   = block_end[b + 1]
            next_vals = rule[next_t1:next_t2]
            next_vals = next_vals[~np.isnan(next_vals)]
            if len(next_vals) > 0:
                next_rule = float(
                    np.bincount(next_vals.astype(int) - 1).argmax() + 1
                )
                block_trans[b] = next_rule
                trans_list.append([block_rule[b], next_rule])

        # Choices and outcomes within block
        c_blk  = c[idx]
        r_blk  = r[idx]
        valid  = ~np.isnan(c_blk)
        c_val  = c_blk[valid]
        r_val  = r_blk[valid]

        if len(c_val) == 0:
            continue

        # Reward rate
        reward_rates[b] = np.nanmean(r_blk)

        # Hit rate: fraction of non-miss trials on hr side
        hit_rates[b] = np.mean(c_val == hr_side_b)

        # Win-stay / lose-switch
        if len(c_val) > 1:
            win_idx = np.where(r_val[:-1] == 1)[0]
            if len(win_idx) > 0:
                p_win_stay[b] = np.mean(c_val[win_idx + 1] == c_val[win_idx])

            lose_idx = np.where(r_val[:-1] == 0)[0]
            if len(lose_idx) > 0:
                p_lose_switch[b] = np.mean(
                    c_val[lose_idx + 1] != c_val[lose_idx]
                )

        # Pre-switch choice = last non-NaN choice in block
        last_valid = np.where(~np.isnan(c_blk))[0]
        if len(last_valid) > 0:
            last_choice = c_blk[last_valid[-1]]
            block_pre_switch_better[b] = float(last_choice == hr_side_b)
            block_pre_switch_worse[b]  = float(last_choice != hr_side_b)

        # Trials-to-criterion = position (1-indexed) of 10th hr-side choice
        better_idx = np.where(c_blk == hr_side_b)[0]
        if len(better_idx) >= 10:
            block_trial_to_crit[b]      = float(better_idx[9] + 1)
            block_trial_random_added[b] = max(0.0, block_length[b] - block_trial_to_crit[b])
        else:
            # nunca llegó a 10 elecciones buenas → criterio/random indefinidos, excluir
            block_trial_to_crit[b]      = np.nan
            block_trial_random_added[b] = np.nan

    # ------------------------------------------------------------------
    # 3. Trial-level hr_side
    # ------------------------------------------------------------------
    hr_side = np.full(n_trials, np.nan)
    for b in range(n_blocks):
        t1       = block_start[b]
        t2       = block_end[b]
        rule_blk = rule[t1:t2]
        if np.all(np.isnan(rule_blk)):
            continue
        first_valid_abs = t1 + int(np.where(~np.isnan(rule_blk))[0][0])
        lp = rewardprob[first_valid_abs, 0]
        rp = rewardprob[first_valid_abs, 1]
        hr_side[t1:t2] = -1.0 if lp > rp else 1.0

    # ------------------------------------------------------------------
    # 4. ruletransList: unique [from, to] pairs
    # ------------------------------------------------------------------
    if trans_list:
        rule_trans_list = np.unique(
            np.array(trans_list, dtype=float), axis=0
        )
    else:
        rule_trans_list = np.zeros((0, 2), dtype=float)

    # ------------------------------------------------------------------
    # 5. Store — field names match MATLAB exactly
    # ------------------------------------------------------------------
    stats["blockLength"]                        = block_length
    stats["blockRule"]                          = block_rule
    stats["blockTrans"]                         = block_trans
    stats["blockTrialtoCrit"]                   = block_trial_to_crit
    stats["blockTrialRandomAdded"]              = block_trial_random_added
    stats["blockPreSwitchBetterChoiceAtSwitch"] = block_pre_switch_better
    stats["blockPreSwitchWorseChoiceAtSwitch"]  = block_pre_switch_worse
    stats["rewardrates"]                        = reward_rates
    stats["hitrates"]                           = hit_rates
    stats["pWinStay"]                           = p_win_stay
    stats["pLooseSwitch"]                       = p_lose_switch
    stats["hr_side"]                            = hr_side
    stats["ruletransList"]                      = rule_trans_list

    return stats
