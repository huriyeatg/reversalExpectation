"""
choice_switch.py
================
Translations of:
    choice_switch_hrside.m          (AC Kwan 170518)
    choice_switch_hrside_random.m   (H Atilgan & AC Kwan 191206)
    choice_switch_stats_random.m    (H Atilgan & AC Kwan)

Analyze choice behavior around high-reward-side switches,
optionally stratified by block-length statistics.
"""

import numpy as np


def choice_switch_hrside(stats: dict, trials_back: int) -> dict:
    """
    Translation of choice_switch_hrside.m.

    Computes the fraction of trials on which the animal chose the
    high-reward side (probh), the low-reward side (probl), or missed
    (probneither), aligned to each block transition.

    n = 0  is the first trial of the NEW block (post-switch).
    n = -1 is the last trial of the OLD block (pre-switch).

    Parameters
    ----------
    stats        : dict  Output of get_trial_stats_more()
    trials_back  : int   Trials to include on each side of the switch

    Returns
    -------
    output : dict with keys
        n           : int array  [-trials_back … +trials_back]
        probh       : float array  P(choose hr side)
        probl       : float array  P(choose lr side)
        probneither : float array  P(miss)
        numtrans    : int  number of transitions included
    """
    win = 1 + 2 * trials_back
    choseh       = np.zeros(win)
    chosel       = np.zeros(win)
    choseneither = np.zeros(win)
    numtrans     = 0

    block_length = stats["blockLength"]
    block_trans  = stats["blockTrans"]
    c            = stats["c"]
    hr_side      = stats["hr_side"]
    n_trials     = len(c)

    for i in range(len(block_length) - 1):
        idx = block_trans[i]
        if np.isnan(idx):
            continue

        switch_trial = int(np.sum(block_length[:i + 1]))   # 0-indexed first trial of next block
        t1 = switch_trial - trials_back
        t2 = switch_trial + trials_back

        if t2 >= n_trials:
            continue
        if t1 < 0:
            continue

        numtrans += 1
        window_c       = c[t1: t2 + 1]
        hr_at_switch   = hr_side[t1]                       # hr side *before* switch

        choseh       += (window_c == hr_at_switch)
        chosel       += (window_c == -hr_at_switch)
        choseneither += np.isnan(window_c)

    if numtrans == 0:
        probh = probl = probneither = np.full(win, np.nan)
    else:
        probh       = choseh       / numtrans
        probl       = chosel       / numtrans
        probneither = choseneither / numtrans

    return {
        "n":           np.arange(-trials_back, trials_back + 1),
        "probh":       probh,
        "probl":       probl,
        "probneither": probneither,
        "numtrans":    numtrans,
    }


def choice_switch_hrside_random(
    stats: dict,
    trials_back: int,
    L1_ranges: np.ndarray,
    L2_ranges: np.ndarray,
) -> dict:
    """
    Translation of choice_switch_hrside_random.m.

    Like choice_switch_hrside() but stratifies transitions by the
    block statistics preceding the switch:
        L1  = blockTrialtoCrit  (criterion-reaching trials)
        L2  = blockTrialRandomAdded  (random-bonus trials)

    Parameters
    ----------
    stats       : dict         Output of get_trial_stats_more()
    trials_back : int          Trials on each side of the switch
    L1_ranges   : (R, 2) array Inclusive [min, max] L1 bands
    L2_ranges   : (R, 2) array Inclusive [min, max] L2 bands

    Returns
    -------
    output : dict with keys
        n           : int array  [-trials_back … +trials_back]
        probh       : (win, R) float array
        probl       : (win, R) float array
        probneither : (win, R) float array
        numtrans    : (R,) int array
        numRange    : int
        L1_ranges, L2_ranges
        L1          : float array  blockTrialtoCrit for all non-NaN transitions
        L2          : float array  blockTrialRandomAdded for all non-NaN transitions
    """
    L1_ranges = np.asarray(L1_ranges)
    L2_ranges = np.asarray(L2_ranges)
    if L1_ranges.shape[0] != L2_ranges.shape[0]:
        raise ValueError(
            "choice_switch_hrside_random: L1_ranges and L2_ranges must "
            "have the same number of rows."
        )

    num_range    = L1_ranges.shape[0]
    win          = 1 + 2 * trials_back
    choseh       = np.zeros((win, num_range))
    chosel       = np.zeros((win, num_range))
    choseneither = np.zeros((win, num_range))
    numtrans     = np.zeros(num_range, dtype=int)

    block_length  = stats["blockLength"]
    block_trans   = stats["blockTrans"]
    block_ttc     = stats["blockTrialtoCrit"]
    block_random  = stats["blockTrialRandomAdded"]
    c             = stats["c"]
    hr_side       = stats["hr_side"]
    n_trials      = len(c)

    for i in range(len(block_length) - 1):
        idx = block_trans[i]
        if np.isnan(idx):
            continue

        # Which L1/L2 range does this block fall into?
        in_range = (
            (block_ttc[i]    >= L1_ranges[:, 0]) & (block_ttc[i]    <= L1_ranges[:, 1]) &
            (block_random[i] >= L2_ranges[:, 0]) & (block_random[i] <= L2_ranges[:, 1])
        )
        if in_range.sum() != 1:
            continue   # must fall in exactly one range

        r_idx = int(np.where(in_range)[0][0])

        switch_trial = int(np.sum(block_length[:i + 1]))
        t1 = switch_trial - trials_back
        t2 = switch_trial + trials_back

        if t2 >= n_trials or t1 < 0:
            continue

        numtrans[r_idx] += 1
        window_c      = c[t1: t2 + 1]
        hr_at_switch  = hr_side[t1]

        choseh[:, r_idx]       += (window_c == hr_at_switch)
        chosel[:, r_idx]       += (window_c == -hr_at_switch)
        choseneither[:, r_idx] += np.isnan(window_c)

    probh       = np.full((win, num_range), np.nan)
    probl       = np.full((win, num_range), np.nan)
    probneither = np.full((win, num_range), np.nan)
    for j in range(num_range):
        if numtrans[j] > 0:
            probh[:, j]       = choseh[:, j]       / numtrans[j]
            probl[:, j]       = chosel[:, j]       / numtrans[j]
            probneither[:, j] = choseneither[:, j] / numtrans[j]

    valid = ~np.isnan(block_trans[:-1])

    return {
        "n":           np.arange(-trials_back, trials_back + 1),
        "probh":       probh,
        "probl":       probl,
        "probneither": probneither,
        "numtrans":    numtrans,
        "numRange":    num_range,
        "L1_ranges":   L1_ranges,
        "L2_ranges":   L2_ranges,
        "L1":          block_ttc[:-1][valid],
        "L2":          block_random[:-1][valid],
    }


def choice_lrandom_start(
    stats: dict,
    trials_back: int,
    L1_ranges: np.ndarray,
    L2_ranges: np.ndarray,
) -> dict:
    """
    Like choice_switch_hrside_random() but aligns to the START of the
    L_Random (random-bonus) period within each block, rather than to the
    block switch (end of L_Random).

    t=0  : first trial of the L_Random period (= trial L_C within the block).
    t < 0: criterion-period trials.
    t > 0: random-bonus trials.

    Blocks are stratified by L_C (L1) and L_R (L2) ranges, same as
    choice_switch_hrside_random().
    """
    L1_ranges = np.asarray(L1_ranges)
    L2_ranges = np.asarray(L2_ranges)
    if L1_ranges.shape[0] != L2_ranges.shape[0]:
        raise ValueError(
            "choice_lrandom_start: L1_ranges and L2_ranges must "
            "have the same number of rows."
        )

    num_range    = L1_ranges.shape[0]
    win          = 1 + 2 * trials_back
    choseh       = np.zeros((win, num_range))
    chosel       = np.zeros((win, num_range))
    choseneither = np.zeros((win, num_range))
    numtrans     = np.zeros(num_range, dtype=int)

    block_length = stats["blockLength"]
    block_ttc    = stats["blockTrialtoCrit"]
    block_random = stats["blockTrialRandomAdded"]
    c            = stats["c"]
    hr_side      = stats["hr_side"]
    n_trials     = len(c)

    for i in range(len(block_length)):
        if np.isnan(block_ttc[i]) or np.isnan(block_random[i]):
            continue

        in_range = (
            (block_ttc[i]    >= L1_ranges[:, 0]) & (block_ttc[i]    <= L1_ranges[:, 1]) &
            (block_random[i] >= L2_ranges[:, 0]) & (block_random[i] <= L2_ranges[:, 1])
        )
        if in_range.sum() != 1:
            continue
        r_idx = int(np.where(in_range)[0][0])

        block_start   = int(np.sum(block_length[:i]))
        lrandom_start = block_start + int(block_ttc[i])

        t1 = lrandom_start - trials_back
        t2 = lrandom_start + trials_back

        if t1 < 0 or t2 >= n_trials:
            continue

        numtrans[r_idx] += 1
        window_c    = c[t1: t2 + 1]
        hr_at_align = hr_side[block_start]

        choseh[:, r_idx]       += (window_c == hr_at_align)
        chosel[:, r_idx]       += (window_c == -hr_at_align)
        choseneither[:, r_idx] += np.isnan(window_c)

    probh       = np.full((win, num_range), np.nan)
    probl       = np.full((win, num_range), np.nan)
    probneither = np.full((win, num_range), np.nan)
    for j in range(num_range):
        if numtrans[j] > 0:
            probh[:, j]       = choseh[:, j]       / numtrans[j]
            probl[:, j]       = chosel[:, j]       / numtrans[j]
            probneither[:, j] = choseneither[:, j] / numtrans[j]

    return {
        "n":           np.arange(-trials_back, trials_back + 1),
        "probh":       probh,
        "probl":       probl,
        "probneither": probneither,
        "numtrans":    numtrans,
        "numRange":    num_range,
        "L1_ranges":   L1_ranges,
        "L2_ranges":   L2_ranges,
    }


def lrandom_normalized_figure(df, n_bins=40, output_dir="figs"):
    """
    P(better/worse) en tiempo de bloque normalizado:
      -1 = block start, 0 = L_Random start, 1 = switch.
    Criterio -> [-1,0], L_Random -> [0,1], para superponer los 4 grupos de
    L_Random pese a largos absolutos distintos. Referencia = lado bueno del
    bloque actual (hr_side[block_start]). Excluye never-crit (ttc NaN) y L_Random==0.
    """
    import numpy as np, matplotlib.pyplot as plt
    from pathlib import Path

    L2 = [(0,4),(5,9),(10,14),(15,30)]
    edges = np.linspace(-1, 1, n_bins+1); cen = (edges[:-1]+edges[1:])/2
    acc = {g: {k: np.zeros(n_bins) for k in "bwmn"} for g in range(4)}

    def tally(a, ph, ch, ref):
        b = int(np.searchsorted(edges, ph, side="right") - 1)
        if b < 0 or b >= n_bins: return
        a["n"][b] += 1
        if   np.isnan(ch): a["m"][b] += 1
        elif ch == ref:    a["b"][b] += 1
        else:              a["w"][b] += 1

    for _, g in df.groupby(["animal", "session_file"], sort=False):
        g = g.reset_index(drop=True)
        blk    = g.groupby("block_idx", sort=True)
        sizes  = blk.size().values.astype(int)
        ttc    = blk.first()["block_trial_to_crit"].values.astype(float)
        rnd    = blk.first()["block_trial_random_added"].values.astype(float)
        c, hr  = g["choice"].values.astype(float), g["hr_side"].values.astype(float)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        last   = len(sizes) - 1
        for i in range(len(sizes)):
            if np.isnan(ttc[i]) or np.isnan(rnd[i]) or i == last: continue
            L = int(rnd[i])
            if L < 1: continue
            gi = next((j for j,(lo,hi) in enumerate(L2) if lo <= L <= hi), None)
            if gi is None: continue
            bs, T = starts[i], int(ttc[i]); ls, be = bs+T, bs+sizes[i]
            ref = hr[bs]
            for k, t in enumerate(range(bs, ls)): tally(acc[gi], -1+(k+0.5)/T, c[t], ref)
            for k, t in enumerate(range(ls, be)): tally(acc[gi],   (k+0.5)/L, c[t], ref)

    oranges = ["#7a2e10","#c0531f","#e8821f","#f2b34d"]
    purples = ["#3b0a6b","#6a2fb0","#9a7bd0","#c3b3e6"]
    fig, ax = plt.subplots(figsize=(9,6))
    for gi,(lo,hi) in enumerate(L2):
        n  = acc[gi]["n"]; ok = n > 0
        pb = np.where(ok, acc[gi]["b"]/np.where(ok,n,1), np.nan)
        pw = np.where(ok, acc[gi]["w"]/np.where(ok,n,1), np.nan)
        ax.plot(cen, pb, "-o", ms=3, color=oranges[gi], label=f"better $L_R$ {lo}-{hi}")
        ax.plot(cen, pw, "-v", ms=3, color=purples[gi], label=f"worse $L_R$ {lo}-{hi}")
    ax.axvline(0, ls="--", color="k", lw=1); ax.axvline(1, ls="-", color="k", lw=1.2)
    ax.set_xlabel("Normalized block time  (-1 start, 0 $L_R$ start, 1 switch)")
    ax.set_ylabel("Fraction of trials"); ax.set_xlim(-1.02,1.02); ax.set_ylim(0,1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower left"); fig.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir)/"switches_lrandom_normalized.png", dpi=150, bbox_inches="tight")

def choice_switch_random(
    stats: dict,
    trials_back: int,
    L1_ranges: np.ndarray,
    L2_ranges: np.ndarray,
) -> dict:
    """
    Translation of choice_switch_random.m (H Atilgan & AC Kwan 191205).

    Like choice_switch_hrside_random() but tracks absolute lateral choices
    (left / right) rather than relative (better / worse), and splits by
    transition TYPE (which rule-pair switched).

    Parameters
    ----------
    stats       : dict         Output of get_trial_stats_more()
    trials_back : int          Trials on each side of the switch
    L1_ranges   : (R, 2) array Inclusive [min, max] L1 bands
    L2_ranges   : (R, 2) array Inclusive [min, max] L2 bands

    Returns
    -------
    output : dict with keys
        n           : int array  [-trials_back … +trials_back]
        probl       : (win, T, R) float  P(left)
        probr       : (win, T, R) float  P(right)
        probneither : (win, T, R) float  P(miss)
        numtransType : int   number of unique transition types T
        transType    : (T, 2) array  unique [from_rule, to_rule] pairs
        numRange     : int
        L1_ranges, L2_ranges
        L1, L2       : per-transition block stats
        rule_labels  : list[str]
    """
    L1_ranges = np.asarray(L1_ranges)
    L2_ranges = np.asarray(L2_ranges)
    if L1_ranges.shape[0] != L2_ranges.shape[0]:
        raise ValueError(
            "choice_switch_random: L1_ranges and L2_ranges must "
            "have the same number of rows."
        )

    rule_trans_list = stats["ruletransList"]
    num_trans_types = len(rule_trans_list)

    if num_trans_types == 0:
        print("No switch of reward probabilities in this session.")
        return {}

    num_range    = L1_ranges.shape[0]
    win          = 1 + 2 * trials_back
    chosel       = np.zeros((win, num_trans_types, num_range))
    choser       = np.zeros((win, num_trans_types, num_range))
    choseneither = np.zeros((win, num_trans_types, num_range))
    numtrans     = np.zeros((num_trans_types, num_range), dtype=int)

    # Pre-build lookup: (from_rule, to_rule) → type index
    trans_lookup = {
        (rule_trans_list[t, 0], rule_trans_list[t, 1]): t
        for t in range(num_trans_types)
    }

    block_length = stats["blockLength"]
    block_trans  = stats["blockTrans"]    # next rule number
    block_rule   = stats["blockRule"]
    block_ttc    = stats["blockTrialtoCrit"]
    block_random = stats["blockTrialRandomAdded"]
    c            = stats["c"]
    n_trials     = len(c)

    for i in range(len(block_length) - 1):
        next_rule = block_trans[i]
        if np.isnan(next_rule):
            continue

        type_idx = trans_lookup.get((block_rule[i], next_rule))
        if type_idx is None:
            continue

        in_range = (
            (block_ttc[i]    >= L1_ranges[:, 0]) & (block_ttc[i]    <= L1_ranges[:, 1]) &
            (block_random[i] >= L2_ranges[:, 0]) & (block_random[i] <= L2_ranges[:, 1])
        )
        if in_range.sum() != 1:
            continue
        r_idx = int(np.where(in_range)[0][0])

        switch_trial = int(np.sum(block_length[:i + 1]))
        t1 = switch_trial - trials_back
        t2 = switch_trial + trials_back

        if t2 >= n_trials or t1 < 0:
            continue

        numtrans[type_idx, r_idx] += 1
        window_c = c[t1: t2 + 1]
        chosel      [:, type_idx, r_idx] += (window_c == -1)
        choser      [:, type_idx, r_idx] += (window_c ==  1)
        choseneither[:, type_idx, r_idx] += np.isnan(window_c)

    probl       = np.full((win, num_trans_types, num_range), np.nan)
    probr       = np.full((win, num_trans_types, num_range), np.nan)
    probneither = np.full((win, num_trans_types, num_range), np.nan)
    for j in range(num_trans_types):
        for k in range(num_range):
            if numtrans[j, k] > 0:
                probl      [:, j, k] = chosel      [:, j, k] / numtrans[j, k]
                probr      [:, j, k] = choser      [:, j, k] / numtrans[j, k]
                probneither[:, j, k] = choseneither[:, j, k] / numtrans[j, k]

    valid = ~np.isnan(block_trans[:-1])
    return {
        "n":            np.arange(-trials_back, trials_back + 1),
        "probl":        probl,
        "probr":        probr,
        "probneither":  probneither,
        "numtransType": num_trans_types,
        "transType":    rule_trans_list,
        "numRange":     num_range,
        "L1_ranges":    L1_ranges,
        "L2_ranges":    L2_ranges,
        "L1":           block_ttc[:-1][valid],
        "L2":           block_random[:-1][valid],
        "rule_labels":  stats.get("rule_labels", []),
    }


def choice_switch_stats_random(
    stats: dict,
    trials_back: int,
    trials_forw: int,
    L1_ranges: np.ndarray,
    L2_ranges: np.ndarray,
) -> dict:
    """
    Translation of choice_switch_stats_random.m.

    Computes summary statistics (first-crossing-of-0.5 and slope) for
    choice curves around block transitions, stratified by L1/L2 ranges.

    Parameters
    ----------
    stats       : dict
    trials_back : int   Trials before switch (n < 0)
    trials_forw : int   Trials after switch  (n > 0)
    L1_ranges   : (R, 2) array
    L2_ranges   : (R, 2) array

    Returns
    -------
    output : dict with keys
        n     : int array  [-trials_back … +trials_forw]
        stath : (3, R)  [first-cross-idx, slope, intercept] for hr-side curve
        statl : (3, R)  same for lr-side curve
        statn : (3, R)  same for miss curve
    """
    L1_ranges = np.asarray(L1_ranges)
    L2_ranges = np.asarray(L2_ranges)
    if L1_ranges.shape[0] != L2_ranges.shape[0]:
        raise ValueError(
            "choice_switch_stats_random: L1_ranges and L2_ranges must "
            "have the same number of rows."
        )

    num_range    = L1_ranges.shape[0]
    win          = 1 + trials_back + trials_forw
    choseh       = np.zeros((win, num_range))
    chosel       = np.zeros((win, num_range))
    choseneither = np.zeros((win, num_range))
    numtrans     = np.zeros(num_range, dtype=int)

    n            = np.arange(-trials_back, trials_forw + 1)
    block_length = stats["blockLength"]
    block_trans  = stats["blockTrans"]
    block_ttc    = stats["blockTrialtoCrit"]
    block_random = stats["blockTrialRandomAdded"]
    c            = stats["c"]
    hr_side      = stats["hr_side"]
    n_trials     = len(c)

    for i in range(len(block_length) - 1):
        idx = block_trans[i]
        if np.isnan(idx):
            continue

        in_range = (
            (block_ttc[i]    >= L1_ranges[:, 0]) & (block_ttc[i]    <= L1_ranges[:, 1]) &
            (block_random[i] >= L2_ranges[:, 0]) & (block_random[i] <= L2_ranges[:, 1])
        )
        if in_range.sum() != 1:
            continue

        r_idx = int(np.where(in_range)[0][0])

        switch_trial = int(np.sum(block_length[:i + 1]))
        t1 = switch_trial - trials_back
        t2 = switch_trial + trials_forw

        if t2 >= n_trials or t1 < 0:
            continue

        numtrans[r_idx] += 1
        window_c     = c[t1: t2 + 1]
        hr_at_switch = hr_side[t1]

        choseh[:, r_idx]       += (window_c == hr_at_switch)
        chosel[:, r_idx]       += (window_c == -hr_at_switch)
        choseneither[:, r_idx] += np.isnan(window_c)

    stath = np.full((3, num_range), np.nan)
    statl = np.full((3, num_range), np.nan)
    statn = np.full((3, num_range), np.nan)
    post  = n > 0
    fit_mask = (n > 0) & (n <= 10)

    for j in range(num_range):
        if numtrans[j] == 0:
            continue
        probh = choseh[:, j] / numtrans[j]
        probl = chosel[:, j] / numtrans[j]
        probn = choseneither[:, j] / numtrans[j]

        # First crossing of 0.5 (1-indexed relative to switch)
        cross_h = np.where(probh[post] <= 0.5)[0]
        stath[0, j] = float(cross_h[0] + 1) if len(cross_h) else np.nan

        cross_l = np.where(probl[post] >= 0.5)[0]
        statl[0, j] = float(cross_l[0] + 1) if len(cross_l) else np.nan

        cross_n = np.where(probn[post] >= 0.5)[0]
        statn[0, j] = float(cross_n[0] + 1) if len(cross_n) else np.nan

        # Linear slope over first 10 post-switch trials
        if fit_mask.sum() >= 2:
            stath[1:, j] = np.polyfit(n[fit_mask], probh[fit_mask], 1)
            statl[1:, j] = np.polyfit(n[fit_mask], probl[fit_mask], 1)
            statn[1:, j] = np.polyfit(n[fit_mask], probn[fit_mask], 1)

    return {
        "n":     n,
        "stath": stath,
        "statl": statl,
        "statn": statn,
    }
