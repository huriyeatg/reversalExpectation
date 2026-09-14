"""
glmhmm_occupancy.py
===================
State-occupancy views of a fitted GLM-HMM: where in time each latent state
tends to occur. Two complementary alignments, both driven by the per-trial hard
(MAP) state in df['glmhmm_state'] (as produced by run_glmhmm.attach_glmhmm_states):

  (a) block-end aligned : P(state | trials before the block ends / switches).
      Answers "is some state more likely just before a contingency switch?"
  (b) within-session    : P(state | normalized position in the session, 0->1).
      Answers "is some state more likely at the start vs the end of a session?"

Each curve is P(state == s) at that time bin with a binomial 95% CI, one line
per state. Occupancies across states sum to 1 at every bin (hard assignment).

Caveat for (a): because blocks have different lengths, far-from-end bins are fed
only by long blocks (survivorship), so the composition there is not a like-for-
like sample -- read the bins near the end (small |x|), which every block reaches.
This is the same alignment confound discussed for the L_Random analyses; the plot
is exploratory, not an anticipation test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _stats
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- palette (matches the slide figures) -----------------------------------
CREAM = "#FBFAF6"; INK = "#1A2E2A"
STATE_COLORS = ["#2C5F2D", "#C9472B", "#E8A33D", "#3A6EA5", "#7B5EA7"]  # by state index
GRID = "#D8D5CC"
MUTE = "#6B6B66"

_BLOCK_KEYS = ["animal", "session_file", "block_idx"]
_SESS_KEYS = ["animal", "session_file"]


def _font():
    for name in ("Georgia", "DejaVu Serif", "Times New Roman", "serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "serif"


def _binom_ci(k, n, z=1.96):
    """Normal-approx binomial CI; returns (p, lo, hi) arrays, NaN where n==0."""
    n = np.asarray(n, float); k = np.asarray(k, float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    se = np.sqrt(np.clip(p * (1 - p), 0, None) / np.where(n > 0, n, np.nan))
    return p, np.clip(p - z * se, 0, 1), np.clip(p + z * se, 0, 1)


def _mean_sem_by_animal(d, group_col, value_col, animal_col="animal",
                        min_n=1, min_animals=5):
    """Aggregate a per-trial value the way Murphy et al. do: average WITHIN each
    animal per group, then take the mean and SEM ACROSS animals (the animal is
    the unit). Returns a DataFrame [group_col, p, lo, hi, n, n_animals] where the
    band lo/hi = mean -/+ 1 SEM. A group is dropped if it has fewer than min_n
    trials or fewer than min_animals contributing animals."""
    out = []
    for gv, g in d.groupby(group_col):
        n = len(g)
        if n < min_n:
            continue
        per = g.groupby(animal_col)[value_col].mean().to_numpy()
        na = int(len(per))
        if na < min_animals:
            continue
        m = float(per.mean())
        sem = float(per.std(ddof=1) / np.sqrt(na)) if na > 1 else 0.0
        out.append({group_col: gv, "p": m, "lo": m - sem, "hi": m + sem,
                    "n": int(n), "n_animals": na})
    return pd.DataFrame(out)


# ===========================================================================
# Position bookkeeping
# ===========================================================================
def add_positions(df):
    """Add block- and session-relative position columns (does not mutate input).

    block_size            : trials in the (animal, session, block) group
    pos_in_block          : 0-based index within the block
    k_from_block_end      : trials remaining until the block's last trial (0 = last)
    block_complete        : block_size == block_trial_to_crit + block_trial_random_added
                            (a block that actually ran to its designed switch)
    is_last_block         : the final block_idx of the session (may be truncated)
    pos_in_session_norm   : trial position within the session scaled to [0, 1]
    """
    df = df.copy()
    gb = df.groupby(_BLOCK_KEYS, sort=False)
    df["block_size"] = gb["block_idx"].transform("size")
    df["pos_in_block"] = gb.cumcount()
    df["k_from_block_end"] = df["block_size"] - 1 - df["pos_in_block"]

    if {"block_trial_to_crit", "block_trial_random_added"} <= set(df.columns):
        exp = (gb["block_trial_to_crit"].transform("first")
               + gb["block_trial_random_added"].transform("first"))
        df["block_complete"] = df["block_size"].eq(exp)
    else:
        df["block_complete"] = True

    last_blk = df.groupby(_SESS_KEYS, sort=False)["block_idx"].transform("max")
    df["is_last_block"] = df["block_idx"].eq(last_blk)

    gs = df.groupby(_SESS_KEYS, sort=False)
    pos = gs.cumcount()
    n = gs["block_idx"].transform("size")
    df["pos_in_session_norm"] = np.where(n > 1, pos / (n - 1), 0.5)
    return df


# ===========================================================================
# Occupancy tables
# ===========================================================================
def occupancy_block_aligned(df, state_col="glmhmm_state", states=(0, 1, 2),
                            k_max=40, min_n=50, complete_only=True,
                            drop_last_block=True, min_animals=5):
    """P(state | trials before block end), as mean +/- SEM across animals.
    Returns long DataFrame (k, trials_before_end, state, p, lo, hi, n,
    n_animals). k=0 is the last trial of the block (the switch). complete_only /
    drop_last_block remove blocks that did not actually run to a switch."""
    d = df
    if complete_only and "block_complete" in d:
        d = d[d["block_complete"]]
    if drop_last_block and "is_last_block" in d:
        d = d[~d["is_last_block"]]
    d = d[d[state_col].notna() & (d["k_from_block_end"] <= k_max)].copy()
    rows = []
    for st in states:
        d["_in"] = (d[state_col].to_numpy() == st).astype(float)
        agg = _mean_sem_by_animal(d, "k_from_block_end", "_in",
                                  min_n=min_n, min_animals=min_animals)
        for _, r in agg.iterrows():
            rows.append({"k": r["k_from_block_end"],
                         "trials_before_end": -r["k_from_block_end"], "state": st,
                         "p": r["p"], "lo": r["lo"], "hi": r["hi"],
                         "n": int(r["n"]), "n_animals": int(r["n_animals"])})
    return pd.DataFrame(rows)


def occupancy_session(df, state_col="glmhmm_state", states=(0, 1, 2),
                      n_bins=20, min_n=50, min_animals=5):
    """P(state | normalized within-session position), as mean +/- SEM across
    animals. Returns long DataFrame (bin_center, state, p, lo, hi, n,
    n_animals)."""
    d = df[df[state_col].notna()].copy()
    pos = d["pos_in_session_norm"].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    d["_bin"] = idx
    rows = []
    for st in states:
        d["_in"] = (d[state_col].to_numpy() == st).astype(float)
        agg = _mean_sem_by_animal(d, "_bin", "_in", min_n=min_n,
                                  min_animals=min_animals)
        for _, r in agg.iterrows():
            rows.append({"bin_center": float(centers[int(r["_bin"])]), "state": st,
                         "p": r["p"], "lo": r["lo"], "hi": r["hi"],
                         "n": int(r["n"]), "n_animals": int(r["n_animals"])})
    return pd.DataFrame(rows)


# ===========================================================================
# Figure
# ===========================================================================
def _draw(ax, occ, xcol, states, state_labels, font, xlabel, title):
    for st in states:
        sub = occ[occ["state"] == st].sort_values(xcol)
        if sub.empty:
            continue
        c = STATE_COLORS[st % len(STATE_COLORS)]
        lab = state_labels[st] if state_labels and st < len(state_labels) else f"state {st}"
        ax.plot(sub[xcol], sub["p"], "-o", ms=3, lw=1.8, color=c, label=lab)
        ax.fill_between(sub[xcol], sub["lo"], sub["hi"], color=c, alpha=0.18, linewidth=0)
    ax.set_title(title, fontsize=11, fontfamily=font, color=INK, weight="bold")
    ax.set_xlabel(xlabel, fontsize=9.5, fontfamily=font)
    ax.set_ylabel("P(state)", fontsize=9.5, fontfamily=font)
    ax.set_ylim(0, 1)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_state_occupancy(df, state_col="glmhmm_state", states=None,
                         state_labels=None,
                         engaged_idx=None, k_max=40, n_bins=20, min_n=50,
                         complete_only=True, drop_last_block=True,
                         lrandom_only=True, lrandom_col="block_trial_random_added",
                         title=None, outfile="glmhmm_state_occupancy.png"):
    """Two-panel state-occupancy figure: (left) aligned to block end/switch,
    (right) across normalized session position. `df` must carry per-trial states
    in `state_col`; positions are added internally if missing.

    states : if None (default), inferred from the states actually present in the
        data, so a K=2 model draws two lines (not a phantom flat-zero third one).
    lrandom_only : if True (default), restrict BOTH panels to each block's
        L_Random phase (k_from_block_end < L_Random, i.e. tau >= 0). This removes
        the L_Criterion trials entirely, so the block-aligned panel shows only
        pre-switch random-window trials and never spans a block boundary (the
        only switch is at k=0). Set False to include the criterion phase.
    title  : if None, set to "GLM-HMM state occupancy (K=<n_states>)"."""
    if "k_from_block_end" not in df.columns or "pos_in_session_norm" not in df.columns:
        df = add_positions(df)
    if lrandom_only:
        if lrandom_col not in df.columns:
            raise KeyError(f"lrandom_only=True needs '{lrandom_col}'")
        df = df[df["k_from_block_end"] < df[lrandom_col]]     # keep L_Random phase only
    if states is None:
        states = tuple(sorted(int(s) for s in pd.unique(df[state_col].dropna())))
    occ_b = occupancy_block_aligned(df, state_col, states, k_max, min_n,
                                    complete_only, drop_last_block)
    occ_s = occupancy_session(df, state_col, states, n_bins, min_n)

    font = _font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.patch.set_facecolor(CREAM)
    for ax in (ax1, ax2):
        ax.set_facecolor(CREAM)

    xlabel_left = ("trials before switch  (0 = switch, L_Random only)"
                   if lrandom_only else "trials before block end  (0 = switch)")
    _draw(ax1, occ_b, "trials_before_end", states, state_labels, font,
          xlabel_left, "Aligned to block end")
    ax1.axvline(0, color=INK, lw=1.0, ls="--", alpha=0.7)
    ax1.legend(frameon=False, fontsize=9, loc="upper left", prop={"family": font})

    _draw(ax2, occ_s, "bin_center", states, state_labels, font,
          "position in session  (0 = start, 1 = end)", "Across the session")

    if title is None:
        suffix = "  ·  L_Random only" if lrandom_only else ""
        title = f"GLM-HMM state occupancy (K={len(states)}){suffix}"
    fig.suptitle(title, fontsize=13,
                 fontfamily=font, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


# ===========================================================================
# L_Random-stratified block-aligned occupancy
# ---------------------------------------------------------------------------
# The plain block-aligned panel pools blocks of every L_Random length, so the
# sample size (and block composition) changes with distance from the switch:
# near the switch every block contributes, but far back only the long blocks
# survive. Stratifying by L_Random (block_trial_random_added) makes each curve
# composition-matched -- blocks with similar random-window length are compared
# directly. L_Criterion is irrelevant here: near the switch (small k) every
# block is in its L_Random phase regardless of how long its criterion phase was.
# ===========================================================================
_LRANDOM_BINS = ((0, 3), (4, 7), (8, 14), (15, 10 ** 9))


def _lrand_label(lo, hi):
    return f"L_Rand {lo}\u2013{hi}" if hi < 10 ** 9 else f"L_Rand {lo}+"


def occupancy_block_aligned_by_lrandom(
        df, state_col="glmhmm_state", states=None, lrandom_bins=_LRANDOM_BINS,
        k_max=20, min_n=30, complete_only=True, drop_last_block=True,
        lrandom_only=True, lrandom_col="block_trial_random_added"):
    """Block-end-aligned P(state | k) computed SEPARATELY within each L_Random
    bin. Returns (long DataFrame with lr_lo/lr_hi columns, states).

    lrandom_only : if True (default), keep ONLY trials in each block's L_Random
        phase (k_from_block_end < L_Random, i.e. tau >= 0). This drops the
        L_Criterion trials that would otherwise appear at large k for short-
        L_Random blocks -- so every plotted point is a genuine pre-switch
        random-window trial. Set False to include the criterion phase too.
    """
    if "k_from_block_end" not in df.columns:
        df = add_positions(df)
    if lrandom_col not in df.columns:
        raise KeyError(f"need '{lrandom_col}' to stratify by L_Random")
    if lrandom_only:
        df = df[df["k_from_block_end"] < df[lrandom_col]]     # keep L_Random phase only
    if states is None:
        states = tuple(sorted(int(s) for s in pd.unique(df[state_col].dropna())))
    out = []
    for lo, hi in lrandom_bins:
        sub = df[(df[lrandom_col] >= lo) & (df[lrandom_col] <= hi)]
        occ = occupancy_block_aligned(sub, state_col, states, k_max, min_n,
                                      complete_only, drop_last_block)
        if len(occ):
            occ["lr_lo"] = lo
            occ["lr_hi"] = hi
            out.append(occ)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), states


def _shade(hex_color, t):
    """Blend hex_color toward white by fraction t in [0,1] (0=full, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * t) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_state_occupancy_by_lrandom(
        df, state_col="glmhmm_state", states=None, state_labels=None,
        lrandom_bins=_LRANDOM_BINS, k_max=20, min_n=30, lrandom_only=True,
        y_zoom=True, title=None, outfile="glmhmm_state_occupancy_by_lrandom.png"):
    """One panel per state; within each, P(state | trials before switch) drawn
    separately for each L_Random bin (darker = longer L_Random). Curves are
    composition-matched, so a real pre-switch trend cannot be an artefact of
    mixing short and long blocks. With lrandom_only=True (default) only the
    L_Random phase of each block is used -- no L_Criterion trials."""
    occ, states = occupancy_block_aligned_by_lrandom(
        df, state_col, states, lrandom_bins, k_max, min_n, lrandom_only=lrandom_only)
    if not len(occ):
        raise ValueError("no L_Random-stratified occupancy rows (check min_n / columns)")
    if states is None:
        states = tuple(sorted(occ["state"].unique()))
    n_states = len(states)
    if state_labels is None:
        state_labels = [f"state {s}" for s in range(max(states) + 1)]
    font = _font()

    bins = [(lo, hi) for lo, hi in lrandom_bins
            if ((occ["lr_lo"] == lo) & (occ["lr_hi"] == hi)).any()]
    shades = np.linspace(0.62, 0.0, len(bins))     # light (short) -> dark (long)

    fig, axes = plt.subplots(1, n_states, figsize=(6.2 * n_states, 4.6), squeeze=False)
    axes = axes[0]
    for ax, st in zip(axes, states):
        base = STATE_COLORS[st % len(STATE_COLORS)]
        pvals = []
        for (lo, hi), t in zip(bins, shades):
            d = occ[(occ["state"] == st) & (occ["lr_lo"] == lo) & (occ["lr_hi"] == hi)]
            d = d.sort_values("trials_before_end")
            if len(d):
                ax.plot(d["trials_before_end"], d["p"], "-o", ms=3, lw=1.8,
                        color=_shade(base, t), label=_lrand_label(lo, hi))
                pvals.append(d["p"].to_numpy())
        ax.axvline(0, ls="--", color=INK, lw=1)
        if y_zoom and pvals:
            allp = np.concatenate(pvals)
            lo_y, hi_y = float(allp.min()), float(allp.max())
            pad = max(0.02, 0.25 * (hi_y - lo_y))     # >=2 pts of headroom
            ax.set_ylim(max(0.0, lo_y - pad), min(1.0, hi_y + pad))
        else:
            ax.set_ylim(0, 1)
        ax.set_xlabel("trials before switch  (0 = switch)", fontsize=9.5, fontfamily=font)
        ax.set_ylabel("P(state)", fontsize=9.5, fontfamily=font)
        lab = state_labels[st] if st < len(state_labels) else f"state {st}"
        ax.set_title(lab, fontsize=11, fontfamily=font, color=INK, weight="bold")
        ax.legend(frameon=False, fontsize=8, prop={"family": font}, title="darker = longer",
                  title_fontsize=7.5)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    if title is None:
        title = f"State occupancy by L_Random length  (K={n_states}, aligned to switch)"
    fig.suptitle(title, fontsize=13, fontfamily=font, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


# ===========================================================================
# tau-aligned occupancy (aligned to the CRITERION, not the switch)
# ---------------------------------------------------------------------------
# Aligning to the block end mixes two clocks (tau-since-criterion vs distance-
# to-switch) that the animal cannot equate, because it does not know the block
# length in advance. Aligning to the criterion instead puts every block's
# L_Random phase on a common origin (tau = 0 = first random-window trial), so
# curves for different L_Random bins can be compared as they diverge from a
# shared start -- the clean way to ask whether long blocks erode differently
# from short ones.
# ===========================================================================
def occupancy_tau_by_lrandom(
        df, state_col="glmhmm_state", states=None, lrandom_bins=_LRANDOM_BINS,
        tau_max=15, min_n=30, complete_only=True, drop_last_block=True,
        normalize_by_animal=False, animal_col="animal", min_animals=5,
        lrandom_col="block_trial_random_added", ttc_col="block_trial_to_crit"):
    """P(state | tau) within each L_Random bin, tau = trials since criterion
    (0 = first L_Random trial). Only pre-switch L_Random trials are used
    (0 <= tau < L_Random). Returns (long DataFrame with lr_lo/lr_hi, states).

    normalize_by_animal : if True, subtract each animal's mean occupancy of the
        state (over the whole L_Random window) from its per-trial indicator
        before aggregating. The plotted value is then the DEVIATION from the
        animal's own baseline (0 = that animal's average). This removes level
        differences driven by which animals contribute long vs short blocks, so a
        surviving separation between L_Random bins reflects within-animal
        structure, not animal composition. With this on, the returned column
        'p' is a mean deviation in [-1, 1] and the binomial CI is replaced by a
        normal SEM (lo/hi = p +/- 1.96*SEM).
    """
    if "pos_in_block" not in df.columns:
        df = add_positions(df)
    for c in (lrandom_col, ttc_col):
        if c not in df.columns:
            raise KeyError(f"occupancy_tau_by_lrandom needs '{c}'")
    d = df.copy()
    if complete_only and "block_complete" in d:
        d = d[d["block_complete"]]
    if drop_last_block and "is_last_block" in d:
        d = d[~d["is_last_block"]]
    d["tau"] = d["pos_in_block"] - d[ttc_col]
    d = d[(d["tau"] >= 0) & (d["tau"] < d[lrandom_col])]     # L_Random, pre-switch
    d = d[d[state_col].notna()]
    if states is None:
        states = tuple(sorted(int(s) for s in pd.unique(d[state_col].dropna())))

    out = []
    for lo, hi in lrandom_bins:
        sub = d[(d[lrandom_col] >= lo) & (d[lrandom_col] <= hi)].copy()
        sub = sub[sub["tau"] <= tau_max]
        for st in states:
            ind = (sub[state_col].to_numpy() == st).astype(float)   # 0/1 in-state
            if normalize_by_animal:
                # per-animal baseline over the whole L_Random window (all bins)
                base = (d.assign(_in=(d[state_col].to_numpy() == st).astype(float))
                        .groupby(animal_col)["_in"].mean())
                ind = ind - sub[animal_col].map(base).to_numpy()
            sub["_v"] = ind
            # mean +/- SEM across animals, per tau
            agg = _mean_sem_by_animal(sub, "tau", "_v", animal_col=animal_col,
                                      min_n=min_n, min_animals=min_animals)
            for _, r in agg.iterrows():
                out.append({"tau": int(r["tau"]), "state": st, "p": float(r["p"]),
                            "lo": float(r["lo"]), "hi": float(r["hi"]),
                            "n": int(r["n"]), "n_animals": int(r["n_animals"]),
                            "lr_lo": lo, "lr_hi": hi})
    return (pd.DataFrame(out) if out else pd.DataFrame()), states


def plot_state_occupancy_by_lrandom_tau(
        df, state=None, state_col="glmhmm_state", state_labels=None,
        lrandom_bins=_LRANDOM_BINS, tau_max=15, min_n=30, y_zoom=True,
        show_ci=True, normalize_by_animal=False, title=None,
        outfile="glmhmm_state_occupancy_tau.png"):
    """Single-panel P(state | tau) by L_Random bin, aligned to the criterion.

    state : index of the state to plot. Default None -> the highest state index
        present (for K=2 with labels [exploit, explore] that is 'explore', where
        the anticipation signal is most visible). Pass an int to choose another.
    normalize_by_animal : subtract each animal's own baseline occupancy so the
        y-axis is deviation from the animal's mean (0 = baseline). Use this to
        check whether the level separation between L_Random bins is real or just
        animal composition.
    """
    occ, states = occupancy_tau_by_lrandom(
        df, state_col, None, lrandom_bins, tau_max, min_n,
        normalize_by_animal=normalize_by_animal)
    if not len(occ):
        raise ValueError("no tau-aligned occupancy rows (check min_n / columns)")
    if state is None:
        state = max(states)
    if state_labels is None:
        state_labels = [f"state {s}" for s in range(max(states) + 1)]
    lab = state_labels[state] if state < len(state_labels) else f"state {state}"
    font = _font()

    bins = [(lo, hi) for lo, hi in lrandom_bins
            if ((occ["lr_lo"] == lo) & (occ["lr_hi"] == hi)).any()]
    shades = np.linspace(0.62, 0.0, len(bins))
    base = STATE_COLORS[state % len(STATE_COLORS)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    pvals = []
    for (lo, hi), t in zip(bins, shades):
        d = occ[(occ["state"] == state) & (occ["lr_lo"] == lo) & (occ["lr_hi"] == hi)]
        d = d.sort_values("tau")
        if not len(d):
            continue
        col = _shade(base, t)
        if show_ci:
            ax.fill_between(d["tau"], d["lo"], d["hi"], color=col, alpha=0.12)
        ax.plot(d["tau"], d["p"], "-o", ms=4, lw=2, color=col,
                label=_lrand_label(lo, hi))
        pvals.append(d["p"].to_numpy())

    if normalize_by_animal:
        ax.axhline(0, color=INK, lw=1, ls="--", alpha=0.6)
    if y_zoom and pvals:
        allp = np.concatenate(pvals)
        lo_y, hi_y = float(allp.min()), float(allp.max())
        pad = max(0.02, 0.25 * (hi_y - lo_y))
        ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlabel("τ  =  trials since criterion  (0 = first L_Random trial)",
                  fontsize=10.5, fontfamily=font)
    ylab = (f"P({lab} state) − animal mean" if normalize_by_animal
            else f"P({lab} state)")
    ax.set_ylabel(ylab, fontsize=10.5, fontfamily=font)
    ax.legend(frameon=False, fontsize=9, prop={"family": font},
              title="darker = longer L_Random", title_fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if title is None:
        norm = ", per-animal normalised" if normalize_by_animal else ""
        title = f"{lab} occupancy by L_Random length  (aligned to criterion, τ{norm})"
    ax.set_title(title, fontsize=12, fontfamily=font, color=INK, weight="bold")
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


# ===========================================================================
# Across-switch occupancy (pre-switch AND post-switch), stratified by L_Random
# ---------------------------------------------------------------------------
# Aligns state occupancy to the block switch and shows a symmetric window that
# CROSSES the boundary: x < 0 = trials before the switch (end of the old block),
# x = 0 = first trial of the new block (reversal in effect), x > 0 = trials
# after. Curves are stratified by the L_Random length of the block that ENDS at
# each switch (the same bins Murphy et al. use in Fig 3G-K). Note: the post-
# switch side is necessarily the new block's L_Criterion phase (the animal
# relearning the better side), and pre-switch trials beyond that block's
# L_Random are its own criterion phase -- crossing the switch inherently spans
# both phases, unlike the L_Random-only figures.
# ===========================================================================
def occupancy_across_switch_by_lrandom(
        df, state_col="glmhmm_state", states=None, lrandom_bins=_LRANDOM_BINS,
        trials_back=10, trials_fwd=10, min_n=30, min_animals=5,
        lrandom_col="block_trial_random_added"):
    """P(state | x) for x in [-trials_back, +trials_fwd] around each block
    switch, stratified by the ENDING block's L_Random, as mean +/- SEM across
    animals. x=0 is the first trial of the new block. Returns (long DataFrame
    with lr_lo/lr_hi, states)."""
    need = ["animal", "session_file", "block_idx", "trial_idx", lrandom_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"occupancy_across_switch needs {miss}")
    if states is None:
        states = tuple(sorted(int(s) for s in pd.unique(df[state_col].dropna())))

    recs = []   # (animal, x, state_value, lr_of_ending_block)
    for (animal, _), ses in df.sort_values(
            ["animal", "session_file", "trial_idx"]).groupby(
            ["animal", "session_file"], sort=False):
        blk = ses["block_idx"].to_numpy()
        st = ses[state_col].to_numpy()
        lr = ses[lrandom_col].to_numpy()
        n = len(ses)
        ends = np.where(blk[:-1] != blk[1:])[0]
        for e in ends:
            lr_end = lr[e]                       # L_Random of the block ending here
            if np.isnan(lr_end):
                continue
            for x in range(-trials_back, trials_fwd + 1):
                # pre-switch side: keep only the ending block's L_Random window
                # (x = -1 is the last pre-switch trial). A trial at x < -lr_end
                # would be that block's L_Criterion phase -> excluded. Post-switch
                # (x >= 0) is the new block's criterion and is kept as-is.
                if x < 0 and x < -lr_end:
                    continue
                j = e + 1 + x                    # x=0 -> first trial of new block
                if 0 <= j < n and not (isinstance(st[j], float) and np.isnan(st[j])):
                    recs.append((animal, x, st[j], lr_end))
    if not recs:
        return pd.DataFrame(), states
    R = pd.DataFrame(recs, columns=["animal", "x", "state", "lr"])

    out = []
    for lo, hi in lrandom_bins:
        sub = R[(R["lr"] >= lo) & (R["lr"] <= hi)].copy()
        for stt in states:
            sub["_in"] = (sub["state"] == stt).astype(float)
            agg = _mean_sem_by_animal(sub, "x", "_in", min_n=min_n,
                                      min_animals=min_animals)
            for _, r in agg.iterrows():
                out.append({"x": int(r["x"]), "state": stt, "p": float(r["p"]),
                            "lo": float(r["lo"]), "hi": float(r["hi"]),
                            "n": int(r["n"]), "n_animals": int(r["n_animals"]),
                            "lr_lo": lo, "lr_hi": hi})
    return (pd.DataFrame(out) if out else pd.DataFrame()), states


def plot_state_occupancy_across_switch(
        df, state_col="glmhmm_state", states=None, state_labels=None,
        lrandom_bins=_LRANDOM_BINS, trials_back=10, trials_fwd=10, min_n=30,
        y_zoom=True, show_ci=True, only_state=None, title=None,
        outfile="glmhmm_state_occupancy_across_switch.png"):
    """P(state | x) around the switch (x in [-trials_back, +trials_fwd], 0 =
    first post-switch trial), one curve per L_Random bin of the ending block
    (darker = longer). show_ci draws the mean +/- SEM band.

    By default one panel per state. Pass only_state=<index> to draw a SINGLE
    panel for just that state (e.g. only_state=1 for a global P(explore) figure
    with the four L_Random-bin curves)."""
    occ, states = occupancy_across_switch_by_lrandom(
        df, state_col, states, lrandom_bins, trials_back, trials_fwd, min_n)
    if not len(occ):
        raise ValueError("no across-switch occupancy rows (check min_n / columns)")
    if state_labels is None:
        state_labels = [f"state {s}" for s in range(max(states) + 1)]
    font = _font()
    bins = [(lo, hi) for lo, hi in lrandom_bins
            if ((occ["lr_lo"] == lo) & (occ["lr_hi"] == hi)).any()]
    shades = np.linspace(0.62, 0.0, len(bins))

    plot_states = [only_state] if only_state is not None else list(states)
    n_panels = len(plot_states)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 4.8), squeeze=False)
    axes = axes[0]
    for ax, st in zip(axes, plot_states):
        base = STATE_COLORS[st % len(STATE_COLORS)]
        pvals = []
        for (lo, hi), t in zip(bins, shades):
            d = occ[(occ["state"] == st) & (occ["lr_lo"] == lo) & (occ["lr_hi"] == hi)]
            d = d.sort_values("x")
            if len(d):
                col = _shade(base, t)
                if show_ci:
                    ax.fill_between(d["x"], d["lo"], d["hi"], color=col, alpha=0.12)
                ax.plot(d["x"], d["p"], "-o", ms=3, lw=1.8, color=col,
                        label=_lrand_label(lo, hi))
                pvals.append(d[["lo", "hi"]].to_numpy().ravel() if show_ci
                             else d["p"].to_numpy())
        ax.axvline(0, ls="--", color=INK, lw=1.2)          # switch
        ax.text(0.15, ax.get_ylim()[1], " switch", fontsize=7.5, color=INK,
                va="top", ha="left")
        if y_zoom and pvals:
            allp = np.concatenate(pvals)
            lo_y, hi_y = float(allp.min()), float(allp.max())
            pad = max(0.02, 0.2 * (hi_y - lo_y))
            ax.set_ylim(max(0.0, lo_y - pad), min(1.0, hi_y + pad))
        ax.set_xlabel("trials relative to switch  (0 = first post-switch trial)",
                      fontsize=9.5, fontfamily=font)
        lab = state_labels[st] if st < len(state_labels) else f"state {st}"
        ax.set_ylabel(f"P({lab})", fontsize=9.5, fontfamily=font)
        ax.set_title(lab, fontsize=11, fontfamily=font, color=INK, weight="bold")
        ax.legend(frameon=False, fontsize=8, prop={"family": font},
                  title="darker = longer L_Random", title_fontsize=7.5)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    if title is None:
        if only_state is not None:
            lab = state_labels[only_state] if only_state < len(state_labels) else f"state {only_state}"
            title = f"P({lab}) across the switch by L_Random length"
        else:
            title = (f"State occupancy across the switch by L_Random length  "
                     f"(K={len(states)})")
    fig.suptitle(title, fontsize=13, fontfamily=font, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def plot_per_session_occupancy(
        df, state_col="glmhmm_state", states=None, state_labels=None,
        smooth_window=15, outdir="figs/occupancy/per_session",
        max_sessions=None, fname_prefix="session", show_switches=True):
    """One figure PER session: smoothed P(state) over the REAL trial axis
    (1..N_trials, unnormalized), both states, no error band (a single session is
    one realization). Vertical dashed lines mark block switches. Smoothing is a
    centered, edge-corrected moving average over `smooth_window` trials.

    Writes <outdir>/<prefix>_<animal>_<session>.png for each session and returns
    the list of written paths. Use max_sessions to cap output while testing."""
    if states is None:
        states = tuple(sorted(int(s) for s in pd.unique(df[state_col].dropna())))
    if state_labels is None:
        state_labels = [f"state {s}" for s in range(max(states) + 1)]
    font = _font()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    written = []
    grp = list(df.groupby(_SESS_KEYS, sort=False))
    if max_sessions is not None:
        grp = grp[:max_sessions]

    for (animal, ses), d in grp:
        d = d.sort_values("trial_idx") if "trial_idx" in d else d
        d = d[d[state_col].notna()]
        if len(d) < smooth_window:
            continue
        sv = d[state_col].to_numpy()
        x = np.arange(1, len(sv) + 1)                      # real trial number

        fig, ax = plt.subplots(figsize=(8.4, 4.3))
        fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)

        # switch positions: where block_idx changes (drawn first, behind curves)
        if show_switches and "block_idx" in d.columns:
            blk = d["block_idx"].to_numpy()
            sw = np.where(blk[1:] != blk[:-1])[0] + 1        # index of first trial of new block
            for i, xs_ in enumerate(sw):
                ax.axvline(x[xs_] - 0.5, color=MUTE, ls="--", lw=0.8, alpha=0.55,
                           label="switch" if i == 0 else None)

        w = min(smooth_window, len(sv))
        kern = np.ones(w) / w
        for st in states:
            ind = (sv == st).astype(float)
            sm = np.convolve(ind, kern, mode="same")
            cov = np.convolve(np.ones_like(ind), kern, mode="same")
            sm = sm / np.clip(cov, 1e-9, None)              # edge-corrected moving avg
            col = STATE_COLORS[st % len(STATE_COLORS)]
            lab = state_labels[st] if st < len(state_labels) else f"state {st}"
            ax.plot(x, sm, "-", lw=2, color=col, label=lab)

        ax.set_ylim(0, 1); ax.set_xlim(0.5, len(sv) + 0.5)
        ax.set_xlabel("trial number", fontsize=10, fontfamily=font)
        ax.set_ylabel("P(state)  (moving average)", fontsize=10, fontfamily=font)
        n_sw = len(sw) if (show_switches and "block_idx" in d.columns) else 0
        ax.set_title(f"{animal}  ·  {Path(str(ses)).stem}  ·  {len(d)} trials, "
                     f"{n_sw} switches",
                     fontsize=11, fontfamily=font, color=INK, weight="bold")
        ax.legend(frameon=False, fontsize=9, prop={"family": font}, loc="upper right")
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        safe_ses = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(Path(str(ses)).stem))
        out = str(Path(outdir) / f"{fname_prefix}_{animal}_{safe_ses}.png")
        fig.savefig(out, facecolor=CREAM, bbox_inches="tight", dpi=120)
        plt.close(fig)
        written.append(out)
    print(f"[per-session] wrote {len(written)} figures to {outdir}")
    return written


def plot_per_session_triptych(
        df, state_col="glmhmm_state", states=None, state_labels=None,
        exploit_state=0, smooth_window=15, outdir="figs/occupancy/per_session_3panel",
        max_sessions=None, fname_prefix="triptych", show_switches=True):
    """Three stacked panels per session, sharing the trial-number x-axis so the
    three signals can be read against each other (Murphy-style):

      (1) State per trial   -- a colour band, one cell per trial (exploit vs
          explore), the raw Viterbi/assigned state.
      (2) P(state)          -- the smoothed moving-average occupancy curves.
      (3) Choice quality    -- a colour band of chose-better vs chose-worse
          (choice == hr_side), with reward ticks along the top.

    Block switches are marked by dashed vertical lines in all three panels.
    Needs choice / hr_side (and optionally rewarded) columns in addition to the
    state; merge them from the main dataframe before calling if the states CSV
    lacks them."""
    need = ["choice", "hr_side"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"plot_per_session_triptych needs {miss} (merge from main df)")
    if states is None:
        states = tuple(sorted(int(s) for s in pd.unique(df[state_col].dropna())))
    if state_labels is None:
        state_labels = [f"state {s}" for s in range(max(states) + 1)]
    font = _font()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    explore_state = 1 if exploit_state == 0 else 0
    C_EXPLOIT = STATE_COLORS[exploit_state % len(STATE_COLORS)]
    C_EXPLORE = STATE_COLORS[explore_state % len(STATE_COLORS)]
    C_BETTER = "#3A6EA5"; C_WORSE = "#E8A33D"      # blue / gold, distinct from state greens

    written = []
    grp = list(df.groupby(_SESS_KEYS, sort=False))
    if max_sessions is not None:
        grp = grp[:max_sessions]

    for (animal, ses), d in grp:
        d = d.sort_values("trial_idx") if "trial_idx" in d else d
        d = d[d[state_col].notna()]
        if len(d) < smooth_window:
            continue
        sv = d[state_col].to_numpy()
        x = np.arange(1, len(sv) + 1)
        choice = d["choice"].to_numpy()
        hr = d["hr_side"].to_numpy()
        rewarded = d["rewarded"].to_numpy() if "rewarded" in d else np.full(len(d), np.nan)
        # switches
        sw = np.array([], int)
        if show_switches and "block_idx" in d.columns:
            blk = d["block_idx"].to_numpy()
            sw = np.where(blk[1:] != blk[:-1])[0] + 1

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10.5, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [0.7, 2.2, 0.9], "hspace": 0.18})
        fig.patch.set_facecolor(CREAM)
        for a in (ax1, ax2, ax3):
            a.set_facecolor(CREAM)

        # ---- Panel 1: state per trial as a colour band --------------------
        band = np.where(sv == exploit_state, 0.0, 1.0).reshape(1, -1)
        from matplotlib.colors import ListedColormap
        ax1.imshow(band, aspect="auto", cmap=ListedColormap([C_EXPLOIT, C_EXPLORE]),
                   extent=[0.5, len(sv) + 0.5, 0, 1], vmin=0, vmax=1, interpolation="nearest")
        ax1.set_yticks([]); ax1.set_ylabel("state", fontsize=9.5, fontfamily=font, rotation=0,
                                           ha="right", va="center")
        import matplotlib.patches as mpatches
        # (no legend on panel 1 -- the same exploit/explore colours are labelled
        #  by panel 2's line legend, and a band legend here collides with the title)

        # ---- Panel 2: smoothed P(state) -----------------------------------
        w = min(smooth_window, len(sv)); kern = np.ones(w) / w
        for st in states:
            ind = (sv == st).astype(float)
            sm = np.convolve(ind, kern, mode="same")
            cov = np.convolve(np.ones_like(ind), kern, mode="same")
            sm = sm / np.clip(cov, 1e-9, None)
            ax2.plot(x, sm, "-", lw=1.8, color=STATE_COLORS[st % len(STATE_COLORS)],
                     label=state_labels[st] if st < len(state_labels) else f"state {st}")
        ax2.set_ylim(0, 1); ax2.set_ylabel("P(state)\n(moving avg)", fontsize=9.5, fontfamily=font)
        ax2.legend(frameon=False, fontsize=8, loc="center left",
                   bbox_to_anchor=(1.005, 0.5), prop={"family": font})
        ax2.grid(True, color=GRID, lw=0.5, alpha=0.5)

        # ---- Panel 3: chose-better / chose-worse band + reward ticks ------
        valid = ~np.isnan(choice) & ~np.isnan(hr)
        qual = np.full(len(sv), np.nan)
        qual[valid] = (choice[valid] == hr[valid]).astype(float)   # 1=better, 0=worse
        qband = np.where(np.isnan(qual), np.nan, 1.0 - qual).reshape(1, -1)  # 0=better,1=worse
        cmapq = ListedColormap([C_BETTER, C_WORSE])
        ax3.imshow(np.ma.masked_invalid(qband), aspect="auto", cmap=cmapq,
                   extent=[0.5, len(sv) + 0.5, 0, 1], vmin=0, vmax=1, interpolation="nearest")
        # reward ticks along the top of panel 3
        rew_x = x[(rewarded == 1)] if not np.all(np.isnan(rewarded)) else np.array([])
        ax3.scatter(rew_x, np.full(len(rew_x), 1.06), marker="|", s=18, color=INK, lw=0.6,
                    clip_on=False)
        ax3.set_yticks([]); ax3.set_ylabel("choice", fontsize=9.5, fontfamily=font, rotation=0,
                                           ha="right", va="center")
        ax3.legend(handles=[mpatches.Patch(color=C_BETTER, label="chose better"),
                            mpatches.Patch(color=C_WORSE, label="chose worse"),
                            plt.Line2D([0], [0], marker="|", color=INK, lw=0, label="reward")],
                   frameon=False, fontsize=8, loc="center left",
                   bbox_to_anchor=(1.005, 0.5), prop={"family": font})
        ax3.set_xlabel("trial number", fontsize=10, fontfamily=font)

        # switches on all panels
        for a in (ax1, ax2, ax3):
            for j in sw:
                a.axvline(x[j] - 0.5, color=MUTE, ls="--", lw=0.8, alpha=0.6)
            a.set_xlim(0.5, len(sv) + 0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)

        ax1.set_title(f"{animal}  ·  {Path(str(ses)).stem}  ·  {len(d)} trials, {len(sw)} switches",
                      fontsize=11, fontfamily=font, color=INK, weight="bold", pad=18)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(Path(str(ses)).stem))
        out = str(Path(outdir) / f"{fname_prefix}_{animal}_{safe}.png")
        fig.savefig(out, facecolor=CREAM, bbox_inches="tight", dpi=130)
        plt.close(fig)
        written.append(out)
    print(f"[triptych] wrote {len(written)} figures to {outdir}")
    return written


def choice_quality_by_state_lrandom(
        df, state_col="glmhmm_state", exploit_state=0, explore_state=1,
        animal_col="animal", min_trials=20, tau_max=None,
        lrandom_col="block_trial_random_added", ttc_col="block_trial_to_crit",
        complete_only=True, drop_last_block=True):
    """
    Correlate STATE (panel 1) with CHOICE QUALITY (panel 3) across sessions:
    per animal, compare P(chose better | exploit) vs P(chose better | explore)
    over the pre-switch L_Random window. 'chose better' = choice == hr_side.

    The animal is the unit: for each animal we compute the two conditional
    probabilities, then test the paired difference (exploit - explore) across
    animals with a Wilcoxon signed-rank test. If P(better|exploit) >>
    P(better|explore), the exploit state really is exploiting the better option
    while explore is closer to chance -- a behavioural validation of the states.

    Returns a dict with the per-animal table and the test summary.
    """
    need = ["choice", "hr_side", lrandom_col, ttc_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"choice_quality_by_state_lrandom needs {miss}")
    if "pos_in_block" not in df.columns:
        df = add_positions(df)
    d = df.copy()
    if complete_only and "block_complete" in d:
        d = d[d["block_complete"]]
    if drop_last_block and "is_last_block" in d:
        d = d[~d["is_last_block"]]
    d["tau"] = d["pos_in_block"] - d[ttc_col]
    d = d[(d["tau"] >= 0) & (d["tau"] < d[lrandom_col])]          # L_Random, pre-switch
    if tau_max is not None:
        d = d[d["tau"] <= tau_max]
    d = d[d[state_col].notna() & d["choice"].notna() & d["hr_side"].notna()]
    d["_better"] = (d["choice"].to_numpy() == d["hr_side"].to_numpy()).astype(float)

    rows = []
    for animal, a in d.groupby(animal_col):
        ex = a[a[state_col] == exploit_state]["_better"]
        xp = a[a[state_col] == explore_state]["_better"]
        if len(ex) < min_trials or len(xp) < min_trials:
            continue
        rows.append({"animal": animal,
                     "p_better_exploit": float(ex.mean()),
                     "p_better_explore": float(xp.mean()),
                     "delta": float(ex.mean() - xp.mean()),
                     "n_exploit": int(len(ex)), "n_explore": int(len(xp))})
    tbl = pd.DataFrame(rows)
    if len(tbl) < 5:
        return {"table": tbl, "n": len(tbl), "median_delta": np.nan,
                "wilcoxon_p": np.nan, "note": "too few animals"}

    W, p = _stats.wilcoxon(tbl["delta"].to_numpy())
    return {"table": tbl, "n": int(len(tbl)),
            "p_better_exploit": float(tbl["p_better_exploit"].mean()),
            "p_better_explore": float(tbl["p_better_explore"].mean()),
            "median_delta": float(tbl["delta"].median()),
            "frac_pos": float((tbl["delta"] > 0).mean()),
            "wilcoxon_W": float(W), "wilcoxon_p": float(p)}


def plot_choice_quality_by_state(res, exploit_label="exploit",
                                 explore_label="explore",
                                 outfile="choice_quality_by_state.png"):
    """Paired per-animal plot of P(chose better) in exploit vs explore."""
    font = _font()
    tbl = res["table"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    C_EX = STATE_COLORS[0]; C_XP = STATE_COLORS[1]
    for _, r in tbl.iterrows():
        ax.plot([0, 1], [r["p_better_exploit"], r["p_better_explore"]],
                "-", color=MUTE, alpha=0.3, lw=0.8)
    ax.plot(np.zeros(len(tbl)), tbl["p_better_exploit"], "o", color=C_EX, ms=6, alpha=0.75)
    ax.plot(np.ones(len(tbl)), tbl["p_better_explore"], "o", color=C_XP, ms=6, alpha=0.75)
    ax.hlines(tbl["p_better_exploit"].mean(), -0.18, 0.18, color=INK, lw=2.5)
    ax.hlines(tbl["p_better_explore"].mean(), 0.82, 1.18, color=INK, lw=2.5)
    ax.axhline(0.5, ls=":", color=GRID, lw=1.2)                  # chance
    ax.text(1.02, 0.5, "chance", fontsize=8, color=MUTE, va="center", fontfamily=font)
    ax.set_xticks([0, 1]); ax.set_xticklabels([exploit_label, explore_label], fontfamily=font)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("P(chose better)  per animal", fontsize=10.5, fontfamily=font)
    ax.set_title(f"Choice quality by state (pre-switch L_Random)\n"
                 f"{exploit_label} {res['p_better_exploit']:.3f} vs "
                 f"{explore_label} {res['p_better_explore']:.3f}, "
                 f"Δ={res['median_delta']:+.3f}, p={res['wilcoxon_p']:.3g} (n={res['n']})",
                 fontsize=12, weight="bold", color=INK, fontfamily=font)
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def compare_two_states_lrandom(
        df, state_a, state_b, state_col="glmhmm_state", animal_col="animal",
        tau_max=15, min_trials=20, min_cells=3, state_labels=None,
        lrandom_col="block_trial_random_added", ttc_col="block_trial_to_crit",
        complete_only=True, drop_last_block=True):
    """
    Compare the occupancy of two GLM-HMM states (e.g. random vs side-biased) over
    the pre-switch L_Random window, with the animal as the unit. Runs two tests:

      A · LEVEL      per-animal Delta_level = mean P(state_a) - mean P(state_b)
                    over L_Random trials; Wilcoxon signed-rank across animals.
      B · TRAJECTORY per-animal slope of each state's occupancy vs tau, then
                    Delta_slope = slope_a - slope_b; Wilcoxon across animals.

    Because GLM-HMM state probabilities are compositional (they sum to 1 with the
    other states), this compares two shares of the same whole -- interpret the
    sign, not an absolute magnitude. Returns a dict with both test summaries and
    the per-animal table.
    """
    if "pos_in_block" not in df.columns:
        df = add_positions(df)
    d = df.copy()
    if complete_only and "block_complete" in d:
        d = d[d["block_complete"]]
    if drop_last_block and "is_last_block" in d:
        d = d[~d["is_last_block"]]
    d["tau"] = d["pos_in_block"] - d[ttc_col]
    d = d[(d["tau"] >= 0) & (d["tau"] < d[lrandom_col]) & d[state_col].notna()]
    d = d[d["tau"] <= tau_max]
    d["_a"] = (d[state_col].to_numpy() == state_a).astype(float)
    d["_b"] = (d[state_col].to_numpy() == state_b).astype(float)

    rows = []
    for animal, a in d.groupby(animal_col):
        if len(a) < min_trials:
            continue
        rec = {"animal": animal, "n": int(len(a)),
               "p_a": float(a["_a"].mean()), "p_b": float(a["_b"].mean())}
        rec["delta_level"] = rec["p_a"] - rec["p_b"]
        # trajectory slopes vs tau (per-animal), each state
        def _slope(col):
            g = a.groupby("tau")[col].agg(["mean", "size"])
            g = g[g["size"] >= 5]
            if len(g) < min_cells:
                return np.nan
            return float(np.polyfit(g.index.to_numpy(float), g["mean"], 1)[0])
        sa, sb = _slope("_a"), _slope("_b")
        rec["slope_a"] = sa; rec["slope_b"] = sb
        rec["delta_slope"] = (sa - sb) if (not np.isnan(sa) and not np.isnan(sb)) else np.nan
        rows.append(rec)
    tbl = pd.DataFrame(rows)

    def _wil(col):
        x = tbl[col].dropna().to_numpy()
        if len(x) < 5:
            return {"n": len(x), "median": np.nan, "frac_pos": np.nan, "p": np.nan}
        W, p = _stats.wilcoxon(x)
        return {"n": int(len(x)), "median": float(np.median(x)),
                "frac_pos": float((x > 0).mean()), "wilcoxon_W": float(W),
                "wilcoxon_p": float(p)}

    la = (state_labels[state_a] if state_labels and state_a < len(state_labels)
          else f"state {state_a}")
    lb = (state_labels[state_b] if state_labels and state_b < len(state_labels)
          else f"state {state_b}")
    return {"labels": (la, lb), "table": tbl,
            "level": _wil("delta_level"), "trajectory": _wil("delta_slope")}


def lrandom_long_vs_short_test(
        df, state=None, state_col="glmhmm_state", animal_col="animal",
        long_bin=(15, 10 ** 9), short_bin=(0, 3), tau_min=0, tau_max=None,
        min_trials=20, lrandom_col="block_trial_random_added",
        ttc_col="block_trial_to_crit", complete_only=True, drop_last_block=True):
    """
    Per-animal test of the 'long-block' effect seen in the tau/normalised figure:
    within each animal, does state occupancy (default = explore) differ between
    LONG L_Random blocks and SHORT ones, over the pre-switch L_Random window?

    For each animal:
        delta = mean P(state | long blocks) - mean P(state | short blocks)
    over L_Random trials (optionally restricted to tau in [tau_min, tau_max]);
    animals need >= min_trials in BOTH bins. delta is then tested across animals
    with a two-sided Wilcoxon signed-rank test (the animal is the unit). A
    negative median delta means long blocks have LESS of the state (less explore)
    -- the pattern the figure suggests. This is the statistical counterpart of
    the eyeballed 15+-vs-rest separation.

    Returns a dict with the per-animal table and the test summary.
    """
    if "pos_in_block" not in df.columns:
        df = add_positions(df)
    d = df.copy()
    if complete_only and "block_complete" in d:
        d = d[d["block_complete"]]
    if drop_last_block and "is_last_block" in d:
        d = d[~d["is_last_block"]]
    d["tau"] = d["pos_in_block"] - d[ttc_col]
    d = d[(d["tau"] >= 0) & (d["tau"] < d[lrandom_col]) & d[state_col].notna()]
    if tau_max is not None:
        d = d[(d["tau"] >= tau_min) & (d["tau"] <= tau_max)]
    if state is None:
        state = max(int(s) for s in pd.unique(d[state_col].dropna()))
    d["_in"] = (d[state_col].to_numpy() == state).astype(float)

    lo_l, hi_l = long_bin
    lo_s, hi_s = short_bin
    d["_long"] = (d[lrandom_col] >= lo_l) & (d[lrandom_col] <= hi_l)
    d["_short"] = (d[lrandom_col] >= lo_s) & (d[lrandom_col] <= hi_s)

    rows = []
    for animal, a in d.groupby(animal_col):
        al = a[a["_long"]]
        ash = a[a["_short"]]
        if len(al) < min_trials or len(ash) < min_trials:
            continue
        rows.append({"animal": animal, "p_long": float(al["_in"].mean()),
                     "p_short": float(ash["_in"].mean()),
                     "delta": float(al["_in"].mean() - ash["_in"].mean()),
                     "n_long": int(len(al)), "n_short": int(len(ash))})
    tbl = pd.DataFrame(rows)
    if len(tbl) < 5:
        return {"table": tbl, "n": len(tbl), "median_delta": np.nan,
                "frac_neg": np.nan, "wilcoxon_p": np.nan,
                "note": "too few animals with data in both bins"}

    W, p = _stats.wilcoxon(tbl["delta"].values)
    return {
        "table": tbl,
        "n": int(len(tbl)),
        "median_delta": float(tbl["delta"].median()),
        "mean_delta": float(tbl["delta"].mean()),
        "frac_neg": float((tbl["delta"] < 0).mean()),
        "wilcoxon_W": float(W),
        "wilcoxon_p": float(p),
        "state": int(state),
        "long_bin": long_bin,
        "short_bin": short_bin,
    }


# ===========================================================================
# Self-test (synthetic states on any bandit-shaped CSV)
# ===========================================================================
if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if csv:
        df = pd.read_csv(csv) if csv.endswith(".csv") else pd.read_pickle(csv)
    else:
        raise SystemExit("usage: python glmhmm_occupancy.py <bandit_csv_or_pkl>")
    rng = np.random.default_rng(0)
    df = add_positions(df)
    # synthetic states with mild structure to exercise the plot
    base = np.array([0.6, 0.25, 0.15])
    pe = df["pos_in_session_norm"].to_numpy()
    p_eng = np.clip(base[0] - 0.2 * pe, 0.05, 0.9)            # engaged fades over session
    st = np.empty(len(df), int)
    for i in range(len(df)):
        p = np.array([p_eng[i], 0.25, 1 - p_eng[i] - 0.25])
        st[i] = rng.choice(3, p=p / p.sum())
    df["glmhmm_state"] = st.astype(float)
    out = plot_state_occupancy(df, outfile="/mnt/user-data/outputs/glmhmm_state_occupancy_preview.png")
    print("wrote", out)