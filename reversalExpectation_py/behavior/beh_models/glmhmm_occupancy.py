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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- palette (matches the slide figures) -----------------------------------
CREAM = "#FBFAF6"; INK = "#1A2E2A"
STATE_COLORS = ["#2C5F2D", "#C9472B", "#E8A33D", "#3A6EA5", "#7B5EA7"]  # by state index
GRID = "#D8D5CC"

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
                            drop_last_block=True):
    """P(state | trials before block end). Returns long DataFrame
    (k, trials_before_end, state, p, lo, hi, n). k=0 is the last trial of the
    block (the switch). complete_only / drop_last_block remove blocks that did
    not actually run to a switch (session-truncated)."""
    d = df
    if complete_only and "block_complete" in d:
        d = d[d["block_complete"]]
    if drop_last_block and "is_last_block" in d:
        d = d[~d["is_last_block"]]
    d = d[d[state_col].notna()]
    k = d["k_from_block_end"].to_numpy()
    s = d[state_col].to_numpy()
    rows = []
    for kk in range(0, k_max + 1):
        sel = k == kk
        n = int(sel.sum())
        if n < min_n:
            continue
        for st in states:
            kcount = int(np.sum(s[sel] == st))
            p, lo, hi = _binom_ci(kcount, n)
            rows.append({"k": kk, "trials_before_end": -kk, "state": st,
                         "p": float(p), "lo": float(lo), "hi": float(hi), "n": n})
    return pd.DataFrame(rows)


def occupancy_session(df, state_col="glmhmm_state", states=(0, 1, 2),
                      n_bins=20, min_n=50):
    """P(state | normalized within-session position). Returns long DataFrame
    (bin_center, state, p, lo, hi, n)."""
    d = df[df[state_col].notna()]
    pos = d["pos_in_session_norm"].to_numpy()
    s = d[state_col].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n < min_n:
            continue
        for st in states:
            kcount = int(np.sum(s[sel] == st))
            p, lo, hi = _binom_ci(kcount, n)
            rows.append({"bin_center": float(centers[b]), "state": st,
                         "p": float(p), "lo": float(lo), "hi": float(hi), "n": n})
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
                         title=None, outfile="glmhmm_state_occupancy.png"):
    """Two-panel state-occupancy figure: (left) aligned to block end/switch,
    (right) across normalized session position. `df` must carry per-trial states
    in `state_col`; positions are added internally if missing.

    states : if None (default), inferred from the states actually present in the
        data, so a K=2 model draws two lines (not a phantom flat-zero third one).
    title  : if None, set to "GLM-HMM state occupancy (K=<n_states>)"."""
    if "k_from_block_end" not in df.columns or "pos_in_session_norm" not in df.columns:
        df = add_positions(df)
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

    _draw(ax1, occ_b, "trials_before_end", states, state_labels, font,
          "trials before block end  (0 = switch)", "Aligned to block end")
    ax1.axvline(0, color=INK, lw=1.0, ls="--", alpha=0.7)
    ax1.legend(frameon=False, fontsize=9, loc="upper left", prop={"family": font})

    _draw(ax2, occ_s, "bin_center", states, state_labels, font,
          "position in session  (0 = start, 1 = end)", "Across the session")

    if title is None:
        title = f"GLM-HMM state occupancy (K={len(states)})"
    fig.suptitle(title, fontsize=13,
                 fontfamily=font, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


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