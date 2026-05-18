"""
plot_session_neuromodulator.py
==============================
Port of plot_session_neuromodulator.m (H Atilgan & AC Kwan).

Creates two figures per session / per animal:

  Figure 1 — session overview (3 subplots):
    1. Reward probabilities over trials (left=red, right=blue)
    2. Trial-by-trial choice + outcome timeline
    3. dF/F heatmap (imagesc over time × trial)

  Figure 2 — snake plot (one subplot per trial type):
    HR-reward, HR-no-reward, LR-reward, LR-no-reward

Saves PNG files to save_path when provided.
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plot_snake import plot_snake


FS       = 20
T_WINDOW = 120       # 6 s × 20 Hz, mirrors create_dff_files.py
T_AXIS   = np.linspace(-1.95, 4.0, T_WINDOW)   # -1.95:1/20:4


def plot_session_neuromodulator(
    stats: dict,
    trials: dict,
    tlabel: str = "",
    save_path: str = None,
) -> tuple:
    """
    Parameters
    ----------
    stats     : dict from get_trial_stats_more() — must contain c, r, rewardprob
    trials    : dict from get_trial_masks() or merge_sessions_neuromodulator()
                — must contain dff (n_trials × T_WINDOW)
    tlabel    : figure title (animal ID or session filename)
    save_path : directory to save figures; None → do not save.

    Returns
    -------
    (fig_session, fig_neural) — two matplotlib Figure objects
    """
    c          = np.asarray(stats.get("c", []), float)
    r          = np.asarray(stats.get("r", []), float)
    rewardprob = np.asarray(stats.get("rewardprob",
                             np.full((len(c), 2), np.nan)), float)

    dff_raw = trials.get("dff", None)
    if dff_raw is None:
        warnings.warn("trials dict has no 'dff' key — neural subplot will be empty.")
        dff_raw = np.full((len(c), T_WINDOW), np.nan)

    dff = np.asarray(dff_raw, float)
    n   = min(len(c), len(dff))
    c          = c[:n]
    r          = r[:n]
    rewardprob = rewardprob[:n, :]
    dff        = dff[:n, :T_WINDOW]

    n_plot = max(n, 100) * int(np.ceil(n / 100)) if n > 0 else 100

    # ------------------------------------------------------------------ #
    # Figure 1 — session overview
    # ------------------------------------------------------------------ #
    fig1, axes = plt.subplots(3, 1, figsize=(14, 9))

    # Subplot 1: reward probabilities
    ax = axes[0]
    ax.plot(rewardprob[:, 0], "r", linewidth=2, label="Left")
    ax.plot(rewardprob[:, 1], "b", linewidth=2, label="Right")
    ax.set_ylabel("Reward probability")
    ax.set_xlim(0, n_plot)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.1, 0.7, 1])
    ax.set_yticklabels(["", "10 %", "70 %", ""])
    ax.legend(frameon=False, fontsize=10)
    ax.set_xticklabels([])
    ax.set_title(tlabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    # Subplot 2: choice + outcome timeline
    ax = axes[1]
    x = np.arange(n)

    left_mask  = (c == -1)
    right_mask = (c ==  1)
    left_rew   = left_mask  & (r == 1)
    right_rew  = right_mask & (r == 1)

    # Mirrors MATLAB: bars at ±0.7 for choice, ±1 for rewarded choice
    h_left  = np.where(left_mask,  -0.7, 0.0)
    h_right = np.where(right_mask,  0.7, 0.0)
    h_lrew  = np.where(left_rew,   -1.0, 0.0)
    h_rrew  = np.where(right_rew,   1.0, 0.0)

    ax.bar(x, h_left,  width=1, color="r", linewidth=0)
    ax.bar(x, h_right, width=1, color="b", linewidth=0)
    ax.bar(x, h_lrew,  width=1, color="k", linewidth=0)
    ax.bar(x, h_rrew,  width=1, color="k", linewidth=0)

    ax.set_ylabel("Choice")
    ax.set_xlim(0, n_plot)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.7, 0.7, 1])
    ax.set_yticklabels(["Reward", "Left", "Right", "Reward"])
    n_rew  = int(np.nansum(r == 1))
    n_resp = int(np.nansum(~np.isnan(r)))
    rr     = n_rew / n_resp if n_resp else float("nan")
    ax.set_title(f"Overall reward rate = {rr:.2f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Subplot 3: dF/F heatmap
    ax = axes[2]
    im = ax.imshow(
        dff,
        aspect="auto",
        extent=[T_AXIS[0], T_AXIS[-1], n, 0],
        cmap="OrRd",
        interpolation="nearest",
    )
    ax.set_xlim(T_AXIS[0], T_AXIS[-1])
    ax.set_ylim(n, 0)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Trial")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig1.tight_layout()

    # ------------------------------------------------------------------ #
    # Figure 2 — snake plots per trial type (4 subplots)
    # ------------------------------------------------------------------ #
    bool_masks = {
        k: np.asarray(v, dtype=bool)[:n]
        for k, v in trials.items()
        if hasattr(v, "__len__") and np.asarray(v).ndim == 1
    }

    def _get(fields1, fields2=None):
        """AND fields1, AND fields2, then OR both groups."""
        def _and(fields):
            out = np.ones(n, dtype=bool)
            for f in fields:
                if f in bool_masks:
                    out &= bool_masks[f]
            return out
        m1 = _and(fields1)
        if fields2:
            return m1 | _and(fields2)
        return m1

    conditions = [
        ("HR reward",    _get(["left","reward","L70R10"], ["right","reward","L10R70"])),
        ("HR no-reward", _get(["left","noreward","L70R10"], ["right","noreward","L10R70"])),
        ("LR reward",    _get(["left","reward","L10R70"], ["right","reward","L70R10"])),
        ("LR no-reward", _get(["left","noreward","L10R70"], ["right","noreward","L70R10"])),
    ]

    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    for ax2, (lbl, mask) in zip(axes2, conditions):
        subset = dff[mask, :] if np.any(mask) else np.full((1, T_WINDOW), np.nan)
        plot_snake(subset, T_AXIS, label=lbl, ax=ax2)

    fig2.suptitle(tlabel, fontsize=12)
    fig2.tight_layout()

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if save_path:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        fig1.savefig(p / "session.png",      dpi=150, bbox_inches="tight")
        fig2.savefig(p / "neuralSignal.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {p / 'session.png'}")
        print(f"  Saved → {p / 'neuralSignal.png'}")

    return fig1, fig2
