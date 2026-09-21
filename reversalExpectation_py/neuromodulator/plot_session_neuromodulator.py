"""
plot_session_neuromodulator.py
==============================
Port of plot_session_neuromodulator.m (H Atilgan & AC Kwan).

Figure 1 ('session') — 3 panels sharing the SAME x axis (trial):
    1. Reward probabilities (left = red, right = blue)
    2. Choice + outcome per trial (bars; black = rewarded, with a white strip
       separating the reward mark from the choice mark)
    3. dF/F as an image: x = trial, y = time (-1.95 to 4 s), as in
       imagesc(1:nTrials, tWindow, trials.dff') in the .m
Figure 2 ('neuralSignal') — ONE snake plot with ALL trials (trials.dff).

Changes vs. the previous port (which departed from the .m):
  - n_plot = 100*ceil(n/100). The port used max(n,100)*ceil(n/100): with
    n~900 that gave 8100 and squeezed the data against the left edge.
  - Panel 3 with trials on x (it had time on x, misaligned with panels 1-2).
  - Figure 2 = a single snake of all trials (it had 4 trial types; that is
    what bandit_neuromodulatorPerSession.m does, not this .m).
  - White bars at ±0.8 as in the .m.
Note: the .m labels sample 1 as -1.95 s, but in creatDffMatFiles sample 41 is
the cue, so sample 1 is -2.00 s (the .m axis is shifted by 50 ms). The .m axis
is kept for fidelity (T_AXIS).
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plot_snake import plot_snake


FS       = 20
T_WINDOW = 120                                   # 6 s × 20 Hz
T_AXIS   = np.round(np.arange(-1.95, 4.0 + 1e-9, 1.0 / FS), 4)   # -1.95:1/20:4 (120 pts)


def plot_session_neuromodulator(
    stats: dict,
    trials: dict,
    tlabel: str = "",
    save_path: str = None,
) -> tuple:
    """
    stats     : dict from get_trial_stats_more() — c, r, rewardprob
    trials    : dict with 'dff' (n_trials × T_WINDOW)
    tlabel    : title (animal or session)
    save_path : folder where 'session.png' and 'neuralSignal.png' are saved
    """
    c = np.asarray(stats.get("c", []), float).ravel()
    r = np.asarray(stats.get("r", []), float).ravel()
    rewardprob = np.asarray(stats.get("rewardprob", np.full((len(c), 2), np.nan)), float)

    dff_raw = trials.get("dff", None)
    if dff_raw is None:
        warnings.warn("trials has no 'dff' — the neural panel will be empty.")
        dff_raw = np.full((len(c), T_WINDOW), np.nan)
    dff = np.asarray(dff_raw, float)[:, :T_WINDOW]

    n_c = len(c)
    n_plot = int(100 * np.ceil(n_c / 100)) if n_c > 0 else 100   # .m: 100*ceil(numel(stats.c)/100)

    # ------------------------------------------------------------------ #
    # Figure 1
    # ------------------------------------------------------------------ #
    fig1, axes = plt.subplots(3, 1, figsize=(14, 9))

    ax = axes[0]
    ax.plot(np.arange(1, len(rewardprob) + 1), rewardprob[:, 0], "r", lw=2, label="Left")
    ax.plot(np.arange(1, len(rewardprob) + 1), rewardprob[:, 1], "b", lw=2, label="Right")
    ax.set_ylabel("Reward probability (%)")
    ax.legend(frameon=False)
    ax.set_xlim(0, n_plot); ax.set_ylim(0, 1)
    ax.set_xticklabels([])
    ax.set_yticks([0, 0.1, 0.7, 1]); ax.set_yticklabels(["", "10", "70", ""])
    ax.set_title(tlabel)

    ax = axes[1]
    x = np.arange(1, n_c + 1)                     # MATLAB bar() places bar i at x = i
    L, R = (c == -1), (c == 1)
    Lr, Rr = L & (r == 1), R & (r == 1)
    # same drawing order as the .m: black, white (gap), color
    for h, col in [(-1.0 * Lr, "k"), (-0.8 * Lr, "w"), (-0.7 * L, "r"),
                   (1.0 * Rr, "k"), (0.8 * Rr, "w"), (0.7 * R, "b")]:
        ax.bar(x, h, width=1, color=col, edgecolor="none")
    ax.set_ylabel("Choice")
    ax.set_xlim(0, n_plot); ax.set_ylim(-1, 1)
    ax.set_xlabel("Trial")
    ax.set_yticks([-1, -0.7, 0.7, 1]); ax.set_yticklabels(["Reward", "Left", "Right", "Reward"])
    n1, n0 = int(np.sum(r == 1)), int(np.sum(r == 0))
    ax.set_title(f"Overall reward rate = {n1 / (n1 + n0) if (n1 + n0) else np.nan:.4g}")

    ax = axes[2]
    n_tr = dff.shape[0]
    cmap = plt.get_cmap("OrRd")
    # imagesc(1:nTrials, tWindow, dff'): x = trial, y = time (y increases upward because of 'hold on')
    ax.imshow(dff.T, aspect="auto", cmap=cmap, origin="lower", interpolation="nearest",
              extent=[0.5, n_tr + 0.5, T_AXIS[0] - 0.025, T_AXIS[-1] + 0.025])
    ax.set_xlim(0, n_plot); ax.set_ylim(-2, 4)
    ax.set_xlabel("Trial"); ax.set_ylabel("Time (sec)")

    fig1.tight_layout()

    # ------------------------------------------------------------------ #
    # Figure 2 — one snake plot with all trials (.m: plot_snake(temp_psth,[0 6.5],...))
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    plot_snake(dff, T_AXIS[: dff.shape[1]], label=" ", ax=ax2)
    ax2.set_xlabel("Time from stimulus (s)")

    if save_path:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        fig1.savefig(p / "session.png", dpi=150, bbox_inches="tight")
        fig2.savefig(p / "neuralSignal.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {p / 'session.png'}")
        print(f"  Saved → {p / 'neuralSignal.png'}")

    return fig1, fig2
