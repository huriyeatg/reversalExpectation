"""
plot_switch_hrside_random.py
=============================
Port of plot_switch_hrside_random.m (AC Kwan 170518).

Plots choice probability around high-reward-side switches,
stratified by block-length statistics (L1 / L2 ranges).

Input is the output of choice_switch_hrside_random(), either a single
dict (one session) or a list of dicts (multiple sessions → mean ± SEM).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def plot_switch_hrside_random(
    input_data,
    tlabel: str = "",
    output_dir: str = "figs",
) -> plt.Figure:
    """
    Parameters
    ----------
    input_data : dict or list of dicts
        Output(s) of choice_switch_hrside_random().
        If a list, the mean ± SEM across sessions is plotted.
    tlabel : str
        Figure title.

    Returns
    -------
    fig : matplotlib Figure
    """
    # ------------------------------------------------------------------ #
    # Aggregate across sessions if a list is provided
    # ------------------------------------------------------------------ #
    if isinstance(input_data, list):
        n         = input_data[0]["n"]
        num_range = input_data[0]["numRange"]
        L1_ranges = input_data[0]["L1_ranges"]
        L2_ranges = input_data[0]["L2_ranges"]

        stack_h = np.stack([d["probh"] for d in input_data], axis=2)   # (win, R, S)
        stack_l = np.stack([d["probl"] for d in input_data], axis=2)
        stack_n = np.stack([d["probneither"] for d in input_data], axis=2)

        n_ses   = len(input_data)
        probh   = np.nanmean(stack_h, axis=2)
        probl   = np.nanmean(stack_l, axis=2)
        probn   = np.nanmean(stack_n, axis=2)
        probh_sem = np.nanstd(stack_h, axis=2) / np.sqrt(n_ses)
        probl_sem = np.nanstd(stack_l, axis=2) / np.sqrt(n_ses)
        probn_sem = np.nanstd(stack_n, axis=2) / np.sqrt(n_ses)
        multi = True
    else:
        n         = input_data["n"]
        num_range = input_data["numRange"]
        L1_ranges = input_data["L1_ranges"]
        L2_ranges = input_data["L2_ranges"]
        probh     = input_data["probh"]      # (win, R)
        probl     = input_data["probl"]
        probn     = input_data["probneither"]
        probh_sem = probl_sem = probn_sem = None
        multi = False

    # ------------------------------------------------------------------ #
    # Colour map — PuOr (10 levels, mirrors MATLAB brewermap)
    # ------------------------------------------------------------------ #
    cmap = cm.get_cmap("PuOr", 10)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(0, color="k", linestyle="--", linewidth=2)

    handles, labels = [], []

    # ---- P(initial better option) — probh ----
    for k in range(num_range):
        color = cmap(k)
        line, = ax.plot(n, probh[:, k], ".-", markersize=10, linewidth=3, color=color)
        if multi:
            for t_idx, t_val in enumerate(n):
                ax.plot(
                    [t_val, t_val],
                    [probh[t_idx, k] - probh_sem[t_idx, k],
                     probh[t_idx, k] + probh_sem[t_idx, k]],
                    "-", linewidth=3, color=color,
                )
        lbl = (f"Initial better option"
               f" (L_C: {int(L1_ranges[k, 0])}-{int(L1_ranges[k, 1])};"
               f" L_R: {int(L2_ranges[k, 0])}-{int(L2_ranges[k, 1])})")
        handles.append(line)
        labels.append(lbl)

    # ---- P(initial worse option) — probl ----
    for k in range(num_range):
        color = cmap(9 - k)
        line, = ax.plot(n, probl[:, k], "v-", markersize=5, linewidth=3, color=color)
        if multi:
            for t_idx, t_val in enumerate(n):
                ax.plot(
                    [t_val, t_val],
                    [probl[t_idx, k] - probl_sem[t_idx, k],
                     probl[t_idx, k] + probl_sem[t_idx, k]],
                    "-", linewidth=3, color=color,
                )
        lbl = (f"Initial worse option"
               f" (L_C: {int(L1_ranges[k, 0])}-{int(L1_ranges[k, 1])};"
               f" L_R: {int(L2_ranges[k, 0])}-{int(L2_ranges[k, 1])})")
        handles.append(line)
        labels.append(lbl)

    # ---- P(miss) — probneither ----
    for k in range(num_range):
        gray = (k + 1) / (num_range + 1)
        color = (gray, gray, gray)
        line, = ax.plot(n, probn[:, k], ".-", markersize=10, linewidth=3, color=color)
        if multi:
            for t_idx, t_val in enumerate(n):
                ax.plot(
                    [t_val, t_val],
                    [probn[t_idx, k] - probn_sem[t_idx, k],
                     probn[t_idx, k] + probn_sem[t_idx, k]],
                    "-", linewidth=3, color=color,
                )
        lbl = (f"Miss"
               f" (L_C: {int(L1_ranges[k, 0])}-{int(L1_ranges[k, 1])};"
               f" L_R: {int(L2_ranges[k, 0])}-{int(L2_ranges[k, 1])})")
        handles.append(line)
        labels.append(lbl)

    ax.legend(handles, labels, frameon=False, fontsize=8)
    ax.set_ylabel("Fraction of trials")
    ax.set_xlabel("Trial from block switch")
    ax.set_xlim(n[0], n[-1])
    ax.set_ylim(0, 1)
    ax.set_title(tlabel)
    fig.tight_layout()

    if output_dir:
        out_path = Path(output_dir) / "switches_hrside_random.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")

    return fig
