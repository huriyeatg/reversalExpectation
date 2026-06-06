"""
plot_switch_random.py
=====================
Port of plot_switch_random.m (AC Kwan 170518).

Plots absolute left/right choice probability around high-reward-side
switches, stratified by block-length statistics (L1 / L2 ranges) and
split into one subplot per rule-transition type.

Input is the output of choice_switch_random(), either a single dict
(one session) or a list of dicts (multiple sessions → mean ± SEM).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def plot_switch_random(input_data, tlabel: str = "", output_dir: str = "figs") -> plt.Figure:
    """
    Parameters
    ----------
    input_data : dict or list of dicts
        Output(s) of choice_switch_random().
    tlabel : str
        Figure super-title.
    output_dir : str or None
        Folder to save the figure. Pass None to skip saving.

    Returns
    -------
    fig : matplotlib Figure
    """
    # ------------------------------------------------------------------ #
    # Aggregate across sessions if a list is provided
    # ------------------------------------------------------------------ #
    if isinstance(input_data, list):
        n             = input_data[0]["n"]
        num_types     = input_data[0]["numtransType"]
        trans_type    = input_data[0]["transType"]
        num_range     = input_data[0]["numRange"]
        L1_ranges     = input_data[0]["L1_ranges"]
        L2_ranges     = input_data[0]["L2_ranges"]
        rule_labels   = input_data[0].get("rule_labels", [])

        stack_l = np.stack([d["probl"]       for d in input_data], axis=3)
        stack_r = np.stack([d["probr"]       for d in input_data], axis=3)
        stack_n = np.stack([d["probneither"] for d in input_data], axis=3)

        n_ses      = len(input_data)
        probl      = np.nanmean(stack_l, axis=3)
        probr      = np.nanmean(stack_r, axis=3)
        probn      = np.nanmean(stack_n, axis=3)
        probl_sem  = np.nanstd(stack_l, axis=3) / np.sqrt(n_ses)
        probr_sem  = np.nanstd(stack_r, axis=3) / np.sqrt(n_ses)
        probn_sem  = np.nanstd(stack_n, axis=3) / np.sqrt(n_ses)
        multi = True
    else:
        n           = input_data["n"]
        num_types   = input_data["numtransType"]
        trans_type  = input_data["transType"]
        num_range   = input_data["numRange"]
        L1_ranges   = input_data["L1_ranges"]
        L2_ranges   = input_data["L2_ranges"]
        rule_labels = input_data.get("rule_labels", [])
        probl       = input_data["probl"]
        probr       = input_data["probr"]
        probn       = input_data["probneither"]
        probl_sem   = probr_sem = probn_sem = None
        multi = False

    # ------------------------------------------------------------------ #
    # Colour map — Oranges (dark → light), mirrors flip(brewermap(5,'Oranges'))
    # ------------------------------------------------------------------ #
    oranges    = cm.get_cmap("Oranges")
    color_vals = np.linspace(0.8, 0.35, num_range)   # dark to light

    fig, axes = plt.subplots(1, num_types, figsize=(7 * num_types, 5), squeeze=False)

    for j in range(num_types):
        ax = axes[0, j]
        ax.axvline(0, color="k", linestyle="--", linewidth=2)

        handles, labels = [], []

        # ---- P(left) ----
        for k in range(num_range):
            color = oranges(color_vals[k])
            line, = ax.plot(n, probl[:, j, k], ".-", markersize=10, linewidth=3, color=color)
            if multi:
                for t_idx, t_val in enumerate(n):
                    ax.plot(
                        [t_val, t_val],
                        [probl[t_idx, j, k] - probl_sem[t_idx, j, k],
                         probl[t_idx, j, k] + probl_sem[t_idx, j, k]],
                        "-", linewidth=3, color=color,
                    )
            lbl = (f"Left"
                   f" (L_C: {int(L1_ranges[k, 0])}-{int(L1_ranges[k, 1])};"
                   f" L_R: {int(L2_ranges[k, 0])}-{int(L2_ranges[k, 1])})")
            handles.append(line)
            labels.append(lbl)

        # ---- P(right) ----
        for k in range(num_range):
            color = oranges(color_vals[k])
            line, = ax.plot(n, probr[:, j, k], "v-", markersize=5, linewidth=3, color=color)
            if multi:
                for t_idx, t_val in enumerate(n):
                    ax.plot(
                        [t_val, t_val],
                        [probr[t_idx, j, k] - probr_sem[t_idx, j, k],
                         probr[t_idx, j, k] + probr_sem[t_idx, j, k]],
                        "-", linewidth=3, color=color,
                    )
            lbl = (f"Right"
                   f" (L_C: {int(L1_ranges[k, 0])}-{int(L1_ranges[k, 1])};"
                   f" L_R: {int(L2_ranges[k, 0])}-{int(L2_ranges[k, 1])})")
            handles.append(line)
            labels.append(lbl)

        # ---- P(miss) ----
        for k in range(num_range):
            gray  = (k + 1) / (num_range + 1)
            color = (gray, gray, gray)
            line, = ax.plot(n, probn[:, j, k], ".-", markersize=10, linewidth=3, color=color)
            if multi:
                for t_idx, t_val in enumerate(n):
                    ax.plot(
                        [t_val, t_val],
                        [probn[t_idx, j, k] - probn_sem[t_idx, j, k],
                         probn[t_idx, j, k] + probn_sem[t_idx, j, k]],
                        "-", linewidth=3, color=color,
                    )
            lbl = (f"Miss"
                   f" (L_C: {int(L1_ranges[k, 0])}-{int(L1_ranges[k, 1])};"
                   f" L_R: {int(L2_ranges[k, 0])}-{int(L2_ranges[k, 1])})")
            handles.append(line)
            labels.append(lbl)

        # Legend on last subplot only
        if j == num_types - 1:
            ax.legend(handles, labels, frameon=False, fontsize=8)

        ax.set_ylabel("Fraction of trials")
        ax.set_xlabel("Trial from block switch")
        ax.set_xlim(n[0], n[-1])
        ax.set_ylim(0, 1)

        # Subplot title: "rule_from → rule_to"
        from_rule = int(trans_type[j, 0])
        to_rule   = int(trans_type[j, 1])
        from_lbl  = rule_labels[from_rule - 1] if from_rule - 1 < len(rule_labels) else str(from_rule)
        to_lbl    = rule_labels[to_rule   - 1] if to_rule   - 1 < len(rule_labels) else str(to_rule)
        ax.set_title(f"{from_lbl} → {to_lbl}")

    if tlabel:
        fig.suptitle(tlabel)
    fig.tight_layout()

    if output_dir:
        out_path = Path(output_dir) / "switches_lateral_random.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
