"""
plot_switch_hrside_random.py
=============================
Port of plot_switch_hrside_random.m (AC Kwan 170518).

Plots choice probability around high-reward-side switches,
stratified by L_Random (L_R) bins with a fixed L_C.

Input is the output of choice_switch_hrside_random(), either a single
dict (one session) or a list of dicts (multiple sessions → mean ± SEM).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def plot_switch_hrside_random(
    input_data,
    tlabel: str = "",
    output_dir: str = "figs",
    xlabel: str = "Trial from block switch",
    output_filename: str = "switches_hrside_random.png",
) -> plt.Figure:
    """
    Parameters
    ----------
    input_data : dict or list of dicts
        Output(s) of choice_switch_hrside_random().
        Expected to be called with a fixed L_C and L_R divided into bins.
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

        stack_h = np.stack([d["probh"]       for d in input_data], axis=2)
        stack_l = np.stack([d["probl"]       for d in input_data], axis=2)
        stack_n = np.stack([d["probneither"] for d in input_data], axis=2)

        n_ses     = len(input_data)
        probh     = np.nanmean(stack_h, axis=2)
        probl     = np.nanmean(stack_l, axis=2)
        probn     = np.nanmean(stack_n, axis=2)
        probh_sem = np.nanstd(stack_h, axis=2) / np.sqrt(n_ses)
        probl_sem = np.nanstd(stack_l, axis=2) / np.sqrt(n_ses)
        probn_sem = np.nanstd(stack_n, axis=2) / np.sqrt(n_ses)
        multi = True
    else:
        n         = input_data["n"]
        num_range = input_data["numRange"]
        L1_ranges = input_data["L1_ranges"]
        L2_ranges = input_data["L2_ranges"]
        probh     = input_data["probh"]
        probl     = input_data["probl"]
        probn     = input_data["probneither"]
        probh_sem = probl_sem = probn_sem = None
        multi = False

    # ------------------------------------------------------------------ #
    # Colour palettes — k=0 (low L_R) is darkest, k=num_range-1 lightest
    # ------------------------------------------------------------------ #
    shade = np.linspace(1.0, 0.45, num_range)      # dark → light
    better_colors = [plt.cm.YlOrBr(v) for v in shade]
    worse_colors  = [plt.cm.Purples(v) for v in shade]
    miss_grays    = np.linspace(0.1, 0.78, num_range)
    miss_colors   = [(g, g, g) for g in miss_grays]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(0, color="k", linestyle="--", linewidth=2)

    handles, labels = [], []

    def add_section_header(name):
        handles.append(Line2D([], [], color="none"))
        labels.append(name)

    def lr_label(k):
        lo = int(L2_ranges[k, 0])
        hi = int(L2_ranges[k, 1])
        return f"$L_{{\\mathrm{{Random}}}}$: {lo} - {hi}"

    def draw_curve(data, sem, color, marker, k):
        line, = ax.plot(
            n, data[:, k],
            f"{marker}-", markersize=8, linewidth=2, color=color,
        )
        if multi and sem is not None:
            for t_idx, t_val in enumerate(n):
                ax.plot(
                    [t_val, t_val],
                    [data[t_idx, k] - sem[t_idx, k],
                     data[t_idx, k] + sem[t_idx, k]],
                    "-", linewidth=2, color=color,
                )
        return line

    # ---- P(initial better option) — probh ----
    add_section_header("Initial better option")
    for k in range(num_range):
        line = draw_curve(probh, probh_sem, better_colors[k], ".", k)
        handles.append(line)
        labels.append(lr_label(k))

    # ---- P(initial worse option) — probl ----
    add_section_header("Initial worse option")
    for k in range(num_range):
        line = draw_curve(probl, probl_sem, worse_colors[k], "v", k)
        handles.append(line)
        labels.append(lr_label(k))

    # ---- P(miss) — probneither ----
    add_section_header("Miss")
    for k in range(num_range):
        line = draw_curve(probn, probn_sem, miss_colors[k], ".", k)
        handles.append(line)
        labels.append(lr_label(k))

    ax.legend(handles, labels, frameon=False, fontsize=8)
    ax.set_ylabel("Fraction of trials")
    ax.set_xlabel(xlabel)
    ax.set_xlim(n[0], n[-1])
    ax.set_ylim(0, 1)
    ax.set_title(tlabel)
    fig.tight_layout()

    if output_dir:
        out_path = Path(output_dir) / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
