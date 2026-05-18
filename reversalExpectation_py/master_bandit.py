"""
master_bandit.py
================
Top-level entry point for all bandit task analyses.
Port of master_bandit.m (H Atilgan & AC Kwan).

To run:
    python master_bandit.py
"""

import socket
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from behavior.master_behavior import (
    run as run_behavior,
    build_trial_dataframe,
    run_model_fitting,
    simulate_sessions,
    plot_model_bic,
    _stats_from_session_df,
)
from behavior.choice_switch             import choice_switch_hrside_random
from behavior.plot_switch_hrside_random import plot_switch_hrside_random
from neuromodulator.master_neuromodulator import run as run_neuromodulator


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_figprop():
    """Port of setup_figprop.m — set default matplotlib figure properties."""
    mpl.rcParams.update({
        "figure.facecolor":    "white",
        "figure.figsize":      [10, 8],
        "axes.linewidth":      2,
        "axes.edgecolor":      "black",
        "axes.labelcolor":     "black",
        "axes.titlesize":      18,
        "axes.labelsize":      18,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.size":    6,
        "ytick.major.size":    6,
        "xtick.color":         "black",
        "ytick.color":         "black",
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":           18,
        "lines.linewidth":     3,
    })


def setup_compprop() -> str:
    """
    Port of setup_compprop.m — return data root path based on hostname.

    Add your computer and data path to the mapping below.
    If the hostname is not recognised you will be prompted to enter a path.
    """
    mapping = {
        "KWANLAB-HA":       r"E:\MATLAB\two-lickport-projects",
        "WIN-AMP016":       r"C:\Users\Huriye\Documents\code\bandit",
        "MWMJ046RNP":       r"E:\MATLAB\PRBehaviour\bandit-master",
        "MWMJ0A8Y18":       r"C:\Users\ho83\Documents\MATLAB\bandit2020\bandit",
        "HURIYEMAC.LOCAL":  "/Users/Huriye/Documents/Code/bandit",
        "DESKTOP-COVVGR0":  r"F:\Bandit Longitudinal Analysis\Bandit Github\bandit",
        "HAKKI_PC":         r"E:\Bandit Longitudinal Analysis\Bandit Github\bandit",
        "AMEA":             r"C:\Users\cande\Documents\reversalExpectation\data\data-behavior",
    }

    hostname = socket.gethostname().upper()
    if hostname in mapping:
        data_root = mapping[hostname]
        print(f"Computer: {hostname}")
    else:
        print(f"Unknown computer: {hostname}")
        data_root = input("Enter path to data-behavior folder: ").strip()

    print(f"Data root: {data_root}")
    return data_root


# ---------------------------------------------------------------------------
# Simulation of RL/Bayesian models
# ---------------------------------------------------------------------------

def run_simulation(
    df: pd.DataFrame = None,
    data_root: str = "data/data-behavior",
    subfolder: str = "bandit_R71_lesion/data",
    n_sims: int = 100,
    n_restarts: int = 5,
    output_dir: str = "figs",
    seed: int = 42
) -> tuple:
    """
    Fit belief and belief-CK models, simulate choices, and plot comparisons.

    Steps
    -----
    1. Fit belief and belief-CK models to every session
    2. Save fitted parameters → analysis/model_fits.csv
    3. Plot BIC comparison → figs/model_bic.png
    4. Simulate choices under each model
    5. Plot switch curves: real vs belief vs belief-CK → figs/switches_hrside_*.png

    Returns
    -------
    (fit_df, sim)
    """
    if df is None:
        print("Loading behavioral data...")
        df = build_trial_dataframe(data_root=data_root, subfolder=subfolder,
                                   output_dir=None)

    if df.empty:
        print("No sessions found — simulation skipped.")
        return pd.DataFrame(), {}

    print(f"\nFitting belief and belief-CK models "
          f"({df['session_file'].nunique()} sessions × {n_restarts} restarts)...")
    fit_df = run_model_fitting(df, n_restarts=n_restarts)

    plot_model_bic(fit_df, output_dir=output_dir)

    print(f"\nSimulating {n_sims} runs per session...")
    sim = simulate_sessions(df, fit_df, n_sims=n_sims, seed=seed)

    print("\nComputing switch curves (real vs simulated)...")
    _plot_switch_comparison(df, sim, output_dir=output_dir)

    return fit_df, sim


def _plot_switch_comparison(
    df: pd.DataFrame,
    sim: dict,
    trials_back: int = 10,
    output_dir: str = "figs",
) -> None:
    """
    Compute hr-side switch curves for real and simulated data, save one
    figure per dataset.  Mirrors Figures 3G-K in Murphy et al. 2024.
    """
    L1_ranges = np.array([[0, 10], [11, 100]])
    L2_ranges = np.array([[0, 100], [0, 100]])

    real_results = _switch_results_from_df(df, trials_back, L1_ranges, L2_ranges)
    _save_switch_fig(real_results, "real", output_dir)

    for model_name, sim_sessions in sim.items():
        if not sim_sessions:
            continue
        model_results = _switch_results_from_sim(
            sim_sessions, df, trials_back, L1_ranges, L2_ranges
        )
        _save_switch_fig(model_results, model_name, output_dir)


def _switch_results_from_df(df, trials_back, L1_ranges, L2_ranges):
    results = []
    for (_, _), df_ses in df.groupby(["animal", "session_file"]):
        try:
            stats  = _stats_from_session_df(df_ses)
            result = choice_switch_hrside_random(stats, trials_back, L1_ranges, L2_ranges)
            results.append(result)
        except Exception as e:
            warnings.warn(f"Switch analysis failed (real): {e}")
    return results


def _switch_results_from_sim(sim_sessions, df, trials_back, L1_ranges, L2_ranges):
    results = []
    for sim_ses in sim_sessions:
        animal, ses_file = sim_ses["animal"], sim_ses["session_file"]
        c_sim   = sim_ses["c_sim"]
        hr_side = sim_ses["hr_side"]

        df_ses = df[(df["animal"] == animal) & (df["session_file"] == ses_file)]
        if df_ses.empty:
            continue

        try:
            stats_real = _stats_from_session_df(df_ses)
        except Exception as e:
            warnings.warn(f"Failed to rebuild stats for {ses_file}: {e}")
            continue

        sim_results_per_run = []
        for s in range(c_sim.shape[1]):
            stats_s            = dict(stats_real)
            stats_s["c"]       = c_sim[:, s]
            stats_s["hr_side"] = hr_side
            try:
                res = choice_switch_hrside_random(stats_s, trials_back, L1_ranges, L2_ranges)
                sim_results_per_run.append(res)
            except Exception:
                pass

        if sim_results_per_run:
            avg = _average_switch_results(sim_results_per_run)
            if avg is not None:
                results.append(avg)

    return results


def _average_switch_results(result_list: list) -> dict:
    if not result_list:
        return None
    avg = dict(result_list[0])
    for key in ("prob_better", "prob_worse", "prob_neither"):
        if key in avg:
            stack    = np.stack([r[key] for r in result_list if key in r], axis=-1)
            avg[key] = np.nanmean(stack, axis=-1)
    return avg


def _save_switch_fig(results, label, output_dir):
    if not results:
        print(f"  No switch results for {label} — skipping figure.")
        return
    try:
        fig = plot_switch_hrside_random(results, output_dir=None)
        if output_dir:
            out = Path(output_dir) / f"switches_hrside_{label}.png"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved → {out}")
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Switch figure failed ({label}): {e}")


# ---------------------------------------------------------------------------
# Not yet ported
# ---------------------------------------------------------------------------
# master_banditlesion        — effects of Cg1/M2 lesion
# master_banditM2stimulation — effects of M2 photo-stimulation
# master_banditlongitudinal  — behavioural changes over time
# master_banditwholecortex   — whole-cortex opto mapping
# master_banditrewardrate    — effect of reward rate (Phase 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plt.close("all")
    setup_figprop()
    data_root = setup_compprop()

    # ── Behaviour analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Project: Behaviour analysis of bandit task")
    print("  Code:  behavior/")
    print("  Data:  data-behavior/bandit_R71_lesion")
    print("=" * 60)
    df = run_behavior(data_root=data_root, subfolder="bandit_R71_lesion/data",
                      fit_models=True)

    # ── Simulation of RL/Bayesian models ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Project: Simulation of RL/Bayesian models")
    print("=" * 60)
    df = pd.read_csv("analysis/bandit_R71_lesion.csv")
    run_simulation(df=df)

    # ── Neuromodulator ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Project: Neuromodulator (NE & ACh miniscope, prelimbic cortex)")
    print("  Code:  neuromodulator/")
    print("  Data:  data-behavior/bandit_neuromodulator")
    print("=" * 60)
    run_neuromodulator(data_root=data_root)
