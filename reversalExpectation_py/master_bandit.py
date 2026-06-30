"""
master_bandit.py
================
Top-level entry point for all bandit task analyses.
Port of master_bandit.m (H Atilgan & AC Kwan).

This script only handles setup (figure properties, data paths) and
orchestration. All analysis and plotting live in `master_behavior.py`.

To run:
    python master_bandit.py

Pipeline overview
-----------------
PART 1  (RUN_ANALYSIS)       raw .log/.mat  ->  analysis/*.csv (behaviour)
MODELS  (RUN_MODELS)         analysis CSV   ->  fit + cross-validate the models
                                               registered in master_model.py
                                               (-> analysis/model_fits.csv, model_cv.csv, figs/)
PART 2  (RUN_PLOTS)          analysis/*.csv ->  figs/*.png
PART 3  (RUN_NEUROMODULATOR) neuromodulator analyses (independent)

Which models run, and the cross-validation schemes / speed knobs, are configured
in master_model.py (the MODELS registry and the CV_* constants). master_bandit.py
only toggles whether to run them.

Inside PART 2, the RUN_SIMULATION toggle controls how much is produced:
    RUN_SIMULATION = False  ->  only figures that do NOT need the model simulation
                                (normalized L_Random, alternation, and the real-data
                                switch / L_Random curves)
    RUN_SIMULATION = True   ->  also overlays the model-simulated switch / L_Random curves
"""

import socket
from pathlib import Path

import pandas as pd
import matplotlib as mpl
mpl.use("Agg")            # render figures to files only; never open windows
import matplotlib.pyplot as plt

from behavior.master_behavior import (
    build_trial_dataframe,
    simulate_sessions,
    plot_switch_comparison,
    plot_lrandom_comparison,
    lrandom_normalized_figure,
    alternation_lrandom_figure,
    alternation_by_lrandom_length_figure,
    lrandom_choice_variability_figure,
    alternation_after_correct_figure,
    make_switch_figure,
    make_lateral_switch_figure,
)
from behavior.beh_models.master_model import run_models
from neuromodulator.master_neuromodulator import run as run_neuromodulator
from behavior.beh_models.belief_vhr_lrandom import run_vhr_lrandom_test

# ===========================================================================
# Setup helpers
# ===========================================================================

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
        "AMEA":             r"C:\Users\cande\Documents\GitHub\reversalExpectation\reversalExpectation_py\data\data-behavior",
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


def filter_lesion_group(df, group, meets_criteria_only=False):
    """
    Subset the trial dataframe by lesion status (see the LESION_GROUP toggle) and,
    optionally, to sessions that meet the performance criterion.

    "naive"        : control + pre-lesion sessions (lesioned is NaN). Replicates the
                     MATLAB master_behavior.m subset `normalSubset = isnan(Lesioned)`.
    "post"         : post-lesion sessions only (lesioned == 1).
    "control"      : animals that were never lesioned (all their sessions are NaN).
    "lesioned_pre" : pre-lesion baseline of animals that were later lesioned.
    "all"          : no filtering (pre + post + controls pooled together).

    meets_criteria_only : if True, additionally keep only meets_criteria == True
                     rows (orthogonal to the lesion group). With group="naive"
                     this reproduces select_naive_meets_criteria — the same
                     inclusion set used by the GLM-HMM and the anticipation test.
    """
    if "lesioned" not in df.columns:
        print(f"[lesion filter] no 'lesioned' column found; skipping (group='{group}').")
        out = df
    else:
        g = str(group).lower()
        if g == "all":
            out = df
        elif g == "naive":
            out = df[df["lesioned"].isna()]
        elif g == "post":
            out = df[df["lesioned"] == 1.0]
        elif g in ("control", "lesioned_pre"):
            has_post = df.groupby("animal")["lesioned"].transform(lambda s: s.notna().any())
            out = df[~has_post] if g == "control" else df[has_post & df["lesioned"].isna()]
        else:
            raise ValueError(
                f"Unknown LESION_GROUP {group!r}; use 'naive', 'post', "
                "'control', 'lesioned_pre' or 'all'."
            )

    if meets_criteria_only:
        if "meets_criteria" in out.columns:
            out = out[out["meets_criteria"] == True]
        else:
            print("[meets_criteria filter] no 'meets_criteria' column found; skipping.")

    out = out.copy()
    n_ses_in  = df.groupby(["animal", "session_file"]).ngroups
    n_ses_out = out.groupby(["animal", "session_file"]).ngroups
    crit_note = " + meets_criteria" if meets_criteria_only else ""
    print(f"[lesion filter] group='{group}'{crit_note}: kept {len(out)} trials, "
          f"{out['animal'].nunique()} animals, {n_ses_out} sessions "
          f"(from {len(df)} trials, {df['animal'].nunique()} animals, {n_ses_in} sessions).")
    return out


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    plt.close("all")
    setup_figprop()

    # --- Paths ------------------------------------------------------------
    SUBFOLDER    = "bandit_R71_lesion/data"
    CSV_PATH     = Path("analysis/bandit_R71_lesion.csv")
    FIT_CSV_PATH = Path("analysis/model_fits.csv")

    # --- Toggles ----------------------------------------------------------
    RUN_ANALYSIS       = False
    RUN_MODELS         = False
    RUN_VHR_LRANDOM    = False
    RUN_PLOTS          = False
    RUN_NEUROMODULATOR = False

    # Lesion-status filter applied to the loaded CSV BEFORE both models and figures.
    # The MATLAB pipeline (behavior/master_behavior.m) runs the reversal analysis and
    # model comparison on `isnan(Lesioned)` = control + pre-lesion only; the pre-vs-post
    # contrast lives in its separate lesion/ pipeline. "naive" replicates that.
    # Use "all" to reproduce the old unfiltered (pre+post+control pooled) behaviour.
    LESION_GROUP = "naive"   # "naive" | "post" | "control" | "lesioned_pre" | "all"

    # Also restrict to sessions that meet the performance criterion. This is the
    # extra filter the GLM-HMM and the anticipation test already apply; turning it
    # on here makes the models and the figures use the SAME inclusion set
    # ("naive" + meets_criteria == the canonical 341,972-trial / 616-session set).
    # Set False to keep every session of the chosen lesion group.
    MEETS_CRITERIA_ONLY = True

    # Within PART 2: whether to run the (slow) model simulation.
    #   False -> only figures that DON'T need the simulation
    #   True  -> all figures (adds the model-simulated curves)
    RUN_SIMULATION = False
    N_SIMS         = 100         # simulated runs per session (if RUN_SIMULATION)

    # Simulation cache (see simulate_sessions in master_behavior.py).
    # Each model is cached per fingerprint of its fitted params, the session data
    # it reads, N_SIMS and seed, so already-simulated models are not re-run.
    #   SIM_CACHE_DIR   directory for the per-model caches; None disables caching.
    #   FORCE_SIMULATION  True -> ignore the cache and re-simulate every model.
    #                     Use this after editing the simulator code itself, since
    #                     the fingerprint does NOT track source-code changes.
    SIM_CACHE_DIR    = "analysis/simulations"
    FORCE_SIMULATION = False
    # ----------------------------------------------------------------------

    # -- PART 1: Behaviour analysis -> CSVs --------------------------------
    if RUN_ANALYSIS:
        print("\n" + "=" * 60)
        print("PART 1 - Behaviour analysis  (raw data -> CSVs)")
        print("=" * 60)
        data_root = setup_compprop()
        df = build_trial_dataframe(data_root=data_root, subfolder=SUBFOLDER)
        print("\nBehaviour CSVs saved to analysis/")

    # -- MODELS: fit + cross-validate the registered models ----------------
    if RUN_MODELS:
        print("\n" + "=" * 60)
        print("MODELS - fit + cross-validation  (CSV -> analysis/ + figs/)")
        print("=" * 60)
        df = pd.read_csv(CSV_PATH)
        df = filter_lesion_group(df, LESION_GROUP, meets_criteria_only=MEETS_CRITERIA_ONLY)
        # Which models run, the CV schemes and speed knobs are all set in
        # master_model.py (MODELS registry + CV_* constants).
        run_models(df)   # -> analysis/model_fits.csv, analysis/model_cv.csv,
                         #    figs/model_bic.png, figs/model_cv.png, + console summary
    # -- VHR L_Random: belief vs belief_vhr (anticipation test) -------------
    if RUN_VHR_LRANDOM:
        print("\n" + "=" * 60)
        print("VHR L_RANDOM - belief vs belief_vhr (anticipation test)")
        print("=" * 60)
        df = pd.read_csv(CSV_PATH)
        df = filter_lesion_group(df, LESION_GROUP, meets_criteria_only=MEETS_CRITERIA_ONLY)
        run_vhr_lrandom_test(df)


    # -- PART 2: CSVs -> Figures -------------------------------------------
    if RUN_PLOTS:
        print("\n" + "=" * 60)
        print("PART 2 - Figures  (CSVs -> figs/)")
        print("=" * 60)
        df     = pd.read_csv(CSV_PATH)
        df     = filter_lesion_group(df, LESION_GROUP, meets_criteria_only=MEETS_CRITERIA_ONLY)
        fit_df = pd.read_csv(FIT_CSV_PATH)

        # --- Figures that do NOT need the simulation ---
        lrandom_normalized_figure(df)               # -> figs/lrandom_choice_normalized_bybin.png
        alternation_lrandom_figure(df)              # -> figs/lrandom_alternation_bytrial_bybin.png
        alternation_by_lrandom_length_figure(df)    # -> figs/lrandom_alternation_bylength.png
        lrandom_choice_variability_figure(df)       # -> figs/lrandom_variability_bylength_lrandom.png
        alternation_after_correct_figure(df, pair="within")   # -> figs/alternation_after_correct_within.png
        alternation_after_correct_figure(df, pair="across")   # -> figs/alternation_after_correct_across.png
        # make_switch_figure(df)                    # -> figs/switches_hrside_random.png
        # make_lateral_switch_figure(df)            # -> figs/switches_lateral_random.png

        # --- Simulation (optional) ---
        # sim = {}        -> comparison calls produce ONLY the real curves.
        # sim populated   -> they also add the model-simulated curves.
        sim = {}
        if RUN_SIMULATION:
            print("\nSimulating sessions...")
            sim = simulate_sessions(df, fit_df, n_sims=N_SIMS,
                                    cache_dir=SIM_CACHE_DIR,
                                    force=FORCE_SIMULATION)

        # --- Switch / L_Random curves (real always; + model if simulated) ---
        plot_switch_comparison(df, sim)        # -> figs/switches_hrside_*.png
        plot_lrandom_comparison(df, sim)       # -> figs/switches_lrandom_*.png

    # -- PART 3: Neuromodulator (independent) ------------------------------
    if RUN_NEUROMODULATOR:
        print("\n" + "=" * 60)
        print("PART 3 - Neuromodulator")
        print("=" * 60)
        if not RUN_ANALYSIS:
            data_root = setup_compprop()
        run_neuromodulator(data_root=data_root)