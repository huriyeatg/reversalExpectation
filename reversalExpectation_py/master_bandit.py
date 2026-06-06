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
PART 1  (RUN_ANALYSIS)       raw .log/.mat  ->  analysis/*.csv
PART 2  (RUN_PLOTS)          analysis/*.csv ->  figs/*.png
PART 3  (RUN_NEUROMODULATOR) neuromodulator analyses (independent)
CV      (RUN_CROSSVAL)       analysis CSV   ->  analysis/model_cv.csv (out-of-sample)

Inside PART 2, the RUN_SIMULATION toggle controls how much is produced:
    RUN_SIMULATION = False  ->  only figures that do NOT need the model
                                simulation (normalized L_Random and the
                                real-data switch / L_Random curves)
    RUN_SIMULATION = True   ->  all of the above PLUS the model comparison
                                (BIC) and the model-simulated curves overlaid
                                on the switch / L_Random figures
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
    plot_model_bic,
    plot_model_cv,
    plot_switch_comparison,
    plot_lrandom_comparison,
    lrandom_normalized_figure,
    make_switch_figure,
    make_lateral_switch_figure,
)
from behavior.beh_models.models_pipeline import model_fit_belief, cross_validate_models
from neuromodulator.master_neuromodulator import run as run_neuromodulator


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
    RUN_ANALYSIS       = False   # PART 1 - raw data -> analysis/*.csv
    RUN_PLOTS          = False    # PART 2 - analysis/*.csv -> figs/*.png
    RUN_NEUROMODULATOR = False   # PART 3 - neuromodulator (independent)

    # Within PART 2: whether to run the (slow) model simulation.
    #   False -> only figures that DON'T need the simulation
    #   True  -> all figures (adds the model-simulated curves)
    RUN_SIMULATION = False
    N_SIMS         = 100         # simulated runs per session (if RUN_SIMULATION)

    # Out-of-sample cross-validation (model comparison). Slow: re-fits per fold.
    RUN_CROSSVAL   = True       # CV - analysis CSV -> analysis/model_cv.csv
    CV_SCHEMES     = ["temporal"]  # any of: "temporal","block","forward"
    CV_N_JOBS      = -1          # CPU cores for CV (-1 = all, 1 = serial)
    CV_N_RESTARTS  = 5           # optimizer restarts per fit (lower = faster)
    # ----------------------------------------------------------------------

    # -- PART 1: Behaviour analysis -> CSVs --------------------------------
    if RUN_ANALYSIS:
        print("\n" + "=" * 60)
        print("PART 1 - Behaviour analysis  (raw data -> CSVs)")
        print("=" * 60)
        data_root = setup_compprop()
        df = build_trial_dataframe(data_root=data_root, subfolder=SUBFOLDER)
        model_fit_belief(df)   # -> analysis/model_fits.csv
        print("\nCSVs saved to analysis/")

    # -- Cross-validation: out-of-sample model comparison -----------------
    if RUN_CROSSVAL:
        print("\n" + "=" * 60)
        print("Cross-validation  (CSV -> analysis/model_cv.csv)")
        print("=" * 60)
        df_cv    = pd.read_csv(CSV_PATH)
        cv_parts = []
        for scheme in CV_SCHEMES:
            print(f"\nScheme: {scheme}")
            # output_dir=None: collect all schemes and save once below.
            cv_parts.append(
                cross_validate_models(df_cv, scheme=scheme,
                                      n_restarts=CV_N_RESTARTS, n_jobs=CV_N_JOBS,
                                      output_dir=None)
            )
        cv_df       = pd.concat(cv_parts, ignore_index=True)
        cv_csv_path = Path("analysis/model_cv.csv")
        cv_csv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(cv_csv_path, index=False)
        print(f"\nSaved -> {cv_csv_path}  "
              f"({len(CV_SCHEMES)} scheme(s), {len(cv_df)} rows)")
        plot_model_cv(cv_df)               # -> figs/model_cv.png (one panel per scheme)

        # --- Console summary: median out-of-sample likelihood + win rate ---
        print("\nCross-validation summary (median per-trial test likelihood):")
        for scheme in CV_SCHEMES:
            piv = (cv_df[cv_df["scheme"] == scheme]
                   .pivot_table(index="session_file", columns="model", values="cv_nlike"))
            if not {"belief", "belief_ck"}.issubset(piv.columns):
                print(f"  {scheme:>8}: incomplete (need both models)")
                continue
            piv = piv.dropna(subset=["belief", "belief_ck"])
            n = len(piv)
            win_bk = float((piv["belief_ck"] > piv["belief"]).mean() * 100) if n else 0.0
            print(f"  {scheme:>8}: n={n:>4}  "
                  f"belief={piv['belief'].median():.3f}  "
                  f"belief-CK={piv['belief_ck'].median():.3f}  "
                  f"| belief-CK wins {win_bk:.0f}%")

    # -- PART 2: CSVs -> Figures -------------------------------------------
    if RUN_PLOTS:
        print("\n" + "=" * 60)
        print("PART 2 - Figures  (CSVs -> figs/)")
        print("=" * 60)
        df     = pd.read_csv(CSV_PATH)
        fit_df = pd.read_csv(FIT_CSV_PATH)

        # --- Figures that do NOT need the simulation ---
        lrandom_normalized_figure(df)          # -> figs/switches_lrandom_normalized.png
        # make_switch_figure(df)               # -> figs/switches_hrside_random.png
        # make_lateral_switch_figure(df)       # -> figs/switches_lateral_random.png

        # --- Simulation (optional) ---
        # sim = {}        -> comparison calls produce ONLY the real curves.
        # sim populated   -> they also add the model-simulated curves.
        sim = {}
        if RUN_SIMULATION:
            print("\nSimulating sessions...")
            sim = simulate_sessions(df, fit_df, n_sims=N_SIMS)
            plot_model_bic(fit_df)             # -> figs/model_bic.png (model comparison)

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