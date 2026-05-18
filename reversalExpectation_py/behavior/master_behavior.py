import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio

from preprocessing.presentation_codes    import REWARD_PROBS, RULE_LABELS
from preprocessing.lesion_index          import add_lesion_info, compute_session_criteria
from preprocessing.log_parser            import parse_logfile, get_session_data, detect_phase
from behavior.trial_processing           import get_trial_masks, get_trial_stats
from behavior.trial_stats_more           import get_trial_stats_more
from behavior.choice_switch              import choice_switch_hrside_random, choice_switch_random
from behavior.plot_switch_hrside_random  import plot_switch_hrside_random
from behavior.plot_switch_random         import plot_switch_random
from behavior.models                     import fit_belief, fit_belief_ck, simulate_belief, simulate_belief_ck


# ---------------------------------------------------------------------------
# Scan folder for session files
# ---------------------------------------------------------------------------

def make_data_index(data_root: Path, subfolder: str = "") -> pd.DataFrame:
    search = Path(data_root) / subfolder if subfolder else Path(data_root)
    rows = []

    for f in sorted(search.rglob("*_beh.mat")):
        rows.append({"animal": f.parent.name.lstrip("Mm"), "session_file": f.name,
                     "beh_path": str(f), "file_type": "mat"})

    for f in sorted(search.rglob("*.log")):
        rows.append({"animal": f.parent.name.lstrip("Mm"), "session_file": f.name,
                     "beh_path": str(f), "file_type": "log"})

    df = pd.DataFrame(rows)
    mat_n = int((df["file_type"] == "mat").sum()) if len(df) else 0
    log_n = int((df["file_type"] == "log").sum()) if len(df) else 0
    print(f"Total *_beh.mat files found: {mat_n}")
    print(f"Total *.log files found: {log_n}")
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _date_from_filename(fname: str) -> float:
    m = re.search(r'_(\d{10})(?:_beh\.mat|\.log)$', fname)
    return float(m.group(1)) if m else float('nan')


def _build_dataframe_from_trial_data(trial_data, subject, n_rules, beh_path, trials, stats):
    n = len(stats["c"])
    prob_map = REWARD_PROBS.get(n_rules, {})

    block_start, t = [], 0
    for bl in stats["blockLength"]:
        block_start.append(t)
        t += int(bl)

    trial_block_idx           = np.full(n, np.nan)
    trial_block_rule          = np.full(n, np.nan)
    trial_block_trans         = np.full(n, np.nan)
    trial_block_ttc           = np.full(n, np.nan)
    trial_block_random        = np.full(n, np.nan)
    trial_block_preswitch_btr = np.full(n, np.nan)
    trial_block_preswitch_wrs = np.full(n, np.nan)
    trial_rewardrate          = np.full(n, np.nan)
    trial_hitrate             = np.full(n, np.nan)
    trial_pwinstay            = np.full(n, np.nan)
    trial_ploseswitch         = np.full(n, np.nan)

    for b, t0 in enumerate(block_start):
        bl = int(stats["blockLength"][b])
        sl = slice(t0, t0 + bl)
        trial_block_idx[sl]           = b + 1
        trial_block_rule[sl]          = stats["blockRule"][b]
        trial_block_trans[sl]         = stats["blockTrans"][b]
        trial_block_ttc[sl]           = stats["blockTrialtoCrit"][b]
        trial_block_random[sl]        = stats["blockTrialRandomAdded"][b]
        trial_block_preswitch_btr[sl] = stats["blockPreSwitchBetterChoiceAtSwitch"][b]
        trial_block_preswitch_wrs[sl] = stats["blockPreSwitchWorseChoiceAtSwitch"][b]
        trial_rewardrate[sl]          = stats["rewardrates"][b]
        trial_hitrate[sl]             = stats["hitrates"][b]
        trial_pwinstay[sl]            = stats["pWinStay"][b]
        trial_ploseswitch[sl]         = stats["pLooseSwitch"][b]

    rt = trial_data["rt"]
    motor_bias = float(abs(
        np.nanmedian(rt[trials["left"].astype(bool)]) -
        np.nanmedian(rt[trials["right"].astype(bool)])
    )) if trials["left"].any() and trials["right"].any() else float('nan')

    rows = []
    for i in range(n):
        rule_idx = stats["rule"][i]
        lp, rp = prob_map.get(int(rule_idx), (np.nan, np.nan)) if not np.isnan(rule_idx) else (np.nan, np.nan)
        c = stats["c"][i]
        hr_side = stats["hr_side"][i]
        correct = float(c == hr_side) if not (np.isnan(c) or np.isnan(hr_side)) else np.nan
        rows.append({
            "animal": subject, "session_file": beh_path.name,
            "date_number": _date_from_filename(beh_path.name),
            "phase": trial_data["presCodeSet"], "n_rules": n_rules,
            "trial_num":  int(np.sum((stats["c"] == -1) | (stats["c"] == 1))),
            "switch_num": int(np.sum(~np.isnan(stats["blockTrans"]))),
            "motor_bias": motor_bias, "trial_idx": i,
            "rule": rule_idx, "reward_prob_left": lp, "reward_prob_right": rp,
            "hr_side": hr_side, "choice": c, "rewarded": stats["r"][i],
            "correct": correct, "rt_s": float(rt[i]), "iti_s": float(trial_data["iti"][i]),
            "block_idx": trial_block_idx[i], "block_rule": trial_block_rule[i],
            "block_trans": trial_block_trans[i], "block_trial_to_crit": trial_block_ttc[i],
            "block_trial_random_added": trial_block_random[i],
            "block_preswitch_better": trial_block_preswitch_btr[i],
            "block_preswitch_worse": trial_block_preswitch_wrs[i],
            "block_rewardrate": trial_rewardrate[i], "block_hitrate": trial_hitrate[i],
            "block_pWinStay": trial_pwinstay[i], "block_pLooseSwitch": trial_ploseswitch[i],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parse a single session (.mat or .log)
# ---------------------------------------------------------------------------

def parse_single_session(beh_path: Path) -> Optional[pd.DataFrame]:
    beh_path = Path(beh_path)
    try:
        mat = sio.loadmat(str(beh_path), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        warnings.warn(f"Cannot load {beh_path.name}: {e}")
        return None
    td, sd = mat["trialData"], mat["sessionData"]
    phase, n_rules = int(td.presCodeSet), int(sd.nRules)
    trial_data = {
        "presCodeSet": phase, "cue": td.cue.astype(float),
        "response": td.response.astype(float), "outcome": td.outcome.astype(float),
        "rule": td.rule.astype(float), "cueTimes": td.cueTimes.astype(float),
        "rt": td.rt.astype(float), "iti": td.iti.astype(float),
        "leftlickTimes": list(td.leftlickTimes), "rightlickTimes": list(td.rightlickTimes),
        "n_rules": n_rules,
    }
    try:
        trials = get_trial_masks(trial_data)
        stats  = get_trial_stats(trials, n_rules=n_rules)
        stats  = get_trial_stats_more(stats)
    except Exception as e:
        warnings.warn(f"Pipeline error in {beh_path.name}: {e}")
        return None
    return _build_dataframe_from_trial_data(trial_data, str(sd.subject), n_rules, beh_path, trials, stats)


def parse_single_log_session(log_path: Path) -> Optional[pd.DataFrame]:
    log_path = Path(log_path)
    try:
        log_data = parse_logfile(log_path)
        phase = detect_phase(log_data["scenario"])
        session_data, trial_data = get_session_data(log_data, phase)
    except Exception as e:
        warnings.warn(f"Cannot parse {log_path.name}: {e}")
        return None
    n_rules = session_data["nRules"]
    trial_data["n_rules"] = n_rules
    try:
        trials = get_trial_masks(trial_data)
        stats  = get_trial_stats(trials, n_rules=n_rules)
        stats  = get_trial_stats_more(stats)
    except Exception as e:
        warnings.warn(f"Pipeline error in {log_path.name}: {e}")
        return None
    return _build_dataframe_from_trial_data(trial_data, session_data["subject"], n_rules, log_path, trials, stats)


# ---------------------------------------------------------------------------
# Full pipeline over all sessions
# ---------------------------------------------------------------------------

def build_trial_dataframe(
    data_root: str,
    subfolder: str = "",
    verbose: bool = True,
    output_dir: str = "analysis",
) -> pd.DataFrame:
    index = make_data_index(Path(data_root), subfolder)
    all_dfs = []

    for i, row in index.iterrows():
        if verbose:
            print(f"  [{i + 1}/{len(index)}] {row['session_file']}")
        parser = parse_single_session if row["file_type"] == "mat" else parse_single_log_session
        df_ses = parser(Path(row["beh_path"]))
        if df_ses is not None:
            all_dfs.append(df_ses)

    if not all_dfs:
        print("No sessions parsed successfully.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = add_lesion_info(df)
    df = compute_session_criteria(df)

    print(f"\nFinal DataFrame: {len(df):,} trials, "
          f"{df['session_file'].nunique()} sessions, "
          f"{df['animal'].nunique()} animals.")

    if output_dir:
        name_source = subfolder or data_root
        csv_stem = Path(name_source).parts[0]
        out_path = Path(output_dir) / f"{csv_stem}.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

    return df


# ---------------------------------------------------------------------------
# Switch figure helpers
# ---------------------------------------------------------------------------

def _stats_from_session_df(df_ses: pd.DataFrame) -> dict:
    """Reconstruct a stats-like dict from a session DataFrame row."""
    blocks     = df_ses.groupby("block_idx", sort=True)
    first      = blocks.first()
    block_rule  = first["block_rule"].values.astype(float)
    block_trans = first["block_trans"].values.astype(float)

    # ruletransList: unique [from_rule, to_rule] pairs across non-NaN transitions
    valid = ~np.isnan(block_trans[:-1])
    if valid.any():
        pairs = np.column_stack([block_rule[:-1][valid], block_trans[:-1][valid]])
        rule_trans_list = np.unique(pairs, axis=0)
    else:
        rule_trans_list = np.zeros((0, 2), dtype=float)

    n_rules = int(df_ses["n_rules"].iloc[0])
    rule_labels = list(RULE_LABELS.get(n_rules, {}).values())

    return {
        "blockLength":           blocks.size().values.astype(float),
        "blockTrans":            block_trans,
        "blockRule":             block_rule,
        "blockTrialtoCrit":      first["block_trial_to_crit"].values.astype(float),
        "blockTrialRandomAdded": first["block_trial_random_added"].values.astype(float),
        "c":                     df_ses["choice"].values.astype(float),
        "hr_side":               df_ses["hr_side"].values.astype(float),
        "ruletransList":         rule_trans_list,
        "rule_labels":           rule_labels,
    }


def make_lateral_switch_figure(
    df: pd.DataFrame,
    trials_back: int = 10,
    L1_ranges: np.ndarray = None,
    L2_ranges: np.ndarray = None,
    output_dir: str = "figs",
):
    if L1_ranges is None:
        L1_ranges = np.array([[0, 10], [11, 100]])
    if L2_ranges is None:
        L2_ranges = np.array([[0, 100], [0, 100]])

    session_results = []
    for (_, _), df_ses in df.groupby(["animal", "session_file"]):
        try:
            stats  = _stats_from_session_df(df_ses)
            result = choice_switch_random(stats, trials_back, L1_ranges, L2_ranges)
            if result:
                session_results.append(result)
        except Exception as e:
            warnings.warn(f"Lateral switch analysis failed: {e}")

    if not session_results:
        print("No sessions produced lateral switch data.")
        return None

    # Only keep sessions whose shape matches the majority (handles edge cases)
    ref_shape = session_results[0]["probl"].shape
    session_results = [r for r in session_results if r["probl"].shape == ref_shape]

    print(f"Lateral switch figure: {len(session_results)} sessions included.")
    return plot_switch_random(session_results, output_dir=output_dir)


def make_switch_figure(
    df: pd.DataFrame,
    trials_back: int = 10,
    L1_ranges: np.ndarray = None,
    L2_ranges: np.ndarray = None,
    output_dir: str = "figs",
):
    if L1_ranges is None:
        L1_ranges = np.array([[0, 10], [11, 100]])
    if L2_ranges is None:
        L2_ranges = np.array([[0, 100], [0, 100]])

    session_results = []
    for (_, _), df_ses in df.groupby(["animal", "session_file"]):
        try:
            stats  = _stats_from_session_df(df_ses)
            result = choice_switch_hrside_random(stats, trials_back, L1_ranges, L2_ranges)
            session_results.append(result)
        except Exception as e:
            warnings.warn(f"Switch analysis failed: {e}")

    if not session_results:
        print("No sessions produced switch data.")
        return None

    print(f"Switch figure: {len(session_results)} sessions included.")
    return plot_switch_hrside_random(session_results, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Computational model fitting and simulation
# ---------------------------------------------------------------------------

def fit_models_session(df_ses: pd.DataFrame, n_restarts: int = 5,
                       rng=None) -> dict:
    """
    Fit belief and belief-CK models to one session.

    Returns dict keyed by model name with fit result dicts
    (keys: model, fitpar, negloglike, bic, nlike).
    """
    c      = df_ses["choice"].values.astype(float)
    r      = df_ses["rewarded"].values.astype(float)
    n_rules = int(df_ses["n_rules"].iloc[0])

    results = {}
    for name, fn in [("belief", fit_belief), ("belief_ck", fit_belief_ck)]:
        try:
            results[name] = fn(c, r, n_rules=n_rules, n_restarts=n_restarts, rng=rng)
        except Exception as e:
            warnings.warn(f"Model fitting failed ({name}): {e}")
            results[name] = None
    return results


def run_model_fitting(
    df: pd.DataFrame,
    n_restarts: int = 5,
    verbose: bool = True,
    output_dir: str = "analysis",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fit belief and belief-CK models to every session in df.

    Returns a DataFrame with one row per session × model containing
    animal, session_file, model name, fitted parameters, BIC, and nlike.
    Saves a CSV when output_dir is given.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    sessions = df.groupby(["animal", "session_file"], sort=False)
    n_ses    = sessions.ngroups

    for idx, ((animal, ses_file), df_ses) in enumerate(sessions):
        if verbose:
            print(f"  [{idx + 1}/{n_ses}] {ses_file}")

        fit = fit_models_session(df_ses, n_restarts=n_restarts, rng=rng)

        for model_name, res in fit.items():
            if res is None:
                continue
            row = {
                "animal":       animal,
                "session_file": ses_file,
                "model":        model_name,
                "negloglike":   res["negloglike"],
                "bic":          res["bic"],
                "nlike":        res["nlike"],
            }
            par = res["fitpar"] if res["fitpar"] is not None else []
            labels = (["H", "beta"] if model_name == "belief"
                      else ["H", "beta", "alpha_k", "beta_k"])
            for lbl, val in zip(labels, par):
                row[f"par_{lbl}"] = float(val)
            rows.append(row)

    fit_df = pd.DataFrame(rows)

    if output_dir and len(fit_df):
        out = Path(output_dir) / "model_fits.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fit_df.to_csv(out, index=False)
        print(f"Saved → {out}")

    return fit_df


def simulate_sessions(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    n_sims: int = 100,
    seed: int = 42,
) -> dict:
    """
    Simulate choices for each session using the fitted parameters.

    Returns dict keyed by model name, each value is a list of dicts:
        {"animal", "session_file", "hr_side", "c_sim", "rule"}
    so that the simulated choices can be fed into the switch-curve functions.
    """
    rng = np.random.default_rng(seed)

    out = {"belief": [], "belief_ck": []}

    for (animal, ses_file), df_ses in df.groupby(["animal", "session_file"], sort=False):
        c     = df_ses["choice"].values.astype(float)
        r     = df_ses["rewarded"].values.astype(float)
        rule  = df_ses["rule"].values.astype(float)

        for model_name in ("belief", "belief_ck"):
            row = fit_df.loc[
                (fit_df["animal"] == animal) &
                (fit_df["session_file"] == ses_file) &
                (fit_df["model"] == model_name)
            ]
            if row.empty:
                continue

            try:
                r0 = row.iloc[0]   # scalar row — avoids Series-as-argument errors
                if model_name == "belief":
                    H, beta = float(r0["par_H"]), float(r0["par_beta"])
                    sim = simulate_belief(c, r, rule, H, beta,
                                         n_sims=n_sims, rng=rng)
                else:
                    H     = float(r0["par_H"])
                    beta  = float(r0["par_beta"])
                    ak    = float(r0["par_alpha_k"])
                    bk    = float(r0["par_beta_k"])
                    sim   = simulate_belief_ck(c, r, rule, H, beta, ak, bk,
                                               n_sims=n_sims, rng=rng)

                out[model_name].append({
                    "animal":       animal,
                    "session_file": ses_file,
                    "hr_side":      df_ses["hr_side"].values.astype(float),
                    "c_sim":        sim["c_sim"],   # (n_trials, n_sims)
                    "rule":         rule,
                })
            except Exception as e:
                warnings.warn(f"Simulation failed ({model_name}, {ses_file}): {e}")

    return out


def plot_model_bic(
    fit_df: pd.DataFrame,
    output_dir: str = "figs",
) -> "plt.Figure":
    """
    Bar chart comparing mean BIC across sessions for belief vs belief-CK,
    mirroring Figure 3F of Murphy et al. 2024.
    """
    import matplotlib.pyplot as plt

    models = ["belief", "belief_ck"]
    labels = ["Belief", "Belief-CK"]

    # Mean BIC per animal × model, then mean/SEM across animals
    per_animal = (
        fit_df.groupby(["animal", "model"])["bic"]
        .mean()
        .reset_index()
    )
    means, sems = [], []
    for m in models:
        vals = per_animal.loc[per_animal["model"] == m, "bic"].values
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals))))

    fig, ax = plt.subplots(figsize=(4, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=sems, color=["#4C72B0", "#DD8452"],
           width=0.5, capsize=5, error_kw={"linewidth": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BIC")
    ax.set_title("Model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if output_dir:
        out = Path(output_dir) / "model_bic.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")

    return fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(data_root: str = "data/data-behavior", subfolder: str = "bandit_R71_lesion/data",
        fit_models: bool = True, n_sims: int = 100):
    df = build_trial_dataframe(data_root=data_root, subfolder=subfolder)
    print(df.shape)
    print(df.dtypes)
    print(df.head())
    make_switch_figure(df)
    make_lateral_switch_figure(df)

    if fit_models:
        print("\n--- Model fitting ---")
        fit_df = run_model_fitting(df)
        plot_model_bic(fit_df)
        sim = simulate_sessions(df, fit_df, n_sims=n_sims)
        return df, fit_df, sim

    return df


if __name__ == "__main__":
    run()
