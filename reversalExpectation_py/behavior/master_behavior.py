"""
master_behavior.py
==================
Behavior-analysis library for the two-armed bandit task.

This module is a library of building blocks (data parsing, stats reconstruction,
switch/L_Random curves, model simulation, BIC comparison). The canonical entry
point that orchestrates these into a full run is `master_bandit.py`.

Layout
------
1. Data indexing            scan folders for session files
2. Session parsing          .mat / .log  ->  per-trial DataFrame
3. Full pipeline            all sessions ->  analysis/*.csv
4. Stats reconstruction     session DataFrame -> stats dict
5. Curve computation        switch- and L_Random-aligned helpers
6. Figure builders          switch / L_Random / normalized / BIC figures
7. Model simulation         simulate fitted belief / belief-CK models
"""

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from preprocessing.presentation_codes    import REWARD_PROBS, RULE_LABELS
from preprocessing.lesion_index          import add_lesion_info, compute_session_criteria
from preprocessing.log_parser            import parse_logfile, get_session_data, detect_phase
from behavior.trial_processing           import get_trial_masks, get_trial_stats
from behavior.trial_stats_more           import get_trial_stats_more
from behavior.choice_switch              import (
    choice_switch_hrside_random,
    choice_switch_random,
    choice_lrandom_start,
)
from behavior.plot_switch_hrside_random  import plot_switch_hrside_random
from behavior.plot_switch_random         import plot_switch_random
from behavior.beh_models.bayesian_models import simulate_belief, simulate_belief_ck
from behavior.beh_models.models_pipeline import model_fit_belief


# ===========================================================================
# 1. Data indexing
# ===========================================================================

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


# ===========================================================================
# 2. Session parsing  (.mat / .log  ->  per-trial DataFrame)
# ===========================================================================

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


# ===========================================================================
# 3. Full pipeline  (all sessions  ->  analysis/*.csv)
# ===========================================================================

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


# ===========================================================================
# 4. Stats reconstruction  (session DataFrame  ->  stats dict)
# ===========================================================================

def _stats_from_session_df(df_ses: pd.DataFrame) -> dict:
    """Reconstruct a stats-like dict from a session DataFrame."""
    blocks      = df_ses.groupby("block_idx", sort=True)
    first       = blocks.first()
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


# ===========================================================================
# 5. Curve computation  (switch- and L_Random-aligned helpers)
# ===========================================================================

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


def _lrandom_results_from_df(df, trials_back, L1_ranges, L2_ranges):
    results = []
    for (_, _), df_ses in df.groupby(["animal", "session_file"]):
        try:
            stats  = _stats_from_session_df(df_ses)
            result = choice_lrandom_start(stats, trials_back, L1_ranges, L2_ranges)
            results.append(result)
        except Exception as e:
            warnings.warn(f"L_Random start analysis failed (real): {e}")
    return results


def _results_from_sim(sim_sessions, df, trials_back, L1_ranges, L2_ranges, curve_fn):
    """Generic per-simulation aggregator for either switch or L_Random curves."""
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
                res = curve_fn(stats_s, trials_back, L1_ranges, L2_ranges)
                sim_results_per_run.append(res)
            except Exception:
                pass

        if sim_results_per_run:
            avg = _average_switch_results(sim_results_per_run)
            if avg is not None:
                results.append(avg)

    return results


def _average_switch_results(result_list: list) -> Optional[dict]:
    if not result_list:
        return None
    avg = dict(result_list[0])
    for key in ("prob_better", "prob_worse", "prob_neither"):
        if key in avg:
            stack    = np.stack([r[key] for r in result_list if key in r], axis=-1)
            avg[key] = np.nanmean(stack, axis=-1)
    return avg


def _save_switch_fig(results, label, output_dir, xlabel="Trial from block switch",
                     prefix="switches_hrside"):
    if not results:
        print(f"  No switch results for {label} — skipping figure.")
        return
    try:
        # Build the figure, then grab it via gcf() so this works whether
        # plot_switch_hrside_random returns the Figure or None.
        plot_switch_hrside_random(results, output_dir=None, xlabel=xlabel)
        fig = plt.gcf()
        if output_dir:
            out = Path(output_dir) / f"{prefix}_{label}.png"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved → {out}")
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Switch figure failed ({label}): {e}")


# ===========================================================================
# 6. Figure builders
# ===========================================================================

def make_switch_figure(
    df: pd.DataFrame,
    trials_back: int = 10,
    L1_ranges: np.ndarray = None,
    L2_ranges: np.ndarray = None,
    output_dir: str = "figs",
):
    if L1_ranges is None:
        L1_ranges = np.array([[1, 10000]] * 4)
    if L2_ranges is None:
        L2_ranges = np.array([[0, 4], [5, 9], [10, 14], [15, 30]])

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


def plot_switch_comparison(
    df: pd.DataFrame,
    sim: dict,
    trials_back: int = 10,
    output_dir: str = "figs",
) -> None:
    """
    Switch-aligned hr-side curves for real (and optionally simulated) data.
    Mirrors Figures 3G-K in Murphy et al. 2024.

    If `sim` is empty, only the real-data figure is produced.
    """
    L1_ranges = np.array([[1, 10000]] * 4)
    L2_ranges = np.array([[0, 4], [5, 9], [10, 14], [15, 30]])

    real_results = _switch_results_from_df(df, trials_back, L1_ranges, L2_ranges)
    _save_switch_fig(real_results, "real", output_dir)

    for model_name, sim_sessions in sim.items():
        if not sim_sessions:
            continue
        model_results = _results_from_sim(
            sim_sessions, df, trials_back, L1_ranges, L2_ranges,
            curve_fn=choice_switch_hrside_random,
        )
        _save_switch_fig(model_results, model_name, output_dir)


def plot_lrandom_comparison(
    df: pd.DataFrame,
    sim: dict,
    trials_back: int = 10,
    output_dir: str = "figs",
) -> None:
    """
    Curves aligned to the start of L_Random (not the block switch) for real
    (and optionally simulated) data.

    If `sim` is empty, only the real-data figure is produced.
    """
    L1_ranges = np.array([[1, 10000]] * 4)
    L2_ranges = np.array([[0, 4], [5, 9], [10, 14], [15, 30]])

    real_results = _lrandom_results_from_df(df, trials_back, L1_ranges, L2_ranges)
    _save_switch_fig(real_results, "real", output_dir,
                     xlabel="Trial from L_Random start",
                     prefix="switches_lrandom")

    for model_name, sim_sessions in sim.items():
        if not sim_sessions:
            continue
        model_results = _results_from_sim(
            sim_sessions, df, trials_back, L1_ranges, L2_ranges,
            curve_fn=choice_lrandom_start,
        )
        _save_switch_fig(model_results, model_name, output_dir,
                         xlabel="Trial from L_Random start",
                         prefix="switches_lrandom")


def lrandom_normalized_figure(df: pd.DataFrame, n_bins: int = 40, output_dir: str = "figs"):
    """
    P(better/worse) on a block-time-normalized axis:
      -1 = block start, 0 = L_Random start, 1 = switch.
    Criterion phase -> [-1, 0], L_Random phase -> [0, 1], so the four L_Random
    groups can be overlaid despite different absolute lengths. Reference = the
    current block's better side (hr_side[block_start]). Excludes never-crit
    blocks (NaN ttc) and L_Random == 0.
    """
    L2 = [(0, 4), (5, 9), (10, 14), (15, 30)]
    edges = np.linspace(-1, 1, n_bins + 1)
    cen = (edges[:-1] + edges[1:]) / 2
    acc = {g: {k: np.zeros(n_bins) for k in "bwmn"} for g in range(4)}

    def tally(a, ph, ch, ref):
        b = int(np.searchsorted(edges, ph, side="right") - 1)
        if b < 0 or b >= n_bins:
            return
        a["n"][b] += 1
        if np.isnan(ch):
            a["m"][b] += 1
        elif ch == ref:
            a["b"][b] += 1
        else:
            a["w"][b] += 1

    for _, g in df.groupby(["animal", "session_file"], sort=False):
        g = g.reset_index(drop=True)
        blk = g.groupby("block_idx", sort=True)
        sizes = blk.size().values.astype(int)
        ttc = blk.first()["block_trial_to_crit"].values.astype(float)
        rnd = blk.first()["block_trial_random_added"].values.astype(float)
        c = g["choice"].values.astype(float)
        hr = g["hr_side"].values.astype(float)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        last = len(sizes) - 1
        for i in range(len(sizes)):
            if np.isnan(ttc[i]) or np.isnan(rnd[i]) or i == last:
                continue
            L = int(rnd[i])
            if L < 1:
                continue
            gi = next((j for j, (lo, hi) in enumerate(L2) if lo <= L <= hi), None)
            if gi is None:
                continue
            bs, T = starts[i], int(ttc[i])
            ls, be = bs + T, bs + sizes[i]
            ref = hr[bs]
            for k, t in enumerate(range(bs, ls)):
                tally(acc[gi], -1 + (k + 0.5) / T, c[t], ref)
            for k, t in enumerate(range(ls, be)):
                tally(acc[gi], (k + 0.5) / L, c[t], ref)

    oranges = ["#7a2e10", "#c0531f", "#e8821f", "#f2b34d"]
    purples = ["#3b0a6b", "#6a2fb0", "#9a7bd0", "#c3b3e6"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for gi, (lo, hi) in enumerate(L2):
        n = acc[gi]["n"]
        ok = n > 0
        pb = np.where(ok, acc[gi]["b"] / np.where(ok, n, 1), np.nan)
        pw = np.where(ok, acc[gi]["w"] / np.where(ok, n, 1), np.nan)
        ax.plot(cen, pb, "-o", ms=3, color=oranges[gi], label=f"better $L_R$ {lo}-{hi}")
        ax.plot(cen, pw, "-v", ms=3, color=purples[gi], label=f"worse $L_R$ {lo}-{hi}")
    ax.axvline(0, ls="--", color="k", lw=1)
    ax.axvline(1, ls="-", color="k", lw=1.2)
    ax.set_xlabel("Normalized block time  (-1 start, 0 $L_R$ start, 1 switch)")
    ax.set_ylabel("Fraction of trials")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "switches_lrandom_normalized.png",
                    dpi=150, bbox_inches="tight")
    return fig


def plot_model_bic(fit_df: pd.DataFrame, output_dir: str = "figs"):
    """
    Scatter plot of BIC(belief) vs BIC(belief_ck) — one point per session.
    Points below the diagonal favour belief_ck; above favour belief.
    """
    bic_belief    = fit_df[fit_df["model"] == "belief"   ].set_index("session_file")["bic"]
    bic_belief_ck = fit_df[fit_df["model"] == "belief_ck"].set_index("session_file")["bic"]
    common = bic_belief.index.intersection(bic_belief_ck.index)

    x = bic_belief[common].values
    y = bic_belief_ck[common].values

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, color="steelblue", alpha=0.7, edgecolors="white", s=60)
    lim = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("BIC  (belief)")
    ax.set_ylabel("BIC  (belief-CK)")
    ax.set_title(f"Model comparison  (n={len(common)} sessions)")
    fig.tight_layout()

    if output_dir:
        out = Path(output_dir) / "model_bic.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig


def plot_model_cv(cv_df: pd.DataFrame, scheme=None, output_dir: str = "figs"):
    """
    Scatter of out-of-sample per-trial likelihood cv_nlike(belief) vs
    cv_nlike(belief_ck) — one point per session. Higher is better, so points
    ABOVE the diagonal favour belief_ck (the opposite convention to the BIC
    plot, where lower wins). The 0.5 reference marks chance for a two-choice
    task. When model_cv.csv holds several cross-validation schemes, one panel
    is drawn per scheme; pass `scheme` to restrict to one.
    """
    if scheme is not None:
        cv_df = cv_df[cv_df["scheme"] == scheme]
    schemes = list(pd.unique(cv_df["scheme"]))

    n = max(len(schemes), 1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, sch in zip(axes[0], schemes):
        sub = cv_df[cv_df["scheme"] == sch]
        nl_belief    = sub[sub["model"] == "belief"   ].set_index("session_file")["cv_nlike"]
        nl_belief_ck = sub[sub["model"] == "belief_ck"].set_index("session_file")["cv_nlike"]
        common = nl_belief.index.intersection(nl_belief_ck.index)

        x = nl_belief[common].values
        y = nl_belief_ck[common].values

        ax.scatter(x, y, color="steelblue", alpha=0.7, edgecolors="white", s=60)
        lo = (min(0.5, float(np.min([x, y]))) - 0.02) if len(common) else 0.48
        lim = [lo, 1.02]
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("cv nlike  (belief)")
        ax.set_ylabel("cv nlike  (belief-CK)")
        win = (float(np.mean(y > x)) * 100) if len(common) else 0.0
        ax.set_title(f"{sch}  (n={len(common)}; belief-CK wins {win:.0f}%)")

    fig.tight_layout()

    if output_dir:
        out = Path(output_dir) / "model_cv.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig


# ===========================================================================
# 7. Model simulation
# ===========================================================================

def simulate_sessions(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    n_sims: int = 100,
    seed: int = 42,
) -> dict:
    """
    Simulate choices under fitted belief and belief-CK models for every session.

    Returns
    -------
    dict keyed by model name ("belief", "belief_ck"), each value is a list of
    dicts with keys: animal, session_file, c_sim (n_trials × n_sims), hr_side.
    """
    rng = np.random.default_rng(seed)
    sim = {"belief": [], "belief_ck": []}

    for (animal, ses_file), df_ses in df.groupby(["animal", "session_file"], sort=False):
        c_real  = df_ses["choice"].values.astype(float)
        r_real  = df_ses["rewarded"].values.astype(float)
        rule    = df_ses["rule"].values.astype(float)
        hr_side = df_ses["hr_side"].values.astype(float)

        ses_fits = fit_df[
            (fit_df["animal"] == animal) &
            (fit_df["session_file"] == ses_file)
        ]

        for model_name in ("belief", "belief_ck"):
            row = ses_fits[ses_fits["model"] == model_name]
            if row.empty:
                continue
            try:
                if model_name == "belief":
                    result = simulate_belief(
                        c_real, r_real, rule,
                        H=float(row["par_H"].iloc[0]),
                        beta=float(row["par_beta"].iloc[0]),
                        n_sims=n_sims, rng=rng,
                    )
                else:
                    result = simulate_belief_ck(
                        c_real, r_real, rule,
                        H=float(row["par_H"].iloc[0]),
                        beta=float(row["par_beta"].iloc[0]),
                        alpha_k=float(row["par_alpha_k"].iloc[0]),
                        beta_k=float(row["par_beta_k"].iloc[0]),
                        n_sims=n_sims, rng=rng,
                    )
                sim[model_name].append({
                    "animal":       animal,
                    "session_file": ses_file,
                    "c_sim":        result["c_sim"],
                    "hr_side":      hr_side,
                })
            except Exception as e:
                warnings.warn(f"Simulation failed ({model_name}, {ses_file}): {e}")

    return sim