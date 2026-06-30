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
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import joblib
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
from behavior.beh_models.belief_vhr import simulate_belief_vhr, prepare_vhr
from behavior.beh_models.master_model import fit_models, model_fit_belief


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


def lrandom_normalized_figure(df: pd.DataFrame, n_bins: int = 40, output_dir: str = "figs",
                              title: str = ""):
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
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "lrandom_choice_normalized_bybin.png",
                    dpi=150, bbox_inches="tight")
    return fig


_MODEL_COLORS = ["#C9472B", "#3A7CA5", "#E8A33D", "#2C5F2D", "#7B3811"]


def _pick_baseline(models, baseline):
    """Default baseline: 'belief' if present, else the first model."""
    models = list(models)
    if baseline is not None:
        return baseline
    return "belief" if "belief" in models else models[0]


def plot_model_bic(fit_df: pd.DataFrame, baseline=None, output_dir: str = "figs"):
    """
    Per-session BIC of each model vs a baseline model (default 'belief').

    One coloured series per non-baseline model. Lower BIC is better, so points
    BELOW the diagonal favour that model over the baseline. The legend reports,
    for each model, the fraction of sessions in which it beats the baseline.
    Works for any number of models (two models -> a single series).
    """
    models = list(pd.unique(fit_df["model"]))
    base = _pick_baseline(models, baseline)
    others = [m for m in models if m != base]

    base_bic = fit_df[fit_df["model"] == base].set_index("session_file")["bic"]

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    vals = [base_bic.values]
    for i, m in enumerate(others):
        m_bic = fit_df[fit_df["model"] == m].set_index("session_file")["bic"]
        common = base_bic.index.intersection(m_bic.index)
        if len(common) == 0:
            continue
        x = base_bic[common].values
        y = m_bic[common].values
        win = float(np.mean(y < x)) * 100   # lower BIC wins
        ax.scatter(x, y, color=_MODEL_COLORS[i % len(_MODEL_COLORS)], alpha=0.7,
                   edgecolors="white", s=55, label=f"{m}  (beats {base} {win:.0f}%)")
        vals.append(y)

    allv = np.concatenate(vals)
    lim = [float(allv.min()) * 0.95, float(allv.max()) * 1.05]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"BIC  ({base})")
    ax.set_ylabel("BIC  (model)")
    ax.set_title("Model comparison  (lower is better; below diagonal beats baseline)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if output_dir:
        out = Path(output_dir) / "model_bic.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig


def plot_model_cv(cv_df: pd.DataFrame, baseline=None, scheme=None,
                  output_dir: str = "figs"):
    """
    Per-session out-of-sample likelihood (cv_nlike) of each model vs a baseline
    (default 'belief'), one panel per cross-validation scheme.

    One coloured series per non-baseline model. Higher is better, so points
    ABOVE the diagonal favour that model; 0.5 marks chance for a two-choice
    task. The legend reports each model's win fraction over the baseline. Works
    for any number of models (two models -> a single series).
    """
    if scheme is not None:
        cv_df = cv_df[cv_df["scheme"] == scheme]
    schemes = list(pd.unique(cv_df["scheme"]))
    models = list(pd.unique(cv_df["model"]))
    base = _pick_baseline(models, baseline)
    others = [m for m in models if m != base]

    n = max(len(schemes), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), squeeze=False)

    for ax, sch in zip(axes[0], schemes):
        sub = cv_df[cv_df["scheme"] == sch]
        base_nl = sub[sub["model"] == base].set_index("session_file")["cv_nlike"]
        lo = 0.5
        for i, m in enumerate(others):
            m_nl = sub[sub["model"] == m].set_index("session_file")["cv_nlike"]
            common = base_nl.index.intersection(m_nl.index)
            if len(common) == 0:
                continue
            x = base_nl[common].values
            y = m_nl[common].values
            win = float(np.mean(y > x)) * 100
            ax.scatter(x, y, color=_MODEL_COLORS[i % len(_MODEL_COLORS)], alpha=0.7,
                       edgecolors="white", s=55, label=f"{m}  (beats {base} {win:.0f}%)")
            lo = min(lo, float(np.min([x, y])))
        lim = [lo - 0.02, 1.02]
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"cv nlike  ({base})")
        ax.set_ylabel("cv nlike  (model)")
        ax.set_title(f"{sch}  (above diagonal = better)")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()

    if output_dir:
        out = Path(output_dir) / "model_cv.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig


def alternation_lrandom_figure(
    df: pd.DataFrame,
    start_j: int = 1,
    min_count: int = 30,
    ylim=(0.0, 0.25),
    show_criterion_baseline: bool = True,
    output_dir: str = "figs",
    save: bool = True,
    title: str = "",
    split_by_reward: bool = True,
    reward_ref: str = "prev",
):
    """
    P(alternation) during the L_Random phase vs absolute trial (not warped).
 
    Alternation at a trial = its choice differs from the previous trial's choice
    (both must be non-miss). The x-axis is the trial index within L_Random
    (0 = first L_Random trial). One curve per L_Random bin (0-4, 5-9, 10-14,
    15-30). Each curve only extends as far as its bin allows, and positions with
    fewer than `min_count` blocks are dropped so the tail is not noisy.
 
    Parameters
    ----------
    start_j : first L_Random position to plot. 1 (default) = alternation strictly
              within L_Random (each trial vs the previous L_Random trial). Use 0
              to include the first L_Random trial, compared to the last criterion
              trial of the same block.
    min_count : minimum blocks contributing to a plotted position.
    ylim : y-axis limits (alternation sits well below the 0.5 chance level). In
              split mode the default (0, 0.25) is treated as "auto" — the y-axis
              is scaled to the data — because the after-no-reward (lose-switch)
              curve sits far above the after-reward (win-stay) one; pass an
              explicit tuple to override. None always autoscales.
    show_criterion_baseline : draw the mean within-criterion alternation as a
              horizontal reference line (computed per condition when split).
    output_dir, save : where/whether to write the PNG.
    title : optional figure title.
    split_by_reward : if True (default), draw two panels — alternation after a
              rewarded trial (win-stay regime) vs after an unrewarded trial
              (lose-switch regime) — sharing the y-axis for direct comparison,
              saved as lrandom_alternation_bytrial_bybin_byreward.png. If False,
              the original single-panel figure (lrandom_alternation_bytrial_bybin.png).
    reward_ref : which trial's outcome conditions the split. "prev" (default)
              uses the previous trial's reward (the outcome that drives the
              stay/switch decision — win-stay/lose-switch framing); "current"
              uses the trial's own reward.
 
    Only blocks that reached criterion (ttc/L_Random non-NaN), have L_Random >= 1,
    and whose row count equals ttc + L_Random are used.
    """
    if reward_ref not in ("prev", "current"):
        raise ValueError("reward_ref must be 'prev' or 'current'")
    colors = {"0-4": "#7B3811", "5-9": "#C0531F", "10-14": "#E78A2E", "15-30": "#F2B33C"}
    lr_bins = [("0-4", 1, 4), ("5-9", 5, 9), ("10-14", 10, 14), ("15-30", 15, np.inf)]
 
    key = ["animal", "session_file", "block_idx"]
    work = df.copy()
    work["_t"] = work.groupby(key, sort=False).cumcount()
    grp = work.groupby(key, sort=False)
    ttc = grp["block_trial_to_crit"].transform("first")
    lr = grp["block_trial_random_added"].transform("first")
    n = grp["_t"].transform("size")
 
    keep = ttc.notna() & lr.notna() & (lr >= 1) & (n == (ttc + lr))
    work = work[keep].copy()
    work["_ttc"] = ttc[keep].astype(int)
    work["_lr"] = lr[keep].astype(int)
 
    # Previous choice / previous reward within the block. L_Random rows (and
    # criterion rows with _t >= 1) are never the first row of a block, so a plain
    # shift stays inside the same block for every row we actually score.
    work["_choice_prev"] = work["choice"].shift(1)
    work["_rew_prev"] = work["rewarded"].shift(1)
    work["_rew_cur"] = work["rewarded"]
    t = work["_t"].to_numpy()
    tc = work["_ttc"].to_numpy()
    alt = (work["choice"].to_numpy() != work["_choice_prev"].to_numpy()).astype(float)
    valid = work["choice"].notna().to_numpy() & work["_choice_prev"].notna().to_numpy()
 
    ref_col = "_rew_prev" if reward_ref == "prev" else "_rew_cur"
    ref_arr = work[ref_col].to_numpy()
 
    def _baseline(extra_mask=None):
        crit = (t >= 1) & (t < tc) & valid
        if extra_mask is not None:
            crit = crit & extra_mask
        return float(np.mean(alt[crit])) if crit.any() else np.nan
 
    # L_Random rows.
    is_lr = t >= tc
    work["_j"] = work["_t"] - work["_ttc"]
    work["_alt"] = alt
    lrdf = work[is_lr & valid & (work["_j"] >= start_j)].copy()
 
    def _bin(v):
        for label, lo, hi in lr_bins:
            if lo <= v <= hi:
                return label
        return None
    lrdf["_bin"] = lrdf["_lr"].map(_bin)
 
    def _plot_bins(ax, sub):
        for label, _, _ in lr_bins:
            s = sub[sub["_bin"] == label]
            if s.empty:
                continue
            agg = s.groupby("_j")["_alt"].agg(["mean", "size"])
            agg = agg[agg["size"] >= min_count]
            if agg.empty:
                continue
            ax.plot(agg.index, agg["mean"], color=colors[label], lw=2.3,
                    marker="o", ms=4, label=f"$L_R$ {label}")
        ax.spines[["top", "right"]].set_visible(False)
 
    # -- single-panel (legacy) --------------------------------------------
    if not split_by_reward:
        fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=130)
        _plot_bins(ax, lrdf)
        baseline = _baseline() if show_criterion_baseline else np.nan
        if show_criterion_baseline and np.isfinite(baseline):
            ax.axhline(baseline, ls="--", color="0.5", lw=1.2)
            ax.text(ax.get_xlim()[1], baseline, f"  criterion baseline ({baseline:.2f})",
                    va="center", ha="left", color="0.4", fontsize=9)
        ax.set_xlabel("Trial from L_Random start", fontsize=13)
        ax.set_ylabel("P(alternation)   choice \u2260 previous", fontsize=13)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(title="L_Random bin", fontsize=10)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save and output_dir:
            out = Path(output_dir) / "lrandom_alternation_bytrial_bybin.png"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved \u2192 {out}")
        return fig
 
    # -- two-panel: rewarded vs unrewarded --------------------------------
    if reward_ref == "prev":
        titles = {1: "after rewarded trial  (win-stay regime)",
                  0: "after unrewarded trial  (lose-switch regime)"}
    else:
        titles = {1: "rewarded trials", 0: "unrewarded trials"}
 
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), dpi=130, sharey=True)
    for ax, val in zip(axes, (1, 0)):           # left = rewarded, right = unrewarded
        cond = ref_arr == val
        _plot_bins(ax, lrdf[lrdf[ref_col] == val])
        if show_criterion_baseline:
            base = _baseline(extra_mask=cond)
            if np.isfinite(base):
                ax.axhline(base, ls="--", color="0.5", lw=1.2)
                ax.text(0.98, base, f" baseline {base:.2f}",
                        transform=ax.get_yaxis_transform(),
                        va="bottom", ha="right", color="0.4", fontsize=8.5)
        ax.set_xlabel("Trial from L_Random start", fontsize=12)
        ax.set_title(titles[val], fontsize=12)
    axes[0].set_ylabel("P(alternation)   choice \u2260 previous", fontsize=12)
    if ylim not in (None, (0.0, 0.25)):
        axes[0].set_ylim(*ylim)
    else:
        axes[0].set_ylim(0, max(ax.get_ylim()[1] for ax in axes))   # shared (sharey)
    axes[1].legend(title="L_Random bin", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    else:
        cond_word = "previous-trial" if reward_ref == "prev" else "current-trial"
        fig.suptitle(f"L_Random alternation split by {cond_word} reward",
                     fontsize=13, y=1.02)
    fig.tight_layout()
 
    if save and output_dir:
        out = Path(output_dir) / "lrandom_alternation_bytrial_bybin_byreward.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved \u2192 {out}")
    return fig


def lrandom_choice_variability_figure(
    df: pd.DataFrame,
    phase: str = "lrandom",
    max_lrandom: int = 30,
    min_blocks: int = 15,
    min_trials: int = 2,
    correct_sampling_noise: bool = True,
    show_counts: bool = True,
    output_dir: str = "figs",
    save: bool = True,
    title: str = "",
):
    """
    Between-block variability of choice as a function of L_Random length.

    For every valid block, the fraction of 'better-side' choices (choice equal to
    the block's high-reward side, hr_side at block start; misses excluded) is
    computed over the selected `phase`. Blocks are then grouped by their L_Random
    length, and the standard deviation of that fraction *across blocks* is plotted
    against L_Random length.

    A block contributes only if it reached criterion (ttc / L_Random non-NaN), has
    1 <= L_Random <= max_lrandom, its row count equals ttc + L_Random, and it has
    at least `min_trials` non-miss choices in the selected phase. L_Random lengths
    with fewer than `min_blocks` contributing blocks are dropped.

    Parameters
    ----------
    phase : {"lrandom", "criterion", "block"}
        Trials over which the better-side fraction is computed for each block.
        "lrandom" (default) uses the random phase only. Note this makes the
        per-block sample size roughly equal to the L_Random length on the x-axis,
        so very short L_Random points are intrinsically noisier (see
        `correct_sampling_noise`).
    correct_sampling_noise : bool
        If True, also plot a sampling-noise-corrected SD. The raw across-block
        variance mixes true between-block heterogeneity with within-block binomial
        sampling noise (~p(1-p)/n per block). Because shorter L_Random phases have
        fewer trials, that noise inflates the raw SD at small L_Random and can
        masquerade as a real trend. The corrected SD subtracts the mean per-block
        binomial variance:  SD_corr = sqrt(max(0, Var_raw - mean[p(1-p)/n])).
    min_blocks : minimum contributing blocks for an L_Random length to be plotted.
    min_trials : minimum non-miss choices in the phase for a block to count.
    show_counts : draw the number of contributing blocks per length on a faint
        secondary axis.
    output_dir, save : where/whether to write the PNG.

    Returns
    -------
    (fig, res) : the Matplotlib figure and a DataFrame with one row per plotted
        L_Random length (columns: L, sd_raw, sd_corr, mean_p, n_blocks).
    """
    if phase not in ("lrandom", "criterion", "block"):
        raise ValueError("phase must be 'lrandom', 'criterion' or 'block'")

    key = ["animal", "session_file", "block_idx"]
    work = df.copy()
    work["_t"] = work.groupby(key, sort=False).cumcount()
    grp = work.groupby(key, sort=False)
    ttc = grp["block_trial_to_crit"].transform("first")
    lr  = grp["block_trial_random_added"].transform("first")
    n   = grp["_t"].transform("size")
    ref = grp["hr_side"].transform("first")

    keep = (ttc.notna() & lr.notna() & (lr >= 1) & (lr <= max_lrandom)
            & (n == (ttc + lr)))
    work = work[keep].copy()
    work["_ttc"] = ttc[keep].astype(int)
    work["_lr"]  = lr[keep].astype(int)
    work["_ref"] = ref[keep]

    t  = work["_t"].to_numpy()
    tc = work["_ttc"].to_numpy()
    if phase == "lrandom":
        in_phase = t >= tc
    elif phase == "criterion":
        in_phase = t < tc
    else:  # "block"
        in_phase = np.ones(len(work), dtype=bool)

    is_better = (work["choice"].to_numpy() == work["_ref"].to_numpy())
    valid = work["choice"].notna().to_numpy() & in_phase
    work["_better"] = np.where(valid, is_better.astype(float), np.nan)

    # Per-block better-side fraction over the phase (misses already dropped).
    blk = work[valid].groupby(key, sort=False)
    per_block = blk.agg(
        L=("_lr", "first"),
        n=("_better", "size"),
        p=("_better", "mean"),
    ).reset_index(drop=True)
    per_block = per_block[per_block["n"] >= min_trials]

    # Aggregate across blocks within each L_Random length.
    rows = []
    for L, sub in per_block.groupby("L", sort=True):
        if len(sub) < min_blocks:
            continue
        p  = sub["p"].to_numpy()
        nn = sub["n"].to_numpy()
        var_raw = float(np.var(p, ddof=1))
        sd_raw  = float(np.sqrt(var_raw))
        v_bin   = float(np.mean(p * (1.0 - p) / nn))   # mean within-block binomial var
        sd_corr = float(np.sqrt(max(0.0, var_raw - v_bin)))
        rows.append((int(L), sd_raw, sd_corr, float(np.mean(p)), int(len(sub))))

    res = pd.DataFrame(rows, columns=["L", "sd_raw", "sd_corr", "mean_p", "n_blocks"])

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=130)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if res.empty:
        warnings.warn("lrandom_choice_variability_figure: no L_Random length "
                      f"reached min_blocks={min_blocks}; nothing to plot.")
        ax.text(0.5, 0.5, "No L_Random length met the thresholds",
                ha="center", va="center", transform=ax.transAxes)
        if save and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / f"lrandom_variability_bylength_{phase}.png",
                        dpi=150, bbox_inches="tight")
        return fig, res

    if show_counts:
        ax2 = ax.twinx()
        ax2.bar(res["L"], res["n_blocks"], width=0.85, color="0.88", zorder=0)
        ax2.set_ylabel("Blocks per L_Random length", color="0.6", fontsize=11)
        ax2.tick_params(axis="y", colors="0.6")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("0.6")
        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)         # let the bars show through

    ax.plot(res["L"], res["sd_raw"], "-o", color="#C0531F", lw=2.3, ms=5,
            label="SD across blocks (raw)", zorder=3)
    if correct_sampling_noise:
        ax.plot(res["L"], res["sd_corr"], "--s", color="#3B0A6B", lw=2.0, ms=4,
                label="SD corrected (binomial noise removed)", zorder=3)

    ax.set_xlabel("L_Random length (trials)", fontsize=13)
    ax.set_ylabel("SD of better-side fraction across blocks", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()

    if save and output_dir:
        out = Path(output_dir) / f"lrandom_variability_bylength_{phase}.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig, res


def alternation_by_lrandom_length_figure(
    df: pd.DataFrame,
    start_j: int = 1,
    max_lrandom: int = 30,
    min_blocks: int = 15,
    min_pairs: int = 1,
    ylim=None,
    show_criterion_baseline: bool = True,
    show_counts: bool = True,
    output_dir: str = "figs",
    save: bool = True,
    title: str = "",
):
    """
    Mean trial-to-trial alternation during L_Random as a function of L_Random length.

    For every valid block, the alternation rate over its L_Random phase is computed
    (fraction of consecutive trial pairs whose choice differs, both non-miss).
    Each block is collapsed to that single rate, blocks are grouped by L_Random
    length, and the *mean across blocks* (± SEM across blocks) is plotted against
    L_Random length.

    Why this avoids the binomial-noise artifact of the between-block SD: the
    per-block rate is an unbiased estimate of that block's switch probability, so
    the across-block mean is unbiased regardless of how many trials each block has.
    The (smaller) per-block sample size at short L_Random only widens the SEM — it
    does not shift the point estimate. Note this measures *within-block* choice
    variability (how much the animal flips), which is conceptually distinct from
    *between-block* heterogeneity.

    A block contributes only if it reached criterion (ttc / L_Random non-NaN), has
    1 <= L_Random <= max_lrandom, its row count equals ttc + L_Random, and it has
    at least `min_pairs` valid consecutive pairs in its L_Random phase. L_Random
    lengths with fewer than `min_blocks` contributing blocks are dropped.

    Parameters
    ----------
    start_j : first L_Random position contributing a pair. 1 (default) = pairs
        strictly within L_Random (each L_Random trial vs the previous L_Random
        trial). Use 0 to also count the first L_Random trial vs the last criterion
        trial of the same block. With start_j=1, blocks with L_Random == 1 have no
        within-phase pair and are dropped.
    min_blocks : minimum contributing blocks for an L_Random length to be plotted.
    min_pairs : minimum valid consecutive pairs in the L_Random phase for a block.
    ylim : optional y-axis limits (alternation sits well below the 0.5 chance level).
    show_criterion_baseline : draw the mean within-criterion alternation as a
        horizontal reference line.
    show_counts : draw the number of contributing blocks per length on a faint
        secondary axis.
    output_dir, save : where/whether to write the PNG.

    Returns
    -------
    (fig, res) : the Matplotlib figure and a DataFrame with one row per plotted
        L_Random length (columns: L, mean_alt, sem_alt, n_blocks, mean_pairs).
    """
    key = ["animal", "session_file", "block_idx"]
    work = df.copy()
    work["_t"] = work.groupby(key, sort=False).cumcount()
    grp = work.groupby(key, sort=False)
    ttc = grp["block_trial_to_crit"].transform("first")
    lr  = grp["block_trial_random_added"].transform("first")
    n   = grp["_t"].transform("size")

    keep = (ttc.notna() & lr.notna() & (lr >= 1) & (lr <= max_lrandom)
            & (n == (ttc + lr)))
    work = work[keep].copy()
    work["_ttc"] = ttc[keep].astype(int)
    work["_lr"]  = lr[keep].astype(int)

    # Previous choice within the block. Any row with _t >= 1 is not the block's
    # first row, so a plain shift stays inside the same block for those rows.
    work["_choice_prev"] = work["choice"].shift(1)
    t  = work["_t"].to_numpy()
    tc = work["_ttc"].to_numpy()
    alt = (work["choice"].to_numpy() != work["_choice_prev"].to_numpy()).astype(float)
    valid = (work["choice"].notna().to_numpy()
             & work["_choice_prev"].notna().to_numpy()
             & (t >= 1))
    work["_j"]   = work["_t"] - work["_ttc"]
    work["_alt"] = alt

    # Criterion-phase baseline: alternation on criterion rows with _t >= 1.
    baseline = np.nan
    if show_criterion_baseline:
        crit = (t >= 1) & (t < tc) & valid
        if crit.any():
            baseline = float(np.mean(alt[crit]))

    # L_Random pairs, then collapse each block to its alternation rate.
    is_lr = t >= tc
    sel = is_lr & valid & (work["_j"].to_numpy() >= start_j)
    lrdf = work[sel]
    per_block = lrdf.groupby(key, sort=False).agg(
        L=("_lr", "first"),
        n_pairs=("_alt", "size"),
        alt=("_alt", "mean"),
    ).reset_index(drop=True)
    per_block = per_block[per_block["n_pairs"] >= min_pairs]

    # Mean ± SEM across blocks within each L_Random length.
    rows = []
    for L, sub in per_block.groupby("L", sort=True):
        if len(sub) < min_blocks:
            continue
        a = sub["alt"].to_numpy()
        mean_alt = float(np.mean(a))
        sem_alt  = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan
        rows.append((int(L), mean_alt, sem_alt, int(len(sub)), float(sub["n_pairs"].mean())))

    res = pd.DataFrame(rows, columns=["L", "mean_alt", "sem_alt", "n_blocks", "mean_pairs"])

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=130)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if res.empty:
        warnings.warn("alternation_by_lrandom_length_figure: no L_Random length "
                      f"reached min_blocks={min_blocks}; nothing to plot.")
        ax.text(0.5, 0.5, "No L_Random length met the thresholds",
                ha="center", va="center", transform=ax.transAxes)
        if save and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / "lrandom_alternation_bylength.png",
                        dpi=150, bbox_inches="tight")
        return fig, res

    if show_counts:
        ax2 = ax.twinx()
        ax2.bar(res["L"], res["n_blocks"], width=0.85, color="0.88", zorder=0)
        ax2.set_ylabel("Blocks per L_Random length", color="0.6", fontsize=11)
        ax2.tick_params(axis="y", colors="0.6")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("0.6")
        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

    ax.errorbar(res["L"], res["mean_alt"], yerr=res["sem_alt"],
                fmt="-o", color="#C0531F", lw=2.3, ms=5, capsize=3,
                ecolor="#C0531F", elinewidth=1.4,
                label="Mean alternation across blocks (± SEM)", zorder=3)

    if show_criterion_baseline and np.isfinite(baseline):
        ax.axhline(baseline, ls="--", color="0.5", lw=1.4, zorder=2,
                   label=f"criterion baseline ({baseline:.2f})")

    ax.set_xlabel("L_Random length (trials)", fontsize=13)
    ax.set_ylabel("P(alternation)   choice ≠ previous", fontsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()

    if save and output_dir:
        out = Path(output_dir) / "lrandom_alternation_bylength.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig, res


def alternation_after_correct_figure(
    df: pd.DataFrame,
    phase: str = "block",
    pair: str = "within",
    unit: str = "animal",
    min_trials: int = 20,
    show_stat: bool = True,
    stat: str = "wilcoxon",
    title: str = "",
    output_dir: str = "figs",
    save: bool = True,
):
    """
    Next-trial alternation after a correct (better-option) choice, split by reward.

    Conditions on trials where the animal chose the better option (choice == hr_side)
    and a within-block next trial exists, then compares P(alternation on the next
    trial) between trials that were rewarded vs unrewarded. Under the 70:10 schedule a
    correct choice is unrewarded on ~30% of trials, so this isolates the effect of
    reward feedback (win-stay vs lose-shift) while holding the chosen option fixed.

    Aggregation is per `unit` (animal by default): P(switch) is computed for each unit
    in each reward condition, and the figure shows the mean across units (± SEM) with
    the paired per-unit values connected. Units with fewer than `min_trials`
    qualifying trials in either condition are dropped.

    Parameters
    ----------
    phase : {"block", "criterion", "lrandom"}
        Which trials count as the *current* trial. "block" (default) uses the whole
        block (the better side is constant within a block, so this assumes the 70:10
        schedule holds until the reversal). Use "criterion" to restrict to the
        structured phase before L_Random.
    pair : {"within", "across"}
        Which (current, next) trial pairs to use. "within" (default) pairs a trial
        with the next trial in the *same* block (lose-shift within a stable block;
        `phase` selects which current trials count). "across" pairs the *last* trial
        of a block with the *first* trial of the next block -- the choice straddling a
        reversal (`phase` is ignored in this mode).
    unit : {"animal", "session"}
        Aggregation unit for the paired comparison.
    min_trials : minimum qualifying trials per condition for a unit to be included.
    show_stat : annotate the statistical test on the figure.
    stat : {"wilcoxon", "glmm"}
        Which test to annotate. "wilcoxon" (default) is the paired Wilcoxon
        signed-rank across units (each animal contributes one paired value, so
        within-animal correlation is handled by aggregation). "glmm" (alias
        "gee") instead fits a trial-level logistic model with the reward
        condition as predictor, clustered by animal (statsmodels GEE, exchangeable
        working correlation, cluster-robust SEs), which accounts for within-animal
        correlation directly and uses all trials; requires statsmodels.
    title, output_dir, save : as in the other figures (title default is no title).

    Returns
    -------
    (fig, res) : the figure and a per-unit DataFrame with columns
        [unit keys, p_rew, p_unr, n_rew, n_unr].
    """
    if phase not in ("block", "criterion", "lrandom"):
        raise ValueError("phase must be 'block', 'criterion' or 'lrandom'")
    if unit not in ("animal", "session"):
        raise ValueError("unit must be 'animal' or 'session'")
    if pair not in ("within", "across"):
        raise ValueError("pair must be 'within' or 'across'")
    if stat not in ("wilcoxon", "glmm", "gee"):
        raise ValueError("stat must be 'wilcoxon', 'glmm' or 'gee'")

    key  = ["animal", "session_file", "block_idx"]
    sess = ["animal", "session_file"]
    work = df.copy()
    work["_t"] = work.groupby(key, sort=False).cumcount()
    grp = work.groupby(key, sort=False)
    ttc   = grp["block_trial_to_crit"].transform("first")
    lr    = grp["block_trial_random_added"].transform("first")
    nrows = grp["_t"].transform("size")

    # The current trial's block must have reached criterion and be complete.
    blk_valid = (ttc.notna() & lr.notna() & (lr >= 1)
                 & (nrows == (ttc + lr))).to_numpy()
    t  = work["_t"].to_numpy()
    tc = ttc.fillna(-1).to_numpy()

    if pair == "within":
        # Pair each trial with the next trial in the SAME block (never crosses a
        # reversal; `phase` selects which current trials count).
        work["_next"] = work.groupby(key, sort=False)["choice"].shift(-1)
        if phase == "criterion":
            in_phase = t < tc
        elif phase == "lrandom":
            in_phase = t >= tc
        else:
            in_phase = np.ones(len(work), dtype=bool)
        scope = blk_valid & in_phase
    else:  # "across": last trial of a block paired with first of the next block
        work["_next"] = work.groupby(sess, sort=False)["choice"].shift(-1)
        nb = work.groupby(sess, sort=False)["block_idx"].shift(-1)
        is_boundary = (nb.notna() & (nb != work["block_idx"])).to_numpy()
        scope = blk_valid & is_boundary

    chose_better = work["choice"].to_numpy() == work["hr_side"].to_numpy()
    cur_ok   = work["choice"].notna().to_numpy() & work["rewarded"].notna().to_numpy()
    next_ok  = work["_next"].notna().to_numpy()
    rewarded = work["rewarded"].to_numpy() == 1
    switch_next = (work["_next"].to_numpy() != work["choice"].to_numpy()).astype(float)

    base = scope & chose_better & cur_ok & next_ok
    sub = work[base].copy()
    sub["_switch"] = switch_next[base]
    sub["_cond"] = np.where(rewarded[base], "rewarded", "unrewarded")

    ucols = ["animal"] if unit == "animal" else ["animal", "session_file"]
    agg = (sub.groupby(ucols + ["_cond"])["_switch"]
              .agg(n="size", p="mean").reset_index())
    p_r = agg[agg["_cond"] == "rewarded"].set_index(ucols)["p"]
    p_u = agg[agg["_cond"] == "unrewarded"].set_index(ucols)["p"]
    n_r = agg[agg["_cond"] == "rewarded"].set_index(ucols)["n"]
    n_u = agg[agg["_cond"] == "unrewarded"].set_index(ucols)["n"]
    res = pd.DataFrame({"p_rew": p_r, "p_unr": p_u, "n_rew": n_r, "n_unr": n_u})
    res = res.dropna(subset=["p_rew", "p_unr"])
    res = res[(res["n_rew"] >= min_trials) & (res["n_unr"] >= min_trials)].reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=130)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if res.empty:
        warnings.warn("alternation_after_correct_figure: no unit met "
                      f"min_trials={min_trials} in both conditions; nothing to plot.")
        ax.text(0.5, 0.5, "No unit met the thresholds",
                ha="center", va="center", transform=ax.transAxes)
        if save and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / f"alternation_after_correct_{pair}.png",
                        dpi=150, bbox_inches="tight")
        return fig, res

    x = [0, 1]
    col_r, col_u = "#3E7CB1", "#C0432B"

    # Paired per-unit lines (faint) + individual points.
    for _, row in res.iterrows():
        ax.plot(x, [row["p_rew"], row["p_unr"]], "-", color="0.75",
                lw=0.8, alpha=0.7, zorder=1)
    ax.plot(np.zeros(len(res)), res["p_rew"], "o", color=col_r, ms=4, alpha=0.5, zorder=2)
    ax.plot(np.ones(len(res)),  res["p_unr"], "o", color=col_u, ms=4, alpha=0.5, zorder=2)

    # Mean ± SEM across units.
    m_r, m_u = res["p_rew"].mean(), res["p_unr"].mean()
    s_r = res["p_rew"].std(ddof=1) / np.sqrt(len(res))
    s_u = res["p_unr"].std(ddof=1) / np.sqrt(len(res))
    ax.errorbar(0, m_r, yerr=s_r, fmt="o", color=col_r, ms=11, capsize=5,
                elinewidth=2, zorder=4)
    ax.errorbar(1, m_u, yerr=s_u, fmt="o", color=col_u, ms=11, capsize=5,
                elinewidth=2, zorder=4)
    ax.plot(x, [m_r, m_u], "-", color="0.3", lw=1.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["Rewarded\n(chose better)", "Unrewarded\n(chose better)"],
                       fontsize=11)
    ax.set_xlim(-0.4, 1.4)
    ylab = ("P(switch on first trial of next block)" if pair == "across"
            else "P(alternation on next trial)")
    ax.set_ylabel(ylab, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    ymax = max(res["p_rew"].max(), res["p_unr"].max(), m_r + s_r, m_u + s_u)
    ax.set_ylim(0, ymax * 1.25)

    if show_stat:
        annot, use = None, stat
        if stat in ("glmm", "gee"):
            # Trial-level logistic model, clustered by animal (cluster-robust SEs).
            try:
                import statsmodels.formula.api as smf
                from statsmodels.genmod.families import Binomial
                from statsmodels.genmod.cov_struct import Exchangeable
                d = sub[sub["animal"].isin(res["animal"])].copy()
                d["y"] = d["_switch"].astype(int)
                d["cond01"] = (d["_cond"] == "unrewarded").astype(int)
                gee = smf.gee("y ~ cond01", groups="animal", data=d,
                              family=Binomial(), cov_struct=Exchangeable()).fit()
                beta, pval = gee.params["cond01"], gee.pvalues["cond01"]
                annot = (f"Logistic GEE (clustered by animal): "
                         f"OR = {np.exp(beta):.2f}, p = {pval:.3g}")
            except ImportError:
                warnings.warn("stat='glmm'/'gee' requires statsmodels "
                              "(pip install statsmodels); falling back to Wilcoxon.")
                use = "wilcoxon"
            except Exception as e:
                warnings.warn(f"GEE fit failed ({e}); falling back to Wilcoxon.")
                use = "wilcoxon"
        if annot is None and use == "wilcoxon" and len(res) >= 6:
            from scipy import stats as _stats
            try:
                _, pval = _stats.wilcoxon(res["p_unr"], res["p_rew"])
                annot = f"Wilcoxon p = {pval:.3g}  (n = {len(res)} {unit}s)"
            except Exception:
                annot = None
        if annot:
            ax.text(0.5, ymax * 1.13, annot, ha="center", va="bottom",
                    fontsize=9.5, color="0.25")

    fig.tight_layout()
    n_pool_r = int((base & rewarded).sum())
    n_pool_u = int((base & ~rewarded).sum())
    print(f"[alternation_after_correct] units={len(res)} | pooled correct trials: "
          f"rewarded={n_pool_r}, unrewarded={n_pool_u}")

    if save and output_dir:
        out = Path(output_dir) / f"alternation_after_correct_{pair}.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig, res


# ===========================================================================
# 7. Model simulation
# ===========================================================================

# Models that simulate_sessions knows how to simulate.
_SIM_MODELS = ("belief", "belief_ck", "belief_vhr")

# df columns a simulation reads; the cache is invalidated if any of these change.
_SIM_DATA_COLS = [
    "animal", "session_file", "choice", "rewarded", "rule", "hr_side",
    "n_rules", "block_trial_to_crit",
]


def _sim_model_rng(seed: int, model_name: str) -> np.random.Generator:
    """Deterministic, model-specific RNG.

    Each model gets its own independent stream derived from (seed, model_name),
    so a model's simulation is reproducible regardless of which other models are
    re-simulated or loaded from cache in the same call.
    """
    h = int(hashlib.sha256(model_name.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng(np.random.SeedSequence([int(seed), h]))


def _sim_fingerprint(model_name: str, df: pd.DataFrame, fit_df: pd.DataFrame,
                     n_sims: int, seed: int) -> str:
    """Hash of everything a model's simulation depends on.

    Combines the model's fitted parameters, the session data it reads, n_sims and
    seed. Any change here (e.g. re-fitting the model, editing the analysis CSV,
    changing n_sims/seed) yields a different hash and so invalidates the cache for
    that model only.
    """
    h = hashlib.sha256()
    h.update(f"{model_name}|n_sims={n_sims}|seed={seed}".encode())

    # Fitted parameters for this model (re-fitting invalidates the cache).
    par_cols = [c for c in fit_df.columns if c.startswith("par_")]
    fit_cols = [c for c in (["animal", "session_file", "model"] + par_cols)
                if c in fit_df.columns]
    fit_sub = (fit_df.loc[fit_df["model"] == model_name, fit_cols]
                     .sort_values(["animal", "session_file"]))
    h.update(fit_sub.to_csv(index=False).encode())

    # Session data the simulation reads (changed data invalidates the cache).
    data_cols = [c for c in _SIM_DATA_COLS if c in df.columns]
    data_sub = df.loc[:, data_cols].sort_values(["animal", "session_file"])
    h.update(data_sub.to_csv(index=False).encode())

    return h.hexdigest()


def _simulate_one_model(model_name: str, df: pd.DataFrame, fit_df: pd.DataFrame,
                        n_sims: int, rng: np.random.Generator) -> list:
    """Simulate a single model across every session that has a fit for it."""
    sessions = []
    for (animal, ses_file), df_ses in df.groupby(["animal", "session_file"], sort=False):
        row = fit_df[
            (fit_df["animal"] == animal) &
            (fit_df["session_file"] == ses_file) &
            (fit_df["model"] == model_name)
        ]
        if row.empty:
            continue

        c_real  = df_ses["choice"].values.astype(float)
        r_real  = df_ses["rewarded"].values.astype(float)
        rule    = df_ses["rule"].values.astype(float)
        hr_side = df_ses["hr_side"].values.astype(float)

        try:
            if model_name == "belief":
                result = simulate_belief(
                    c_real, r_real, rule,
                    H=float(row["par_H"].iloc[0]),
                    beta=float(row["par_beta"].iloc[0]),
                    n_sims=n_sims, rng=rng,
                )
            elif model_name == "belief_ck":
                result = simulate_belief_ck(
                    c_real, r_real, rule,
                    H=float(row["par_H"].iloc[0]),
                    beta=float(row["par_beta"].iloc[0]),
                    alpha_k=float(row["par_alpha_k"].iloc[0]),
                    beta_k=float(row["par_beta_k"].iloc[0]),
                    n_sims=n_sims, rng=rng,
                )
            else:  # belief_vhr — needs tau (trials since criterion)
                tau = prepare_vhr(df_ses)[3]
                result = simulate_belief_vhr(
                    c_real, r_real, rule, tau,
                    a=float(row["par_a"].iloc[0]),
                    b=float(row["par_b"].iloc[0]),
                    beta=float(row["par_beta"].iloc[0]),
                    n_sims=n_sims, rng=rng,
                )
            sessions.append({
                "animal":       animal,
                "session_file": ses_file,
                "c_sim":        result["c_sim"],
                "hr_side":      hr_side,
            })
        except Exception as e:
            warnings.warn(f"Simulation failed ({model_name}, {ses_file}): {e}")

    return sessions


def simulate_sessions(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    n_sims: int = 100,
    seed: int = 42,
    cache_dir="analysis/simulations",
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Simulate choices under each fitted model for every session, with a per-model
    on-disk cache so already-simulated models are not re-run.

    For each model, a fingerprint is computed from its fitted parameters, the
    session data it reads, n_sims and seed. If a cache file with a matching
    fingerprint exists, the model is loaded and skipped; otherwise it is simulated
    and the result cached. This means re-fitting a model (or changing the data /
    n_sims / seed) automatically invalidates *only* that model's cache.

    Parameters
    ----------
    df, fit_df : DataFrames
        Behaviour CSV and model fits (model_fits.csv).
    n_sims : int
        Simulated runs per session.
    seed : int
        Base seed; each model derives an independent, reproducible RNG from it.
    cache_dir : path-like or None
        Directory for the per-model caches. None disables caching entirely
        (always simulate, never read or write).
    force : bool
        If True, ignore existing caches and re-simulate every model.
    verbose : bool
        Print a one-line status per model (cache hit / stale / simulating).

    Returns
    -------
    dict keyed by model name ("belief", "belief_ck", "belief_vhr"), each value a
    list of dicts with keys: animal, session_file, c_sim (n_trials × n_sims),
    hr_side. Models with no fits in fit_df map to an empty list.
    """
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    fitted_models = set(pd.unique(fit_df["model"]))
    sim = {m: [] for m in _SIM_MODELS}

    for model_name in _SIM_MODELS:
        if model_name not in fitted_models:
            if verbose:
                print(f"[simulate_sessions] '{model_name}': no fits -> skipping")
            continue

        fingerprint = _sim_fingerprint(model_name, df, fit_df, n_sims, seed)
        cfile = cache_path / f"sim_{model_name}.joblib" if cache_path is not None else None

        # --- Try the cache ---
        if cfile is not None and not force and cfile.exists():
            try:
                cached = joblib.load(cfile)
            except Exception as e:
                cached = None
                warnings.warn(f"Could not read sim cache for '{model_name}': {e}")
            if cached is not None and cached.get("fingerprint") == fingerprint:
                sim[model_name] = cached["sessions"]
                if verbose:
                    print(f"[simulate_sessions] '{model_name}': cache hit "
                          f"({len(sim[model_name])} sessions) -> skipping")
                continue
            elif cached is not None and verbose:
                print(f"[simulate_sessions] '{model_name}': cache stale "
                      f"(fits/data/n_sims/seed changed) -> re-simulating")

        # --- (Re-)simulate this model ---
        if verbose:
            print(f"[simulate_sessions] '{model_name}': simulating "
                  f"(n_sims={n_sims})...")
        rng = _sim_model_rng(seed, model_name)
        sessions = _simulate_one_model(model_name, df, fit_df, n_sims, rng)
        sim[model_name] = sessions

        if cfile is not None:
            try:
                joblib.dump(
                    {"model": model_name, "fingerprint": fingerprint,
                     "n_sims": n_sims, "seed": seed, "sessions": sessions},
                    cfile, compress=3,
                )
            except Exception as e:
                warnings.warn(f"Could not write sim cache for '{model_name}': {e}")

    return sim