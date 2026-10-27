"""
lesion_modelfree.py
===================
Steps 0 and 1 of the lesion re-analysis: data audit and model-free behaviour,
with the ANIMAL as the statistical unit.

Groups (preprocessing/lesion_index.py, LESION_SIDE): 1 = left ACAd/MOs lesion,
2 = right, 3 = bilateral, 4 = saline (unilateral injection; side not recorded).
Left and right are pooled as "unilateral". Every lesioned/sham animal has
sessions before (pre) and after (post) surgery.

Why a new analysis (the preprint version is not robust enough):
  - sessions were treated as independent replicates (pseudoreplication);
  - lesion groups were never contrasted with the saline group (an effect being
    significant in one group and not in another is not a difference);
  - the session criterion differed from the one stated in the text.

Design here
  Step 0 (audit): sessions, sessions passing the criterion, trials per session
    and miss rate, per group x phase, to expose selection effects (e.g. the
    bilateral group performs fewer trials).
  Step 1 (behaviour): per-session metrics -> per-animal pre and post means ->
    per-animal change (post - pre). Then, with exact permutation tests:
      (a) within group: is the mean change different from 0?   (sign flips)
      (b) vs saline: is the change different from the saline change?
          (group labels permuted among the animals of the two groups)
      (c) unilateral only: is the change different for blocks whose better
          side is contralateral vs ipsilateral to the lesion?   (sign flips)
    Exact permutation p-values have a floor set by n (4 vs 4 animals: 1/35 for
    a two-sided test), which is the honest resolution of this dataset.

Metrics (pre-switch L_Random window = 0 <= tau < L_Random, complete blocks,
last block of each session dropped; tau = trials since criterion):
  n_trials, miss_rate, p_better (L_Random), p_better_early (tau 0-3),
  p_better_late (tau >= 6), late_minus_early (anticipation proxy),
  trials_to_crit (blocks after the first of the session), win_stay, lose_shift.
  Unilateral animals also get p_ipsi (choice bias toward the lesioned side) and
  the block-type split: blocks whose better side is ipsi vs contra to lesion.

Run from reversalExpectation_py (revExp env):
    python -X utf8 behavior\\lesion\\lesion_modelfree.py > analysis\\lesion_step01.log 2>&1
    python -X utf8 behavior\\lesion\\lesion_modelfree.py --all-sessions   (sensitivity: no criterion)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_DIR = ROOT / "analysis" / "lesion"
FIG_DIR = ROOT / "figs" / "lesion"

GROUP = {1.0: "unilateral", 2.0: "unilateral", 3.0: "bilateral", 4.0: "saline"}
SIDE_SIGN = {1.0: -1.0, 2.0: 1.0}      # lesioned side in choice coding (-1 = left, +1 = right)
EARLY_TAU, LATE_TAU = 3, 6             # early = tau 0..3, late = tau >= 6
METRICS = ["n_trials", "miss_rate", "p_better", "p_better_early", "p_better_late",
           "late_minus_early", "trials_to_crit", "win_stay", "lose_shift"]
SIDE_METRICS = ["p_better", "trials_to_crit", "win_stay", "lose_shift"]
MAX_EXACT = 200_000                    # beyond this many relabelings, use Monte Carlo
N_MC = 100_000
RNG = np.random.default_rng(20261025)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(all_sessions: bool) -> pd.DataFrame:
    cols = ["animal", "session_file", "phase", "trial_idx", "choice", "rewarded", "correct",
            "hr_side", "block_idx", "block_trial_to_crit", "block_trial_random_added",
            "lesioned", "lesion_side", "meets_criteria"]
    df = pd.read_csv(CSV_PATH, usecols=cols)
    df = df[df["lesion_side"].notna()].copy()            # the 17 operated animals only
    df["group"] = df["lesion_side"].map(GROUP)
    df["phase_ls"] = np.where(df["lesioned"].notna(), "post", "pre")
    df["keep"] = True if all_sessions else (df["meets_criteria"] == True)  # noqa: E712
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Step 0: what enters the analysis, per group x phase."""
    ses = df.groupby("session_file").agg(
        animal=("animal", "first"), group=("group", "first"), phase_ls=("phase_ls", "first"),
        task_phase=("phase", "first"), keep=("keep", "first"),
        n_trials=("trial_idx", "size"), miss_rate=("choice", lambda c: c.isna().mean()))
    out = ses.groupby(["group", "phase_ls"]).agg(
        animals=("animal", "nunique"), sessions=("keep", "size"),
        sessions_kept=("keep", "sum"), trials_median=("n_trials", "median"),
        miss_rate_median=("miss_rate", "median"))
    out["frac_kept"] = out["sessions_kept"] / out["sessions"]
    out["task_phase8_sessions"] = ses[ses["task_phase"] == 8].groupby(["group", "phase_ls"]).size()
    return out.fillna({"task_phase8_sessions": 0})


# ---------------------------------------------------------------------------
# Per-session metrics
# ---------------------------------------------------------------------------
def _ws_ls(d: pd.DataFrame) -> tuple[float, float]:
    """Win-stay and lose-shift over consecutive responded trials."""
    c = d["choice"].to_numpy()
    r = d["rewarded"].to_numpy()
    prev_c, cur_c, prev_r = c[:-1], c[1:], r[:-1]
    ok = np.isfinite(prev_c) & np.isfinite(cur_c) & np.isfinite(prev_r)
    stay = (cur_c == prev_c)
    win = ok & (prev_r == 1)
    lose = ok & (prev_r == 0)
    ws = stay[win].mean() if win.sum() >= 5 else np.nan
    ls = (~stay[lose]).mean() if lose.sum() >= 5 else np.nan
    return ws, ls


def session_metrics(d: pd.DataFrame, side_sign: float | None) -> dict:
    d = d.sort_values("trial_idx")
    out = {"n_trials": len(d), "miss_rate": d["choice"].isna().mean()}
    # position within block and tau
    d = d.assign(pos=d.groupby("block_idx").cumcount())
    d["tau"] = d["pos"] - d["block_trial_to_crit"]
    last_block = d["block_idx"].max()
    in_lr = ((d["tau"] >= 0) & d["block_trial_random_added"].notna()
             & (d["tau"] < d["block_trial_random_added"]) & (d["block_idx"] < last_block)
             & d["choice"].notna())
    lr = d[in_lr]
    out["p_better"] = lr["correct"].mean() if len(lr) else np.nan
    e = lr[lr["tau"] <= EARLY_TAU]["correct"]
    l = lr[lr["tau"] >= LATE_TAU]["correct"]
    out["p_better_early"] = e.mean() if len(e) >= 5 else np.nan
    out["p_better_late"] = l.mean() if len(l) >= 5 else np.nan
    out["late_minus_early"] = out["p_better_late"] - out["p_better_early"]
    blocks = d.groupby("block_idx").agg(ttc=("block_trial_to_crit", "first"),
                                        hr=("hr_side", "first"))
    blocks = blocks[blocks.index > blocks.index.min()]       # first block: no prior rule
    out["trials_to_crit"] = blocks["ttc"].mean() if len(blocks) else np.nan
    out["win_stay"], out["lose_shift"] = _ws_ls(d)

    if side_sign is not None:                                 # unilateral: ipsi / contra split
        resp = d[d["choice"].notna()]
        out["p_ipsi"] = (resp["choice"] == side_sign).mean()
        for name, mask_hr in [("ipsi", d["hr_side"] == side_sign),
                              ("contra", d["hr_side"] == -side_sign)]:
            sub_lr = lr[lr["hr_side"] == (side_sign if name == "ipsi" else -side_sign)]
            out[f"p_better_{name}"] = sub_lr["correct"].mean() if len(sub_lr) >= 5 else np.nan
            b = blocks[blocks["hr"] == (side_sign if name == "ipsi" else -side_sign)]
            out[f"trials_to_crit_{name}"] = b["ttc"].mean() if len(b) else np.nan
            ws, ls = _ws_ls(d[mask_hr])
            out[f"win_stay_{name}"], out[f"lose_shift_{name}"] = ws, ls
    return out


def all_session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ses, d in df[df["keep"]].groupby("session_file"):
        ls = d["lesion_side"].iloc[0]
        m = session_metrics(d, SIDE_SIGN.get(ls))
        m.update(session_file=ses, animal=str(d["animal"].iloc[0]), group=d["group"].iloc[0],
                 lesion_side=ls, phase_ls=d["phase_ls"].iloc[0])
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exact permutation tests (animal = unit)
# ---------------------------------------------------------------------------
def signflip_p(x: np.ndarray) -> float:
    """Two-sided exact sign-flip test of mean(x) = 0."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    obs = abs(x.mean())
    if 2 ** n <= MAX_EXACT:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = RNG.choice([-1, 1], size=(N_MC, n))
    null = np.abs((signs * x).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def twogroup_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact permutation test of mean(a) - mean(b) = 0."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.r_[a, b]
    n, k = len(pooled), len(a)
    obs = abs(a.mean() - b.mean())
    combos = itertools.combinations(range(n), k)
    from math import comb
    if comb(n, k) <= MAX_EXACT:
        diffs = []
        for idx in combos:
            m = np.zeros(n, bool)
            m[list(idx)] = True
            diffs.append(abs(pooled[m].mean() - pooled[~m].mean()))
        null = np.array(diffs)
    else:
        null = np.empty(N_MC)
        for i in range(N_MC):
            p = RNG.permutation(n)
            null[i] = abs(pooled[p[:k]].mean() - pooled[p[k:]].mean())
    return float(np.mean(null >= obs - 1e-12))


def animal_changes(sm: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-animal pre mean, post mean and change for each metric."""
    g = sm.groupby(["animal", "group", "lesion_side", "phase_ls"])[metrics].mean()
    pre = g.xs("pre", level="phase_ls")
    post = g.xs("post", level="phase_ls")
    pre, post = pre.align(post, join="inner")
    delta = post - pre
    return pd.concat({"pre": pre, "post": post, "delta": delta}, axis=1)


def run_tests(ch: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    d = ch["delta"].reset_index()
    sal = d[d["group"] == "saline"]
    for m in metrics:
        for grp in ["unilateral", "bilateral", "saline"]:
            x = d.loc[d["group"] == grp, m].to_numpy(float)
            row = {"metric": m, "group": grp, "n_animals": int(np.isfinite(x).sum()),
                   "pre_mean": ch["pre"].reset_index().loc[lambda t: t["group"] == grp, m].mean(),
                   "post_mean": ch["post"].reset_index().loc[lambda t: t["group"] == grp, m].mean(),
                   "delta_mean": np.nanmean(x), "p_within": signflip_p(x),
                   "n_delta_pos": int(np.sum(x > 0))}
            if grp != "saline":
                row["delta_minus_saline"] = np.nanmean(x) - np.nanmean(sal[m].to_numpy(float))
                row["p_vs_saline"] = twogroup_p(x, sal[m].to_numpy(float))
            rows.append(row)
    return pd.DataFrame(rows)


def side_tests(ch: pd.DataFrame) -> pd.DataFrame:
    """Unilateral: change in contra-better blocks vs ipsi-better blocks, within animal;
    and each side's change vs the (side-averaged) saline change."""
    d = ch["delta"].reset_index()
    uni = d[d["group"] == "unilateral"]
    sal = d[d["group"] == "saline"]
    rows = []
    for m in SIDE_METRICS:
        ci, ip = uni[f"{m}_contra"].to_numpy(float), uni[f"{m}_ipsi"].to_numpy(float)
        rows.append({"metric": m, "delta_contra": np.nanmean(ci), "delta_ipsi": np.nanmean(ip),
                     "contra_minus_ipsi": np.nanmean(ci - ip), "p_contra_vs_ipsi": signflip_p(ci - ip),
                     "p_contra_vs_saline": twogroup_p(ci, sal[m].to_numpy(float)),
                     "p_ipsi_vs_saline": twogroup_p(ip, sal[m].to_numpy(float)),
                     "n_animals": int(np.isfinite(ci - ip).sum())})
    x = uni["p_ipsi"].to_numpy(float)
    # change in the fraction of choices toward the lesioned side (sign flips)
    rows.append({"metric": "p_ipsi (choice bias)", "delta_contra": np.nan, "delta_ipsi": np.nanmean(x),
                 "contra_minus_ipsi": np.nan, "p_contra_vs_ipsi": signflip_p(x),
                 "p_contra_vs_saline": np.nan, "p_ipsi_vs_saline": np.nan,
                 "n_animals": int(np.isfinite(x).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure: per-animal pre -> post, one panel per metric
# ---------------------------------------------------------------------------
def make_figure(ch: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"unilateral": "#C9472B", "bilateral": "#2C5F2D", "saline": "#7A7A7A"}
    order = ["unilateral", "bilateral", "saline"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, m in zip(axes.ravel(), METRICS):
        for gi, grp in enumerate(order):
            pre_t, post_t = ch["pre"].reset_index(), ch["post"].reset_index()
            pre = pre_t.loc[pre_t["group"] == grp, m].to_numpy(float)
            post = post_t.loc[post_t["group"] == grp, m].to_numpy(float)
            x0, x1 = gi * 3, gi * 3 + 1
            for a, b in zip(pre, post):
                ax.plot([x0, x1], [a, b], color=colors[grp], alpha=0.5, lw=1.5)
            ax.plot([x0, x1], [np.nanmean(pre), np.nanmean(post)], color=colors[grp], lw=4)
        ax.set_xticks([0.5, 3.5, 6.5], ["uni", "bi", "saline"])
        ax.set_title(m)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Per-animal pre (left of each pair) vs post (right); thick = group mean")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    all_sessions = "--all-sessions" in sys.argv
    tag = "allsessions" if all_sessions else "criterion"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load(all_sessions)

    print(f"=== Step 0: data audit ({'no session criterion' if all_sessions else 'sessions with >= 4 switches'}) ===")
    au = audit(df)
    print(au.round(3).to_string())
    au.to_csv(OUT_DIR / f"step0_audit_{tag}.csv")

    sm = all_session_metrics(df)
    sm.to_csv(OUT_DIR / f"step1_session_metrics_{tag}.csv", index=False)
    # side-split columns (unilateral only); p_ipsi already ends with "_ipsi"
    side_cols = [c for c in sm.columns if c.endswith(("_ipsi", "_contra"))]
    ch = animal_changes(sm, METRICS + side_cols)
    ch.to_csv(OUT_DIR / f"step1_animal_changes_{tag}.csv")

    print("\n=== Step 1: per-animal change (post - pre), exact permutation tests ===")
    print("p_within: change != 0 in that group (sign flips) | p_vs_saline: change differs from saline")
    res = run_tests(ch, METRICS)
    res.to_csv(OUT_DIR / f"step1_tests_{tag}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))

    print("\n=== Step 1c: unilateral, blocks with the better side contra vs ipsi to the lesion ===")
    st = side_tests(ch)
    st.to_csv(OUT_DIR / f"step1_side_tests_{tag}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(st.round(4).to_string(index=False))

    make_figure(ch, FIG_DIR / f"step1_pre_post_{tag}.png")
    print(f"\nResults in {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
