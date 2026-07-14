"""
directed_switch.py
==================
Test 1 of the alternative anticipation tests (Undermind memo): a DIRECTIONAL
decomposition of switching inside the pre-switch L_Random window.

The raw anticipation curve, P(choose worse | tau), can be challenged as a
by-product of local win-stay/lose-shift plus block structure. This module asks
the sharper question: do mice become specifically more likely to move from the
currently BETTER option to the currently WORSE one as tau (trials since
criterion) grows? Genuine anticipation predicts:

    P(better -> worse | tau)  rises with tau
    P(worse  -> better | tau)  stays flatter

computed as CONDITIONAL rates (better->worse among trials whose previous choice
was the better option; worse->better among trials whose previous choice was the
worse option). Statistics: a per-animal slope of each conditional rate over tau
(one line per mouse), tested across mice with a Wilcoxon signed-rank test — the
same subject-level logic as anticipation_test.py. If statsmodels is available,
a pooled logistic GEE with recent reward/choice history controls (R_{t-1},
R_{t-2}, C_{t-1}) and a mouse-level correlation is added, so the tau effect is
reported after holding local history fixed.

A secondary distance-to-switch view (indexed by trials remaining until the
actual switch) is included as a VISUALISATION only: the animal cannot know how
many trials remain (L_Random is a truncated geometric), so a ramp in that view
is intuitive but is not causal evidence of anticipation on its own.

Better/worse are read from `hr_side` (the high-reward side, -1 left / +1 right)
vs `choice`: chose_better = (choice == hr_side).

Entry point: `run_directed_switch(df, output_dir, figs_dir, ...)`, meant to be
called from the RUN_PLOTS branch of master_bandit.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CREAM = "#FBFAF6"; INK = "#1A2E2A"; GREEN = "#2C5F2D"
CORAL = "#C9472B"; GOLD = "#E8A33D"; GRID = "#D8D5CC"; MUTE = "#6B6B66"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def prepare_directed(df: pd.DataFrame, drop_last_block: bool = True) -> pd.DataFrame:
    """
    Return the pre-switch L_Random, non-miss trials with the fields needed for
    the directional analysis, one row per trial:

        animal, tau, dtos, prev_better, now_better, now_worse,
        r1, r2, c1   (recent history: prev reward, prev-prev reward, prev choice)

    tau  = trials since criterion (within-block index - block_trial_to_crit),
           kept only where 0 <= tau < L_Random (the pre-switch random window).
    dtos = trials remaining until the switch = L_Random - 1 - tau.
    prev_better/now_better use hr_side; transitions are formed from consecutive
    NON-MISS trials within the same block's L_Random window.
    """
    need = ["animal", "session_file", "block_idx", "trial_idx", "choice",
            "hr_side", "block_trial_to_crit", "block_trial_random_added"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"directed_switch needs columns {missing}")

    d = df.sort_values(["animal", "session_file", "block_idx", "trial_idx"]).copy()
    d["wb"] = d.groupby(["animal", "session_file", "block_idx"]).cumcount()
    ttc = d["block_trial_to_crit"]
    lr = d["block_trial_random_added"]
    d["tau"] = d["wb"] - ttc
    d["lrandom"] = lr

    win = (d["tau"] >= 0) & (d["tau"] < lr) & ttc.notna() & lr.notna()
    d = d[win].copy()

    if drop_last_block:                      # the last block has no observed switch
        last = d.groupby(["animal", "session_file"])["block_idx"].transform("max")
        d = d[d["block_idx"] < last].copy()

    d["dtos"] = d["lrandom"] - 1.0 - d["tau"]          # trials remaining to switch

    # recent history BEFORE dropping misses (so r1/r2/c1 reflect real adjacency)
    g_all = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["r1"] = g_all["rewarded"].shift(1) if "rewarded" in d.columns else np.nan
    d["r2"] = g_all["rewarded"].shift(2) if "rewarded" in d.columns else np.nan
    d["c1"] = g_all["choice"].shift(1)

    # transitions from consecutive NON-MISS trials within the L_Random window
    d = d[d["choice"].notna()].copy()
    g = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["prev_choice"] = g["choice"].shift(1)
    d["prev_hr"] = g["hr_side"].shift(1)
    d = d.dropna(subset=["prev_choice", "prev_hr"])

    d["prev_better"] = (d["prev_choice"] == d["prev_hr"]).astype(int)
    d["now_better"] = (d["choice"] == d["hr_side"]).astype(int)
    d["now_worse"] = 1 - d["now_better"]
    return d


# ---------------------------------------------------------------------------
# Conditional curves + per-animal slopes
# ---------------------------------------------------------------------------
def _rate_curve(sub, outcome, max_tau, min_n):
    g = sub[sub["tau"] <= max_tau].groupby("tau")[outcome].agg(["mean", "size"])
    g = g[g["size"] >= min_n]
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    return g.index.to_numpy(float), g["mean"].to_numpy(), se.to_numpy(), g["size"].to_numpy()


def _per_animal_slopes(sub, outcome, max_tau, min_per_cell):
    """One slope of `outcome` vs tau per animal (>= min_per_cell trials/tau)."""
    out = {}
    for animal, a in sub.groupby("animal"):
        c = a[a["tau"] <= max_tau].groupby("tau")[outcome].agg(["mean", "size"])
        c = c[c["size"] >= min_per_cell]
        if len(c) >= 3:
            out[animal] = float(np.polyfit(c.index.to_numpy(float), c["mean"], 1)[0])
    return pd.Series(out, name=f"slope_{outcome}")


def directional_stats(d: pd.DataFrame, max_tau: int = 14, min_n: int = 200,
                      min_per_cell: int = 10):
    """Conditional curves and per-animal slope tests for both directions."""
    prev_better = d[d["prev_better"] == 1]     # was on the better option
    prev_worse = d[d["prev_better"] == 0]      # was on the worse option

    tb, pb, seb, nb = _rate_curve(prev_better, "now_worse", max_tau, min_n)
    tw, pw, sew, nw = _rate_curve(prev_worse, "now_better", max_tau, min_n)

    sl_bw = _per_animal_slopes(prev_better, "now_worse", max_tau, min_per_cell)
    sl_wb = _per_animal_slopes(prev_worse, "now_better", max_tau, min_per_cell)

    def _wilcox(s):
        s = s.dropna()
        if len(s) < 5:
            return dict(median=np.nan, frac_pos=np.nan, p=np.nan, n=len(s))
        # two-sided: is the slope distribution shifted from 0?
        W, p = stats.wilcoxon(s.values)
        return dict(median=float(s.median()), frac_pos=float((s > 0).mean()),
                    p=float(p), n=int(len(s)))

    return {
        "bw": {"tau": tb, "p": pb, "se": seb, "n": nb,
               "slopes": sl_bw, "stats": _wilcox(sl_bw)},
        "wb": {"tau": tw, "p": pw, "se": sew, "n": nw,
               "slopes": sl_wb, "stats": _wilcox(sl_wb)},
    }


def history_controlled_tau(d: pd.DataFrame, max_tau: int = 14):
    """
    Optional pooled logistic GEE for P(better -> worse) with recent-history
    controls, clustered by mouse (needs statsmodels). Returns the tau
    coefficient with 95% CI and p, or None if statsmodels is unavailable.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception:
        return None
    sub = d[(d["prev_better"] == 1) & (d["tau"] <= max_tau)].copy()
    sub = sub.dropna(subset=["r1", "r2", "c1"])
    if sub["animal"].nunique() < 5 or len(sub) < 500:
        return None
    try:
        m = smf.gee("now_worse ~ tau + r1 + r2 + c1", groups="animal", data=sub,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
        res = m.fit()
        ci = res.conf_int().loc["tau"].tolist()
        return {"beta_tau": float(res.params["tau"]), "ci": [float(ci[0]), float(ci[1])],
                "p": float(res.pvalues["tau"]), "n": int(len(sub))}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_directed_switch(res, hist=None, outfile="directed_switch.png"):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.patch.set_facecolor(CREAM)
    for a in ax:
        a.set_facecolor(CREAM)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(True, color=GRID, lw=0.6, alpha=0.6)

    # Panel A: the two conditional curves vs tau
    bw, wb = res["bw"], res["wb"]
    ax[0].fill_between(bw["tau"], bw["p"] - bw["se"], bw["p"] + bw["se"], color=GREEN, alpha=0.18)
    ax[0].plot(bw["tau"], bw["p"], "-o", color=GREEN, ms=4, lw=2, label="better → worse")
    ax[0].fill_between(wb["tau"], wb["p"] - wb["se"], wb["p"] + wb["se"], color=CORAL, alpha=0.18)
    ax[0].plot(wb["tau"], wb["p"], "-o", color=CORAL, ms=4, lw=2, label="worse → better")
    ax[0].set_xlabel("τ  =  trials since criterion  (L_Random window)", fontsize=10)
    ax[0].set_ylabel("transition probability", fontsize=10)
    ax[0].set_title("Directional switching vs τ", fontsize=12, weight="bold", color=INK)
    ax[0].legend(frameon=False, fontsize=9)
    sb, sw = bw["stats"], wb["stats"]
    txt = (f"better→worse: slope median {sb['median']:+.4f}, "
           f"{sb['frac_pos']*100:.0f}% >0, p={sb['p']:.3g} (n={sb['n']})\n"
           f"worse→better: slope median {sw['median']:+.4f}, "
           f"{sw['frac_pos']*100:.0f}% >0, p={sw['p']:.3g} (n={sw['n']})")
    if hist:
        txt += (f"\nhistory-controlled better→worse: β_τ={hist['beta_tau']:+.4f} "
                f"[{hist['ci'][0]:+.4f},{hist['ci'][1]:+.4f}], p={hist['p']:.3g}")
    ax[0].text(0.02, -0.30, txt, transform=ax[0].transAxes, fontsize=8.3,
               color=INK, va="top", family="monospace")

    # Panel B: per-animal slopes, both directions
    s_bw = bw["slopes"].dropna(); s_wb = wb["slopes"].dropna()
    for i, (s, col, lab) in enumerate([(s_bw, GREEN, "better→worse"),
                                       (s_wb, CORAL, "worse→better")]):
        x = np.full(len(s), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(s))
        ax[1].scatter(x, s.values, s=22, color=col, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax[1].hlines(s.median(), i - 0.2, i + 0.2, color=INK, lw=2)
    ax[1].axhline(0, color=MUTE, lw=1, ls="--")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["better→worse", "worse→better"], fontsize=10)
    ax[1].set_ylabel("per-mouse slope of transition rate vs τ", fontsize=10)
    ax[1].set_title("Per-mouse directional slopes", fontsize=12, weight="bold", color=INK)

    fig.suptitle("Directed-switch test of anticipation  (pre-switch L_Random)",
                 fontsize=13, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def plot_distance_to_switch(d: pd.DataFrame, max_d: int = 14, min_n: int = 200,
                            outfile="directed_switch_distance.png"):
    """Secondary VISUALISATION: rates indexed by trials remaining to the switch."""
    pb = d[d["prev_better"] == 1]
    sub = pb[pb["dtos"] <= max_d].groupby("dtos")["now_worse"].agg(["mean", "size"])
    sub = sub[sub["size"] >= min_n]
    se = np.sqrt(sub["mean"] * (1 - sub["mean"]) / sub["size"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    ax.fill_between(sub.index, sub["mean"] - se, sub["mean"] + se, color=GREEN, alpha=0.18)
    ax.plot(sub.index, sub["mean"], "-o", color=GREEN, ms=4, lw=2)
    ax.invert_xaxis()   # switch at 0 on the right
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax.set_xlabel("d  =  trials remaining until switch  (0 = switch)", fontsize=10)
    ax.set_ylabel("P(better → worse)", fontsize=10)
    ax.set_title("Distance-to-switch view (visualisation only)", fontsize=12,
                 weight="bold", color=INK)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Entry point for the RUN_PLOTS branch
# ---------------------------------------------------------------------------
def run_directed_switch(df: pd.DataFrame, output_dir: str = "analysis",
                        figs_dir: str = "figs", max_tau: int = 14,
                        min_n: int = 200, min_per_cell: int = 10,
                        distance_view: bool = True) -> dict:
    """
    Full directed-switch analysis, wired to the plots toggle. Prints a summary,
    writes the per-animal slope table to analysis/, the figure(s) to figs/, and
    returns the stats dict.
    """
    print("\n=== Directed-switch test (better→worse vs worse→better in L_Random) ===")
    d = prepare_directed(df)
    n_mice = d["animal"].nunique()
    print(f"  pre-switch L_Random transitions: {len(d)}  ·  {n_mice} mice")

    res = directional_stats(d, max_tau=max_tau, min_n=min_n, min_per_cell=min_per_cell)
    hist = history_controlled_tau(d, max_tau=max_tau)

    sb, sw = res["bw"]["stats"], res["wb"]["stats"]
    print(f"  better→worse : per-mouse slope median {sb['median']:+.4f}, "
          f"{sb['frac_pos']*100:.0f}% >0, Wilcoxon p={sb['p']:.3g} (n={sb['n']})")
    print(f"  worse→better : per-mouse slope median {sw['median']:+.4f}, "
          f"{sw['frac_pos']*100:.0f}% >0, Wilcoxon p={sw['p']:.3g} (n={sw['n']})")
    if hist:
        print(f"  history-controlled better→worse β_τ = {hist['beta_tau']:+.4f} "
              f"[{hist['ci'][0]:+.4f}, {hist['ci'][1]:+.4f}], p={hist['p']:.3g}")
    else:
        print("  (statsmodels not available -> skipped history-controlled GEE)")

    # per-animal slope table
    tbl = pd.concat([res["bw"]["slopes"], res["wb"]["slopes"]], axis=1).reset_index()
    tbl = tbl.rename(columns={"index": "animal"})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tbl.to_csv(Path(output_dir) / "directed_switch_per_animal.csv", index=False)

    fig = plot_directed_switch(res, hist=hist,
                               outfile=str(Path(figs_dir) / "directed_switch.png"))
    print(f"  wrote {fig}")
    if distance_view:
        figd = plot_distance_to_switch(d, max_d=max_tau,
                                       outfile=str(Path(figs_dir) / "directed_switch_distance.png"))
        print(f"  wrote {figd}")

    return {"stats": res, "history": hist, "n_mice": n_mice}