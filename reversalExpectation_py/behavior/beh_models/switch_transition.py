"""
switch_transition.py
====================
Is the change of GLM-HMM strategy time-locked to the block switch?

Two analyses over a window of +/- `win` trials around every switch (x = 0 is
the first trial of the new block; x < 0 pre-switch, x > 0 post-switch):

  1. TRANSITION RATE   P(state changes between x and x+1) as a function of x,
     mean +/- SEM across animals. A peak near x = 0 means the switch triggers
     strategy reorganization. Pre (x in [-win,-1]) vs post (x in [0,win-1])
     transition rate is compared per-animal (Wilcoxon).

  2. COMPOSITION       per-animal P(exploit) in the pre vs post window, compared
     with a Wilcoxon signed-rank test -- does the occupied state change across
     the switch (e.g. exploit -> explore then recovery)?

The animal is the unit throughout. Designed for K=2 (exploit/explore) but works
for any K. Needs per-trial states plus block_idx / trial_idx; other block
columns are not required.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as _stats

CREAM = "#FBFAF6"; INK = "#1A2E2A"; GREEN = "#2C5F2D"
CORAL = "#C9472B"; GOLD = "#E8A33D"; GRID = "#D8D5CC"; MUTE = "#6B6B66"


def _font():
    from matplotlib import font_manager
    for name in ("Georgia", "DejaVu Serif", "serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False); return name
        except Exception:
            continue
    return "serif"


# ---------------------------------------------------------------------------
# Build switch-aligned records (per animal)
# ---------------------------------------------------------------------------
def _switch_records(df, state_col, win):
    """For every switch, emit rows (animal, x, state, changed) for x in
    [-win, win], where x=0 is the first trial of the new block, `state` is the
    per-trial state, and `changed` = 1 if the state differs from the previous
    trial's state (NaN where the previous trial is outside the session)."""
    recs = []
    for (animal, _), ses in df.sort_values(
            ["animal", "session_file", "trial_idx"]).groupby(
            ["animal", "session_file"], sort=False):
        s = ses[state_col].to_numpy()
        blk = ses["block_idx"].to_numpy()
        n = len(ses)
        # per-trial "changed from previous trial" indicator
        changed = np.full(n, np.nan)
        changed[1:] = (s[1:] != s[:-1]).astype(float)
        ends = np.where(blk[:-1] != blk[1:])[0]          # last trial of a block
        for e in ends:
            for x in range(-win, win + 1):
                j = e + 1 + x                            # x=0 -> first new-block trial
                if 0 <= j < n and not (isinstance(s[j], float) and np.isnan(s[j])):
                    recs.append((animal, x, s[j], changed[j]))
    return pd.DataFrame(recs, columns=["animal", "x", "state", "changed"])


def _mean_sem(per_animal_series):
    x = per_animal_series.dropna().to_numpy()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0), n


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def switch_transition_analysis(df, state_col="glmhmm_state", exploit_state=0,
                               win=10, min_trials_per_cell=3):
    """Run both analyses. Returns a dict with the transition-rate curve, the
    pre/post transition-rate test, and the pre/post composition test."""
    R = _switch_records(df, state_col, win)
    if not len(R):
        raise ValueError("no switch-aligned records (check block_idx/trial_idx)")

    # --- 1. transition-rate curve: per-animal mean(changed) at each x -------
    xs = np.arange(-win, win + 1)
    curve = []
    for x in xs:
        rx = R[(R["x"] == x) & R["changed"].notna()]
        per = rx.groupby("animal")["changed"].mean()
        # require a few animals with data
        m, se, na = _mean_sem(per)
        curve.append({"x": int(x), "rate": m, "sem": se, "n_animals": na})
    curve = pd.DataFrame(curve)

    # --- pre vs post transition rate, per animal ---------------------------
    pre = R[(R["x"] >= -win) & (R["x"] <= -1) & R["changed"].notna()]
    post = R[(R["x"] >= 0) & (R["x"] <= win - 1) & R["changed"].notna()]
    tr_pre = pre.groupby("animal")["changed"].mean()
    tr_post = post.groupby("animal")["changed"].mean()
    tr = pd.concat([tr_pre.rename("pre"), tr_post.rename("post")], axis=1).dropna()
    tr["delta"] = tr["post"] - tr["pre"]
    Wt, pt = _stats.wilcoxon(tr["delta"].to_numpy()) if len(tr) >= 5 else (np.nan, np.nan)
    trans_test = {"n": int(len(tr)), "pre": float(tr["pre"].mean()),
                  "post": float(tr["post"].mean()),
                  "median_delta": float(tr["delta"].median()),
                  "wilcoxon_p": float(pt) if not np.isnan(pt) else np.nan,
                  "table": tr}

    # --- 2. composition: per-animal P(exploit) pre vs post -----------------
    R["_ex"] = (R["state"] == exploit_state).astype(float)
    cpre = R[(R["x"] >= -win) & (R["x"] <= -1)].groupby("animal")["_ex"].mean()
    cpost = R[(R["x"] >= 0) & (R["x"] <= win - 1)].groupby("animal")["_ex"].mean()
    comp = pd.concat([cpre.rename("pre"), cpost.rename("post")], axis=1).dropna()
    comp["delta"] = comp["post"] - comp["pre"]
    Wc, pc = _stats.wilcoxon(comp["delta"].to_numpy()) if len(comp) >= 5 else (np.nan, np.nan)
    comp_test = {"n": int(len(comp)), "pre_exploit": float(comp["pre"].mean()),
                 "post_exploit": float(comp["post"].mean()),
                 "median_delta": float(comp["delta"].median()),
                 "wilcoxon_p": float(pc) if not np.isnan(pc) else np.nan,
                 "table": comp}

    return {"curve": curve, "transition_test": trans_test,
            "composition_test": comp_test, "win": win}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_switch_transition(res, exploit_label="exploit", outfile="switch_transition.png"):
    font = _font()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.patch.set_facecolor(CREAM)
    for a in ax:
        a.set_facecolor(CREAM)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(True, color=GRID, lw=0.6, alpha=0.6)

    # panel 1: transition-rate curve
    c = res["curve"]
    ax[0].fill_between(c["x"], c["rate"] - c["sem"], c["rate"] + c["sem"],
                       color=GREEN, alpha=0.18)
    ax[0].plot(c["x"], c["rate"], "-o", color=GREEN, ms=4, lw=2)
    ax[0].axvline(0, ls="--", color=INK, lw=1.2)
    ax[0].text(0.2, ax[0].get_ylim()[1], " switch", fontsize=8, color=INK,
               va="top", fontfamily=font)
    ax[0].set_xlabel("trials relative to switch  (0 = first post-switch trial)",
                     fontsize=10, fontfamily=font)
    ax[0].set_ylabel("P(state changes vs previous trial)", fontsize=10, fontfamily=font)
    tt = res["transition_test"]
    ax[0].set_title(f"Transition rate around the switch\n"
                    f"pre {tt['pre']:.3f} vs post {tt['post']:.3f}, "
                    f"p={tt['wilcoxon_p']:.3g} (n={tt['n']})",
                    fontsize=11, weight="bold", color=INK, fontfamily=font)

    # panel 2: composition pre vs post (per-animal paired)
    comp = res["composition_test"]["table"]
    for _, r in comp.iterrows():
        ax[1].plot([0, 1], [r["pre"], r["post"]], "-", color=MUTE, alpha=0.3, lw=0.8)
    ax[1].plot(np.zeros(len(comp)), comp["pre"], "o", color=CORAL, ms=5, alpha=0.7)
    ax[1].plot(np.ones(len(comp)), comp["post"], "o", color=GREEN, ms=5, alpha=0.7)
    ax[1].hlines(comp["pre"].mean(), -0.15, 0.15, color=INK, lw=2)
    ax[1].hlines(comp["post"].mean(), 0.85, 1.15, color=INK, lw=2)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["pre-switch", "post-switch"], fontfamily=font)
    ax[1].set_ylabel(f"P({exploit_label})  per animal", fontsize=10, fontfamily=font)
    ct = res["composition_test"]
    ax[1].set_title(f"{exploit_label} occupancy pre vs post\n"
                    f"Δ={ct['median_delta']:+.3f}, p={ct['wilcoxon_p']:.3g} (n={ct['n']})",
                    fontsize=11, weight="bold", color=INK, fontfamily=font)

    fig.suptitle(f"Strategy change locked to the switch  (±{res['win']} trials)",
                 fontsize=13, weight="bold", color=INK, fontfamily=font)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def explore_latency_vs_lrandom(
        df, state_col="glmhmm_state", explore_state=1,
        lrandom_bins=((0, 3), (4, 7), (8, 14), (15, 10 ** 9)),
        min_blocks=10, min_animals=5,
        lrandom_col="block_trial_random_added", ttc_col="block_trial_to_crit"):
    """
    Latency to the first explore state within each block's pre-switch L_Random
    window, as a function of L_Random length.

    For every block, tau = trials since criterion (0 = first L_Random trial);
    latency = the smallest tau >= 0 (and tau < L_Random) at which the state is
    `explore_state`. Blocks that never enter explore before the switch are
    CENSORED and excluded from the latency, but their fraction is reported per
    L_Random bin. The animal is the unit: per-animal mean latency is computed in
    each bin, then compared across bins, and a per-animal slope of latency vs
    bin index is tested with a Wilcoxon signed-rank test.

    Needs block_idx / trial_idx plus the block-clock columns. Returns a dict with
    the per-bin table, the per-animal slope test, and the censoring rates.
    """
    for c in (lrandom_col, ttc_col, "block_idx", "trial_idx"):
        if c not in df.columns:
            raise KeyError(f"explore_latency_vs_lrandom needs '{c}'")
    d = df.sort_values(["animal", "session_file", "block_idx", "trial_idx"]).copy()
    d = d[d[state_col].notna()]
    gb = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["wb"] = gb.cumcount()
    d["tau"] = d["wb"] - d[ttc_col]
    # restrict to pre-switch L_Random trials
    lr = d[lrandom_col]
    d = d[(d["tau"] >= 0) & (d["tau"] < lr) & lr.notna()].copy()
    # drop last block per session (no switch)
    last = d.groupby(["animal", "session_file"])["block_idx"].transform("max")
    d = d[d["block_idx"] < last].copy()

    # one row per block: L_Random length, whether it explored, and the latency
    recs = []
    for (animal, ses, blk), g in d.groupby(["animal", "session_file", "block_idx"],
                                           sort=False):
        lrand = float(g[lrandom_col].iloc[0])
        ex = g[g[state_col] == explore_state]
        explored = len(ex) > 0
        latency = float(ex["tau"].min()) if explored else np.nan
        recs.append({"animal": animal, "lrandom": lrand,
                     "explored": explored, "latency": latency})
    B = pd.DataFrame(recs)

    def _bin(v):
        for i, (lo, hi) in enumerate(lrandom_bins):
            if lo <= v <= hi:
                return i
        return np.nan
    B["bin"] = B["lrandom"].apply(_bin)

    # per-bin summary (across-animal mean of per-animal means) + censoring
    rows = []
    for i, (lo, hi) in enumerate(lrandom_bins):
        sub = B[B["bin"] == i]
        if len(sub) < min_blocks:
            continue
        # per-animal mean latency among explored blocks
        per = sub[sub["explored"]].groupby("animal")["latency"].mean()
        na = len(per)
        m = float(per.mean()) if na else np.nan
        sem = float(per.std(ddof=1) / np.sqrt(na)) if na > 1 else 0.0
        cens = 1.0 - sub["explored"].mean()          # fraction never exploring
        rows.append({"bin": i, "lrandom_range": f"{lo}-{hi if hi < 1e9 else '+'}",
                     "n_blocks": int(len(sub)), "n_animals": int(na),
                     "mean_latency": m, "sem_latency": sem,
                     "frac_censored": float(cens),
                     "median_lrandom": float(sub["lrandom"].median())})
    tab = pd.DataFrame(rows)

    # per-animal slope of latency vs bin index (explored blocks only)
    slopes = {}
    for animal, a in B[B["explored"]].groupby("animal"):
        per_bin = a.groupby("bin")["latency"].mean()
        if per_bin.notna().sum() >= 3:
            xb = per_bin.index.to_numpy(float)
            slopes[animal] = float(np.polyfit(xb, per_bin.to_numpy(), 1)[0])
    slopes = pd.Series(slopes, name="latency_slope")
    if len(slopes.dropna()) >= min_animals:
        W, p = _stats.wilcoxon(slopes.dropna().to_numpy())
        slope_test = {"n": int(slopes.notna().sum()),
                      "median_slope": float(slopes.median()),
                      "frac_pos": float((slopes > 0).mean()),
                      "wilcoxon_p": float(p)}
    else:
        slope_test = {"n": int(slopes.notna().sum()), "median_slope": np.nan,
                      "frac_pos": np.nan, "wilcoxon_p": np.nan}

    return {"per_bin": tab, "slope_test": slope_test, "slopes": slopes,
            "blocks": B, "explore_state": explore_state}


def plot_explore_latency(res, outfile="explore_latency_vs_lrandom.png"):
    font = _font()
    tab = res["per_bin"]
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(CREAM); ax1.set_facecolor(CREAM)
    xb = np.arange(len(tab))
    ax1.errorbar(xb, tab["mean_latency"], yerr=tab["sem_latency"], fmt="-o",
                 color=CORAL, ms=6, lw=2, capsize=3, label="latency to explore")
    ax1.set_xticks(xb); ax1.set_xticklabels(tab["lrandom_range"], fontfamily=font)
    ax1.set_xlabel("L_Random length (trials)", fontsize=10, fontfamily=font)
    ax1.set_ylabel("τ of first explore  (mean ± SEM across mice)", fontsize=10,
                   fontfamily=font, color=CORAL)
    ax2 = ax1.twinx()
    ax2.plot(xb, 100 * tab["frac_censored"], "--s", color=MUTE, ms=5,
             label="% never explored")
    ax2.set_ylabel("% blocks that never explored", fontsize=10, fontfamily=font, color=MUTE)
    ax2.set_ylim(0, 100)
    st = res["slope_test"]
    ax1.set_title(f"Latency to explore vs L_Random length\n"
                  f"per-mouse slope median {st['median_slope']:+.3f}, "
                  f"p={st['wilcoxon_p']:.3g} (n={st['n']})",
                  fontsize=11, weight="bold", color=INK, fontfamily=font)
    for a in (ax1, ax2):
        for sp in ("top",):
            a.spines[sp].set_visible(False)
    ax1.grid(True, color=GRID, lw=0.6, alpha=0.6)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def explore_hazard_by_lrandom(
        df, state_col="glmhmm_state", explore_state=1, tau_max=15,
        lrandom_bins=((0, 3), (4, 7), (8, 14), (15, 10 ** 9)), min_risk=20,
        min_animals=5, lrandom_col="block_trial_random_added",
        ttc_col="block_trial_to_crit"):
    """
    Discrete-time HAZARD of the first explore state, deconfounded from block
    length. For each tau, hazard(tau) = P(first explore at tau | the block
    reached tau and has not explored yet). A block contributes to the risk set
    at tau only while it is at least tau+1 L_Random trials long AND has not yet
    explored; once it explores (event) or the switch arrives (it leaves the risk
    set), it stops contributing. This removes the truncation artefact that
    inflates the raw latency: short blocks simply exit the risk set early rather
    than biasing the estimate.

    Computed within each L_Random bin. If the hazard curves coincide across bins,
    the internal 'clock' to explore is FIXED (pure belief erosion); if long-block
    bins have a lower/higher hazard at matched tau, block length modulates it.
    Per-animal hazard (pooled over tau) is compared across bins with a Wilcoxon
    slope test. Returns a dict with the per-bin hazard curves and the test.
    """
    for c in (lrandom_col, ttc_col, "block_idx", "trial_idx"):
        if c not in df.columns:
            raise KeyError(f"explore_hazard_by_lrandom needs '{c}'")
    d = df.sort_values(["animal", "session_file", "block_idx", "trial_idx"]).copy()
    d = d[d[state_col].notna()]
    gb = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["wb"] = gb.cumcount()
    d["tau"] = d["wb"] - d[ttc_col]
    lr = d[lrandom_col]
    d = d[(d["tau"] >= 0) & (d["tau"] < lr) & lr.notna()].copy()
    last = d.groupby(["animal", "session_file"])["block_idx"].transform("max")
    d = d[d["block_idx"] < last].copy()

    # one row per block: L_Random length, first-explore tau (or inf), and the
    # state at tau=0 (entry into the L_Random window)
    recs = []
    for (animal, ses, blk), g in d.groupby(["animal", "session_file", "block_idx"],
                                           sort=False):
        lrand = float(g[lrandom_col].iloc[0])
        g0 = g[g["tau"] == 0]
        if not len(g0):
            continue
        entry_state = int(g0[state_col].iloc[0])          # state at first L_Random trial
        ex = g[g[state_col] == explore_state]
        t_ev = float(ex["tau"].min()) if len(ex) else np.inf
        recs.append({"animal": animal, "lrandom": lrand, "t_event": t_ev,
                     "entry_state": entry_state})
    B = pd.DataFrame(recs)

    # Condition on entering the window in EXPLOIT: blocks already in explore at
    # tau=0 carried that state over from the criterion phase, so counting them as
    # a "first explore at tau=0" produces a spurious hazard peak that only
    # reflects the ~20% baseline explore occupancy, not a transition. Keeping
    # only exploit-entry blocks makes the hazard measure genuine exploit->explore
    # transitions from a common starting state.
    B = B[B["entry_state"] != explore_state].copy()

    def _bin(v):
        for i, (lo, hi) in enumerate(lrandom_bins):
            if lo <= v <= hi:
                return i
        return np.nan
    B["bin"] = B["lrandom"].apply(_bin)

    def _hazard_rows(sub):
        """discrete hazard per tau over a set of blocks."""
        out = []
        for tt in range(0, tau_max + 1):
            at_risk = sub[(sub["lrandom"] > tt) & (sub["t_event"] >= tt)]  # reached tau, not yet explored
            n = len(at_risk)
            if n < min_risk:
                continue
            events = int((at_risk["t_event"] == tt).sum())
            out.append({"tau": tt, "hazard": events / n, "n_risk": n})
        return pd.DataFrame(out)

    # per-bin hazard curves
    curves = {}
    for i, (lo, hi) in enumerate(lrandom_bins):
        sub = B[B["bin"] == i]
        h = _hazard_rows(sub)
        if len(h):
            curves[i] = h

    # per-animal test: compare hazard at MATCHED tau between two bins. Rather
    # than the shortest vs longest bin (which share only tau 0..2 because the
    # short bin ends early), pick the pair of bins that SHARE THE MOST tau values
    # -- a longer, more informative overlap and a less fragile test.
    keys = sorted(curves)
    best_pair, best_overlap = None, -1
    for ii in range(len(keys)):
        for jj in range(ii + 1, len(keys)):
            ov = set(curves[keys[ii]]["tau"]) & set(curves[keys[jj]]["tau"])
            if len(ov) > best_overlap:
                best_overlap, best_pair = len(ov), (keys[ii], keys[jj])
    lo_bin, hi_bin = best_pair          # lo_bin = shorter L_Random, hi_bin = longer
    shared_tau = sorted(set(curves[lo_bin]["tau"]) & set(curves[hi_bin]["tau"]))
    diffs = {}
    for animal, a in B.groupby("animal"):
        hs = _hazard_rows(a[a["bin"] == lo_bin]) if len(a[a["bin"] == lo_bin]) else pd.DataFrame()
        hl = _hazard_rows(a[a["bin"] == hi_bin]) if len(a[a["bin"] == hi_bin]) else pd.DataFrame()
        if not len(hs) or not len(hl):
            continue
        hs = hs.set_index("tau")["hazard"]; hl = hl.set_index("tau")["hazard"]
        common = [t for t in shared_tau if t in hs.index and t in hl.index]
        if len(common) >= 2:
            diffs[animal] = float((hl.loc[common] - hs.loc[common]).mean())  # long - short
    diffs = pd.Series(diffs, name="hazard_long_minus_short")
    lrl = lrandom_bins[lo_bin]; lrh = lrandom_bins[hi_bin]
    pair_label = (f"L_Rand {lrl[0]}-{lrl[1] if lrl[1] < 1e9 else '+'}"
                  f" vs {lrh[0]}-{lrh[1] if lrh[1] < 1e9 else '+'}")
    if diffs.notna().sum() >= min_animals:
        W, p = _stats.wilcoxon(diffs.dropna().to_numpy())
        test = {"n": int(diffs.notna().sum()),
                "median_long_minus_short": float(diffs.median()),
                "frac_pos": float((diffs > 0).mean()), "wilcoxon_p": float(p),
                "pair": pair_label, "n_shared_tau": len(shared_tau),
                "note": "hazard at matched tau, longer bin minus shorter bin "
                        f"({pair_label}, {len(shared_tau)} shared τ); "
                        "n.s. => fixed clock (no modulation by block length)"}
    else:
        test = {"n": int(diffs.notna().sum()), "median_long_minus_short": np.nan,
                "frac_pos": np.nan, "wilcoxon_p": np.nan, "pair": pair_label}

    return {"curves": curves, "bins": lrandom_bins, "slope_test": test,
            "diffs": diffs}


def plot_explore_hazard(res, outfile="explore_hazard_by_lrandom.png"):
    font = _font()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    bins = res["bins"]
    shades = np.linspace(0.62, 0.0, len(res["curves"]))
    base = CORAL
    for (i, h), t in zip(sorted(res["curves"].items()), shades):
        lo, hi = bins[i]
        c = _shade(base, t)
        lab = f"L_Rand {lo}-{hi if hi < 1e9 else '+'}"
        ax.plot(h["tau"], h["hazard"], "-o", ms=4, lw=1.8, color=c, label=lab)
    ax.set_xlabel("τ  =  trials since criterion", fontsize=10, fontfamily=font)
    ax.set_ylabel("hazard: P(first explore at τ | reached τ, not yet explored)",
                  fontsize=9.5, fontfamily=font)
    st = res["slope_test"]
    ax.set_title(f"Deconfounded hazard of first explore\n"
                 f"{st.get('pair','')} at matched τ: median {st['median_long_minus_short']:+.4f}, "
                 f"p={st['wilcoxon_p']:.3g} (n={st['n']})",
                 fontsize=11, weight="bold", color=INK, fontfamily=font)
    ax.legend(frameon=False, fontsize=8.5, prop={"family": font},
              title="darker = longer L_Random", title_fontsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def _shade(hex_color, t):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * t) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_switch_transition(df, state_col="glmhmm_state", exploit_state=0,
                          exploit_label="exploit", win=10,
                          output_dir="analysis", figs_dir="figs/occupancy"):
    print(f"\n=== Switch-locked strategy change (±{win} trials) ===")
    res = switch_transition_analysis(df, state_col, exploit_state, win)
    tt, ct = res["transition_test"], res["composition_test"]
    print("[1] transition rate (state changes):")
    print(f"    pre  {tt['pre']:.3f}  vs  post {tt['post']:.3f}  "
          f"(Δ median {tt['median_delta']:+.3f}, Wilcoxon p={tt['wilcoxon_p']:.3g}, n={tt['n']})")
    print(f"[2] {exploit_label} occupancy:")
    print(f"    pre  {ct['pre_exploit']:.3f}  vs  post {ct['post_exploit']:.3f}  "
          f"(Δ median {ct['median_delta']:+.3f}, Wilcoxon p={ct['wilcoxon_p']:.3g}, n={ct['n']})")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    res["curve"].to_csv(Path(output_dir) / "switch_transition_curve.csv", index=False)
    fig = plot_switch_transition(res, exploit_label,
                                 str(Path(figs_dir) / "switch_transition.png"))
    print(f"    wrote {fig}")

    # --- explore-latency vs L_Random (needs block-clock columns) -----------
    explore_state = 1 if exploit_state == 0 else 0
    if {"block_trial_random_added", "block_trial_to_crit"} <= set(df.columns):
        lat = explore_latency_vs_lrandom(df, state_col, explore_state=explore_state)
        print(f"[3] latency to explore vs L_Random length:")
        for _, r in lat["per_bin"].iterrows():
            print(f"    L_Rand {r['lrandom_range']:>4}: mean τ={r['mean_latency']:.2f}"
                  f"±{r['sem_latency']:.2f}  ({r['frac_censored']*100:.0f}% never explored,"
                  f" n={r['n_animals']} mice)")
        stp = lat["slope_test"]
        print(f"    per-mouse slope of latency vs bin: median {stp['median_slope']:+.3f}, "
              f"Wilcoxon p={stp['wilcoxon_p']:.3g} (n={stp['n']})")
        lat["per_bin"].to_csv(Path(output_dir) / "explore_latency_vs_lrandom.csv", index=False)
        figl = plot_explore_latency(lat, str(Path(figs_dir) / "explore_latency_vs_lrandom.png"))
        print(f"    wrote {figl}")
        res["latency"] = lat

        # --- deconfounded hazard of first explore --------------------------
        haz = explore_hazard_by_lrandom(df, state_col, explore_state=explore_state)
        print(f"[4] deconfounded hazard of first explore (survival):")
        sth = haz["slope_test"]
        print(f"    hazard at matched τ (long − short bin): median {sth['median_long_minus_short']:+.4f}, "
              f"Wilcoxon p={sth['wilcoxon_p']:.3g} (n={sth['n']})")
        print(f"    -> if p is n.s., the internal clock to explore is FIXED "
              f"(erosion), not modulated by block length")
        figh = plot_explore_hazard(haz, str(Path(figs_dir) / "explore_hazard_by_lrandom.png"))
        print(f"    wrote {figh}")
        res["hazard"] = haz
    else:
        print("[3] skipped latency (needs block_trial_random_added / block_trial_to_crit)")
    return res


if __name__ == "__main__":
    import sys
    st = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "analysis/glmhmm_states_K2.csv")
    run_switch_transition(st, output_dir=".", figs_dir=".")