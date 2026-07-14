"""
anticipation_integrated.py
==========================
Integrated anticipation pipeline (Undermind "Integrated anticipation analysis
plan"), linking three analyses into one conservative sequence:

  1. History-length selection  -> a justified local-history control depth K*
     Logistic prediction of side choice from lagged (choice, reward, choice x
     reward) terms, K in {1,2,3,4,5,6,8,10}, mouse-stratified session-level
     5-fold CV; K* = smallest K within 1 SEM of the best held-out NLL.

  2. History-matched anticipation  -> mouse-level late-minus-early effect
     Pre-switch L_Random trials only, early/late split within block, EXACT
     matching on M_t = (C_{t-1}, R_{t-1}, C_{t-2}, R_{t-2}, Q_t), primary
     outcome = chose the currently WORSE option. Per-mouse harmonic-weighted
     Delta_m, Wilcoxon + bootstrap across mice.

  3. Directed better->worse test  -> directional specificity
     Same matched estimator with outcome = better->worse transition, plus the
     worse->better negative control, plus a complementary trial-level logistic
     regression (late + compact history) as a robustness check.

Everything uses the mouse as the unit of inference. Reads the standard bandit
dataframe columns: animal, session_file, block_idx, trial_idx, choice (-1/+1),
rewarded (0/1), hr_side (-1/+1 better side), block_trial_to_crit,
block_trial_random_added.

Entry point: run_integrated_anticipation(df, output_dir, figs_dir).
Meant to be called from the RUN_PLOTS / RUN_MODELS branch of master_bandit.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as _stats

CREAM = "#FBFAF6"; INK = "#1A2E2A"; GREEN = "#2C5F2D"
CORAL = "#C9472B"; GOLD = "#E8A33D"; GRID = "#D8D5CC"; MUTE = "#6B6B66"

CAND_LAGS = (1, 2, 3, 4, 5, 6, 8, 10)


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def _nonmiss_session_lags(df, max_k):
    """Add centered choice/reward lags over the sequence of NON-MISS trials
    within each session (so t-1 is the previous non-miss trial). Adds cC, cR
    (centered current) and cC_k, cR_k for k=1..max_k. Returns the non-miss
    subframe with a within-session order preserved."""
    d = df[df["choice"].notna() & df["rewarded"].notna()].copy()
    d = d.sort_values(["animal", "session_file", "trial_idx"])
    d["cC"] = np.where(d["choice"].to_numpy() > 0, 0.5, -0.5)      # right=+0.5
    d["cR"] = np.where(d["rewarded"].to_numpy() > 0, 0.5, -0.5)    # rewarded=+0.5
    g = d.groupby(["animal", "session_file"], sort=False)
    for k in range(1, max_k + 1):
        d[f"cC_{k}"] = g["cC"].shift(k)
        d[f"cR_{k}"] = g["cR"].shift(k)
    return d


def _design_for_K(d, K):
    """Feature matrix for lag horizon K: for each lag, choice, reward, and their
    (centered) interaction. Returns (X, y, mouse, session, valid_mask)."""
    cols = []
    for k in range(1, K + 1):
        d[f"cCR_{k}"] = d[f"cC_{k}"] * d[f"cR_{k}"]
        cols += [f"cC_{k}", f"cR_{k}", f"cCR_{k}"]
    valid = d[cols].notna().all(axis=1)
    X = d.loc[valid, cols].to_numpy(float)
    y = (d.loc[valid, "choice"].to_numpy() > 0).astype(int)        # Y = 1[C_t = R]
    mouse = d.loc[valid, "animal"].to_numpy()
    session = d.loc[valid, "session_file"].to_numpy()
    return X, y, mouse, session


# ---------------------------------------------------------------------------
# Analysis 1: history-length selection
# ---------------------------------------------------------------------------
def _mouse_fold_assignment(mouse, session, n_folds, seed=0):
    """Assign each session to a fold, balanced within mouse (round-robin over a
    shuffled session list per mouse). Returns an array of fold ids per trial."""
    rng = np.random.default_rng(seed)
    key = pd.Series(list(zip(mouse, session)))
    uniq = key.drop_duplicates().tolist()
    by_mouse = {}
    for m, s in uniq:
        by_mouse.setdefault(m, []).append((m, s))
    fold_of = {}
    for m, lst in by_mouse.items():
        rng.shuffle(lst)
        for i, ms in enumerate(lst):
            fold_of[ms] = i % n_folds
    return key.map(fold_of).to_numpy()


def select_history_length(df, cand_lags=CAND_LAGS, n_folds=5, seed=0,
                          mouse_fe=True, one_sem_rule=True):
    """Analysis 1. Mouse-stratified session-level CV of lag-K logistic choice
    prediction. Returns dict with the per-K table and the selected K*."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score
    except Exception as e:
        raise ImportError("select_history_length needs scikit-learn "
                          f"(pip install scikit-learn): {e}")

    d = _nonmiss_session_lags(df, max(cand_lags))
    Kmax = max(cand_lags)
    _, _, mouse_all, session_all = _design_for_K(d, Kmax)   # deepest-valid rows
    # fold ids defined on the deepest-valid set, then reused per K via re-mask
    fold_all = _mouse_fold_assignment(mouse_all, session_all, n_folds, seed)
    # map fold back onto d rows that are valid at Kmax
    valid_max = d[[f"cC_{k}" for k in range(1, Kmax + 1)]
                  + [f"cR_{k}" for k in range(1, Kmax + 1)]].notna().all(axis=1)
    d = d.loc[valid_max].copy()
    d["_fold"] = fold_all

    if mouse_fe:
        mouse_dummies = pd.get_dummies(d["animal"], prefix="m").to_numpy(float)

    rows = []
    for K in cand_lags:
        cols = []
        for k in range(1, K + 1):
            d[f"cCR_{k}"] = d[f"cC_{k}"] * d[f"cR_{k}"]
            cols += [f"cC_{k}", f"cR_{k}", f"cCR_{k}"]
        Xh = d[cols].to_numpy(float)
        X = np.hstack([Xh, mouse_dummies]) if mouse_fe else Xh
        y = (d["choice"].to_numpy() > 0).astype(int)
        folds = d["_fold"].to_numpy()

        nlls, aucs = [], []
        for f in range(n_folds):
            tr, te = folds != f, folds == f
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1e6, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            nlls.append(log_loss(y[te], p, labels=[0, 1]))
            if len(np.unique(y[te])) == 2:
                aucs.append(roc_auc_score(y[te], p))
        rows.append({"K": K,
                     "nll_mean": float(np.mean(nlls)),
                     "nll_sem": float(np.std(nlls, ddof=1) / np.sqrt(len(nlls))),
                     "auc_mean": float(np.mean(aucs)) if aucs else np.nan,
                     "n_folds": len(nlls)})
    tab = pd.DataFrame(rows)

    best_i = int(tab["nll_mean"].idxmin())
    best_nll = tab.loc[best_i, "nll_mean"]
    best_sem = tab.loc[best_i, "nll_sem"]
    if one_sem_rule:
        within = tab[tab["nll_mean"] <= best_nll + best_sem]
        k_star = int(within["K"].min())
    else:
        k_star = int(tab.loc[best_i, "K"])
    return {"table": tab, "k_star": k_star, "best_K": int(tab.loc[best_i, "K"]),
            "best_nll": float(best_nll)}


# ---------------------------------------------------------------------------
# Build the matched pre-switch L_Random trial set (Analyses 2 & 3)
# ---------------------------------------------------------------------------
def build_matched_set(df, min_block_elig=4):
    """Return pre-switch L_Random, non-miss trials with t-1/t-2 in the same
    block, the exact-matching signature M_t, better/worse outcomes, and the
    within-block early/late split. Only blocks with >= min_block_elig eligible
    trials are kept."""
    need = ["animal", "session_file", "block_idx", "trial_idx", "choice",
            "hr_side", "block_trial_to_crit", "block_trial_random_added"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"build_matched_set needs {miss}")
    d = df.sort_values(["animal", "session_file", "block_idx", "trial_idx"]).copy()
    if "rewarded" not in d.columns:
        d["rewarded"] = np.nan
    gb = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["wb"] = gb.cumcount()
    d["tau"] = d["wb"] - d["block_trial_to_crit"]
    # pre-switch L_Random window
    d = d[(d["tau"] >= 0) & (d["tau"] < d["block_trial_random_added"])].copy()
    # drop last block per session (no observed switch)
    last = d.groupby(["animal", "session_file"])["block_idx"].transform("max")
    d = d[d["block_idx"] < last].copy()

    # lags within block over the (already L_Random) sequence -- but t-1/t-2 must
    # be real adjacent trials in the block; recompute lags on the full block,
    # then restrict. Simplest: use within-block shift on the L_Random rows.
    gbl = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["C1"] = gbl["choice"].shift(1)
    d["C2"] = gbl["choice"].shift(2)
    d["R1"] = gbl["rewarded"].shift(1)
    d["R2"] = gbl["rewarded"].shift(2)
    d["hr1"] = gbl["hr_side"].shift(1)
    d = d.dropna(subset=["choice", "C1", "C2", "R1", "R2", "hr1"])

    # signature + outcomes
    d["Q"] = (d["C1"] == d["hr1"]).astype(int)                 # prev choice was better
    d["chose_worse"] = (d["choice"] != d["hr_side"]).astype(int)
    d["prev_better"] = (d["C1"] == d["hr1"]).astype(int)
    d["now_better"] = (d["choice"] == d["hr_side"]).astype(int)
    d["bw"] = ((d["prev_better"] == 1) & (d["now_better"] == 0)).astype(int)
    d["wb"] = ((d["prev_better"] == 0) & (d["now_better"] == 1)).astype(int)
    d["sig"] = list(zip(d["C1"].astype(int), d["R1"].astype(int),
                        d["C2"].astype(int), d["R2"].astype(int), d["Q"].astype(int)))

    # keep blocks with >= min_block_elig eligible trials, then early/late split
    blk = d.groupby(["animal", "session_file", "block_idx"], sort=False)
    d["_nblk"] = blk["choice"].transform("size")
    d = d[d["_nblk"] >= min_block_elig].copy()
    d["_rank"] = blk.cumcount()
    d["_half"] = np.where(d["_rank"] < (d["_nblk"] // 2), "early", "late")
    return d


def _matched_delta(d, outcome, split_col="_half"):
    """Per-mouse harmonic-weighted late-minus-early Delta for a binary outcome,
    over exact (mouse, signature) strata with >=1 early and >=1 late trial.
    Returns (per-mouse Series Delta_m, retention fraction)."""
    used = 0
    per_mouse = {}
    for mouse, dm in d.groupby("animal"):
        num = den = 0.0
        for _, g in dm.groupby("sig"):
            e = g[g[split_col] == "early"][outcome]
            l = g[g[split_col] == "late"][outcome]
            nE, nL = len(e), len(l)
            if nE < 1 or nL < 1:
                continue
            w = nE * nL / (nE + nL)
            num += w * (l.mean() - e.mean())
            den += w
            used += nE + nL
        if den > 0:
            per_mouse[mouse] = num / den
    retention = used / max(len(d), 1)
    return pd.Series(per_mouse, name=f"delta_{outcome}"), retention


def _group_test(delta, n_boot=5000, seed=0):
    """Median, Wilcoxon, frac>0, bootstrap 95% CI over mice."""
    x = delta.dropna().to_numpy()
    if len(x) < 5:
        return {"n": len(x), "median": np.nan, "frac_pos": np.nan,
                "wilcoxon_p": np.nan, "ci": (np.nan, np.nan)}
    W, p = _stats.wilcoxon(x)
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return {"n": int(len(x)), "median": float(np.median(x)),
            "frac_pos": float((x > 0).mean()), "wilcoxon_p": float(p),
            "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))}


def history_matched_anticipation(d, outcome="chose_worse", n_boot=5000, seed=0):
    """Analysis 2 (or 3, by outcome): matched late-vs-early estimator + group
    test. `d` is the output of build_matched_set."""
    delta, retention = _matched_delta(d, outcome)
    res = _group_test(delta, n_boot=n_boot, seed=seed)
    res.update({"outcome": outcome, "retention": float(retention), "delta": delta})
    return res


# ---------------------------------------------------------------------------
# Analysis 3: complementary regression (robustness for better->worse)
# ---------------------------------------------------------------------------
def complementary_regression(d, Kc=2):
    """Trial-level logistic P(better->worse) ~ late + compact history + mouse FE.
    Returns the 'late' coefficient (log-odds) or None if sklearn unavailable."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    d = d.copy()
    d["late"] = (d["_half"] == "late").astype(int)
    feats = ["late"]
    # compact centered history (up to Kc lags already present as C1,R1,C2,R2)
    d["cC1"] = np.where(d["C1"] > 0, 0.5, -0.5); d["cR1"] = np.where(d["R1"] > 0, 0.5, -0.5)
    feats += ["cC1", "cR1"]
    if Kc >= 2:
        d["cC2"] = np.where(d["C2"] > 0, 0.5, -0.5); d["cR2"] = np.where(d["R2"] > 0, 0.5, -0.5)
        feats += ["cC2", "cR2"]
    X = np.hstack([d[feats].to_numpy(float),
                   pd.get_dummies(d["animal"], prefix="m").to_numpy(float)])
    y = d["bw"].to_numpy(int)
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, C=1e6, solver="lbfgs").fit(X, y)
    return {"beta_late": float(clf.coef_[0][0]), "Kc": Kc}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _font():
    from matplotlib import font_manager
    for name in ("Georgia", "DejaVu Serif", "serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False); return name
        except Exception:
            continue
    return "serif"


def make_summary_figure(hist, cw, bw, wb, outfile):
    font = _font()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(CREAM)
    for a in ax:
        a.set_facecolor(CREAM)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(True, color=GRID, lw=0.6, alpha=0.6)

    # (1) history-length CV
    t = hist["table"]
    ax[0].errorbar(t["K"], t["nll_mean"], yerr=t["nll_sem"], fmt="-o", color=GREEN,
                   ms=5, capsize=3)
    kstar = hist["k_star"]
    ax[0].axvline(kstar, ls="--", color=CORAL, lw=1.5)
    ax[0].text(kstar, ax[0].get_ylim()[1], f" K*={kstar}", color=CORAL,
               va="top", fontsize=9, fontfamily=font)
    ax[0].set_xlabel("history lag horizon K", fontsize=10, fontfamily=font)
    ax[0].set_ylabel("held-out NLL / trial", fontsize=10, fontfamily=font)
    ax[0].set_title("1 · History-length selection", fontsize=11, weight="bold",
                    color=INK, fontfamily=font)

    # (2) chose-worse late-minus-early per mouse
    x = cw["delta"].dropna().to_numpy()
    ax[1].axhline(0, color=MUTE, ls="--", lw=1)
    jit = np.random.default_rng(0).uniform(-0.08, 0.08, len(x))
    ax[1].scatter(jit, x, s=24, color=GREEN, alpha=0.7, edgecolor="white", lw=0.5)
    ax[1].hlines(np.median(x), -0.2, 0.2, color=INK, lw=2)
    ax[1].set_xticks([0]); ax[1].set_xticklabels(["chose worse"], fontfamily=font)
    ax[1].set_ylabel("late − early  (per mouse)", fontsize=10, fontfamily=font)
    ax[1].set_title(f"2 · History-matched\nmedian {cw['median']:+.4f}, p={cw['wilcoxon_p']:.3g}",
                    fontsize=11, weight="bold", color=INK, fontfamily=font)

    # (3) directed bw vs wb per mouse
    for i, (res, col, lab) in enumerate([(bw, GREEN, "better->worse"),
                                         (wb, CORAL, "worse->better")]):
        xx = res["delta"].dropna().to_numpy()
        j = np.full(len(xx), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(xx))
        ax[2].scatter(j, xx, s=24, color=col, alpha=0.7, edgecolor="white", lw=0.5)
        ax[2].hlines(np.median(xx), i - 0.2, i + 0.2, color=INK, lw=2)
    ax[2].axhline(0, color=MUTE, ls="--", lw=1)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["better->worse", "worse->better"], fontfamily=font)
    ax[2].set_ylabel("late − early  (per mouse)", fontsize=10, fontfamily=font)
    ax[2].set_title("3 · Directed test", fontsize=11, weight="bold", color=INK, fontfamily=font)

    fig.suptitle("Integrated anticipation pipeline  (mouse-level, history-controlled)",
                 fontsize=13, weight="bold", color=INK, fontfamily=font)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_integrated_anticipation(df, output_dir="analysis", figs_dir="figs",
                                n_boot=5000, seed=0):
    """Run the full 3-step pipeline, print a report, save CSVs + a summary
    figure, and return all results."""
    print("\n=== Integrated anticipation pipeline ===")

    # Analysis 1
    hist = select_history_length(df, seed=seed)
    print("[1] history-length selection")
    for _, r in hist["table"].iterrows():
        print(f"    K={int(r['K']):>2}: NLL={r['nll_mean']:.4f}±{r['nll_sem']:.4f}  "
              f"AUC={r['auc_mean']:.3f}")
    print(f"    -> K* = {hist['k_star']} (best K={hist['best_K']})")

    # Matched set (Analyses 2 & 3)
    d = build_matched_set(df)
    n_mice = d["animal"].nunique()
    print(f"[matched set] {len(d)} pre-switch L_Random trials, {n_mice} mice")

    # Analysis 2
    cw = history_matched_anticipation(d, "chose_worse", n_boot, seed)
    print("[2] history-matched anticipation (chose worse)")
    print(f"    median Δ={cw['median']:+.4f}, {cw['frac_pos']*100:.0f}% >0, "
          f"Wilcoxon p={cw['wilcoxon_p']:.3g}, 95% CI "
          f"[{cw['ci'][0]:+.4f},{cw['ci'][1]:+.4f}], n={cw['n']}, "
          f"retention={cw['retention']*100:.0f}%")

    # Analysis 3
    bw = history_matched_anticipation(d, "bw", n_boot, seed)
    wb = history_matched_anticipation(d, "wb", n_boot, seed)
    Kc = 2 if hist["k_star"] <= 4 else 3
    reg = complementary_regression(d, Kc=Kc)
    print("[3] directed test")
    print(f"    better→worse: median Δ={bw['median']:+.4f}, p={bw['wilcoxon_p']:.3g}, n={bw['n']}")
    print(f"    worse→better: median Δ={wb['median']:+.4f}, p={wb['wilcoxon_p']:.3g}, n={wb['n']}")
    if reg:
        print(f"    complementary regression β_late(better→worse) = {reg['beta_late']:+.4f} (Kc={reg['Kc']})")

    # Diagnostics
    print("[diagnostics]")
    print(f"    match retention {cw['retention']*100:.0f}% (target >=50%)")
    print(f"    mouse coverage n={cw['n']} (target >=25)")
    print(f"    directional specificity: |bw|>|wb|? "
          f"{abs(bw['median']) > abs(wb['median'])}")

    # Save
    outd = Path(output_dir); outd.mkdir(parents=True, exist_ok=True)
    hist["table"].to_csv(outd / "anticip_history_length_cv.csv", index=False)
    pd.concat([cw["delta"].rename("chose_worse"), bw["delta"].rename("bw"),
               wb["delta"].rename("wb")], axis=1).to_csv(
        outd / "anticip_per_mouse_deltas.csv")
    fig = make_summary_figure(hist, cw, bw, wb,
                              str(Path(figs_dir) / "anticipation_integrated.png"))
    print(f"    wrote {fig}")

    return {"history": hist, "chose_worse": cw, "bw": bw, "wb": wb,
            "regression": reg, "n_mice": n_mice}