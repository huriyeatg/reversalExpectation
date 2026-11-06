"""
glmhmm_model_selection_figures.py
=================================
Two summary figures for the GLM-HMM model selection, built from the CV outputs
already on disk (no refitting):

  Figure 1 - "Why K = 2"   (from glmhmm_kselection_cv.py)
    A  Held-out LL gain over K = 1, per animal (thin lines) and fold mean +- SEM.
    B  Incremental gain of each K over K - 1, per animal, with the number of
       animals that improve.
    C  Fit stability: fraction of EM restarts that reach the best solution,
       per fold.

  Figure 2 - "GLM versions" (from glmhmm_lapse_comparison.py)
    A  Held-out LL of each model relative to the free GLM-HMM (K = 2), per
       animal and fold mean +- SEM.
    B  Zoom on the closest competitor (GLM-HMM with a pure Markov-lapse state)
       vs. the free GLM-HMM, per animal.

All held-out LLs are per observed trial (nats), 5-fold CV with sessions
stratified by animal (same folds everywhere), prior sigma = 1 on GLM weights.

Inputs  (relative to reversalExpectation_py/):
    analysis/glmhmm_kselection_cv/{checkpoint.csv, cv_per_fold.csv}
    analysis/glmhmm_lapse_comparison/{cv_per_animal.csv, cv_per_fold.csv}
Outputs (analysis/glmhmm_model_selection_figures/):
    fig_kselection.png / .pdf
    fig_glm_versions.png / .pdf

Usage (from reversalExpectation_py/):
    python behavior/beh_models/glmhmm_model_selection_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy.stats import wilcoxon         # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
KSEL_DIR = REPO_ROOT / "analysis" / "glmhmm_kselection_cv"
LAPSE_DIR = REPO_ROOT / "analysis" / "glmhmm_lapse_comparison"
OUT_DIR = REPO_ROOT / "analysis" / "glmhmm_model_selection_figures"

# Colors: one highlight (the chosen model) and neutral greys for everything else
INK = "#1A2E2A"
MUTED = "#6B6F6A"
ANIMAL = "#C9C9C3"          # thin per-animal marks
HIGHLIGHT = "#2C5F2D"       # chosen model (K = 2 / free GLM-HMM)
OTHER = "#8A8F88"           # competing models

# Readable names (and free-parameter counts) for the lapse comparison models
MODEL_LABELS = {
    "M1_GLM": "GLM (3)",
    "M2_lapse_const": "GLM + constant lapse (5)",
    "M3_lapse_session": "GLM + lapse(trial in session) (10)",
    "M4_lapse_session_reward": "GLM + lapse(trial, reward rate) (11)",
    "M5_HMM_markov_lapse": "HMM: GLM + Markov lapse state (7)",
    "M6_GLMHMM": "GLM-HMM, K = 2 (9)",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fmt_p(p):
    return "p < 1e-5" if p < 1e-5 else f"p = {p:.1g}"


def panel_label(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom")


def jitter(n, width=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


# ---------------------------------------------------------------------------
# Figure 1: K selection
# ---------------------------------------------------------------------------
def figure_kselection(ksel_dir: Path, out_dir: Path):
    ck = pd.read_csv(ksel_dir / "checkpoint.csv")
    per_fold = pd.read_csv(ksel_dir / "cv_per_fold.csv")

    # Per-animal held-out LL per trial (pooled over that animal's test sessions)
    pa = (ck.groupby(["K", "animal"])[["ll", "n_obs"]].sum()
            .assign(llt=lambda x: x["ll"] / x["n_obs"])["llt"].unstack("K"))
    pf = per_fold.pivot(index="fold", columns="K", values="ll_per_trial")
    Ks = sorted(pf.columns)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})

    # ---- A: gain over K = 1 -----------------------------------------------------
    ax = axes[0]
    gain_a = pa.sub(pa[1], axis=0)
    gain_f = pf.sub(pf[1], axis=0)
    for _, row in gain_a.iterrows():
        ax.plot(Ks, row.values, color=ANIMAL, lw=0.7, zorder=1)
    m, s = gain_f.mean(), gain_f.std(ddof=1) / np.sqrt(len(gain_f))
    ax.errorbar(Ks, m, yerr=s, color=INK, lw=2, marker="o", ms=5, capsize=3, zorder=3)
    ax.plot(2, m[2], "o", color=HIGHLIGHT, ms=9, zorder=4)
    ax.set_xticks(Ks)
    ax.set_xlabel("Number of states K")
    ax.set_ylabel("Held-out LL gain over K = 1\n(nats / trial)")
    ax.set_title("Held-out likelihood")
    ax.text(0.97, 0.05, "grey: individual mice (n = 31)\nblack: mean ± SEM over 5 folds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    panel_label(ax, "A")

    # ---- B: incremental gain K vs K - 1 ----------------------------------------
    ax = axes[1]
    labels = []
    for i, K in enumerate(Ks[1:]):
        d_a = pa[K] - pa[K - 1]
        d_f = pf[K] - pf[K - 1]
        col = HIGHLIGHT if K == 2 else OTHER
        ax.scatter(i + jitter(len(d_a), seed=K), d_a, s=10, color=col, alpha=0.55,
                   lw=0, zorder=2)
        ax.errorbar(i, d_f.mean(), yerr=d_f.std(ddof=1) / np.sqrt(len(d_f)),
                    color=INK, marker="_", ms=22, mew=2, capsize=0, lw=1.5, zorder=3)
        n_up = int((d_a > 0).sum())
        ax.text(i, max(d_a.max(), 0) + 0.004, f"{n_up}/{len(d_a)} mice\n{fmt_p(wilcoxon(d_a).pvalue)}",
                ha="center", va="bottom", fontsize=7.5, color=MUTED)
        labels.append(f"{K - 1}→{K}")
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.01)
    ax.set_xlabel("Step in K")
    ax.set_ylabel("Incremental held-out LL gain\n(nats / trial)")
    ax.set_title("Gain of each extra state")
    panel_label(ax, "B")

    # ---- C: restart stability --------------------------------------------------
    ax = axes[2]
    hmm = per_fold[per_fold["K"] >= 2].assign(frac=lambda x: x["agree"] / x["n_restarts"])
    for i, K in enumerate(sorted(hmm["K"].unique())):
        f = hmm[hmm["K"] == K]["frac"].values
        col = HIGHLIGHT if K == 2 else OTHER
        ax.bar(i, f.mean(), width=0.6, color=col, alpha=0.85, zorder=1)
        ax.scatter(i + jitter(len(f), 0.15, seed=K), f, s=14, color=INK, zorder=2, lw=0)
        ax.text(i, f.mean() + 0.03, f"{f.mean():.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(3), ["2", "3", "4"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Number of states K")
    ax.set_ylabel("Restarts reaching best fit")
    ax.set_title("Fit stability")
    panel_label(ax, "C")

    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_kselection.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: GLM versions (lapse vs. strategy)
# ---------------------------------------------------------------------------
def figure_glm_versions(lapse_dir: Path, out_dir: Path):
    pa = pd.read_csv(lapse_dir / "cv_per_animal.csv").pivot(
        index="animal", columns="model", values="ll_per_trial")
    pf = pd.read_csv(lapse_dir / "cv_per_fold.csv").pivot(
        index="fold", columns="model", values="ll_per_trial")
    ref = "M6_GLMHMM"
    models = list(MODEL_LABELS)                       # bottom-to-top order below
    d_a = pa.sub(pa[ref], axis=0)
    d_f = pf.sub(pf[ref], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [1.7, 1]})

    # ---- A: all models relative to the GLM-HMM ---------------------------------
    ax = axes[0]
    for y, mdl in enumerate(models):
        col = HIGHLIGHT if mdl == ref else OTHER
        ax.scatter(d_a[mdl], y + jitter(len(d_a), 0.15, seed=y), s=10, color=col,
                   alpha=0.55, lw=0, zorder=2)
        mu = d_f[mdl].mean()
        se = d_f[mdl].std(ddof=1) / np.sqrt(len(d_f))
        ax.errorbar(mu, y, xerr=se, color=INK, marker="|", ms=16, mew=2, lw=1.5, zorder=3)
        if mdl != ref:
            n_worse = int((d_a[mdl] < 0).sum())
            ax.text(0.004, y, f"{mu:+.4f}   {n_worse}/{len(d_a)} worse",
                    va="center", fontsize=7.5, color=MUTED)
    ax.axvline(0, color=HIGHLIGHT, lw=1, ls="--", zorder=0)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.set_xlim(right=0.035)
    ax.set_xlabel("Held-out LL relative to GLM-HMM (nats / trial)")
    ax.set_title("Lapse models vs. GLM-HMM")
    ax.text(0.0, -0.3, "(n) = free parameters. Dots: mice; bars: mean ± SEM over 5 folds.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    panel_label(ax, "A")

    # ---- B: closest competitor, per animal -------------------------------------
    ax = axes[1]
    close = "M5_HMM_markov_lapse"
    x = pa[ref].values
    yv = pa[close].values
    diff = (yv - x) * 1e3                             # millinats / trial
    order = np.argsort(diff)
    cols = np.where(diff[order] < 0, HIGHLIGHT, OTHER)
    ax.barh(range(len(diff)), diff[order], color=cols, height=0.75)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks([])
    ax.set_ylabel("Mice (sorted)")
    ax.set_xlabel("Markov lapse − GLM-HMM\n(10⁻³ nats / trial)")
    p = wilcoxon(yv - x).pvalue
    n_better = int((diff < 0).sum())
    ax.set_title("Closest competitor")
    ax.text(0.03, 0.97, f"GLM-HMM better in {n_better}/{len(diff)} mice\nWilcoxon {fmt_p(p)}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK)
    panel_label(ax, "B")

    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_glm_versions.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksel", type=Path, default=KSEL_DIR)
    ap.add_argument("--lapse", type=Path, default=LAPSE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figure_kselection(args.ksel, args.out)
    figure_glm_versions(args.lapse, args.out)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
