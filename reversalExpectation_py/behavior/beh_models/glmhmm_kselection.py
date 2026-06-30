"""
glmhmm_kselection.py
====================
Support for defending the choice of K in the GLM-HMM pipeline.

Two things live here:

1. State-stability analysis. Cross-validated log-likelihood plateaus after
   K=2 (the K=2 -> K=3 and K=3 -> K=4 gains both fall below the fold-to-fold
   noise of ~0.009 LL/trial), so K cannot be chosen on held-out LL alone. The
   argument for K=3 rests on the recovered states being (a) interpretable and
   (b) reproducible. This module quantifies (b): it refits the K-state model
   many times -- across random-init restarts and across CV folds -- matches the
   states between fits with the Hungarian algorithm, and reports how tightly the
   matched states agree (weight cosine similarity, and dwell-time consistency).
   A stable K=3 solution (matched cosine ~> 0.99 across restarts/folds) is the
   evidence that the third state is a real strategy, not an init artifact.

2. Slide figures, in the project palette:
     - plot_cv_curve:        held-out LL/trial vs K, elbow + plateau annotated.
     - plot_state_stability: per-state weights with restart spread (error bars).

Integration assumptions (from hmmGlm.py):
    fit_fn(choices, inputs, masks, K=..., seed=..., **kw) -> ssm model
        e.g. functools.partial(fit_global_glmhmm, glm_weights=glm_weights)
        or any wrapper that takes K and seed and returns a fitted model.
    weights_fn(model) -> per-state weights: either an ndarray (K, M) or a
        DataFrame whose non-"state" columns are the weights (glmhmm_weights).
    transmat_fn(model) -> (K, K) transition matrix (glmhmm_transition_matrix).
The fit calls are NOT executed in this file's self-test; only the matching and
plotting logic is exercised on synthetic weight matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Project palette
# ---------------------------------------------------------------------------
CREAM = "#FBFAF6"
INK = "#1A2E2A"
GREEN = "#2C5F2D"
CORAL = "#C9472B"
GOLD = "#E8A33D"
GRID = "#D8D5CC"
STATE_COLORS = [GREEN, CORAL, GOLD, "#5B6C8F"]  # engaged, random, biased, +1


def _serif():
    """Prefer Georgia (present on the user's Windows box); fall back gracefully."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Georgia", "Times New Roman", "DejaVu Serif", "serif"):
        if name in available or name == "serif":
            return name
    return "serif"


# ===========================================================================
# State matching
# ===========================================================================
def _as_weight_matrix(W) -> np.ndarray:
    """Coerce glmhmm_weights output (DataFrame with a 'state' col, or ndarray)
    into a numeric (K, M) array of per-state weights."""
    if isinstance(W, pd.DataFrame):
        cols = [c for c in W.columns if c != "state"]
        return W[cols].to_numpy(dtype=float)
    return np.asarray(W, dtype=float)


def match_states(ref_W: np.ndarray, W: np.ndarray):
    """Permute the states of `W` to best match `ref_W`.

    Assignment uses Euclidean distance between weight vectors (Hungarian,
    minimising total distance). Euclidean is used rather than cosine because a
    disengaged/random state sits near the origin: its *direction* is undefined,
    so cosine would mis-match it, whereas distance correctly pairs small-to-small.

    Returns (perm, sims) where W[perm] is aligned to ref_W and sims[i] is the
    cosine similarity of ref state i with its matched W state (reported for
    diagnostics; meaningful only for states with non-negligible weight norm).
    """
    ref_W = np.asarray(ref_W, float)
    W = np.asarray(W, float)
    # pairwise squared Euclidean distance, D[i, j] = ||ref_i - W_j||^2
    D = ((ref_W[:, None, :] - W[None, :, :]) ** 2).sum(-1)
    row, col = linear_sum_assignment(D)             # minimise total distance
    rn = ref_W / (np.linalg.norm(ref_W, axis=1, keepdims=True) + 1e-12)
    wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    K = ref_W.shape[0]
    perm = np.empty(K, dtype=int)
    sims = np.empty(K, dtype=float)
    for i, j in zip(row, col):
        perm[i] = j
        sims[i] = float(rn[i] @ wn[j])
    return perm, sims


def _allpairs_stability(W_list, P_list=None, norm_thresh=0.5):
    """All-pairs state agreement across a list of fits.

    Aligns every fit to fit 0 (Euclidean matching), then per state reports:
      - mean weight vector and its SD across fits (the figure's error bars),
      - mean weight norm (how "structured" the state is),
      - pairwise cosine similarity across fits.
    Cosine is only a sensible stability metric for structured states; a state
    whose mean norm < `norm_thresh` (e.g. the disengaged/random state) is judged
    stable by its small, consistent magnitude instead. `overall_*` cosine stats
    are computed over structured states only.
    """
    W_list = [_as_weight_matrix(W) for W in W_list]
    n = len(W_list)
    K = W_list[0].shape[0]

    ref = W_list[0]
    aligned_W = [ref]
    aligned_P = [np.asarray(P_list[0], float)] if P_list is not None else None
    for k in range(1, n):
        perm, _ = match_states(ref, W_list[k])
        aligned_W.append(W_list[k][perm])
        if P_list is not None:
            aligned_P.append(np.asarray(P_list[k], float)[np.ix_(perm, perm)])
    A = np.stack(aligned_W)                                  # (n, K, M)

    sims_per_state = [[] for _ in range(K)]
    for i in range(n):
        for j in range(i + 1, n):
            wi = A[i] / (np.linalg.norm(A[i], axis=1, keepdims=True) + 1e-12)
            wj = A[j] / (np.linalg.norm(A[j], axis=1, keepdims=True) + 1e-12)
            for s in range(K):
                sims_per_state[s].append(float(wi[s] @ wj[s]))

    mean_W = A.mean(0)                                       # (K, M)
    norms = np.linalg.norm(mean_W, axis=1)                   # (K,)
    structured = norms >= norm_thresh
    struct_sims = [v for s in range(K) if structured[s] for v in sims_per_state[s]]

    stats = {
        "K": K, "n_fits": n,
        "mean_weights": mean_W,                              # (K, M)
        "sd_weights": A.std(0),                              # (K, M)
        "state_norm": norms.tolist(),
        "structured": structured.tolist(),
        "mean_cos_per_state": [float(np.mean(s)) if s else float("nan")
                               for s in sims_per_state],
        "min_cos_per_state": [float(np.min(s)) if s else float("nan")
                              for s in sims_per_state],
        "max_weight_sd": float(A.std(0).max()),
        "overall_mean_cos_structured": float(np.mean(struct_sims)) if struct_sims else float("nan"),
        "overall_min_cos_structured": float(np.min(struct_sims)) if struct_sims else float("nan"),
        "aligned_weights": A,
    }

    if P_list is not None:
        Ps = np.stack(aligned_P)                            # (n, K, K)
        self_p = np.einsum("nkk->nk", Ps)
        dwell = 1.0 / (1.0 - np.clip(self_p, 0, 1 - 1e-9))
        stats["dwell_mean"] = dwell.mean(0).tolist()
        stats["dwell_sd"] = dwell.std(0).tolist()
        stats["aligned_transitions"] = Ps
    return stats


def collect_fits(datasets, K, seeds, fit_fn, weights_fn, transmat_fn=None,
                 fit_kwargs=None):
    """Fit the K-state model once per (dataset, seed) and collect weights and
    (optionally) transition matrices.

    - Restart stability: pass one dataset and many seeds, e.g.
        collect_fits([(ch, inp, mk)], K=3, seeds=range(10), ...)
    - Fold stability: pass the per-fold TRAINING sets and a single seed each,
        collect_fits(fold_train_sets, K=3, seeds=[0]*n_folds, ...)
    Each dataset is a (choices, inputs, masks) tuple. seeds is a list matched
    to datasets, OR a list of seeds applied to a single dataset (restart mode).
    """
    fit_kwargs = fit_kwargs or {}
    if len(datasets) == 1 and len(seeds) > 1:
        datasets = datasets * len(seeds)               # restart mode
    assert len(datasets) == len(seeds), "datasets and seeds length mismatch"
    W_list, P_list = [], [] if transmat_fn else None
    for (ch, inp, mk), sd in zip(datasets, seeds):
        model = fit_fn(ch, inp, mk, K=K, seed=int(sd), **fit_kwargs)
        W_list.append(_as_weight_matrix(weights_fn(model)))
        if transmat_fn:
            P_list.append(np.asarray(transmat_fn(model), float))
    return _allpairs_stability(W_list, P_list)


# ===========================================================================
# Figures
# ===========================================================================
def _style_ax(ax):
    ax.set_facecolor(CREAM)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_cv_curve(cv_df, chosen_K=3, baselines=None, n_folds=None,
                  ll_col="test_ll", k_col="K", fold_col="fold",
                  outfile="cv_kselection.png", show_deltas=True):
    """Slide figure: held-out LL/trial vs K with elbow + plateau annotated.

    cv_df: long DataFrame with columns [k_col, fold_col, ll_col], one row per
           (K, fold). LL is per-trial held-out log-likelihood (negative).
    baselines: optional {"GLM": val, "lapse": val} horizontal references.
    chosen_K: highlighted as the final model.
    """
    font = _serif()
    g = cv_df.groupby(k_col)[ll_col]
    Ks = np.sort(cv_df[k_col].unique())
    mean = g.mean().reindex(Ks).to_numpy()
    n = g.count().reindex(Ks).to_numpy()
    sem = (g.std(ddof=1).reindex(Ks).to_numpy()) / np.sqrt(np.maximum(n, 1))
    if n_folds is None:
        n_folds = int(np.nanmax(n))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(CREAM)
    _style_ax(ax)

    ax.errorbar(Ks, mean, yerr=sem, color=GREEN, lw=2.2, marker="o", ms=7,
                mfc=GREEN, mec=INK, mew=0.8, capsize=4, zorder=3,
                label="GLM-HMM (test LL/trial)")

    # highlight chosen K
    ci = int(np.where(Ks == chosen_K)[0][0])
    ax.plot([chosen_K], [mean[ci]], marker="o", ms=14, mfc=GOLD, mec=CORAL,
            mew=2.2, zorder=4)
    ax.annotate(f"final model\nK = {chosen_K}",
                xy=(chosen_K, mean[ci]), xytext=(chosen_K + 0.18, mean[ci] - 0.0),
                fontsize=10.5, fontfamily=font, color=INK, va="center",
                fontweight="bold")

    # baselines
    if baselines:
        for name, val in baselines.items():
            ax.axhline(val, ls=(0, (5, 3)), color=INK, lw=1.1, alpha=0.55)
            ax.annotate(name, xy=(Ks[0], val), xytext=(Ks[0] - 0.02, val),
                        fontsize=9, color=INK, va="bottom", ha="left",
                        fontfamily=font, alpha=0.8)

    # delta annotations (elbow vs plateau)
    if show_deltas and len(Ks) >= 2:
        deltas = np.diff(mean)
        span = (np.nanmax(mean) - np.nanmin(mean)) or 1.0
        for i, d in enumerate(deltas):
            x = (Ks[i] + Ks[i + 1]) / 2
            y = max(mean[i], mean[i + 1]) + span * 0.06
            big = abs(d) >= 0.01
            txt = ("+" if d >= 0 else "\u2212") + f"{abs(d):.4f}".rstrip("0").rstrip(".")
            ax.annotate(txt, xy=(x, y), fontsize=9.5,
                        color=CORAL if big else INK,
                        alpha=1.0 if big else 0.55,
                        fontfamily=font, ha="center", va="bottom",
                        fontweight="bold" if big else "normal")
        # plateau bracket over the last segment(s)
        if len(Ks) >= 3:
            ylo = min(mean[1:]) - (max(mean) - min(mean)) * 0.18
            ax.annotate("", xy=(Ks[1], ylo), xytext=(Ks[-1], ylo),
                        arrowprops=dict(arrowstyle="-", color=INK, lw=1.0,
                                        alpha=0.5))
            ax.text((Ks[1] + Ks[-1]) / 2, ylo, "plateau (within fold noise)",
                    ha="center", va="top", fontsize=9, color=INK, alpha=0.7,
                    fontfamily=font)

    ax.set_xticks(Ks)
    ax.set_xlabel("number of latent states  K", fontsize=11.5, color=INK,
                  fontfamily=font)
    ax.set_ylabel("held-out log-likelihood / trial", fontsize=11.5, color=INK,
                  fontfamily=font)
    ax.set_title("Cross-validated model selection", fontsize=15, color=INK,
                 fontfamily=font, fontweight="bold", loc="left", pad=12)
    cap = (f"session-level CV stratified by animal  ·  {n_folds} folds  ·  "
           "mean \u00b1 SEM across folds")
    fig.text(0.125, 0.005, cap, fontsize=8.5, color=INK, alpha=0.7,
             fontfamily=font)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_state_stability(stats, regressor_names, state_labels=None,
                         outfile="state_stability.png"):
    """Slide figure: per-state weights with restart/fold spread (error bars).

    stats: output of collect_fits (uses stats['aligned_weights'], shape (n,K,M)).
    A tight clustering (small error bars, well-separated state profiles) is the
    visual evidence that the K states are reproducible.
    """
    font = _serif()
    W = np.asarray(stats["aligned_weights"], float)      # (n, K, M)
    n, K, M = W.shape
    mean = W.mean(0)                                     # (K, M)
    sd = W.std(0)                                        # (K, M)
    if state_labels is None:
        defaults = ["engaged", "random", "side-biased", "state 4"]
        state_labels = [defaults[k] if k < len(defaults) else f"state {k+1}"
                        for k in range(K)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(CREAM)
    _style_ax(ax)

    x = np.arange(M)
    width = 0.8 / K
    for k in range(K):
        ax.bar(x + (k - (K - 1) / 2) * width, mean[k], width,
               yerr=sd[k], color=STATE_COLORS[k % len(STATE_COLORS)],
               ec=INK, lw=0.6, capsize=3, label=state_labels[k], zorder=3,
               error_kw=dict(ecolor=INK, lw=1.0))
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(regressor_names, fontsize=10, color=INK, fontfamily=font)
    ax.set_ylabel("weight (log-odds of right)", fontsize=11.5, color=INK,
                  fontfamily=font)
    ax.set_title(f"State weights are stable across {n} fits",
                 fontsize=15, color=INK, fontfamily=font, fontweight="bold",
                 loc="left", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="best", prop={"family": font})
    sub = (f"mean \u00b1 SD across {n} fits  ·  structured-state matched cosine "
           f"min = {stats['overall_min_cos_structured']:.3f}  ·  "
           f"max weight SD = {stats['max_weight_sd']:.3f}")
    fig.text(0.125, 0.005, sub, fontsize=8.5, color=INK, alpha=0.7,
             fontfamily=font)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight")
    plt.close(fig)
    return outfile


# ===========================================================================
# Self-test: matching + plotting logic on synthetic data (no ssm / no real fits)
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- 1. matching logic ----
    # 3 "true" states with distinct profiles: [bias, prev_choice, wsls]
    true_W = np.array([[0.1,  2.5,  1.8],    # engaged: strong persev + wsls
                       [0.0,  0.05, 0.02],   # random: ~flat
                       [1.4,  0.3,  0.1]])   # side-biased: big bias
    # simulate 10 restarts: same states, permuted + small noise
    W_list, P_list = [], []
    for _ in range(10):
        perm = rng.permutation(3)
        noise = 0.04 * rng.standard_normal((3, 3))
        W_list.append(true_W[perm] + noise)
        P = np.full((3, 3), 0.02); np.fill_diagonal(P, 0.96)
        P = P[np.ix_(perm, perm)] + 0.001 * rng.standard_normal((3, 3))
        P = np.clip(P, 1e-3, None); P /= P.sum(1, keepdims=True)
        P_list.append(P)

    stats = _allpairs_stability(W_list, P_list)
    print("=== matching self-test ===")
    print("state norms:", [round(v, 2) for v in stats["state_norm"]],
          "| structured:", stats["structured"])
    print("per-state mean cosine:", [round(v, 4) for v in stats["mean_cos_per_state"]])
    print("structured-state min cosine (should be ~1.0):",
          round(stats["overall_min_cos_structured"], 4))
    print("max weight SD across fits:", round(stats["max_weight_sd"], 4))
    print("dwell mean (expect ~25 = 1/(1-0.96)):",
          [round(v, 1) for v in stats["dwell_mean"]])
    # structured states (engaged, biased) must align nearly perfectly;
    # the near-origin random state is judged by its small, low-variance norm.
    assert stats["overall_min_cos_structured"] > 0.97, "structured states mis-aligned!"
    rec = stats["mean_weights"]
    # recovered mean weights should match the true profiles up to the permutation
    perm, _ = match_states(rec, true_W)
    assert np.abs(rec - true_W[perm]).max() < 0.1, "weights not recovered!"
    assert stats["max_weight_sd"] < 0.1, "weight SD too large for stable scenario!"

    # also test an unstable scenario (states genuinely differ across fits)
    W_unstable = [true_W + 0.9 * rng.standard_normal((3, 3)) for _ in range(8)]
    s2 = _allpairs_stability(W_unstable)
    print("unstable scenario: structured min cosine (should be lower):",
          round(s2["overall_min_cos_structured"], 3),
          "| max weight SD:", round(s2["max_weight_sd"], 3))

    # ---- 2. CV curve figure (ILLUSTRATIVE numbers; replace with real cv_df) ----
    base = {1: -0.620, 2: -0.589, 3: -0.5875, 4: -0.5854}   # ~remembered deltas
    rows = []
    for K, m in base.items():
        for f in range(5):
            rows.append({"K": K, "fold": f,
                         "test_ll": m + 0.0035 * rng.standard_normal()})
    cv_df = pd.DataFrame(rows)
    f1 = plot_cv_curve(cv_df, chosen_K=3,
                       baselines={"GLM": -0.612, "lapse": -0.605}, n_folds=5,
                       outfile="/mnt/user-data/outputs/cv_kselection_preview.png")
    print("wrote", f1)

    # ---- 3. stability figure ----
    f2 = plot_state_stability(
        stats, regressor_names=["bias", "prev_choice", "wsls"],
        state_labels=["engaged", "random", "side-biased"],
        outfile="/mnt/user-data/outputs/state_stability_preview.png")
    print("wrote", f2)
    print("ALL SELF-TESTS PASSED")