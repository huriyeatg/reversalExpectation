"""
session_profiles.py
===================
Cluster the 616 sessions into behavioural "profiles" by the SHAPE of their
GLM-HMM state trajectory across the session.

Each session is represented not by its PNG but by the DATA behind it: the
smoothed P(exploit) curve vs normalized session position, resampled onto a
common grid so every session is an equal-length vector. K-means then groups
these curves; K is estimated by the elbow of the within-cluster inertia and by
the silhouette score. exploit and explore are complementary (sum to 1), so the
exploit curve alone carries the full shape.

Outputs: a per-session cluster assignment CSV, a K-selection diagnostic figure,
and a profiles figure (one centroid curve per cluster + all sessions faintly
behind it). Entry point: run_session_profiles(df, ...).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CREAM = "#FBFAF6"; INK = "#1A2E2A"; GRID = "#D8D5CC"; MUTE = "#6B6B66"
# a distinct colour per cluster
PALETTE = ["#2C5F2D", "#C9472B", "#E8A33D", "#3A6EA5", "#7B5EA7",
           "#4C8C4A", "#B0793A", "#8A5A44"]
_SESS_KEYS = ["animal", "session_file"]


def _font():
    from matplotlib import font_manager
    for name in ("Georgia", "DejaVu Serif", "serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False); return name
        except Exception:
            continue
    return "serif"


# ---------------------------------------------------------------------------
# Build one smoothed exploit curve per session on a common grid
# ---------------------------------------------------------------------------
def build_session_curves(df, state_col="glmhmm_state", exploit_state=0,
                         smooth_window=15, n_points=50, min_trials=30):
    """Return (curves, keys): curves is an (n_sessions x n_points) array of the
    smoothed P(exploit) vs normalized session position; keys is a DataFrame of
    [animal, session_file, n_trials] aligned to the rows of curves.

    exploit_state is the state index whose occupancy defines the curve (default
    0 = the most perseverative state / 'exploit' under the K=2 convention)."""
    need = ["animal", "session_file", "trial_idx", state_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"build_session_curves needs {miss}")
    xs = np.linspace(0, 1, n_points)
    rows, keys = [], []
    for (animal, ses), d in df.groupby(_SESS_KEYS, sort=False):
        d = d.sort_values("trial_idx")
        d = d[d[state_col].notna()]
        if len(d) < min_trials:
            continue
        pos = np.linspace(0, 1, len(d))               # even positions in session
        ind = (d[state_col].to_numpy() == exploit_state).astype(float)
        w = min(smooth_window, len(ind))
        kern = np.ones(w) / w
        sm = np.convolve(ind, kern, mode="same")
        cov = np.convolve(np.ones_like(ind), kern, mode="same")
        sm = sm / np.clip(cov, 1e-9, None)            # edge-corrected moving avg
        rows.append(np.interp(xs, pos, sm))
        keys.append({"animal": animal, "session_file": ses, "n_trials": int(len(d))})
    return np.asarray(rows), pd.DataFrame(keys)


# ---------------------------------------------------------------------------
# K selection + K-means
# ---------------------------------------------------------------------------
def estimate_k(curves, k_range=range(2, 9), seed=0):
    """Fit K-means for each K; return a DataFrame with inertia and silhouette,
    plus the silhouette-optimal K."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(curves)
        sil = silhouette_score(curves, km.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    tab = pd.DataFrame(rows)
    k_best = int(tab.loc[tab["silhouette"].idxmax(), "k"])
    return tab, k_best


def cluster_sessions(curves, k, seed=0):
    """K-means with k clusters; returns (labels, centroids) with clusters
    reordered by mean exploit level (cluster 0 = lowest exploit)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(curves)
    order = np.argsort([curves[km.labels_ == c].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in km.labels_])
    centroids = km.cluster_centers_[order]
    return labels, centroids


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_k_selection(tab, k_best, outfile, font=None):
    font = font or _font()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(CREAM); ax1.set_facecolor(CREAM)
    ax1.plot(tab["k"], tab["inertia"], "-o", color=INK, label="inertia")
    ax1.set_xlabel("number of clusters K", fontsize=10, fontfamily=font)
    ax1.set_ylabel("within-cluster inertia", fontsize=10, fontfamily=font, color=INK)
    ax2 = ax1.twinx()
    ax2.plot(tab["k"], tab["silhouette"], "-s", color="#C9472B", label="silhouette")
    ax2.set_ylabel("silhouette", fontsize=10, fontfamily=font, color="#C9472B")
    ax1.axvline(k_best, ls="--", color="#E8A33D", lw=1.5)
    ax1.text(k_best, ax1.get_ylim()[1], f" K={k_best}", color="#B0793A",
             va="top", fontsize=9, fontfamily=font)
    for a in (ax1, ax2):
        for sp in ("top",):
            a.spines[sp].set_visible(False)
    ax1.set_title("Cluster-count selection", fontsize=12, weight="bold",
                  color=INK, fontfamily=font)
    ax1.grid(True, color=GRID, lw=0.6, alpha=0.6)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


def plot_profiles(curves, labels, centroids, outfile, font=None):
    font = font or _font()
    k = centroids.shape[0]
    xs = np.linspace(0, 1, curves.shape[1])
    ncol = min(k, 4); nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow),
                             squeeze=False)
    for c in range(k):
        ax = axes[c // ncol][c % ncol]
        ax.set_facecolor(CREAM)
        mem = curves[labels == c]
        for row in mem:                              # faint individual sessions
            ax.plot(xs, row, color=PALETTE[c % len(PALETTE)], alpha=0.06, lw=0.8)
        ax.plot(xs, centroids[c], color=PALETTE[c % len(PALETTE)], lw=2.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"Profile {c}  ·  n={len(mem)}  ·  "
                     f"mean exploit {mem.mean():.2f}", fontsize=10,
                     fontfamily=font, color=INK, weight="bold")
        ax.set_xlabel("position in session", fontsize=8.5, fontfamily=font)
        ax.set_ylabel("P(exploit)", fontsize=8.5, fontfamily=font)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for j in range(k, nrow * ncol):                  # hide unused axes
        axes[j // ncol][j % ncol].axis("off")
    fig.patch.set_facecolor(CREAM)
    fig.suptitle(f"Session profiles by exploit trajectory  (K-means, K={k})",
                 fontsize=13, weight="bold", color=INK, fontfamily=font)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return outfile


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_session_profiles(df, state_col="glmhmm_state", exploit_state=0,
                         k=None, k_range=range(2, 9), smooth_window=15,
                         n_points=50, min_trials=30, output_dir="analysis",
                         figs_dir="figs/occupancy", seed=0):
    """Cluster sessions by exploit-trajectory shape. If k is None, K is chosen by
    the silhouette score. Saves the per-session assignment CSV and two figures;
    returns the assignment table + diagnostics."""
    print("\n=== Session profile clustering (exploit trajectory) ===")
    curves, keys = build_session_curves(df, state_col, exploit_state,
                                        smooth_window, n_points, min_trials)
    print(f"  {len(curves)} sessions with >= {min_trials} trials")

    ksel, k_best = estimate_k(curves, k_range, seed)
    if k is None:
        k = k_best
    print(f"  K selection (silhouette-optimal K={k_best}); using K={k}")
    for _, r in ksel.iterrows():
        print(f"    K={int(r['k'])}: inertia={r['inertia']:.2f}, silhouette={r['silhouette']:.3f}")

    labels, centroids = cluster_sessions(curves, k, seed)
    keys = keys.copy(); keys["cluster"] = labels
    counts = keys["cluster"].value_counts().sort_index()
    print("  cluster sizes:", dict(counts))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    keys.to_csv(Path(output_dir) / "session_profiles.csv", index=False)
    f1 = plot_k_selection(ksel, k_best, str(Path(figs_dir) / "session_profiles_kselection.png"))
    f2 = plot_profiles(curves, labels, centroids, str(Path(figs_dir) / "session_profiles.png"))
    print(f"  wrote {f1}\n  wrote {f2}")
    return {"assignments": keys, "centroids": centroids, "k": k,
            "k_selection": ksel, "curves": curves}


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "analysis/bandit_R71_lesion.csv"
    df = pd.read_csv(csv, low_memory=False)
    if "lesioned" in df and "meets_criteria" in df:
        df = df[df["lesioned"].isna() & (df["meets_criteria"] == True)]
    run_session_profiles(df, output_dir=".", figs_dir=".")