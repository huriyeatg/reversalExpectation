"""
belief_vhr_lrandom.py
=====================
The definitive anticipation test: does a variable hazard H(tau) = sigmoid(a + b*tau)
improve the belief model SPECIFICALLY within the L_Random window?

belief_vhr nests the constant-hazard belief model at b = 0 (H = sigmoid(a)), so
the whole question is the slope b. We fit both (b free vs b = 0) by maximising the
likelihood RESTRICTED to L_Random trials (score_mask = in_Lrandom) while letting
the belief propagate causally through every trial, and compare them by a
likelihood-ratio test. b > 0 with a significant LR == the animal runs an internal
clock that erodes its block belief as the switch draws near, beyond anything the
outcomes alone justify -- anticipation.

Reported:
  - global pooled fit: b_hat, LR, p (boundary-corrected for b >= 0), dBIC,
    dLL/trial on L_Random;
  - per-animal pooled fits (animal as the unit of replication): distribution of
    b, Wilcoxon signed-rank vs 0;
  - figure: inferred behavioural hazard H(tau) vs the constant baseline and the
    empirical TASK hazard, plus the per-animal b distribution.

Run from the revExp environment (belief_vhr / bayesian_models must import).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, wilcoxon
from joblib import Parallel, delayed

# single source of truth for the belief math + the task-hazard diagnostic
from behavior.beh_models.belief_vhr import belief_vhr_trial_loglikes, empirical_hazard

CREAM = "#FBFAF6"; INK = "#1A2E2A"; GREEN = "#2C5F2D"
CORAL = "#C9472B"; GOLD = "#E8A33D"; GRID = "#D8D5CC"


def _serif():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Georgia", "Times New Roman", "DejaVu Serif", "serif"):
        if n in avail or n == "serif":
            return n
    return "serif"


def select_naive_meets_criteria(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reproduce the GLM-HMM inclusion filter WITHOUT importing hmmGlm (which
    pulls in ssm, absent from the revExp env): naive (lesioned is NaN), all
    phases unified, meets_criteria == True. Keeps the belief_vhr test
    apples-to-apples with the GLM-HMM (same 616-session set)."""
    out = df[df["lesioned"].isna()]
    out = out[out["meets_criteria"] == True].copy()
    if verbose:
        n_ses = out.groupby(["animal", "session_file"]).ngroups
        print(f"[vhr_lrandom select] naive + meets_criteria: {len(out)} trials, "
              f"{out['animal'].nunique()} animals, {n_ses} sessions.")
    return out


# ===========================================================================
# Build per-session (c, r, n_rules, tau, L_Random mask)
# ===========================================================================
def _session_inputs(df_ses: pd.DataFrame):
    """(choice, reward, n_rules, tau, lrandom_mask) for one session.

    tau = trials since criterion, clipped to >= 0 (same as prepare_vhr);
    lrandom_mask = raw_tau >= 0 (post-criterion window) AND the block reached
    criterion. The belief still propagates through every trial; only L_Random
    trials are scored.
    """
    c = df_ses["choice"].to_numpy(dtype=float)
    r = df_ses["rewarded"].to_numpy(dtype=float)
    n_rules = int(df_ses["n_rules"].iloc[0])
    t_block = df_ses.groupby("block_idx", sort=False).cumcount().to_numpy(dtype=float)
    ttc = df_ses["block_trial_to_crit"].to_numpy(dtype=float)
    raw_tau = t_block - ttc
    tau = np.clip(np.where(np.isnan(ttc), 0.0, raw_tau), 0.0, None)
    mask = (raw_tau >= 0) & ~np.isnan(ttc)
    return c, r, n_rules, tau, mask


def prepare_sessions(df: pd.DataFrame):
    """List of (animal, (c, r, n_rules, tau, mask)) over all sessions."""
    out = []
    for (animal, _ses), df_ses in df.groupby(["animal", "session_file"], sort=False):
        ci = _session_inputs(df_ses)
        if ci[4].sum() > 0:                 # keep sessions with >=1 L_Random trial
            out.append((animal, ci))
    return out


# ===========================================================================
# Pooled fit + likelihood-ratio test
# ===========================================================================
def _pooled_nll(theta, sessions, b_free):
    a = theta[0]
    b, beta = (theta[1], theta[2]) if b_free else (0.0, theta[1])
    if beta <= 0 or b < 0:
        return 1e12
    tot = 0.0
    for (c, r, n_rules, tau, mask) in sessions:
        ll = belief_vhr_trial_loglikes([a, b, beta], c, r, n_rules, tau)
        tot += -np.nansum(ll[mask])
    return tot


def _fit_pooled(sessions, b_free, n_restarts, rng):
    bounds = [(-12, 12), (0, 5), (1e-3, 100)] if b_free else [(-12, 12), (1e-3, 100)]
    best = (np.inf, None)
    for _ in range(n_restarts):
        x0 = ([rng.uniform(-4, 0), rng.uniform(0, 0.5), rng.uniform(0.5, 15)]
              if b_free else [rng.uniform(-4, 0), rng.uniform(0.5, 15)])
        res = minimize(_pooled_nll, x0, args=(sessions, b_free),
                       method="L-BFGS-B", bounds=bounds)
        if res.fun < best[0]:
            best = (res.fun, res.x)
    return best


def lr_test(sessions, n_restarts: int = 8, rng=None) -> dict:
    """belief_vhr (b free) vs belief (b = 0) on the pooled L_Random likelihood."""
    rng = rng or np.random.default_rng(0)
    nll_full, par_full = _fit_pooled(sessions, True, n_restarts, rng)
    nll_red, par_red = _fit_pooled(sessions, False, n_restarts, rng)
    n_scored = sum(int(np.sum(m & ~np.isnan(c))) for (c, r, nr, tau, m) in sessions)
    LR = max(2 * (nll_red - nll_full), 0.0)
    p_naive = float(chi2.sf(LR, 1))
    a, b, beta = par_full
    a0 = float(par_red[0])
    return dict(a=float(a), b=float(b), beta=float(beta), a0=a0,
                LR=float(LR), p_naive=p_naive, p_boundary=0.5 * p_naive,
                dBIC=float(LR - np.log(max(n_scored, 1))),     # BIC_red - BIC_full
                dLL_per_trial=float((nll_red - nll_full) / max(n_scored, 1)),
                n_scored=int(n_scored), base_hazard=float(expit(a)))


def _animal_test(animal, sessions, n_restarts, seed):
    res = lr_test(sessions, n_restarts=n_restarts, rng=np.random.default_rng(seed))
    res["animal"] = animal
    return res


def per_animal_test(df: pd.DataFrame, n_restarts: int = 6, n_jobs: int = -1):
    """Fit the pooled LR test within each animal (animal = unit of replication).
    Returns (per_animal_df, summary_dict with the Wilcoxon test on b vs 0)."""
    sess = prepare_sessions(df)
    by_animal = {}
    for animal, ci in sess:
        by_animal.setdefault(animal, []).append(ci)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_animal_test)(a, s, n_restarts, i)
        for i, (a, s) in enumerate(by_animal.items()))
    pa = pd.DataFrame(rows).sort_values("animal").reset_index(drop=True)
    b = pa["b"].to_numpy()
    # one-sided Wilcoxon signed-rank that b > 0 (drop exact zeros)
    nz = b[np.abs(b) > 1e-9]
    if len(nz) >= 1:
        try:
            W, p = wilcoxon(nz, alternative="greater")
        except ValueError:
            W, p = np.nan, np.nan
    else:
        W, p = np.nan, 1.0
    summary = dict(n_animals=len(pa), median_b=float(np.median(b)),
                   frac_b_pos=float(np.mean(b > 0)),
                   wilcoxon_W=float(W), wilcoxon_p=float(p),
                   median_dLL=float(np.median(pa["dLL_per_trial"])))
    return pa, summary


# ===========================================================================
# Figure
# ===========================================================================
def plot_vhr_lrandom(df, global_res, per_animal_df, max_tau: int = 20,
                     outfile="belief_vhr_lrandom.png"):
    """(a) inferred behavioural hazard H(tau) vs constant baseline vs task hazard;
       (b) per-animal b distribution with the Wilcoxon test."""
    font = _serif()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=200,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(CREAM)

    # --- (a) hazards ---
    ax = axes[0]; ax.set_facecolor(CREAM)
    taus = np.arange(0, max_tau + 1)
    H_fit = expit(global_res["a"] + global_res["b"] * taus)
    H_const = expit(global_res["a0"]) * np.ones_like(taus, float)
    try:
        haz, p_hat = empirical_hazard(df, max_lrandom=max_tau)
        solid = haz["n_at_risk"].to_numpy() >= 20
        ax.errorbar(haz.loc[solid, "tau"], haz.loc[solid, "hazard"],
                    yerr=haz.loc[solid, "se"], fmt="o", color=GOLD, ms=4, lw=1.4,
                    capsize=2, alpha=0.9, label="task hazard (empirical)", zorder=2)
    except Exception:
        pass
    ax.plot(taus, H_const, ls=(0, (5, 3)), color=CORAL, lw=1.8,
            label=f"belief (constant H = {expit(global_res['a0']):.3f})", zorder=3)
    ax.plot(taus, H_fit, color=GREEN, lw=2.6,
            label=f"belief_vhr  H(τ)=σ(a+bτ),  b={global_res['b']:.3f}", zorder=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("τ  =  trials since criterion  (L_Random)", fontsize=11, fontfamily=font, color=INK)
    ax.set_ylabel("hazard  H(τ)", fontsize=11, fontfamily=font, color=INK)
    ax.set_ylim(0, max(0.5, float(np.nanmax(H_fit)) * 1.1))
    sig = ("p<0.001" if global_res["p_boundary"] < 1e-3
           else f"p={global_res['p_boundary']:.3g}")
    ax.set_title(f"Inferred vs task hazard  (ΔBIC={global_res['dBIC']:+.0f}, {sig})",
                 fontsize=13, fontweight="bold", loc="left", fontfamily=font, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", prop={"family": font})

    # --- (b) per-animal b ---
    ax2 = axes[1]; ax2.set_facecolor(CREAM)
    b = per_animal_df["b"].to_numpy()
    x = np.random.default_rng(0).normal(1.0, 0.05, size=len(b))
    ax2.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax2.scatter(x, b, color=GREEN, s=28, alpha=0.75, edgecolor=INK, linewidth=0.4, zorder=3)
    ax2.boxplot(b, positions=[1], widths=0.45, showfliers=False,
                medianprops=dict(color=CORAL, lw=2),
                boxprops=dict(color=INK), whiskerprops=dict(color=INK),
                capprops=dict(color=INK))
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.yaxis.grid(True, color=GRID, lw=0.8); ax2.set_axisbelow(True)
    ax2.set_xticks([1]); ax2.set_xticklabels(["per animal"], fontfamily=font, fontsize=10)
    ax2.set_ylabel("hazard slope  b", fontsize=11, fontfamily=font, color=INK)
    med = float(np.median(b)); frac = float(np.mean(b > 0))
    ax2.set_title(f"b by animal  (median={med:.3f}, {frac*100:.0f}% > 0)",
                  fontsize=13, fontweight="bold", loc="left", fontfamily=font, color=INK, pad=10)

    fig.tight_layout()
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight")
    plt.close(fig)
    return outfile


# ===========================================================================
# Runner
# ===========================================================================
def run_vhr_lrandom_test(df, output_dir="analysis", figs_dir="figs", n_restarts_global: int = 8,
                         n_restarts_animal: int = 6, n_jobs: int = -1):
    """Full test: global LR + per-animal Wilcoxon + figure. Returns (global_res,
    per_animal_df, summary)."""
    sessions = prepare_sessions(df)
    print(f"[vhr_lrandom] {len(sessions)} sessions with an L_Random window")
    ci_list = [ci for (_animal, ci) in sessions]          # strip animal tag
    global_res = lr_test(ci_list, n_restarts=n_restarts_global)
    print(f"[vhr_lrandom] GLOBAL: b={global_res['b']:.3f}  "
          f"LR={global_res['LR']:.1f}  p={global_res['p_boundary']:.3g}  "
          f"dBIC={global_res['dBIC']:+.1f}  dLL/trial={global_res['dLL_per_trial']:+.4f}  "
          f"(n={global_res['n_scored']})")
    pa, summary = per_animal_test(df, n_restarts=n_restarts_animal, n_jobs=n_jobs)
    print(f"[vhr_lrandom] PER ANIMAL: median b={summary['median_b']:.3f}, "
          f"{summary['frac_b_pos']*100:.0f}% > 0, "
          f"Wilcoxon p={summary['wilcoxon_p']:.3g}")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    figs = Path(figs_dir); figs.mkdir(parents=True, exist_ok=True)
    pa.to_csv(out / "belief_vhr_lrandom_per_animal.csv", index=False)
    fig = plot_vhr_lrandom(df, global_res, pa, outfile=str(figs / "belief_vhr_lrandom.png"))
    print(f"[vhr_lrandom] wrote {fig}")
    return global_res, pa, summary


if __name__ == "__main__":
    import sys
    CSV = sys.argv[1] if len(sys.argv) > 1 else "analysis/bandit_R71_lesion.csv"
    df = pd.read_csv(CSV)
    # same inclusion criteria as the GLM-HMM, WITHOUT importing ssm
    df = select_naive_meets_criteria(df)
    run_vhr_lrandom_test(df)