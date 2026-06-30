"""
anticipation_test.py
====================
Does the animal ANTICIPATE the block switch, i.e. does it choose the currently
"worse" option (the soon-to-be-better side) more as time-in-block grows, BEFORE
any outcome signals the switch?

The clean, exogenous window is L_Random: after the performance criterion is met
the contingency has NOT changed and outcomes are still consistent with the
current block, yet the switch draws nearer on a random (geometric) clock. So:

    P(choose worse | tau) rising with tau WITHIN L_Random  ==  anticipation,
    because a purely reactive policy has no outcome-based reason to drift there.

This module:
  1. prepare_anticipation : per-trial tau, in_Lrandom, worse_side, chose_worse.
  2. pworse_curve         : P(chose_worse | tau) in L_Random, with Wilson CIs,
                            optionally restricted to one GLM-HMM state.
  3. pstate_curve         : P(state | tau) in L_Random (the disengagement check
                            that separates "within-engaged drift" from "the animal
                            just slipped into the random/biased state").
  4. simulate_reactive_glmhmm : a YOKED forward simulation of the fitted
                            (reactive) GLM-HMM under the real block schedule. Its
                            P(worse | tau) is the reactive null band; the data's
                            EXCESS over it is the anticipatory component.
  5. plot_anticipation    : slide-style figure in the project palette.

The GLM-HMM state per trial is supplied by the caller as a df column (hard MAP in
`state_col`, soft posterior in `p_engaged_col`); see the runner note at the
bottom for how to fill it from glmhmm_posteriors. The reactive simulator uses the
`ashwood_wsls` coding documented in `_ashwood_step` — confirm it matches your
build_glmhmm_inputs coding (that is the one integration point to check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:  # production: the task's reward-probability table
    from preprocessing.presentation_codes import REWARD_PROBS as _DEFAULT_RP
except Exception:  # standalone / self-test: caller passes reward_probs
    _DEFAULT_RP = None

# ---- palette ----
CREAM = "#FBFAF6"; INK = "#1A2E2A"; GREEN = "#2C5F2D"
CORAL = "#C9472B"; GOLD = "#E8A33D"; GRID = "#D8D5CC"


def _serif():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Georgia", "Times New Roman", "DejaVu Serif", "serif"):
        if n in avail or n == "serif":
            return n
    return "serif"


# ===========================================================================
# 1. Per-trial anticipation columns
# ===========================================================================
def prepare_anticipation(df: pd.DataFrame, reward_probs: dict | None = None,
                         tau_min: int = 0) -> pd.DataFrame:
    """
    Add columns: t_block, raw_tau, in_Lrandom, worse_side, chose_worse.

    raw_tau    = within-block trial index - block_trial_to_crit
                 (<0 criterion phase, >=0 = L_Random window).
    in_Lrandom = raw_tau >= tau_min and the block has a criterion.
    worse_side = the lower-reward-probability side of the CURRENT rule
                 (-1 left / +1 right); the side that becomes better after switch.
    chose_worse= 1.0 if choice == worse_side, 0.0 if chose better, NaN on miss.
    """
    rp = reward_probs if reward_probs is not None else _DEFAULT_RP
    if rp is None:
        raise ValueError("reward_probs not available; pass reward_probs=REWARD_PROBS.")
    df = df.copy()

    # within-block trial index and trials-since-criterion
    df["t_block"] = (df.groupby(["animal", "session_file", "block_idx"], sort=False)
                       .cumcount().astype(float))
    ttc = df["block_trial_to_crit"].to_numpy(dtype=float)
    raw_tau = df["t_block"].to_numpy() - ttc
    df["raw_tau"] = raw_tau
    df["in_Lrandom"] = (raw_tau >= tau_min) & ~np.isnan(ttc)

    # worse side from the current rule's reward probs
    n_rules = int(df["n_rules"].iloc[0])
    prob_map = rp.get(n_rules, rp.get(2))
    worse = np.full(len(df), np.nan)
    rules = df["rule"].to_numpy(dtype=float)
    for ridx, (lp, rp_) in prob_map.items():
        better = -1.0 if lp > rp_ else 1.0          # higher-prob side
        worse[rules == ridx] = -better              # lower-prob side
    df["worse_side"] = worse

    ch = df["choice"].to_numpy(dtype=float)
    cw = np.where(np.isnan(ch) | np.isnan(worse), np.nan, (ch == worse).astype(float))
    df["chose_worse"] = cw
    return df


# ===========================================================================
# 2-3. Curves vs tau with Wilson CIs
# ===========================================================================
def _wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (vectorised)."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, centre - half, centre + half


def _binary_curve(values, tau, tau_grid):
    """P(value==1 | tau) over tau_grid, with counts, ignoring NaNs."""
    values = np.asarray(values, float); tau = np.asarray(tau, float)
    k = np.array([np.nansum(values[tau == t] == 1) for t in tau_grid], float)
    n = np.array([np.sum((tau == t) & ~np.isnan(values)) for t in tau_grid], float)
    p, lo, hi = _wilson(k, n)
    return pd.DataFrame({"tau": tau_grid, "n": n.astype(int), "p": p, "lo": lo, "hi": hi})


def pworse_curve(df, state_col=None, state_value=None, max_tau=20, min_n=20):
    """
    P(chose_worse | tau) inside the L_Random window.

    state_col / state_value : restrict to one GLM-HMM state (e.g. the engaged
    state) to test within-state drift; None = all states pooled.
    Rows with fewer than `min_n` trials are returned but flagged via `n`.
    """
    sub = df[df["in_Lrandom"]].copy()
    if state_col is not None and state_value is not None:
        sub = sub[sub[state_col] == state_value]
    grid = np.arange(0, max_tau + 1)
    out = _binary_curve(sub["chose_worse"].to_numpy(), sub["raw_tau"].to_numpy(), grid)
    out["enough"] = out["n"] >= min_n
    return out


def pstate_curve(df, state_col, target_value, max_tau=20):
    """P(state == target_value | tau) inside L_Random (disengagement check)."""
    sub = df[df["in_Lrandom"]].copy()
    isval = (sub[state_col].to_numpy(dtype=float) == target_value).astype(float)
    grid = np.arange(0, max_tau + 1)
    return _binary_curve(isval, sub["raw_tau"].to_numpy(), grid)


# ===========================================================================
# 4. Yoked reactive GLM-HMM forward simulation (the reactive null)
# ===========================================================================
def _ashwood_step(last_choice, last_reward):
    """
    One-trial `ashwood_wsls` regressors from the last NON-missed trial, as
    log-odds-of-RIGHT predictors (sign convention of glmhmm_weights):

        prev_choice : last choice (+1 right, -1 left); >0 weight = perseveration
        wsls        : +1 if (won & chose right) or (lost & chose left) -> predict right
                      i.e. wsls = last_choice if rewarded else -last_choice
    Returns (prev_choice, wsls); both 0 before any valid trial.
    CONFIRM this matches build_glmhmm_inputs' coding for `ashwood_wsls`.
    """
    if last_choice is None or np.isnan(last_choice):
        return 0.0, 0.0
    pc = float(last_choice)
    wsls = pc if last_reward == 1 else -pc
    return pc, wsls


def _stationary(P):
    """Stationary distribution of a row-stochastic matrix P (power iteration)."""
    K = P.shape[0]
    pi = np.full(K, 1.0 / K)
    for _ in range(2000):
        nxt = pi @ P
        if np.max(np.abs(nxt - pi)) < 1e-12:
            break
        pi = nxt
    return pi / pi.sum()


def simulate_reactive_glmhmm(df, weights_right, trans_mat, n_sims=200, rng=None,
                             reward_probs: dict | None = None):
    """
    Yoked forward simulation of the fitted reactive GLM-HMM.

    For every real session the simulated agent runs under the SAME block schedule
    (same per-trial rule and the same miss mask), so tau / L_Random align exactly
    with the data; only the policy is the model's. Choices feed back into the
    WSLS / prev_choice regressors (closed loop), and rewards are drawn from the
    real per-trial rule. Transitions are the sticky Markov chain (not input-driven),
    matching the current model.

    weights_right : (K, M) per-state weights as log-odds of RIGHT (glmhmm_weights),
                    columns ordered [bias, prev_choice, wsls].
    trans_mat     : (K, K) transition matrix.

    Returns (c_sim, z_sim), each (n_trials_total, n_sims) aligned to df row
    order. c_sim is NaN on real-miss trials; z_sim is the simulated state, so the
    null band can be restricted to the same state as the data curve.
    """
    rng = rng or np.random.default_rng()
    rp = reward_probs if reward_probs is not None else _DEFAULT_RP
    if rp is None:
        raise ValueError("reward_probs not available; pass reward_probs=REWARD_PROBS.")
    W = np.asarray(weights_right, float)              # (K, M)
    P = np.asarray(trans_mat, float)
    K = W.shape[0]
    pi0 = _stationary(P)
    n_rules = int(df["n_rules"].iloc[0])
    prob_map = rp.get(n_rules, rp.get(2))

    ch = df["choice"].to_numpy(dtype=float)
    rule = df["rule"].to_numpy(dtype=float)
    miss = np.isnan(ch)
    c_sim = np.full((len(df), n_sims), np.nan)
    z_sim = np.full((len(df), n_sims), -1, dtype=int)   # simulated state per trial

    # session boundaries (contiguous rows per session, df is trial-ordered)
    sess_id = (df["animal"].astype(str) + "|" + df["session_file"].astype(str)).to_numpy()
    bounds = np.flatnonzero(np.r_[True, sess_id[1:] != sess_id[:-1]])
    starts = list(bounds); ends = list(bounds[1:]) + [len(df)]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for s in range(n_sims):
        for a, b in zip(starts, ends):
            z = rng.choice(K, p=pi0)
            last_c, last_r = np.nan, np.nan
            for t in range(a, b):
                z_sim[t, s] = z
                if miss[t]:
                    z = rng.choice(K, p=P[z])         # state still evolves on a miss
                    continue
                pc, wsls = _ashwood_step(last_c, last_r)
                x = np.array([1.0, pc, wsls])
                p_right = sigmoid(float(W[z] @ x))
                c = 1.0 if rng.random() < p_right else -1.0   # +1 right / -1 left
                c_sim[t, s] = c
                ridx = rule[t]
                if not np.isnan(ridx):
                    lp, rp_ = prob_map.get(int(ridx), (0.5, 0.5))
                    prew = lp if c == -1 else rp_
                    last_r = 1.0 if rng.random() < prew else 0.0
                else:
                    last_r = np.nan
                last_c = c
                z = rng.choice(K, p=P[z])
    return c_sim, z_sim


def reactive_pworse_band(df, c_sim, z_sim=None, state_value=None, max_tau=20):
    """
    P(chose_worse | tau) in L_Random for each simulation -> reactive null band.

    If z_sim and state_value are given, the band is computed only over simulated
    trials in that state, so it is the correct null for a state-restricted data
    curve (e.g. compare the engaged-data curve to the engaged-only reactive null,
    not to a null that also includes the side-biased state's worse-choices).
    Returns DataFrame(tau, mean, lo, hi); lo/hi are 2.5/97.5 percentiles over sims.
    """
    worse = df["worse_side"].to_numpy(dtype=float)
    inwin = df["in_Lrandom"].to_numpy(bool)
    tau = df["raw_tau"].to_numpy(dtype=float)
    grid = np.arange(0, max_tau + 1)
    n_sims = c_sim.shape[1]
    curves = np.full((len(grid), n_sims), np.nan)
    for j in range(n_sims):
        cw = np.where(np.isnan(c_sim[:, j]) | np.isnan(worse), np.nan,
                      (c_sim[:, j] == worse).astype(float))
        keep = inwin.copy()
        if z_sim is not None and state_value is not None:
            keep = keep & (z_sim[:, j] == state_value)
        for i, t in enumerate(grid):
            m = keep & (tau == t) & ~np.isnan(cw)
            if m.sum() > 0:
                curves[i, j] = np.nanmean(cw[m])
    return pd.DataFrame({
        "tau": grid,
        "mean": np.nanmean(curves, axis=1),
        "lo": np.nanpercentile(curves, 2.5, axis=1),
        "hi": np.nanpercentile(curves, 97.5, axis=1),
    })


# ===========================================================================
# 4b. Inference on the excess slope: per-animal replication + animal bootstrap
#     (reuses an EXISTING (c_sim, z_sim); no re-simulation needed)
# ===========================================================================
def _excess_slope(cur, band, min_n=20):
    """Slope (and mean) of the data-minus-null excess vs tau, over tau bins that
    have >= min_n data trials and a finite null. Shared by the runner and the
    per-animal / bootstrap routines so they all define the slope identically."""
    p = cur["p"].to_numpy(float)
    nm = band["mean"].to_numpy(float)
    tau = cur["tau"].to_numpy(float)
    enough = cur["enough"].to_numpy(bool) if "enough" in cur else (cur["n"].to_numpy() >= min_n)
    ok = enough & np.isfinite(p) & np.isfinite(nm)
    if ok.sum() < 3:
        return np.nan, np.nan
    exc = (p - nm)[ok]
    slope = float(np.polyfit(tau[ok], exc, 1)[0])
    return slope, float(np.nanmean(exc))


def excess_slope_stats(df, c_sim, z_sim, state_value, max_tau=14):
    """
    Precompute per-(animal, tau) sufficient statistics for the engaged excess
    slope, so per-animal and bootstrap slopes are pure aggregation of an EXISTING
    simulation -- the expensive forward sim is never repeated.

    For each animal a and tau t it stores, for the DATA engaged curve, the worse
    count Dk[a,t] and trial count Dn[a,t]; and for the reactive null, per sim j,
    the worse count Sk[a,j,t] and trial count Sn[a,j,t] over simulated trials in
    `state_value`. Summing these over any set of animals reproduces exactly the
    pooled data curve and the pooled per-sim null rate, so the slope on any
    animal subset matches pworse_curve / reactive_pworse_band on that subset.
    """
    animals = list(pd.unique(df["animal"]))
    ai_of = pd.factorize(df["animal"])[0]
    grid = np.arange(0, max_tau + 1)
    nA, nT, nS = len(animals), len(grid), c_sim.shape[1]

    worse = df["worse_side"].to_numpy(float)
    inwin = df["in_Lrandom"].to_numpy(bool)
    tau = df["raw_tau"].to_numpy(float)
    state = df["glmhmm_state"].to_numpy(float)
    cw_data = df["chose_worse"].to_numpy(float)

    intau = inwin & (tau >= 0) & (tau <= max_tau)
    ti = np.zeros(len(df), dtype=int)
    ti[intau] = tau[intau].astype(int)

    # --- data engaged worse-counts ---
    dmask = intau & (state == state_value) & ~np.isnan(cw_data)
    flat = ai_of[dmask] * nT + ti[dmask]
    Dn = np.bincount(flat, minlength=nA * nT).reshape(nA, nT).astype(float)
    Dk = np.bincount(flat, weights=(cw_data[dmask] == 1).astype(float),
                     minlength=nA * nT).reshape(nA, nT)

    # --- reactive null, per simulation ---
    Sk = np.zeros((nA, nS, nT)); Sn = np.zeros((nA, nS, nT))
    for j in range(nS):
        cj = c_sim[:, j]
        okj = intau & (z_sim[:, j] == state_value) & ~np.isnan(cj) & ~np.isnan(worse)
        flatj = ai_of[okj] * nT + ti[okj]
        Sn[:, j, :] = np.bincount(flatj, minlength=nA * nT).reshape(nA, nT)
        wj = (cj[okj] == worse[okj]).astype(float)
        Sk[:, j, :] = np.bincount(flatj, weights=wj, minlength=nA * nT).reshape(nA, nT)

    return {"animals": animals, "grid": grid, "Dk": Dk, "Dn": Dn, "Sk": Sk, "Sn": Sn}


def _slope_from_stats(stats, animal_idx, min_n=20):
    """Excess slope (and mean excess) for the animals at positions `animal_idx`
    (an array of indices into stats['animals']; repeats allowed for bootstrap)."""
    idx = np.asarray(animal_idx, int)
    Dk = stats["Dk"][idx].sum(0); Dn = stats["Dn"][idx].sum(0)         # (nT,)
    Sk = stats["Sk"][idx].sum(0); Sn = stats["Sn"][idx].sum(0)         # (nS, nT)
    grid = stats["grid"]
    data_p = np.divide(Dk, Dn, out=np.full_like(Dn, np.nan), where=Dn > 0)
    sim_rate = np.divide(Sk, Sn, out=np.full_like(Sn, np.nan), where=Sn > 0)
    null_mean = np.nanmean(sim_rate, axis=0)                           # (nT,)
    ok = (Dn >= min_n) & np.isfinite(data_p) & np.isfinite(null_mean)
    if ok.sum() < 3:
        return np.nan, np.nan
    exc = (data_p - null_mean)[ok]
    slope = float(np.polyfit(grid[ok], exc, 1)[0])
    return slope, float(np.nanmean(exc))


def per_animal_excess_slope(stats, min_n=10):
    """Per-animal engaged excess slope -- the subject-level replication of the
    anticipation effect (analogous to the per-animal hazard slope b of belief_vhr).
    Returns DataFrame(animal, slope, mean_excess). min_n is per-animal, so it is
    lower than the pooled default."""
    rows = []
    for ai, a in enumerate(stats["animals"]):
        s, me = _slope_from_stats(stats, [ai], min_n=min_n)
        rows.append({"animal": a, "slope": s, "mean_excess": me})
    return pd.DataFrame(rows)


def bootstrap_excess_slope(stats, n_boot=2000, rng=None, min_n=20):
    """Bootstrap the GLOBAL engaged excess slope over animals (the unit of
    replication). Returns dict(slope_boot, ci_lo, ci_hi, p_boot). p_boot is the
    two-sided bootstrap p that the slope differs from 0."""
    rng = rng or np.random.default_rng(0)
    nA = len(stats["animals"])
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = rng.integers(0, nA, nA)            # resample animals with replacement
        boot[b], _ = _slope_from_stats(stats, draw, min_n=min_n)
    valid = boot[np.isfinite(boot)]
    if len(valid) == 0:
        return {"slope_boot": boot, "ci_lo": np.nan, "ci_hi": np.nan, "p_boot": np.nan}
    lo, hi = np.percentile(valid, [2.5, 97.5])
    frac_le = float(np.mean(valid <= 0))
    p_boot = min(1.0, 2 * min(frac_le, 1 - frac_le))
    return {"slope_boot": boot, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": float(p_boot)}


def anticipation_inference(df, c_sim, z_sim, state_value, max_tau=14, min_n=20,
                           per_animal_min_n=10, n_boot=2000, rng=None, verbose=True):
    """
    Full inference on the engaged anticipation signal, reusing an EXISTING
    (c_sim, z_sim) so it adds essentially no runtime over the forward sim.

    Returns a dict with the global excess slope, its animal-bootstrap 95% CI and
    two-sided p, and the per-animal slope table with a one-sided Wilcoxon
    signed-rank test that the slope is > 0 (subject-level replication).
    """
    rng = rng or np.random.default_rng(0)
    stats = excess_slope_stats(df, c_sim, z_sim, state_value, max_tau=max_tau)
    g_slope, g_mean = _slope_from_stats(stats, np.arange(len(stats["animals"])), min_n=min_n)
    bs = bootstrap_excess_slope(stats, n_boot=n_boot, rng=rng, min_n=min_n)
    pa = per_animal_excess_slope(stats, min_n=per_animal_min_n)

    sl = pa["slope"].to_numpy(float); sl = sl[np.isfinite(sl)]
    wil_W, wil_p = np.nan, np.nan
    if len(sl) >= 6:
        from scipy.stats import wilcoxon
        try:
            wil_W, wil_p = wilcoxon(sl, alternative="greater")
        except ValueError:
            pass

    res = {
        "global_slope": g_slope, "global_mean_excess": g_mean,
        "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "p_boot": bs["p_boot"],
        "n_animals": int(len(sl)),
        "median_slope": float(np.nanmedian(sl)) if len(sl) else np.nan,
        "frac_pos": float(np.mean(sl > 0)) if len(sl) else np.nan,
        "wilcoxon_W": float(wil_W), "wilcoxon_p": float(wil_p),
        "per_animal": pa, "boot": bs["slope_boot"],
    }
    if verbose:
        print(f"  global excess slope = {g_slope:+.4f}/trial   "
              f"95% CI [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]   p_boot = {bs['p_boot']:.3g}")
        print(f"  per-animal slope    : median = {res['median_slope']:+.4f}, "
              f"{res['frac_pos']*100:.0f}% > 0, Wilcoxon p = {res['wilcoxon_p']:.3g} "
              f"(n = {res['n_animals']} animals)")
    return res


# ===========================================================================
# 5. Slide figure
# ===========================================================================
def plot_anticipation(curve_all, curve_engaged, sim_band, pstate=None,
                      state_name="engaged", outfile="anticipation_test.png",
                      max_tau=14, inference=None):
    """
    Two-panel slide figure:
      (a) P(choose worse | tau) in L_Random: data (all states + engaged) vs the
          reactive GLM-HMM null band. Data rising above the band = anticipation.
      (b) P(state | tau): is any pre-switch drift within the engaged state, or
          just disengagement?

    inference : optional dict from anticipation_inference(); if given, the
        excess slope, its 95% CI / bootstrap p and the per-animal summary are
        annotated on panel (a).
    """
    font = _serif()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor(CREAM)

    ax = axes[0]; ax.set_facecolor(CREAM)
    # reactive null band
    sb = sim_band[sim_band["tau"] <= max_tau]
    ax.fill_between(sb["tau"], sb["lo"], sb["hi"], color=CORAL, alpha=0.16,
                    label="reactive GLM-HMM (null band)", zorder=1)
    ax.plot(sb["tau"], sb["mean"], color=CORAL, lw=1.4, ls="--", alpha=0.8, zorder=2)
    # data: all states
    ca = curve_all[(curve_all["tau"] <= max_tau) & curve_all["enough"]]
    ax.plot(ca["tau"], ca["p"], color=INK, lw=1.6, marker="o", ms=4, alpha=0.55,
            label="data — all states", zorder=3)
    # data: engaged only
    ce = curve_engaged[(curve_engaged["tau"] <= max_tau) & curve_engaged["enough"]]
    ax.errorbar(ce["tau"], ce["p"], yerr=[ce["p"] - ce["lo"], ce["hi"] - ce["p"]],
                color=GREEN, lw=2.2, marker="o", ms=6, capsize=3, zorder=4,
                label=f"data — {state_name} state")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("τ  =  trials since criterion  (L_Random window)", fontsize=11, fontfamily=font, color=INK)
    ax.set_ylabel("P(choose the worse option)", fontsize=11, fontfamily=font, color=INK)
    ax.set_title("Anticipation test", fontsize=14, fontweight="bold", loc="left",
                 fontfamily=font, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", prop={"family": font})

    # annotate the excess-slope inference (mirrors the belief_vhr deltaBIC label)
    if inference is not None:
        def _p(x):
            if x is None or not np.isfinite(x):
                return "n/a"
            return "<0.001" if x < 1e-3 else f"{x:.3f}"
        sl = inference.get("global_slope", np.nan)
        lo = inference.get("ci_lo", np.nan); hi = inference.get("ci_hi", np.nan)
        lines = [f"excess slope = {sl:+.4f}/trial"]
        if np.isfinite(lo) and np.isfinite(hi):
            lines.append(f"95% CI [{lo:+.4f}, {hi:+.4f}],  p = {_p(inference.get('p_boot'))}")
        med = inference.get("median_slope", np.nan)
        fpos = inference.get("frac_pos", np.nan)
        nan = inference.get("n_animals", 0)
        if np.isfinite(med):
            lines.append(f"per-animal: median {med:+.4f},  {fpos*100:.0f}% > 0")
            lines.append(f"Wilcoxon p = {_p(inference.get('wilcoxon_p'))}  (n = {nan})")
        ax.text(0.97, 0.05, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, fontfamily=font, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85,
                          edgecolor=GREEN, linewidth=1.0))

    ax2 = axes[1]; ax2.set_facecolor(CREAM)
    if pstate is not None:
        ps = pstate[pstate["tau"] <= max_tau]
        ax2.fill_between(ps["tau"], ps["lo"], ps["hi"], color=GREEN, alpha=0.15)
        ax2.plot(ps["tau"], ps["p"], color=GREEN, lw=2.0, marker="o", ms=5)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.yaxis.grid(True, color=GRID, lw=0.8); ax2.set_axisbelow(True)
    ax2.set_xlabel("τ  (L_Random window)", fontsize=11, fontfamily=font, color=INK)
    ax2.set_ylabel(f"P({state_name} state)", fontsize=11, fontfamily=font, color=INK)
    ax2.set_title("Disengagement check", fontsize=14, fontweight="bold", loc="left",
                  fontfamily=font, color=INK, pad=10)

    fig.tight_layout()
    fig.savefig(outfile, facecolor=CREAM, bbox_inches="tight")
    plt.close(fig)
    return outfile


# ===========================================================================
# Self-test: reactive vs anticipatory synthetic agents (no ssm / no real data)
# ===========================================================================
def _make_synth(n_animals=4, blocks_per_session=12, rng=None, anticipatory=False,
                drift=0.06):
    """Build a synthetic trial table with the real schema, choices from either a
    reactive WSLS policy or the same policy plus a tau-driven drift toward the
    worse side during L_Random."""
    rng = rng or np.random.default_rng(0)
    RP = {2: {0: (0.7, 0.1), 1: (0.1, 0.7)}}          # rule 0 = high-left
    rows = []
    for ani in range(n_animals):
        for ses in range(3):
            rule = rng.integers(0, 2)
            last_c, last_r = np.nan, np.nan
            for blk in range(blocks_per_session):
                ttc = int(rng.integers(8, 18))         # trials to criterion
                lran = int(rng.geometric(1 / 8))       # L_Random ~ geometric mean 8
                lran = min(lran, 30)
                lp, rp = RP[2][rule]
                better = -1 if lp > rp else 1
                worse = -better
                for tb in range(ttc + lran):
                    raw_tau = tb - ttc
                    # reactive WSLS policy with the REAL GLM-HMM design
                    # [bias, prev_choice, wsls] only -- the better side is tracked
                    # through WSLS (win -> stay), NOT a 'better' regressor.
                    pc = 0.0 if np.isnan(last_c) else last_c
                    wsls = 0.0 if np.isnan(last_c) else (last_c if last_r == 1 else -last_c)
                    logit = 0.0 + 1.8 * pc + 1.2 * wsls
                    if anticipatory and raw_tau >= 0:
                        logit += drift * raw_tau * worse     # drift toward worse with tau
                    p_right = 1 / (1 + np.exp(-logit))
                    c = 1 if rng.random() < p_right else -1
                    prew = lp if c == -1 else rp
                    r = 1 if rng.random() < prew else 0
                    rows.append(dict(animal=ani, session_file=f"s{ses}", block_idx=blk,
                                     block_trial_to_crit=ttc,
                                     block_trial_random_added=lran,
                                     n_rules=2, rule=rule, choice=c, rewarded=r))
                    last_c, last_r = c, r
                rule = 1 - rule                          # switch
    return pd.DataFrame(rows), RP


if __name__ == "__main__":
    RP = {2: {0: (0.7, 0.1), 1: (0.1, 0.7)}}
    # reactive null uses the SAME design as the generator: [bias, prev_choice, wsls]
    W_NULL = np.array([[0.0, 1.8, 1.2]])      # K=1, design-matched
    TRANS = np.array([[1.0]])
    MAXT = 12

    def excess_profile(df, drift=0.0, anticipatory=False, seed=2):
        df, _ = _make_synth(rng=np.random.default_rng(seed), anticipatory=anticipatory,
                            drift=drift) if df is None else (df, None)
        df = prepare_anticipation(df, reward_probs=RP)
        cur = pworse_curve(df, max_tau=MAXT, min_n=30)
        c_sim, z_sim = simulate_reactive_glmhmm(df, W_NULL, TRANS, n_sims=120,
                                                rng=np.random.default_rng(7), reward_probs=RP)
        band = reactive_pworse_band(df, c_sim, max_tau=MAXT)
        ok = cur["enough"].to_numpy() & ~np.isnan(band["mean"].to_numpy())
        excess = (cur["p"] - band["mean"]).to_numpy()[ok]
        taus = cur["tau"].to_numpy()[ok]
        mean_excess = float(np.nanmean(excess))
        exc_slope = float(np.polyfit(taus, excess, 1)[0])
        return df, cur, band, mean_excess, exc_slope

    print("=== reactive agent: data should sit INSIDE its own reactive null band ===")
    df_r, cur_r, band_r, me_r, es_r = excess_profile(None, anticipatory=False, seed=2)
    print(f" mean excess over null = {me_r:+.3f}  (expect ~0)")
    print(f" excess slope vs tau   = {es_r:+.4f}/trial  (expect ~0)")

    print("\n=== anticipatory agent: data should rise ABOVE the reactive null band ===")
    df_a, cur_a, band_a, me_a, es_a = excess_profile(None, anticipatory=True, drift=0.08, seed=2)
    print(f" mean excess over null = {me_a:+.3f}  (expect > 0)")
    print(f" excess slope vs tau   = {es_a:+.4f}/trial  (expect > 0)")

    assert abs(me_r) < 0.05, "reactive agent should sit inside its null band!"
    assert abs(es_r) < 0.006, "reactive excess should be ~flat!"
    assert me_a > me_r + 0.05, "anticipation not separated from reactive null!"
    assert es_a > 0.006, "anticipatory excess should grow with tau!"

    fig = plot_anticipation(cur_a, cur_a, band_a, pstate=None, state_name="engaged",
                            outfile="/mnt/user-data/outputs/anticipation_test_preview.png")
    print("\nwrote", fig)
    print(f"\nseparation: reactive mean-excess {me_r:+.3f} vs anticipatory {me_a:+.3f}")
    print("ALL SELF-TESTS PASSED")