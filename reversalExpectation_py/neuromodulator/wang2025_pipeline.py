"""
wang2025_pipeline.py
====================
Python replication of the fluorescence analysis pipeline in

    Wang, Ortega, ... Kwan (2025) Sci. Adv. 11, eadr9916
    "Frontal noradrenergic and cholinergic transients exhibit distinct
     spatiotemporal dynamics during competitive decision-making"

Source of truth: the official MATLAB code (github.com/Kwan-Lab/wangortega2025),
not only the Methods text. Where paper and code disagree, the PAPER is the
default and the code's value is exposed as a parameter (see DIVERGENCES).

Replicated steps (original .m file in brackets):
  1. dF/F with F0 = 10th percentile in a centered 2-min moving window [calc_dFF.m]
  2. Interpolation to 10 ms + alignment to cue, window [-4, 6] s  [linear_regr.m, align_signal.m]
  3. Mean per 100-ms bin between -3 and 5 s (80 bins)              [linear_regr.m]
  4. Multiple linear regression per bin (OLS, 14 predictors)        [MP_GRAB_MLR.m + regstats]
       C(n+1..n-2), R(n+1..n-2), X=C*R (n+1..n-2), rMA (20 trials), normalized rCum
       Coding: choice ipsi=0 / contra=1; reward 0/1; miss -> NaN (row excluded)
  5. Unit "modulated" by a predictor: p<0.01 in >=3 consecutive bins
     or >= N bins in total                                          [getRegautoCorrData.m]
  6. Fraction of significant units per bin vs. 1% (binomial)       [MP_plot_regr_fluo.m]
  7. Overlap between variables: 2x2 chi-square of independence     [chi2ind.m]
     Conditional probabilities Pr(v1|v2) per session + median test [MP_GRAB_tempComp.m]
  8. Hierarchical clustering of outcome coefficients
     (complete linkage, distance = 1 - Pearson, 2 clusters) and
     assignment to group 1/2 by the sign of the mean cluster AUC    [regCoef_cluster.m,
                                                                     MP_GRAB_temporalCorrSummary.m]
  9. Time-to-peak / peak value: smooth (5 pts) -> interp 10 ms ->
     maximum of the sign-oriented curve, t>0; per session the
     median and variance of time-to-peak and median peak value;
     NE vs ACh compared with rank-sum                               [MP_GRAB_temporalCorrSummary.m,
                                                                     MP_GRAB_tempComp.m]
 10. Switch regression: s_n = 0 stay, +1 switch to contra, -1 switch to ipsi
     dF/F = b0 + b1 s_n + b2 r_n + b3 s_n r_n + b4 rMA + b5 rCum    [Methods, switch equation]

DIVERGENCES paper vs. code (parameters below):
  - Total-bins threshold: paper = 10, code = 8               -> MIN_TOTAL_SIG_BINS
  - AUC window used to group clusters: paper = 0-2 s,
    regCoef_cluster.m uses 1-2.5 s only to order clusters   -> AUC_WINDOW
  - Test on fraction of ROIs: paper says chi-square vs 1%,
    the figure code uses a per-bin binomial test             -> both are returned
  - Percentile: MATLAB prctile interpolates differently than numpy;
    numerically negligible difference in F0                  (documented, not parametrized)

Deliberately NOT replicated (does not apply to this dataset):
  - 28x28 ROI grid over the 2P FOV (here each trace column of the CSV = 1 unit)
  - End-of-session cut by 3-choice entropy < 1 [cutoff.m]: in a reversal
    bandit, exploitation produces low entropy by design.
    Implemented as `running_entropy()` but off by default.
  - Baseline-jump correction by change points [MP_GRAB_normalizedFF.m]:
    artifact of triggering one .tif per trial in 2P; optional.

Dependencies: numpy, pandas, scipy (statsmodels NOT needed; the revExp env lacks it).
Running `python wang2025_pipeline.py` executes a self-test on synthetic data.
"""

from __future__ import annotations

import warnings

import numpy as np

# np.trapezoid exists from numpy 2.0 and np.trapz was removed in numpy 2.4;
# the ssm env needs numpy < 2, so support both
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Parameters (values from the paper / original code)
# ---------------------------------------------------------------------------
BASELINE_WIN_S      = 120.0          # window for F0 (2 min, centered)
BASELINE_PRCTILE    = 10.0           # 10th percentile
INTERP_DT           = 0.01           # interpolate to 10 ms before aligning
REG_WINDOW          = np.round(np.arange(-3.0, 5.0 + 1e-9, 0.1), 10)  # bin edges
ALIGN_PAD_S         = 1.0            # alignment window = [-3-1, 5+1]
PVAL_THRESH         = 0.01           # threshold per coefficient and per bin
MIN_CONSEC_SIG_BINS = 3              # >= 3 consecutive significant bins
MIN_TOTAL_SIG_BINS  = 10             # paper = 10 (code = 8)
RMA_WINDOW          = 20             # reward rate: 20-trial moving average
N_CLUSTERS          = 2
CLUSTER_T_RANGE     = (-3.0, 5.0)    # time range used in clusterdata
AUC_WINDOW          = (0.0, 2.0)     # paper: AUC 0-2 s defines group 1 (+) / group 2 (-)

PREDICTOR_NAMES = [
    "C(n+1)", "C(n)", "C(n-1)", "C(n-2)",
    "R(n+1)", "R(n)", "R(n-1)", "R(n-2)",
    "X(n+1)", "X(n)", "X(n-1)", "X(n-2)",
    "rMA", "rCum",
]   # same order as concat_event() in MP_GRAB_MLR.m; output col 0 = intercept


# ===========================================================================
# 1. dF/F  [calc_dFF.m]
# ===========================================================================
def calc_dff(f: np.ndarray, frame_rate: float,
             win_s: float = BASELINE_WIN_S,
             prctile: float = BASELINE_PRCTILE) -> tuple[np.ndarray, np.ndarray]:
    """dF/F = (F - F0) / F0, F0 = 10th percentile in a centered 2-min window.

    Replicates the .m indices: idx1 = max(1, round(j - win/2)),
    idx2 = min(N, round(j + win/2)); the window shrinks at the edges.
    Returns (dff, baseline).
    """
    f = np.asarray(f, dtype=float)
    win = win_s * frame_rate
    half = int(round(win / 2))
    # centered pandas rolling with min_periods=1 reproduces the window that
    # shrinks at the edges; size = 2*half + 1 samples
    baseline = (pd.Series(f)
                  .rolling(window=2 * half + 1, center=True, min_periods=1)
                  .quantile(prctile / 100.0, interpolation="linear")
                  .to_numpy())
    dff = (f - baseline) / baseline
    return dff, baseline


def check_raw_trace(f: np.ndarray) -> dict:
    """QC before calc_dff: dF/F only makes sense if F is positive raw
    fluorescence. If the trace is already processed (centered at 0, negative),
    dividing by F0 ~ 0 blows up -> exactly the ~1e10 symptom seen before."""
    f = np.asarray(f, dtype=float)
    q = np.nanpercentile(f, [0.1, 10, 50])
    return {
        "min": np.nanmin(f), "p10": q[1], "median": q[2],
        "frac_nonpositive": float(np.mean(f <= 0)),
        "looks_raw": bool(q[0] > 0),        # all positive -> probably raw F
    }


# ===========================================================================
# 2-3. Alignment and binning  [align_signal.m, linear_regr.m]
# ===========================================================================
def interpolate_signal(t: np.ndarray, signal: np.ndarray, dt: float = INTERP_DT):
    """Linear interp1 onto a regular 10-ms grid (as in linear_regr.m)."""
    inter_t = np.arange(t[0], t[-1] + dt / 2, dt)
    inter_sig = interp1d(t, signal, kind="linear", bounds_error=False)(inter_t)
    return inter_t, inter_sig


def align_signal(t: np.ndarray, signal: np.ndarray, event_times: np.ndarray,
                 window: tuple[float, float]):
    """Literal translation of align_signal.m (1-based -> 0-based indices).

    Returns sig_by_trial (n_time x n_events) and t_by_trial (n_time,).
    NaN events (miss) or out-of-range events -> NaN column.
    """
    dt = np.nanmean(np.diff(t))
    k0, k1 = int(round(window[0] / dt)), int(round(window[1] / dt))
    t_by_trial = dt * np.arange(k0, k1 + 1)
    out = np.full((len(t_by_trial), len(event_times)), np.nan)
    n = len(signal)
    for j, ev in enumerate(event_times):
        if np.isnan(ev):
            continue
        event_idx = int(np.sum(ev >= t))           # 1-based in MATLAB
        start, stop = event_idx + k0, event_idx + k1  # 1-based, inclusive
        if start > 0 and stop <= n:
            out[:, j] = signal[start - 1:stop]
    return out, t_by_trial


def bin_aligned(sig_by_trial: np.ndarray, t_by_trial: np.ndarray,
                edges: np.ndarray = REG_WINDOW):
    """Mean per 100-ms bin, with the same indices as linear_regr.m:
    idx1 = sum(t <= center - dt/2), idx2 = sum(t < center + dt/2)."""
    step = np.nanmean(np.diff(edges))
    centers = np.arange(edges[0] + step / 2, edges[-1] - step / 2 + 1e-9, step)
    binned = np.full((len(centers), sig_by_trial.shape[1]), np.nan)
    for i, c in enumerate(centers):
        idx1 = int(np.sum(t_by_trial <= c - step / 2))   # 1-based
        idx2 = int(np.sum(t_by_trial < c + step / 2))
        seg = sig_by_trial[max(idx1 - 1, 0):idx2, :]
        # trials without an imaging stamp are all-NaN columns: expected, silence
        # numpy's "Mean of empty slice" RuntimeWarning
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            binned[i] = np.nanmean(seg, axis=0)
    return binned, centers


# ===========================================================================
# 4. Design matrix and regression  [MP_GRAB_MLR.m, regstats]
# ===========================================================================
def _lag(x: np.ndarray, k: int) -> np.ndarray:
    """k>0: value k trials back; k<0: |k| trials ahead (NaN at the edges)."""
    out = np.full_like(x, np.nan, dtype=float)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def encode_choice_ipsi_contra(choice_lr: np.ndarray, recording_site: str) -> np.ndarray:
    """choice_lr: -1 = left, +1 = right, NaN = miss.
    Returns ipsi=0 / contra=1 relative to the recorded hemisphere."""
    c = np.asarray(choice_lr, dtype=float)
    out = np.full_like(c, np.nan)
    if recording_site == "left":
        out[c == -1], out[c == 1] = 0, 1
    elif recording_site == "right":
        out[c == -1], out[c == 1] = 1, 0
    else:
        raise ValueError("recording_site must be 'left' or 'right'")
    return out


def build_design_matrix(choice_ic: np.ndarray, rewarded: np.ndarray,
                        responded: np.ndarray) -> np.ndarray:
    """14 predictors in the order of MP_GRAB_MLR.m (no intercept).

    choice_ic : ipsi 0 / contra 1 / NaN miss
    rewarded  : 1 rewarded, 0 not (for all trials; miss = 0)
    responded : bool, trial with a response
    """
    c = np.asarray(choice_ic, dtype=float)
    r_all = np.asarray(rewarded, dtype=float)                  # trials.reward
    r = np.where(responded, r_all, np.nan)                      # miss -> NaN
    C = [_lag(c, -1), c, _lag(c, 1), _lag(c, 2)]
    R = [_lag(r, -1), r, _lag(r, 1), _lag(r, 2)]
    X = [ci * ri for ci, ri in zip(C, R)]
    n = len(c)
    rma = np.array([r_all[:k + 1].sum() / (k + 1) if k < RMA_WINDOW
                    else r_all[k - RMA_WINDOW + 1:k + 1].sum() / RMA_WINDOW
                    for k in range(n)])                         # includes trial n
    tot = r_all.sum()
    rcum = np.cumsum(r_all) / tot if tot > 0 else np.full(n, np.nan)
    return np.column_stack(C + R + X + [rma, rcum])


def build_switch_design(choice_ic: np.ndarray, rewarded: np.ndarray,
                        responded: np.ndarray) -> np.ndarray:
    """Switch regression from Methods: s_n (0 stay, +1 to contra, -1 to ipsi),
    r_n, s_n*r_n, rMA, rCum."""
    c = np.asarray(choice_ic, dtype=float)
    prev = _lag(c, 1)
    s = np.where(c == prev, 0.0, np.where(c == 1, 1.0, -1.0))
    s[np.isnan(c) | np.isnan(prev)] = np.nan
    full = build_design_matrix(choice_ic, rewarded, responded)
    r = full[:, 5]
    return np.column_stack([s, r, s * r, full[:, 12], full[:, 13]])


def ols_regstats(y: np.ndarray, X: np.ndarray):
    """Equivalent to regstats(y, X, 'linear'): intercept + X, rows with NaN
    excluded, two-tailed t p-values. Returns (beta, pval), NaN if the
    matrix is singular (like the .m try/catch)."""
    k = X.shape[1] + 1
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy, XX = y[ok], np.column_stack([np.ones(ok.sum()), X[ok]])
    n = len(yy)
    if n <= k or np.linalg.matrix_rank(XX) < k:
        return np.full(k, np.nan), np.full(k, np.nan)
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX @ beta
    mse = resid @ resid / (n - k)
    se = np.sqrt(np.diag(mse * np.linalg.inv(XX.T @ XX)))
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), df=n - k)
    return beta, pval


def linear_regr(t: np.ndarray, signal: np.ndarray, event_times: np.ndarray,
                X: np.ndarray, edges: np.ndarray = REG_WINDOW) -> dict:
    """Full linear_regr.m pipeline for ONE unit (ROI/column)."""
    it, isig = interpolate_signal(t, signal)
    sig_by_trial, t_by_trial = align_signal(
        it, isig, event_times, (edges[0] - ALIGN_PAD_S, edges[-1] + ALIGN_PAD_S))
    binned, centers = bin_aligned(sig_by_trial, t_by_trial, edges)
    coeff = np.full((len(centers), X.shape[1] + 1), np.nan)
    pval = np.full_like(coeff, np.nan)
    for i in range(len(centers)):
        coeff[i], pval[i] = ols_regstats(binned[i], X)
    return {"regr_time": centers, "coeff": coeff, "pval": pval}


# ===========================================================================
# 5-7. Significance, fractions, overlap  [getRegautoCorrData.m, chi2ind.m]
# ===========================================================================
def is_modulated(pvals: np.ndarray, alpha: float = PVAL_THRESH,
                 min_consec: int = MIN_CONSEC_SIG_BINS,
                 min_total: int = MIN_TOTAL_SIG_BINS) -> bool:
    """True if >= min_consec consecutive bins with p<alpha, or >= min_total in total."""
    sig = np.nan_to_num(pvals, nan=1.0) < alpha
    if sig.sum() >= min_total:
        return True
    run = best = 0
    for s in sig:
        run = run + 1 if s else 0
        best = max(best, run)
    return best >= min_consec


def fraction_significant_per_bin(pval_stack: np.ndarray, pred_col: int,
                                 alpha: float = PVAL_THRESH) -> pd.DataFrame:
    """pval_stack: (n_units, n_bins, n_coef). Fraction of units with p<alpha per
    bin + binomial test vs alpha (code) and chi-square vs alpha (paper)."""
    sig = np.nan_to_num(pval_stack[:, :, pred_col], nan=1.0) < alpha
    n_units, n_bins = sig.shape
    rows = []
    for b in range(n_bins):
        k = int(sig[:, b].sum())
        p_binom = stats.binomtest(k, n_units, alpha).pvalue
        exp = np.array([n_units * alpha, n_units * (1 - alpha)])
        p_chi = stats.chisquare([k, n_units - k], exp).pvalue
        rows.append((b, k / n_units, p_binom, p_chi))
    return pd.DataFrame(rows, columns=["bin", "frac_sig", "p_binom", "p_chi2"])


def overlap_chi2(mod_a: np.ndarray, mod_b: np.ndarray) -> dict:
    """Pearson chi-square test of independence on the 2x2 table (no Yates)."""
    a, b = np.asarray(mod_a, bool), np.asarray(mod_b, bool)
    table = np.array([[np.sum(a & b), np.sum(a & ~b)],
                      [np.sum(~a & b), np.sum(~a & ~b)]])
    chi2, p, *_ = stats.chi2_contingency(table, correction=False)
    return {"table": table, "chi2": chi2, "p": p}


def conditional_prob(mod_v1: np.ndarray, mod_v2: np.ndarray) -> float:
    """Pr(v1|v2) = N(v1 & v2) / N(v2) within one session."""
    v1, v2 = np.asarray(mod_v1, bool), np.asarray(mod_v2, bool)
    return np.nan if v2.sum() == 0 else float(np.sum(v1 & v2) / v2.sum())


# ===========================================================================
# 8-9. Clustering and temporal dynamics
# ===========================================================================
def cluster_outcome_coeffs(coeffs: np.ndarray, t: np.ndarray,
                           n_clusters: int = N_CLUSTERS,
                           t_range: tuple = CLUSTER_T_RANGE,
                           auc_window: tuple = AUC_WINDOW) -> np.ndarray:
    """coeffs: (n_significant_units, n_bins) of ONE session.
    clusterdata(... 'Linkage','complete','distance','correlation','Maxclust',2),
    then each cluster -> group 1 if AUC(0-2 s) of its mean coefficient > 0, else group 2.
    Returns the group (1/2) of each unit."""
    if coeffs.shape[0] == 0:
        return np.array([], dtype=int)
    if coeffs.shape[0] == 1:
        labels = np.array([1])
    else:
        tm = (t > t_range[0]) & (t < t_range[1])
        Z = linkage(coeffs[:, tm], method="complete", metric="correlation")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    wm = (t > auc_window[0]) & (t < auc_window[1])
    group = np.zeros(len(labels), dtype=int)
    for lab in np.unique(labels):
        mean_curve = np.nanmean(coeffs[labels == lab], axis=0)
        auc = _trapz(mean_curve[wm], t[wm])
        group[labels == lab] = 1 if auc > 0 else 2
    return group


def matlab_smooth(y: np.ndarray, span: int = 5) -> np.ndarray:
    """MATLAB smooth() default: 5-point moving average whose window
    shrinks symmetrically at the edges."""
    y = np.asarray(y, float)
    n, h = len(y), span // 2
    out = np.empty(n)
    for i in range(n):
        w = min(h, i, n - 1 - i)
        out[i] = np.nanmean(y[i - w:i + w + 1])
    return out


def time_to_peak(coeff: np.ndarray, t: np.ndarray, sign: int) -> tuple[float, float]:
    """smooth -> interp 10 ms -> max of sign*curve for t>0 (MP_GRAB_temporalCorrSummary.m).
    Returns (time_to_peak [s from cue], sign-oriented peak_value)."""
    s = matlab_smooth(sign * np.asarray(coeff, float))
    ti = np.arange(t[0], t[-1] + 1e-9, 0.01)
    si = np.interp(ti, t, s)
    pos = ti > 0
    k = int(np.nanargmax(si[pos]))
    return float(ti[pos][k]), float(si[pos][k])


def session_temporal_summary(ttp: np.ndarray, peak: np.ndarray) -> dict:
    """Per session: median and variance (ddof=1, like nanvar) of time-to-peak,
    and median peak value."""
    return {"median_ttp": np.nanmedian(ttp),
            "var_ttp": np.nanvar(ttp, ddof=1) if np.sum(np.isfinite(ttp)) > 1 else np.nan,
            "median_peak": np.nanmedian(peak)}


def compare_groups(a: np.ndarray, b: np.ndarray) -> dict:
    """MATLAB ranksum (two-sided Wilcoxon rank-sum) and Mood's median test."""
    a, b = np.asarray(a)[np.isfinite(a)], np.asarray(b)[np.isfinite(b)]
    out = {"n_a": len(a), "n_b": len(b)}
    out["p_ranksum"] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    out["p_median_test"] = stats.median_test(a, b)[1]
    return out


# ===========================================================================
# Optional utility: entropy-based cut [cutoff.m] -- off by default
# ===========================================================================
def running_entropy(choice: np.ndarray, win: int = 30) -> np.ndarray:
    """Entropy of 3-choice sequences in a 30-trial moving window."""
    c = np.asarray(choice)
    out = np.full(len(c), np.nan)
    for i in range(win, len(c) + 1):
        seg = c[i - win:i]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 3:
            continue
        pats = [tuple(seg[k:k + 3]) for k in range(len(seg) - 2)]
        _, cnt = np.unique(np.array(pats), axis=0, return_counts=True)
        p = cnt / cnt.sum()
        out[i - 1] = -np.sum(p * np.log2(p))
    return out


# ===========================================================================
# Self-test on synthetic data
# ===========================================================================
def _self_test(seed: int = 0):
    rng = np.random.default_rng(seed)
    fs, n_trials = 20.0, 400
    # ITI 10-12 s: no overlap of transients between neighboring trials (with
    # short ITIs + very high synthetic SNR, jittered overlap creates spurious
    # effects through model misspecification, not a pipeline bug; the
    # false-positive rate with a null signal was verified at ~1%)
    iti = rng.uniform(10, 12, n_trials)
    cue = np.cumsum(iti) + 30
    t = np.arange(0, cue[-1] + 20, 1 / fs)
    choice_lr = rng.choice([-1.0, 1.0], n_trials)
    choice_lr[rng.random(n_trials) < 0.03] = np.nan            # misses
    responded = np.isfinite(choice_lr)
    rewarded = (responded & (rng.random(n_trials) < 0.6)).astype(float)

    # positive raw F with a reward transient (peak ~0.4 s) + drift
    f = 100 + 5 * np.sin(2 * np.pi * t / 600) + rng.normal(0, 1.5, t.size)
    for k in np.where(rewarded == 1)[0]:
        dtt = t - cue[k]
        m = (dtt > 0) & (dtt < 4)
        f[m] += 3.0 * (dtt[m] / 0.4) * np.exp(1 - dtt[m] / 0.4)

    qc = check_raw_trace(f)
    dff, _ = calc_dff(f, fs)
    c_ic = encode_choice_ipsi_contra(choice_lr, "left")
    X = build_design_matrix(c_ic, rewarded, responded)
    res = linear_regr(t, dff, cue, X)

    assert res["coeff"].shape == (80, 15), res["coeff"].shape
    col_rn = 1 + PREDICTOR_NAMES.index("R(n)")
    col_cn = 1 + PREDICTOR_NAMES.index("C(n)")
    mod_r = is_modulated(res["pval"][:, col_rn])
    mod_c = is_modulated(res["pval"][:, col_cn])
    ttp, pk = time_to_peak(res["coeff"][:, col_rn], res["regr_time"], +1)
    grp = cluster_outcome_coeffs(res["coeff"][None, :, col_rn], res["regr_time"])

    print("Raw-trace QC:", qc)
    print(f"R(n) modulated: {mod_r} (expected True) | C(n) modulated: {mod_c} (expected False)")
    print(f"time-to-peak R(n): {ttp:.2f} s (simulated transient peaks at 0.4 s + bin width)")
    print(f"outcome cluster group: {grp} (expected [1], positive response)")
    assert mod_r and not mod_c and grp[0] == 1 and 0.2 < ttp < 1.0
    # already-processed trace -> the QC must flag it
    assert not check_raw_trace(dff)["looks_raw"]
    print("SELF-TEST OK")


if __name__ == "__main__":
    _self_test()
