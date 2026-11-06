"""
neuromodulator_state_regression.py
==================================
Objective 5: does the NE / ACh signal carry the LATENT variables of the two
behavioral models, beyond the observable task variables?

Three nested regressions on the same trials and the same 100-ms bins, per
session (Wang et al. 2025 scheme: one OLS per bin, so every coefficient is a
time course):

    M1  base (Wang)   : the 14 predictors of MP_GRAB_MLR.m
                        (choice, outcome and their interaction at n+1..n-2,
                         reward moving average, cumulative reward)
    M2  + state       : p(engaged) from the frozen GLM-HMM (primary spec, sigma = 1),
                        one-step-ahead PREDICTED posterior P(z_t | y_1..t-1)
                        (uses trials < t only; the smoothed one would leak
                         future choices into a question about timing)
    M3  + belief      : confidence |2b-1|, value of the chosen option, unsigned
                        prediction error |RPE|, and tau
                        (signed RPE is NOT used: the base already contains the
                         current outcome R(n), so RPE = R(n) - V_chosen would add
                         only -V_chosen and be read as an error signal by mistake.
                         V_chosen is defined before the outcome; |RPE| is the
                         outcome surprise and is not a linear function of R and V)
                        (trials since criterion) from the fitted belief_vhr

Each step is tested with a partial F-test per bin, so the question is not "does
the signal correlate with the state" but "does the state add anything once
choice, outcome and history are accounted for".

Why these two levels are not redundant (measured on the behavioral dataset):
inside the pre-switch L_Random window, exploit occupancy is flat across tau
(0.796 -> 0.793, p = 0.54) while accuracy falls (0.888 -> 0.822). The GLM-HMM
has no block clock and belief_vhr has no disengagement, so a signal tracking
tau cannot be explained by the state, and vice versa.

Predictions to test (these recordings are PRELIMBIC cortex, not M2 as in
Wang et al., so their region-specific result is a lead, not a prediction):
  - In M2, Wang et al. found NE (not ACh) promotes switching. If that role is
    shared by prelimbic cortex, NE should track the explore state or precede
    it; a null here is as informative, since the regions differ.
  - Yu & Dayan: ACh ~ expected uncertainty -> should track belief confidence.
  - The behavioral anticipation result (b = 0.051) predicts a signal that
    grows with tau inside the L_Random window.

Requires, in analysis/:
  - glmhmm_states_neuromodulator_K2.csv  (glmhmm_decode.py, primary sigma = 1 model)
  - belief_vhr_lrandom_per_animal.csv / _global.csv  (belief_vhr_lrandom.py)

Run from reversalExpectation_py (revExp env):
    python neuromodulator\\neuromodulator_state_regression.py
"""

import importlib.util
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

from behavior.beh_models.belief_vhr import belief_trajectory  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wp = _load("wang2025_pipeline")
runner = _load("run_wang2025_neuromodulator")
ntr = _load("neuromodulator_trials")

PHASES = [31]
EDGES = np.round(np.arange(-3.0, 5.0 + 1e-9, 0.1), 10)     # Wang: -3 to 5 s, 100 ms
STATE_CSV = ROOT / "analysis" / "glmhmm_states_neuromodulator_K2.csv"
# belief_vhr parameters for the latents. Preferred: the pooled fit on THIS
# (imaging) cohort's own behaviour (belief_vhr_neuromod.py). Fallback: the main
# behavioural cohort. Per-animal parameters are deliberately NOT used: the
# split-half reliability of per-animal b is low (rho = 0.21, belief_vhr_splithalf.py).
VHR_NEUROMOD = ROOT / "analysis" / "belief_vhr_neuromod_global.csv"
VHR_GLOBAL = ROOT / "analysis" / "belief_vhr_lrandom_global.csv"
OUT_DIR = ROOT / "data" / "figs" / "neuromodulator-state"

STATE_PRED = ["p_exploit"]                                  # M2 adds
BELIEF_PRED = ["confidence", "v_chosen", "abs_rpe", "tau"]  # M3 adds
PVAL = 0.01                    # Wang's threshold
MIN_CONSEC, MIN_TOTAL = 3, 10  # Wang's modulation criterion
MIN_TRIALS = 100
# session criterion, identical to the behavioural analyses of this cohort
# (glmhmm_heldout_validation.py, belief_vhr_neuromod.py): > 100 responded trials
# and >= 4 switches (block_idx.max() > 3), so neural sessions are a subset of them
MIN_RESPONDED, MIN_SWITCHES = 100, 3
N_SHIFT_NULL = 100             # circular-shift repeats per session (empirical null)
MIN_SHIFT = 30                 # minimum shift in trials
COLORS = {"NE": "#2A9D8F", "ACh": "#E9A93A"}
COHORT = {1.0: "NE", 2.0: "ACh"}


# ---------------------------------------------------------------------------
# OLS with the residual sum of squares, for the nested F-tests
# ---------------------------------------------------------------------------
def _ols(y, X):
    """Returns (beta, pval, rss, n, k) with an intercept; NaN rows dropped."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy = y[ok]
    XX = np.column_stack([np.ones(ok.sum()), X[ok]])
    n, k = len(yy), XX.shape[1]
    if n <= k + 1 or np.linalg.matrix_rank(XX) < k:
        return np.full(k, np.nan), np.full(k, np.nan), np.nan, n, k
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX @ beta
    rss = float(resid @ resid)
    dof = n - k
    s2 = rss / dof
    xtx_inv = np.linalg.pinv(XX.T @ XX)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, p, rss, n, k


def _f_test(rss_small, k_small, rss_big, k_big, n):
    """Partial F-test between nested models fit on the SAME rows."""
    if not np.isfinite(rss_small) or not np.isfinite(rss_big) or k_big <= k_small:
        return np.nan, np.nan
    df1, df2 = k_big - k_small, n - k_big
    if df2 <= 0 or rss_big <= 0:
        return np.nan, np.nan
    F = ((rss_small - rss_big) / df1) / (rss_big / df2)
    return float(F), float(stats.f.sf(F, df1, df2))


def _modulated(pvals):
    """Wang's criterion: p < 0.01 in >= 3 consecutive bins or >= 10 in total."""
    sig = np.isfinite(pvals) & (pvals < PVAL)
    if sig.sum() >= MIN_TOTAL:
        return True
    run = 0
    for s in sig:
        run = run + 1 if s else 0
        if run >= MIN_CONSEC:
            return True
    return False


def _z(v):
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else v * np.nan


def _regress_bins(Y, centers, X1, X2, X3, names) -> pd.DataFrame:
    """M1 / M2 / M3 per time bin: added-predictor coefficients and nested F-tests."""
    rows = []
    XA = np.column_stack([X1, X2, X3])
    X12 = np.column_stack([X1, X2])
    for i, tc in enumerate(centers):
        y = Y[:, i]
        keep = np.isfinite(y) & np.all(np.isfinite(XA), axis=1)   # same rows in all models
        yk = y[keep]
        _, _, r1, n1, k1 = _ols(yk, X1[keep])
        _, _, r2, _, k2 = _ols(yk, X12[keep])
        b3, p3, r3, _, k3 = _ols(yk, XA[keep])
        F12, pF12 = _f_test(r1, k1, r2, k2, n1)
        F23, pF23 = _f_test(r2, k2, r3, k3, n1)
        rec = {"t": tc, "n": n1, "F_state": F12, "p_F_state": pF12,
               "F_belief": F23, "p_F_belief": pF23}
        for j, nm in enumerate(names):                            # coefficients from M3
            ok3 = np.all(np.isfinite(b3))
            rec[f"beta_{nm}"] = b3[1 + X1.shape[1] + j] if ok3 else np.nan
            rec[f"p_{nm}"] = p3[1 + X1.shape[1] + j] if ok3 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
def build_session_design(row, states, vhr_par):
    """Load one session and build the signal and the three nested design matrices.

    Shared by the per-bin analysis (process_session, below) and the alternative
    per-trial window analysis (neuromodulator_state_baseline.py), so both use
    exactly the same sessions, alignment, predictors and z-scoring.

    Returns (out, design): `out` is the session record with its status; `design`
    is None when the session is skipped, else a dict with
        Y (n_trials, n_bins) binned dF/F aligned to the cue, centers (n_bins,),
        X1 base (Wang), X2 state, X3 belief, n_tr, ses.
    Exceptions are left to the caller.
    """
    ses = Path(str(row["session_file"])).stem
    out = {"session": ses, "animal": str(row["animal"]),
           "cohort": COHORT.get(row.get("experiment"), "?"), "status": "ok"}
    if pd.isna(row.get("cell_created")) or pd.isna(row.get("time_events_created")):
        out["status"] = "no_neural_data"
        return out, None
    beh = ntr.build_trial_table(row["beh_path"], out["animal"], ses)
    if not (int(beh["responded"].sum()) > MIN_RESPONDED
            and int(beh["block_idx"].max()) > MIN_SWITCHES):
        out["status"] = "fails_criteria"
        return out, None
    st = states[states["session_file"] == ses]
    if st.empty:
        out["status"] = "no_decoded_states"
        return out, None
    st = st.set_index("trial_idx").reindex(beh["trial_idx"])

    t, f = runner.load_trace(Path(row["neural_data_path"]) / row["cell_filename"])
    stamps = runner.load_trial_stamps(Path(row["neural_data_path"]) / row["time_events_filename"])
    match = ntr.match_stamps_to_trials(stamps, beh["cue_time_log"].to_numpy())
    qc = wp.check_raw_trace(f)
    out.update(n_trials=len(beh), n_imaged=match["n_matched"], stamp_lag=match["lag"])
    if not qc["looks_raw"]:
        out["status"] = "trace_not_raw"
        return out, None
    if not match["ok"] or match["n_matched"] < MIN_TRIALS:
        out["status"] = "bad_alignment" if not match["ok"] else "too_few_trials"
        return out, None

    fs = 1.0 / np.median(np.diff(t))
    dff, _ = wp.calc_dff(f, fs)
    it, isig = wp.interpolate_signal(t, dff)
    sig, tb = wp.align_signal(it, isig, match["cue_time_imaging"], (EDGES[0] - 1, EDGES[-1] + 1))
    binned, centers = wp.bin_aligned(sig, tb, EDGES)      # (n_bins, n_trials)
    Y = binned.T                                          # (n_trials, n_bins)

    # --- predictors -----------------------------------------------------
    # same convention as the Wang runner: with RECORDING_SITE empty, C is
    # coded right=1 / left=0, NOT ipsi/contra (warned about in main())
    site = runner.RECORDING_SITE.get(out["animal"], runner.DEFAULT_SITE)
    c_ic = wp.encode_choice_ipsi_contra(beh["choice"].to_numpy(), site)
    X1 = wp.build_design_matrix(c_ic, np.nan_to_num(beh["rewarded"].to_numpy(), nan=0.0),
                                beh["responded"].to_numpy())
    # M2: ONE-STEP-AHEAD PREDICTED posterior of the engaged state,
    # P(z_t | y_1..t-1): only choices/outcomes before trial t, i.e. known at cue
    # onset. (The earlier ssm-based decoder's "p_engaged_filtered" came from
    # ssm's HMM.filter(), which returns exactly this predicted quantity, so the
    # previous results already used it.) The true filtered posterior
    # P(z_t | y_1..t) includes the choice of trial t; it is kept as X2_filtered
    # for comparison only.
    p_exploit = _z(st["p_engaged_predicted"].to_numpy())
    p_filtered = _z(st["p_engaged_filtered"].to_numpy())
    # M3: belief latents with the animal's own belief_vhr parameters
    par = vhr_par["__global__"]                  # one cohort-level parameter set
    tr = belief_trajectory(par, beh["choice"].to_numpy(), beh["rewarded"].to_numpy(),
                           int(beh["rule"].max() if beh["rule"].notna().any() else 2),
                           beh["tau"].clip(lower=0).to_numpy())
    X2 = np.column_stack([p_exploit])
    # value of the chosen option under the belief, E[r | belief, chosen side]:
    # recovered from rpe = r - E[r], so v_chosen = r - rpe (NaN on no-response)
    r_obs = np.nan_to_num(beh["rewarded"].to_numpy().astype(float), nan=0.0)
    v_chosen = r_obs - tr["rpe"]
    abs_rpe = np.abs(tr["rpe"])
    X3 = np.column_stack([_z(tr["confidence"]), _z(v_chosen), _z(abs_rpe),
                          _z(beh["tau"].clip(lower=0).to_numpy())])
    out["vhr_params"] = vhr_par["__source__"]

    # correlation between the two latent levels (collinearity check)
    ok = np.isfinite(p_exploit) & np.isfinite(X3[:, 0])
    out["corr_state_confidence"] = float(np.corrcoef(p_exploit[ok], X3[ok, 0])[0, 1]) if ok.sum() > 10 else np.nan
    ok = np.isfinite(p_exploit) & np.isfinite(X3[:, 3])
    out["corr_state_tau"] = float(np.corrcoef(p_exploit[ok], X3[ok, 3])[0, 1]) if ok.sum() > 10 else np.nan
    ok = np.isfinite(p_exploit) & np.isfinite(X3[:, 1])
    out["corr_state_vchosen"] = float(np.corrcoef(p_exploit[ok], X3[ok, 1])[0, 1]) if ok.sum() > 10 else np.nan
    ok = np.isfinite(X3[:, 0]) & np.isfinite(X3[:, 1])
    out["corr_conf_vchosen"] = float(np.corrcoef(X3[ok, 0], X3[ok, 1])[0, 1]) if ok.sum() > 10 else np.nan

    # --- strictly PRE-CUE alternatives (only information available before the
    # cue of trial n), used by neuromodulator_state_baseline.py --spec precue:
    #   X1_precue    : history-only baseline. Of the Wang predictors, keep choice,
    #                  outcome and their interaction at n-1 and n-2; drop n+1 and n.
    #                  Reward moving average and cumulative reward are shifted
    #                  by one trial so they end at n-1 (the Wang versions include
    #                  trial n; the cumulative one is divided by the session total,
    #                  a per-session constant that z-scoring/OLS absorbs).
    #   state        : X2 (already the one-step-ahead predicted posterior)
    #   X3_precue    : belief latents known at cue onset. confidence and tau are
    #                  already defined at decision time; v_chosen and |RPE| depend
    #                  on the choice / outcome of trial n, so the n-1 values are used.
    lag1 = lambda v: wp._lag(np.asarray(v, float), 1)                  # noqa: E731
    lag2 = lambda v: wp._lag(np.asarray(v, float), 2)                  # noqa: E731
    C, R, CR = X1[:, 1], X1[:, 5], X1[:, 9]                 # columns at lag 0 (trial n)
    X1_precue = np.column_stack([lag1(C), lag2(C), lag1(R), lag2(R), lag1(CR), lag2(CR),
                                 lag1(X1[:, 12]), lag1(X1[:, 13])])
    X2_filtered = np.column_stack([p_filtered])
    X3_precue = np.column_stack([_z(tr["confidence"]), _z(lag1(v_chosen)), _z(lag1(abs_rpe)),
                                 _z(beh["tau"].clip(lower=0).to_numpy())])
    return out, {"Y": Y, "centers": centers, "X1": X1, "X2": X2, "X3": X3,
                 "X1_precue": X1_precue, "X2_filtered": X2_filtered,
                 "X3_precue": X3_precue, "n_tr": len(beh), "ses": ses}


def process_session(row, states, vhr_par):
    try:
        out, design = build_session_design(row, states, vhr_par)
        if design is None:
            return out, None
        Y, centers = design["Y"], design["centers"]
        X1, X2, X3 = design["X1"], design["X2"], design["X3"]
        ses = design["ses"]

        # --- one regression per bin ----------------------------------------
        names = STATE_PRED + BELIEF_PRED
        res = _regress_bins(Y, centers, X1, X2, X3, names)

        # --- empirical null: circular shift of the ADDED regressors ---------
        # p_exploit and tau are slow and dF/F is autocorrelated, so OLS p-values
        # are anti-conservative here. A circular shift keeps each regressor's own
        # autocorrelation and their mutual correlation but destroys the trial-wise
        # link to the signal, giving the false-positive rate of the SAME criterion.
        # stable seed: Python's hash() of a str changes between runs, crc32 does not
        rng = np.random.default_rng(zlib.crc32(ses.encode("utf-8")))
        n_tr = design["n_tr"]
        null = {f"null_{k}": [] for k in ["state_F", "belief_F"] + names}
        for _ in range(N_SHIFT_NULL):
            k_sh = int(rng.integers(MIN_SHIFT, max(n_tr - MIN_SHIFT, MIN_SHIFT + 1)))
            r_ = _regress_bins(Y, centers, X1, np.roll(X2, k_sh, axis=0),
                               np.roll(X3, k_sh, axis=0), names)
            null["null_state_F"].append(_modulated(r_["p_F_state"].to_numpy()))
            null["null_belief_F"].append(_modulated(r_["p_F_belief"].to_numpy()))
            for nm in names:
                null[f"null_{nm}"].append(_modulated(r_[f"p_{nm}"].to_numpy()))
        out.update({k: float(np.mean(v)) for k, v in null.items()})
        res.insert(0, "session", ses)
        res.insert(1, "animal", out["animal"])
        res.insert(2, "cohort", out["cohort"])

        out["mod_state_F"] = _modulated(res["p_F_state"].to_numpy())
        out["mod_belief_F"] = _modulated(res["p_F_belief"].to_numpy())
        for nm in names:
            out[f"mod_{nm}"] = _modulated(res[f"p_{nm}"].to_numpy())
        return out, res
    except Exception as e:
        ses = Path(str(row["session_file"])).stem
        out = {"session": ses, "animal": str(row["animal"]),
               "cohort": COHORT.get(row.get("experiment"), "?"),
               "status": f"error: {e!r}"}
        return out, None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(coef: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = STATE_PRED + BELIEF_PRED
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7), sharex=True)
    for j, nm in enumerate(names):
        top, bot = axes[0][j], axes[1][j]
        for coh, g in coef.groupby("cohort"):
            # sessions -> animal -> cohort, so animals weigh equally
            per_animal = g.pivot_table(index="animal", columns="t", values=f"beta_{nm}")
            mu = per_animal.mean()
            se = per_animal.std(ddof=1) / np.sqrt(len(per_animal))
            top.plot(mu.index, mu.values, color=COLORS.get(coh, "gray"),
                     label=f"{coh} (n={len(per_animal)})")
            top.fill_between(mu.index, mu - se, mu + se, color=COLORS.get(coh, "gray"),
                             alpha=0.2, lw=0)
            frac = g.pivot_table(index="session", columns="t", values=f"p_{nm}",
                                 aggfunc="first").lt(PVAL).mean()
            bot.plot(frac.index, frac.values, color=COLORS.get(coh, "gray"))
        for ax in (top, bot):
            ax.axvline(0, color="k", lw=0.8)
        top.axhline(0, color="gray", lw=0.6)
        bot.axhline(PVAL, ls=":", color="k", lw=1)
        top.set_title(f"beta: {nm}")
        bot.set(title=f"fraction of sessions p<{PVAL}", xlabel="time from cue (s)")
        if j == 0:
            top.set_ylabel("coefficient (dF/F per SD)")
            bot.set_ylabel("fraction")
            top.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_state_belief_coefficients.png", dpi=150)
    plt.close(fig)

    # nested F-tests
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (col, lab) in zip(axes, [("p_F_state", "M2 vs M1: + GLM-HMM state"),
                                     ("p_F_belief", "M3 vs M2: + belief / tau")]):
        for coh, g in coef.groupby("cohort"):
            frac = g.pivot_table(index="session", columns="t", values=col,
                                 aggfunc="first").lt(PVAL).mean()
            ax.plot(frac.index, frac.values, color=COLORS.get(coh, "gray"), label=coh)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(PVAL, ls=":", color="k", lw=1)
        ax.set(title=lab, xlabel="time from cue (s)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(f"fraction of sessions p<{PVAL}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_nested_Ftests.png", dpi=150)
    plt.close(fig)


def load_vhr_params():
    """Cohort-level belief_vhr parameters (a, b, beta) for every session.

    Order of preference:
      1. pooled fit on the imaging cohort's own behaviour (belief_vhr_neuromod.py);
      2. pooled fit on the main behavioural cohort (belief_vhr_lrandom.py);
      3. the reported main-cohort fit, as a last resort.
    """
    for path, src in [(VHR_NEUROMOD, "imaging_cohort"), (VHR_GLOBAL, "main_cohort")]:
        if path.exists():
            g = pd.read_csv(path).iloc[0]
            par = [float(g["a"]), float(g["b"]), float(g["beta"])]
            print(f"[belief_vhr params] {src}: a={par[0]:.3f} b={par[1]:.3f} beta={par[2]:.3f}")
            return {"__global__": par, "__source__": src}
    print("NOTE: no belief_vhr fit found; using the reported main-cohort fit")
    return {"__global__": [-4.4, 0.051, 5.0], "__source__": "reported"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_CSV.exists():
        sys.exit(f"{STATE_CSV} not found: run glmhmm_decode.py in the ssm env first")
    states = pd.read_csv(STATE_CSV)
    states["session_file"] = states["session_file"].astype(str)
    vhr_par = load_vhr_params()

    if not runner.RECORDING_SITE:
        warnings.warn("RECORDING_SITE is empty: C(n) is coded right=1 / left=0, not "
                      "ipsi/contra. It affects the sign of the CHOICE coefficients of "
                      "the Wang base, not the added state / belief predictors.")
    master = _load("master_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(PHASES)]
    idx = ntr.drop_excluded_animals(idx)

    rows, coefs = [], []
    for k, (_, r) in enumerate(idx.iterrows(), 1):
        out, res = process_session(r, states, vhr_par)
        rows.append(out)
        if res is not None:
            coefs.append(res)
        print(f"[{k}/{len(idx)}] {out['session']}: {out['status']}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "session_summary.csv", index=False)
    print("\n=== Session status ===")
    print(summary.groupby(["cohort", "status"]).size().to_string())
    if not coefs:
        return
    coef = pd.concat(coefs, ignore_index=True)
    coef.to_csv(OUT_DIR / "coefficients_by_bin.csv", index=False)
    ok = summary[summary["status"] == "ok"]

    print("\n=== Do the latents add anything over the Wang base? (partial F per bin) ===")
    print("fraction of sessions 'modulated' (Wang criterion: p<0.01 in >=3 consecutive "
          "bins or >=10 total; 1% expected by chance)")
    cols = ["mod_state_F", "mod_belief_F"] + [f"mod_{n}" for n in STATE_PRED + BELIEF_PRED]
    tab = ok.groupby("cohort")[cols].mean().round(3)
    tab["n_sessions"] = ok.groupby("cohort").size()
    print(tab.to_string())
    null_cols = [c for c in ok.columns if c.startswith("null_")]
    if null_cols:
        print("\nempirical null (same criterion after circularly shifting the added "
              f"regressors, {N_SHIFT_NULL} repeats/session): the real fractions above "
              "must exceed THIS, not 1%")
        print(ok.groupby("cohort")[null_cols].mean().round(3).to_string())

        # real vs null, session level and animal level. Sessions of one animal are
        # not independent, so the animal count is the one to trust.
        print("\n=== Real vs circular-shift null ===")
        print("  sessions: observed vs expected under the null (Poisson-binomial, "
              "normal approx., one-sided); animals: how many have real > null")
        for k in ["state_F", "belief_F"] + STATE_PRED + BELIEF_PRED:
            for coh, g in ok.groupby("cohort"):
                r = g[f"mod_{k}"].astype(float)
                pn = g[f"null_{k}"].astype(float)
                obs, mu, var = r.sum(), pn.sum(), (pn * (1 - pn)).sum()
                z = (obs - 0.5 - mu) / np.sqrt(var) if var > 0 else np.nan
                pz = stats.norm.sf(z) if np.isfinite(z) else np.nan
                d = (g.assign(_r=r, _n=pn).groupby("animal")[["_r", "_n"]].mean()
                     .pipe(lambda x: x["_r"] - x["_n"]))
                print(f"  {k:11s} {coh:4s} {int(obs):3d}/{len(g):3d} vs {mu:5.1f}  "
                      f"z={z:+.2f} p={pz:.3g}   animals {int((d > 0).sum())}/{len(d)}")

    print("\n=== NE vs ACh (rank-sum on the per-session peak |beta|) ===")
    for nm in STATE_PRED + BELIEF_PRED:
        pk = coef.groupby(["cohort", "session"])[f"beta_{nm}"].apply(lambda s: np.nanmax(np.abs(s)))
        ne = pk.loc["NE"].dropna() if "NE" in pk.index.get_level_values(0) else pd.Series(dtype=float)
        ach = pk.loc["ACh"].dropna() if "ACh" in pk.index.get_level_values(0) else pd.Series(dtype=float)
        if len(ne) > 3 and len(ach) > 3:
            u = stats.ranksums(ne, ach)
            print(f"  {nm:12s} NE median {ne.median():.4f} (n={len(ne)})  "
                  f"ACh {ach.median():.4f} (n={len(ach)})  p={u.pvalue:.3g}")

    print("\n=== Collinearity between the two latent levels (per-session correlation) ===")
    print(ok.groupby("cohort")[["corr_state_confidence", "corr_state_tau",
                                "corr_state_vchosen", "corr_conf_vchosen"]]
            .median().round(3).to_string())

    make_figures(coef, OUT_DIR)
    print(f"\nResults in {OUT_DIR}")


if __name__ == "__main__":
    main()
