"""
glmhmm_sanity_report.py
=======================
Sanity-check report for the final K=2 `ashwood_wsls` GLM-HMM (exploit / explore).

Answers, with numbers, the questions raised about the model:

  1. Design matrix. How are the regressors coded? Exports the exact per-trial
     design matrix fed to the GLM-HMM, rebuilt with the same rules as
     hmmGlm.build_session_arrays (bias, prev_choice = +-1, wsls = prev_choice x
     (+1 rewarded / -1 omission), history forward-filled across missed trials).

  2. Predicted vs. empirical choice probabilities. P(stay | state, previous
     outcome), where "stay" = repeat the previous (non-missed) choice:
       - predicted: straight from the fitted per-state weights;
       - empirical: from the data, weighting each trial by the model's
         ONE-STEP-AHEAD state probability P(z_t | y_1..y_{t-1}). The predictive
         (not the smoothed) probability is used so that the choice being scored
         does not leak into its own weight (no circularity).
     Pooled over trials and as mean +- SEM across animals.

  3. Posterior predictive check. Model-implied P(stay | previous outcome),
     marginalised over states with the same predictive probabilities, against
     the empirical rate.

  4. Fitting facts. The HMM restarts at every SESSION (each session is its own
     sequence, starting from pi0). Reports pi0, transition matrix, expected
     dwell times, MAP prior, EM settings and the cross-validated held-out
     log-likelihood by K (read from analysis/glmhmm_cv.csv if present).

  5. Pipeline check. Re-runs forward-backward in plain numpy from the saved
     parameters and compares the smoothed P(exploit) with the cached
     analysis/glmhmm_states_K2.csv (should match to ~1e-6).

The script does NOT need the `ssm` package: it reads the saved parameters
from the pickle with a stub unpickler, so it runs in any env with numpy,
pandas and matplotlib (revExp or ssm).

Inputs  (relative to reversalExpectation_py/):
    analysis/bandit_R71_lesion.csv
    analysis/glmhmm_model_ashwood_wsls_K2.pkl
    analysis/glmhmm_states_K2.csv           (optional, for check 5)
    analysis/glmhmm_cv.csv                  (optional, for check 4)
Outputs (analysis/glmhmm_sanity/):
    design_matrix_full.csv.gz     every trial: regressors, P(state), P(right|state)
    design_matrix_example.csv     first 60 trials of one example session (readable)
    model_parameters.csv          weights, pi0, transitions, dwell, prior
    pstay_by_state_outcome.csv    predicted vs empirical P(stay), pooled
    pstay_per_animal.csv          per-animal empirical P(stay) by state x outcome
    pstay_across_animals.csv      mean +- SEM across animals
    posterior_predictive.csv      model vs data P(stay | outcome), marginal
    report.txt                    human-readable summary of all of the above
    pstay_predicted_vs_empirical.png

Usage (from reversalExpectation_py/):
    python behavior/beh_models/glmhmm_sanity_report.py
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # headless: write PNG only
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../behavior/beh_models
REPO_ROOT = HERE.parents[1]                     # .../reversalExpectation_py
ANALYSIS = REPO_ROOT / "analysis"
CSV_PATH = ANALYSIS / "bandit_R71_lesion.csv"
PKL_PATH = ANALYSIS / "glmhmm_model_ashwood_wsls_K2.pkl"
STATES_CSV = ANALYSIS / "glmhmm_states_K2.csv"
CV_CSV = ANALYSIS / "glmhmm_cv.csv"
OUT_DIR = ANALYSIS / "glmhmm_sanity"

MIN_TRIALS = 20                 # same as hmmGlm.build_glmhmm_inputs(min_trials=20)
STATE_LABELS = {0: "exploit", 1: "explore"}
EXAMPLE_N_TRIALS = 60
KEYS = ["animal", "session_file", "block_idx", "trial_idx"]


# ---------------------------------------------------------------------------
# 0. Load the saved model without importing ssm
# ---------------------------------------------------------------------------
class _Stub:
    """Placeholder for ssm/autograd classes: just keeps the pickled state."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _NoSsmUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("ssm", "autograd")):
            return _Stub
        return super().find_class(module, name)


def load_model_params(path: Path) -> dict:
    """Return weights (log-odds of RIGHT), transition matrix, pi0 and prior."""
    with open(path, "rb") as fh:
        d = _NoSsmUnpickler(fh).load()
    m = d["model"]
    # ssm's Wk = log-odds of LEFT (category 0); negate -> log-odds of RIGHT,
    # exactly as hmmGlm.glmhmm_weights() does.
    W_right = -np.asarray(m.observations.Wk)[:, 0, :]          # (K, M)
    P = np.exp(np.asarray(m.transitions.log_Ps))
    P = P / P.sum(1, keepdims=True)                             # row-stochastic
    pi0 = np.exp(np.asarray(m.init_state_distn.log_pi0))
    pi0 = pi0 / pi0.sum()
    return {
        "W": W_right, "P": P, "pi0": pi0, "K": int(d["K"]),
        "parametrization": d["parametrization"],
        "engaged_idx": int(d.get("engaged_idx", 0)),
        "prior_sigma": float(m.observations.prior_sigma),
        "trans_alpha": float(m.transitions.__dict__.get("alpha", np.nan)),
        "trans_kappa": float(m.transitions.__dict__.get("kappa", np.nan)),
        "weights_saved": d.get("weights"),
    }


# ---------------------------------------------------------------------------
# 1. Data selection + design matrix (mirrors hmmGlm exactly)
# ---------------------------------------------------------------------------
def select_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """hmmGlm.select_glmhmm_sessions defaults: naive, all phases, meets_criteria."""
    out = df[df["lesioned"].isna()]
    out = out[out["meets_criteria"] == True]                    # noqa: E712
    out = out.copy()
    n_ses = out.groupby(["animal", "session_file"]).ngroups
    print(f"[select] naive + meets_criteria: {len(out)} trials, "
          f"{out['animal'].nunique()} animals, {n_ses} sessions")
    return out


def session_design(df_ses: pd.DataFrame):
    """Return (observed, choice_right01, X, prev_choice, prev_reward) for one session.

    Identical rules to hmmGlm.build_session_arrays(parametrization="ashwood_wsls"):
      prev_choice / prev_reward taken from the nearest preceding non-missed
      trial (forward fill, then shift by one); first trial has no history (0);
      wsls = (2*prev_reward - 1) * prev_choice, set to 0 when there is no history.
    """
    c = df_ses["choice"].to_numpy(dtype=float)                  # -1 / +1 / NaN
    r = np.nan_to_num(df_ses["rewarded"].to_numpy(dtype=float), nan=0.0)
    observed = ~np.isnan(c)
    signed = np.where(observed, c, 0.0)

    T = len(c)
    filled_c, filled_r = signed.copy(), r.copy()
    last_c, last_r = 0.0, 0.0
    for t in range(T):                                          # forward fill
        if observed[t]:
            last_c, last_r = signed[t], r[t]
        else:
            filled_c[t], filled_r[t] = last_c, last_r
    prev_choice = np.zeros(T)
    prev_reward = np.zeros(T)
    prev_choice[1:] = filled_c[:-1]
    prev_reward[1:] = filled_r[:-1]

    prev_reward_signed = 2.0 * prev_reward - 1.0
    prev_reward_signed[prev_choice == 0] = 0.0
    wsls = prev_reward_signed * prev_choice
    X = np.column_stack([np.ones(T), prev_choice, wsls])        # bias, prev_choice, wsls

    y = np.zeros(T, dtype=int)
    y[observed] = (c[observed] > 0).astype(int)                 # 1 = right
    return observed, y, X, prev_choice, prev_reward


# ---------------------------------------------------------------------------
# 2. Forward-backward (numpy), with masked (missed) trials
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_backward(y, observed, X, W, P, pi0):
    """Scaled forward-backward for an input-driven Bernoulli HMM.

    Returns
      p_right   : (T, K) P(choose right | z_t = k, x_t)
      pred      : (T, K) one-step-ahead P(z_t | y_1..y_{t-1})
      post      : (T, K) smoothed P(z_t | y_1..y_T)
      loglik    : log P(y_1..y_T)
    Missed trials contribute likelihood 1 (as ssm's mask does).
    """
    T, K = X.shape[0], W.shape[0]
    p_right = sigmoid(X @ W.T)                                  # (T, K)
    lik = np.where(y[:, None] == 1, p_right, 1.0 - p_right)
    lik[~observed] = 1.0

    alpha = np.zeros((T, K))
    pred = np.zeros((T, K))
    scale = np.zeros(T)
    prior = pi0.copy()
    for t in range(T):
        pred[t] = prior
        a = prior * lik[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
        prior = alpha[t] @ P

    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        b = P @ (lik[t + 1] * beta[t + 1])
        beta[t] = b / scale[t + 1]
    post = alpha * beta
    post /= post.sum(1, keepdims=True)
    return p_right, pred, post, float(np.log(scale).sum())


# ---------------------------------------------------------------------------
# 3. Build the full per-trial table
# ---------------------------------------------------------------------------
def build_trial_table(df: pd.DataFrame, prm: dict) -> pd.DataFrame:
    W, P, pi0 = prm["W"], prm["P"], prm["pi0"]
    K = W.shape[0]
    parts, n_skipped, total_ll, total_obs = [], 0, 0.0, 0
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        observed, y, X, pc, pr = session_design(d)
        if observed.sum() < MIN_TRIALS:                         # same skip rule
            n_skipped += 1
            continue
        p_right, pred, post, ll = forward_backward(y, observed, X, W, P, pi0)
        total_ll += ll
        total_obs += int(observed.sum())
        t = d[KEYS + ["choice", "rewarded"]].copy()
        t["observed"] = observed
        t["bias"], t["prev_choice"], t["wsls"] = X[:, 0], X[:, 1], X[:, 2]
        t["prev_reward"] = pr
        for k in range(K):
            lab = STATE_LABELS.get(k, f"s{k}")
            t[f"p_right_{lab}"] = p_right[:, k]
            t[f"pred_{lab}"] = pred[:, k]                       # one-step-ahead
            t[f"post_{lab}"] = post[:, k]                       # smoothed
        t["p_right_model"] = (pred * p_right).sum(1)            # predictive P(right)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    print(f"[fb] {len(parts)} sessions kept, {n_skipped} skipped (<{MIN_TRIALS} "
          f"observed trials); in-sample LL/trial = {total_ll / total_obs:.4f}")
    out.attrs["ll_per_trial"] = total_ll / total_obs
    return out


# ---------------------------------------------------------------------------
# 4. P(stay) tables
# ---------------------------------------------------------------------------
def scored_trials(tt: pd.DataFrame) -> pd.DataFrame:
    """Observed trials that have a previous choice: stay = repeat previous side."""
    s = tt[tt["observed"] & (tt["prev_choice"] != 0)].copy()
    s["stay"] = (np.sign(s["choice"]) == s["prev_choice"]).astype(float)
    s["prev_outcome"] = np.where(s["prev_reward"] == 1, "reward", "omission")
    for lab in STATE_LABELS.values():
        # P(stay | state) = P(choose prev side | state)
        pr = s[f"p_right_{lab}"]
        s[f"pstay_{lab}"] = np.where(s["prev_choice"] > 0, pr, 1.0 - pr)
    s["pstay_model"] = sum(s[f"pred_{lab}"] * s[f"pstay_{lab}"]
                           for lab in STATE_LABELS.values())
    return s


def weight_formula_table(W: np.ndarray) -> pd.DataFrame:
    """Logits implied by the weights, for each state x prev outcome x prev side."""
    rows = []
    for k in range(W.shape[0]):
        b, wc, ww = W[k]
        for outcome, rs in (("reward", +1.0), ("omission", -1.0)):
            for side, pc in (("right", +1.0), ("left", -1.0)):
                logit_right = b + wc * pc + ww * rs * pc
                p_right = sigmoid(logit_right)
                p_stay = p_right if pc > 0 else 1 - p_right
                rows.append({"state": STATE_LABELS.get(k, k), "prev_outcome": outcome,
                             "prev_side": side, "logit_right": logit_right,
                             "logit_stay": logit_right * pc,
                             "p_stay_formula": p_stay})
    return pd.DataFrame(rows)


def pooled_table(s: pd.DataFrame, W: np.ndarray) -> pd.DataFrame:
    rows = []
    for k, lab in STATE_LABELS.items():
        for outcome in ("reward", "omission"):
            g = s[s["prev_outcome"] == outcome]
            w = g[f"pred_{lab}"].to_numpy()
            # predicted: model P(stay | state) averaged over the same trials/weights
            pred = np.average(g[f"pstay_{lab}"], weights=w)
            emp = np.average(g["stay"], weights=w)
            n_eff = w.sum() ** 2 / (w ** 2).sum()               # Kish effective n
            se = np.sqrt(emp * (1 - emp) / n_eff)
            # hard-assignment version (argmax of the predictive probability)
            hard = g[g[f"pred_{lab}"] > 0.5]
            rows.append({"state": lab, "prev_outcome": outcome,
                         "p_stay_predicted": pred, "p_stay_empirical": emp,
                         "se_empirical": se, "n_trials": len(g), "n_eff": n_eff,
                         "p_stay_empirical_hard": hard["stay"].mean(),
                         "n_hard": len(hard)})
    return pd.DataFrame(rows)


def per_animal_table(s: pd.DataFrame):
    rows = []
    for animal, ga in s.groupby("animal"):
        for lab in STATE_LABELS.values():
            for outcome in ("reward", "omission"):
                g = ga[ga["prev_outcome"] == outcome]
                w = g[f"pred_{lab}"].to_numpy()
                if w.sum() < 5:                                 # too little mass
                    continue
                rows.append({"animal": animal, "state": lab, "prev_outcome": outcome,
                             "p_stay_empirical": np.average(g["stay"], weights=w),
                             "p_stay_predicted": np.average(g[f"pstay_{lab}"], weights=w),
                             "weight_mass": w.sum()})
    pa = pd.DataFrame(rows)
    agg = (pa.groupby(["state", "prev_outcome"])
             .agg(n_animals=("animal", "nunique"),
                  emp_mean=("p_stay_empirical", "mean"),
                  emp_sem=("p_stay_empirical", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                  pred_mean=("p_stay_predicted", "mean"),
                  pred_sem=("p_stay_predicted", lambda x: x.std(ddof=1) / np.sqrt(len(x))))
             .reset_index())
    return pa, agg


def calibration_table(s: pd.DataFrame) -> pd.DataFrame:
    """Data vs model P(stay), binned by the one-step-ahead P(exploit).

    This is the correct predicted-vs-empirical comparison: the model's
    prediction for each trial is the state MIXTURE (pred_k x P(stay | k)),
    so it is compared with the data within bins of the mixing weight.
    """
    bins = [0, .2, .4, .6, .8, .9, .95, 1.0]
    s = s.assign(bin=pd.cut(s["pred_exploit"], bins))
    return (s.groupby(["prev_outcome", "bin"], observed=True)
              .agg(n_trials=("stay", "size"), p_stay_data=("stay", "mean"),
                   p_stay_model=("pstay_model", "mean"))
              .reset_index())


def switch_attribution(s: pd.DataFrame) -> pd.DataFrame:
    """Which state does the (smoothed) posterior assign switch trials to?"""
    rows = []
    for outcome in ("all", "reward", "omission"):
        g = s if outcome == "all" else s[s["prev_outcome"] == outcome]
        sw, st = g[g["stay"] == 0], g[g["stay"] == 1]
        rows.append({"prev_outcome": outcome, "n_switches": len(sw),
                     "frac_switches_in_explore": (sw["post_explore"] > 0.5).mean(),
                     "mean_P_explore_on_switches": sw["post_explore"].mean(),
                     "mean_P_explore_on_stays": st["post_explore"].mean()})
    return pd.DataFrame(rows)


def posterior_predictive(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ("reward", "omission"):
        g = s[s["prev_outcome"] == outcome]
        rows.append({"prev_outcome": outcome, "n_trials": len(g),
                     "p_stay_data": g["stay"].mean(),
                     "p_stay_model": g["pstay_model"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Fitting facts parsed from the source (no ssm import needed)
# ---------------------------------------------------------------------------
def parse_constants(path: Path, names) -> dict:
    out = {}
    if not path.exists():
        return out
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for n in names:
        m = re.search(rf"^{n}\s*=\s*([^#\n]+)", txt, flags=re.M)
        if m:
            out[n] = m.group(1).strip()
    return out


def cv_summary() -> pd.DataFrame | None:
    if not CV_CSV.exists():
        return None
    cv = pd.read_csv(CV_CSV)
    piv = cv.pivot(index="fold", columns="K", values="test_ll_per_trial")
    rows = []
    for K in piv.columns:
        rows.append({"K": K, "mean_ll": piv[K].mean(),
                     "sem_ll": piv[K].std(ddof=1) / np.sqrt(len(piv))})
    Ks = list(piv.columns)
    for a, b in zip(Ks[:-1], Ks[1:]):
        d = piv[b] - piv[a]
        rows.append({"K": f"{a}->{b}", "mean_ll": d.mean(),
                     "sem_ll": d.std(ddof=1) / np.sqrt(len(d)),
                     "folds_improving": int((d > 0).sum()), "n_folds": len(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Figure
# ---------------------------------------------------------------------------
def make_figure(pooled: pd.DataFrame, agg: pd.DataFrame, path: Path):
    INK, GREEN, CORAL, GOLD = "#1A2E2A", "#2C5F2D", "#C9472B", "#E8A33D"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, lab, col in zip(axes, STATE_LABELS.values(), (GREEN, CORAL)):
        a = agg[agg["state"] == lab].set_index("prev_outcome").loc[["reward", "omission"]]
        p = pooled[pooled["state"] == lab].set_index("prev_outcome").loc[["reward", "omission"]]
        x = np.arange(2)
        ax.bar(x - 0.18, p["p_stay_predicted"], 0.34, color=col, alpha=0.45,
               label="predicted (weights)")
        ax.bar(x + 0.18, a["emp_mean"], 0.34, yerr=a["emp_sem"], color=col,
               capsize=3, label="empirical (mean ± SEM animals)")
        ax.axhline(0.5, color=INK, lw=0.8, ls="--")
        ax.set_xticks(x, ["after reward", "after omission"])
        ax.set_title(lab, color=INK)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(repeat previous choice)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prm = load_model_params(PKL_PATH)
    if prm["parametrization"] != "ashwood_wsls" or prm["K"] != 2:
        sys.exit(f"expected ashwood_wsls K=2, got {prm['parametrization']} K={prm['K']}")
    W, P, pi0 = prm["W"], prm["P"], prm["pi0"]

    # --- data, design matrix, forward-backward --------------------------------
    df = select_sessions(pd.read_csv(CSV_PATH))
    tt = build_trial_table(df, prm)
    tt.to_csv(OUT_DIR / "design_matrix_full.csv.gz", index=False, compression="gzip")
    ex_key = tt[["animal", "session_file"]].iloc[0]
    ex = tt[(tt["animal"] == ex_key["animal"]) &
            (tt["session_file"] == ex_key["session_file"])].head(EXAMPLE_N_TRIALS)
    ex.to_csv(OUT_DIR / "design_matrix_example.csv", index=False)

    # --- check 5: posteriors vs cached state file ------------------------------
    post_check = "states CSV not found"
    if STATES_CSV.exists():
        st = pd.read_csv(STATES_CSV)
        m = tt.merge(st[KEYS + ["p_engaged", "glmhmm_state"]], on=KEYS, how="inner")
        diff = (m["post_exploit"] - m["p_engaged"]).abs()
        agree = (np.where(m["post_exploit"] > 0.5, 0, 1) == m["glmhmm_state"]).mean()
        post_check = (f"{len(m)} trials matched; |P(exploit) numpy - cached| "
                      f"median={diff.median():.2e}, max={diff.max():.2e}; "
                      f"argmax state agreement={agree:.4f}")
    print("[check posteriors]", post_check)

    # --- P(stay) tables -------------------------------------------------------
    s = scored_trials(tt)
    formula = weight_formula_table(W)
    pooled = pooled_table(s, W)
    pa, agg = per_animal_table(s)
    ppc = posterior_predictive(s)
    calib = calibration_table(s)
    swa = switch_attribution(s)
    calib.to_csv(OUT_DIR / "calibration_by_pexploit.csv", index=False)
    swa.to_csv(OUT_DIR / "switch_attribution.csv", index=False)
    pooled.to_csv(OUT_DIR / "pstay_by_state_outcome.csv", index=False)
    pa.to_csv(OUT_DIR / "pstay_per_animal.csv", index=False)
    agg.to_csv(OUT_DIR / "pstay_across_animals.csv", index=False)
    ppc.to_csv(OUT_DIR / "posterior_predictive.csv", index=False)

    # --- parameters -----------------------------------------------------------
    dwell = 1.0 / (1.0 - np.diag(P))
    stat = np.linalg.eig(P.T)[1][:, 0].real
    stat = stat / stat.sum()
    par_rows = []
    for k in range(2):
        lab = STATE_LABELS[k]
        par_rows.append({"state": lab, "bias": W[k, 0], "prev_choice": W[k, 1],
                         "wsls": W[k, 2], "pi0": pi0[k],
                         f"P_to_{STATE_LABELS[0]}": P[k, 0],
                         f"P_to_{STATE_LABELS[1]}": P[k, 1],
                         "expected_dwell_trials": dwell[k],
                         "stationary_occupancy": stat[k]})
    params = pd.DataFrame(par_rows)
    params.to_csv(OUT_DIR / "model_parameters.csv", index=False)

    consts = parse_constants(HERE / "hmmGlm.py",
                             ["N_ITERS", "TOLERANCE", "INIT_NOISE", "STICKY_SELF",
                              "PRIOR_SIGMA", "TRANSITION_ALPHA", "TRANSITION_KAPPA"])
    consts.update(parse_constants(HERE / "run_glmhmm.py",
                                  ["PARAMETRIZATION", "N_RESTARTS", "N_FOLDS", "K_RANGE"]))
    cvs = cv_summary()

    make_figure(pooled, agg, OUT_DIR / "pstay_predicted_vs_empirical.png")

    # --- human-readable report ------------------------------------------------
    pd.set_option("display.width", 140)
    fmt = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.4f}")  # noqa: E731
    lines = [
        "GLM-HMM sanity report  (ashwood_wsls, K=2, global fit)",
        "=" * 60,
        f"Data: {tt['animal'].nunique()} animals, "
        f"{tt.groupby(['animal','session_file']).ngroups} sessions, "
        f"{int(tt['observed'].sum())} observed trials (naive + meets_criteria).",
        "",
        "1. DESIGN MATRIX",
        "   bias = 1",
        "   prev_choice = +1 (right) / -1 (left), from the last non-missed trial",
        "   wsls = prev_choice x (+1 if that trial was rewarded, -1 if omitted)",
        "   first trial of each session: prev_choice = wsls = 0",
        "   Weights are log-odds of choosing RIGHT; logit(stay) = prev_choice*logit(right).",
        "",
        "2. PARAMETERS",
        fmt(params),
        "",
        "   Logits implied by the weights (bias included, by previous side):",
        fmt(formula),
        "",
        "3. P(STAY | STATE, PREVIOUS OUTCOME)  -- pooled over trials,",
        "   empirical weighted by the one-step-ahead P(state):",
        fmt(pooled),
        "",
        "   Across animals (mean +- SEM):",
        fmt(agg),
        "",
        "   CAUTION: the one-step-ahead P(exploit) never exceeds ~0.91 (sticky",
        "   transitions), so these 'empirical' columns are always a MIXTURE of both",
        "   states and cannot be compared one-to-one with the state-pure predicted",
        "   column. The correct comparison is the calibration table in section 4.",
        "",
        "4. POSTERIOR PREDICTIVE CHECKS",
        "   (a) states marginalised:",
        fmt(ppc),
        "",
        "   (b) calibration: data vs model mixture by one-step-ahead P(exploit):",
        fmt(calib),
        "",
        "   (c) which state the smoothed posterior assigns switch trials to:",
        fmt(swa),
        "",
        "5. FITTING",
        "   HMM restarts at every SESSION: each session is a separate sequence",
        "   starting from pi0 (shared across sessions). One global parameter set",
        "   for all 31 animals; only the state sequence is per session.",
        "   Settings parsed from the source:",
        *[f"     {k} = {v}" for k, v in consts.items()],
        "   Fit = Ashwood hierarchical init (GLM -> global GLM-HMM), EM with MAP",
        "   M-step, N_RESTARTS random restarts around the GLM weights; best LL kept.",
        f"   In-sample LL/trial (numpy forward pass) = {tt.attrs['ll_per_trial']:.4f}",
        "",
        "   Cross-validated held-out LL/trial by K (5-fold, sessions stratified by animal):",
        fmt(cvs) if cvs is not None else "   (glmhmm_cv.csv not found)",
        "",
        "6. PIPELINE CHECK (numpy forward-backward vs cached states):",
        f"   {post_check}",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
