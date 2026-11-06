"""
glmhmm_decode.py
================
Decode GLM-HMM states on NEW sessions with the frozen global model.

The neuromodulator imaging sessions (phase 31/32) are not part of the
behavioral dataset the global GLM-HMM was fit on. Re-fitting on them would
change what the states mean; instead, the states are INFERRED on these
sessions with the global parameters held fixed (weights, transition matrix,
initial distribution), so a state label means the same thing as in the
behavioral validation.

Model: the PRIMARY specification (ashwood_wsls, K = 2, prior sigma = 1), read
from the JSON written by glmhmm_fit_primary.py:
    analysis/glmhmm_primary_K2_sigma1.json
Numpy only (no ssm, no pickle). The design matrix is built exactly as for the
fit (glmhmm_regularization.session_design == hmmGlm.build_session_arrays,
ashwood_wsls branch).

Three posteriors of the engaged state are returned per trial t:
    p_engaged_predicted : ONE-STEP-AHEAD PREDICTED  P(z_t | y_1..t-1)
                          = filtered(t-1) @ P, and pi0 on the first trial.
                          Uses only choices/outcomes BEFORE trial t, i.e. what
                          is known at cue onset of trial t. Use it for any
                          pre-cue neural window.
    p_engaged_filtered  : FILTERED  P(z_t | y_1..t). Includes the choice on
                          trial t itself, so it is NOT pre-cue.
    p_engaged           : SMOOTHED  P(z_t | y_1..T). Uses the whole session
                          (future choices too); for description only.
On a missed trial nothing is observed, so filtered(t) = predicted(t).

Inputs (relative to reversalExpectation_py/):
    analysis/glmhmm_primary_K2_sigma1.json
    behaviour files of the neuromodulator sessions (master_neuromodulator index)
Output:
    analysis/glmhmm_states_neuromodulator_K2.csv

Run from reversalExpectation_py/ (revExp env, no ssm needed):
    python behavior/beh_models/glmhmm_decode.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
import glmhmm_regularization as gr  # noqa: E402

MODEL_JSON = REPO_ROOT / "analysis" / "glmhmm_primary_K2_sigma1.json"
OUT_CSV = REPO_ROOT / "analysis" / "glmhmm_states_neuromodulator_K2.csv"
PHASES = [31]


def load_model(path: Path = MODEL_JSON) -> dict:
    """Load the frozen primary model written by glmhmm_fit_primary.py."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m["parametrization"] != "ashwood_wsls":
        raise ValueError(f"decoder only supports ashwood_wsls, got {m['parametrization']}")
    return {"W": np.asarray(m["W_logodds_right"], float),
            "w_glm": np.asarray(m["glm_w_logodds_right"], float)
            if "glm_w_logodds_right" in m else None,
            "P": np.asarray(m["transition_matrix"], float),
            "pi0": np.asarray(m["pi0"], float),
            "engaged_idx": int(m["engaged_idx"]),
            "K": int(m["K"]), "prior_sigma": float(m["prior_sigma"])}


def posteriors(model: dict, y: np.ndarray, observed: np.ndarray, X: np.ndarray):
    """Predicted, filtered and smoothed state posteriors of one session.

    y        : (T,) 1 = right, 0 = left (ignored where not observed)
    observed : (T,) bool, trial with a response
    X        : (T, 3) design [bias, prev_choice, wsls]
    Returns three (T, K) arrays: predicted, filtered, smoothed.
    """
    W, P, pi0 = model["W"], model["P"], model["pi0"]
    T, K = len(y), len(pi0)
    p_right = expit(X @ W.T)                                   # (T, K)
    lik = np.where(y[:, None] == 1, p_right, 1.0 - p_right)
    lik[~observed] = 1.0                                       # a miss carries no evidence

    pred = np.empty((T, K))
    filt = np.empty((T, K))
    prior = pi0.copy()
    for t in range(T):
        pred[t] = prior                                        # P(z_t | y_<t)
        a = prior * lik[t]
        filt[t] = a / a.sum()                                  # P(z_t | y_<=t)
        prior = filt[t] @ P

    # backward pass for the smoothed posterior (scaled by the filter normalisers)
    beta = np.ones(K)
    smooth = np.empty((T, K))
    smooth[-1] = filt[-1]
    for t in range(T - 2, -1, -1):
        beta = P @ (lik[t + 1] * beta)
        beta /= beta.sum()
        s = filt[t] * beta
        smooth[t] = s / s.sum()
    return pred, filt, smooth


def session_loglik(model: dict, choice: np.ndarray, rewarded: np.ndarray):
    """Log-likelihood of one session under the frozen GLM-HMM and, if saved, the
    1-state GLM (same prior). Returns (ll_glmhmm, ll_glm or NaN, n_observed)."""
    df_ses = pd.DataFrame({"choice": np.asarray(choice, float),
                           "rewarded": np.asarray(rewarded, float)})
    y, obs, X = gr.session_design(df_ses)
    W, P = model["W"], model["P"]
    p_right = expit(X @ W.T)
    lik = np.where(y[:, None] == 1, p_right, 1.0 - p_right)
    lik[~obs] = 1.0
    ll, prior = 0.0, model["pi0"].copy()
    for t in range(len(y)):
        a = prior * lik[t]
        s = a.sum()
        ll += np.log(s)
        prior = (a / s) @ P
    ll_glm = np.nan
    if model.get("w_glm") is not None:
        q = expit(X @ model["w_glm"])
        ll_glm = float(np.sum(np.where(y == 1, np.log(q), np.log1p(-q))[obs]))
    return float(ll), ll_glm, int(obs.sum())


def decode_session(model: dict, choice: np.ndarray, rewarded: np.ndarray) -> pd.DataFrame:
    """Decode one session.

    choice   : -1 left / +1 right / NaN miss (same coding as the behavioral CSV)
    rewarded : 1 / 0 (NaN or 0 on miss)
    Returns a per-trial DataFrame (row order = input order).
    """
    df_ses = pd.DataFrame({"choice": np.asarray(choice, float),
                           "rewarded": np.asarray(rewarded, float)})
    y, obs, X = gr.session_design(df_ses)
    pred, filt, smooth = posteriors(model, y, obs, X)
    e = model["engaged_idx"]
    return pd.DataFrame({
        "trial_idx": np.arange(len(df_ses)),
        "glmhmm_state": smooth.argmax(1),
        "is_exploit": (smooth.argmax(1) == e).astype(int),
        "p_engaged": smooth[:, e],
        "p_engaged_filtered": filt[:, e],
        "p_engaged_predicted": pred[:, e],
    })


def decode_trial_table(model: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Decode every session in a trial table with columns
    animal, session_file, choice, rewarded (trials in session order)."""
    out = []
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        dec = decode_session(model, d["choice"].to_numpy(), d["rewarded"].to_numpy())
        dec.insert(0, "session_file", ses)
        dec.insert(0, "animal", animal)
        out.append(dec)
    return pd.concat(out, ignore_index=True)


def _load_sibling(folder: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    model = load_model()
    print(f"loaded {MODEL_JSON.name}: K={model['K']}, sigma={model['prior_sigma']}, "
          f"engaged_idx={model['engaged_idx']}")

    # Same session index and behavior loader as the Wang pipeline
    nm = REPO_ROOT / "neuromodulator"
    master = _load_sibling(nm, "master_neuromodulator")
    runner = _load_sibling(nm, "run_wang2025_neuromodulator")
    idx = master.build_index()
    idx = idx[idx["phase"].isin(PHASES)]

    rows = []
    for _, r in idx.iterrows():
        try:
            beh = runner.load_behavior(Path(r["beh_path"]))
        except Exception as exc:
            print(f"  skip {r['session_file']}: {exc!r}")
            continue
        dec = decode_session(model, beh["choice_lr"].to_numpy(), beh["rewarded"].to_numpy())
        dec.insert(0, "session_file", Path(str(r["session_file"])).stem)
        dec.insert(0, "animal", str(r["animal"]))
        dec["experiment"] = r["experiment"]
        rows.append(dec)
    out = pd.concat(rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    occ = out.groupby(["experiment", "animal"])["is_exploit"].mean().round(3)
    print("\nFraction of trials in the engaged state (smoothed posterior):")
    print(occ.to_string())
    c = out[["p_engaged_predicted", "p_engaged_filtered", "p_engaged"]].corr().round(3)
    print("\nCorrelation between the three posteriors:")
    print(c.to_string())
    print(f"\nwrote {OUT_CSV}  ({out['session_file'].nunique()} sessions, {len(out)} trials)")


if __name__ == "__main__":
    main()
