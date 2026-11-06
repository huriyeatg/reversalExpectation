"""
glmhmm_fit_primary.py
=====================
PRIMARY GLM-HMM specification of the project, used for every final result
(behavioural figures and the neuromodulator state trajectories):

    parametrization : ashwood_wsls  (bias, prev_choice, wsls)
    K               : 2
    prior on weights: Gaussian, sigma = 1   (MAP; see glmhmm_regularization.py:
                      sigma = 1 costs < 0.0001 nats/trial of held-out LL and
                      removes the non-identified quasi-separated exploit weights)
    transitions     : Dirichlet alpha = 1 (no prior), kappa = 0
    fit             : numpy EM of glmhmm_regularization.py, Ashwood hierarchical
                      init (GLM -> GLM-HMM), N_RESTARTS restarts, best
                      log-posterior kept
    data            : naive mice, sessions meeting criteria, >= 20 trials
                      (594 sessions, 31 animals; same selection as hmmGlm.py)

The 1-state GLM (K = 1, same prior) is fitted and saved too, as the reference
model of the held-out comparisons (glmhmm_heldout_validation.py).

The fitted parameters are written to a plain JSON (no pickle, no ssm), read by
glmhmm_decode.py to infer states on new sessions with the model frozen.

Convention in the JSON: W[k] = weights of state k on [bias, prev_choice, wsls]
as log-odds of a RIGHT choice. States are ordered by prev_choice weight
(descending), so state 0 is the engaged / high history-dependence state.

Inputs  (relative to reversalExpectation_py/):
    analysis/bandit_R71_lesion.csv
Outputs:
    analysis/glmhmm_primary_K2_sigma1.json

Usage (from reversalExpectation_py/):
    python behavior/beh_models/glmhmm_fit_primary.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glmhmm_regularization as gr          # noqa: E402

REPO_ROOT = HERE.parents[1]
CSV_PATH = REPO_ROOT / "analysis" / "bandit_R71_lesion.csv"
OUT_JSON = REPO_ROOT / "analysis" / "glmhmm_primary_K2_sigma1.json"

PARAMETRIZATION = "ashwood_wsls"
K = 2
SIGMA = 1.0
N_RESTARTS = 10
SEED = 0
REGRESSORS = ["bias", "prev_choice", "wsls"]
STATE_LABELS = ["engaged", "disengaged"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    args = ap.parse_args()

    sessions, tags = gr.load_sessions(args.csv)
    b = gr.Batch(sessions)
    print(f"[data] {len(sessions)} sessions, {len({t[0] for t in tags})} animals, "
          f"{int(b.n_obs)} observed trials", flush=True)

    t0 = time.time()
    fit = gr.fit_glmhmm(b, SIGMA, args.restarts, SEED, K=K)
    print(f"[fit] done in {time.time() - t0:.0f}s, restarts agree "
          f"{fit['restarts_agree']}/{fit['n_restarts']}", flush=True)

    W, P, pi0 = np.asarray(fit["W"]), np.asarray(fit["P"]), np.asarray(fit["pi0"])
    # 1-state GLM with the same prior: the K = 1 reference for held-out comparisons
    w_glm = np.asarray(gr.fit_glm(b, SIGMA))
    ll, *_ = gr.forward_backward(b, W, P, pi0, need_posteriors=False)

    # P(stay) per state after reward / omission, for a readable summary
    p_stay = {}
    for k in range(K):
        for outcome, sgn in (("reward", 1.0), ("omission", -1.0)):
            # previous choice = right (+1): stay = choose right again
            logit = W[k, 0] + W[k, 1] * 1.0 + W[k, 2] * sgn
            p_stay[f"{STATE_LABELS[k]}_after_{outcome}"] = float(1 / (1 + np.exp(-logit)))

    out = {
        "parametrization": PARAMETRIZATION,
        "K": K,
        "prior_sigma": SIGMA,
        "regressors": REGRESSORS,
        "state_labels": STATE_LABELS,
        "engaged_idx": 0,
        "W_logodds_right": W.tolist(),
        "glm_w_logodds_right": w_glm.tolist(),
        "transition_matrix": P.tolist(),
        "pi0": pi0.tolist(),
        "dwell_trials": (1 / (1 - np.diag(P))).tolist(),
        "p_stay": p_stay,
        "train_loglik": float(ll),
        "train_loglik_per_trial": float(ll / b.n_obs),
        "n_sessions": len(sessions),
        "n_observed_trials": int(b.n_obs),
        "n_restarts": int(fit["n_restarts"]),
        "restarts_agree": int(fit["restarts_agree"]),
        "data_csv": args.csv.name,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
