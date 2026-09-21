"""
glmhmm_decode.py
================
Decode GLM-HMM states on NEW sessions with the frozen global model.

The neuromodulator imaging sessions (phase 31/32) are not part of the
behavioral dataset the global GLM-HMM was fit on. Re-fitting on them would
change what "exploit" and "explore" mean; instead, the states are INFERRED on
these sessions with the global parameters held fixed (weights, transition
matrix, initial distribution), so a state label means the same thing as in the
behavioral validation.

Requires the pickle written by run_glmhmm.py after the final fit:
    analysis/glmhmm_model_<parametrization>_K<K>.pkl

Two posteriors are returned per trial:
    p_engaged          : SMOOTHED posterior (forward-backward, uses the whole
                         session). Same quantity as glmhmm_states_K2.csv.
    p_engaged_filtered : FILTERED posterior (forward pass only, uses trials
                         <= t). Causal: prefer it when relating the state at
                         trial t to neural activity at or before trial t, since
                         the smoothed posterior already "knows" choices t+1, t+2...

Run from behavior/beh_models/ in the `ssm` env:
    conda activate ssm
    python glmhmm_decode.py
"""

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
import hmmGlm as g  # noqa: E402

MODEL_PKL = REPO_ROOT / "analysis" / "glmhmm_model_ashwood_wsls_K2.pkl"
OUT_CSV = REPO_ROOT / "analysis" / "glmhmm_states_neuromodulator_K2.csv"
PHASES = [31]


def load_model(path: Path = MODEL_PKL) -> dict:
    """Load the bundle written by run_glmhmm.py."""
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    for key in ("model", "parametrization", "n_lags", "engaged_idx"):
        if key not in bundle:
            raise KeyError(f"{path.name} has no '{key}' -- re-run run_glmhmm.py")
    return bundle


def _filtered_posterior(model, choices, inputs, mask) -> np.ndarray:
    """Forward-only posterior P(z_t | y_1..t). Uses ssm's filter() when
    available, otherwise an explicit forward pass with the model's own terms."""
    if hasattr(model, "filter"):
        try:
            return np.asarray(model.filter(choices, input=inputs, mask=mask))
        except TypeError:
            pass
    pi0 = model.init_state_distn.initial_state_distn
    Ps = model.transitions.transition_matrices(choices, inputs, mask, None)
    log_lik = model.observations.log_likelihoods(choices, inputs, mask, None)
    T, K = log_lik.shape
    alpha = np.zeros((T, K))
    a = pi0 * np.exp(log_lik[0] - log_lik[0].max())
    alpha[0] = a / a.sum()
    for t in range(1, T):
        P = Ps[t - 1] if Ps.ndim == 3 else Ps
        a = (alpha[t - 1] @ P) * np.exp(log_lik[t] - log_lik[t].max())
        alpha[t] = a / a.sum()
    return alpha


def decode_session(bundle: dict, choice: np.ndarray, rewarded: np.ndarray) -> pd.DataFrame:
    """Decode one session.

    choice   : -1 left / +1 right / NaN miss (same coding as the behavioral CSV)
    rewarded : 1 / 0 (NaN or 0 on miss)
    Returns a per-trial DataFrame (row order = input order).
    """
    df_ses = pd.DataFrame({"choice": np.asarray(choice, float),
                           "rewarded": np.asarray(rewarded, float)})
    ch, X, mk = g.build_session_arrays(df_ses, n_lags=bundle["n_lags"],
                                       parametrization=bundle["parametrization"])
    model, e = bundle["model"], bundle["engaged_idx"]
    post = np.asarray(model.expected_states(ch, input=X, mask=mk)[0])
    filt = _filtered_posterior(model, ch, X, mk)
    return pd.DataFrame({
        "trial_idx": np.arange(len(df_ses)),
        "glmhmm_state": post.argmax(1),
        "is_exploit": (post.argmax(1) == e).astype(int),
        "p_engaged": post[:, e],
        "p_engaged_filtered": filt[:, e],
    })


def decode_trial_table(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Decode every session in a trial table with columns
    animal, session_file, choice, rewarded (trials in session order)."""
    out = []
    for (animal, ses), d in df.groupby(["animal", "session_file"], sort=False):
        dec = decode_session(bundle, d["choice"].to_numpy(), d["rewarded"].to_numpy())
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
    bundle = load_model()
    print(f"loaded {MODEL_PKL.name}: K={bundle['K']}, {bundle['parametrization']}, "
          f"engaged_idx={bundle['engaged_idx']}")

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
        choice = beh["choice_lr"].to_numpy()
        rewarded = beh["rewarded"].to_numpy()
        dec = decode_session(bundle, choice, rewarded)
        dec.insert(0, "session_file", Path(str(r["session_file"])).stem)
        dec.insert(0, "animal", str(r["animal"]))
        dec["experiment"] = r["experiment"]
        rows.append(dec)
    out = pd.concat(rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    occ = out.groupby(["experiment", "animal"])["is_exploit"].mean().round(3)
    print("\nFraction of trials in the exploit state (smoothed posterior):")
    print(occ.to_string())
    print(f"\nwrote {OUT_CSV}  ({out['session_file'].nunique()} sessions, {len(out)} trials)")


if __name__ == "__main__":
    main()
