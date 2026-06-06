"""
models_pipeline.py
================
Model fitting pipeline.

In-sample fitting (model_fit_belief) is unchanged. Added below it is
out-of-sample cross-validation (cross_validate_models_session /
cross_validate_models), which re-fits each model on a TRAIN subset of trials
and scores the held-out TEST trials with the fitted parameters. Because the
belief / choice-kernel state is sequential, the test log-likelihood is taken
from a single full-session pass (so the latent state is warmed up through the
preceding trials) rather than by re-initialising on the test segment.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from behavior.beh_models.bayesian_models import (
    fit_belief,
    fit_belief_ck,
    belief_trial_loglikes,
    belief_ck_trial_loglikes,
)


def fit_models_session(df_ses: pd.DataFrame, n_restarts: int = 5,
                       rng=None) -> dict:
    """
    Fit belief and belief-CK models to one session.

    Returns dict keyed by model name with fit result dicts
    (keys: model, fitpar, negloglike, bic, nlike).
    """
    c       = df_ses["choice"].values.astype(float)
    r       = df_ses["rewarded"].values.astype(float)
    n_rules = int(df_ses["n_rules"].iloc[0])

    results = {}
    for name, fn in [("belief", fit_belief), ("belief_ck", fit_belief_ck)]:
        try:
            results[name] = fn(c, r, n_rules=n_rules, n_restarts=n_restarts, rng=rng)
        except Exception as e:
            warnings.warn(f"Model fitting failed ({name}): {e}")
            results[name] = None
    return results


def _fit_one_session(animal, ses_file, df_ses, n_restarts, seed_seq):
    """Worker: fit all models for one session; returns a list of row dicts."""
    rng = np.random.default_rng(seed_seq)
    fit = fit_models_session(df_ses, n_restarts=n_restarts, rng=rng)
    rows = []
    for model_name, res in fit.items():
        if res is None:
            continue
        row = {
            "animal":       animal,
            "session_file": ses_file,
            "model":        model_name,
            "negloglike":   res["negloglike"],
            "bic":          res["bic"],
            "nlike":        res["nlike"],
        }
        par = res["fitpar"] if res["fitpar"] is not None else []
        labels = (["H", "beta"] if model_name == "belief"
                  else ["H", "beta", "alpha_k", "beta_k"])
        for lbl, val in zip(labels, par):
            row[f"par_{lbl}"] = float(val)
        rows.append(row)
    return rows


def model_fit_belief(
    df: pd.DataFrame,
    n_restarts: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
    output_dir: str = "analysis",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fit belief and belief-CK models to every session in df.

    Sessions are fit in parallel across CPU cores (n_jobs=-1 uses all cores,
    n_jobs=1 runs serially). Each session gets its own seeded RNG, so results
    are reproducible regardless of the number of workers.

    Returns a DataFrame with one row per session × model containing
    animal, session_file, model name, fitted parameters, BIC, and nlike.
    Saves a CSV when output_dir is given.
    """
    sessions = list(df.groupby(["animal", "session_file"], sort=False))
    seeds = np.random.SeedSequence(seed).spawn(len(sessions))

    results = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
        delayed(_fit_one_session)(animal, ses_file, df_ses, n_restarts, ss)
        for ss, ((animal, ses_file), df_ses) in zip(seeds, sessions)
    )
    rows = [row for session_rows in results for row in session_rows]

    fit_df = pd.DataFrame(rows)

    if output_dir and len(fit_df):
        out = Path(output_dir) / "model_fits.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fit_df.to_csv(out, index=False)
        print(f"Saved → {out}")

    return fit_df


# ===========================================================================
# Cross-validation (out-of-sample model comparison)
# ===========================================================================
#
# Both schemes propagate the latent state through the whole session and only
# the SCORED trials enter the objective (fit) or the reported loss (test):
#
#   scheme="temporal" : single split. Train = first (1 - test_frac) of the
#                       trials, test = the remaining suffix. Strictly causal /
#                       leakage-free: scored train trials all precede the test
#                       trials, so warming the belief through the test segment
#                       cannot affect the fit.
#   scheme="block"    : leave-one-block-out. One fold per block_idx; the held-
#                       out block is the test set, the rest is train. Note: the
#                       latent state is still propagated through the held-out
#                       block during fitting (it is not scored), so an early
#                       held-out block can influence the state of later train
#                       blocks — a mild, well-known optimism for sequential
#                       models. Use "temporal" or "forward" if strict causality
#                       matters.
#   scheme="forward"  : forward-chaining by block (expanding window). Fold i
#                       trains on blocks 0..i-1 and tests on block i, for i from
#                       min_train_blocks onward. Block-structured AND strictly
#                       causal (the test block always follows every train
#                       block, so there is no leakage), at the cost of using
#                       less data in the early folds.

def _make_cv_folds(df_ses: pd.DataFrame, scheme: str, test_frac: float,
                   min_train_blocks: int = 1):
    """Return a list of (train_mask, test_mask) boolean arrays over trials."""
    n = len(df_ses)
    idx = np.arange(n)

    if scheme == "temporal":
        cut = int(round(n * (1.0 - test_frac)))
        cut = max(1, min(cut, n - 1))
        train = idx < cut
        test = idx >= cut
        return [(train, test)]

    if scheme == "block":
        if "block_idx" not in df_ses.columns:
            raise ValueError("scheme='block' requires a 'block_idx' column")
        blocks = df_ses["block_idx"].values
        folds = []
        for b in pd.unique(blocks):           # in order of appearance
            test = (blocks == b)
            train = ~test
            if train.sum() == 0 or test.sum() == 0:
                continue
            folds.append((train, test))
        return folds

    if scheme == "forward":
        if "block_idx" not in df_ses.columns:
            raise ValueError("scheme='forward' requires a 'block_idx' column")
        blocks = df_ses["block_idx"].values
        uniq = pd.unique(blocks)              # in order of appearance
        folds = []
        for i in range(max(min_train_blocks, 1), len(uniq)):
            train = np.isin(blocks, uniq[:i])  # blocks 0..i-1
            test = (blocks == uniq[i])         # block i
            if train.sum() == 0 or test.sum() == 0:
                continue
            folds.append((train, test))
        return folds

    raise ValueError(
        f"Unknown scheme: {scheme!r} (use 'temporal', 'block', or 'forward')")


def cross_validate_models_session(
    df_ses: pd.DataFrame,
    scheme: str = "temporal",
    test_frac: float = 0.30,
    n_restarts: int = 5,
    min_train_blocks: int = 1,
    rng=None,
) -> dict:
    """
    Cross-validate belief and belief-CK on one session.

    For every fold the model is re-fit on the train trials (score_mask = train)
    and the held-out test trials are scored with the fitted parameters from a
    single full-session pass through the per-trial log-likelihood. Test losses
    are summed across folds. Lower cv_negloglike (equivalently higher cv_nlike)
    wins — this is the out-of-sample analogue of the BIC comparison.

    Parameters
    ----------
    df_ses    : one session's trials (needs choice, rewarded, n_rules; plus
                block_idx when scheme="block" or "forward").
    scheme    : "temporal", "block", or "forward" (see module notes above).
    test_frac : test fraction for scheme="temporal".
    n_restarts: random restarts per fit.
    min_train_blocks : first block index tested by scheme="forward".
    rng       : np.random.Generator.

    Returns
    -------
    dict keyed by model name, each with:
        model, scheme, n_folds, cv_negloglike, n_test, cv_nlike
    """
    if rng is None:
        rng = np.random.default_rng()

    c       = df_ses["choice"].values.astype(float)
    r       = df_ses["rewarded"].values.astype(float)
    n_rules = int(df_ses["n_rules"].iloc[0])

    folds = _make_cv_folds(df_ses, scheme, test_frac, min_train_blocks)

    specs = [
        ("belief",    fit_belief,    belief_trial_loglikes),
        ("belief_ck", fit_belief_ck, belief_ck_trial_loglikes),
    ]

    out = {}
    for name, fit_fn, loglike_fn in specs:
        cv_nll = 0.0
        n_test = 0
        for train_mask, test_mask in folds:
            try:
                res = fit_fn(c, r, n_rules=n_rules, n_restarts=n_restarts,
                             rng=rng, score_mask=train_mask)
            except Exception as e:
                warnings.warn(f"CV fit failed ({name}): {e}")
                continue
            if res["fitpar"] is None:
                continue
            # Full-session pass with fitted params, then score the test trials.
            ll = loglike_fn(res["fitpar"], c, r, n_rules)
            cv_nll += float(-np.nansum(ll[test_mask]))
            n_test += int(np.sum(~np.isnan(c) & test_mask))

        out[name] = {
            "model":         name,
            "scheme":        scheme,
            "n_folds":       len(folds),
            "cv_negloglike": cv_nll,
            "n_test":        n_test,
            "cv_nlike":      float(np.exp(-cv_nll / max(n_test, 1))),
        }
    return out


def _cv_one_session(animal, ses_file, df_ses, scheme, test_frac, n_restarts,
                    min_train_blocks, seed_seq):
    """Worker: cross-validate all models for one session; returns row dicts."""
    rng = np.random.default_rng(seed_seq)
    res = cross_validate_models_session(
        df_ses, scheme=scheme, test_frac=test_frac, n_restarts=n_restarts,
        min_train_blocks=min_train_blocks, rng=rng,
    )
    return [{"animal": animal, "session_file": ses_file, **d} for d in res.values()]


def cross_validate_models(
    df: pd.DataFrame,
    scheme: str = "temporal",
    test_frac: float = 0.30,
    n_restarts: int = 5,
    min_train_blocks: int = 1,
    n_jobs: int = -1,
    verbose: bool = True,
    output_dir: str = "analysis",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cross-validate belief and belief-CK for every session in df.

    Sessions are cross-validated in parallel across CPU cores (n_jobs=-1 uses
    all cores, n_jobs=1 runs serially). Each session gets its own seeded RNG, so
    results are reproducible regardless of the number of workers.

    Returns a DataFrame with one row per session × model containing the
    out-of-sample loss (cv_negloglike), the held-out trial count (n_test), and
    the per-trial geometric-mean test likelihood (cv_nlike). Saves a CSV when
    output_dir is given.
    """
    sessions = list(df.groupby(["animal", "session_file"], sort=False))
    seeds = np.random.SeedSequence(seed).spawn(len(sessions))

    results = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
        delayed(_cv_one_session)(animal, ses_file, df_ses, scheme, test_frac,
                                 n_restarts, min_train_blocks, ss)
        for ss, ((animal, ses_file), df_ses) in zip(seeds, sessions)
    )
    rows = [row for session_rows in results for row in session_rows]

    cv_df = pd.DataFrame(rows)

    if output_dir and len(cv_df):
        out = Path(output_dir) / "model_cv.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(out, index=False)
        print(f"Saved → {out}")

    return cv_df