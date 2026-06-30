"""
master_model.py
================
Model orchestration: registry of models + generic in-sample fitting and
out-of-sample cross-validation that loop over whatever models are registered.

Add a new model type by writing its own file in this folder (e.g. bayesian_models.py,
belief_vhr.py) that exposes a fit function, a per-trial log-likelihood function and
its parameter labels, then registering it in the MODELS list below. Turning a model
on/off is a one-line edit to MODELS — nothing else changes, and the entry point
(master_bandit.py) only has a single "run the models" toggle.

Each model is described by a ModelSpec:
    name          : str
    fit_fn        : fit_fn(*inputs, n_restarts, rng, score_mask=None) -> result dict
                    (keys: fitpar, negloglike, bic, nlike). If the fit function
                    also accepts an `optimizer` keyword (the belief models do),
                    the registry passes the module-level OPTIMIZER through to it;
                    fit functions without that keyword are called unchanged.
    loglike_fn    : loglike_fn(params, *inputs) -> per-trial log-likelihood array
                    (NaN on miss); the latent state propagates through every trial
    param_labels  : list[str], names of the fitted parameters (order matches fitpar)
    prepare       : prepare(df_ses) -> tuple of per-session inputs. The first element
                    MUST be the choice array (used for the miss mask). The default
                    returns (choice, reward, n_rules); a model that needs extra inputs
                    (e.g. belief_vhr needs trials-since-criterion) provides its own.

Cross-validation re-fits each model on a TRAIN subset and scores the held-out TEST
trials. Because the latent state is sequential, the test log-likelihood is taken from
a single full-session pass (warmed up through the preceding trials).
"""

import inspect
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from behavior.beh_models.bayesian_models import (
    fit_belief,
    fit_belief_ck,
    belief_trial_loglikes,
    belief_ck_trial_loglikes,
    _HAVE_BADS,
)
# Re-exported so callers can reach the hazard diagnostic through this module too.
from behavior.beh_models.belief_vhr import (
    empirical_hazard,
    plot_empirical_hazard,
    fit_belief_vhr,
    belief_vhr_trial_loglikes,
    prepare_vhr,
    simulate_belief_vhr,
)


# ===========================================================================
# Model registry  (toggle models on/off here)
# ===========================================================================

def _prepare_choice_reward(df_ses: pd.DataFrame):
    """Default model inputs: (choice, reward, n_rules)."""
    c = df_ses["choice"].values.astype(float)
    r = df_ses["rewarded"].values.astype(float)
    n_rules = int(df_ses["n_rules"].iloc[0])
    return (c, r, n_rules)


@dataclass
class ModelSpec:
    name: str
    fit_fn: Callable
    loglike_fn: Callable
    param_labels: List[str]
    prepare: Callable = _prepare_choice_reward


@lru_cache(maxsize=None)
def _accepts_optimizer(fit_fn: Callable) -> bool:
    """True if fit_fn takes an `optimizer` keyword (cached per function)."""
    try:
        return "optimizer" in inspect.signature(fit_fn).parameters
    except (TypeError, ValueError):
        return False


def _fit_kwargs(fit_fn, n_restarts, rng, score_mask=None, optimizer=None):
    """Assemble the keyword args for a fit_fn call, adding `optimizer` only when
    the fit function accepts it (keeps the registry generic across model types)."""
    kw = {"n_restarts": n_restarts, "rng": rng}
    if score_mask is not None:
        kw["score_mask"] = score_mask
    if optimizer is not None and _accepts_optimizer(fit_fn):
        kw["optimizer"] = optimizer
    return kw


def _require_optimizer(optimizer: str):
    """Fail fast if BADS is requested but pybads is missing.

    Without this, every per-session fit would raise, get caught, warn, and skip
    — leaving an empty model_fits.csv behind a wall of warnings. Better to stop
    immediately with one clear message.
    """
    if optimizer == "bads" and not _HAVE_BADS:
        raise ImportError(
            "optimizer='bads' but pybads is not installed. Install it with "
            "`pip install pybads`, or set OPTIMIZER='lbfgs' in master_model.py "
            "(or pass optimizer='lbfgs')."
        )


# The active models. Comment/uncomment lines to toggle which models are run.
MODELS: List[ModelSpec] = [
    ModelSpec("belief",    fit_belief,    belief_trial_loglikes, ["H", "beta"]),
    ModelSpec("belief_ck", fit_belief_ck, belief_ck_trial_loglikes,
              ["H", "beta", "alpha_k", "beta_k"]),
    ModelSpec("belief_vhr", fit_belief_vhr, belief_vhr_trial_loglikes,
              ["a", "b", "beta"], prepare=prepare_vhr),   # H(tau) = sigmoid(a + b*tau)
]


# ===========================================================================
# Cross-validation defaults  (configured here, not in master_bandit.py)
# ===========================================================================

CV_SCHEMES = ["temporal"]   # any of: "temporal", "block", "forward"
CV_N_RESTARTS = 5           # optimizer restarts per fit (lower = faster)
CV_N_JOBS = -1              # CPU cores (-1 = all, 1 = serial)
OPTIMIZER = "bads"          # "bads" (PyBADS, as in Murphy et al.) or "lbfgs"
                            # (scipy fallback). Passed to fit functions that
                            # accept an `optimizer` kwarg.


# ===========================================================================
# In-sample fitting
# ===========================================================================

def fit_models_session(df_ses: pd.DataFrame, models=None, n_restarts: int = 5,
                       rng=None, optimizer: str = None) -> dict:
    """Fit every registered model to one session. Returns {name: result dict|None}."""
    models = MODELS if models is None else models
    optimizer = OPTIMIZER if optimizer is None else optimizer
    out = {}
    for spec in models:
        try:
            kw = _fit_kwargs(spec.fit_fn, n_restarts, rng, optimizer=optimizer)
            out[spec.name] = spec.fit_fn(*spec.prepare(df_ses), **kw)
        except Exception as e:
            warnings.warn(f"Model fitting failed ({spec.name}): {e}")
            out[spec.name] = None
    return out


def _fit_one_session(animal, ses_file, df_ses, models, n_restarts, seed_seq,
                     optimizer="bads"):
    """Worker: fit every model for one session; returns a list of row dicts."""
    rng = np.random.default_rng(seed_seq)
    rows = []
    for spec in models:
        try:
            kw = _fit_kwargs(spec.fit_fn, n_restarts, rng, optimizer=optimizer)
            res = spec.fit_fn(*spec.prepare(df_ses), **kw)
        except Exception as e:
            warnings.warn(f"Model fitting failed ({spec.name}): {e}")
            continue
        if res is None or res.get("fitpar") is None:
            continue
        row = {
            "animal":       animal,
            "session_file": ses_file,
            "model":        spec.name,
            "negloglike":   res["negloglike"],
            "bic":          res["bic"],
            "nlike":        res["nlike"],
        }
        for lbl, val in zip(spec.param_labels, res["fitpar"]):
            row[f"par_{lbl}"] = float(val)
        rows.append(row)
    return rows


def fit_models(
    df: pd.DataFrame,
    models=None,
    n_restarts: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
    output_dir: str = "analysis",
    seed: int = 42,
    optimizer: str = None,
) -> pd.DataFrame:
    """
    Fit every registered model to every session in df (in parallel across cores).

    Each session gets its own seeded RNG, so results are reproducible regardless
    of the number of workers. Returns one row per session × model (parameters,
    BIC, nlike) and saves analysis/model_fits.csv when output_dir is given.
    """
    models = MODELS if models is None else models
    optimizer = OPTIMIZER if optimizer is None else optimizer
    _require_optimizer(optimizer)
    sessions = list(df.groupby(["animal", "session_file"], sort=False))
    seeds = np.random.SeedSequence(seed).spawn(len(sessions))

    results = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
        delayed(_fit_one_session)(animal, ses_file, df_ses, models, n_restarts, ss,
                                  optimizer)
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


# Backwards-compatible alias (the pipeline used to be belief-specific).
model_fit_belief = fit_models


# ===========================================================================
# Cross-validation (out-of-sample model comparison)
# ===========================================================================
#
# All schemes propagate the latent state through the whole session and only the
# SCORED trials enter the objective (fit) or the reported loss (test):
#
#   scheme="temporal" : single split. Train = first (1 - test_frac) of the
#                       trials, test = the remaining suffix. Strictly causal /
#                       leakage-free.
#   scheme="block"    : leave-one-block-out. One fold per block_idx; the held-
#                       out block is the test set. The latent state is still
#                       propagated through the held-out block during fitting (it
#                       is not scored), so an early held-out block can influence
#                       the state of later train blocks — a mild, well-known
#                       optimism for sequential models. Use "temporal"/"forward"
#                       if strict causality matters.
#   scheme="forward"  : forward-chaining by block (expanding window). Fold i
#                       trains on blocks 0..i-1 and tests on block i, from
#                       min_train_blocks onward. Block-structured AND strictly
#                       causal, at the cost of less data in the early folds.

def _make_cv_folds(df_ses: pd.DataFrame, scheme: str, test_frac: float,
                   min_train_blocks: int = 1):
    """Return a list of (train_mask, test_mask) boolean arrays over trials."""
    n = len(df_ses)
    idx = np.arange(n)

    if scheme == "temporal":
        cut = int(round(n * (1.0 - test_frac)))
        cut = max(1, min(cut, n - 1))
        return [(idx < cut, idx >= cut)]

    if scheme == "block":
        if "block_idx" not in df_ses.columns:
            raise ValueError("scheme='block' requires a 'block_idx' column")
        blocks = df_ses["block_idx"].values
        folds = []
        for b in pd.unique(blocks):
            test = (blocks == b)
            train = ~test
            if train.sum() and test.sum():
                folds.append((train, test))
        return folds

    if scheme == "forward":
        if "block_idx" not in df_ses.columns:
            raise ValueError("scheme='forward' requires a 'block_idx' column")
        blocks = df_ses["block_idx"].values
        uniq = pd.unique(blocks)
        folds = []
        for i in range(max(min_train_blocks, 1), len(uniq)):
            train = np.isin(blocks, uniq[:i])
            test = (blocks == uniq[i])
            if train.sum() and test.sum():
                folds.append((train, test))
        return folds

    raise ValueError(
        f"Unknown scheme: {scheme!r} (use 'temporal', 'block', or 'forward')")


def cross_validate_models_session(
    df_ses: pd.DataFrame,
    models=None,
    scheme: str = "temporal",
    test_frac: float = 0.30,
    n_restarts: int = 5,
    min_train_blocks: int = 1,
    rng=None,
    optimizer: str = None,
) -> dict:
    """
    Cross-validate every registered model on one session.

    For each fold the model is re-fit on the train trials (score_mask = train)
    and the held-out test trials are scored from a single full-session pass.
    Lower cv_negloglike (equivalently higher cv_nlike) wins. Returns a dict
    keyed by model name with model, scheme, n_folds, cv_negloglike, n_test,
    cv_nlike.
    """
    models = MODELS if models is None else models
    optimizer = OPTIMIZER if optimizer is None else optimizer
    if rng is None:
        rng = np.random.default_rng()

    folds = _make_cv_folds(df_ses, scheme, test_frac, min_train_blocks)

    out = {}
    for spec in models:
        inputs = spec.prepare(df_ses)
        choice = inputs[0]                     # first input is always the choice array
        cv_nll = 0.0
        n_test = 0
        for train_mask, test_mask in folds:
            try:
                kw = _fit_kwargs(spec.fit_fn, n_restarts, rng,
                                 score_mask=train_mask, optimizer=optimizer)
                res = spec.fit_fn(*inputs, **kw)
            except Exception as e:
                warnings.warn(f"CV fit failed ({spec.name}): {e}")
                continue
            if res["fitpar"] is None:
                continue
            ll = spec.loglike_fn(res["fitpar"], *inputs)
            cv_nll += float(-np.nansum(ll[test_mask]))
            n_test += int(np.sum(~np.isnan(choice) & test_mask))

        out[spec.name] = {
            "model":         spec.name,
            "scheme":        scheme,
            "n_folds":       len(folds),
            "cv_negloglike": cv_nll,
            "n_test":        n_test,
            "cv_nlike":      float(np.exp(-cv_nll / max(n_test, 1))),
        }
    return out


def _cv_one_session(animal, ses_file, df_ses, models, scheme, test_frac,
                    n_restarts, min_train_blocks, seed_seq, optimizer="bads"):
    """Worker: cross-validate every model for one session; returns row dicts."""
    rng = np.random.default_rng(seed_seq)
    res = cross_validate_models_session(
        df_ses, models=models, scheme=scheme, test_frac=test_frac,
        n_restarts=n_restarts, min_train_blocks=min_train_blocks, rng=rng,
        optimizer=optimizer,
    )
    return [{"animal": animal, "session_file": ses_file, **d} for d in res.values()]


def cross_validate_models(
    df: pd.DataFrame,
    models=None,
    scheme: str = "temporal",
    test_frac: float = 0.30,
    n_restarts: int = 5,
    min_train_blocks: int = 1,
    n_jobs: int = -1,
    verbose: bool = True,
    output_dir: str = "analysis",
    seed: int = 42,
    optimizer: str = None,
) -> pd.DataFrame:
    """
    Cross-validate every registered model for every session in df (in parallel).

    Returns one row per session × model (cv_negloglike, n_test, cv_nlike) and
    saves analysis/model_cv.csv when output_dir is given.
    """
    models = MODELS if models is None else models
    optimizer = OPTIMIZER if optimizer is None else optimizer
    _require_optimizer(optimizer)
    sessions = list(df.groupby(["animal", "session_file"], sort=False))
    seeds = np.random.SeedSequence(seed).spawn(len(sessions))

    results = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
        delayed(_cv_one_session)(animal, ses_file, df_ses, models, scheme,
                                 test_frac, n_restarts, min_train_blocks, ss,
                                 optimizer)
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


# ===========================================================================
# Orchestrator — the single entry point used by master_bandit.py
# ===========================================================================

def run_models(
    df: pd.DataFrame,
    models=None,
    schemes=None,
    n_restarts=None,
    n_jobs=None,
    output_dir: str = "analysis",
    make_plots: bool = True,
    verbose: bool = True,
    seed: int = 42,
    optimizer: str = None,
):
    """
    Fit and cross-validate every registered model, save the CSVs, print a
    summary, and (optionally) draw the model-comparison figures.

    Which models run is controlled by MODELS; the cross-validation schemes,
    speed knobs and optimizer default to the CV_* / OPTIMIZER constants in this
    module. This is the only function master_bandit.py needs to call.

    Returns (fit_df, cv_df).
    """
    models = MODELS if models is None else models
    schemes = list(CV_SCHEMES if schemes is None else schemes)
    n_restarts = CV_N_RESTARTS if n_restarts is None else n_restarts
    n_jobs = CV_N_JOBS if n_jobs is None else n_jobs
    optimizer = OPTIMIZER if optimizer is None else optimizer
    _require_optimizer(optimizer)   # stop now if BADS requested but unavailable

    print("Active models: " + ", ".join(s.name for s in models)
          + f"  |  optimizer: {optimizer}")

    # In-sample fit -> analysis/model_fits.csv
    print("\nIn-sample fit...")
    fit_df = fit_models(df, models=models, n_restarts=n_restarts, n_jobs=n_jobs,
                        verbose=verbose, output_dir=output_dir, seed=seed,
                        optimizer=optimizer)

    # Cross-validation across schemes -> analysis/model_cv.csv
    cv_parts = []
    for scheme in schemes:
        print(f"\nCross-validation (scheme = {scheme})...")
        cv_parts.append(cross_validate_models(
            df, models=models, scheme=scheme, n_restarts=n_restarts,
            n_jobs=n_jobs, verbose=verbose, output_dir=None, seed=seed,
            optimizer=optimizer))
    cv_df = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
    if output_dir and len(cv_df):
        out = Path(output_dir) / "model_cv.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(out, index=False)
        print(f"Saved → {out}")

    # Console summary: median out-of-sample likelihood per model, per scheme.
    if len(cv_df):
        print("\nCross-validation summary (median per-trial test likelihood):")
        for scheme in schemes:
            sub = cv_df[cv_df["scheme"] == scheme]
            if sub.empty:
                continue
            med = sub.groupby("model")["cv_nlike"].median().sort_values(ascending=False)
            n_ses = sub["session_file"].nunique()
            line = "  ".join(f"{m}={v:.3f}" for m, v in med.items())
            print(f"  {scheme:>8} (n={n_ses}): {line}  | best: {med.index[0]}")

    # Model-comparison figures. Lazy import avoids a circular dependency
    # (master_behavior imports fit_models from this module).
    if make_plots:
        try:
            from behavior.master_behavior import plot_model_bic, plot_model_cv
            if len(fit_df):
                plot_model_bic(fit_df)
            if len(cv_df):
                plot_model_cv(cv_df)
        except Exception as e:
            warnings.warn(f"Model-comparison plots skipped: {e}")

    return fit_df, cv_df