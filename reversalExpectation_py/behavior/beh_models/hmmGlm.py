"""
hmmGlm.py
=========
GLM-HMM (input-driven HMM) for the two-armed bandit task, built on Ashwood's
fork of the Linderman-lab `ssm` package (github.com/zashwood/ssm), which
handles missed ("violation") trials via masking. Each trial's choice is a
Bernoulli GLM whose log-odds weights depend on a latent discrete state; the
state evolves as a first-order Markov chain.

This module mirrors the analysis pipeline of
    Ashwood et al. (2022) "Mice alternate between discrete strategies during
    perceptual decision-making", Nature Neuroscience.
adapted to a value-based task. The IBL task's dominant regressor is the
*stimulus*, which has no analogue in a bandit, so the design matrix is
necessarily different; everything else (fitting, hierarchical initialisation,
model comparison, cross-validation) follows their recipe.

Two design-matrix parametrisations are provided:

  parametrization="ashwood_wsls"  (faithful, L=1; M = 3)
      bias, previous_choice (+-1), wsls (= prev_reward(+-1) x prev_choice(+-1))
      Drops the stimulus column of Ashwood's design matrix. previous_choice
      and previous_reward are forward-filled across misses (their imputation
      of violation trials by the nearest preceding non-violation trial).

  parametrization="reward_perseveration"  (extension; default; M = 1 + 2L)
      bias + per-lag reward-seeking and perseveration kernels
      (cf. Beron et al. 2022; Miller et al.). At L=1 this spans the same
      2-D history space as ashwood_wsls (a linear reparametrisation).

Data conventions (from master_behavior.py per-trial DataFrame):
    choice    : -1 = left, +1 = right, NaN = miss
    rewarded  :  0 / 1  (NaN treated as 0)
    sessions  : df.groupby(["animal", "session_file"], sort=False), trial order kept

Model selection uses session-level cross-validation on held-out per-trial
log-likelihood (not BIC). Missed trials are masked, never dropped.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from joblib import Parallel, delayed

import ssm

# choice categories for ssm: 0 = left, 1 = right
N_CATEGORIES = 2
OBS_DIM = 1

# defaults (override from caller / master_bandit.py)
PARAMETRIZATION = "reward_perseveration"
N_LAGS = 3
K_RANGE = (1, 2, 3, 4, 5)
N_RESTARTS = 15
N_ITERS = 200
TOLERANCE = 1e-4
CV_N_FOLDS = 5
CV_N_JOBS = -1

# hierarchical-init hyperparameters (Ashwood recipe)
INIT_NOISE = 0.2        # std of Gaussian noise added to GLM weights per state
STICKY_SELF = 0.95      # initial self-transition probability

# MAP prior hyperparameters (Ashwood global-fit values). With these the EM
# M-step performs MAP rather than MLE, preventing weights from diverging on
# near-deterministic states. prior_sigma is the std of the Gaussian prior on
# the GLM weights (smaller = stronger shrinkage); transition_alpha is the
# Dirichlet concentration on the transition rows (1 = ~MLE); transition_kappa
# adds extra mass to self-transitions (0 = none, rely on init).
PRIOR_SIGMA = 100.0
TRANSITION_ALPHA = 1.0
TRANSITION_KAPPA = 0.0


# ===========================================================================
# 1. Design-matrix construction
# ===========================================================================

def _signed_history(df_ses):
    """Return (observed, signed_choice, reward) per trial for one session."""
    c = df_ses["choice"].to_numpy(dtype=float)                            # -1 / +1 / NaN
    r = np.nan_to_num(df_ses["rewarded"].to_numpy(dtype=float), nan=0.0)  # 0 / 1
    observed = ~np.isnan(c)
    signed_choice = np.where(observed, c, 0.0)                            # +-1, 0 on miss
    return observed, signed_choice, r


def _forward_fill_previous(observed, signed_choice, reward):
    """Forward-fill choice/reward across misses, then shift by 1 trial.

    Implements Ashwood's violation handling for the history regressors: a
    missed trial's "previous choice/reward" is taken from the nearest
    preceding non-missed trial. The first trial has no history (set to 0).
    """
    T = observed.shape[0]
    filled_c = signed_choice.copy()
    filled_r = reward.copy()
    last_c, last_r = 0.0, 0.0
    for t in range(T):
        if observed[t]:
            last_c, last_r = signed_choice[t], reward[t]
        else:
            filled_c[t] = last_c
            filled_r[t] = last_r
    prev_choice = np.zeros(T)
    prev_reward = np.zeros(T)
    prev_choice[1:] = filled_c[:-1]
    prev_reward[1:] = filled_r[:-1]
    return prev_choice, prev_reward


def build_session_arrays(df_ses: pd.DataFrame, n_lags: int = N_LAGS,
                         parametrization: str = PARAMETRIZATION):
    """Build (choices, inputs, mask) for a single session DataFrame.

    Returns
    -------
    choices : (T, 1) int array in {0, 1}   (0 = left, 1 = right; placeholder 0 on misses)
    inputs  : (T, M) float array
    mask    : (T, 1) bool array            (True = choice observed)
    """
    observed, signed_choice, reward = _signed_history(df_ses)
    T = observed.shape[0]

    # observation labels for ssm: -1 -> 0 (left), +1 -> 1 (right)
    c = df_ses["choice"].to_numpy(dtype=float)
    choices = np.zeros((T, 1), dtype=int)
    choices[observed, 0] = (c[observed] > 0).astype(int)

    if parametrization == "ashwood_wsls":
        # faithful Ashwood design (minus stimulus), lag 1 only -> M = 3
        prev_choice, prev_reward = _forward_fill_previous(observed, signed_choice, reward)
        prev_reward_signed = 2.0 * prev_reward - 1.0          # {-1, +1}
        prev_reward_signed[prev_choice == 0] = 0.0            # no history -> 0
        wsls = prev_reward_signed * prev_choice               # {-1, 0, +1}
        X = np.column_stack([np.ones(T), prev_choice, wsls])

    elif parametrization == "reward_perseveration":
        # reward-seeking + perseveration kernels over n_lags -> M = 1 + 2L
        rew_signed = signed_choice * reward                   # +-1 if rewarded, else 0
        X = np.zeros((T, 1 + 2 * n_lags))
        X[:, 0] = 1.0
        for k in range(1, n_lags + 1):
            X[k:, 2 * k - 1] = rew_signed[:-k]                # reward-seeking kernel
            X[k:, 2 * k] = signed_choice[:-k]                 # perseveration kernel

    else:
        raise ValueError(f"unknown parametrization: {parametrization!r}")

    return choices, X, observed.reshape(T, 1)


def regressor_names(n_lags: int = N_LAGS, parametrization: str = PARAMETRIZATION):
    """Column names matching build_session_arrays order."""
    if parametrization == "ashwood_wsls":
        return ["bias", "prev_choice", "wsls"]
    names = ["bias"]
    for k in range(1, n_lags + 1):
        names += [f"rew_choice_{k}", f"choice_{k}"]
    return names


def n_regressors(n_lags: int = N_LAGS, parametrization: str = PARAMETRIZATION) -> int:
    return 3 if parametrization == "ashwood_wsls" else 1 + 2 * n_lags


# ---------------------------------------------------------------------------
# Inclusion criteria for the GLM-HMM input dataset (fixed for this analysis)
# ---------------------------------------------------------------------------
# The trial subset fed to the GLM-HMM is defined by three explicit criteria:
#
#   1. lesion = "naive": control + pre-lesion sessions only (df["lesioned"] is
#      NaN). This is the same subset the rest of the pipeline analyses, i.e.
#      filter_lesion_group(df, "naive") in master_bandit.py and the MATLAB
#      normalSubset = isnan(Lesioned). Post-lesion sessions (lesioned == 1) are
#      excluded.
#
#   2. phases unified: phase 3 (main bandit) and phase 8 (pupillometry) are
#      KEPT TOGETHER. They share the identical 70:10 two-reward-probability
#      rule set (RuleCodes2; see presentation_codes.py), so the task is
#      behaviourally the same; phase 8 only adds pupillometry recording.
#
#   3. performance = meets_criteria: only sessions flagged
#      df["meets_criteria"] == True are kept, removing low-performance /
#      early-training sessions (compute_session_criteria in lesion_index.py).
#
# NOTE: the Ashwood et al. minimum-sessions-per-animal inclusion rule is
# deliberately NOT applied here.
GLMHMM_LESION_GROUP = "naive"     # "naive" | "post" | "all"
GLMHMM_PHASES = None              # None = all phases unified; e.g. (3,) to restrict
GLMHMM_REQUIRE_CRITERIA = True    # keep only meets_criteria == True sessions


def select_glmhmm_sessions(df: pd.DataFrame,
                           lesion_group: str = GLMHMM_LESION_GROUP,
                           phases=GLMHMM_PHASES,
                           require_criteria: bool = GLMHMM_REQUIRE_CRITERIA,
                           verbose: bool = True) -> pd.DataFrame:
    """Apply the fixed inclusion criteria for the GLM-HMM input dataset.

    See the module-level notes above for the rationale. By default this
    reproduces: naive (control + pre-lesion), both phases unified, and
    meets_criteria == True. Returns the filtered DataFrame (a copy).
    """
    out = df

    # 1. lesion status
    if lesion_group == "naive":
        out = out[out["lesioned"].isna()]            # control + pre-lesion
    elif lesion_group == "post":
        out = out[out["lesioned"] == 1.0]            # post-lesion only
    elif lesion_group == "all":
        pass                                          # no lesion filtering
    else:
        raise ValueError(f"unknown lesion_group: {lesion_group!r} "
                         "(use 'naive', 'post' or 'all')")

    # 2. phase (None keeps phases 3 and 8 unified)
    if phases is not None:
        out = out[out["phase"].isin(list(phases))]

    # 3. performance criterion
    if require_criteria:
        out = out[out["meets_criteria"] == True]

    out = out.copy()
    if verbose:
        n_ses = out.groupby(["animal", "session_file"]).ngroups
        print(f"[glm-hmm select] lesion={lesion_group}, "
              f"phases={'all' if phases is None else tuple(phases)}, "
              f"meets_criteria={require_criteria}: kept {len(out)} trials, "
              f"{out['animal'].nunique()} animals, {n_ses} sessions.")
    return out


def build_glmhmm_inputs(df: pd.DataFrame, n_lags: int = N_LAGS,
                        parametrization: str = PARAMETRIZATION,
                        min_trials: int = 20):
    """Build aligned lists of choices / inputs / masks / tags over all sessions.

    Sessions with fewer than `min_trials` observed trials are skipped.
    """
    choices, inputs, masks, tags = [], [], [], []
    for (animal, ses_file), df_ses in df.groupby(["animal", "session_file"], sort=False):
        ch, X, m = build_session_arrays(df_ses, n_lags=n_lags,
                                        parametrization=parametrization)
        if int(m.sum()) < min_trials:
            continue
        choices.append(ch)
        inputs.append(X)
        masks.append(m)
        tags.append((animal, ses_file))
    return choices, inputs, masks, tags


# ===========================================================================
# 2. Fitting  (multiple EM restarts, random init)
# ===========================================================================

def _new_hmm(K, M):
    return ssm.HMM(K, OBS_DIM, M, observations="input_driven_obs",
                   observation_kwargs=dict(C=N_CATEGORIES, prior_sigma=PRIOR_SIGMA),
                   transitions="sticky",
                   transition_kwargs=dict(alpha=TRANSITION_ALPHA, kappa=TRANSITION_KAPPA))


def _fit_once(K, M, choices, inputs, masks, seed, n_iters, tolerance):
    np.random.seed(seed)
    model = _new_hmm(K, M)
    ll = model.fit(choices, inputs=inputs, masks=masks, method="em",
                   num_iters=n_iters, tolerance=tolerance, verbose=0)
    return model, float(ll[-1])


def fit_glmhmm(choices, inputs, masks, K, M=None,
               n_restarts: int = N_RESTARTS, n_iters: int = N_ITERS,
               tolerance: float = TOLERANCE, seed: int = 0, n_jobs: int = CV_N_JOBS):
    """Fit a K-state GLM-HMM with `n_restarts` random initialisations.

    Keeps the highest-training-LL restart, then relabels states by descending
    occupancy for reproducible interpretation.
    """
    if M is None:
        M = inputs[0].shape[1]
    seeds = np.random.SeedSequence(seed).generate_state(n_restarts)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_once)(K, M, choices, inputs, masks, int(s), n_iters, tolerance)
        for s in seeds)
    model, _ = max(results, key=lambda mr: mr[1])
    _sort_states_by_occupancy(model, choices, inputs, masks)
    return model


def _sort_states_by_occupancy(model, choices, inputs, masks):
    """Permute states so state 0 is the most occupied (label-switching fix)."""
    if model.K == 1:
        return
    occ = np.zeros(model.K)
    for ch, X, m in zip(choices, inputs, masks):
        occ += model.expected_states(ch, input=X, mask=m)[0].sum(axis=0)
    model.permute(list(np.argsort(occ)[::-1]))


# ===========================================================================
# 3. Hierarchical initialisation  (Ashwood recipe: GLM -> global -> individual)
# ===========================================================================

def fit_glm(choices, inputs, masks, n_restarts: int = 5, n_iters: int = N_ITERS,
            tolerance: float = TOLERANCE, seed: int = 0, n_jobs: int = CV_N_JOBS):
    """Fit the 1-state GLM baseline. Its weights initialise the global GLM-HMM.

    Returns (model, glm_weights) where glm_weights has shape (M,).
    """
    model = fit_glmhmm(choices, inputs, masks, K=1, n_restarts=n_restarts,
                       n_iters=n_iters, tolerance=tolerance, seed=seed, n_jobs=n_jobs)
    glm_weights = np.asarray(model.observations.Wk)[0, 0, :].copy()
    return model, glm_weights


def _sticky_log_Ps(K, self_p=STICKY_SELF):
    off = (1.0 - self_p) / max(K - 1, 1)
    P = np.full((K, K), off)
    np.fill_diagonal(P, self_p)
    return np.log(P)


def _fit_global_once(K, M, choices, inputs, masks, glm_weights, seed,
                     init_noise, self_p, n_iters, tolerance):
    """One global-GLM-HMM fit, initialised from GLM weights + per-state noise."""
    rng = np.random.default_rng(seed)
    model = _new_hmm(K, M)
    Wk = np.tile(glm_weights, (K, 1)).reshape(K, 1, M)
    Wk = Wk + init_noise * rng.standard_normal((K, 1, M))
    model.observations.Wk = Wk
    if K > 1:
        model.transitions.log_Ps = _sticky_log_Ps(K, self_p)
    ll = model.fit(choices, inputs=inputs, masks=masks, method="em",
                   num_iters=n_iters, tolerance=tolerance, verbose=0,
                   initialize=False)
    return model, float(ll[-1])


def fit_global_glmhmm(choices, inputs, masks, K, glm_weights,
                      n_restarts: int = N_RESTARTS, init_noise: float = INIT_NOISE,
                      self_p: float = STICKY_SELF, n_iters: int = N_ITERS,
                      tolerance: float = TOLERANCE, seed: int = 0,
                      n_jobs: int = CV_N_JOBS):
    """Fit the global GLM-HMM (all sessions), initialised from the GLM fit.

    Each state's weights start at the GLM weights plus Gaussian noise; the
    transition matrix starts sticky. `n_restarts` differ only in the init
    noise draw. Returns the best (highest train LL) model, states sorted.
    """
    M = glm_weights.shape[0]
    seeds = np.random.SeedSequence(seed).generate_state(n_restarts)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_global_once)(K, M, choices, inputs, masks, glm_weights,
                                  int(s), init_noise, self_p, n_iters, tolerance)
        for s in seeds)
    model, _ = max(results, key=lambda mr: mr[1])
    _sort_states_by_occupancy(model, choices, inputs, masks)
    return model


def fit_individual_glmhmm(choices, inputs, masks, global_model,
                          n_restarts: int = 3, init_noise: float = 0.05,
                          n_iters: int = N_ITERS, tolerance: float = TOLERANCE,
                          seed: int = 0, n_jobs: int = CV_N_JOBS):
    """Fit one animal's GLM-HMM, initialised from the fitted global model.

    Copies the global weights and transitions (with small noise across
    restarts) and refits on this animal's sessions only.
    """
    K = global_model.K
    M = np.asarray(global_model.observations.Wk).shape[2]
    Wk0 = np.asarray(global_model.observations.Wk).copy()
    logPs0 = np.asarray(global_model.transitions.log_Ps).copy()

    def _one(seed_i):
        rng = np.random.default_rng(seed_i)
        model = _new_hmm(K, M)
        model.observations.Wk = Wk0 + init_noise * rng.standard_normal((K, 1, M))
        if K > 1:
            model.transitions.log_Ps = logPs0.copy()
        ll = model.fit(choices, inputs=inputs, masks=masks, method="em",
                       num_iters=n_iters, tolerance=tolerance, verbose=0,
                       initialize=False)
        return model, float(ll[-1])

    seeds = np.random.SeedSequence(seed).generate_state(n_restarts)
    results = Parallel(n_jobs=n_jobs)(delayed(_one)(int(s)) for s in seeds)
    model, _ = max(results, key=lambda mr: mr[1])
    _sort_states_by_occupancy(model, choices, inputs, masks)
    return model


def fit_individual_glmhmms(df, global_model, n_lags: int = N_LAGS,
                           parametrization: str = PARAMETRIZATION, **kwargs):
    """Fit a per-animal GLM-HMM for every animal, all initialised from global.

    Returns {animal: fitted_model}.
    """
    out = {}
    for animal, df_a in df.groupby("animal", sort=False):
        ch, inp, mk, _ = build_glmhmm_inputs(df_a, n_lags=n_lags,
                                              parametrization=parametrization)
        if not ch:
            continue
        out[animal] = fit_individual_glmhmm(ch, inp, mk, global_model, **kwargs)
        print(f"[individual] {animal}: fitted ({len(ch)} sessions)")
    return out


# ===========================================================================
# 4. Lapse-model baseline  (for model comparison)
# ===========================================================================

def _stack_observed(choices, inputs, masks):
    """Concatenate observed (non-miss) trials across sessions: (X, y)."""
    X = np.vstack([inp[m[:, 0]] for inp, m in zip(inputs, masks)])
    y = np.concatenate([ch[m[:, 0], 0] for ch, m in zip(choices, masks)])
    return X, y.astype(float)


def _lapse_negloglike(theta, X, y, n_lapses):
    """NLL of the lapse model. p = gR + (1 - gL - gR) * sigmoid(X w)."""
    M = X.shape[1]
    w = theta[:M]
    if n_lapses == 1:
        gL = gR = theta[M]
    else:
        gL, gR = theta[M], theta[M + 1]
    p = gR + (1.0 - gL - gR) * expit(X @ w)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_lapse_model(choices, inputs, masks, n_lapses: int = 2, seed: int = 0):
    """Fit a GLM with 1 or 2 lapse parameters (no latent state).

    Returns dict with keys: weights (M,), lapses (gL, gR), loglike, n_params.
    """
    X, y = _stack_observed(choices, inputs, masks)
    M = X.shape[1]
    rng = np.random.default_rng(seed)
    theta0 = np.concatenate([0.1 * rng.standard_normal(M),
                             np.full(1 if n_lapses == 1 else 2, 0.05)])
    bounds = [(None, None)] * M + [(0.0, 0.5)] * (1 if n_lapses == 1 else 2)
    res = minimize(_lapse_negloglike, theta0, args=(X, y, n_lapses),
                   method="L-BFGS-B", bounds=bounds)
    w = res.x[:M]
    gL = res.x[M]
    gR = res.x[M] if n_lapses == 1 else res.x[M + 1]
    return {"weights": w, "lapses": (gL, gR), "loglike": -res.fun,
            "n_params": M + n_lapses}


def lapse_log_likelihood(params, choices, inputs, masks):
    """Total log-likelihood of a fitted lapse model on (possibly held-out) data."""
    X, y = _stack_observed(choices, inputs, masks)
    w = params["weights"]
    gL, gR = params["lapses"]
    p = np.clip(gR + (1.0 - gL - gR) * expit(X @ w), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ===========================================================================
# 5. Cross-validation over K  +  model comparison
# ===========================================================================

def _session_folds(animals, n_folds, seed):
    """Assign each session to a CV fold, stratified by animal.

    Each animal's sessions are independently shuffled and split across the
    n_folds, so every animal contributes to every fold (matching Ashwood's
    create_train_test_sessions per-animal fold assignment). Animals with fewer
    sessions than n_folds simply appear in a subset of folds.

    Parameters
    ----------
    animals : sequence of length n_sessions; the animal label for each session,
              in the same order as choices/inputs/masks (i.e. [t[0] for t in tags]).

    Returns
    -------
    list of n_folds arrays of session indices.
    """
    animals = np.asarray(animals, dtype=object)
    rng = np.random.default_rng(seed)
    fold_assign = np.full(len(animals), -1, dtype=int)
    for a in pd.unique(animals):
        idx = rng.permutation(np.where(animals == a)[0])
        for f, chunk in enumerate(np.array_split(idx, n_folds)):
            fold_assign[chunk] = f
    return [np.where(fold_assign == f)[0] for f in range(n_folds)]


def _split(lst, idx):
    return [lst[i] for i in idx]


def cross_validate_glmhmm(df: pd.DataFrame, K_range=K_RANGE, n_lags: int = N_LAGS,
                          parametrization: str = PARAMETRIZATION,
                          n_folds: int = CV_N_FOLDS, n_restarts: int = N_RESTARTS,
                          n_iters: int = N_ITERS, tolerance: float = TOLERANCE,
                          seed: int = 0, n_jobs: int = CV_N_JOBS,
                          hierarchical: bool = True) -> pd.DataFrame:
    """Session-level CV of held-out per-trial log-likelihood across K.

    With hierarchical=True, each training fit follows the Ashwood recipe
    (GLM -> global GLM-HMM init). Returns tidy DataFrame [K, fold, test_ll_per_trial].
    """
    choices, inputs, masks, tags = build_glmhmm_inputs(df, n_lags=n_lags,
                                                       parametrization=parametrization)
    n_ses = len(choices)
    folds = _session_folds([t[0] for t in tags], n_folds, seed)   # stratified by animal
    rows = []

    for f, te in enumerate(folds):
        te = list(te)
        tr = [i for i in range(n_ses) if i not in set(te)]
        tr_ch, tr_in, tr_mk = _split(choices, tr), _split(inputs, tr), _split(masks, tr)
        te_ch, te_in, te_mk = _split(choices, te), _split(inputs, te), _split(masks, te)
        n_te = max(int(sum(int(m.sum()) for m in te_mk)), 1)

        glm_w = None
        if hierarchical:
            _, glm_w = fit_glm(tr_ch, tr_in, tr_mk, seed=seed + f, n_jobs=n_jobs)

        for K in K_range:
            if hierarchical and K > 1:
                model = fit_global_glmhmm(tr_ch, tr_in, tr_mk, K, glm_w,
                                          n_restarts=n_restarts, n_iters=n_iters,
                                          tolerance=tolerance, seed=seed + f, n_jobs=n_jobs)
            else:
                model = fit_glmhmm(tr_ch, tr_in, tr_mk, K, n_restarts=n_restarts,
                                   n_iters=n_iters, tolerance=tolerance,
                                   seed=seed + f, n_jobs=n_jobs)
            test_ll = model.log_likelihood(te_ch, inputs=te_in, masks=te_mk)
            rows.append({"K": K, "fold": f, "test_ll_per_trial": test_ll / n_te})
            print(f"[CV] fold {f}  K={K}  held-out LL/trial = {test_ll / n_te:.4f}")

    return pd.DataFrame(rows)


def model_comparison(df: pd.DataFrame, K_states: int, n_lags: int = N_LAGS,
                     parametrization: str = PARAMETRIZATION, n_folds: int = CV_N_FOLDS,
                     n_restarts: int = N_RESTARTS, n_iters: int = N_ITERS,
                     tolerance: float = TOLERANCE, seed: int = 0,
                     n_jobs: int = CV_N_JOBS) -> pd.DataFrame:
    """Held-out per-trial LL for GLM, lapse model, and GLM-HMM(K_states).

    Mirrors Ashwood's model comparison. Returns tidy DataFrame
    [model, fold, test_ll_per_trial].
    """
    choices, inputs, masks, tags = build_glmhmm_inputs(df, n_lags=n_lags,
                                                       parametrization=parametrization)
    n_ses = len(choices)
    folds = _session_folds([t[0] for t in tags], n_folds, seed)   # stratified by animal
    rows = []

    for f, te in enumerate(folds):
        te = list(te)
        tr = [i for i in range(n_ses) if i not in set(te)]
        tr_ch, tr_in, tr_mk = _split(choices, tr), _split(inputs, tr), _split(masks, tr)
        te_ch, te_in, te_mk = _split(choices, te), _split(inputs, te), _split(masks, te)
        n_te = max(int(sum(int(m.sum()) for m in te_mk)), 1)

        glm, glm_w = fit_glm(tr_ch, tr_in, tr_mk, seed=seed + f, n_jobs=n_jobs)
        lapse = fit_lapse_model(tr_ch, tr_in, tr_mk, seed=seed + f)
        ghmm = fit_global_glmhmm(tr_ch, tr_in, tr_mk, K_states, glm_w,
                                 n_restarts=n_restarts, n_iters=n_iters,
                                 tolerance=tolerance, seed=seed + f, n_jobs=n_jobs)

        rows += [
            {"model": "GLM", "fold": f,
             "test_ll_per_trial": glm.log_likelihood(te_ch, inputs=te_in, masks=te_mk) / n_te},
            {"model": "lapse", "fold": f,
             "test_ll_per_trial": lapse_log_likelihood(lapse, te_ch, te_in, te_mk) / n_te},
            {"model": f"GLM-HMM (K={K_states})", "fold": f,
             "test_ll_per_trial": ghmm.log_likelihood(te_ch, inputs=te_in, masks=te_mk) / n_te},
        ]
        print(f"[compare] fold {f} done")

    return pd.DataFrame(rows)


# ===========================================================================
# 6. Interpretation helpers
# ===========================================================================

def glmhmm_weights(model, n_lags: int = N_LAGS,
                   parametrization: str = PARAMETRIZATION) -> pd.DataFrame:
    """Per-state GLM weights as log-odds of choosing RIGHT, as a tidy DataFrame.

    ssm's InputDrivenObservations stores Wk as the logits of category 0 (LEFT),
    with category 1 (RIGHT) as the zero-weight reference (see calculate_logits:
    a zero row is stacked for the last category). So model.observations.Wk is
    the log-odds of choosing LEFT; we negate it here to report the more
    intuitive log-odds of choosing RIGHT. With this convention a POSITIVE
    prev_choice weight means perseveration (repeat) and a POSITIVE wsls weight
    means win-stay-lose-shift. NOTE: fit_glm returns the raw (un-negated) Wk
    for initialisation, which must stay in ssm's native convention.
    """
    W = -np.asarray(model.observations.Wk)[:, 0, :]   # -> log-odds of choosing right
    out = pd.DataFrame(W, columns=regressor_names(n_lags, parametrization))
    out.insert(0, "state", np.arange(model.K))
    return out


def glmhmm_posteriors(model, choices, inputs, masks):
    """Per-session posterior state probabilities, each (T_i, K)."""
    return [model.expected_states(ch, input=X, mask=m)[0]
            for ch, X, m in zip(choices, inputs, masks)]


def glmhmm_transition_matrix(model) -> np.ndarray:
    """K x K Markov transition matrix between latent states."""
    return model.transitions.transition_matrix