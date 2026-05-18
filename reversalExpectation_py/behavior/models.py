"""
models.py
=========
Bayesian belief models for two-armed bandit reversal task.
Port of funbelief.m / funbelief_CK.m (AC Kwan / H Atilgan lab).

Reference: Murphy et al. 2024 eLife, Figure 3, Methods pp. 21-22.

Models
------
belief      (2 params): H, beta
belief_CK   (4 params): H, beta, alpha_k, beta_k

Encoding convention (same as the rest of the pipeline)
-------------------------------------------------------
choice  : -1 = left,  1 = right,  NaN = miss
reward  :  1 = rewarded,  0 = not rewarded,  NaN = miss
rule    :  integer index mapping into REWARD_PROBS
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid

from preprocessing.presentation_codes import REWARD_PROBS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_reward_probs(n_rules: int):
    """Return {rule_idx: (p_left, p_right)} for this task variant."""
    return REWARD_PROBS.get(n_rules, REWARD_PROBS[2])


def _transition(b: float, H: float) -> float:
    """Chapman-Kolmogorov step for a two-state chain with hazard rate H."""
    return b * (1.0 - H) + (1.0 - b) * H


def _belief_update(b_hat: float, choice: float, reward: float,
                   p_high: float = 0.70, p_low: float = 0.10) -> float:
    """
    Bayes update of belief b_hat = P(high-left state) after observing
    (choice, reward).  Returns unchanged b_hat on miss.
    """
    if np.isnan(choice) or np.isnan(reward):
        return b_hat

    if choice == -1:                     # chose left
        L1 = p_high if reward else (1.0 - p_high)   # high-left state
        L2 = p_low  if reward else (1.0 - p_low)    # high-right state
    else:                                # chose right
        L1 = p_low  if reward else (1.0 - p_low)
        L2 = p_high if reward else (1.0 - p_high)

    denom = b_hat * L1 + (1.0 - b_hat) * L2
    if denom < 1e-300:
        return b_hat
    return b_hat * L1 / denom


def _q_diff(b_hat: float, p_high: float = 0.70, p_low: float = 0.10) -> float:
    """Expected-reward difference: Q(left) - Q(right) given belief b_hat."""
    q_left  = p_high * b_hat + p_low  * (1.0 - b_hat)
    q_right = p_low  * b_hat + p_high * (1.0 - b_hat)
    return q_left - q_right


def _reward_probs_for_n_rules(n_rules: int):
    """Return (p_high, p_low) from the first entry in REWARD_PROBS[n_rules]."""
    probs = REWARD_PROBS.get(n_rules, REWARD_PROBS[2])
    # p_high is the larger of the two reward probs in rule 1
    lp, rp = probs[1]
    return max(lp, rp), min(lp, rp)


# ---------------------------------------------------------------------------
# Belief model — negative log-likelihood
# ---------------------------------------------------------------------------

def belief_negloglike(params, c, r, n_rules: int = 2) -> float:
    """
    Negative log-likelihood for the belief model.

    Parameters
    ----------
    params   : [H, beta]
    c        : (n_trials,) choices  (-1/1/NaN)
    r        : (n_trials,) rewards  (0/1/NaN)
    n_rules  : number of rules / hidden states (default 2)
    """
    H, beta = params
    if H <= 0 or H >= 1 or beta <= 0:
        return 1e9

    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    b   = 0.5
    nll = 0.0

    for t in range(len(c)):
        b_hat = _transition(b, H)
        qdiff = _q_diff(b_hat, p_high, p_low)
        p_l   = expit(beta * qdiff)

        ct = c[t]
        if not np.isnan(ct):
            p_chosen = p_l if ct == -1 else (1.0 - p_l)
            nll -= np.log(np.clip(p_chosen, 1e-15, 1.0 - 1e-15))

        b = _belief_update(b_hat, ct, r[t], p_high, p_low)

    return nll


# ---------------------------------------------------------------------------
# Belief-CK model — negative log-likelihood
# ---------------------------------------------------------------------------

def belief_ck_negloglike(params, c, r, n_rules: int = 2) -> float:
    """
    Negative log-likelihood for the belief + choice-kernel model.

    Parameters
    ----------
    params   : [H, beta, alpha_k, beta_k]
    c        : (n_trials,) choices
    r        : (n_trials,) rewards
    n_rules  : number of rules (default 2)
    """
    H, beta, alpha_k, beta_k = params
    if H <= 0 or H >= 1 or beta <= 0 or alpha_k < 0 or alpha_k > 1 or beta_k <= 0:
        return 1e9

    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    b   = 0.5
    ck  = np.zeros(2)   # [CK_left, CK_right]
    nll = 0.0

    for t in range(len(c)):
        b_hat   = _transition(b, H)
        qdiff   = _q_diff(b_hat, p_high, p_low)
        ck_diff = ck[0] - ck[1]
        logit   = beta * qdiff + beta_k * ck_diff
        p_l     = expit(logit)

        ct = c[t]
        if not np.isnan(ct):
            p_chosen = p_l if ct == -1 else (1.0 - p_l)
            nll -= np.log(np.clip(p_chosen, 1e-15, 1.0 - 1e-15))

        b = _belief_update(b_hat, ct, r[t], p_high, p_low)

        if not np.isnan(ct):
            if ct == -1:                       # chose left
                ck[0] = ck[0] + alpha_k * (1.0 - ck[0])
                ck[1] = ck[1] * (1.0 - alpha_k)
            else:                              # chose right
                ck[1] = ck[1] + alpha_k * (1.0 - ck[1])
                ck[0] = ck[0] * (1.0 - alpha_k)

    return nll


# ---------------------------------------------------------------------------
# Model fitting (L-BFGS-B with random restarts)
# ---------------------------------------------------------------------------

def fit_belief(c, r, n_rules: int = 2, n_restarts: int = 5,
               rng=None) -> dict:
    """
    Fit the belief model to observed choices and rewards.

    Returns
    -------
    dict with keys: model, fitpar, negloglike, bic, nlike
        fitpar = [H, beta]
    """
    if rng is None:
        rng = np.random.default_rng()
    c, r = np.asarray(c, float), np.asarray(r, float)
    n_obs = int(np.sum(~np.isnan(c)))

    best_nll, best_par = np.inf, None
    for _ in range(n_restarts):
        x0  = [rng.uniform(0.01, 0.49), rng.uniform(0.5, 15.0)]
        res = minimize(
            belief_negloglike, x0, args=(c, r, n_rules),
            method="L-BFGS-B",
            bounds=[(1e-6, 1.0 - 1e-6), (1e-3, 100.0)],
        )
        if res.fun < best_nll:
            best_nll, best_par = res.fun, res.x

    bic = 2.0 * best_nll + 2.0 * np.log(max(n_obs, 1))
    return {
        "model":      "belief",
        "fitpar":     best_par,
        "negloglike": best_nll,
        "bic":        bic,
        "nlike":      np.exp(-best_nll / max(n_obs, 1)),
    }


def fit_belief_ck(c, r, n_rules: int = 2, n_restarts: int = 5,
                  rng=None) -> dict:
    """
    Fit the belief-CK model to observed choices and rewards.

    Returns
    -------
    dict with keys: model, fitpar, negloglike, bic, nlike
        fitpar = [H, beta, alpha_k, beta_k]
    """
    if rng is None:
        rng = np.random.default_rng()
    c, r = np.asarray(c, float), np.asarray(r, float)
    n_obs = int(np.sum(~np.isnan(c)))

    best_nll, best_par = np.inf, None
    for _ in range(n_restarts):
        x0 = [
            rng.uniform(0.01, 0.49),
            rng.uniform(0.5,  15.0),
            rng.uniform(0.01, 0.49),
            rng.uniform(0.5,  10.0),
        ]
        res = minimize(
            belief_ck_negloglike, x0, args=(c, r, n_rules),
            method="L-BFGS-B",
            bounds=[
                (1e-6, 1.0 - 1e-6),   # H
                (1e-3, 100.0),         # beta
                (1e-6, 1.0 - 1e-6),   # alpha_k
                (1e-3, 100.0),         # beta_k
            ],
        )
        if res.fun < best_nll:
            best_nll, best_par = res.fun, res.x

    bic = 2.0 * best_nll + 4.0 * np.log(max(n_obs, 1))
    return {
        "model":      "belief_ck",
        "fitpar":     best_par,
        "negloglike": best_nll,
        "bic":        bic,
        "nlike":      np.exp(-best_nll / max(n_obs, 1)),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_belief(c_real, r_real, rule, H: float, beta: float,
                    n_sims: int = 100, rng=None) -> dict:
    """
    Simulate choices under the belief model.

    On each simulation the model uses:
    - the same miss-trial mask as the real session
    - the true block rules to draw synthetic rewards

    Parameters
    ----------
    c_real  : (n_trials,) real choices — only used for miss masking
    r_real  : (n_trials,) real rewards — not used in simulation
    rule    : (n_trials,) true rule index per trial
    H, beta : model parameters
    n_sims  : number of independent repeats
    rng     : np.random.Generator

    Returns
    -------
    dict with keys:
        c_sim   : (n_trials, n_sims)  simulated choices (-1 / 1)
        belief  : (n_trials, n_sims)  P(high-left state) after transition, before update
        p_left  : (n_trials, n_sims)  P(choose left) on each trial
    """
    if rng is None:
        rng = np.random.default_rng()

    c_real = np.asarray(c_real, float)
    rule   = np.asarray(rule,   float)
    n      = len(c_real)
    miss   = np.isnan(c_real)

    # Determine p_high/p_low from the rules present
    unique_rules = rule[~np.isnan(rule)]
    n_rules = 2 if unique_rules.max() <= 2 else int(unique_rules.max())
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    prob_map = _get_reward_probs(n_rules)

    c_sim  = np.full((n, n_sims), np.nan)
    belief = np.full((n, n_sims), np.nan)
    p_left = np.full((n, n_sims), np.nan)

    for s in range(n_sims):
        b = 0.5
        for t in range(n):
            b_hat = _transition(b, H)
            belief[t, s] = b_hat

            if miss[t]:
                b = b_hat   # no observation → just propagate
                continue

            qdiff      = _q_diff(b_hat, p_high, p_low)
            pl         = expit(beta * qdiff)
            p_left[t, s] = pl

            ct = -1.0 if rng.random() < pl else 1.0
            c_sim[t, s] = ct

            # Synthetic reward from true block rule
            r_idx = rule[t]
            if not np.isnan(r_idx):
                lp, rp = prob_map.get(int(r_idx), (0.5, 0.5))
                prew   = lp if ct == -1 else rp
                rt     = 1.0 if rng.random() < prew else 0.0
            else:
                rt = np.nan

            b = _belief_update(b_hat, ct, rt, p_high, p_low)

    return {"c_sim": c_sim, "belief": belief, "p_left": p_left}


def simulate_belief_ck(c_real, r_real, rule,
                       H: float, beta: float,
                       alpha_k: float, beta_k: float,
                       n_sims: int = 100, rng=None) -> dict:
    """
    Simulate choices under the belief + choice-kernel model.

    Parameters
    ----------
    c_real, r_real, rule : see simulate_belief
    H, beta, alpha_k, beta_k : model parameters
    n_sims, rng : see simulate_belief

    Returns
    -------
    dict with keys: c_sim, belief, p_left, ck_diff
        ck_diff : (n_trials, n_sims)  CK_left - CK_right at decision time
    """
    if rng is None:
        rng = np.random.default_rng()

    c_real = np.asarray(c_real, float)
    rule   = np.asarray(rule,   float)
    n      = len(c_real)
    miss   = np.isnan(c_real)

    unique_rules = rule[~np.isnan(rule)]
    n_rules = 2 if (len(unique_rules) == 0 or unique_rules.max() <= 2) else int(unique_rules.max())
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    prob_map = _get_reward_probs(n_rules)

    c_sim   = np.full((n, n_sims), np.nan)
    belief  = np.full((n, n_sims), np.nan)
    p_left  = np.full((n, n_sims), np.nan)
    ck_diff = np.full((n, n_sims), np.nan)

    for s in range(n_sims):
        b  = 0.5
        ck = np.zeros(2)   # [CK_left, CK_right]

        for t in range(n):
            b_hat = _transition(b, H)
            belief[t, s] = b_hat

            if miss[t]:
                b = b_hat
                continue

            qdiff       = _q_diff(b_hat, p_high, p_low)
            ck_d        = ck[0] - ck[1]
            logit       = beta * qdiff + beta_k * ck_d
            pl          = expit(logit)

            p_left[t, s]  = pl
            ck_diff[t, s] = ck_d

            ct = -1.0 if rng.random() < pl else 1.0
            c_sim[t, s] = ct

            r_idx = rule[t]
            if not np.isnan(r_idx):
                lp, rp = prob_map.get(int(r_idx), (0.5, 0.5))
                prew   = lp if ct == -1 else rp
                rt     = 1.0 if rng.random() < prew else 0.0
            else:
                rt = np.nan

            b = _belief_update(b_hat, ct, rt, p_high, p_low)

            if ct == -1:
                ck[0] = ck[0] + alpha_k * (1.0 - ck[0])
                ck[1] = ck[1] * (1.0 - alpha_k)
            else:
                ck[1] = ck[1] + alpha_k * (1.0 - ck[1])
                ck[0] = ck[0] * (1.0 - alpha_k)

    return {"c_sim": c_sim, "belief": belief, "p_left": p_left, "ck_diff": ck_diff}
