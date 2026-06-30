"""
belief_vhr.py
=============
Belief model with a variable hazard rate, H(tau), where tau = trials since the
criterion was reached (the experimenter's block clock).

For now this module holds the empirical-hazard diagnostic used to choose the
functional form of H(tau): the discrete hazard of the block switch as a
function of trials since criterion, computed directly from the L_Random
distribution. The fitted variable-hazard model will be added here once the
H(tau) form is fixed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid

# Reuse the belief math from the constant-hazard model (single source of truth).
from behavior.beh_models.bayesian_models import (
    _transition,
    _belief_update,
    _q_diff,
    _reward_probs_for_n_rules,
    _get_reward_probs,
    _run_bads,
    _HAVE_BADS,
)


def empirical_hazard(df: pd.DataFrame, max_lrandom: int = 30):
    """
    Discrete switch hazard as a function of tau = trials since criterion.

    H(tau) = P(switch at exactly tau | reached tau)
           = #{blocks with L_Random == tau} / #{blocks with L_Random >= tau}

    A block "survives" to tau if its L_Random >= tau (it had not switched yet),
    and the "event" at tau is the switch (L_Random == tau). One L_Random value
    per block is used (criterion reached => non-NaN). Blocks with
    L_Random > max_lrandom are excluded so the truncation boundary is clean
    (the hazard then reaches 1 exactly at tau = max_lrandom).

    Parameters
    ----------
    df : trial table (needs animal, session_file, block_idx, block_trial_random_added).
    max_lrandom : drop blocks whose L_Random exceeds this (default 30).

    Returns
    -------
    haz : DataFrame with columns tau, n_event, n_at_risk, hazard, se
        se = binomial standard error sqrt(h (1 - h) / n_at_risk).
    p_hat : float
        Pooled constant (geometric) hazard = N / sum(L_Random + 1)
        = 1 / (mean(L_Random) + 1).
    """
    lr = (df.groupby(["animal", "session_file", "block_idx"], sort=False)
            ["block_trial_random_added"].first().dropna().astype(int))
    lr = lr[lr <= max_lrandom]
    n_blocks = len(lr)
    if n_blocks == 0:
        raise ValueError("No blocks with a valid L_Random <= max_lrandom.")

    taus = np.arange(0, int(lr.max()) + 1)
    n_event = np.array([(lr == t).sum() for t in taus], dtype=float)   # switch at tau
    n_risk = np.array([(lr >= t).sum() for t in taus], dtype=float)    # at risk at tau
    hazard = n_event / n_risk
    se = np.sqrt(hazard * (1.0 - hazard) / n_risk)

    p_hat = n_blocks / float((lr + 1).sum())   # geometric MLE = 1 / (mean(L_Random) + 1)

    haz = pd.DataFrame({
        "tau":       taus,
        "n_event":   n_event.astype(int),
        "n_at_risk": n_risk.astype(int),
        "hazard":    hazard,
        "se":        se,
    })
    return haz, p_hat


def plot_empirical_hazard(df: pd.DataFrame, max_lrandom: int = 30,
                          min_at_risk: int = 20, output_dir: str = "figs",
                          save: bool = True):
    """
    Plot the empirical switch hazard vs trials since criterion.

    Shows the discrete hazard H(tau) with binomial error bars and a dashed line
    at the pooled constant (geometric) hazard for reference. The empirical curve
    sits on the constant line while the task is memoryless and rises toward 1 as
    tau approaches the truncation boundary. Positions with fewer than
    `min_at_risk` surviving blocks are drawn faded (noisy tail). Saves
    figs/hazard_vs_tau.png.
    """
    haz, p_hat = empirical_hazard(df, max_lrandom=max_lrandom)
    solid = haz["n_at_risk"].to_numpy() >= min_at_risk

    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=130)
    ax.errorbar(haz.loc[solid, "tau"], haz.loc[solid, "hazard"],
                yerr=haz.loc[solid, "se"], fmt="o-", color="#2C5F2D", lw=2, ms=4,
                capsize=2, label="empirical hazard  H(τ)")
    if (~solid).any():
        ax.plot(haz.loc[~solid, "tau"], haz.loc[~solid, "hazard"], "o",
                color="#9BB89A", ms=4, alpha=0.7,
                label=f"tail (<{min_at_risk} blocks at risk)")
    ax.axhline(p_hat, ls="--", color="#C9472B", lw=1.6,
               label=f"constant hazard (geometric) = {p_hat:.3f}")
    ax.set_xlabel("τ  =  trials since criterion", fontsize=13)
    ax.set_ylabel("Hazard  H(τ) = P(switch at τ | reached τ)", fontsize=13)
    ax.set_xlim(-1, max_lrandom + 1)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()

    if save and output_dir:
        out = Path(output_dir) / "hazard_vs_tau.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    return fig, haz, p_hat


# ===========================================================================
# Belief model with a variable hazard rate  (belief_vhr)
# ===========================================================================
#
# Same belief filter as the constant-hazard model, but the per-trial hazard
# grows with tau = trials since criterion (the task's block clock):
#
#     H(tau) = sigmoid(a + b * tau)
#
# - tau is provided per trial by prepare_vhr (0 throughout the criterion phase,
#   then 0, 1, 2, ... across the L_Random phase). The decision at trial t is made
#   on a belief eroded by H(tau[t]); under Murphy's decide-then-update ordering
#   the end-of-trial transition uses tau[t+1] (see belief_vhr_trial_loglikes).
#   Keeping tau = 0 before
#   criterion is causal: the switch cannot happen until criterion is reached,
#   so the model is not given any look-ahead.
# - The sigmoid keeps H in (0, 1) without clipping. b >= 0 makes the hazard
#   non-decreasing in tau (the anticipation hypothesis). b = 0 recovers the
#   constant-hazard belief model exactly, with H = sigmoid(a); so belief_vhr
#   nests `belief` and the BIC / CV penalty for the extra parameter is the test
#   of whether anticipation helps.
#
# Parameters: [a, b, beta]
#   a    : hazard logit intercept   (baseline hazard = sigmoid(a))
#   b    : hazard logit slope in tau (anticipation rate; >= 0)
#   beta : softmax inverse temperature on the value difference

_VHR_PARAM_LABELS = ["a", "b", "beta"]


def prepare_vhr(df_ses: pd.DataFrame):
    """
    Model inputs for belief_vhr: (choice, reward, n_rules, tau).

    tau = trials since criterion, per trial:
        criterion phase  -> 0   (switch cannot happen yet; causal, no look-ahead)
        L_Random phase   -> 0, 1, 2, ... (trials elapsed since criterion)
    Computed from the within-block trial index and block_trial_to_crit. Blocks
    with no criterion (NaN block_trial_to_crit, e.g. never-crit) get tau = 0
    throughout.
    """
    c = df_ses["choice"].values.astype(float)
    r = df_ses["rewarded"].values.astype(float)
    n_rules = int(df_ses["n_rules"].iloc[0])

    t_block = df_ses.groupby("block_idx", sort=False).cumcount().to_numpy(dtype=float)
    ttc = df_ses["block_trial_to_crit"].to_numpy(dtype=float)
    tau = t_block - ttc                       # trials since criterion (negative pre-criterion)
    tau = np.where(np.isnan(ttc), 0.0, tau)   # no criterion -> baseline
    tau = np.clip(tau, 0.0, None)             # criterion phase -> 0
    return (c, r, n_rules, tau)


def belief_vhr_trial_loglikes(params, c, r, n_rules: int, tau) -> np.ndarray:
    """
    Per-trial log-likelihood under the variable-hazard belief model.

    Murphy trial order: the choice at trial t is made on the belief carried from
    the end of trial t-1; the end-of-trial update is the transition (Eq. 7) then
    the Bayes update (Eq. 8). The transition that produces the belief used at
    trial t+1 uses the hazard H = sigmoid(a + b * tau[t+1]); so the belief at the
    moment of deciding trial t is eroded by H(tau[t]) — the same anticipation
    semantics as before, now under Murphy's decide-then-update ordering. tau is
    exogenous (the experimenter's block clock), so indexing tau[t+1] is causal.
    With b = 0 the hazard is constant (= sigmoid(a)) and this reduces exactly to
    the constant-hazard belief model. Returns one log-likelihood per trial
    (NaN on miss).
    """
    a, b, beta = params
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    tau = np.asarray(tau, float)
    n = len(c)

    belief = 0.5
    ll = np.full(n, np.nan)
    for t in range(n):
        p_l = expit(beta * _q_diff(belief, p_high, p_low))   # decide on carried belief

        ct = c[t]
        if not np.isnan(ct):
            p_chosen = p_l if ct == -1 else (1.0 - p_l)
            ll[t] = np.log(np.clip(p_chosen, 1e-15, 1.0 - 1e-15))

        # End-of-trial update: transition (Eq. 7) then Bayes (Eq. 8). The
        # hazard governs the step into trial t+1, so it uses tau[t+1].
        tau_step = tau[t + 1] if (t + 1) < n else tau[t]
        H_step = expit(a + b * tau_step)
        belief = _belief_update(_transition(belief, H_step), ct, r[t], p_high, p_low)

    return ll


def belief_vhr_negloglike(params, c, r, n_rules: int = 2, tau=None,
                          score_mask=None) -> float:
    """
    Negative log-likelihood for belief_vhr (see `belief_vhr_trial_loglikes`).

    With score_mask=None all choice trials are summed; with a boolean mask only
    those trials contribute, while the latent state still propagates through
    every trial (used for cross-validation).
    """
    a, b, beta = params
    if beta <= 0 or b < 0:
        return 1e9
    ll = belief_vhr_trial_loglikes(params, c, r, n_rules, tau)
    if score_mask is not None:
        ll = ll[np.asarray(score_mask, bool)]
    return float(-np.nansum(ll))


def fit_belief_vhr(c, r, n_rules: int = 2, tau=None, n_restarts: int = 5,
                   rng=None, score_mask=None, optimizer: str = "bads") -> dict:
    """
    Fit the variable-hazard belief model.

    optimizer : "bads" (default; PyBADS, as in Murphy et al.) or "lbfgs". The
    first restart starts from a constant-hazard anchor (b = 0, baseline hazard
    ~0.1, beta = 5) so the search begins at the nested belief model.

    Returns dict with keys: model, fitpar, negloglike, bic, nlike
        fitpar = [a, b, beta]
    """
    if rng is None:
        rng = np.random.default_rng()
    c, r = np.asarray(c, float), np.asarray(r, float)
    if score_mask is None:
        n_obs = int(np.sum(~np.isnan(c)))
    else:
        n_obs = int(np.sum(~np.isnan(c) & np.asarray(score_mask, bool)))

    # a: baseline hazard logit; b: hazard slope in tau (>= 0); beta: inverse temp.
    lb  = [-12.0, 0.0, 0.0]
    ub  = [12.0,  5.0, 100.0]
    plb = [-6.0,  0.05, 1.0]
    pub = [2.0,   1.5, 20.0]
    fun = lambda x: belief_vhr_negloglike(x, c, r, n_rules, tau, score_mask)
    anchor_x0 = [-2.2, 0.1, 5.0]   # near-nested start (small slope), H~0.1

    best_nll, best_par = np.inf, None
    for i in range(n_restarts):
        x0 = anchor_x0 if i == 0 else [
            rng.uniform(-4.0, 0.0),
            rng.uniform(0.0, 0.5),
            rng.uniform(0.5, 15.0),
        ]
        if optimizer == "bads":
            par, nll = _run_bads(fun, x0, lb, ub, plb, pub)
        else:
            res = minimize(belief_vhr_negloglike, x0,
                           args=(c, r, n_rules, tau, score_mask),
                           method="L-BFGS-B",
                           bounds=[(-12.0, 12.0), (0.0, 5.0), (1e-3, 100.0)])
            par, nll = res.x, res.fun
        if nll < best_nll:
            best_nll, best_par = nll, par

    bic = 2.0 * best_nll + 3.0 * np.log(max(n_obs, 1))   # 3 parameters
    return {
        "model":      "belief_vhr",
        "fitpar":     best_par,
        "negloglike": best_nll,
        "bic":        bic,
        "nlike":      np.exp(-best_nll / max(n_obs, 1)),
    }


def simulate_belief_vhr(c_real, r_real, rule, tau, a: float, b: float, beta: float,
                        n_sims: int = 100, rng=None) -> dict:
    """
    Simulate choices under the variable-hazard belief model.

    Mirrors simulate_belief / simulate_belief_ck: each simulation uses the same
    miss-trial mask as the real session and the true block rules to draw
    synthetic rewards. The per-trial hazard is H_t = sigmoid(a + b * tau[t]),
    with tau = trials since criterion (use prepare_vhr to build it).

    Parameters
    ----------
    c_real  : (n_trials,) real choices — only used for miss masking
    r_real  : (n_trials,) real rewards — not used in simulation
    rule    : (n_trials,) true rule index per trial (for synthetic rewards)
    tau     : (n_trials,) trials since criterion per trial
    a, b, beta : model parameters
    n_sims  : number of independent repeats
    rng     : np.random.Generator

    Returns
    -------
    dict with keys c_sim, belief, p_left, hazard — each (n_trials, n_sims).
    """
    if rng is None:
        rng = np.random.default_rng()

    c_real = np.asarray(c_real, float)
    rule   = np.asarray(rule, float)
    tau    = np.asarray(tau, float)
    n      = len(c_real)
    miss   = np.isnan(c_real)

    unique_rules = rule[~np.isnan(rule)]
    n_rules = 2 if (len(unique_rules) == 0 or unique_rules.max() <= 2) else int(unique_rules.max())
    p_high, p_low = _reward_probs_for_n_rules(n_rules)
    prob_map = _get_reward_probs(n_rules)

    c_sim  = np.full((n, n_sims), np.nan)
    belief = np.full((n, n_sims), np.nan)
    p_left = np.full((n, n_sims), np.nan)
    hazard = np.full((n, n_sims), np.nan)

    for s in range(n_sims):
        bel = 0.5
        for t in range(n):
            # Decision uses the belief carried into the trial (Murphy order).
            belief[t, s] = bel
            # Hazard governing the step into the next trial (tau[t+1]); recorded
            # here for diagnostics.
            tau_step = tau[t + 1] if (t + 1) < n else tau[t]
            H_step = expit(a + b * tau_step)
            hazard[t, s] = H_step

            if miss[t]:
                bel = _transition(bel, H_step)   # transition only on misses
                continue

            pl = expit(beta * _q_diff(bel, p_high, p_low))
            p_left[t, s] = pl

            ct = -1.0 if rng.random() < pl else 1.0
            c_sim[t, s] = ct

            r_idx = rule[t]
            if not np.isnan(r_idx):
                lp, rp = prob_map.get(int(r_idx), (0.5, 0.5))
                prew   = lp if ct == -1 else rp
                rt     = 1.0 if rng.random() < prew else 0.0
            else:
                rt = np.nan

            # End-of-trial update: transition (Eq. 7) then Bayes (Eq. 8).
            bel = _belief_update(_transition(bel, H_step), ct, rt, p_high, p_low)

    return {"c_sim": c_sim, "belief": belief, "p_left": p_left, "hazard": hazard}