"""
changepoint_probability.py
==========================
Translation of changepointprobability.m
(H Atilgan 200417, adapted from MR Nassar / getOptimalLRs.m).

Computes the trial-by-trial change-point probability (CPP) for a
binary outcome sequence under a change-point prior.

Reference:
    Wilson, R.C., Nassar, M.R., & Gold, J.I. (2013).
    A delta-rule approximation to Bayesian inference in change-point
    problems. PLoS Computational Biology, 9(7), e1003150.
"""

import numpy as np


def changepoint_probability(
    input_seq: np.ndarray,
    hazardrate: float,
) -> np.ndarray:
    """
    Translation of changepointprobability.m.

    Parameters
    ----------
    input_seq  : float array
        Binary outcome sequence (0 / 1). NaN values are ignored during
        the computation and reinserted in the output at their original
        positions.
    hazardrate : float
        Prior probability of a change-point at each trial (0 < H < 1).

    Returns
    -------
    CPP : float array (same length as input_seq)
        Trial-by-trial change-point probability. NaN where input is NaN.
    """
    input_seq = np.asarray(input_seq, dtype=float)
    data      = input_seq[~np.isnan(input_seq)]
    m         = len(data)

    # Grid over possible outcome probabilities
    ps       = np.linspace(0, 1, 101)           # 0.00, 0.01, … 1.00
    cp_prior = np.ones(len(ps)) / len(ps)
    p        = cp_prior.copy()

    data_ll = np.full(m, np.nan)

    for i in range(m - 1):
        if data[i]:
            cond_prob = p * ps
        else:
            cond_prob = p * (1 - ps)

        data_ll[i + 1] = np.log(np.sum(cond_prob))

        # Change-point update
        p = p * (1 - hazardrate) + cp_prior * hazardrate
        p = p / np.nansum(p)

        # Likelihood update
        if data[i]:
            p = p * ps
        else:
            p = p * (1 - ps)
        p = p / np.nansum(p)

    # CPP from data log-likelihood and hazard rate
    Q        = (0.5 * hazardrate) / (np.exp(data_ll) * (1 - hazardrate))
    temp_cpp = Q / (1 + Q)

    # Reinsert NaNs at original positions
    CPP = np.full(len(input_seq), np.nan)
    CPP[~np.isnan(input_seq)] = temp_cpp

    return CPP
