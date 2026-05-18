"""
logistic_regression.py
======================
Translations of:
    logreg_RCUC.m     (AC Kwan 170518)
    logreg_RCUC_LR.m  (AC Kwan 170518)

Logistic regression analyses of choice behavior using
rewarded / unrewarded choice history as regressors.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _logistic_fit(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit a logistic regression via scikit-learn (mirrors MATLAB glmfit).

    Returns (coeff, pvals) where coeff[0] is the intercept and
    pvals are two-sided Wald p-values.
    """
    from scipy import stats as scipy_stats

    clf = LogisticRegression(
        fit_intercept=True,
        max_iter=1000,
        solver="lbfgs",
        C=1e9,          # virtually no regularisation → matches glmfit
    )
    clf.fit(X, y)

    intercept = clf.intercept_[0]
    coefs     = clf.coef_[0]
    b         = np.concatenate([[intercept], coefs])

    # Wald p-values
    pred  = clf.predict_proba(X)[:, 1]
    W     = pred * (1 - pred)
    XB    = np.column_stack([np.ones(len(X)), X])
    try:
        cov   = np.linalg.inv(XB.T @ np.diag(W) @ XB)
        se    = np.sqrt(np.diag(cov))
        z     = b / se
        pvals = 2 * scipy_stats.norm.sf(np.abs(z))
    except np.linalg.LinAlgError:
        pvals = np.full(len(b), np.nan)

    return b, pvals


def _neg_log_like(c: np.ndarray, pr: np.ndarray) -> float:
    nll = 0.0
    for k in range(len(c)):
        if c[k] == 1:
            nll -= np.log(np.clip(pr[k], 1e-12, 1))
        elif c[k] == -1:
            nll -= np.log(np.clip(1 - pr[k], 1e-12, 1))
    return nll


# ---------------------------------------------------------------------------
# logreg_RCUC
# ---------------------------------------------------------------------------

def logreg_RCUC(stats: dict, step_back: int) -> tuple:
    """
    Translation of logreg_RCUC.m.

    Logistic regression with two regressors per lag:
        YR  — rewarded choice   (+1 right, -1 left)
        NR  — unrewarded choice (+1 right, -1 left)

    Parameters
    ----------
    stats     : dict  Output of get_trial_stats() or get_trial_stats_more()
    step_back : int   Number of past trials to include

    Returns
    -------
    output      : dict  (see below)
    negloglike  : float
    bic         : float
    nlike       : float  normalised likelihood (Ito & Doya 2015)

    output keys
    -----------
        n           : int array  [-1, -2, … -step_back]
        b_bias      : float   intercept
        pval_bias   : float
        b_coeff     : (step_back, 2) array  [:, 0]=YR  [:, 1]=NR
        pval_coeff  : (step_back, 2) array
        b_label     : ['Rewarded choice', 'Unrewarded choice']
    """
    c = stats["c"]
    r = stats["r"]
    n = len(c)

    YR = np.zeros(n)
    YR = r * ((c == 1) & (r > 0)).astype(float) + \
         (-1) * r * ((c == -1) & (r > 0)).astype(float)

    NR = np.zeros(n)
    NR = ((c == 1) & (r == 0)).astype(float) + \
         (-1) * ((c == -1) & (r == 0)).astype(float)

    # Build regressor matrix  (n - step_back) × (2 * step_back)
    rmat = np.zeros((n - step_back, 2 * step_back))
    for i in range(step_back, n):
        for j in range(1, step_back + 1):
            rmat[i - step_back, j - 1]            = YR[i - j]
            rmat[i - step_back, j - 1 + step_back] = NR[i - j]

    crit1      = np.concatenate([np.zeros(step_back), np.ones(n - step_back)])
    crit2      = ~np.isnan(c)
    good_trial = (crit1 == 1) & crit2

    c_fit    = (c == 1).astype(float)[good_trial]
    rmat_fit = rmat[good_trial[step_back:], :]
    rmat_fit = np.nan_to_num(rmat_fit, nan=0.0)

    b, pvals = _logistic_fit(rmat_fit, c_fit)

    output = {
        "n":          np.arange(-1, -step_back - 1, -1),
        "b_bias":     b[0],
        "pval_bias":  pvals[0],
        "b_coeff":    np.column_stack([b[1:step_back + 1], b[step_back + 1:]]),
        "pval_coeff": np.column_stack([pvals[1:step_back + 1], pvals[step_back + 1:]]),
        "b_label":    ["Rewarded choice", "Unrewarded choice"],
    }

    # Negative log-likelihood using sequential prediction
    pr = np.full(n, 0.5)
    for k in range(1, n):
        back = min(k, step_back)
        stop = None if k - back - 1 < 0 else k - back - 1
        b_temp = output["b_bias"]
        b_temp += np.nansum(output["b_coeff"][:back, 0] * YR[k - 1:stop:-1])
        b_temp += np.nansum(output["b_coeff"][:back, 1] * NR[k - 1:stop:-1])
        pr[k] = np.exp(b_temp) / (1 + np.exp(b_temp))

    negloglike = _neg_log_like(c, pr)
    n_params   = len(b)
    n_valid    = int(np.sum(~np.isnan(c)))
    bic        = negloglike + n_params * np.log(n_valid)
    nlike      = np.exp(-negloglike) ** (1 / n_valid)

    return output, negloglike, bic, nlike


# ---------------------------------------------------------------------------
# logreg_RCUC_LR
# ---------------------------------------------------------------------------

def logreg_RCUC_LR(stats: dict, step_back: int) -> tuple:
    """
    Translation of logreg_RCUC_LR.m.

    Like logreg_RCUC() but uses four separate regressors per lag:
        YR_right  — rewarded right choice
        YR_left   — rewarded left choice   (encoded as -1)
        NR_right  — unrewarded right choice
        NR_left   — unrewarded left choice  (encoded as -1)

    Returns the same (output, negloglike, bic, nlike) tuple.
    output["b_coeff"] is (step_back, 4) with columns in the order above.
    """
    c = stats["c"]
    r = stats["r"]
    n = len(c)

    YR_right = r * ((c == 1) & (r > 0)).astype(float)
    YR_left  = (-1) * r * ((c == -1) & (r > 0)).astype(float)
    NR_right = ((c == 1) & (r == 0)).astype(float)
    NR_left  = (-1) * ((c == -1) & (r == 0)).astype(float)

    rmat = np.zeros((n - step_back, 4 * step_back))
    for i in range(step_back, n):
        for j in range(1, step_back + 1):
            rmat[i - step_back, j - 1]                   = YR_right[i - j]
            rmat[i - step_back, j - 1 +     step_back]   = YR_left[i - j]
            rmat[i - step_back, j - 1 + 2 * step_back]   = NR_right[i - j]
            rmat[i - step_back, j - 1 + 3 * step_back]   = NR_left[i - j]

    crit1      = np.concatenate([np.zeros(step_back), np.ones(n - step_back)])
    crit2      = ~np.isnan(c)
    good_trial = (crit1 == 1) & crit2

    c_fit    = (c == 1).astype(float)[good_trial]
    rmat_fit = rmat[good_trial[step_back:], :]
    rmat_fit = np.nan_to_num(rmat_fit, nan=0.0)

    b, pvals = _logistic_fit(rmat_fit, c_fit)

    s = step_back
    output = {
        "n":     np.arange(-1, -s - 1, -1),
        "b_bias":     b[0],
        "pval_bias":  pvals[0],
        "b_coeff": np.column_stack([
            b[1:s + 1],
            b[s + 1:2*s + 1],
            b[2*s + 1:3*s + 1],
            b[3*s + 1:4*s + 1],
        ]),
        "pval_coeff": np.column_stack([
            pvals[1:s + 1],
            pvals[s + 1:2*s + 1],
            pvals[2*s + 1:3*s + 1],
            pvals[3*s + 1:4*s + 1],
        ]),
        "b_label": [
            "Rewarded right choice",
            "Rewarded left choice",
            "Unrewarded right choice",
            "Unrewarded left choice",
        ],
    }

    regressors = [YR_right, YR_left, NR_right, NR_left]
    pr = np.full(n, 0.5)
    for k in range(1, n):
        back   = min(k, s)
        stop   = None if k - back - 1 < 0 else k - back - 1
        b_temp = output["b_bias"]
        for col, reg in enumerate(regressors):
            b_temp += np.nansum(
                output["b_coeff"][:back, col] * reg[k - 1:stop:-1]
            )
        pr[k] = np.exp(b_temp) / (1 + np.exp(b_temp))

    negloglike = _neg_log_like(c, pr)
    n_params   = len(b)
    n_valid    = int(np.sum(~np.isnan(c)))
    bic        = negloglike + n_params * np.log(n_valid)
    nlike      = np.exp(-negloglike) ** (1 / n_valid)

    return output, negloglike, bic, nlike
