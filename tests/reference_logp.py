"""An independent reference for the M0 log density, written from the mathematics.

This file deliberately imports nothing from ``autobayes`` and nothing from PyMC or
Bambi. It builds its own design matrices from the data frame and evaluates

    log p(theta_value | y)  (up to the constant that PyMC also drops: none, for these
    densities; PyMC's Normal/HalfNormal/StudentT logps are fully normalised)

in PyMC's *value* space: scales enter as their logs and the Jacobian ``+ log(scale)``
of the log transform is added when ``jacobian=True``.

Model (centred form, matching brms and Bambi):

    mu_i     = a_c + sum_p (x_ip - xbar_p) b_p + sd * z[g_i]
    y_i      ~ Normal(mu_i, sigma)
    z[j]     ~ Normal(0, 1)
    a_c      ~ prior_intercept,  b_p ~ prior_b[p],  sd ~ prior_sd,  sigma ~ prior_sigma
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


def prior_logpdf(name: str, params: dict, x: float) -> float:
    """Log density of the named prior at x (natural scale), fully normalised."""
    if name == "Normal":
        return stats.norm.logpdf(x, loc=params["mu"], scale=params["sigma"])
    if name == "StudentT":
        return stats.t.logpdf(x, df=params["nu"], loc=params["mu"], scale=params["sigma"])
    if name == "HalfNormal":
        return stats.halfnorm.logpdf(x, scale=params["sigma"])
    if name == "HalfStudentT":
        nu, s = params["nu"], params["sigma"]
        if x <= 0:
            return -np.inf
        # 2 * t_nu(x / s) / s, written out to avoid depending on scipy's half-t
        return (np.log(2) + gammaln((nu + 1) / 2) - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi) - np.log(s)
                - (nu + 1) / 2 * np.log1p((x / s) ** 2 / nu))
    if name == "Exponential":
        return stats.expon.logpdf(x, scale=1.0 / params["lam"])
    if name == "HalfCauchy":
        return stats.halfcauchy.logpdf(x, scale=params["beta"])
    raise ValueError(name)


def _labels(s: pd.Series) -> np.ndarray:
    """String labels of a categorical column, integers in decimal form (the library's rule)."""
    if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        return np.array([str(int(v)) for v in s.to_numpy()], dtype=object)
    return np.array([str(v) for v in s.astype(object)], dtype=object)


def design_columns(data: pd.DataFrame, predictors: list[str], categorical: dict | None = None) -> list[str]:
    """Design-column labels in formula order: a numeric term as itself, a categorical term as
    ``z[level]`` for every sorted level except the reference (treatment coding)."""
    categorical = categorical or {}
    cols = []
    for term in predictors:
        if term in categorical:
            levels = sorted(set(_labels(data[term]).tolist()))
            cols.extend(f"{term}[{l}]" for l in levels if l != str(categorical[term]))
        else:
            cols.append(term)
    return cols


def build_design(data: pd.DataFrame, response: str, predictors: list[str], group: str, center: bool,
                 categorical: dict | None = None):
    """``predictors`` are the formula's predictor terms in order; ``categorical`` maps a term to
    its reference level. Dummies are 0/1 indicators of the non-reference levels; centring
    subtracts every column's mean (a dummy's mean is the level's proportion)."""
    categorical = categorical or {}
    y = data[response].to_numpy(dtype=float)
    cols = []
    for term in predictors:
        if term in categorical:
            lab = _labels(data[term])
            for l in sorted(set(lab.tolist())):
                if l != str(categorical[term]):
                    cols.append((lab == l).astype(float))
        else:
            cols.append(data[term].to_numpy(dtype=float))
    X = np.column_stack(cols) if cols else np.zeros((len(y), 0))
    xbar = X.mean(0) if center else np.zeros(X.shape[1])
    levels = sorted(data[group].astype(str).unique())
    gidx = np.array([levels.index(v) for v in data[group].astype(str)])
    return y, X - xbar, xbar, gidx, levels


def reference_logp(point: dict, data: pd.DataFrame, *, response: str, predictors: list[str],
                   group: str, priors: dict, center: bool = True, jacobian: bool = True,
                   categorical: dict | None = None) -> float:
    """``point`` holds a_c, b (array, one entry per design column), log_sigma, log_sd, z (array).

    ``priors`` maps 'intercept', 'b' (dict by design column), 'sd', 'sigma' to (name, params).
    """
    y, Xc, _, gidx, levels = build_design(data, response, predictors, group, center, categorical)
    columns = design_columns(data, predictors, categorical)
    a, b = float(point["a_c"]), np.atleast_1d(point["b"])
    log_sigma, log_sd = float(point["log_sigma"]), float(point["log_sd"])
    z = np.asarray(point["z"], dtype=float)
    assert z.shape == (len(levels),)
    sigma, sd = np.exp(log_sigma), np.exp(log_sd)

    mu = a + Xc @ b + sd * z[gidx]
    lp = np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    lp += np.sum(stats.norm.logpdf(z, 0.0, 1.0))
    lp += prior_logpdf(*priors["intercept"], a)
    for i, p in enumerate(columns):
        lp += prior_logpdf(*priors["b"][p], b[i])
    lp += prior_logpdf(*priors["sd"], sd)
    lp += prior_logpdf(*priors["sigma"], sigma)
    if jacobian:
        lp += log_sigma + log_sd
    return float(lp)


def conditional_gaussian_posterior(data, *, response, predictors, group, center, sigma, sd,
                                   prior_intercept, prior_b):
    """Exact posterior of (a_c, b, z) given fixed sigma and sd, all priors Normal.

    Returns (mean, cov) in the order [a_c, b_1..b_p, z_1..z_J].
    """
    y, Xc, _, gidx, levels = build_design(data, response, predictors, group, center)
    n, p, J = len(y), Xc.shape[1], len(levels)
    Z = np.zeros((n, J)); Z[np.arange(n), gidx] = 1.0
    W = np.column_stack([np.ones(n), Xc, sd * Z])
    prior_prec = np.zeros(1 + p + J)
    prior_mean = np.zeros(1 + p + J)
    prior_prec[0] = 1.0 / prior_intercept["sigma"] ** 2; prior_mean[0] = prior_intercept["mu"]
    for i, pname in enumerate(predictors):
        prior_prec[1 + i] = 1.0 / prior_b[pname]["sigma"] ** 2; prior_mean[1 + i] = prior_b[pname]["mu"]
    prior_prec[1 + p:] = 1.0
    P = np.diag(prior_prec) + W.T @ W / sigma**2
    cov = np.linalg.inv(P)
    mean = cov @ (prior_prec * prior_mean + W.T @ y / sigma**2)
    return mean, cov, P


def reference_logp_bernoulli(point: dict, data: pd.DataFrame, *, response: str, predictors: list[str],
                             group: str, priors: dict, center: bool = True, jacobian: bool = True,
                             categorical: dict | None = None) -> float:
    """M1 reference: Bernoulli likelihood with a logit link, same structure as ``reference_logp``.

    ``point`` holds a_c, b (array), log_sd, z (array); there is no residual scale. The
    likelihood is written with ``log_expit`` so it is exact at any logit, which is what the
    Bambi-built model is compared against (and where it deviates, see docs/m1-decision.md).
    """
    from scipy.special import log_expit

    y, Xc, _, gidx, levels = build_design(data, response, predictors, group, center, categorical)
    columns = design_columns(data, predictors, categorical)
    assert set(np.unique(y)) <= {0.0, 1.0}
    a, b = float(point["a_c"]), np.atleast_1d(point["b"])
    log_sd = float(point["log_sd"])
    z = np.asarray(point["z"], dtype=float)
    assert z.shape == (len(levels),)
    sd = np.exp(log_sd)

    eta = a + Xc @ b + sd * z[gidx]
    lp = np.sum(y * log_expit(eta) + (1.0 - y) * log_expit(-eta))
    lp += np.sum(stats.norm.logpdf(z, 0.0, 1.0))
    lp += prior_logpdf(*priors["intercept"], a)
    for i, p in enumerate(columns):
        lp += prior_logpdf(*priors["b"][p], b[i])
    lp += prior_logpdf(*priors["sd"], sd)
    if jacobian:
        lp += log_sd
    return float(lp)
