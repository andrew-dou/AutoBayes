"""An independent, high-precision reference for the M2 binomial log density and gradient.

Imports nothing from ``autobayes``, PyMC or Bambi, and does not use ``scipy.special.gammaln``:
the row terms are computed in 60-digit ``decimal`` arithmetic, including ``log C(n, k)`` from
``lnGamma`` values that are exact sums for arguments up to 2000 and a Stirling series with a
stated remainder bound above that (docs/m2-decision.md, "Independent reference and its error
budget"). Prior terms and the Jacobian reuse the SciPy reference of ``tests/reference_logp.py``.
"""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pandas as pd

from tests.reference_logp import build_design, design_columns, prior_logpdf, slope_terms

getcontext().prec = 60
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
HALF_LN_2PI = (2 * PI).ln() / 2
# Bernoulli numbers B2..B12 for the Stirling series; the first omitted term (B14 = 7/6) bounds the remainder.
BERNOULLI = [Decimal(1) / 6, Decimal(-1) / 30, Decimal(1) / 42, Decimal(-1) / 30, Decimal(5) / 66, Decimal(-691) / 2730]
EXACT_UPTO = 2000
_lgamma_cache: dict[int, Decimal] = {}


def lgamma_dec(z: int) -> Decimal:
    """ln Gamma(z) for an integer z >= 1: exact summation ln((z-1)!) for z <= 2000, Stirling
    series above (remainder below |B14| / (14 * 13 * z^13) < 1e-45 for z > 2000)."""
    if z in _lgamma_cache:
        return _lgamma_cache[z]
    if z <= EXACT_UPTO:
        total = Decimal(0)
        for i in range(2, z):
            total += Decimal(i).ln()
        out = total
    else:
        zd = Decimal(z)
        out = (zd - Decimal("0.5")) * zd.ln() - zd + HALF_LN_2PI
        for m, b in enumerate(BERNOULLI, start=1):
            out += b / (Decimal(2 * m * (2 * m - 1)) * zd ** (2 * m - 1))
    _lgamma_cache[z] = out
    return out


def stirling_remainder_bound(z: int) -> Decimal:
    return (Decimal(7) / 6) / (Decimal(14 * 13) * Decimal(z) ** 13)


def log_binom_dec(n: int, k: int) -> Decimal:
    return lgamma_dec(n + 1) - lgamma_dec(k + 1) - lgamma_dec(n - k + 1)


def row_loglik_dec(k: int, n: int, eta: float, *, coefficient: bool = True) -> Decimal:
    """log C(n, k) - k softplus(-eta) - (n - k) softplus(eta), in decimal."""
    e = Decimal(eta)
    sp_pos = (1 + e.exp()).ln()          # softplus(eta)
    sp_neg = (1 + (-e).exp()).ln()       # softplus(-eta)
    out = -Decimal(k) * sp_neg - Decimal(n - k) * sp_pos
    if coefficient:
        out += log_binom_dec(n, k)
    return out


def row_grad_dec(k: int, n: int, eta: float) -> Decimal:
    """d/d eta of the row log likelihood: k expit(-eta) - (n - k) expit(eta)."""
    e = Decimal(eta)
    return Decimal(k) / (1 + e.exp()) - Decimal(n - k) / (1 + (-e).exp())


def reference_logp_binomial(point: dict, data: pd.DataFrame, *, response: str, trials: str,
                            predictors: list[str], group: str, priors: dict, center: bool = True, categorical: dict | None = None, interactions: list | None = None,
                            jacobian: bool = True, coefficient: bool = True, varying_slope: str | None = None) -> float:
    """``point`` holds a_c, b (array), log_sd, z (array). ``data`` are the fitted rows."""
    y, Xc, _, gidx, levels = build_design(data, response, predictors, group, center, categorical, interactions)
    k = data[response].to_numpy(dtype=np.int64)
    n = data[trials].to_numpy(dtype=np.int64)
    a, b = float(point["a_c"]), np.atleast_1d(point["b"])
    log_sd = float(point["log_sd"])
    z = np.asarray(point["z"], dtype=float)
    assert z.shape == (len(levels),)
    sd = np.exp(log_sd)
    eta_slope, lp_slope = slope_terms(point, data, gidx, levels, priors, varying_slope, jacobian)
    eta = a + Xc @ b + sd * z[gidx] + eta_slope          # float64, like the model's linear predictor
    lik = Decimal(0)
    for ki, ni, ei in zip(k, n, eta):
        lik += row_loglik_dec(int(ki), int(ni), float(ei), coefficient=coefficient)
    lp = float(lik)
    lp += float(np.sum(__import__("scipy").stats.norm.logpdf(z, 0.0, 1.0)))
    lp += prior_logpdf(*priors["intercept"], a)
    for i, p in enumerate(design_columns(data, predictors, categorical, interactions)):
        lp += prior_logpdf(*priors["b"][p], b[i])
    lp += prior_logpdf(*priors["sd"], sd)
    lp += lp_slope
    if jacobian:
        lp += log_sd
    return lp


def reference_grad_binomial(point: dict, data: pd.DataFrame, *, response: str, trials: str,
                            predictors: list[str], group: str, prior_intercept_sigma: float, categorical: dict | None = None, interactions: list | None = None,
                            prior_b_sigma: float, jacobian: bool = True, center: bool = True) -> dict:
    """Gradient w.r.t. the value variables for Normal(0, s) intercept/slope priors and a
    HalfNormal(1) group SD (the priors the tests use); row terms in decimal."""
    y, Xc, _, gidx, levels = build_design(data, response, predictors, group, center, categorical, interactions)
    k = data[response].to_numpy(dtype=np.int64)
    n = data[trials].to_numpy(dtype=np.int64)
    a, b = float(point["a_c"]), np.atleast_1d(point["b"])
    sd = float(np.exp(point["log_sd"]))
    z = np.asarray(point["z"], dtype=float)
    eta = a + Xc @ b + sd * z[gidx]
    r = np.array([float(row_grad_dec(int(ki), int(ni), float(ei))) for ki, ni, ei in zip(k, n, eta)])
    out = {"Intercept": r.sum() - a / prior_intercept_sigma ** 2}
    for i, p in enumerate(design_columns(data, predictors, categorical, interactions)):
        out[p] = (r * Xc[:, i]).sum() - b[i] / prior_b_sigma ** 2
    out["1|g_sigma_log__"] = (r * z[gidx]).sum() * sd - sd ** 2 + (1.0 if jacobian else 0.0)
    out["1|g_offset"] = np.array([r[gidx == j].sum() * sd - z[j] for j in range(len(levels))])
    return out
