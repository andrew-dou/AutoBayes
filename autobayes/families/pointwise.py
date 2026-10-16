"""Pointwise log densities in NumPy: each family's log density of one observed response per posterior draw, which
``log_lik()`` evaluates (brms-parity C07, docs/post-processing.md). brms's ``log_lik()`` is the per-row term of brms's
likelihood; these are the per-row terms of this library's likelihoods (``autobayes.likelihoods`` and PyMC's own
distributions) in the same stable forms, written for arrays of draws: the argument shapes broadcast against each other,
``(chain, draw, row)`` for the linear predictor and ``(row,)`` for the response. ``tests/test_post_processing.py`` checks
every family's against the model's own elementwise log density at the frozen brms points of the oracle cases.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import betaln, gammaln, log_ndtr

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def softplus(x):
    """``log(1 + exp(x))``, exact for every finite ``x``."""
    return np.logaddexp(0.0, x)


def log1mexp(a):
    """``log(1 - exp(a))`` for ``a <= 0``: ``log(-expm1(a))`` above ``-log 2``, ``log1p(-exp(a))`` below."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > -math.log(2.0), np.log(-np.expm1(np.minimum(a, 0.0))), np.log1p(-np.exp(np.minimum(a, 0.0))))


def log_binomial_coefficient(n, k):
    """``log C(n, k)`` for whole ``0 <= k <= n`` as ``-log(n + 1) - log B(n - k + 1, k + 1)``, exact for large counts (SciPy's
    ``betaln``), where a difference of three ``gammaln`` values of magnitude ``n log n`` would not be."""
    n, k = np.asarray(n, dtype=float), np.asarray(k, dtype=float)
    return -np.log1p(n) - betaln(n - k + 1.0, k + 1.0)


# Under each link of a probability family: (log P(y = 1 | eta), log P(y = 0 | eta)), finite for every finite eta.
LINK_LOG_PROBABILITIES = {
    "logit": (lambda eta: -softplus(-eta), lambda eta: -softplus(eta)),
    "probit": (log_ndtr, lambda eta: log_ndtr(-eta)),
}


def normal(y, mu, sigma):
    """``log Normal(y | mu, sigma)``."""
    z = (y - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - _HALF_LOG_2PI


def student_t(y, nu, mu, sigma):
    """``log StudentT(y | nu, mu, sigma)`` (brms's ``student_t_lpdf``: the location-scale t)."""
    z = (y - mu) / sigma
    return (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * math.pi) - np.log(sigma)
            - 0.5 * (nu + 1.0) * np.log1p(z * z / nu))


def poisson_log(y, eta):
    """``log Poisson(y | exp(eta))`` = ``y eta - exp(eta) - log(y!)`` (Stan's ``poisson_log_lpmf``)."""
    return y * eta - np.exp(eta) - gammaln(y + 1.0)


def neg_binomial_2_log(y, eta, shape):
    """``log NegBinomial2(y | exp(eta), shape)`` (brms's ``neg_binomial_2_log_lpmf``: mean ``mu = exp(eta)``, variance
    ``mu + mu^2 / shape``): ``log C(y + shape - 1, y) + shape log(shape / (shape + mu)) + y log(mu / (shape + mu))``, the
    coefficient as ``-log(y + shape) - log B(shape, y + 1)`` and ``log(shape + mu)`` as ``logaddexp(log shape, eta)``."""
    log_total = np.logaddexp(np.log(shape), eta)
    return (-np.log(y + shape) - betaln(shape, y + 1.0) + shape * (np.log(shape) - log_total) + y * (eta - log_total))


def gamma_log(y, eta, shape):
    """``log Gamma(y | shape, rate = shape / exp(eta))`` (brms's ``gamma_lpdf(y | shape, shape / mu)``)."""
    return (shape * np.log(shape) - gammaln(shape) - shape * eta + (shape - 1.0) * np.log(y) - shape * y * np.exp(-eta))


def lognormal(y, mu, sigma):
    """``log LogNormal(y | mu, sigma)``: the Normal log density of ``log y`` minus ``log y``."""
    log_y = np.log(y)
    return normal(log_y, mu, sigma) - log_y


def beta_logit(y, eta, phi):
    """``log Beta(y | mu phi, (1 - mu) phi)`` with ``mu = expit(eta)`` and ``1 - mu = expit(-eta)``."""
    from scipy.special import expit

    a, b = expit(eta) * phi, expit(-eta) * phi
    return -betaln(a, b) + (a - 1.0) * np.log(y) + (b - 1.0) * np.log1p(-y)


def bernoulli(y, eta, link: str):
    """``log Bernoulli(y | p)`` with ``p`` the inverse of ``link`` at ``eta``, from the link's stable log probabilities."""
    log_p1, log_p0 = LINK_LOG_PROBABILITIES[link]
    return np.where(y == 1, log_p1(eta), log_p0(eta))


def binomial(y, trials, eta, link: str):
    """``log Binomial(y | trials, p)`` with ``p`` the inverse of ``link`` at ``eta``: the coefficient plus counts times the
    link's stable log probabilities (a count of zero contributes zero)."""
    log_p1, log_p0 = LINK_LOG_PROBABILITIES[link]
    failures = trials - y
    return (log_binomial_coefficient(trials, y) + np.where(y > 0, y * log_p1(eta), 0.0)
            + np.where(failures > 0, failures * log_p0(eta), 0.0))


def zero_inflated_poisson_log(y, eta, zi):
    """brms's ``zero_inflated_poisson_log_lpmf``: ``log(zi + (1 - zi) exp(-mu))`` at zero, ``log(1 - zi) + log Poisson(y |
    mu)`` above, ``mu = exp(eta)``."""
    zero = np.logaddexp(np.log(zi), np.log1p(-zi) - np.exp(eta))
    return np.where(y == 0, zero, np.log1p(-zi) + poisson_log(y, eta))


def hurdle_poisson_log(y, eta, hu):
    """brms's ``hurdle_poisson_log_lpmf``: ``log hu`` at zero, ``log(1 - hu) + log Poisson(y | mu) - log(1 - exp(-mu))``
    above, ``mu = exp(eta)``, the truncation term as ``log(-expm1(-mu))``."""
    positive = np.log1p(-hu) + poisson_log(y, eta) - log1mexp(-np.exp(eta))
    return np.where(y == 0, np.log(hu) + 0.0 * eta, positive)


def ordered_logistic(y, eta, thresholds):
    """``log P(y | eta, thresholds)`` of brms's ``cumulative("logit")`` for 1-based categories ``y``: ``thresholds`` is
    ``(..., K - 1)`` broadcasting against ``eta``'s ``(...)`` with a trailing axis; the stable form of
    ``likelihoods.ordered_logistic_logp``, ``-softplus(eta - tau_y) - softplus(tau_{y-1} - eta) + log(1 - exp(tau_{y-1} -
    tau_y))``, without the end terms at the first and last category."""
    eta = np.asarray(eta, dtype=float)
    tau = np.broadcast_to(np.asarray(thresholds, dtype=float), eta.shape + np.shape(thresholds)[-1:])
    K1 = tau.shape[-1]
    idx = np.broadcast_to(np.asarray(y, dtype=np.int64), eta.shape)
    upper = np.take_along_axis(tau, np.clip(idx - 1, 0, K1 - 1)[..., None], axis=-1)[..., 0]
    lower = np.take_along_axis(tau, np.clip(idx - 2, 0, K1 - 1)[..., None], axis=-1)[..., 0]
    has_upper, has_lower = idx <= K1, idx >= 2
    both = has_upper & has_lower
    out = (np.where(has_upper, -softplus(eta - upper), 0.0) + np.where(has_lower, -softplus(lower - eta), 0.0)
           + np.where(both, log1mexp(np.where(both, lower - upper, -1.0)), 0.0))
    return out


def gaussian_mixture(y, mu, sigma, theta):
    """``log sum_k theta_k Normal(y | mu_k, sigma_k)`` of brms's ``mixture(gaussian, ..., gaussian)`` (its ``log_mix`` of
    ``normal_lpdf``): ``mu`` is ``(..., K)``, the components' linear predictors, ``sigma`` and ``theta`` broadcast against it
    and ``y`` against its leading axes; the form of ``likelihoods.gaussian_mixture_logp``, the sum over the components with
    the largest term factored out."""
    z = (np.asarray(y, dtype=float)[..., None] - mu) / sigma
    terms = np.log(theta) - 0.5 * z * z - np.log(sigma) - _HALF_LOG_2PI
    m = terms.max(axis=-1, keepdims=True)
    return m[..., 0] + np.log(np.exp(terms - m).sum(axis=-1))


def categorical_logit(y, eta):
    """``log P(y | eta)`` of brms's ``categorical()`` for 1-based categories ``y``: ``eta`` is ``(..., K - 1)``, the linear
    predictors of the non-reference categories; ``eta_y - log_sum_exp(0, eta)``."""
    eta = np.asarray(eta, dtype=float)
    full = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], axis=-1)
    idx = np.broadcast_to(np.asarray(y, dtype=np.int64) - 1, eta.shape[:-1])
    picked = np.take_along_axis(full, idx[..., None], axis=-1)[..., 0]
    m = full.max(axis=-1)
    return picked - (m + np.log(np.exp(full - m[..., None]).sum(axis=-1)))
