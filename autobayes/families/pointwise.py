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


def ordinal(y, d, kind: str):
    """``log P(y | d)`` of the ordinal model ``kind`` (D04: ``cumulative_probit``, ``sratio_logit``, ``cratio_logit``,
    ``acat_logit``) for 1-based categories ``y``, from the gaps ``d = tau - eta`` (``(..., K - 1)``): the row's entry of
    ``likelihoods.ordinal_log_probabilities``, the stable forms of ``likelihoods.ordinal_logp_rows``."""
    from autobayes.likelihoods import ordinal_log_probabilities

    logp = ordinal_log_probabilities(kind, d)
    idx = np.broadcast_to(np.asarray(y, dtype=np.int64) - 1, logp.shape[:-1])
    return np.take_along_axis(logp, idx[..., None], axis=-1)[..., 0]


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


# -- D01, docs/continuous-families.md ------------------------------------------------------------------------------------------
def weibull(y, eta, shape):
    """brms's ``weibull_lpdf(y | shape, mu / Gamma(1 + 1 / shape))`` with ``mu = exp(eta)``: the Weibull of shape ``shape``
    and scale ``s``, ``log shape - log s + (shape - 1) log(y / s) - (y / s)^shape``, ``log s = eta - lgamma(1 + 1 / shape)``."""
    log_scale = eta - gammaln(1.0 + 1.0 / shape)
    log_ratio = np.log(y) - log_scale
    return np.log(shape) - log_scale + (shape - 1.0) * log_ratio - np.exp(shape * log_ratio)


def exponential_log(y, eta):
    """brms's ``exponential_lpdf(y | 1 / mu)`` with ``mu = exp(eta)``: ``-eta - y exp(-eta)``."""
    return -eta - y * np.exp(-eta)


def skew_normal(y, mu, sigma, alpha):
    """brms's ``skew_normal``, mean ``mu``, SD ``sigma``, skewness ``alpha``: Azzalini's ``log 2 - log omega + log phi(z) +
    log Phi(alpha z)`` with ``z = (y - xi) / omega`` (``likelihoods.skew_normal_location_scale``)."""
    from autobayes.likelihoods import skew_normal_location_scale

    xi, omega, _ = skew_normal_location_scale(np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float), np.asarray(alpha, dtype=float))
    z = (y - xi) / omega
    return math.log(2.0) - np.log(omega) - 0.5 * z * z - _HALF_LOG_2PI + log_ndtr(alpha * z)


def exgaussian(y, mu, sigma, beta):
    """brms's ``exgaussian``, mean ``mu`` (Stan's ``exp_mod_normal_lpdf(y | mu - beta, sigma, 1 / beta)``): ``-log beta +
    (mu - beta - y) / beta + sigma^2 / (2 beta^2) + log Phi((y - mu + beta) / sigma - sigma / beta)``."""
    r = sigma / beta
    return -np.log(beta) + (mu - beta - y) / beta + 0.5 * r * r + log_ndtr((y - mu + beta) / sigma - r)


def shifted_lognormal(y, mu, sigma, ndt):
    """brms's ``lognormal_lpdf(y - ndt | mu, sigma)``."""
    return lognormal(y - ndt, mu, sigma)


# -- D02, docs/discrete-families.md --------------------------------------------------------------------------------------------
def beta_binomial_logit(y, trials, eta, phi):
    """brms's ``beta_binomial_lpmf(y | trials, mu phi, (1 - mu) phi)`` with ``mu = expit(eta)`` and ``1 - mu = expit(-eta)``:
    ``log C(trials, y) + log B(y + a, trials - y + b) - log B(a, b)``."""
    from scipy.special import expit

    a, b = expit(eta) * phi, expit(-eta) * phi
    return log_binomial_coefficient(trials, y) + betaln(y + a, trials - y + b) - betaln(a, b)


# -- D03, docs/zero-inflated-hurdle.md -----------------------------------------------------------------------------------------
def _log_softplus(x):
    """``log(log(1 + exp(x)))``: ``x + log1p(-exp(x) / 2)`` below -20 (as ``likelihoods.log_softplus``), else its log."""
    x = np.asarray(x, dtype=float)
    low = x < -20.0
    with np.errstate(divide="ignore"):
        return np.where(low, x + np.log1p(-0.5 * np.exp(np.minimum(x, -20.0))), np.log(softplus(np.maximum(x, -20.0))))


def _cloglog_log_p1(t):
    """``log(1 - exp(-exp(t)))``: ``t + log1p(-exp(t) / 2)`` below -20 (as ``likelihoods.cloglog_log_p1``), else
    ``log(-expm1(-exp(t)))``."""
    t = np.asarray(t, dtype=float)
    with np.errstate(over="ignore", divide="ignore"):
        return np.where(t < -20.0, t + np.log1p(-0.5 * np.exp(np.minimum(t, -20.0))), np.log(-np.expm1(-np.exp(np.maximum(t, -20.0)))))


def neg_binomial_log_zero(eta, shape):
    """``log NB(0 | exp(eta), shape)`` = ``-shape softplus(eta - log shape)``."""
    return -shape * softplus(eta - np.log(shape))


def zero_inflated_negbinomial_log(y, eta, shape, zi):
    """brms's ``zero_inflated_neg_binomial_log_lpmf``: ``log(zi + (1 - zi) NB(0 | mu, shape))`` at zero, ``log(1 - zi) + log
    NB(y | mu, shape)`` above, ``mu = exp(eta)``."""
    zero = np.logaddexp(np.log(zi), np.log1p(-zi) + neg_binomial_log_zero(eta, shape))
    return np.where(y == 0, zero, np.log1p(-zi) + neg_binomial_2_log(y, eta, shape))


def hurdle_negbinomial_log(y, eta, shape, hu):
    """brms's ``hurdle_neg_binomial_log_lpmf``: ``log hu`` at zero, ``log(1 - hu) + log NB(y | mu, shape) - log(1 - NB(0 | mu,
    shape))`` above, the truncation as ``log(1 - exp(-exp(t)))`` with ``t = log shape + log softplus(eta - log shape)``."""
    log_shape = np.log(shape)
    positive = np.log1p(-hu) + neg_binomial_2_log(y, eta, shape) - _cloglog_log_p1(log_shape + _log_softplus(eta - log_shape))
    return np.where(y == 0, np.log(hu) + 0.0 * eta, positive)


def zero_inflated_binomial_logit(y, trials, eta, zi):
    """brms's ``zero_inflated_binomial_logit_lpmf``: ``log(zi + (1 - zi) (1 - p)^n)`` at zero, ``log(1 - zi) + log Binomial(y |
    n, p)`` above, ``p = expit(eta)``."""
    zero = np.logaddexp(np.log(zi), np.log1p(-zi) - trials * softplus(eta))
    return np.where(y == 0, zero, np.log1p(-zi) + binomial(y, trials, eta, "logit"))


def zero_inflated_beta_logit(y, eta, phi, zi):
    """brms's ``zero_inflated_beta_lpdf``: ``log zi`` at zero, ``log(1 - zi) + log Beta(y | mu phi, (1 - mu) phi)`` above, ``mu =
    expit(eta)`` (the Beta part evaluated at y = 1/2 and eta = 0 on the zero rows, where its density is infinite at 0 and its
    shape parameters may underflow)."""
    zero = y == 0
    positive = np.log1p(-zi) + beta_logit(np.where(zero, 0.5, y), np.where(zero, 0.0, eta), phi)
    return np.where(y == 0, np.log(zi) + 0.0 * eta, positive)


def hurdle_gamma_log(y, eta, shape, hu):
    """brms's ``hurdle_gamma_lpdf``: ``log hu`` at zero, ``log(1 - hu) + log Gamma(y | shape, shape / exp(eta))`` above."""
    positive = np.log1p(-hu) + gamma_log(np.where(y == 0, 1.0, y), eta, shape)
    return np.where(y == 0, np.log(hu) + 0.0 * eta, positive)


def hurdle_lognormal(y, mu, sigma, hu):
    """brms's ``hurdle_lognormal_lpdf``: ``log hu`` at zero, ``log(1 - hu) + log LogNormal(y | mu, sigma)`` above."""
    positive = np.log1p(-hu) + lognormal(np.where(y == 0, 1.0, y), mu, sigma)
    return np.where(y == 0, np.log(hu) + 0.0 * mu, positive)


# -- E06, docs/links.md --------------------------------------------------------------------------------------------------------
def poisson_sqrt(y, eta):
    """``log Poisson(y | eta^2)`` (brms's ``poisson("sqrt")``): ``2 y log|eta| - eta^2 - log(y!)``, the first term 0 at ``y = 0``."""
    eta = np.asarray(eta, dtype=float)
    with np.errstate(divide="ignore"):
        log_term = np.where(y > 0, 2.0 * y * np.log(np.abs(np.where(y > 0, eta, 1.0))), 0.0)
    return log_term - eta * eta - gammaln(y + 1.0)


def neg_binomial_2_identity(y, eta, shape):
    """``log NegBinomial2(y | eta, shape)`` (brms's ``negbinomial("identity")``): ``neg_binomial_2_log`` at ``log(eta)``, ``-inf``
    where ``eta <= 0``."""
    eta = np.asarray(eta, dtype=float)
    ok = eta > 0
    return np.where(ok, neg_binomial_2_log(y, np.log(np.where(ok, eta, 1.0)), shape), -np.inf)


def gamma_link(y, eta, shape, link: str):
    """``log Gamma(y | shape, rate)`` under brms's ``Gamma("inverse")`` (``rate = shape eta``) or ``Gamma("identity")`` (``rate =
    shape / eta``), ``-inf`` where ``eta <= 0``."""
    eta = np.asarray(eta, dtype=float)
    ok = eta > 0
    e = np.where(ok, eta, 1.0)
    log_rate, rate = (np.log(shape) + np.log(e), shape * e) if link == "inverse" else (np.log(shape) - np.log(e), shape / e)
    return np.where(ok, shape * log_rate - gammaln(shape) + (shape - 1.0) * np.log(y) - rate * y, -np.inf)


def _log_cloglog_complement(eta):
    """``log(1 - (1 - exp(-exp(eta)))) = -exp(eta)``, at ``eta`` capped at 700 (as ``likelihoods.cloglog_log_p0``)."""
    return -np.exp(np.minimum(eta, 700.0))


def _lgamma_of_shape(x, log_x):
    """``likelihoods._lgamma_of_shape``: ``gammaln(x)`` from 1e-10 up, ``-log_x - gamma x`` below."""
    tiny = log_x < math.log(1e-10)
    return np.where(tiny, -log_x - 0.5772156649015329 * np.exp(np.minimum(log_x, 0.0)), gammaln(np.where(tiny, 1.0, x)))


def beta_link(y, eta, phi, link: str):
    """``log Beta(y | mu phi, (1 - mu) phi)`` under brms's ``Beta("probit")`` or ``Beta("cloglog")``, ``mu`` and ``1 - mu`` each
    computed directly and the log-gamma of a shape parameter below 1e-10 from its logarithm (``likelihoods.beta_link_logp``)."""
    from scipy.special import ndtr

    eta = np.asarray(eta, dtype=float)
    if link == "probit":
        mu, cm, log_mu, log_cm = ndtr(eta), ndtr(-eta), log_ndtr(eta), log_ndtr(-eta)
    else:
        e = np.exp(np.minimum(eta, 700.0))
        mu, cm = -np.expm1(-e), np.exp(-e)
        log_mu, log_cm = _cloglog_log_p1(eta), _log_cloglog_complement(eta)
    a, b = mu * phi, cm * phi
    log_phi = np.log(phi)
    lbeta = _lgamma_of_shape(a, log_phi + log_mu) + _lgamma_of_shape(b, log_phi + log_cm) - gammaln(a + b)
    return (a - 1.0) * np.log(y) + (b - 1.0) * np.log1p(-y) - lbeta
