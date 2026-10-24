"""Censored and truncated likelihoods beyond the Gaussian family (brms-parity E04, docs/cens-trunc-families.md).

brms 2.23 writes a censored row of a continuous family as the family's log CDF (``cens = -1``: the response is an upper
bound), log CCDF (``1``: a lower bound) or, under interval censoring ``cens(c, y2)`` (``2``), the log of the CDF difference
``log(F(y2) - F(y))``, and an observed row (``0``) as its log density; a truncated count row as its log mass minus the log mass
of the allowed range ``P(lb <= Y <= ub)``, the bounds inclusive (checked on the frozen cases ``weibull_censored``,
``lognormal_censored``, ``gamma_censored_interval``, ``poisson_truncated`` and ``negbinomial_truncated``). Here, from the
published distributions, each in a form that is finite with a finite gradient wherever its value is:

- Weibull (shape ``k``, scale ``s = mu / Gamma(1 + 1 / k)``): with ``u = k (log y - log s)``, ``F(y) = 1 - exp(-exp(u))``, the
  cloglog link's CDF in ``u``: ``log F = cloglog_log_p1(u)``, ``log(1 - F) = -exp(u)`` and the interval's
  ``cloglog_log_mass(u(y), u(y2))`` (A12, E06; never a difference of two probabilities).
- lognormal: the Normal tails of ``log y`` (``normal_log_cdf``, ``normal_log_ccdf``, ``normal_log_mass``, B02).
- Gamma (shape ``a``, rate ``a / mu``): ``F(y) = P(a, a y / mu)``, the regularized lower incomplete gamma function, and
  ``1 - F(y) = Q(a, a y / mu)``; an interval from the tail it lies in, ``log Q(x1) + log1mexp(log Q(x2) - log Q(x1))`` above
  the shape (``x1 > a``) and ``log P(x2) + log1mexp(log P(x1) - log P(x2))`` below it. ``P`` and ``Q`` are PyTensor's values
  (Cephes in C) and JAX's (``jax.scipy.special.gammainc``) on the two paths; their gradient in the shape ``a`` is this
  module's own on the PyTensor path (``gammainc_shape_derivative``: the series of ``P`` differentiated term by term below
  ``x = a + 1``, Legendre's continued fraction of ``Q`` differentiated through the modified Lentz recurrence above it, both
  to double precision), because PyTensor's (a port of Stan's ``grad_reg_inc_gamma``, stopped at 1e-6) is off by up to 0.3%
  relative on the frozen case's rows; JAX's (XLA's ``igamma_grad_a``) is within 1e-14 there.
- Poisson and negative binomial: ``log P(lb <= Y <= ub)`` as the log of a sum of the masses over the range (``trunc(ub = b)``:
  ``k = 0..b``; a lower bound alone: ``log1mexp`` of the masses below it, which for ``lb = 1`` is the exact
  ``log(1 - P(0))``), the masses in log space, never a difference of CDFs.

``Tails.neutral`` gives, from the family's parameters, a pair of response values well inside the support (the location and
twice it) that are substituted on the rows a censoring branch does not take, so that an untaken branch is never evaluated at
an extreme argument (whose zero adjoint would otherwise meet an infinite derivative, 0 * inf = NaN).

The NumPy forms (``numpy_tails``) give ``ab.log_lik()`` brms's censored and truncated pointwise log likelihood, and the count
families' truncated means and draws are exact sums over the support (``E[Y | Y <= b]``), or for a lower bound alone the
identities ``E[Y | Y >= l] = mu P(Y >= l - 1) / P(Y >= l)`` (Poisson) and ``mu P(Y+ >= l - 1) / P(Y >= l)`` with ``Y+`` the
negative binomial of shape ``shape + 1`` and the same success probability (both from ``k p(k) = mu p+(k - 1)``).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.scalar.basic import BinaryScalarOp, exp as _s_exp, log as _s_log, upgrade_to_float_no_complex
from pytensor.scalar.math import GammaInc, GammaIncC, gammaln as _s_gammaln
from pytensor.tensor.elemwise import Elemwise
from scipy import stats
from scipy.special import digamma, erf, gammainc as _sp_gammainc, gammaincc as _sp_gammaincc, gammaln, log_ndtr

from autobayes.likelihoods import (Tails, cloglog_log_mass, cloglog_log_p0, cloglog_log_p1, normal_log_ccdf, normal_log_cdf,
                                   normal_log_mass)

_SQRT2 = math.sqrt(2.0)
_TWO = np.float64(2.0)
_MAX_STEPS = 100_000            # the series and the continued fraction stop far earlier (about 9 sqrt(a) steps at most)
_STOP = 1e-18                   # a term below this relative to the sum, a factor within it of one: converged


# -- the regularized incomplete gamma function's derivative in its shape --------------------------------------------------
def _series_shape_derivative(a, x):
    """``dP(a, x)/da`` for ``0 < x < a + 1``: ``P = exp(a log x - x - lgamma(a + 1)) S`` with ``S = sum_n c_n``, ``c_0 = 1``,
    ``c_n = c_{n-1} x / (a + n)`` (positive terms, ratios below one), so ``dP/da = exp(...) (S (log x - digamma(a + 1)) +
    sum_n c_n D_n)`` with ``D_n = -sum_{j<=n} 1 / (a + j)``."""
    log_x = np.log(x)
    term, total, dlog, dsum = np.ones_like(a), np.ones_like(a), np.zeros_like(a), np.zeros_like(a)
    active = np.ones(a.shape, dtype=bool)
    for n in range(1, _MAX_STEPS + 1):
        new_term = term * x / (a + n)
        new_dlog = dlog - 1.0 / (a + n)
        new_total, new_dsum = total + new_term, dsum + new_term * new_dlog
        term, dlog = np.where(active, new_term, term), np.where(active, new_dlog, dlog)
        total, dsum = np.where(active, new_total, total), np.where(active, new_dsum, dsum)
        active &= ~((new_term <= _STOP * new_total) & (np.abs(new_term * new_dlog) <= _STOP * np.abs(new_dsum)))
        if not active.any():
            break
    else:
        total = np.where(active, np.nan, total)
    return np.exp(a * log_x - x - gammaln(a + 1.0)) * (total * (log_x - digamma(a + 1.0)) + dsum)


def _fraction_shape_derivative(a, x):
    """``(Q(a, x), d log Q(a, x)/da)`` for ``x >= a + 1``: ``Q = exp(a log x - x - lgamma(a)) h`` with ``h`` Legendre's continued
    fraction ``1 / (x + 1 - a - 1 (1 - a) / (x + 3 - a - 2 (2 - a) / ...))`` by the modified Lentz recurrence, carried with
    its derivative in ``a`` (forward mode, each update's own derivative), so ``d log Q/da = log x - digamma(a) + h' / h``."""
    tiny = 1e-300
    b = x + 1.0 - a                                   # db/da = -1 throughout
    c, dc = np.full_like(a, 1.0 / tiny), np.zeros_like(a)
    d, dd = 1.0 / b, 1.0 / (b * b)
    h, dh = d.copy(), dd.copy()
    active = np.ones(a.shape, dtype=bool)
    for i in range(1, _MAX_STEPS + 1):
        an, dan = -i * (i - a), float(i)
        b = b + 2.0
        nd, ndd = an * d + b, dan * d + an * dd - 1.0
        flat = np.abs(nd) < tiny
        nd, ndd = np.where(flat, tiny, nd), np.where(flat, 0.0, ndd)
        nc, ndc = b + an / c, -1.0 + dan / c - an * dc / (c * c)
        flat = np.abs(nc) < tiny
        nc, ndc = np.where(flat, tiny, nc), np.where(flat, 0.0, ndc)
        ndd, nd = -ndd / (nd * nd), 1.0 / nd
        delta, ddelta = nd * nc, ndd * nc + nd * ndc
        nh, ndh = h * delta, dh * delta + h * ddelta
        d, dd = np.where(active, nd, d), np.where(active, ndd, dd)
        c, dc = np.where(active, nc, c), np.where(active, ndc, dc)
        h, dh = np.where(active, nh, h), np.where(active, ndh, dh)
        active &= ~((np.abs(delta - 1.0) <= _STOP) & (np.abs(h * ddelta) <= _STOP * np.abs(ndh) + 1e-300))
        if not active.any():
            break
    else:
        h = np.where(active, np.nan, h)
    log_x = np.log(x)
    return np.exp(a * log_x - x - gammaln(a) + np.log(h)), log_x - digamma(a) + dh / h


def gammainc_shape_derivative(a, x):
    """``dP(a, x)/da`` of the regularized lower incomplete gamma function ``P`` (``-dQ/da``), to double precision: the
    series below ``x = a + 1``, the continued fraction above (``-Q d log Q/da``); 0 at ``x = 0`` and ``x = inf``, NaN outside
    ``a > 0, x >= 0``. NumPy, elementwise."""
    a, x = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(x, dtype=float))
    out = np.zeros(a.shape)
    valid = np.isfinite(a) & (a > 0) & (x >= 0) & ~np.isnan(x)
    out[~valid] = np.nan
    series = valid & (x > 0) & (x < a + 1.0)
    upper = valid & np.isfinite(x) & (x >= a + 1.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if series.any():
            out[series] = _series_shape_derivative(a[series], x[series])
        if upper.any():
            q, dlog = _fraction_shape_derivative(a[upper], x[upper])
            out[upper] = -q * dlog
    return out if out.ndim else float(out)


class _GammaIncShapeDerivative(BinaryScalarOp):
    """``gammainc_shape_derivative`` as a scalar op (Python; NumPy's vectorised function through ``nfunc_spec``), the shape
    gradient of ``gammainc_exact`` and ``gammaincc_exact``. It has no gradient of its own (first derivatives only)."""

    nfunc_spec = ("autobayes.tails.gammainc_shape_derivative", 2, 1)

    def impl(self, a, x):
        return np.float64(gammainc_shape_derivative(a, x))


_gammainc_da = _GammaIncShapeDerivative(upgrade_to_float_no_complex, name="gammainc_shape_derivative")


class ExactShapeGammaInc(GammaInc):
    """PyTensor's ``gammainc(a, x)`` (same value and C code, and the same derivative in ``x``) whose derivative in ``a`` is
    ``gammainc_shape_derivative``; its JAX lowering is ``jax.scipy.special.gammainc`` (``nfunc_spec``), whose own derivative
    JAX's autodiff takes."""

    def grad(self, inputs, grads):
        k, x = inputs
        (gz,) = grads
        return [gz * _gammainc_da(k, x), gz * _s_exp(-x + (k - 1) * _s_log(x) - _s_gammaln(k))]


class ExactShapeGammaIncC(GammaIncC):
    """PyTensor's ``gammaincc(a, x)``, likewise, with ``dQ/da = -dP/da``."""

    def grad(self, inputs, grads):
        k, x = inputs
        (gz,) = grads
        return [gz * -_gammainc_da(k, x), gz * -_s_exp(-x + (k - 1) * _s_log(x) - _s_gammaln(k))]


gammainc_exact = Elemwise(ExactShapeGammaInc(upgrade_to_float_no_complex, name="gammainc_exact"))
gammaincc_exact = Elemwise(ExactShapeGammaIncC(upgrade_to_float_no_complex, name="gammaincc_exact"))

try:  # the shape derivative's JAX lowering, for a gradient graph compiled to JAX (JAX's autodiff of the value needs none)
    import jax.lax as _lax
    from pytensor.link.jax.dispatch import jax_funcify

    @jax_funcify.register(_GammaIncShapeDerivative)
    def _jax_funcify_gammainc_shape_derivative(op, **kwargs):
        return _lax.igamma_grad_a
except ImportError:  # pragma: no cover - JAX is an optional extra; the pinned test environment always has it
    pass


# -- NumPy helpers ------------------------------------------------------------------------------------------------------
def _np_log1mexp(v):
    """``log(1 - exp(v))`` for ``v <= 0``, from whichever of ``log(-expm1(v))`` and ``log1p(-exp(v))`` is exact."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > -math.log(2.0), np.log(-np.expm1(v)), np.log1p(-np.exp(v)))


def np_normal_log_mass(a, b):
    """``log(Phi(b) - Phi(a))`` for standardised bounds ``a < b`` (NumPy): from the upper tail above zero, the lower tail below
    it, a sum of two erfs across it (``likelihoods.normal_log_mass``'s three forms)."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = log_ndtr(-a) + _np_log1mexp(log_ndtr(-b) - log_ndtr(-a))
        lower = log_ndtr(b) + _np_log1mexp(log_ndtr(a) - log_ndtr(b))
        middle = np.log(0.5 * (erf(b / _SQRT2) - erf(a / _SQRT2)))
    return np.where(a > 0, upper, np.where(b < 0, lower, middle))


def _np_cloglog_log_mass(u1, u2):
    """``log(exp(-exp(u1)) - exp(-exp(u2)))`` for ``u1 < u2`` (NumPy): ``-t1 + log1mexp(-(t2 - t1))``, ``t2 - t1`` formed as
    ``exp(u2) (-expm1(u1 - u2))``."""
    with np.errstate(over="ignore", invalid="ignore"):
        return -np.exp(u1) + _np_log1mexp(-np.exp(u2) * -np.expm1(u1 - u2))


def _np_gamma_log_mass(a, x1, x2):
    """``log(P(a, x2) - P(a, x1))`` from the tail the interval lies in (NumPy)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lq1, lq2 = np.log(_sp_gammaincc(a, x1)), np.log(_sp_gammaincc(a, x2))
        lp1, lp2 = np.log(_sp_gammainc(a, x1)), np.log(_sp_gammainc(a, x2))
        return np.where(x1 > a, lq1 + _np_log1mexp(lq2 - lq1), lp2 + _np_log1mexp(lp1 - lp2))


# -- the continuous families' tails ----------------------------------------------------------------------------------
def _weibull_u(value, alpha, beta):
    return alpha * (pt.log(value) - pt.log(beta))


def _weibull_numpy(model, post, eta):
    from autobayes.families.spec import auxiliary

    shape = auxiliary(model, post, "shape")
    log_scale = eta - gammaln(1.0 + 1.0 / shape)

    def u(v):
        return shape * (np.log(v) - log_scale)

    def log_ccdf(v):
        with np.errstate(over="ignore"):
            return -np.exp(u(v))

    return SimpleNamespace(log_cdf=lambda v: _np_log1mexp(log_ccdf(v)), log_ccdf=log_ccdf,
                           log_mass=lambda lo, hi: _np_cloglog_log_mass(u(lo), u(hi)))


WEIBULL_TAILS = Tails(   # Bambi's Weibull(alpha = shape, beta = mu / Gamma(1 + 1 / shape))
    log_cdf=lambda value, *, alpha, beta: cloglog_log_p1(_weibull_u(value, alpha, beta)),
    log_ccdf=lambda value, *, alpha, beta: cloglog_log_p0(_weibull_u(value, alpha, beta)),
    log_mass=lambda lower, upper, *, alpha, beta: cloglog_log_mass(_weibull_u(lower, alpha, beta), _weibull_u(upper, alpha, beta)),
    neutral=lambda *, alpha, beta: (beta, _TWO * beta),
    numpy_tails=_weibull_numpy,
)


def _lognormal_numpy(model, post, eta):
    from autobayes.families.spec import auxiliary

    sigma = auxiliary(model, post, "sigma")
    return SimpleNamespace(log_cdf=lambda v: log_ndtr((np.log(v) - eta) / sigma), log_ccdf=lambda v: log_ndtr((eta - np.log(v)) / sigma),
                           log_mass=lambda lo, hi: np_normal_log_mass((np.log(lo) - eta) / sigma, (np.log(hi) - eta) / sigma))


LOGNORMAL_TAILS = Tails(   # likelihoods.lognormal_family: PyMC's LogNormal(mu, sigma), mu the location of log y
    log_cdf=lambda value, *, mu, sigma: normal_log_cdf(pt.log(value), mu, sigma),
    log_ccdf=lambda value, *, mu, sigma: normal_log_ccdf(pt.log(value), mu, sigma),
    log_mass=lambda lower, upper, *, mu, sigma: normal_log_mass(pt.log(lower), pt.log(upper), mu, sigma),
    neutral=lambda *, mu, sigma: (pt.exp(mu), _TWO * pt.exp(mu)),
    numpy_tails=_lognormal_numpy,
)


def _gamma_shape_rate(mu, sigma):
    """PyMC's Gamma(mu, sigma) (Bambi's ``sigma = mu / sqrt(shape)``) as its shape and rate, as PyMC forms them."""
    return mu ** 2 / sigma ** 2, mu / sigma ** 2


def _gamma_log_cdf(value, *, mu, sigma):
    a, rate = _gamma_shape_rate(mu, sigma)
    return pt.log(gammainc_exact(a, value * rate))


def _gamma_log_ccdf(value, *, mu, sigma):
    a, rate = _gamma_shape_rate(mu, sigma)
    return pt.log(gammaincc_exact(a, value * rate))


def _gamma_log_mass(lower, upper, *, mu, sigma):
    """``log(P(a, x2) - P(a, x1))``, ``x = y a / mu``, from the upper tail where ``x1 > a`` and the lower tail elsewhere, each
    form evaluated at ``(a, 2 a)`` (well inside both tails) on the rows it does not take."""
    a, rate = _gamma_shape_rate(mu, sigma)
    x1, x2 = lower * rate, upper * rate
    above = x1 > a
    u1, u2 = pt.switch(above, x1, a), pt.switch(above, x2, _TWO * a)
    lq1, lq2 = pt.log(gammaincc_exact(a, u1)), pt.log(gammaincc_exact(a, u2))
    l1, l2 = pt.switch(above, a, x1), pt.switch(above, _TWO * a, x2)
    lp1, lp2 = pt.log(gammainc_exact(a, l1)), pt.log(gammainc_exact(a, l2))
    return pt.switch(above, lq1 + pt.log1mexp(lq2 - lq1), lp2 + pt.log1mexp(lp1 - lp2))


def _gamma_numpy(model, post, eta):
    from autobayes.families.spec import auxiliary

    a = auxiliary(model, post, "shape")
    rate = a * np.exp(-eta)
    with np.errstate(divide="ignore"):
        return SimpleNamespace(log_cdf=lambda v: np.log(_sp_gammainc(a, v * rate)), log_ccdf=lambda v: np.log(_sp_gammaincc(a, v * rate)),
                               log_mass=lambda lo, hi: _np_gamma_log_mass(a, lo * rate, hi * rate))


GAMMA_TAILS = Tails(   # Bambi's Gamma(mu, sigma = mu / sqrt(shape)), the log link
    log_cdf=_gamma_log_cdf, log_ccdf=_gamma_log_ccdf, log_mass=_gamma_log_mass,
    neutral=lambda *, mu, sigma: (mu, _TWO * mu),
    numpy_tails=_gamma_numpy,
)


# -- the count families' truncation ----------------------------------------------------------------------------------------
def _logsumexp(x):
    """``log sum exp`` over the last axis with the largest term factored out."""
    m = pt.max(x, axis=-1, keepdims=True)
    return m[..., 0] + pt.log(pt.sum(pt.exp(x - m), axis=-1))


def _count_log_range(lpmf, lower, upper):
    """``log P(lower <= Y <= upper)`` (inclusive integer bounds, ``None`` for an open side) of a count whose log mass at the
    constant points ``k`` is ``lpmf(k)`` (``(..., K)``): the log of the sum of the masses over the range when it is bounded
    above, ``log1mexp`` of the masses below ``lower`` when it is not (``lower <= 0``: the whole support, 0)."""
    if upper is not None:
        return _logsumexp(lpmf(np.arange(0 if lower is None else max(int(lower), 0), int(upper) + 1, dtype=np.float64)))
    if lower <= 0:
        return pt.zeros_like(lpmf(np.zeros(1))[..., 0])
    return pt.log1mexp(_logsumexp(lpmf(np.arange(0, int(lower), dtype=np.float64))))


def _poisson_lpmf(mu):
    m = pt.shape_padright(pt.as_tensor_variable(mu))
    return lambda k: k * pt.log(m) - m - gammaln(k + 1.0)


def _negbinomial_lpmf(mu, alpha):
    """``log NegBinomial2(k | mu, alpha)``: ``lgamma(k + alpha) - lgamma(alpha) - lgamma(k + 1) - alpha log1p(mu / alpha) +
    k (log mu - log(mu + alpha))``."""
    m, s = pt.shape_padright(pt.as_tensor_variable(mu)), pt.shape_padright(pt.as_tensor_variable(alpha))
    return lambda k: (pt.gammaln(k + s) - pt.gammaln(s) - gammaln(k + 1.0) - s * pt.log1p(m / s)
                      + k * (pt.log(m) - pt.log(m + s)))


def _np_count_log_range(logpmf, lower, upper, shape):
    """``_count_log_range`` in NumPy, one point of the support at a time (``logpmf(k)`` broadcasts to ``shape``)."""
    lo, hi = (0 if lower is None or lower <= 0 else int(lower)), (None if upper is None else int(upper))
    if hi is not None:
        acc = np.full(shape, -np.inf)
        for k in range(lo, hi + 1):
            acc = np.logaddexp(acc, logpmf(k))
        return acc
    if lo == 0:
        return np.zeros(shape)
    acc = np.full(shape, -np.inf)
    for k in range(lo):
        acc = np.logaddexp(acc, logpmf(k))
    return _np_log1mexp(acc)


def _count_distribution(model, post, eta):
    """The scipy distribution of each draw and row's count, its mean and its shape (``(chain, draw, row)``)."""
    from autobayes.families.spec import auxiliary, exp_mean

    mu = exp_mean(eta, "truncated expected count")
    if model.family_spec.scale_parameter is None:                         # Poisson
        return stats.poisson(mu), mu, None
    shape = np.broadcast_to(auxiliary(model, post, model.family_spec.scale_parameter), mu.shape)
    return stats.nbinom(shape, shape / (shape + mu)), mu, shape


def _count_truncated_mean(model, post, eta, lb, ub):
    """``E[Y | lb <= Y <= ub]`` per draw and row: the support's exact weighted sum when ``ub`` is finite (log weights, one
    point at a time), otherwise ``mu P(Y >= lb - 1) / P(Y >= lb)`` (Poisson) or ``mu P(Y+ >= lb - 1) / P(Y >= lb)`` with ``Y+``
    the negative binomial of shape ``shape + 1`` and the same success probability, each tail from SciPy's survival function."""
    dist, mu, shape = _count_distribution(model, post, eta)
    lo = 0 if not np.isfinite(lb) or lb <= 0 else int(lb)
    if np.isfinite(ub):
        num, den = np.full(mu.shape, -np.inf), np.full(mu.shape, -np.inf)
        with np.errstate(divide="ignore"):
            for k in range(lo, int(ub) + 1):
                lp = dist.logpmf(k)
                den = np.logaddexp(den, lp)
                if k > 0:
                    num = np.logaddexp(num, math.log(k) + lp)
        return np.exp(num - den)
    if lo == 0:
        return mu
    if shape is None:
        above = stats.poisson(mu).sf(lo - 2) if lo >= 2 else 1.0
    else:
        above = stats.nbinom(shape + 1.0, shape / (shape + mu)).sf(lo - 2) if lo >= 2 else 1.0
    return mu * above / dist.sf(lo - 1)


def _truncated_count_inverse(dist, lb, ub, u):
    """The counts of the frozen SciPy distribution ``dist`` truncated to ``[lb, ub]`` (``-inf`` / ``inf`` for an open side) at
    the uniforms ``u``: the truncated CDF inverted over the support, one point at a time, when ``ub`` is finite (its masses in
    log space, so a range far in a tail keeps its mass), otherwise SciPy's inverse survival function of the upper tail."""
    lo = 0 if not np.isfinite(lb) or lb <= 0 else int(lb)
    if np.isfinite(ub):
        total = np.full(u.shape, -np.inf)
        with np.errstate(divide="ignore"):
            for k in range(lo, int(ub) + 1):
                total = np.logaddexp(total, dist.logpmf(k))
            out, cum = np.full(u.shape, int(ub), dtype=np.int64), np.zeros(u.shape)
            open_ = np.ones(u.shape, dtype=bool)
            for k in range(lo, int(ub) + 1):
                cum = cum + np.exp(dist.logpmf(k) - total)
                hit = open_ & (u <= cum)
                out[hit], open_ = k, open_ & ~hit
        return out
    y = dist.isf(u * dist.sf(lo - 1))
    return np.maximum(np.where(np.isfinite(y), y, lo), lo).astype(np.int64)


def _count_draw_truncated(model, post, eta, lb, ub, rng):
    """brms's posterior_predict of a truncated count model: each draw and row's count from the truncated distribution."""
    dist, mu, _ = _count_distribution(model, post, eta)
    return _truncated_count_inverse(dist, lb, ub, rng.uniform(size=mu.shape))


def _truncated_count_rv(frozen, lower, upper, size, *params):
    """A random variable of the count truncated to ``[lower, upper]`` (``None`` for an open side) for prior-predictive draws, by
    ``_truncated_count_inverse`` (PyMC's Truncated rejects draws, which never ends when the range has almost no mass)."""
    lb, ub = (-np.inf if lower is None else lower), (np.inf if upper is None else upper)

    def random(*values, rng=None, size=None):
        shape = tuple(size) if size is not None else np.broadcast_shapes(*(np.shape(v) for v in values))
        dist = frozen(*(np.broadcast_to(v, shape) for v in values))
        return _truncated_count_inverse(dist, lb, ub, rng.uniform(size=shape))

    return pm.CustomDist.dist(*params, random=random, dtype="int64", size=size)


def _count_numpy(model, post, eta):
    dist, _, _ = _count_distribution(model, post, eta)
    return SimpleNamespace(log_truncation=lambda lower, upper: _np_count_log_range(dist.logpmf, lower, upper, eta.shape))


POISSON_TAILS = Tails(   # PyMC's Poisson(mu); trunc() only (no cens(): divergence D122)
    truncated=lambda lower, upper, *, size=None, mu: _truncated_count_rv(lambda m: stats.poisson(m), lower, upper, size, mu),
    truncated_mean=_count_truncated_mean, draw_truncated=_count_draw_truncated,
    log_truncation=lambda lower, upper, *, mu: _count_log_range(_poisson_lpmf(mu), lower, upper),
    numpy_tails=_count_numpy,
)

NEGBINOMIAL_TAILS = Tails(   # PyMC's NegativeBinomial(mu, alpha = shape); trunc() only
    truncated=lambda lower, upper, *, size=None, mu, alpha: _truncated_count_rv(lambda m, s: stats.nbinom(s, s / (s + m)), lower, upper,
                                                                                size, mu, alpha),
    truncated_mean=_count_truncated_mean, draw_truncated=_count_draw_truncated,
    log_truncation=lambda lower, upper, *, mu, alpha: _count_log_range(_negbinomial_lpmf(mu, alpha), lower, upper),
    numpy_tails=_count_numpy,
)
