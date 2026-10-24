"""The Wiener diffusion model of response times and decisions (brms ``wiener()``, brms-parity F04, docs/wiener.md).

A Wiener process with drift ``v`` and unit diffusion starts at ``a w`` (``0 < w < 1``) between two absorbing boundaries, 0 and
``a > 0``; the decision time is the time it first reaches one of them, the decision which one. Navarro and Fuss (2009, "Fast
and accurate calculations for first-passage times in Wiener diffusion models", J. Math. Psych. 53) write the density of
reaching the lower boundary at time ``t`` as

    f(t | v, a, w) = a^-2 exp(-v a w - v^2 t / 2) f(t / a^2 | 0, 1, w)

with two series for the standard density ``f(u | 0, 1, w)``, each exact in the limit:

    small time: (2 pi u^3)^(-1/2) sum_k (w + 2k) exp(-(w + 2k)^2 / (2u)), over every integer k;
    large time: pi sum_{k >= 1} k exp(-k^2 pi^2 u / 2) sin(k pi w).

The upper boundary's density is the lower one's with ``(v, w)`` replaced by ``(-v, 1 - w)`` (the process reflected). brms's
``wiener()`` (Stan's ``wiener_lpdf(rt | bs, ndt, bias, mu)``) is the upper boundary's density at ``t = rt - ndt`` for a row with
``dec = 1`` and, for ``dec = 0``, ``wiener_lpdf(rt | bs, ndt, 1 - bias, -mu)``, the lower boundary's: with ``a = bs``, ``v = mu``
and ``w = bias``.

Here both series are summed with a fixed number of terms, each where it converges fast: the small-time one for ``u <= 1/2``
(k from -5 to 4, or -4 to 4), the large-time one above (k from 1 to 5), which keeps the omitted terms below 1e-20 of the sum.
Their terms are grouped so that no two of opposite sign and nearly equal size are added: the small-time terms in pairs
around 0 for ``w <= 1/2`` (k with -k, as ``w (A_k + B_k) + 2k B_k expm1(-4 k w / u)``) and around 1 for ``w > 1/2`` (k with
-(k + 1)); the large-time sine of ``k pi w`` as ``(-1)^(k + 1) sin(k pi (1 - w))`` for ``w > 1/2``; and ``1 - w`` is passed
beside ``w`` (``e``), so that it is ``bias`` itself on the rows where ``w = 1 - bias``. Against a 60-digit evaluation of both
series the log density is exact to 4e-15 (absolute below 1, relative above) for ``1e-12 <= w <= 1 - 1e-12`` and ``1e-4 <= u <=
200``, and its gradient to 1e-11 (tests/test_wiener.py). Stan 2.32's ``wiener_lpdf`` truncates the series and is within 1.5e-10 of the exact log density at the
frozen brms points (divergence D150).

The functions take an ``ops`` namespace of elementwise functions, ``NUMPY_OPS`` or ``pytensor_ops()``, so that the model's
density (PyTensor, and so JAX) and ``log_lik()`` (NumPy) are one implementation. Every constant is a float64.

The mean decision time (Wald's identity ``a P(upper) = a w + v E[T]``): ``E[T] = (a P(upper) - a w) / v`` with ``P(upper) =
(1 - exp(-2 v a w)) / (1 - exp(-2 v a))``, and ``a^2 w (1 - w)`` at ``v = 0``. ``mean_decision_time`` writes it as ``E[T] = a w
expm1(L) / v`` with ``L = log(a P(upper) / (a w))`` for ``v > 0``, in forms that subtract no two nearly equal numbers at any
``v``, and the reflection ``E[T](v, a, w) = E[T](-v, a, 1 - w)`` for ``v < 0``. ``E[rt] = ndt + E[T]``; brms's
``posterior_epred`` is not this mean (divergence D148).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

U_SWITCH = np.float64(0.5)        # the small-time series at u <= U_SWITCH, the large-time one above
SMALL_TIME_TERMS = 4              # k = 1..4 on either side of the pair at 0 (w <= 1/2) or at 1 (w > 1/2)
LARGE_TIME_TERMS = 5              # k = 1..5
_PI = np.float64(math.pi)
_PI2_HALF = np.float64(math.pi ** 2 / 2.0)
_LOG_PI = np.float64(math.log(math.pi))
_HALF_LOG_2PI = np.float64(0.5 * math.log(2.0 * math.pi))
_ONE, _TWO, _HALF, _ONE_AND_HALF, _ZERO = (np.float64(1.0), np.float64(2.0), np.float64(0.5), np.float64(1.5), np.float64(0.0))
_NEG_INF = np.float64(-np.inf)
# sinh(x) / x - 1 = sum_{k >= 1} x^(2k) / (2k + 1)!: its coefficients for k = 1..9, every term positive (below x = 1 the omitted
# ones are under 1e-17 of the sum)
_SINHC_SERIES = tuple(np.float64(1.0 / math.factorial(2 * k + 1)) for k in range(1, 10))

NUMPY_OPS = SimpleNamespace(exp=np.exp, expm1=np.expm1, log=np.log, log1p=np.log1p, sin=np.sin, where=np.where)


def pytensor_ops() -> SimpleNamespace:
    """The PyTensor elementwise functions of ``NUMPY_OPS`` (``where`` is ``pt.switch``), which PyMC's JAX backend lowers."""
    import pytensor.tensor as pt

    return SimpleNamespace(exp=pt.exp, expm1=pt.expm1, log=pt.log, log1p=pt.log1p, sin=pt.sin, where=pt.switch)


def _small_time_sum(u, w, e, ops):
    """``sum_k (w + 2k) exp(-2k (w + k) / u)``, the small-time series divided by its k = 0 term's ``exp(-w^2 / (2u))``, its
    terms paired around 0 (``w <= 1/2``) or around 1 (``w > 1/2``); ``e = 1 - w``."""
    low = w                                                   # k = 0, then k with -k
    for k in range(1, SMALL_TIME_TERMS + 1):
        a = ops.exp(-_TWO * k * (k + w) / u)                  # term k's exponential
        b = ops.exp(-_TWO * k * (k - w) / u)                  # term -k's
        low = low + w * (a + b) + _TWO * k * b * ops.expm1(-np.float64(4.0 * k) * w / u)
    high = _ZERO                                              # k with -(k + 1), k = 0..: (2k + w) and -(2k + 2 - w) = -(2k + 1 + e)
    for k in range(0, SMALL_TIME_TERMS + 1):
        m = np.float64(2 * k + 1)
        a = _ONE if k == 0 else ops.exp(-_TWO * k * (k + w) / u)
        high = high + a * (-(np.float64(2 * k) + w) * ops.expm1(-_TWO * m * e / u) - _TWO * e * ops.exp(-_TWO * m * e / u))
    return ops.where(w <= _HALF, low, high)


def _large_time_sum(u, w, e, ops):
    """``sum_{k >= 1} k exp(-(k^2 - 1) pi^2 u / 2) sin(k pi w)``, the large-time series divided by ``pi exp(-pi^2 u / 2)``, with
    ``sin(k pi w) = (-1)^(k + 1) sin(k pi e)`` for ``w > 1/2``; ``e = 1 - w``."""
    flip = w > _HALF
    r = ops.where(flip, e, w)
    total = _ZERO
    for k in range(1, LARGE_TIME_TERMS + 1):
        term = np.float64(k) * ops.exp(-np.float64(k * k - 1) * _PI2_HALF * u) * ops.sin(np.float64(k) * _PI * r)
        total = total + (term if k % 2 else ops.where(flip, -term, term))
    return total


def log_standard_density(u, w, e, ops=NUMPY_OPS):
    """``log f(u | 0, 1, w)``: the log density of the standard process (no drift, boundaries 0 and 1) reaching the lower
    boundary at time ``u > 0`` from ``w`` in (0, 1), ``e = 1 - w`` given beside it. Each series is evaluated at its own side
    of ``U_SWITCH`` (the other at the switch itself), so that the branch not taken is finite with a zero gradient."""
    small = u <= U_SWITCH
    us = ops.where(small, u, U_SWITCH)
    ul = ops.where(small, U_SWITCH, u)
    log_small = -_HALF_LOG_2PI - _ONE_AND_HALF * ops.log(us) - w * w / (_TWO * us) + ops.log(_small_time_sum(us, w, e, ops))
    log_large = _LOG_PI - _PI2_HALF * ul + ops.log(_large_time_sum(ul, w, e, ops))
    return ops.where(small, log_small, log_large)


def log_lower_density(t, v, a, w, e, ops=NUMPY_OPS):
    """``log f(t | v, a, w)`` of reaching the lower boundary at decision time ``t`` (drift ``v``, boundary separation ``a``,
    relative start ``w``, ``e = 1 - w``): ``-v a w - v^2 t / 2 - 2 log a + log f(t / a^2 | 0, 1, w)``; ``-inf`` for ``t <= 0``."""
    positive = t > _ZERO
    ts = ops.where(positive, t, a * a)                         # the standard density at u = 1 where t <= 0, then discarded
    value = -v * a * w - v * v * ts / _TWO - _TWO * ops.log(a) + log_standard_density(ts / (a * a), w, e, ops)
    return ops.where(positive, value, _NEG_INF)


def wiener_log_density(rt, upper, mu, bs, ndt, bias, ops=NUMPY_OPS):
    """brms's ``wiener()`` log density of each response time ``rt`` with its decision ``upper`` (brms's ``dec = 1``; a boolean
    array): Stan's ``wiener_lpdf(rt | bs, ndt, bias, mu)`` for ``dec = 1``, ``wiener_lpdf(rt | bs, ndt, 1 - bias, -mu)`` for
    ``dec = 0``; the drift ``mu``, boundary separation ``bs``, non-decision time ``ndt`` and relative starting point ``bias``
    broadcast against ``rt``."""
    w = ops.where(upper, _ONE - bias, bias)
    e = ops.where(upper, bias, _ONE - bias)
    v = ops.where(upper, -mu, mu)
    return log_lower_density(rt - ndt, v, bs, w, e, ops)


def wiener_logp(value, mu, bs, ndt, bias, upper):
    """The model's density (PyTensor): ``wiener_log_density`` of the observed response times ``value``."""
    return wiener_log_density(value, np.asarray(upper, dtype=bool), mu, bs, ndt, bias, pytensor_ops())


DRAWS_REFUSED = ("draws of the wiener family's response (first-passage times and decisions) are not offered: brms's "
                 "posterior_predict needs the R package RWiener, and no sampler of the Wiener first-passage time is implemented "
                 "here (docs/wiener.md, divergence D149)")


def _refuse_draws(*args, **kwargs):
    raise ValueError(DRAWS_REFUSED)


def wiener_family(upper):
    """Bambi family of brms's ``wiener()``: the identity link on the drift ``mu`` and the parameters ``bs``, ``ndt`` and
    ``bias`` with their priors (brms's links ``log``, ``log`` and ``logit`` where one has its own formula: ``bs``'s here);
    ``upper`` is each fitted row's decision (brms's ``dec = 1``). The likelihood draws nothing (``DRAWS_REFUSED``)."""
    from autobayes.likelihoods import _mean_family

    up = np.asarray(upper, dtype=bool)

    def logp(value, mu, bs, ndt, bias):
        return wiener_logp(value, mu, bs, ndt, bias, up)

    return _mean_family("wiener", "Wiener", ("mu", "bs", "ndt", "bias"),
                        {"mu": "identity", "bs": "log", "ndt": "log", "bias": "logit"}, logp, _refuse_draws)


def _sinhc_series(x2):
    """``sinh(x) / x - 1`` from ``x^2`` by its positive series (Horner, k = 1..9), exact below ``x = 1``."""
    series = 0.0
    for c in reversed(_SINHC_SERIES):
        series = (series + c) * x2
    return series


def log_sinhc(x):
    """``log(sinh(x) / x)`` for ``x >= 0`` (NumPy; 0 at 0): ``log1p`` of the positive series of ``sinh(x) / x - 1`` below 1, and
    ``x + log(-expm1(-2x)) - log(2x)`` at and above, which does not overflow."""
    x = np.asarray(x, dtype=float)
    small = x < 1.0
    xs = np.where(small, x, 0.5)
    xl = np.where(small, 1.0, x)
    return np.where(small, np.log1p(_sinhc_series(xs * xs)), xl + np.log(-np.expm1(-2.0 * xl)) - np.log(2.0 * xl))


def _log_sinhc_drop(y, e):
    """``log_sinhc(y (1 - e)) - log_sinhc(y)`` for ``0 < y < 2`` and ``0 <= e <= 1/2`` without subtracting the two: below
    ``y = 1`` the series' difference ``sum_k c_k y^(2k) expm1(2k log1p(-e))``, every term of one sign; from 1 the ratio
    ``sinh(y - d) / sinh(y) = 1 + 2 sinh(d / 2)^2 - sinh(d) / tanh(y)`` with ``d = y e``, less ``log1p(-e)``."""
    small = y < 1.0
    ys = np.where(small, y, 0.5)
    y2, l1e = ys * ys, np.log1p(-e)
    drop, power = 0.0, 1.0
    for k, c in enumerate(_SINHC_SERIES, start=1):
        power = power * y2
        drop = drop + c * power * np.expm1(2.0 * k * l1e)
    yl = np.where(small, 1.5, y)
    d = yl * e
    ratio = 2.0 * np.sinh(0.5 * d) ** 2 - np.sinh(d) / np.tanh(yl)
    return np.where(small, np.log1p(drop / (1.0 + _sinhc_series(y2))), np.log1p(ratio) - l1e)


def mean_decision_time(v, a, w):
    """``E[T]``, the mean first-passage time of a Wiener process with drift ``v``, unit diffusion, boundaries 0 and ``a`` and
    start ``z = a w`` (NumPy, broadcast). For ``v > 0`` it is ``z expm1(L) / v`` with ``L = log(a P(upper) / z)``, which is
    ``v a (1 - w) + log_sinhc(v z) - log_sinhc(v a)`` (``P(upper) = exp(v (a - z)) sinh(v z) / sinh(v a)``), in the form that
    subtracts no two nearly equal numbers: below ``v a = 2`` that sum, its last two terms as ``_log_sinhc_drop`` for ``w > 1/2``;
    from ``v a = 2``, where ``log_sinhc(v a)`` is close to ``v a``, ``log(2 v a) - v z - log(-expm1(-2 v a)) + log_sinhc(v z)``
    for ``v z < 1`` and ``-log1p(-(1 - w)) + log1p(exp(-2 v z) expm1(-2 v a (1 - w)) / -expm1(-2 v a))`` otherwise. For
    ``v < 0`` the reflection ``E[T](v, a, w) = E[T](-v, a, 1 - w)``, and ``a^2 w (1 - w)`` at ``v = 0``. Within 3e-15 of a
    60-digit evaluation of ``(a P(upper) - z) / v`` for drifts from -400 to 400 and w from 1e-9 to 1 - 1e-9
    (tests/test_wiener.py)."""
    v, a, w = np.broadcast_arrays(np.asarray(v, dtype=float), np.asarray(a, dtype=float), np.asarray(w, dtype=float))
    neg = v < 0
    s = np.abs(v)
    ww, ee = np.where(neg, 1.0 - w, w), np.where(neg, w, 1.0 - w)
    z = a * ww
    sa, sz = s * a, s * z
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        drop = np.where(ww > 0.5, _log_sinhc_drop(np.minimum(sa, 1.999), np.minimum(ee, 0.5)), log_sinhc(sz) - log_sinhc(sa))
        near = sa * ee + drop                                                                   # v a < 2
        wide_low = np.log(2.0 * sa) - sz - np.log(-np.expm1(-2.0 * sa)) + log_sinhc(sz)        # v a >= 2, v z < 1
        wide_high = -np.log1p(-ee) + np.log1p(np.exp(-2.0 * sz) * np.expm1(-2.0 * sa * ee) / -np.expm1(-2.0 * sa))   # v z >= 1
        lv = np.where(sa < 2.0, near, np.where(sz < 1.0, wide_low, wide_high))
        moving = z * np.expm1(lv) / s
    return np.where(s > 0, moving, z * a * ee)


def expected_response_time(mu, bs, ndt, bias):
    """``E[rt] = ndt + E[T]`` (``mean_decision_time(mu, bs, bias)``), the mean response time over both boundaries."""
    return np.asarray(ndt, dtype=float) + mean_decision_time(mu, bs, bias)


def brms_epred(mu, bs, ndt, bias):
    """brms 2.23's ``posterior_epred`` of a ``wiener()`` model, ``ndt - bias / mu + bs / mu (exp(-2 mu bias) - 1) / (exp(-2 mu
    bs) - 1)`` (the case notes of the frozen fixtures): ``E[rt]`` with the relative ``bias`` where the absolute starting point
    ``bs * bias`` belongs, equal to it only for ``bs = 1``. Recorded for the comparison with brms (divergence D148), not used
    by ``predict()``."""
    mu, bs, ndt, bias = (np.asarray(x, dtype=float) for x in (mu, bs, ndt, bias))
    return ndt - bias / mu + bs / mu * np.expm1(-2.0 * mu * bias) / np.expm1(-2.0 * mu * bs)
