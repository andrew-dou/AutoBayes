#!/usr/bin/env python
"""High-precision references for the brms oracle cases whose brms side is inexact at some frozen points (P6).

    python oracle/reference.py <case-id> [<case-id> ...]          # write tests/oracle/references/<case-id>.json
    python oracle/reference.py --check <case-id> [...] | --all    # recompute and compare with the committed file

A case that opts in to two-reference acceptance (its ``gate`` field ``{"expect": "pass", "reference": "<case-id>.json",
"divergence": ..., "fallback": [...]}``, docs/brms-parity/LOOP.md) is compared with brms wherever brms is accurate and
with this reference where brms is not. The reference is an independent third implementation: it uses the standard
library only (nothing from autobayes or tests/), reads the frozen fixture's ``case_brms.json``, ``data.csv``,
``newdata.csv`` and the seeded points' unconstrained values in ``fixture.json``, applies brms's constraining transforms
and their Jacobians itself (identity for ``b``, ``Intercept`` and ``z_1``; ``exp`` with log-Jacobian ``u`` for ``sigma``
and ``sd_1``; Stan's ``ordered`` transform of the cumulative model's thresholds, ``Intercept[1] = u_1`` and
``Intercept[k] = Intercept[k - 1] + exp(u_k)`` with log-Jacobian ``u_2 + ... + u_K``; Stan's lower-upper transform of the
wiener family's ``ndt`` (upper bound ``min_Y``, the smallest response time) and ``bias`` (upper bound 1), ``ub inv_logit(u)``
with log-Jacobian ``log(ub) + log inv_logit(u) + log(1 - inv_logit(u))``; as the fixture's probed ``upars`` table and its
declarations say), and evaluates the model's log density in Python's decimal arithmetic at 50 significant digits. The models, in
brms's parameterisation (``eta = Intercept + b (x - mean(x)) + sd_1 z_1[g]``, the separate centred intercept and
non-centred group effects of a varying intercept on ``g``; the cumulative model's ``eta`` has no intercept, its
thresholds ``c_1 < ... < c_K`` are brms's ``Intercept``, and ``c_0 = -inf``, ``c_{K+1} = inf``):

    bernoulli, cloglog   yb ~ x + (1 | g)                     yb log(1 - exp(-exp(eta))) - (1 - yb) exp(eta)
    binomial, cloglog    k | trials(size) ~ x + (1 | g)       log C(size, k) + k log(1 - exp(-exp(eta))) - (size - k) exp(eta)
    gaussian             y ~ x + (1 | g)                      the log density of y given eta and sigma
    gaussian, censored   y | cens(cens_ind) ~ x + (1 | g)     by code 0 / 1 / -1: log of the density, of 1 - Phi or of Phi
                                                              of (y - eta) / sigma (the density's -log sigma included)
    gaussian, truncated  ypos | trunc(lb = 0) ~ x + (1 | g)   the density minus log(1 - Phi((lb - eta) / sigma))
    negbinomial, log, truncated
                         ytr | trunc(lb = 1) ~ x + (1 | g)    with mu = exp(eta) and e = shape log(1 + mu / shape): log Gamma(y +
                                                              shape) - log Gamma(y + 1) - log Gamma(shape) + y (eta - log(mu +
                                                              shape)) - e, the log mass of Stan's neg_binomial_2 (its log Gamma
                                                              terms as the log of prod_{j < y} (shape + j) / (j + 1)), minus
                                                              log P(Y >= 1) = log(1 - exp(-e)), P(Y = 0) = exp(-e) (``negbinomial_lpmf``)
    cumulative, probit   ord ~ x + (1 | g)                    log(Phi(c_ord - eta) - Phi(c_{ord-1} - eta)), ord in 1..K+1,
                                                              the difference taken in the tail it lies in (``log_normal_interval``)
    Gamma, log, censored ypos | cens(cens_i, ypos2) ~ x + (1 | g)
                                                              with mu = exp(eta), rate = shape / mu and t = rate y: by code 0 / 1
                                                              / -1 / 2 the log density, log Q(shape, t), log P(shape, t) and
                                                              log(P(shape, t2) - P(shape, t)), P and Q the regularized incomplete
                                                              gamma functions (``log_gamma_interval``)
    wiener               rt | dec(dec) ~ x + (1 | g), and with bs ~ z (bs = exp(Intercept_bs + Xc_bs b_bs), Xc_bs the dummies of z
                                                              centred at the fitted rows' means): Stan's wiener_lpdf(rt | bs, ndt,
                                                              bias, mu) for dec = 1, wiener_lpdf(rt | bs, ndt, 1 - bias, -mu) for
                                                              dec = 0, mu = eta; the density of the lower boundary f(t | v, a, w) =
                                                              a^-2 exp(-v a w - v^2 t / 2) f(t / a^2 | 0, 1, w) at t = rt - ndt (the
                                                              upper boundary's with (-v, 1 - w)), f(u | 0, 1, w) by Navarro and
                                                              Fuss's small-time series for u <= 1 and their large-time series above
                                                              (``log_first_passage``)

with brms's priors: ``normal(m, s)`` on ``Intercept`` and ``b`` (``normal_lpdf``), ``normal(m, s)`` on ``sd_1`` and ``sigma``,
which are declared ``<lower=0>`` (``normal_lpdf(. | m, s) - normal_lccdf(0 | m, s)``), ``gamma(a, b)`` on the Gamma and the
negative binomial families' ``shape`` (``gamma_lpdf``; its support is the declared one, so no truncation term), and
``std_normal`` on ``z_1``; for the
wiener family ``gamma(a, b)`` on ``bs`` or ``normal(m, s)`` on its formula's ``Intercept_bs`` and ``b_bs``, ``normal(m, s)`` on
``ndt`` truncated to (0, min_Y) (``normal_lpdf - log(Phi((min_Y - m) / s) - Phi(-m / s))``, brms's ``log_diff_exp`` of two
``normal_lcdf``, checked against brms's ``lprior`` at every point) and ``beta(a, b)`` on ``bias``. Every
normalising constant is included, as ``rstan::log_prob`` includes them. The inputs are the doubles brms and AutoBayes
evaluate, converted to decimal exactly: the CSV's numbers parsed as binary64 (the generator rounds them to four decimals
so that R and Python parse the same double) and the fixture's ``u``; everything from there on, the centring of ``x``
included, is exact to 50 digits.

Per seeded point the file holds the log density with and without the Jacobian, the gradient with and without it in
brms's unconstrained order (without: a fourth-order central difference of the 50-digit density with step 1e-10, whose
truncation and rounding errors are both below 1e-35; with: plus 1 in each log-transformed coordinate and in each
threshold coordinate after the first, the derivative of its log-Jacobian ``u``, and ``1 - 2 inv_logit(u)`` in each lower-upper
coordinate), and, for the truncated model, every expected value gate G4 compares: ``posterior_epred``'s
truncated mean ``eta + sigma phi(a) / (1 - Phi(a))``, ``a = (lb - eta) / sigma``, on the fitted rows and on ``newdata``
(``x`` centred at the fitted mean), with and without the group effects; for the wiener family the same blocks of the mean
response time ``ndt + (bs P_up - bs bias) / mu``, ``P_up = (1 - exp(-2 mu bs bias)) / (1 - exp(-2 mu bs))``, and ``ndt + bs^2
bias (1 - bias)`` at ``mu = 0`` (``wiener_mean``; brms's own posterior_epred is not this mean, divergence D148); for the
truncated negative binomial the same blocks of the mean of the distribution its likelihood truncates, ``E[Y | Y >= 1] = mu /
(1 - exp(-e))`` (``count_truncated_mean``; brms's own posterior_epred is the mean above 2, divergence D121).

A case whose fixture records loo's ``loo_model_weights()`` (its ``post`` block) also gets the exact stacking weights of the
two models it compares (``stacking_weight``; Yao, Vehtari, Simpson and Gelman 2018, eq. 4): the maximiser over the simplex
of ``F(w) = sum_i log(w a_i + (1 - w) b_i)``, with ``a_i`` and ``b_i`` the exponentials of the two models' pointwise
``elpd_loo`` (the fixture's ``loo_compare.elpd_loo``, the matrix brms hands loo) less their larger one. ``F`` is concave and
its derivative ``F'(w) = sum_i (a_i - b_i) / (w a_i + (1 - w) b_i)`` decreases in ``w``; the Karush-Kuhn-Tucker conditions on
the simplex are ``w = 0`` where ``F'(0) <= 0``, ``w = 1`` where ``F'(1) >= 0``, and otherwise ``F'(w) = 0``, whose root is
bisected in 65-digit arithmetic to an interval of width 1e-60: the weight is then known to within the larger of that width
and the 65-digit sum's rounding error over ``|F''|``, far below the 30 digits written. The file holds the models' names and
both weights (the first ``w``, the second ``1 - w``). It is independent of AutoBayes's ``stacking_weights`` (an active-set
Newton method in double precision).

Values are written as strings with 30
significant digits, with the sha256 of every file read. Erf and erfc: the series of positive terms for ``|x| < 3`` (with
15 guard digits) and the Laplace continued fraction above; the gradient's step and the guard digits are constants here.
Re-running the script reproduces a committed file byte for byte (``--check``); a committed reference is never
overwritten (a changed fixture would be a new case id).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from decimal import ROUND_HALF_EVEN, Context, Decimal, getcontext, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "oracle" / "fixtures"
REFERENCES = REPO / "tests" / "oracle" / "references"

PREC = 50                       # significant digits of the arithmetic
GUARD = 15                      # further digits inside erf, erfc, pi and the constants
DIGITS = 30                     # significant digits written
STEP = Decimal("1e-10")         # step of the fourth-order central difference
CF_FROM = Decimal(3)            # erfc by the continued fraction from here on, by the series below
CONTEXT = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)

MODELS = {   # the brms formulas this script implements -> (family, link); Case.row_term is the likelihood of each
    "yb ~ x + (1 | g)": ("bernoulli", "cloglog"),
    "k | trials(size) ~ x + (1 | g)": ("binomial", "cloglog"),
    "y | cens(cens_ind) ~ x + (1 | g)": ("gaussian", "identity"),
    "ypos | trunc(lb = 0) ~ x + (1 | g)": ("gaussian", "identity"),
    "ord ~ x + (1 | g)": ("cumulative", "probit"),
    "ypos | cens(cens_i, ypos2) ~ x + (1 | g)": ("Gamma", "log"),
    "rt | dec(dec) ~ x + (1 | g)": ("wiener", "identity"),
    "rt | dec(dec) ~ x + (1 | g), bs ~ z": ("wiener", "identity"),       # a list of formulas, joined with ", "
    "y ~ x + (1 | g)": ("gaussian", "identity"),
    "ytr | trunc(lb = 1) ~ x + (1 | g)": ("negbinomial", "log"),
}
DEFAULT_LINK = {"negbinomial": "log"}   # brms's default link of a family whose case names none (otherwise identity)
SHAPED = ("Gamma", "negbinomial")       # the families with a shape parameter, gamma(a, b) on it, log-transformed
BISECT_WIDTH = Decimal("1e-60")         # the stacking weight's bisection stops at an interval this wide
_GAMMA = re.compile(r"gamma\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)")
_NORMAL = re.compile(r"normal\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)")
_BETA = re.compile(r"beta\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)")


# -- 50-digit special functions ---------------------------------------------------------------------------------------
def _guarded():
    ctx = CONTEXT.copy()
    ctx.prec = PREC + GUARD
    return localcontext(ctx)


@lru_cache(maxsize=None)
def _pi() -> Decimal:
    """pi from Machin's formula 16 atan(1/5) - 4 atan(1/239), at PREC + 2 GUARD digits."""
    ctx = CONTEXT.copy()
    ctx.prec = PREC + 2 * GUARD
    with localcontext(ctx):
        eps = Decimal(10) ** -(ctx.prec + 2)

        def atan_inv(n: int) -> Decimal:
            power, total, k = Decimal(1) / n, Decimal(0), 0
            while power > eps:
                total += power / (2 * k + 1) * (-1 if k % 2 else 1)
                power /= n * n
                k += 1
            return total
        return 16 * atan_inv(5) - 4 * atan_inv(239)


def _erf_series(x: Decimal) -> Decimal:
    """erf(x) for 0 <= x < 3 in the caller's context: 2 / sqrt(pi) exp(-x^2) sum_n x (2 x^2)^n / (1 3 5 ... (2n + 1)), a sum
    of positive terms (no cancellation inside it)."""
    two_x2 = 2 * x * x
    term = total = x
    n = 0
    eps = Decimal(10) ** -(getcontext().prec + 2)
    while True:
        n += 1
        term = term * two_x2 / (2 * n + 1)
        total += term
        if n > two_x2 and term <= eps * total:
            break
    return 2 / _pi().sqrt() * (-(x * x)).exp() * total


def _erfc_cf(x: Decimal) -> Decimal:
    """erfc(x) for x >= 3 in the caller's context: exp(-x^2) / (sqrt(pi) K) with Laplace's continued fraction K = x + (1/2) /
    (x + (2/2) / (x + (3/2) / (x + ...))) (Abramowitz and Stegun 7.1.14), by the modified Lentz method; every partial
    numerator and denominator is positive."""
    eps = Decimal(10) ** -(getcontext().prec + 2)
    f = c = x
    d = Decimal(0)
    n = 0
    while True:
        n += 1
        a = Decimal(n) / 2
        d = 1 / (x + a * d)
        c = x + a / c
        delta = c * d
        f *= delta
        if abs(delta - 1) <= eps:
            break
    return (-(x * x)).exp() / (_pi().sqrt() * f)


def _erfc(x: Decimal) -> Decimal:
    """erfc(x) for every x, relative to its value, in the caller's (guarded) context."""
    if x >= CF_FROM:
        return _erfc_cf(x)
    if x >= 0:
        return 1 - _erf_series(x)              # erfc(3) = 2.2e-5: at most 5 of the guard digits are lost
    if x > -CF_FROM:
        return 1 + _erf_series(-x)
    return 2 - _erfc_cf(-x)


def log_cdf(t: Decimal) -> Decimal:
    """log Phi(t), the standard Normal log-CDF, as log(erfc(-t / sqrt 2) / 2)."""
    with _guarded():
        r = (_erfc(-t / Decimal(2).sqrt()) / 2).ln()
    with localcontext(CONTEXT):
        return +r


def log_ccdf(t: Decimal) -> Decimal:
    """log(1 - Phi(t)) as log(erfc(t / sqrt 2) / 2)."""
    with _guarded():
        r = (_erfc(t / Decimal(2).sqrt()) / 2).ln()
    with localcontext(CONTEXT):
        return +r


def log_normal_interval(a: Decimal | None, b: Decimal | None) -> Decimal:
    """log(Phi(b) - Phi(a)) for a < b, ``None`` standing for -inf (a) or inf (b): in the upper tail (a >= 0) as
    (erfc(a / sqrt 2) - erfc(b / sqrt 2)) / 2, in the lower tail (b <= 0) as (erfc(-b / sqrt 2) - erfc(-a / sqrt 2)) / 2,
    and across 0 as 1 - erfc(b / sqrt 2) / 2 - erfc(-a / sqrt 2) / 2, each a difference of terms of the size of the result
    or smaller; when the two terms of a difference agree in their leading digits, it is recomputed with as many further
    digits as they cancel."""
    if a is None and b is None:
        return Decimal(0)
    if a is None:
        return log_cdf(b)
    if b is None:
        return log_ccdf(a)
    if not a < b:
        raise ValueError(f"log_normal_interval needs a < b, got {a} and {b}")
    extra = 0
    while True:
        ctx = CONTEXT.copy()
        ctx.prec = PREC + GUARD + extra
        with localcontext(ctx):
            root2 = Decimal(2).sqrt()
            if a >= 0:
                hi, lo = _erfc(a / root2) / 2, _erfc(b / root2) / 2
            elif b <= 0:
                hi, lo = _erfc(-b / root2) / 2, _erfc(-a / root2) / 2
            else:
                hi, lo = 1 - _erfc(-a / root2) / 2, _erfc(b / root2) / 2
            diff = hi - lo
            lost = max(0, hi.adjusted() - diff.adjusted())
            if lost <= GUARD // 2 + extra:
                r = diff.ln()
                break
        extra = lost + GUARD // 2
    with localcontext(CONTEXT):
        return +r


@lru_cache(maxsize=None)
def _bernoulli_even(k: int) -> Fraction:
    """The Bernoulli number B_{2k} (B_2 = 1/6, B_4 = -1/30, ...), exactly, from the recurrence sum_j C(m + 1, j) B_j = 0."""
    B = [Fraction(1)]
    for m in range(1, 2 * k + 1):
        B.append(-sum(math.comb(m + 1, j) * B[j] for j in range(m)) / (m + 1))
    return B[2 * k]


def _log_gamma(a: Decimal) -> Decimal:
    """log Gamma(a) for a > 0 in the caller's (guarded) context: Stirling's series at z = a + N >= 60 with 30 terms (the next
    is below 1e-70 there), less sum_{j < N} log(a + j)."""
    if not a > 0:
        raise ValueError(f"log Gamma needs a positive argument, got {a}")
    N = max(0, 60 - int(a))
    z = a + N
    total = (z - Decimal("0.5")) * z.ln() - z + (2 * _pi()).ln() / 2
    power = z
    for k in range(1, 31):
        B = _bernoulli_even(k)
        total += Decimal(B.numerator) / Decimal(B.denominator) / (2 * k * (2 * k - 1) * power)
        power *= z * z
    for j in range(N):
        total -= (a + j).ln()
    return total


def _gamma_p_series(a: Decimal, x: Decimal) -> Decimal:
    """P(a, x) = x^a e^-x / Gamma(a + 1) sum_n x^n / ((a + 1) ... (a + n)), a sum of positive terms (for x < a + 1)."""
    eps = Decimal(10) ** -(getcontext().prec + 2)
    term = total = Decimal(1)
    n = 0
    while True:
        n += 1
        term = term * x / (a + n)
        total += term
        if term <= eps * total:
            break
    return (a * x.ln() - x - _log_gamma(a + 1)).exp() * total


def _gamma_q_cf(a: Decimal, x: Decimal) -> Decimal:
    """Q(a, x) = x^a e^-x / Gamma(a) / (x + 1 - a - 1 (1 - a) / (x + 3 - a - 2 (2 - a) / (x + 5 - a - ...))), Legendre's continued
    fraction, by the modified Lentz method (for x >= a + 1)."""
    eps = Decimal(10) ** -(getcontext().prec + 2)
    tiny = Decimal(10) ** -(getcontext().prec * 2)
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    i = 0
    while True:
        i += 1
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) <= eps:
            break
    return (a * x.ln() - x - _log_gamma(a)).exp() * h


def _gamma_p_q(a: Decimal, x: Decimal) -> tuple[Decimal, Decimal]:
    """(P(a, x), Q(a, x)), the regularized incomplete gamma functions, the smaller of the two computed directly."""
    if x < a + 1:
        p = _gamma_p_series(a, x)
        return p, 1 - p
    q = _gamma_q_cf(a, x)
    return 1 - q, q


def log_gamma_interval(a: Decimal, x1: Decimal | None, x2: Decimal | None) -> Decimal:
    """log(P(a, x2) - P(a, x1)) for 0 <= x1 < x2, ``None`` standing for 0 (x1) or infinity (x2): log P, log Q, or the
    difference taken in the tail both lie in (Q(x1) - Q(x2) above a + 1, P(x2) - P(x1) below), recomputed with as many
    further digits as the two terms cancel."""
    extra = 0
    while True:
        ctx = CONTEXT.copy()
        ctx.prec = PREC + GUARD + extra
        with localcontext(ctx):
            if x1 is None and x2 is None:
                return Decimal(0)
            if x1 is None:
                r = _gamma_p_q(a, x2)[0].ln()
                break
            if x2 is None:
                r = _gamma_p_q(a, x1)[1].ln()
                break
            (p1, q1), (p2, q2) = _gamma_p_q(a, x1), _gamma_p_q(a, x2)
            hi, lo = (q1, q2) if x1 >= a + 1 else (p2, p1)
            diff = hi - lo
            lost = max(0, hi.adjusted() - diff.adjusted())
            if lost <= GUARD // 2 + extra:
                r = diff.ln()
                break
        extra = lost + GUARD // 2
    with localcontext(CONTEXT):
        return +r


def gamma_lpdf(v: Decimal, a: Decimal, rate: Decimal) -> Decimal:
    """log Gamma(v | a, rate) = a log(rate) - log Gamma(a) + (a - 1) log v - rate v."""
    with _guarded():
        r = a * rate.ln() - _log_gamma(a) + (a - 1) * v.ln() - rate * v
    with localcontext(CONTEXT):
        return +r


def inverse_mills(t: Decimal) -> Decimal:
    """phi(t) / (1 - Phi(t)), the standard Normal density over its upper tail mass."""
    with _guarded():
        r = (-(t * t) / 2).exp() / (2 * _pi()).sqrt() / (_erfc(t / Decimal(2).sqrt()) / 2)
    with localcontext(CONTEXT):
        return +r


@lru_cache(maxsize=None)
def _half_log_two_pi() -> Decimal:
    with _guarded():
        r = (2 * _pi()).ln() / 2
    with localcontext(CONTEXT):
        return +r


def normal_lpdf(v: Decimal, m: Decimal, s: Decimal) -> Decimal:
    z = (v - m) / s
    return -s.ln() - _half_log_two_pi() - z * z / 2


def log1m_exp_neg(e: Decimal) -> Decimal:
    """log(1 - exp(-e)) for e > 0, with as many further digits as 1 - exp(-e) cancels (the cloglog link's log P(y = 1))."""
    ctx = CONTEXT.copy()
    ctx.prec = PREC + GUARD + max(0, -e.adjusted())
    with localcontext(ctx):
        r = (1 - (-e).exp()).ln()
    with localcontext(CONTEXT):
        return +r


def _negbinomial_e(eta: Decimal, shape: Decimal) -> Decimal:
    """e = shape log(1 + mu / shape), mu = exp(eta), in the caller's (guarded) context: P(Y = 0) = (shape / (mu + shape))^shape =
    exp(-e); log(1 + x) with as many further digits as 1 + x loses of a small x."""
    x = eta.exp() / shape
    ctx = getcontext().copy()
    ctx.prec += max(0, -x.adjusted())
    with localcontext(ctx):
        r = shape * (1 + x).ln()
    return +r


def negbinomial_lpmf(y: int, eta: Decimal, shape: Decimal) -> Decimal:
    """log NB(y | mu = exp(eta), shape), Stan's neg_binomial_2_log_lpmf: log Gamma(y + shape) - log Gamma(y + 1) - log Gamma(shape)
    + y (eta - log(mu + shape)) - shape log(1 + mu / shape), the three log Gamma as the log of the product of (shape + j) / (j +
    1) over j = 0..y - 1 (Gamma(y + shape) = Gamma(shape) shape (shape + 1) ... (shape + y - 1))."""
    with _guarded():
        coefficient = Decimal(1)
        for j in range(y):
            coefficient *= (shape + j) / (j + 1)
        r = coefficient.ln() + y * (eta - (eta.exp() + shape).ln()) - _negbinomial_e(eta, shape)
    with localcontext(CONTEXT):
        return +r


def negbinomial_log_p_positive(eta: Decimal, shape: Decimal) -> Decimal:
    """log P(Y >= 1) = log(1 - exp(-e)) of the negative binomial, the term trunc(lb = 1) subtracts from each row."""
    with _guarded():
        e = _negbinomial_e(eta, shape)
    return log1m_exp_neg(e)


def negbinomial_positive_mean(eta: Decimal, shape: Decimal) -> Decimal:
    """E[Y | Y >= 1] = mu / (1 - exp(-e)) of the negative binomial (the zeros add nothing to the mean mu), with as many further
    digits as 1 - exp(-e) cancels."""
    with _guarded():
        e = _negbinomial_e(eta, shape)
        mu = eta.exp()
    ctx = CONTEXT.copy()
    ctx.prec = PREC + GUARD + max(0, -e.adjusted())
    with localcontext(ctx):
        r = mu / (1 - (-e).exp())
    with localcontext(CONTEXT):
        return +r


def stacking_weight(first: list[Decimal], second: list[Decimal]) -> Decimal:
    """The stacking weight of the first of two models from their pointwise elpd: the maximiser over [0, 1] of F(w) = sum_i log(w
    a_i + (1 - w) b_i), a_i and b_i the exponentials of the two elpd less their larger one. F'(w) = sum_i (a_i - b_i) / (w a_i
    + (1 - w) b_i) decreases in w, so the Karush-Kuhn-Tucker conditions are w = 0 where F'(0) <= 0, w = 1 where F'(1) >= 0, and
    otherwise F'(w) = 0, whose root is bisected at PREC + GUARD digits to an interval of width ``BISECT_WIDTH``."""
    with _guarded():
        a, b = [], []
        for u, v in zip(first, second):
            top = max(u, v)
            a.append((u - top).exp())
            b.append((v - top).exp())

        def slope(w: Decimal) -> Decimal:
            return sum((x - y) / (w * x + (1 - w) * y) for x, y in zip(a, b))
        if slope(Decimal(0)) <= 0:
            w = Decimal(0)
        elif slope(Decimal(1)) >= 0:
            w = Decimal(1)
        else:
            lo, hi = Decimal(0), Decimal(1)
            while hi - lo > BISECT_WIDTH:
                mid = (lo + hi) / 2
                if slope(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            w = (lo + hi) / 2
    with localcontext(CONTEXT):
        return +w


def _sin(x: Decimal) -> Decimal:
    """sin(x) in the caller's context: x reduced to [-pi, pi] by a multiple of 2 pi (pi at PREC + 2 GUARD digits), then
    its Taylor series, summed until a term is below the context's last digit."""
    two_pi = 2 * _pi()
    x = x - two_pi * (x / two_pi).to_integral_value()
    eps = Decimal(10) ** -(getcontext().prec + 2)
    term = total = x
    n = 1
    while abs(term) > eps:
        term = -term * x * x / ((n + 1) * (n + 2))
        total += term
        n += 2
    return total


def beta_lpdf(v: Decimal, a: Decimal, b: Decimal) -> Decimal:
    """log Beta(v | a, b) = (a - 1) log v + (b - 1) log(1 - v) - log Gamma(a) - log Gamma(b) + log Gamma(a + b)."""
    with _guarded():
        r = (a - 1) * v.ln() + (b - 1) * (1 - v).ln() - _log_gamma(a) - _log_gamma(b) + _log_gamma(a + b)
    with localcontext(CONTEXT):
        return +r


def log_first_passage(u: Decimal, w: Decimal) -> Decimal:
    """log f(u | 0, 1, w), the density of a Wiener process without drift, started at w in (0, 1) between the boundaries 0 and
    1, reaching the lower one at time u (Navarro and Fuss 2009): for u <= 1 the small-time series (2 pi u^3)^(-1/2) sum_k (w +
    2k) exp(-(w + 2k)^2 / (2u)), k = 0, +-1, +-2, ..., above it the large-time series pi sum_{k>=1} k exp(-k^2 pi^2 u / 2)
    sin(k pi w). Terms are added until the next one's envelope (|w + 2k| exp(-(w + 2k)^2 / (2u)) for both signs of k, k
    exp(-k^2 pi^2 u / 2)) is below the context's last digit of the sum; when the terms cancel (the sum's leading digit below
    the largest term's), it is recomputed with as many further digits as they cancel."""
    extra = 0
    while True:
        ctx = CONTEXT.copy()
        ctx.prec = PREC + GUARD + extra
        with localcontext(ctx):
            eps = Decimal(10) ** -(ctx.prec + 2)
            if u <= 1:
                total = biggest = w * (-(w * w) / (2 * u)).exp()
                k = 0
                while True:
                    k += 1
                    up, down = w + 2 * k, w - 2 * k
                    pair = (up * (-(up * up) / (2 * u)).exp(), down * (-(down * down) / (2 * u)).exp())
                    total += pair[0] + pair[1]
                    biggest = max(biggest, abs(pair[0]), abs(pair[1]))
                    if abs(pair[0]) + abs(pair[1]) <= eps * abs(total):
                        break
                log_scale = -(2 * _pi() * u * u * u).ln() / 2
            else:
                pi = _pi()
                total, biggest, k = Decimal(0), Decimal(0), 0
                while True:
                    k += 1
                    envelope = k * (-(k * k) * pi * pi * u / 2).exp()
                    term = envelope * _sin(k * pi * w)
                    total += term
                    biggest = max(biggest, abs(term))
                    if k >= 2 and envelope <= eps * abs(total):
                        break
                log_scale = pi.ln()
            lost = max(0, biggest.adjusted() - total.adjusted())
            if lost <= GUARD // 2 + extra:
                r = total.ln() + log_scale
                break
        extra = lost + GUARD // 2
    with localcontext(CONTEXT):
        return +r


def wiener_lower(t: Decimal, v: Decimal, a: Decimal, w: Decimal) -> Decimal:
    """log f(t | v, a, w) of reaching the lower boundary at decision time t > 0 (drift v, boundary separation a, relative
    start w): -v a w - v^2 t / 2 - 2 log a + log f(t / a^2 | 0, 1, w)."""
    with _guarded():
        head = -v * a * w - v * v * t / 2 - 2 * a.ln()
        u = t / (a * a)
    r = head + log_first_passage(u, w)
    with localcontext(CONTEXT):
        return +r


def wiener_mean(mu: Decimal, bs: Decimal, ndt: Decimal, bias: Decimal) -> Decimal:
    """The mean response time over both boundaries, ndt + (bs P_up - bs bias) / mu with P_up = (1 - exp(-2 mu bs bias)) /
    (1 - exp(-2 mu bs)) (Wald's identity), and ndt + bs^2 bias (1 - bias) at mu = 0; recomputed with as many further digits
    as P_up - bias cancels (as mu -> 0)."""
    if mu == 0:
        with _guarded():
            r = ndt + bs * bs * bias * (1 - bias)
        with localcontext(CONTEXT):
            return +r
    extra = max(0, -(2 * mu * bs).adjusted())
    while True:
        ctx = CONTEXT.copy()
        ctx.prec = PREC + GUARD + extra
        with localcontext(ctx):
            p_up = (1 - (-2 * mu * bs * bias).exp()) / (1 - (-2 * mu * bs).exp())
            diff = p_up - bias
            lost = max(0, max(p_up.adjusted(), bias.adjusted()) - diff.adjusted())
            if lost <= GUARD // 2 + extra:
                r = ndt + bs * diff / mu
                break
        extra = lost + GUARD // 2
    with localcontext(CONTEXT):
        return +r


def _log_inv_logit(v: Decimal) -> Decimal:
    """log(1 / (1 + exp(-v)))."""
    with _guarded():
        r = -(1 + (-v).exp()).ln()
    with localcontext(CONTEXT):
        return +r


# -- the case: data, priors, parameters ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text: str) -> Decimal:
    """A CSV number as the double R and Python parse it to, exactly."""
    return Decimal(float(text))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Case:
    """The frozen fixture of ``case_id`` as this script needs it: the model, the rows (doubles, exactly), the priors and the
    unconstrained points, each checked against what the script implements. Build it, and use it, inside
    ``localcontext(CONTEXT)``."""

    def __init__(self, case_id: str):
        self.id = case_id
        self.dir = FIXTURES / case_id
        self.sources = {name: self.dir / name for name in ("case_brms.json", "data.csv", "newdata.csv", "fixture.json")
                        if (self.dir / name).exists()}
        brms = json.loads((self.dir / "case_brms.json").read_text(encoding="utf-8"))
        self.fixture = json.loads((self.dir / "fixture.json").read_text(encoding="utf-8"))
        self.formula = brms["formula"] if isinstance(brms["formula"], str) else ", ".join(brms["formula"])
        if self.formula not in MODELS:
            raise SystemExit(f"{case_id}: no reference model for brms formula {self.formula!r}; this script implements {sorted(MODELS)}")
        family, link = MODELS[self.formula]
        if (brms["family"].get("family"), brms["family"].get("link", DEFAULT_LINK.get(family, "identity"))) != (family, link):
            raise SystemExit(f"{case_id}: the formula {self.formula!r} is implemented for {family}({link}), not {brms['family']}")
        self.family, self.link = family, link
        lower = re.search(r"\| trunc\(lb = (\d+)\)", self.formula)          # trunc(lb = 0) (Gaussian), trunc(lb = 1) (counts)
        self.lb = Decimal(lower.group(1)) if lower else None
        self.censored = "| cens(" in self.formula
        self.levels = list(brms["factors"]["g"])
        self.bs_formula = self.formula.endswith(", bs ~ z")      # wiener's bs ~ z: bs = exp(Intercept_bs + Xc_bs b_bs)
        if [grp["levels"] for grp in self.fixture["groups"]] != [self.levels]:
            raise SystemExit(f"{case_id}: brms's group table {self.fixture['groups']} is not one varying intercept on g with the levels {self.levels}")
        self.priors = self._priors(brms["priors"])
        self.upars = [(e["par"], int(e["elem"]), e["transform"]) for e in self.fixture["upars"]]
        if family == "cumulative":     # the ordered thresholds are one transform of the whole vector, which the probe calls "joint"
            self.nthres = int(self.fixture["standata"]["nthres"]["values"][0])
            expected = [("b", "identity")] + [("Intercept", "joint")] * self.nthres + [("sd_1", "log")] \
                + [("z_1", "identity")] * len(self.levels)
            if [(par, transform) for par, _, transform in self.upars] != expected:
                raise SystemExit(f"{case_id}: brms's unconstrained coordinates are {self.upars}; this script implements {expected}")
        elif family == "wiener":       # bs (log) or its formula's b_bs and Intercept_bs; ndt and bias Stan's lower-upper ("other")
            middle = [("b_bs", 1, "identity"), ("b_bs", 2, "identity"), ("Intercept_bs", 1, "identity")] if self.bs_formula \
                else [("bs", 1, "log")]
            expected = [("b", 1, "identity"), ("Intercept", 1, "identity")] + middle + [("ndt", 1, "other"), ("bias", 1, "other")] \
                + [("sd_1", 1, "log")] + [("z_1", j, "identity") for j in range(1, len(self.levels) + 1)]
            if self.upars != expected:
                raise SystemExit(f"{case_id}: brms's unconstrained coordinates are {self.upars}; this script implements {expected}")
            declared = self.fixture["declarations"]
            if (declared["ndt"], declared["bias"]) != ("real<lower=0,upper=min_Y> ndt;", "real<lower=0,upper=1> bias;"):
                raise SystemExit(f"{case_id}: brms declares {declared['ndt']!r} and {declared['bias']!r}")
        else:
            expected = [("b", 1, "identity"), ("Intercept", 1, "identity")] + ([("sigma", 1, "log")] if family == "gaussian" else []) \
                + ([("shape", 1, "log")] if family in SHAPED else []) \
                + [("sd_1", 1, "log")] + [("z_1", j, "identity") for j in range(1, len(self.levels) + 1)]
            if self.upars != expected:
                raise SystemExit(f"{case_id}: brms's unconstrained coordinates are {self.upars}; this script implements {expected}")
        fitted = read_csv(self.dir / "data.csv")
        response = {"bernoulli": "yb", "binomial": "k", "cumulative": "ord", "Gamma": "ypos", "wiener": "rt", "negbinomial": "ytr"}.get(
            family, "ypos" if self.lb is not None else "y")
        X, Y = self.fixture["standata"]["X"], self.fixture["standata"]["Y"]
        j, n = X["colnames"].index("x"), len(fitted)
        if [float(r["x"]) for r in fitted] != X["values"][j * n:(j + 1) * n] or [float(r[response]) for r in fitted] != Y["values"]:
            raise SystemExit(f"{case_id}: data.csv's x or {response} is not, double for double, brms's standata X or Y")
        if family == "negbinomial" and (self.fixture["standata"]["lb"]["values"] != [int(self.lb)] * n
                                        or self.fixture["standata"]["ub"]["values"] != ["Inf"] * n):
            raise SystemExit(f"{case_id}: brms's standata lb and ub are not {self.lb} and Inf on every row")
        self.x_mean = sum(number(r["x"]) for r in fitted) / len(fitted)
        if family == "wiener":                               # min_Y, the upper bound of ndt; brms's dec; bs's centred dummies of z
            self.min_y = min(number(r["rt"]) for r in fitted)
            if [int(r["dec"]) for r in fitted] != self.fixture["standata"]["dec"]["values"]:
                raise SystemExit(f"{case_id}: data.csv's dec is not brms's standata dec")
            self.z_means = {lvl: Decimal(sum(r["z"] == lvl for r in fitted)) / len(fitted) for lvl in ("q", "r")} if self.bs_formula else {}
            if self.bs_formula:
                Xb = self.fixture["standata"]["X_bs"]
                if Xb["colnames"] != ["Intercept", "zq", "zr"] or [float(r["z"] == lvl) for lvl in ("q", "r") for r in fitted] != Xb["values"][n:]:
                    raise SystemExit(f"{case_id}: data.csv's z is not brms's standata X_bs (Intercept, zq, zr)")
        self.rows = [self._row(r) for r in fitted]
        self.new_rows = [self._row(r, response=False) for r in read_csv(self.dir / "newdata.csv")] \
            if "newdata.csv" in self.sources else []
        self.points = [[Decimal(float(v)) for v in p["u"]["values"]] for p in self.fixture["points"]]
        for k, (u, p) in enumerate(zip(self.points, self.fixture["points"])):     # the transforms are brms's, value by value
            mine = self.constrain(u)[0]
            for name, theirs in p["pars"].items():
                for a, b in zip(mine.get(name, ()), theirs["values"]):
                    if abs(float(a) - b) > 1e-14 * max(1.0, abs(b)):
                        raise SystemExit(f"{case_id}: point {k}: {name} from u is {float(a)!r}, brms's constrained value {b!r}")
            if family == "wiener":                            # and brms's lprior, ndt's truncation term included, is the script's
                theirs = p["pars"]["lprior"]["values"][0]
                if abs(float(self.lprior(mine)) - theirs) > 1e-12 * max(1.0, abs(theirs)):
                    raise SystemExit(f"{case_id}: point {k}: lprior {float(self.lprior(mine))!r}, brms's {theirs!r}")

    def _priors(self, rows: list[dict]) -> dict[str, tuple[Decimal, Decimal]]:
        out = {}
        self.shape_prior = None
        if self.family == "wiener":
            return self._wiener_priors(rows)
        for p in rows:
            g = _GAMMA.fullmatch(p["prior"].strip())
            if self.family in SHAPED and p["class"] == "shape" and g is not None and self.shape_prior is None:
                self.shape_prior = (Decimal(g.group(1)), Decimal(g.group(2)))      # gamma(a, b) on the family's shape
                continue
            m = _NORMAL.fullmatch(p["prior"].strip())
            key = p["class"]
            if m is None or key in out or p.get("coef") or (key == "sd") != ("group" in p) or p.get("group", "g") != "g":
                raise SystemExit(f"{self.id}: prior {p} is not a normal(m, s) prior of one class this script implements")
            out[key] = (Decimal(m.group(1)), Decimal(m.group(2)))
        wanted = {"Intercept", "b", "sd"} | ({"sigma"} if self.family == "gaussian" else set())
        if set(out) != wanted or (self.family in SHAPED) != (self.shape_prior is not None):
            raise SystemExit(f"{self.id}: priors on {sorted(out)}; the model needs exactly {sorted(wanted)}"
                             + (" and gamma(a, b) on shape" if self.family in SHAPED else ""))
        return out

    def _wiener_priors(self, rows: list[dict]) -> dict[str, tuple[Decimal, Decimal]]:
        """wiener's priors: normal(m, s) on Intercept, b and sd (g), and on bs's formula's Intercept and b (dpar bs); gamma(a, b)
        on bs without a formula; normal(m, s) on ndt (truncated to (0, min_Y) by brms's lprior); beta(a, b) on bias."""
        out, self.dpar_priors, self.bs_prior, self.ndt_prior, self.bias_prior = {}, {}, None, None, None
        for p in rows:
            text_, cls, dpar = p["prior"].strip(), p["class"], p.get("dpar", "")
            normal, gamma, beta = _NORMAL.fullmatch(text_), _GAMMA.fullmatch(text_), _BETA.fullmatch(text_)
            if p.get("coef") or (cls == "sd") != ("group" in p) or p.get("group", "g") != "g":
                raise SystemExit(f"{self.id}: prior {p} is not one of a class this script implements")
            if dpar == "bs" and normal is not None and cls in ("Intercept", "b") and cls not in self.dpar_priors:
                self.dpar_priors[cls] = (Decimal(normal.group(1)), Decimal(normal.group(2)))
            elif not dpar and cls in ("Intercept", "b", "sd") and normal is not None and cls not in out:
                out[cls] = (Decimal(normal.group(1)), Decimal(normal.group(2)))
            elif not dpar and cls == "bs" and gamma is not None and self.bs_prior is None:
                self.bs_prior = (Decimal(gamma.group(1)), Decimal(gamma.group(2)))
            elif not dpar and cls == "ndt" and normal is not None and self.ndt_prior is None:
                self.ndt_prior = (Decimal(normal.group(1)), Decimal(normal.group(2)))
            elif not dpar and cls == "bias" and beta is not None and self.bias_prior is None:
                self.bias_prior = (Decimal(beta.group(1)), Decimal(beta.group(2)))
            else:
                raise SystemExit(f"{self.id}: prior {p} is not one this script implements for wiener")
        if set(out) != {"Intercept", "b", "sd"} or self.ndt_prior is None or self.bias_prior is None \
                or (set(self.dpar_priors) == {"Intercept", "b"}) != self.bs_formula or (self.bs_prior is None) != self.bs_formula:
            raise SystemExit(f"{self.id}: wiener's priors are {rows}; the model needs normal on Intercept, b, sd, ndt, beta on bias and "
                             + ("normal on bs's Intercept and b" if self.bs_formula else "gamma on bs"))
        return out

    def _row(self, r: dict[str, str], response: bool = True) -> dict:
        row = {"xc": number(r["x"]) - self.x_mean, "level": self.levels.index(r["g"])}
        if self.family == "wiener" and self.bs_formula:      # brms's Xc_bs: the dummies of z centred at the fitted rows' means
            row["zc"] = [Decimal(int(r["z"] == lvl)) - self.z_means[lvl] for lvl in ("q", "r")]
        if not response:
            return row
        if self.formula.startswith("yb "):
            row["y"] = int(r["yb"])
        elif self.family == "cumulative":
            row["y"] = int(r["ord"])
            if not 1 <= row["y"] <= self.nthres + 1:
                raise SystemExit(f"{self.id}: category {row['y']} outside 1..{self.nthres + 1}")
        elif self.family == "Gamma":                         # cens(cens_i, ypos2): the code, the response and the interval's end
            row["y"], row["cens"], row["y2"] = number(r["ypos"]), int(r["cens_i"]), number(r["ypos2"])
            if row["cens"] not in (-1, 0, 1, 2) or (row["cens"] == 2 and not row["y2"] > row["y"]):
                raise SystemExit(f"{self.id}: censoring code {row['cens']} on ({row['y']}, {row['y2']})")
        elif self.formula.startswith("k "):
            row["k"], row["n"] = int(r["k"]), int(r["size"])
            row["log_choose"] = Decimal(math.comb(row["n"], row["k"])).ln()
        elif self.family == "wiener":                        # the response time and its decision (1 upper, 0 lower)
            row["y"], row["dec"] = number(r["rt"]), int(r["dec"])
            if row["dec"] not in (0, 1) or not row["y"] > 0:
                raise SystemExit(f"{self.id}: decision {row['dec']} at the response time {row['y']}")
        elif self.family == "negbinomial":                   # a count at or above the truncation bound
            row["y"] = int(r["ytr"])
            if not row["y"] >= self.lb:
                raise SystemExit(f"{self.id}: count {row['y']} below the truncation bound {self.lb}")
        elif self.lb is None:                                # cens(cens_ind): its code; y ~ x + (1 | g): every row observed (0)
            row["y"], row["cens"] = number(r["y"]), int(r["cens_ind"]) if self.censored else 0
            if row["cens"] not in (-1, 0, 1):
                raise SystemExit(f"{self.id}: censoring code {row['cens']}")
        else:
            row["y"] = number(r["ypos"])
        return row

    # -- the target ------------------------------------------------------------------------------------------------
    def constrain(self, u: list[Decimal]) -> tuple[dict[str, list[Decimal]], Decimal]:
        """brms's constrained parameters at the unconstrained point ``u`` and the log-Jacobian of the transforms."""
        pars, jacobian = {}, Decimal(0)
        for (par, _, transform), v in zip(self.upars, u):
            if transform == "log":
                pars.setdefault(par, []).append(v.exp())
                jacobian += v
            elif transform == "joint" and par in pars:          # Stan's ordered: each threshold the previous plus exp(u)
                pars[par].append(pars[par][-1] + v.exp())
                jacobian += v
            elif transform == "other" and self.family == "wiener":   # Stan's lower-upper: ub inv_logit(u), log-Jacobian
                upper = self.min_y if par == "ndt" else Decimal(1)  # log(ub) + log inv_logit(u) + log(1 - inv_logit(u))
                pars.setdefault(par, []).append(upper / (1 + (-v).exp()))
                jacobian += upper.ln() + _log_inv_logit(v) + _log_inv_logit(-v)
            else:
                pars.setdefault(par, []).append(v)
        return pars, jacobian

    def jacobian_slopes(self, u: list[Decimal]) -> list:
        """d (log-Jacobian) / d u_i: 1 in each log-transformed coordinate and each threshold coordinate after the first, and
        1 - 2 inv_logit(u_i) in each of wiener's lower-upper coordinates (ndt, bias)."""
        out, seen = [], set()
        for (par, _, transform), v in zip(self.upars, u):
            if transform == "other" and self.family == "wiener":
                out.append(1 - 2 / (1 + (-v).exp()))
            else:
                out.append(1 if transform == "log" or (transform == "joint" and par in seen) else 0)
            seen.add(par)
        return out

    def eta(self, pars: dict, row: dict, groups: bool = True) -> Decimal:
        if self.family == "cumulative":                     # the thresholds take the intercept's place
            eta = pars["b"][0] * row["xc"]
        else:
            eta = pars["Intercept"][0] + pars["b"][0] * row["xc"]
        return eta + pars["sd_1"][0] * pars["z_1"][row["level"]] if groups else eta

    def boundary_separation(self, pars: dict, row: dict) -> Decimal:
        """wiener's bs: the parameter, or with bs ~ z exp(Intercept_bs + Xc_bs b_bs) of the row."""
        if not self.bs_formula:
            return pars["bs"][0]
        with _guarded():
            r = (pars["Intercept_bs"][0] + sum(c * b for c, b in zip(row["zc"], pars["b_bs"]))).exp()
        with localcontext(CONTEXT):
            return +r

    def row_term(self, pars: dict, row: dict) -> Decimal:
        eta = self.eta(pars, row)
        if self.family == "wiener":   # Stan's wiener_lpdf: the upper boundary (dec 1) is the lower one's with (-mu, 1 - bias)
            bs, bias, t = self.boundary_separation(pars, row), pars["bias"][0], row["y"] - pars["ndt"][0]
            return wiener_lower(t, -eta, bs, 1 - bias) if row["dec"] == 1 else wiener_lower(t, eta, bs, bias)
        if self.family == "cumulative":                     # probit: Phi(c_y - eta) - Phi(c_{y-1} - eta)
            c, y = pars["Intercept"], row["y"]
            return log_normal_interval(c[y - 2] - eta if y > 1 else None, c[y - 1] - eta if y <= self.nthres else None)
        if self.family == "Gamma":                           # rate = shape / mu, mu = exp(eta)
            a = pars["shape"][0]
            with _guarded():
                rate = a / eta.exp()
                t = rate * row["y"]
                t2 = rate * row["y2"]
            if row["cens"] == 0:
                return gamma_lpdf(row["y"], a, rate)
            return log_gamma_interval(a, *{1: (t, None), -1: (None, t), 2: (t, t2)}[row["cens"]])
        if self.link == "cloglog":
            e = eta.exp()
            if "k" in row:
                return row["log_choose"] + row["k"] * log1m_exp_neg(e) - (row["n"] - row["k"]) * e
            return log1m_exp_neg(e) if row["y"] == 1 else -e
        if self.family == "negbinomial":                     # trunc(lb = 1): the log mass minus log P(Y >= 1)
            shape = pars["shape"][0]
            return negbinomial_lpmf(row["y"], eta, shape) - negbinomial_log_p_positive(eta, shape)
        sigma = pars["sigma"][0]
        if self.lb is not None:
            return normal_lpdf(row["y"], eta, sigma) - log_ccdf((self.lb - eta) / sigma)
        if row["cens"] == 0:
            return normal_lpdf(row["y"], eta, sigma)
        t = (row["y"] - eta) / sigma
        return log_ccdf(t) if row["cens"] == 1 else log_cdf(t)

    def lprior(self, pars: dict) -> Decimal:
        """brms's lprior of the wiener model: normal on Intercept and b (and on bs's formula's Intercept_bs and b_bs), the
        normal on sd_1 truncated to (0, inf), gamma on bs, the normal on ndt less log(Phi((min_Y - m) / s) - Phi((0 - m) / s)),
        its mass on (0, min_Y), and beta on bias."""
        lp = Decimal(0)
        for name, cls in (("Intercept", "Intercept"), ("b", "b")):
            m, s = self.priors[cls]
            lp += sum(normal_lpdf(v, m, s) for v in pars[name])
        if self.bs_formula:
            for name, cls in (("Intercept_bs", "Intercept"), ("b_bs", "b")):
                m, s = self.dpar_priors[cls]
                lp += sum(normal_lpdf(v, m, s) for v in pars[name])
        else:
            lp += gamma_lpdf(pars["bs"][0], *self.bs_prior)
        m, s = self.ndt_prior
        lp += normal_lpdf(pars["ndt"][0], m, s) - log_normal_interval((0 - m) / s, (self.min_y - m) / s)
        lp += beta_lpdf(pars["bias"][0], *self.bias_prior)
        m, s = self.priors["sd"]
        lp += sum(normal_lpdf(v, m, s) - log_ccdf((0 - m) / s) for v in pars["sd_1"])
        return lp

    def lp_nojac(self, u: list[Decimal]) -> Decimal:
        """brms's target without the Jacobian: the likelihood, lprior and std_normal(z_1)."""
        pars, _ = self.constrain(u)
        if self.family == "wiener":
            lp = self.lprior(pars)
            lp += sum(normal_lpdf(v, Decimal(0), Decimal(1)) for v in pars["z_1"])
            return lp + sum(self.row_term(pars, row) for row in self.rows)
        lp = Decimal(0)
        for name, cls in (("Intercept", "Intercept"), ("b", "b")):
            m, s = self.priors[cls]
            lp += sum(normal_lpdf(v, m, s) for v in pars[name])
        for name, cls in (("sd_1", "sd"), ("sigma", "sigma")):
            if name in pars:                       # <lower=0>: the normal truncated to (0, inf), brms's - normal_lccdf(0 | m, s)
                m, s = self.priors[cls]
                lp += sum(normal_lpdf(v, m, s) - log_ccdf((0 - m) / s) for v in pars[name])
        if self.family in SHAPED:                            # gamma(a, b) on shape, its declared support (0, inf)
            lp += gamma_lpdf(pars["shape"][0], *self.shape_prior)
        lp += sum(normal_lpdf(v, Decimal(0), Decimal(1)) for v in pars["z_1"])
        lp += sum(self.row_term(pars, row) for row in self.rows)
        return lp

    def gradient_nojac(self, u: list[Decimal]) -> list[Decimal]:
        """d lp_nojac / d u_i by the fourth-order central difference (f(-2h) - 8 f(-h) + 8 f(h) - f(2h)) / (12 h)."""
        out = []
        for i in range(len(u)):
            def at(step: Decimal) -> Decimal:
                v = list(u)
                v[i] = u[i] + step
                return self.lp_nojac(v)
            out.append((at(-2 * STEP) - 8 * at(-STEP) + 8 * at(STEP) - at(2 * STEP)) / (12 * STEP))
        return out

    def wiener_epred(self, pars: dict, row: dict, groups: bool) -> Decimal:
        """The mean response time of the row (``wiener_mean``), mu its linear predictor with or without the group effects."""
        return wiener_mean(self.eta(pars, row, groups), self.boundary_separation(pars, row), pars["ndt"][0], pars["bias"][0])

    def truncated_mean(self, pars: dict, row: dict, groups: bool) -> Decimal:
        """brms's posterior_epred of the truncated Gaussian: eta + sigma phi(a) / (1 - Phi(a)), a = (lb - eta) / sigma."""
        eta, sigma = self.eta(pars, row, groups), pars["sigma"][0]
        return eta + sigma * inverse_mills((self.lb - eta) / sigma)

    def count_truncated_mean(self, pars: dict, row: dict, groups: bool) -> Decimal:
        """The mean of the negative binomial its likelihood truncates to Y >= 1, E[Y | Y >= 1] (brms's posterior_epred is the mean
        above 2, divergence D121)."""
        return negbinomial_positive_mean(self.eta(pars, row, groups), pars["shape"][0])

    def stacking(self) -> dict:
        """loo_model_weights()'s stacking weights of the two models the fixture compares, exactly (``stacking_weight``), from the
        pointwise elpd_loo brms handed loo (``post.loo_compare.elpd_loo``, a column per model in the order of the weights)."""
        post = self.fixture["post"]
        weights, table = post["loo_model_weights"], post["loo_compare"]["elpd_loo"]
        models, n = list(weights["models"]), len(self.rows)
        if weights.get("method") != "stacking" or len(models) != 2 or table.get("colnames") != models or table["dims"] != [n, 2]:
            raise SystemExit(f"{self.id}: loo_model_weights weighs {models} by {weights.get('method')!r} and loo_compare's elpd_loo is "
                             f"{table['dims']} on {table.get('colnames')}; this script implements stacking of two models on the {n} rows")
        first, second = ([Decimal(float(v)) for v in table["values"][j * n:(j + 1) * n]] for j in range(2))
        if not all(v.is_finite() for v in first + second):
            raise SystemExit(f"{self.id}: loo_compare's elpd_loo is not finite")
        w = stacking_weight(first, second)
        return {"models": models, "values": [text(w), text(1 - w)]}


# -- the file ----------------------------------------------------------------------------------------------------------
def text(v: Decimal) -> str:
    return format(v, f".{DIGITS - 1}E")


def build(case_id: str) -> str:
    """The reference file of ``case_id`` as text."""
    with localcontext(CONTEXT):
        case = Case(case_id)
        points = []
        for u in case.points:
            _, jacobian = case.constrain(u)
            lp_nojac = case.lp_nojac(u)
            grad_nojac = case.gradient_nojac(u)
            grad = [g + slope for g, slope in zip(grad_nojac, case.jacobian_slopes(u))]
            points.append({"lp": text(lp_nojac + jacobian), "lp_nojac": text(lp_nojac), "grad": [text(g) for g in grad],
                           "grad_nojac": [text(g) for g in grad_nojac]})
        if case.family == "wiener":                     # its own description of the model and the arithmetic
            model, arithmetic = wiener_text(case)
        out = {
            "case": case_id,
            "generator": "oracle/reference.py",
            "model": model if case.family == "wiener" else f"{case.formula}, brms {case.family}({case.link}); priors "
                     + ", ".join(f"{cls} normal({m}, {s})" for cls, (m, s) in case.priors.items())
                     + (f", shape gamma({case.shape_prior[0]}, {case.shape_prior[1]})" if case.shape_prior is not None else "")
                     + ", z_1 std_normal",
            "arithmetic": arithmetic if case.family == "wiener" else
                          f"decimal, {PREC} significant digits ({GUARD} guard digits in erf, erfc and the constants); values "
                          f"written with {DIGITS}; the gradient without the Jacobian is a fourth-order central difference of the "
                          f"{PREC}-digit density with step {STEP}, with it plus 1 in each log-transformed coordinate",
            "sources": {str(path.relative_to(REPO)): sha256(path) for path in case.sources.values()},
            "upars": [f"{par}[{elem}]" for par, elem, _ in case.upars],
            "points": points,
        }
        if case.family == "negbinomial":                # its expected values and the digits its log masses need
            out["model"] += "; expected values E[Y | Y >= 1] = mu / (1 - (shape / (mu + shape))^shape)"
            out["arithmetic"] = (f"decimal, {PREC} significant digits ({GUARD} guard digits in log Gamma, the logarithms and the "
                                 f"constants, more where 1 - P(Y = 0) cancels); values written with {DIGITS}; the gradient without the "
                                 f"Jacobian is a fourth-order central difference of the {PREC}-digit density with step {STEP}, with it "
                                 f"plus 1 in each log-transformed coordinate")
        if case.lb is not None:                         # G4's expected values: the truncated mean (Gaussian: posterior_epred; D121)
            pars = [case.constrain(u)[0] for u in case.points]
            mean = case.truncated_mean if case.family == "gaussian" else case.count_truncated_mean
            out["predictions"] = {
                block: {kind: [[text(mean(p, row, groups)) for row in rows] for p in pars]
                        for kind, groups in (("epred", True), ("epred_nogroup", False))}
                for block, rows in (("fitted", case.rows), ("newdata", case.new_rows)) if rows}
        if case.family == "wiener":                     # G4's expected values: the mean response time (D148)
            pars = [case.constrain(u)[0] for u in case.points]
            out["predictions"] = {
                block: {kind: [[text(case.wiener_epred(p, row, groups)) for row in rows] for p in pars]
                        for kind, groups in (("epred", True), ("epred_nogroup", False))}
                for block, rows in (("fitted", case.rows), ("newdata", case.new_rows)) if rows}
        if "loo_model_weights" in (case.fixture.get("post") or {}):   # the post gate's stacking weights (D126)
            out["weights"] = case.stacking()
            out["arithmetic"] += (f"; the stacking weights the root of the Karush-Kuhn-Tucker condition F'(w) = 0 of the two models' "
                                  f"objective, bisected at {PREC + GUARD} digits to an interval of width {BISECT_WIDTH}")
    return json.dumps(out, indent=1, ensure_ascii=False) + "\n"


def wiener_text(case: Case) -> tuple[str, str]:
    """The ``model`` and ``arithmetic`` entries of a wiener reference."""
    priors = ", ".join(f"{cls} normal({m}, {s})" for cls, (m, s) in case.priors.items())
    if case.bs_formula:
        priors += ", " + ", ".join(f"{cls} normal({m}, {s}) (dpar bs)" for cls, (m, s) in case.dpar_priors.items())
    else:
        priors += f", bs gamma({case.bs_prior[0]}, {case.bs_prior[1]})"
    priors += (f", ndt normal({case.ndt_prior[0]}, {case.ndt_prior[1]}) on (0, min_Y), bias beta({case.bias_prior[0]}, "
               f"{case.bias_prior[1]}), z_1 std_normal")
    model = (f"{case.formula}, brms wiener(identity{', bs log' if case.bs_formula else ''}); Stan's wiener_lpdf(rt | bs, ndt, bias, mu) "
             f"for dec = 1 and wiener_lpdf(rt | bs, ndt, 1 - bias, -mu) for dec = 0; priors {priors}; expected values the mean "
             f"response time ndt + (bs P_up - bs bias) / mu, P_up = (1 - exp(-2 mu bs bias)) / (1 - exp(-2 mu bs))")
    arithmetic = (f"decimal, {PREC} significant digits ({GUARD} guard digits in the first-passage series, erfc, log Gamma and the "
                  f"constants, more where terms cancel); the Navarro-Fuss small-time series for t / bs^2 <= 1, the large-time one "
                  f"above, each summed until the next term's envelope is below the last digit; values written with {DIGITS}; the "
                  f"gradient without the Jacobian is a fourth-order central difference of the {PREC}-digit density with step "
                  f"{STEP}, with it plus 1 in each log-transformed coordinate and 1 - 2 inv_logit(u) in ndt's and bias's")
    return model, arithmetic


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--check", action="store_true", help="recompute and compare with the committed file byte for byte")
    ap.add_argument("--all", action="store_true", help="with --check: every committed reference")
    args = ap.parse_args(argv)
    ids = list(args.ids)
    if args.all:
        if not args.check:
            ap.error("--all goes with --check")
        ids = sorted(p.stem for p in REFERENCES.glob("*.json"))
    if not ids:
        ap.error("give case ids (or --check --all)")
    failed = False
    for cid in ids:
        path = REFERENCES / f"{cid}.json"
        if args.check:
            if not path.exists():
                print(f"{cid}: no committed reference")
                failed = True
                continue
            same = build(cid) == path.read_text(encoding="utf-8")
            print(f"{cid}: {'identical' if same else 'DIFFERS from the committed file'}")
            failed = failed or not same
        elif path.exists():
            print(f"{cid}: already written (a committed reference is never overwritten; --check compares)")
        else:
            REFERENCES.mkdir(parents=True, exist_ok=True)
            path.write_text(build(cid), encoding="utf-8")
            print(f"{cid}: written to {path.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
