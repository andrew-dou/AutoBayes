"""More zero-inflated and hurdle families (brms-parity D03, docs/zero-inflated-hurdle.md): ``zero_inflated_negbinomial``,
``hurdle_negbinomial``, ``zero_inflated_binomial``, ``zero_inflated_beta``, ``hurdle_gamma`` and ``hurdle_lognormal``, and
``weights()`` for these and B06's two.

- the row densities: the negative binomial and binomial ones against a 50-digit reference (the coefficient as an exact sum
  of logarithms), value and gradient in ``eta``, the shape and ``zi`` / ``hu`` on both paths (PyTensor's gradient and JAX's
  autodiff of the jaxified graph) from ``eta = -1000`` to ``1000`` and shapes from 1e-6 to 1e9; the Beta, Gamma and
  lognormal ones against SciPy, their zero rows exact with a zero ``eta`` gradient at every ``eta``;
- the samplers: the exact zero-truncated negative binomial draws against the pmf, and brms 2.23's recorded
  ``posterior_predict()`` draws of ``hurdle_negbinomial`` (``tests/zero_hurdle_probes``), which do not follow it (D86);
- each model's density and gradient by hand, priors and their refusals, ``value_point``, responses, predictions by hand
  (means and seeded draws), the weighted density and ``log_lik`` / ``loo`` / ``waic`` under ``weights()``, every other
  construct refused naming it or built (``gp()``), ``conditional_effects()``, ``describe()``, prior-predictive draws and the
  fit record. Nothing samples; the brms comparison is the oracle's eight frozen cases (tests/oracle)."""
import json
import math
import re
import warnings
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import stats
from scipy.special import digamma, expit, logit

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import get_family
from autobayes.post_processing import loo_from_log_lik, waic_from_log_lik
from tests.oracle import adapter, datasets, gates

NEW = ["zero_inflated_negbinomial", "hurdle_negbinomial", "zero_inflated_binomial", "zero_inflated_beta", "hurdle_gamma",
       "hurdle_lognormal"]
ALL = NEW + ["zero_inflated_poisson", "hurdle_poisson"]
SPECS = {                                   # family -> (left-hand side on DATA, its auxiliary priors)
    "zero_inflated_negbinomial": ("zcnt", {"shape": ab.InverseGamma(0.4, 0.3), "zi": ab.Beta(2, 2)}),
    "hurdle_negbinomial": ("zcnt", {"shape": ab.InverseGamma(0.4, 0.3), "hu": ab.Beta(2, 2)}),
    "zero_inflated_binomial": ("zk | trials(size)", {"zi": ab.Beta(2, 2)}),
    "zero_inflated_beta": ("zprop", {"phi": ab.Gamma(0.01, 0.01), "zi": ab.Beta(2, 2)}),
    "hurdle_gamma": ("zpos", {"shape": ab.Gamma(0.01, 0.01), "hu": ab.Beta(2, 2)}),
    "hurdle_lognormal": ("zpos", {"sigma": ab.HalfNormal(1), "hu": ab.Beta(2, 2)}),
    "zero_inflated_poisson": ("zcnt", {"zi": ab.Beta(2, 2)}),
    "hurdle_poisson": ("zcnt", {"hu": ab.Beta(2, 2)}),
}
DATA = datasets.tier_d(7, 40)               # zero-inflated counts zcnt, successes zk of size, zprop in [0, 1), zpos >= 0, weights w
CASES = {"zero_inflated_negbinomial": "zero_inflated_negbinomial_varint", "hurdle_negbinomial": "hurdle_negbinomial_varint",
         "zero_inflated_binomial": "zero_inflated_binomial_varint", "zero_inflated_beta": "zero_inflated_beta_varint",
         "hurdle_gamma": "hurdle_gamma_varint", "hurdle_lognormal": "hurdle_lognormal_varint"}


def priors(family, group=True, **override):
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **({"sd": ab.HalfNormal(1)} if group else {}))
    return ab.Priors(**{**base, **SPECS[family][1], **override})


def build(family, rhs="x + (1 | g)", frame=None, lhs=None, **override):
    group = "|" in rhs
    return getattr(ab, family)(f"{lhs or SPECS[family][0]} ~ {rhs}", DATA if frame is None else frame,
                               priors(family, group, **override))


def weighted_lhs(lhs: str) -> str:
    return f"{lhs} + weights(w)" if "|" in lhs else f"{lhs} | weights(w)"


# -- 1. the negative binomial and binomial rows: a 50-digit reference -----------------------------------------------------
def _log1p(x):
    return (1 + x).ln() if abs(x) > Decimal("1e-25") else x - x * x / 2 + x ** 3 / 3


def _expm1(x):
    return x.exp() - 1 if abs(x) > Decimal("1e-25") else x + x * x / 2 + x ** 3 / 6


def _log_factorial(y):
    return sum((Decimal(k).ln() for k in range(2, y + 1)), Decimal(0))


def _nb(y, e, s):
    """log NB(y | mu = exp(e), s) and its derivatives in e and s; log NB(0) and its derivatives (Decimal): the coefficient
    log C(y + s - 1, y) is the exact sum of log(s + j) over j < y minus log y!."""
    mu = e.exp()
    ratio = mu / s
    log_total = s.ln() + _log1p(ratio)                                  # log(mu + s)
    value = sum(((s + j).ln() for j in range(y)), Decimal(0)) - _log_factorial(y) - s * _log1p(ratio) + y * (e - log_total)
    d_e = s * (y - mu) / (mu + s)
    d_s = sum((1 / (s + j) for j in range(y)), Decimal(0)) - _log1p(ratio) + (mu - y) / (mu + s)
    return value, d_e, d_s, -s * _log1p(ratio), -s * mu / (mu + s), -_log1p(ratio) + mu / (mu + s)


def exact_nb_row(family, y, eta, shape, p):
    """(log density, d/d eta, d/d shape, d/d p) of one row of brms's zero_inflated_negbinomial (p = zi) or
    hurdle_negbinomial (p = hu) at 50 digits."""
    with localcontext() as ctx:
        ctx.prec = 50
        e, s, P = Decimal(float(eta)), Decimal(float(shape)), Decimal(float(p))
        value, d_e, d_s, log_q, q_e, q_s = _nb(y, e, s)
        q, one_minus_q = log_q.exp(), -_expm1(log_q)                    # NB(0), P(y > 0)
        if family == "zero_inflated_negbinomial":
            if y == 0:
                S = P + (1 - P) * q
                out = (S.ln(), (1 - P) * q * q_e / S, (1 - P) * q * q_s / S, one_minus_q / S)
            else:
                out = ((1 - P).ln() + value, d_e, d_s, -1 / (1 - P))
        elif y == 0:
            out = (P.ln(), Decimal(0), Decimal(0), 1 / P)
        else:                                                           # minus log P(y > 0), the zero truncation
            r = q / one_minus_q
            out = ((1 - P).ln() + value - one_minus_q.ln(), d_e + r * q_e, d_s + r * q_s, -1 / (1 - P))
        return tuple(float(v) for v in out)


def exact_zib_row(y, n, eta, p):
    """(log density, d/d eta, d/d zi) of one row of brms's zero_inflated_binomial (logit link) at 50 digits."""
    with localcontext() as ctx:
        ctx.prec = 50
        e, P = Decimal(float(eta)), Decimal(float(p))
        log_p, log_1mp = -_log1p((-e).exp()), -_log1p(e.exp())
        prob = log_p.exp()
        if y == 0:
            q = (n * log_1mp).exp()                                     # (1 - p)^n
            S = P + (1 - P) * q
            out = (S.ln(), -(1 - P) * q * n * prob / S, -_expm1(n * log_1mp) / S)
        else:
            coef = _log_factorial(n) - _log_factorial(y) - _log_factorial(n - y)
            out = ((1 - P).ln() + coef + y * log_p + (n - y) * log_1mp, y - n * prob, -1 / (1 - P))
        return tuple(float(v) for v in out)


NB_ETAS = [-1000.0, -745.5, -700.0, -100.0, -40.0, -20.5, -20.0, -3.0, -1.0, 0.0, 0.5, 2.0, 5.0, 20.0, 40.0, 100.0, 700.0, 1000.0]
SHAPES = [1e-6, 0.3, 1.0, 7.5, 1e4, 1e9]
PROBS = [1e-6, 0.3, 0.97]
NB_LOGP = {"zero_inflated_negbinomial": lk.zero_inflated_negbinomial_logp, "hurdle_negbinomial": lk.hurdle_negbinomial_logp}


def _paths(fn, mode, ints, floats):
    """``fn(*ints, *floats)`` per row and its gradient in each float input, on one path: PyTensor's gradient graph, or JAX's
    autodiff of the jaxified graph (the path NumPyro samples on). ``ints`` are data (integer or real arrays), no gradient."""
    iv = [pt.vector(f"i{k}", dtype=np.asarray(a).dtype) for k, a in enumerate(ints)]
    fv = [pt.dvector(f"f{k}") for k in range(len(floats))]
    rows = fn(*iv, *fv)
    if mode == "pytensor":
        f = pytensor.function([*iv, *fv], [rows, *(pt.grad(rows.sum(), v) for v in fv)])
        return [np.asarray(a, dtype=float) for a in f(*ints, *floats)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[*iv, *fv], outputs=[rows])
    out = [np.asarray(jf(*ints, *floats)[0], dtype=float)]
    for k in range(len(floats)):
        def total(v, k=k):
            args = list(floats)
            args[k] = v
            return jf(*ints, *args)[0].sum()
        out.append(np.asarray(jax.grad(total)(floats[k]), dtype=float))
    return out


def _close(got, ref, tol=1e-12):
    return np.isfinite(got) and abs(got - ref) <= tol * max(1.0, abs(ref))


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("family", sorted(NB_LOGP))
def test_negative_binomial_rows_are_exact_and_never_nan_on_both_paths(family, mode):
    grid = [(y, e, s, p) for y in (0, 1, 3, 40) for e in NB_ETAS for s in SHAPES for p in PROBS]
    y, eta, shape, p = (np.array([r[k] for r in grid], dtype=np.int64 if k == 0 else float) for k in range(4))
    value, d_eta, d_shape, d_p = _paths(NB_LOGP[family], mode, [y], [eta, shape, p])
    assert not any(np.isnan(a).any() for a in (value, d_eta, d_shape, d_p))
    for i, row in enumerate(grid):
        want = list(exact_nb_row(family, *row))
        # the shape's derivative as the sampler sees it, in log(shape) (alpha_log__): at shape 1e-6 its terms are 1 / shape
        # = 1e6 each and cancel, so d/d shape itself is exact to 1e6 eps, and d/d log(shape) to eps. On the PyTensor path the
        # coefficient's gradient goes through PyTensor's digamma (gammaln's gradient), 1.5e-10 off relative between 1 and 12
        # (JAX's is exact to rounding; NumPyro samples on JAX)
        want[2] *= row[2]
        tolerances = (1e-12, 1e-12, 1e-12 if mode == "jax" else 2e-9, 1e-12)
        for got, ref, tol in zip((value[i], d_eta[i], d_shape[i] * row[2], d_p[i]), want, tolerances):
            assert _close(got, ref, tol), (family, mode, row, got, ref)


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_zero_inflated_binomial_rows_are_exact_on_both_paths(mode):
    grid = [(y, n, e, p) for n in (1, 7, 60) for y in sorted({0, 1, n // 2, n}) for e in (-1000.0, -40.0, -5.0, -1.0, 0.0, 0.7, 3.0,
                                                                                           40.0, 1000.0) for p in PROBS]
    y, n, eta, p = (np.array([r[k] for r in grid], dtype=np.int64 if k < 2 else float) for k in range(4))
    value, d_eta, d_p = _paths(lambda yv, nv, ev, pv: lk.zero_inflated_binomial_logp(yv, ev, pv, nv), mode, [y, n], [eta, p])
    for i, row in enumerate(grid):
        for got, ref in zip((value[i], d_eta[i], d_p[i]), exact_zib_row(*row)):
            assert _close(got, ref), (mode, row, got, ref)


def test_the_coefficient_and_the_truncation_hold_where_lgamma_differences_and_subtraction_fail():
    """At shape 1e12 the coefficient's two lgamma values are 2.6e13 each, so their difference keeps no digit of the
    O(100) answer; ``log_nb_coefficient`` is exact there. ``log(1 - NB(0))`` at mu = 1e-30 is log(mu) + O(mu): ``1 - (shape
    / (mu + shape))^shape`` by subtraction is 0."""
    yv, sv, ev = pt.dvector("y"), pt.dvector("s"), pt.dvector("e")
    coef = pytensor.function([yv, sv], lk.log_nb_coefficient(yv, sv))
    for y, s in ((40, 1e12), (3, 1e-6), (1000, 2.5), (0, 3.0)):
        with localcontext() as ctx:
            ctx.prec = 50
            S = Decimal(s)
            want = float(sum(((S + j).ln() for j in range(y)), Decimal(0)) - _log_factorial(y))
        assert abs(float(coef(np.array([y], float), np.array([s]))[0]) - want) <= 1e-13 * max(1.0, abs(want)), (y, s)
    with np.errstate(divide="ignore"):
        assert np.log(1 - (2.0 / (1e-30 + 2.0)) ** 2.0) == -np.inf
    positive = pytensor.function([ev, sv], lk.neg_binomial_log_positive(ev, sv))
    got = float(positive(np.array([math.log(1e-30)]), np.array([2.0]))[0])
    assert abs(got - math.log(1e-30)) < 1e-14 * 70                     # log(1e-30) - 7.5e-31: log(1e-30) to double precision


# -- 2. the Beta, Gamma and lognormal rows ----------------------------------------------------------------------------------
def _beta_row(y, eta, phi, p):
    if y == 0:
        return math.log(p), 0.0, 0.0, 1 / p
    m = expit(eta)
    a, b = m * phi, (1 - m) * phi
    da, db = digamma(phi) - digamma(a) + math.log(y), digamma(phi) - digamma(b) + math.log1p(-y)
    return (math.log1p(-p) + stats.beta.logpdf(y, a, b), (da - db) * phi * m * (1 - m), m * da + (1 - m) * db, -1 / (1 - p))


def _gamma_row(y, eta, shape, p):
    if y == 0:
        return math.log(p), 0.0, 0.0, 1 / p
    mu = math.exp(eta)
    return (math.log1p(-p) + stats.gamma.logpdf(y, shape, scale=mu / shape), -shape + shape * y / mu,
            math.log(shape) + 1 - digamma(shape) - eta + math.log(y) - y / mu, -1 / (1 - p))


def _lognormal_row(y, mu, sigma, p):
    if y == 0:
        return math.log(p), 0.0, 0.0, 1 / p
    z = (math.log(y) - mu) / sigma
    return math.log1p(-p) + stats.lognorm.logpdf(y, sigma, scale=math.exp(mu)), z / sigma, (z * z - 1) / sigma, -1 / (1 - p)


CONTINUOUS = {"zero_inflated_beta": (lk.zero_inflated_beta_logp, _beta_row, (0.0, 0.03, 0.5, 0.97)),
              "hurdle_gamma": (lk.hurdle_gamma_logp, _gamma_row, (0.0, 1e-3, 0.8, 30.0)),
              "hurdle_lognormal": (lk.hurdle_lognormal_logp, _lognormal_row, (0.0, 1e-3, 0.8, 30.0))}


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("family", sorted(CONTINUOUS))
def test_continuous_rows_and_their_zero_rows_on_both_paths(family, mode):
    fn, reference, ys = CONTINUOUS[family]
    grid = [(y, e, s, p) for y in ys for e in (-12.0, -2.0, 0.0, 0.4, 3.0, 12.0) for s in (0.05, 1.0, 40.0) for p in PROBS]
    y, eta, aux, p = (np.array([r[k] for r in grid], dtype=float) for k in range(4))
    value, d_eta, d_aux, d_p = _paths(fn, mode, [y], [eta, aux, p])
    for i, row in enumerate(grid):                     # the gradients through digamma: PyTensor's is 1.5e-10 off (see above)
        for got, ref, tol in zip((value[i], d_eta[i], d_aux[i], d_p[i]), reference(*row), (1e-12, 1e-8, 1e-8, 1e-12)):
            assert _close(got, ref, tol), (family, mode, row, got, ref)
    # a zero row is log(p) with a zero eta gradient at every eta, where the positive part's density is infinite at y = 0
    eta = np.array([-1000.0, -745.0, 0.0, 709.0, 1000.0])
    value, d_eta, d_aux, d_p = _paths(fn, mode, [np.zeros(5)], [eta, np.full(5, 2.0), np.full(5, 0.3)])
    assert np.all(value == math.log(0.3)) and np.all(d_eta == 0.0) and np.all(d_aux == 0.0)
    np.testing.assert_allclose(d_p, 1 / 0.3, rtol=1e-14)


def test_responses_outside_the_support_have_no_mass():
    one = pt.dvector("y")
    for fn, bad in ((lk.zero_inflated_beta_logp, [1.0, -0.2, 1.3]), (lk.hurdle_gamma_logp, [-0.5]), (lk.hurdle_lognormal_logp, [-0.5])):
        f = pytensor.function([one], fn(one, pt.constant(0.3), pt.constant(2.0), pt.constant(0.25)))
        assert np.all(f(np.array(bad)) == -np.inf), fn.__name__
    for fn in NB_LOGP.values():
        f = pytensor.function([], fn(pt.constant(np.array([-1])), pt.constant(0.3), pt.constant(2.0), pt.constant(0.2)))
        assert f()[0] == -np.inf


# -- 3. the samplers ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mu, shape", [(0.5, 0.5), (3.0, 2.0), (0.05, 0.2), (1.0, 1e-8), (1e-3, 1e8), (1e6, 0.5), (1e-12, 1e-12),
                                       (7.0, 1.0)])
def test_zero_truncated_negbinomial_draws_have_the_truncated_distribution(mu, shape):
    n = 100000
    y = lk.zero_truncated_negbinomial_draws(np.full(n, mu), np.full(n, shape), np.random.default_rng(5))
    assert y.min() >= 1 and y.dtype == np.int64
    positive = -math.expm1(-shape * math.log1p(mu / shape))
    for k in (1, 2, 3):                                           # the pmf, NB(k | mu, shape) / (1 - NB(0 | mu, shape))
        pk = stats.nbinom.pmf(k, shape, shape / (shape + mu)) / positive
        assert abs(np.mean(y == k) - pk) < 4.5 * math.sqrt(pk * (1 - pk) / n) + 1e-12, k
    if shape >= 0.5:                                              # a finite variance to compare the mean with
        mean = mu / positive
        var = (mu + mu * mu / shape + mu * mu) / positive - mean ** 2
        assert abs(y.mean() - mean) < 4.5 * math.sqrt(var / n)


def test_zero_truncated_negbinomial_limits_and_refusals():
    rng = np.random.default_rng(1)
    assert np.all(lk.zero_truncated_negbinomial_draws(np.array([0.0, 1e-320]), np.array([2.0, 2.0]), rng) == 1)
    for mu, shape in ((1.0, 0.0), (1.0, -1.0), (np.nan, 1.0), (-1.0, 1.0), (1.0, np.inf)):
        with pytest.raises(ValueError, match="need a finite mean mu >= 0 and a finite shape > 0"):
            lk.zero_truncated_negbinomial_draws(np.array([mu]), np.array([shape]), rng)
    with pytest.raises(ValueError, match="integer random-generator limit"):       # a log-series tail beyond 1e297
        lk.zero_truncated_negbinomial_draws(np.full(100, 2.0), np.full(100, 1e-300), rng)


PROBE = json.loads((Path(__file__).parent / "zero_hurdle_probes" / "brms-2.23.0-zero-hurdle.json").read_text())


@pytest.mark.parametrize("point", PROBE["hurdle_negbinomial_draws"], ids=lambda p: f"mu={p['mu']},shape={p['shape']}")
def test_brms_posterior_predict_of_hurdle_negbinomial_is_not_the_truncated_distribution(point):
    """D86: brms 2.23's posterior_predict() draws of hurdle_negbinomial at a fixed point (recorded by the probe) put far
    more mass on 1 than the zero-truncated negative binomial, whose mean brms's own posterior_epred() reports; this
    library's draws follow that distribution."""
    mu, shape, hu = point["mu"], point["shape"], point["hu"]
    counts = np.asarray(point["counts"], dtype=float)
    n, positive_n = counts.sum(), counts[1:].sum()
    assert n == point["n"]
    positive = -math.expm1(-shape * math.log1p(mu / shape))
    assert abs(float(point["epred"]) - (1 - hu) * mu / positive) < 1e-14 * mu / positive   # brms's epred: the truncated mean
    p1 = stats.nbinom.pmf(1, shape, shape / (shape + mu)) / positive
    brms_p1 = counts[1] / positive_n
    assert abs(brms_p1 - p1) > 20 * math.sqrt(p1 * (1 - p1) / positive_n)                 # brms's draws: not the distribution
    rng = np.random.default_rng(11)
    ours = lk.hurdle_negbinomial_random(np.log(mu), shape, hu, rng=rng, size=(int(n),))
    pos = ours[ours > 0]
    assert abs(np.mean(pos == 1) - p1) < 4.5 * math.sqrt(p1 * (1 - p1) / pos.size)
    assert abs(np.mean(ours == 0) - hu) < 4.5 * math.sqrt(hu * (1 - hu) / n)


@pytest.mark.parametrize("family", NEW)
def test_likelihood_objects_draw_their_distribution(family):
    import pymc as pm

    eta = np.array([-1.0, 0.4, 1.5])
    p, aux, n = 0.3, 2.0, np.array([3, 5, 8])
    dist = {"zero_inflated_negbinomial": lambda: lk.zero_inflated_negbinomial_family().likelihood.dist.dist(log_mu=eta, alpha=aux, zi=p),
            "hurdle_negbinomial": lambda: lk.hurdle_negbinomial_family().likelihood.dist.dist(log_mu=eta, alpha=aux, hu=p),
            "zero_inflated_binomial": lambda: lk.zero_inflated_binomial_family().likelihood.dist.dist(logit_p=eta, zi=p, n=n),
            "zero_inflated_beta": lambda: lk.zero_inflated_beta_family().likelihood.dist.dist(logit_mu=eta, phi=aux, zi=p),
            "hurdle_gamma": lambda: lk.hurdle_gamma_family().likelihood.dist.dist(log_mu=eta, alpha=aux, hu=p),
            "hurdle_lognormal": lambda: lk.hurdle_lognormal_family().likelihood.dist.dist(mu=eta, sigma=aux, hu=p)}[family]()
    S = 40000
    draws = pm.draw(dist, draws=S, random_seed=3)
    assert draws.shape == (S, 3) and draws.min() >= 0
    zero_mass = {"zero_inflated_negbinomial": p + (1 - p) * (aux / (np.exp(eta) + aux)) ** aux,
                 "zero_inflated_binomial": p + (1 - p) * expit(-eta) ** n}.get(family, np.full(3, p))
    assert np.all(np.abs((draws == 0).mean(0) - zero_mass) < 4.5 * np.sqrt(zero_mass * (1 - zero_mass) / S))
    mean = {"zero_inflated_negbinomial": (1 - p) * np.exp(eta),
            "hurdle_negbinomial": (1 - p) * np.exp(eta) / (1 - (aux / (np.exp(eta) + aux)) ** aux),
            "zero_inflated_binomial": (1 - p) * n * expit(eta), "zero_inflated_beta": (1 - p) * expit(eta),
            "hurdle_gamma": (1 - p) * np.exp(eta), "hurdle_lognormal": (1 - p) * np.exp(eta + aux ** 2 / 2)}[family]
    sd = draws.std(0)
    assert np.all(np.abs(draws.mean(0) - mean) < 5 * sd / math.sqrt(S)), (draws.mean(0), mean)


# -- 4. the models: density and gradient by hand ------------------------------------------------------------------------------
FRAME = pd.DataFrame({"x": [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7], "g": ["a", "a", "b", "b", "c", "c", "a", "b"],
                      "zcnt": [0, 2, 5, 0, 1, 0, 3, 0], "zk": [0, 2, 3, 0, 1, 0, 4, 2], "size": [3, 4, 5, 2, 3, 6, 4, 2],
                      "zprop": [0.0, 0.3, 0.8, 0.0, 0.55, 0.0, 0.2, 0.9], "zpos": [0.0, 1.2, 3.5, 0.0, 0.4, 0.0, 2.2, 0.7],
                      "w": [1.5, 0.6, 2.0, 1.0, 0.8, 1.2, 0.7, 1.9]})
AUX = {"shape": 1.7, "phi": 3.2, "sigma": 0.8}


def _by_hand_rows(family, eta, aux, p):
    y, f = FRAME[SPECS[family][0].split()[0]].to_numpy(), FRAME
    zero = y == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if family == "zero_inflated_negbinomial":
            pos = stats.nbinom.logpmf(y, aux, aux / (aux + np.exp(eta)))
            return np.where(zero, np.log(p + (1 - p) * np.exp(pos)), np.log1p(-p) + pos)
        if family == "hurdle_negbinomial":
            q = (aux / (aux + np.exp(eta))) ** aux
            return np.where(zero, np.log(p), np.log1p(-p) + stats.nbinom.logpmf(y, aux, aux / (aux + np.exp(eta))) - np.log1p(-q))
        if family == "zero_inflated_binomial":
            pos = stats.binom.logpmf(y, f["size"].to_numpy(), expit(eta))
            return np.where(zero, np.log(p + (1 - p) * np.exp(pos)), np.log1p(-p) + pos)
        if family == "zero_inflated_beta":
            return np.where(zero, np.log(p), np.log1p(-p) + stats.beta.logpdf(y, expit(eta) * aux, expit(-eta) * aux))
        if family == "hurdle_gamma":
            return np.where(zero, np.log(p), np.log1p(-p) + stats.gamma.logpdf(y, aux, scale=np.exp(eta) / aux))
        return np.where(zero, np.log(p), np.log1p(-p) + stats.lognorm.logpdf(y, aux, scale=np.exp(eta)))


def _prior_logpdf(prior, value):
    name, kw = prior.name, prior.params
    return {"InverseGamma": lambda: stats.invgamma.logpdf(value, kw["alpha"], scale=kw["beta"]),
            "Gamma": lambda: stats.gamma.logpdf(value, kw["alpha"], scale=1 / kw["beta"]),
            "HalfNormal": lambda: stats.halfnorm.logpdf(value, scale=kw["sigma"])}[name]()


def _linpred(m, point):
    pmm = m.pymc_model
    return np.ravel(pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars,
                                   on_unused_input="ignore")(point))


@pytest.mark.parametrize("family", NEW)
def test_hierarchical_density_and_gradient_by_hand_both_paths(family):
    fam = get_family(family)
    zero, scale = fam.zero_parameter, fam.scale_parameter
    m = build(family, frame=FRAME, **{zero: ab.Beta(2, 3)})
    nat = dict(intercept=0.3, b=[-0.4], sd=0.5, z=np.array([0.2, -0.4, 0.1]))
    aux = {scale: AUX[scale]} if scale else {}
    point = m.value_point(**nat, **aux, **{zero: 0.35})
    np.testing.assert_allclose(point[f"{zero}_logodds__"], logit(0.35), rtol=1e-15)
    x = FRAME.x.to_numpy()
    eta = 0.3 - 0.4 * (x - x.mean()) + 0.5 * nat["z"][pd.Categorical(FRAME.g).codes]
    np.testing.assert_allclose(_linpred(m, point), eta, rtol=1e-14)            # the graph's parent is eta itself
    lik = _by_hand_rows(family, eta, AUX.get(scale), 0.35).sum()
    prior = (stats.norm.logpdf(0.3, 0, 2) + stats.norm.logpdf(-0.4, 0, 1) + stats.halfnorm.logpdf(0.5)
             + stats.norm.logpdf(nat["z"]).sum() + stats.beta.logpdf(0.35, 2, 3))
    jac = np.log(0.5) + np.log(0.35) + np.log1p(-0.35)                         # sd on the log scale, zi / hu on the logit scale
    if scale:
        prior += _prior_logpdf(getattr(m.priors, scale), AUX[scale])
        jac += np.log(AUX[scale])                                              # shape, phi, sigma on the log scale
    pmm = m.pymc_model
    np.testing.assert_allclose(pmm.compile_logp()(point), lik + prior + jac, rtol=1e-12)
    np.testing.assert_allclose(pmm.compile_logp(jacobian=False)(point), lik + prior, rtol=1e-12)
    import jax
    from pymc.sampling.jax import get_jaxified_logp

    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    args = [point[n] for n in names]
    np.testing.assert_allclose(jf(args), lik + prior + jac, rtol=1e-12)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    g_pt = np.asarray(pmm.compile_dlogp()(point), dtype=float)
    np.testing.assert_allclose(g_jax, g_pt, rtol=1e-9, atol=1e-12)       # PyTensor's digamma: 1.5e-10 (section 1)
    # d/d logit(p) by hand: the rows' d/dp times p (1 - p), the Beta(2, 3) prior's (1 - p) - 2 p and the Jacobian's 1 - 2 p
    h, q = 1e-6, 0.35
    rows_dp = (_by_hand_rows(family, eta, AUX.get(scale), q + h) - _by_hand_rows(family, eta, AUX.get(scale), q - h)).sum() / (2 * h)
    want = rows_dp * q * (1 - q) + (1 - q) - 2 * q + 1 - 2 * q
    np.testing.assert_allclose(g_pt[names.index(f"{zero}_logodds__")], want, rtol=1e-7)


# -- 5. priors, value_point, responses -------------------------------------------------------------------------------------
def test_priors_are_required_and_refused_by_name():
    with pytest.raises(ab.PriorError, match=r"the zero_inflated_negbinomial family needs a prior on the shape \(overdispersion\) "
                                            r"parameter: pass shape=\.\.\."):
        ab.zero_inflated_negbinomial("zcnt ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), zi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match="the hurdle_negbinomial family needs a prior on the hurdle probability hu"):
        build("hurdle_negbinomial", "x", hu=None)
    with pytest.raises(ab.PriorError, match="the zero_inflated_beta family needs a prior on the precision parameter phi"):
        build("zero_inflated_beta", "x", phi=None)
    with pytest.raises(ab.PriorError, match="the hurdle_lognormal family needs a prior on the SD of log y"):
        build("hurdle_lognormal", "x", sigma=None)
    with pytest.raises(ab.PriorError, match=r"the hurdle_gamma family has no zero-inflation probability; do not pass zi. Its "
                                            r"auxiliary parameter is shape: pass shape=\.\.\. Its hurdle probability is hu"):
        build("hurdle_gamma", "x", zi=ab.Beta(1, 1))
    with pytest.raises(ab.PriorError, match=r"the zero_inflated_binomial family has no residual SD; do not pass sigma. Its "
                                            r"zero-inflation probability is zi: pass zi=\.\.\. Its intercept and slope priors are on "
                                            r"the logit scale"):
        build("zero_inflated_binomial", "x", sigma=ab.HalfNormal(1))
    with pytest.raises(ab.PriorError, match="the zero_inflated_beta family has no shape"):
        build("zero_inflated_beta", "x", shape=ab.Gamma(1, 1))
    with pytest.raises(ab.PriorError, match=r"prior for 'shape' must be a positive-support prior"):
        build("hurdle_negbinomial", "x", shape=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"prior for 'zi' must be a prior on \(0, 1\)"):
        build("zero_inflated_negbinomial", "x", zi=ab.HalfNormal(1))


@pytest.mark.parametrize("family", NEW)
def test_names_and_value_point(family):
    fam = get_family(family)
    zero, scale = fam.zero_parameter, fam.scale_parameter
    other = "hu" if zero == "zi" else "zi"
    m = build(family)
    assert m.names[zero] == zero and other not in m.names and m.names["linpred"] == fam.parent
    if scale:
        assert m.names[scale] == fam.bambi_scale_parameter and f"{fam.bambi_scale_parameter}_log__" in {v.name for v in m.pymc_model.value_vars}
    nat = dict(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(6))
    aux = {scale: 1.3} if scale else {}
    p = m.value_point(**nat, **aux, **{zero: 0.25})
    assert set(p) == {v.name for v in m.pymc_model.value_vars}
    with pytest.raises(ValueError, match=rf"{zero} \(the .* probability\) is required for the {family} family"):
        m.value_point(**nat, **aux)
    if scale:
        with pytest.raises(ValueError, match=re.escape(fam.scale_rule)):
            m.value_point(**nat, **{zero: 0.25})
    for bad in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match=f"{zero} must be a finite scalar strictly between 0 and 1"):
            m.value_point(**nat, **aux, **{zero: bad})


@pytest.mark.parametrize("family, values, message", [
    ("zero_inflated_negbinomial", [-1, 2], "negative value"), ("hurdle_negbinomial", [0.5, 2], "non-integral"),
    ("zero_inflated_negbinomial", [2e6, 1], "exceeds the Poisson input-count limit"),
    ("zero_inflated_beta", [0.0, 1.0], r"value outside \[0, 1\) \(1.0\).*an exact 1 is refused \(brms's zero_one_inflated_beta"),
    ("zero_inflated_beta", [-0.1, 0.5], r"outside \[0, 1\)"), ("zero_inflated_beta", [np.inf, 0.5], "non-finite"),
    ("hurdle_gamma", [0.0, -0.5], r"negative value \(-0.5\); the hurdle Gamma family needs non-negative responses"),
    ("hurdle_lognormal", [np.inf, 1.0], "non-finite"), ("hurdle_lognormal", [True, False], "numeric response"),
])
def test_invalid_responses(family, values, message):
    with pytest.raises(ab.FormulaError, match=message):
        getattr(ab, family)(f"{SPECS[family][0]} ~ 1", pd.DataFrame({SPECS[family][0]: values}), priors(family, False, b=None))


def test_zero_inflated_binomial_responses_and_all_zero_warnings():
    with pytest.raises(ab.FormulaError, match="exceed"):
        ab.zero_inflated_binomial("zk | trials(size) ~ 1", pd.DataFrame({"zk": [3, 1], "size": [2, 2]}), priors("zero_inflated_binomial", False, b=None))
    m = ab.zero_inflated_binomial("zk | trials(size) ~ 1", pd.DataFrame({"zk": [0, 1, 0], "size": [0, 2, 3]}),
                                  priors("zero_inflated_binomial", False, b=None))
    assert m.design.n == 2                                         # the zero-trial row is dropped (D4), as for the binomial family
    for family in NEW:
        lhs = SPECS[family][0]
        frame = FRAME.assign(**{lhs.split()[0]: 0})
        m = build(family, "x", frame=frame)
        assert len(m.warnings) == 1 and ("zero" in m.warnings[0]), (family, m.warnings)


# -- 6. predictions -----------------------------------------------------------------------------------------------------------
def draws(family, a=0.3, p=0.25, aux=1.7, group=True):
    fam = get_family(family)
    post = {"Intercept": np.full((2, 4), a), "x": np.full((2, 4), -0.2), fam.zero_parameter: np.full((2, 4), p)}
    if fam.scale_parameter:
        post[fam.bambi_scale_parameter] = np.full((2, 4), aux)
    if group:
        post.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.1, -0.1, 0.3], (2, 4, 3))})
    return az.from_dict(posterior=post, coords={"g__factor_dim": ["a", "b", "c"]}, dims={"1|g": ["g__factor_dim"]})


def _expected(family, eta, aux, p, trials):
    if family in ("zero_inflated_negbinomial", "hurdle_gamma"):
        return (1 - p) * np.exp(eta)
    if family == "hurdle_negbinomial":
        mu = np.exp(eta)
        return (1 - p) * mu / -np.expm1(-aux * np.log1p(mu / aux))
    if family == "zero_inflated_binomial":
        return (1 - p) * trials * expit(eta)
    if family == "zero_inflated_beta":
        return (1 - p) * expit(eta)
    return (1 - p) * np.exp(eta + aux ** 2 / 2)


def _draws_by_hand(family, eta, aux, p, trials, rng):
    """The seeded response draws as predict() makes them: the zero part's uniform first, then the other part's draws."""
    zero = rng.uniform(size=eta.shape) < p
    mu = np.exp(eta)
    other = {"zero_inflated_negbinomial": lambda: rng.negative_binomial(aux, aux / (aux + mu)),
             "hurdle_negbinomial": lambda: lk.zero_truncated_negbinomial_draws(mu, np.full(eta.shape, aux), rng),
             "zero_inflated_binomial": lambda: rng.binomial(trials, expit(eta)),
             "zero_inflated_beta": lambda: rng.beta(expit(eta) * aux, expit(-eta) * aux),
             "hurdle_gamma": lambda: rng.gamma(aux, mu / aux),
             "hurdle_lognormal": lambda: rng.lognormal(eta, aux)}[family]()
    return np.where(zero, 0, other)


@pytest.mark.parametrize("family", NEW)
def test_fitted_and_new_row_predictions_by_hand(family):
    fam = get_family(family)
    m = build(family, frame=FRAME)
    post = draws(family)
    u = np.array([0.1, -0.1, 0.3])[pd.Categorical(FRAME.g).codes]
    eta = np.broadcast_to(0.3 - 0.2 * FRAME.x.to_numpy() + u, (2, 4, 8))
    trials = FRAME["size"].to_numpy()
    lin = m.predict(post, kind="linear")
    np.testing.assert_allclose(lin.eta, eta, rtol=1e-14)
    assert lin.attrs["link"] == fam.link
    out = m.predict(post, kind="response", seed=9)
    assert set(out.data_vars) == {fam.mean_name, "y"}
    np.testing.assert_allclose(out[fam.mean_name], _expected(family, eta, 1.7, 0.25, trials), rtol=1e-13)
    np.testing.assert_array_equal(out.y, _draws_by_hand(family, eta, 1.7, 0.25, trials, np.random.default_rng(9)))
    assert out.y.dtype == (np.int64 if fam.discrete else np.float64)
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["b", "c"], "size": [5, 9]})
    eta_new = np.broadcast_to(0.3 - 0.2 * new.x.to_numpy() + np.array([-0.1, 0.3]), (2, 4, 2))
    mean = m.predict(post, newdata=new, kind="mean")[fam.mean_name]
    np.testing.assert_allclose(mean, _expected(family, eta_new, 1.7, 0.25, new["size"].to_numpy()), rtol=1e-13)
    if fam.has_trials:                                             # brms's expected count needs each new row's trials
        with pytest.raises(Exception, match="size"):
            m.predict(post, newdata=new.drop(columns="size"), kind="mean")
    with pytest.raises(ValueError, match="kind"):
        m.predict(post, kind="probability")


def test_prediction_limits():
    nb, hnb = build("zero_inflated_negbinomial", frame=FRAME), build("hurdle_negbinomial", frame=FRAME)
    with pytest.raises(ValueError, match="overflow"):
        nb.predict(draws("zero_inflated_negbinomial", a=800.0), kind="mean")
    with pytest.raises(ValueError, match="integer random-generator limit"):
        hnb.predict(draws("hurdle_negbinomial", a=60.0), kind="response", seed=1)
    tiny = hnb.predict(draws("hurdle_negbinomial", a=-800.0), kind="response", seed=1)   # mu underflows: E[y | y > 0] -> 1
    np.testing.assert_allclose(tiny["mean"], 0.75)
    assert set(np.unique(tiny.y)) <= {0, 1}
    with pytest.raises(ValueError, match="underflowed to zero"):
        build("zero_inflated_beta", frame=FRAME).predict(draws("zero_inflated_beta", a=800.0), kind="response", seed=1)


# -- 7. weights() ----------------------------------------------------------------------------------------------------------
def _point(model, seed=3):
    rng = np.random.default_rng(seed)
    return {k: np.asarray(v + rng.normal(0.0, 0.3, np.shape(v))) for k, v in model.pymc_model.initial_point().items()}


def _rowwise(model):
    pmm = model.pymc_model
    return pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False)[0], inputs=pmm.value_vars, on_unused_input="ignore")


@pytest.mark.parametrize("family", ALL)
def test_weighted_log_density_is_the_rows_times_their_weights(family):
    """brms adds weights[n] * lpmf(Y[n] | ...) row by row: the weighted density is the unweighted one plus
    sum((w - 1) * log p(y_i)), on both paths; unit weights give the unweighted density."""
    lhs = SPECS[family][0]
    m0 = build(family, frame=FRAME)
    mw = build(family, frame=FRAME, lhs=weighted_lhs(lhs))
    ones = build(family, frame=FRAME.assign(w=1.0), lhs=weighted_lhs(lhs))
    point = _point(m0)
    l0, lw, l1 = (float(m.pymc_model.compile_logp()(point)) for m in (m0, mw, ones))
    rows = np.asarray(_rowwise(m0)(point), dtype=float)
    w = mw.design.weights
    assert np.array_equal(w, FRAME.w.to_numpy()) and np.all(np.isfinite(rows))
    assert abs((lw - l0) - np.sum((w - 1.0) * rows)) <= 1e-10 * max(1.0, abs(lw))
    np.testing.assert_allclose(np.asarray(_rowwise(mw)(point), dtype=float), w * rows, rtol=1e-12, atol=1e-12)
    assert abs(l1 - l0) <= 1e-12 * max(1.0, abs(l0))
    from pymc.sampling.jax import get_jaxified_logp

    names = [v.name for v in mw.pymc_model.value_vars]
    assert abs(float(get_jaxified_logp(mw.pymc_model)([point[n] for n in names])) - lw) <= 1e-12 * max(1.0, abs(lw))
    assert mw.target_identity["sha256"] != m0.target_identity["sha256"]
    assert "weights   : w, each fitted row's log likelihood multiplied by its weight" in mw.describe()


@pytest.mark.parametrize("case_id", ["zero_inflated_poisson_weighted", "hurdle_poisson_weighted"])
def test_weights_multiply_log_lik_loo_and_waic_and_not_the_predictions(case_id):
    """brms's log_lik() multiplies each row's log density by its weight (C07, verified against binomial_weighted_post), and
    loo() and waic() read it; posterior_epred ignores the weights (G4 checks it against brms on this case)."""
    b = gates.Bundle(case_id)
    family = b.model.family
    plain = getattr(ab, family)("zcnt ~ x + (1 | g)", b.data, b.model.priors)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    w = b.model.design.weights
    assert not np.allclose(w, 1.0)
    weighted = ab.log_lik(b.model, idata)
    expected = w * ab.log_lik(plain, idata).to_numpy()
    np.testing.assert_allclose(weighted.to_numpy(), expected, rtol=1e-14)
    ll = weighted.copy(data=expected)
    pd.testing.assert_frame_equal(ab.waic(b.model, idata).estimates, waic_from_log_lik(ll).estimates)
    pd.testing.assert_frame_equal(ab.loo(b.model, idata, r_eff=1).estimates, loo_from_log_lik(ll, r_eff=1).estimates)
    for kind in ("linear", "mean"):
        got, want = b.model.predict(idata, kind=kind), plain.predict(idata, kind=kind)
        for var in got.data_vars:
            np.testing.assert_allclose(got[var].to_numpy(), want[var].to_numpy(), rtol=1e-14)


# -- 8. every other construct: refused naming it, or built -------------------------------------------------------------------
CONTEXTS = [
    ("zi_formula", lambda f, lhs, zero, scale: (getattr(ab, f), [f"{lhs} ~ x", f"{zero} ~ w"]), r"brms's (zi|hu) ~ \.\.\. is not offered here"),
    ("scale_formula", lambda f, lhs, zero, scale: (getattr(ab, f), [f"{lhs} ~ x", f"{scale} ~ w"]), r"brms's (shape|phi|sigma) ~ \.\.\. is not offered here"),
    ("cens", lambda f, lhs, zero, scale: (getattr(ab, f), f"{_add(lhs, 'cens(c)')} ~ x"), r"'cens\(c\)' is not available for the \w+ family"),
    ("trunc", lambda f, lhs, zero, scale: (getattr(ab, f), f"{_add(lhs, 'trunc(lb = 0)')} ~ x"), r"'trunc\(lb = 0\)' is not available"),
    ("mo", lambda f, lhs, zero, scale: (getattr(ab, f), f"{lhs} ~ x + mo(ord)"), r"the monotonic term 'mo\(ord\)' is not available for the \w+ family"),
    ("ar", lambda f, lhs, zero, scale: (getattr(ab, f), f"{lhs} ~ x + ar(time = t, gr = g)"), r"the autocorrelation term .* is not available.*D70"),
    ("me", lambda f, lhs, zero, scale: (getattr(ab, f), f"{lhs} ~ me(xobs, sdx)"), r"the measurement-error term 'me\(xobs, sdx\)' is not available"),
    ("mm", lambda f, lhs, zero, scale: (getattr(ab, f), f"{lhs} ~ x + (1 | mm(g, g2))"), r"the multi-membership term .* is not available for the \w+ family"),
    ("gr_by", lambda f, lhs, zero, scale: (getattr(ab, f), f"{lhs} ~ x + (1 | gr(g, by = gby))"), r"the term with gr\(by = \) .* is not available"),
    ("mvbind", lambda f, lhs, zero, scale: (getattr(ab, f), f"mvbind({lhs.split()[0]}, cnt) ~ x"), r"a multivariate \w+ model is not offered.*D58"),
]


def _add(lhs, term):
    return f"{lhs} + {term}" if "|" in lhs else f"{lhs} | {term}"


@pytest.mark.parametrize("family", NEW)
@pytest.mark.parametrize("label", [c[0] for c in CONTEXTS])
def test_other_constructs_are_refused_naming_them(family, label):
    _, make, message = next(c for c in CONTEXTS if c[0] == label)
    fam = get_family(family)
    if label == "scale_formula" and fam.scale_parameter is None:
        pytest.skip("the zero-inflated binomial family has no scale parameter")
    if label == "zi_formula" and fam.zero_parameter in fam.distributional_parameters:   # E03 (tests/test_contexts_newer_families.py)
        pytest.skip(f"E03: the {family} family's {fam.zero_parameter} ~ ... is taken ({family}_distributional_{fam.zero_parameter})")
    ctor, formula = make(family, SPECS[family][0], fam.zero_parameter, fam.scale_parameter)
    frame = DATA.assign(c=DATA.cens_ind)
    extra = {"simo": ab.Dirichlet(1)} if label == "mo" else {"ar": ab.Normal(0, 0.5)} if label == "ar" else {}
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1) if "(1 |" in str(formula) else None,
                    **SPECS[family][1], **extra)
    with pytest.raises(ab.FormulaError, match=message):
        ctor(formula, frame, pri)


@pytest.mark.parametrize("family", NEW)
def test_links_mixtures_and_nonlinear_formulas_are_refused(family):
    lhs, extra = SPECS[family]
    fam = get_family(family)
    other = "probit" if fam.link != "probit" else "logit"
    with pytest.raises(ValueError, match=rf"the {family} family has the {fam.link} link only; link='{other}' is not available"):
        ab.Model(f"{lhs} ~ x", DATA, priors(family, False), family=family, link=other)
    assert ab.Model(f"{lhs} ~ x", DATA, priors(family, False), family=family, link=fam.link).link == fam.link
    with pytest.raises(ValueError, match=rf"a {family} mixture component is not offered.*D52"):
        ab.mixture("y ~ x", DATA, ab.Priors(components=[ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)),
                                                       ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **extra)],
                                           theta=ab.Dirichlet(2)), families=["gaussian", family])
    with pytest.raises(ab.FormulaError, match=rf"the {family} family is not offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel([f"{lhs.split()[0]} ~ a * x", "a ~ 1"], DATA, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, **extra),
                          family=family)


@pytest.mark.parametrize("family", NEW)
def test_the_term_model_and_gaussian_processes_compose(family):
    """The term model is shared (categorical predictors, interactions, no intercept, independent and correlated slopes, the
    offset, which enters the linear predictor), and a Gaussian process enters it before the link as for every family but
    the categorical one (C03; its contribution is checked against the data at build)."""
    lhs, extra = SPECS[family]
    slopes = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}
    for formula, pri, kw in (("x * z", dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)), {"categorical": {"z": "p"}}),
                             ("0 + x", dict(b=ab.Normal(0, 1)), {}),
                             ("x + (1 + x || g)", dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=slopes), {}),
                             ("x + (1 + x | g) + (1 | h)", dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                               sd={**slopes, "h": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)}), {})):
        m = getattr(ab, family)(f"{lhs} ~ {formula}", DATA, ab.Priors(**pri, **extra), **kw)
        assert f"{get_family(family).zero_parameter}_logodds__" in {v.name for v in m.pymc_model.value_vars}
    m = getattr(ab, family)(f"{lhs} ~ x + offset(log(exposure))", DATA, priors(family, False))
    nat = {get_family(family).scale_parameter: 1.1} if get_family(family).scale_parameter else {}
    point = m.value_point(intercept=0.2, b=[0.5], **nat, **{get_family(family).zero_parameter: 0.4})
    x = DATA.x.to_numpy()
    np.testing.assert_allclose(_linpred(m, point), 0.2 + 0.5 * (x - x.mean()) + np.log(DATA.exposure.to_numpy()), rtol=1e-13)
    gp = getattr(ab, family)(f"{lhs} ~ gp(x)", DATA, ab.Priors(intercept=ab.Normal(0, 1), sdgp=ab.HalfNormal(1),
                                                               lscale=ab.InverseGamma(3, 1), **extra))
    assert len(gp.design.gp) == 1 and ab.default_priors(f"{lhs} ~ gp(x)", DATA, family=family).lscale is not None


@pytest.mark.parametrize("family", NEW)
def test_conditional_effects_and_hypothesis(family):
    """D06's conditional_effects() summarises predict(kind="mean") without group effects on its grid, which G4 checks against
    brms's posterior_epred on these cases; the binomial-type family is refused (its grid needs trials), as the binomial is."""
    b = gates.Bundle(CASES[family])
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    h = ab.hypothesis(b.model, idata, ["x > 0"])
    np.testing.assert_allclose(h.table["Estimate"].iloc[0], idata.posterior["x"].mean().item(), rtol=1e-12)
    if b.model.family_spec.has_trials:
        with pytest.raises(ValueError, match="needs the trials of each grid row"):
            ab.conditional_effects(b.model, idata, ["x"])
        return
    table = ab.conditional_effects(b.model, idata, ["x"], resolution=7)["x"]
    mean = b.model.predict(idata, newdata=table[["x"]].reset_index(drop=True), kind="mean", include_group_effects=False)
    draws_ = mean[b.model.names["mean"]].to_numpy().reshape(-1, len(table))
    np.testing.assert_allclose(table["estimate__"].to_numpy(), np.median(draws_, axis=0), rtol=1e-12)


# -- 9. describe(), identity, prior predictive, defaults, the fit record ------------------------------------------------------
LIKELIHOOD_LINES = {
    "zero_inflated_negbinomial": "P(y_i = 0) = zi + (1 - zi) (shape / (mu_i + shape))^shape",
    "hurdle_negbinomial": "P(y_i = k) = (1 - hu) NB(k | mu_i, shape) / (1 - NB(0 | mu_i, shape))",
    "zero_inflated_binomial": "P(zk_i = 0) = zi + (1 - zi) (1 - p_i)^size_i",
    "zero_inflated_beta": "the density is zi at 0 and (1 - zi) Beta(y_i | mu_i phi, (1 - mu_i) phi)",
    "hurdle_gamma": "the density is hu at 0 and (1 - hu) Gamma(y_i | shape, shape / mu_i) above",
    "hurdle_lognormal": "E[y_i] = (1 - hu) exp(mu_i + sigma^2 / 2)",
}


@pytest.mark.parametrize("family", NEW)
def test_describe_identity_and_registry_constructor(family):
    fam = get_family(family)
    m = build(family)
    text = m.describe()
    assert text.startswith(f"AutoBayes {fam.tag} model") and f"family    : {family}, {fam.link} link" in text
    assert f"{fam.zero_parameter} ~ Beta(alpha=2, beta=2)" in text and f"{fam.zero_parameter}_logodds__" in text
    assert LIKELIHOOD_LINES[family] in text
    if fam.scale_parameter:
        assert f"{fam.scale_parameter} ~ {getattr(m.priors, fam.scale_parameter)}" in text
    same = ab.Model(f"{SPECS[family][0]} ~ x + (1 | g)", DATA, priors(family), family=family)
    assert same.describe() == text and same.target_identity == m.target_identity
    assert build(family, **{fam.zero_parameter: ab.Beta(1, 1)}).target_identity != m.target_identity


def test_target_identity_tells_the_families_apart():
    ids = {build(f).target_identity["sha256"] for f in ("zero_inflated_negbinomial", "hurdle_negbinomial")}
    ids |= {build(f, lhs="zcnt").target_identity["sha256"] for f in ("zero_inflated_poisson", "hurdle_poisson")}
    ids |= {build(f).target_identity["sha256"] for f in ("hurdle_gamma", "hurdle_lognormal")}
    assert len(ids) == 6


@pytest.mark.parametrize("family", NEW)
def test_prior_predictive(family):
    fam = get_family(family)
    proper = {"shape": ab.Gamma(2, 0.5)} if family == "hurdle_gamma" else {"phi": ab.Gamma(2, 0.5)} if family == "zero_inflated_beta" else {}
    m = build(family, **proper)                   # gamma(0.01, 0.01) draws below 1e-10 three times in four: degenerate parts
    out = m.prior_predictive(draws=40, seed=3)
    zero = fam.zero_parameter
    assert out.prior[zero].shape == (1, 40) and ((out.prior[zero].values > 0) & (out.prior[zero].values < 1)).all()
    y = out.prior_predictive[m.names["y"]].values
    assert y.shape == (1, 40, m.design.n) and (y >= 0).all() and (y == 0).any()
    if fam.discrete:
        assert y.dtype.kind == "i"
    if family == "zero_inflated_binomial":
        assert (y <= DATA["size"].to_numpy()).all()
    if family == "zero_inflated_beta":                  # a Beta draw within 1e-16 of 1 rounds to 1, as the Beta family's do
        assert (y <= 1).all()


@pytest.mark.parametrize("family, defaults", [
    ("zero_inflated_negbinomial", {"shape": ab.InverseGamma(0.4, 0.3), "zi": ab.Beta(1, 1)}),
    ("hurdle_negbinomial", {"shape": ab.InverseGamma(0.4, 0.3), "hu": ab.Beta(1, 1)}),
    ("zero_inflated_binomial", {"zi": ab.Beta(1, 1)}),
    ("zero_inflated_beta", {"phi": ab.Gamma(0.01, 0.01), "zi": ab.Beta(1, 1)}),
    ("hurdle_gamma", {"shape": ab.Gamma(0.01, 0.01), "hu": ab.Beta(1, 1)}),
])
def test_default_priors_build_the_model(family, defaults):
    """brms's defaults of the auxiliary parameters (G6 checks every row of brms's default_prior() table on each frozen case,
    hurdle_lognormal's data-dependent sigma included)."""
    lhs = SPECS[family][0]
    pri = ab.default_priors(f"{lhs} ~ x + (1 | g)", DATA, family=family)
    for name, prior in defaults.items():
        assert getattr(pri, name) == prior and getattr(pri, name).source == "default"
    m = getattr(ab, family)(f"{lhs} ~ x + (1 | g)", DATA, pri)
    assert get_family(family).zero_parameter in m.default_prior_labels


def test_fit_record_and_summary_rows(monkeypatch):
    import autobayes.session as session

    m = build("hurdle_negbinomial", frame=FRAME)
    post = draws("hurdle_negbinomial")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "1|g_sigma", "alpha", "hu"}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "hurdle_negbinomial" and rec["model"]["link"] == "log"
    assert rec["model"]["priors"]["hu"] == "Beta(alpha=2, beta=2)" and rec["model"]["priors"]["shape"] == "InverseGamma(alpha=0.4, beta=0.3)"
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
