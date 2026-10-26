"""cens() and trunc() beyond the Gaussian family (brms-parity E04, docs/cens-trunc-families.md).

- the regularized incomplete gamma function's derivative in its shape (``tails.gammainc_shape_derivative``) and the ops
  ``gammainc_exact`` / ``gammaincc_exact`` on both paths (PyTensor's gradient graph, JAX's autodiff of the jaxified value)
  against an 80-digit reference (the series of ``P`` in ``Decimal``, fourth-order central differences), where PyTensor's own
  gradient is off by 0.3%;
- every family's tails on both paths against high-precision references: the Weibull's (cloglog forms in ``u = shape log(y /
  scale)``), the lognormal's (the Normal tails of ``log y``), the Gamma's (with the interval from the tail it lies in), and the
  count families' log masses of a truncation range (sums of masses); untaken censoring branches stay finite at extreme
  parameters on both paths;
- each model's row densities by hand (SciPy) at a value point: right, left and interval censoring, upper truncation of counts,
  with ``weights()``, a distributional formula, ``mo()``, slopes and offsets; the two cases accepted against two references
  (the Gamma family's ``cens()``, D120; a count response's lower bound, D121) against brms on every gate but the one brms's
  arithmetic or definition fails, and AutoBayes's numbers there against exact ones;
- predictions (the truncated means and draws of counts by hand, censoring changing none), ``log_lik()`` as each model's own
  pointwise density (and the Gaussian's censored and truncated rows, which E04 brings into ``log_lik()``), ``loo()``,
  ``hypothesis()``, ``conditional_effects()``'s refusal, prior-predictive draws, ``describe()``, the target identity, the fit
  record and default priors; every refusal by message (a count response's lower bound, withheld before D121 was accepted
  against two references, is offered: tests/test_releases.py);
- the branches no frozen case reaches: the shape derivative's Python ``impl`` and its JAX lowering in a gradient graph, an
  unconverged series, the Gaussian's one-sided truncation in ``log_lik()``, a count lower bound at or below 0.

Nothing samples; the comparisons with brms are the oracle's (tests/oracle: ``weibull_censored``, ``lognormal_censored``,
``poisson_truncated``, ``lognormal_censored_interval_weighted``, ``poisson_truncated_weighted_post``,
``weibull_censored_interval_post``, ``negbinomial_truncated_ub``; ``gamma_censored_interval`` and ``negbinomial_truncated`` by
two-reference acceptance, P6)."""
import math
import warnings
from dataclasses import replace
from decimal import Decimal, getcontext, localcontext
from types import MappingProxyType

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
import pytest
from scipy import stats
from scipy.special import gammainc, gammaincc, gammaln, log_ndtr

import autobayes as ab
import autobayes.families as fams
import autobayes.formula as F
from autobayes import likelihoods as lk
from autobayes import tails as T
from autobayes.families import get_family
from tests.oracle import adapter, gates
from tests.oracle.adapter import OracleMismatch
from tests.oracle.two_reference import ReferenceFallback
from tests.test_bounded_families import _derivatives, _lgamma, _paths

FRAME = pd.DataFrame({
    "x": [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7, 0.9, -1.4], "g": list("aabbccabca"),
    "z": [0.2, -0.5, 1.0, 0.3, -1.2, 0.8, 0.1, -0.3, 0.6, -0.9],
    "ypos": [0.8, 1.3, 2.9, 0.4, 1.1, 2.2, 0.6, 1.7, 3.1, 0.9],
    "c": [0, 1, -1, 2, 0, 2, 1, -1, 0, 2],                  # every code of cens(c, y2)
    "c01": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],                  # survival style: right censoring only
    "cnt": [0, 2, 5, 1, 3, 6, 0, 2, 4, 1], "ytr": [1, 2, 5, 1, 3, 6, 1, 2, 4, 1], "ymid": [2, 3, 5, 2, 3, 6, 2, 4, 4, 2],
    "w": [1.5, 0.6, 2.0, 1.0, 0.8, 1.2, 0.7, 1.9, 1.1, 0.5], "ord": [1, 2, 3, 1, 2, 3, 1, 2, 3, 2],
    "exposure": [1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.8, 1.2, 2.5, 0.9],
})
FRAME["ypos2"] = FRAME["ypos"] * 1.6
NAT = {"weibull": {"shape": 1.7}, "lognormal": {"sigma": 0.6}, "gamma": {"shape": 2.3}, "poisson": {}, "negbinomial": {"shape": 1.4},
       "gaussian": {"sigma": 0.7}}
SCALE = {"weibull": ab.Gamma(0.01, 0.01), "gamma": ab.Gamma(0.01, 0.01), "negbinomial": ab.Gamma(0.01, 0.01), "lognormal": ab.HalfNormal(1),
         "gaussian": ab.HalfNormal(1)}
CENSORED = ["weibull", "lognormal", "gamma"]               # gamma: with its park lifted
TRUNCATED = ["poisson", "negbinomial"]


def priors(family, group=True):
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **({"sd": ab.HalfNormal(1)} if group else {}))
    name = get_family(family).scale_parameter
    return ab.Priors(**base, **({name: SCALE[family]} if name else {}))


@pytest.fixture
def lifted(monkeypatch):
    """The E04 parks lifted: the Gamma family's cens() (D120) and a count response's lower bound (D121)."""
    monkeypatch.setattr(F, "PARKED_TRUNCATION", MappingProxyType({}))
    gamma = replace(fams.FAMILIES["gamma"], parked_additions=MappingProxyType({}))
    monkeypatch.setattr(fams, "FAMILIES", MappingProxyType({**fams.FAMILIES, "gamma": gamma}))


# -- 0. references: the regularized incomplete gamma function at 80 digits ---------------------------------------------------
def _log_p(a, x):
    """log P(a, x) (Decimal): ``a log x - x - lgamma(a + 1) + log sum_n x^n / ((a + 1) ... (a + n))``, positive terms."""
    tiny = Decimal(10) ** (-(getcontext().prec + 5))
    term = total = Decimal(1)
    n = 0
    while True:
        n += 1
        term *= x / (a + n)
        total += term
        if x < a + n and term < tiny * total:
            return a * x.ln() - x - _lgamma(a + 1) + total.ln()


def _log_q(a, x):
    return (1 - _log_p(a, x).exp()).ln()


def exact(f, *args, prec=80, h="1e-20"):
    """``f`` and its fourth-order central difference in each argument at the float ``args`` (truncation h^4, rounding
    10^-prec / h), as floats."""
    with localcontext() as ctx:
        ctx.prec = prec
        return [float(v) for v in _derivatives(f, [Decimal(float(a)) for a in args], Decimal(h))]


def _close(got, ref, tol):
    return np.isfinite(got) and abs(got - ref) <= tol * max(1.0, abs(ref))


GAMMAINC_GRID = [(a, x) for a in (0.31, 0.51, 1.0, 2.24, 7.5, 40.0) for x in (1e-3, 0.11, 0.8, 2.45, 7.44, 24.6, 60.0)]
_TABLE = {}


def _table(name, grid, row):
    """``row(*point)`` (the reference values of a grid point) at every point of ``grid``, computed once for both paths."""
    if name not in _TABLE:
        _TABLE[name] = [row(*point) for point in grid]
    return _TABLE[name]


def test_the_reference_is_the_incomplete_gamma_function():
    with localcontext() as ctx:
        ctx.prec = 80
        for a, x in ((0.31, 0.2), (2.24, 2.45), (7.5, 24.6), (40.0, 30.0), (1.0, 3.0)):
            assert abs(float(_log_p(Decimal(a), Decimal(x))) - math.log(gammainc(a, x))) <= 2e-14 * max(1.0, abs(math.log(gammainc(a, x))))
            assert abs(float(_log_q(Decimal(a), Decimal(x))) - math.log(gammaincc(a, x))) <= 2e-14 * max(1.0, abs(math.log(gammaincc(a, x))))
        assert abs(float(_log_q(Decimal(1), Decimal(3))) + 3.0) < 1e-60 * 1e40       # Q(1, x) = exp(-x)


# -- 1. the shape derivative and the exact ops on both paths ------------------------------------------------------------------
def test_shape_derivative_is_exact_in_both_regimes():
    """``dP/da`` from the series (x < a + 1) and the differentiated continued fraction (above), as ``d log P/da`` and ``d log
    Q/da``, against the 80-digit reference."""
    a, x = (np.array([p[k] for p in GAMMAINC_GRID]) for k in range(2))
    dp = T.gammainc_shape_derivative(a, x)
    lp_table = _table("log_p", GAMMAINC_GRID, lambda *p: exact(_log_p, *p))
    lq_table = _table("log_q", GAMMAINC_GRID, lambda *p: exact(_log_q, *p))
    for i, (ai, xi) in enumerate(GAMMAINC_GRID):
        assert _close(dp[i] / gammainc(ai, xi), lp_table[i][1], 1e-13), (ai, xi, dp[i] / gammainc(ai, xi), lp_table[i][1])
        assert _close(-dp[i] / gammaincc(ai, xi), lq_table[i][1], 1e-13), (ai, xi, -dp[i] / gammaincc(ai, xi), lq_table[i][1])
    assert T.gammainc_shape_derivative(2.0, 0.0) == 0.0 and T.gammainc_shape_derivative(2.0, np.inf) == 0.0
    assert np.isnan(T.gammainc_shape_derivative(-1.0, 2.0)) and np.isnan(T.gammainc_shape_derivative(2.0, np.nan))
    assert np.shape(T.gammainc_shape_derivative(np.ones((2, 3)), np.full(3, 2.0))) == (2, 3)


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_exact_incomplete_gamma_ops_on_both_paths(mode):
    """log P and log Q with their gradients in a and x: PyTensor's values (Cephes) and gradient graph (this module's shape
    derivative, the density in x), and JAX's ``gammainc`` / ``gammaincc`` under JAX's autodiff."""
    a, x = (np.array([p[k] for p in GAMMAINC_GRID]) for k in range(2))
    lp = _paths(lambda av, xv: pt.log(T.gammainc_exact(av, xv)), mode, [], [a, x])
    lq = _paths(lambda av, xv: pt.log(T.gammaincc_exact(av, xv)), mode, [], [a, x])
    for got, table in ((lp, _table("log_p", GAMMAINC_GRID, lambda *p: exact(_log_p, *p))), (lq, _table("log_q", GAMMAINC_GRID, lambda *p: exact(_log_q, *p)))):
        for i, point in enumerate(GAMMAINC_GRID):
            for k, (ref, tol) in enumerate(zip(table[i], (2e-13, 1e-12, 1e-12))):
                assert _close(got[k][i], ref, tol), (mode, point, k, got[k][i], ref)


def test_pytensors_own_shape_gradient_is_not_exact():
    """PyTensor's ``gammaincc`` gradient in ``a`` (a port of Stan's grad_reg_inc_gamma, stopped at 1e-6) is 0.28% off at a row of
    the frozen case gamma_censored_interval; ``gammaincc_exact``'s is exact there. The two ops share the value."""
    av, xv = pt.dscalar("a"), pt.dscalar("x")
    ours, theirs = pt.log(T.gammaincc_exact(av, xv)), pt.log(pt.gammaincc(av, xv))
    f = pytensor.function([av, xv], [ours, theirs, pt.grad(ours, av), pt.grad(theirs, av)])
    a, x = 2.100245825833418, 7.439141245482232
    v_ours, v_theirs, g_ours, g_theirs = (float(v) for v in f(a, x))
    ref = exact(_log_q, a, x)[1]
    assert v_ours == v_theirs and _close(g_ours, ref, 1e-13) and abs(g_theirs - ref) / abs(ref) > 2e-3
    assert type(T.gammaincc_exact.scalar_op) is not type(pt.gammaincc(av, xv).owner.op.scalar_op)


# -- 2. the families' tails on both paths --------------------------------------------------------------------------------
def _weibull_rows(kind):
    def f(v, w, a, s):
        t = (v / s) ** a
        if kind == "log_cdf":
            return (1 - (-t).exp()).ln()
        if kind == "log_ccdf":
            return -t
        return ((-t).exp() - (-((w / s) ** a)).exp()).ln()
    return f


WEIBULL_GRID = [(v, 1.6 * v, a, s) for v in (1e-6, 0.05, 0.9, 3.0, 8.0) for a, s in ((0.4, 1.3), (1.7, 0.8), (6.0, 2.5))]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("kind", ["log_cdf", "log_ccdf", "log_mass"])
def test_weibull_tails_are_exact_on_both_paths(mode, kind):
    v, w, a, s = (np.array([p[k] for p in WEIBULL_GRID]) for k in range(4))
    tails = T.WEIBULL_TAILS
    if kind == "log_mass":
        got = _paths(lambda vv, wv, av, sv: tails.log_mass(vv, wv, alpha=av, beta=sv), mode, [v, w], [a, s])
    else:
        got = _paths(lambda vv, av, sv: getattr(tails, kind)(vv, alpha=av, beta=sv), mode, [v], [a, s])
    ref = _table(f"weibull_{kind}", WEIBULL_GRID, lambda *p: exact(lambda a_, s_, p=p: _weibull_rows(kind)(Decimal(p[0]), Decimal(p[1]), a_, s_),
                                                                  p[2], p[3], prec=120, h="1e-30"))   # 1 - exp(-4e-39) keeps 80 digits
    for i, point in enumerate(WEIBULL_GRID):
        for k in range(3):
            assert _close(got[k][i], ref[i][k], 1e-12), (mode, kind, point, k, got[k][i], ref[i][k])


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_lognormal_tails_are_the_normal_tails_of_log_y(mode):
    if mode == "JAX":
        pytest.importorskip("jax")
    v, mu, s = pt.dvector("v"), pt.dvector("mu"), pt.dvector("s")
    tails = T.LOGNORMAL_TAILS
    ours = [tails.log_cdf(v, mu=mu, sigma=s), tails.log_ccdf(v, mu=mu, sigma=s), tails.log_mass(v, 1.6 * v, mu=mu, sigma=s)]
    normal = [lk.normal_log_cdf(pt.log(v), mu, s), lk.normal_log_ccdf(pt.log(v), mu, s), lk.normal_log_mass(pt.log(v), pt.log(1.6 * v), mu, s)]
    grads = [g for e in ours + normal for g in pt.grad(e.sum(), [mu, s])]
    f = pytensor.function([v, mu, s], ours + normal + grads, mode=mode)
    vals = [np.asarray(r) for r in f(np.array([0.01, 0.4, 1.0, 3.0, 40.0]), np.array([1.5, -0.2, 0.0, 0.3, -1.0]), np.array([0.3, 0.6, 1.0, 2.0, 0.5]))]
    for k in range(3):
        np.testing.assert_allclose(vals[k], vals[3 + k], rtol=1e-15)
    np.testing.assert_allclose(vals[6:12], vals[12:18], rtol=1e-15)
    z = (np.log(np.array([0.01, 0.4, 1.0, 3.0, 40.0])) - np.array([1.5, -0.2, 0.0, 0.3, -1.0])) / np.array([0.3, 0.6, 1.0, 2.0, 0.5])
    np.testing.assert_allclose(vals[0], log_ndtr(z), rtol=1e-13)
    np.testing.assert_allclose(vals[1], log_ndtr(-z), rtol=1e-13)


def _gamma_rows(kind):
    """The Gamma's tails in Bambi's parameters (mu, sigma = mu / sqrt(shape)): shape mu^2 / sigma^2, rate mu / sigma^2."""
    def f(v, w, mu, sigma):
        a, rate = mu * mu / (sigma * sigma), mu / (sigma * sigma)
        if kind == "log_cdf":
            return _log_p(a, v * rate)
        if kind == "log_ccdf":
            return _log_q(a, v * rate)
        return (_log_p(a, w * rate).exp() - _log_p(a, v * rate).exp()).ln()
    return f


# (value, upper, mu, shape): the intervals below, around and far above the mode, each tail
GAMMA_GRID = [(v, f * v, mu, sh) for mu, sh in ((0.4, 0.31), (1.7, 2.2), (6.0, 30.0)) for v, f in ((0.02 * mu, 3.0), (0.9 * mu, 1.3), (3.5 * mu, 1.5))]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("kind", ["log_cdf", "log_ccdf", "log_mass"])
def test_gamma_tails_are_exact_on_both_paths(mode, kind):
    v, w, mu, sh = (np.array([p[k] for p in GAMMA_GRID]) for k in range(4))
    sigma = mu / np.sqrt(sh)
    tails = T.GAMMA_TAILS
    if kind == "log_mass":
        got = _paths(lambda vv, wv, mv, sv: tails.log_mass(vv, wv, mu=mv, sigma=sv), mode, [v, w], [mu, sigma])
    else:
        got = _paths(lambda vv, mv, sv: getattr(tails, kind)(vv, mu=mv, sigma=sv), mode, [v], [mu, sigma])
    points = [(p[0], p[1], p[2], p[2] / math.sqrt(p[3])) for p in GAMMA_GRID]
    ref = _table(f"gamma_{kind}", points, lambda *p: exact(lambda m_, s_, p=p: _gamma_rows(kind)(Decimal(p[0]), Decimal(p[1]), m_, s_),
                                                          p[2], p[3]))
    for i, point in enumerate(GAMMA_GRID):
        for k, tol in enumerate((2e-13, 1e-11, 1e-11)):
            assert _close(got[k][i], ref[i][k], tol), (mode, kind, point, k, got[k][i], ref[i][k])


def _poisson_range(lo, hi):
    """log P(lo <= Y <= hi) of a Poisson(mu) count (Decimal): the masses summed over the range, or one minus those below lo."""
    def f(mu):
        def pmf(j):
            return (-mu).exp() * mu ** j / math.factorial(j)
        if hi is not None:
            return sum((pmf(j) for j in range(0 if lo is None else lo, hi + 1)), Decimal(0)).ln()
        return (1 - sum((pmf(j) for j in range(lo)), Decimal(0))).ln()
    return f


def _negbinomial_range(lo, hi):
    def f(mu, s):
        def pmf(j):
            return (_lgamma(j + s) - _lgamma(s) - _lgamma(Decimal(j + 1)) + s * (s / (s + mu)).ln() + j * (mu / (s + mu)).ln()).exp()
        if hi is not None:
            return sum((pmf(j) for j in range(0 if lo is None else lo, hi + 1)), Decimal(0)).ln()
        return (1 - sum((pmf(j) for j in range(lo)), Decimal(0))).ln()
    return f


COUNT_RANGES = [(None, 0), (None, 3), (None, 6), (None, 40), (1, None), (2, None), (5, None), (2, 6)]
COUNT_MEANS = np.array([0.03, 0.8, 3.5, 12.0, 60.0])


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("lo,hi", COUNT_RANGES)
def test_count_truncation_masses_are_exact_on_both_paths(mode, lo, hi):
    """log P(lo <= Y <= hi), the bounds inclusive: sums of masses, exact for every mean when the range is bounded above. A lower
    bound alone is log1mexp of the masses below it: exact for lb = 1 (log(1 - P(0))), and otherwise to about 1e-16 / P(Y >= lb)
    relative, which the withheld lower bound (D121) documents."""
    shape = np.array([0.4, 1.4, 5.0, 0.9, 30.0])
    pois = _paths(lambda m: T.POISSON_TAILS.log_truncation(lo, hi, mu=m), mode, [], [COUNT_MEANS])
    nb = _paths(lambda m, s: T.NEGBINOMIAL_TAILS.log_truncation(lo, hi, mu=m, alpha=s), mode, [], [COUNT_MEANS, shape])
    for i in range(len(COUNT_MEANS)):
        for name, got, ref in (("poisson", pois, exact(_poisson_range(lo, hi), COUNT_MEANS[i], prec=50, h="1e-12")),
                               ("negbinomial", nb, exact(_negbinomial_range(lo, hi), COUNT_MEANS[i], shape[i], prec=50, h="1e-12"))):
            mass = math.exp(ref[0])
            tol = 1e-12 if hi is not None or lo == 1 else max(1e-12, 4e-16 / mass)
            for k in range(len(got)):
                assert _close(got[k][i], ref[k], tol * (1 if k == 0 else 100)), (name, mode, lo, hi, COUNT_MEANS[i], k, got[k][i], ref[k])


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_untaken_censoring_branches_stay_finite(mode):
    """At extreme linear predictors and shapes every censored model's log density and gradient are finite: a branch is
    evaluated at a neutral value (the location and twice it) on the rows it does not take."""
    for family in ("weibull", "lognormal", "gamma"):
        m = build_lifted(family, "ypos | cens(c, ypos2) ~ x")
        for intercept, scale in ((40.0, 0.05), (-40.0, 0.05), (25.0, 20.0), (-25.0, 20.0)):
            point = m.value_point(intercept=intercept, b=[3.0], **{get_family(family).scale_parameter: scale})
            if mode == "pytensor":
                lp = float(m.pymc_model.compile_logp()(point))
                grad = np.asarray(m.pymc_model.compile_dlogp()(point), dtype=float)
            else:
                pytest.importorskip("jax")
                import jax
                from pymc.sampling.jax import get_jaxified_logp
                names = [v.name for v in m.pymc_model.value_vars]
                jf = get_jaxified_logp(m.pymc_model)
                vals = [np.asarray(point[n], dtype=float) for n in names]
                lp = float(jf(vals))
                grad = np.concatenate([np.ravel(g) for g in jax.grad(lambda vs: jf(vs))(vals)])
            if np.isfinite(lp):
                assert np.all(np.isfinite(grad)), (family, mode, intercept, scale, grad)


# -- 3. models: their row densities by hand ----------------------------------------------------------------------------------
def build_lifted(family, lhs_rhs, frame=None, group=False, **kwargs):
    """A model built with the parks lifted for the duration of the call (the model keeps its spec)."""
    gamma = replace(fams.FAMILIES["gamma"], parked_additions=MappingProxyType({}))
    saved = fams.FAMILIES, F.PARKED_TRUNCATION
    fams.FAMILIES, F.PARKED_TRUNCATION = MappingProxyType({**fams.FAMILIES, "gamma": gamma}), MappingProxyType({})
    try:
        return getattr(ab, family)(lhs_rhs, FRAME if frame is None else frame, priors(family, group), **kwargs)
    finally:
        fams.FAMILIES, F.PARKED_TRUNCATION = saved


def _rowwise(model):
    pmm = model.pymc_model
    return pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False)[0], inputs=pmm.value_vars, on_unused_input="ignore")


def scipy_distribution(family, eta, aux):
    mu = np.exp(eta)
    if family == "gaussian":
        return stats.norm(eta, aux["sigma"])
    if family == "weibull":
        k = aux["shape"]
        return stats.weibull_min(k, scale=mu / math.gamma(1 + 1 / k))
    if family == "lognormal":
        return stats.lognorm(aux["sigma"], scale=np.exp(eta))
    if family == "gamma":
        return stats.gamma(aux["shape"], scale=mu / aux["shape"])
    if family == "poisson":
        return stats.poisson(mu)
    return stats.nbinom(aux["shape"], aux["shape"] / (aux["shape"] + mu))


def censored_rows(family, eta, aux, codes, frame=FRAME):
    d, y, y2 = scipy_distribution(family, eta, aux), frame.ypos.to_numpy(), frame.ypos2.to_numpy()
    with np.errstate(divide="ignore"):
        interval = np.log(d.cdf(y2) - d.cdf(y))
    return np.where(codes == 0, d.logpdf(y), np.where(codes == 1, d.logsf(y), np.where(codes == -1, d.logcdf(y), interval)))


def truncated_rows(family, eta, aux, lb, ub, response="cnt"):
    d, y = scipy_distribution(family, eta, aux), FRAME[response].to_numpy()
    lo = 0 if lb is None else lb
    mass = (d.cdf(ub) if ub is not None else 1.0) - (d.cdf(lo - 1) if lo > 0 else 0.0)
    return d.logpmf(y) - np.log(mass)


XBAR = FRAME.x.mean()
ETA = 0.3 - 0.4 * (FRAME.x.to_numpy() - XBAR)               # the centred intercept 0.3 (at the mean of x), slope -0.4


@pytest.mark.parametrize("family", CENSORED + ["gaussian"])
@pytest.mark.parametrize("codes", ["c", "c01"])
def test_censored_rows_by_hand(family, codes):
    """Each censored row's log CDF (-1), log CCDF (1) or interval log mass (2) and each other row's log density, SciPy's; the
    Gaussian's interval censoring too (E04 offers cens(c, y2) for every family with cens(); its tails are B02's)."""
    lhs = f"ypos | cens({codes}, ypos2)" if codes == "c" else f"ypos | cens({codes})"
    m = build_lifted(family, f"{lhs} ~ x")
    point = m.value_point(intercept=0.3, b=[-0.4], **NAT[family])
    rows = np.asarray(_rowwise(m)(point), dtype=float)
    np.testing.assert_allclose(rows, censored_rows(family, ETA, NAT[family], FRAME[codes].to_numpy()), rtol=1e-11)
    if codes == "c":
        assert np.array_equal(m.design.rcens, np.where(FRAME.c == 2, FRAME.ypos2, 0.0)) and np.array_equal(m.design.cens, FRAME.c)


@pytest.mark.parametrize("family", TRUNCATED)
@pytest.mark.parametrize("lb,ub", [(None, 6), (None, 8), (1, None), (1, 7)])
def test_truncated_rows_by_hand(family, lb, ub):
    term = ", ".join(f"{s} = {v}" for s, v in (("lb", lb), ("ub", ub)) if v is not None)
    response = "ytr" if lb else "cnt"
    m = build_lifted(family, f"{response} | trunc({term}) ~ x")
    point = m.value_point(intercept=0.3, b=[-0.4], **NAT[family])
    rows = np.asarray(_rowwise(m)(point), dtype=float)
    np.testing.assert_allclose(rows, truncated_rows(family, ETA, NAT[family], lb, ub, response), rtol=1e-11)
    assert np.array_equal(m.design.lb, np.full(10, -np.inf if lb is None else lb)) and np.array_equal(m.design.ub, np.full(10, np.inf if ub is None else ub))


def test_weights_distributional_formulas_and_the_term_model_compose():
    """weights() multiplies each censored or truncated row as a whole (lognormal_censored_interval_weighted and
    poisson_truncated_weighted_post verify it against brms); a formula on sigma or shape gives each row its own; slopes, an
    offset and mo() enter through the linear predictor."""
    w = FRAME.w.to_numpy()
    m = ab.lognormal("ypos | cens(c, ypos2) + weights(w) ~ x", FRAME, priors("lognormal", False))
    rows = np.asarray(_rowwise(m)(m.value_point(intercept=0.3, b=[-0.4], sigma=0.6)), dtype=float)
    np.testing.assert_allclose(rows, w * censored_rows("lognormal", ETA, {"sigma": 0.6}, FRAME.c.to_numpy()), rtol=1e-11)
    m = ab.negbinomial("cnt | trunc(ub = 7) + weights(w) ~ x", FRAME, priors("negbinomial", False))
    rows = np.asarray(_rowwise(m)(m.value_point(intercept=0.3, b=[-0.4], shape=1.4)), dtype=float)
    np.testing.assert_allclose(rows, w * truncated_rows("negbinomial", ETA, {"shape": 1.4}, None, 7), rtol=1e-11)
    # a formula on sigma: each row's sigma = exp(-0.5 + 0.3 (z - mean z))
    p = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma={"intercept": ab.StudentT(3, 0, 2.5), "b": ab.Normal(0, 1)})
    m = ab.lognormal(["ypos | cens(c, ypos2) ~ x", "sigma ~ z"], FRAME, p)
    rows = np.asarray(_rowwise(m)(m.value_point(intercept=0.3, b=[-0.4], sigma={"intercept": -0.5, "b": [0.3]})), dtype=float)
    sig = np.exp(-0.5 + 0.3 * (FRAME.z.to_numpy() - FRAME.z.mean()))
    np.testing.assert_allclose(rows, censored_rows("lognormal", ETA, {"sigma": sig}, FRAME.c.to_numpy()), rtol=1e-11)
    p = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape={"intercept": ab.StudentT(3, 0, 2.5), "b": ab.Normal(0, 1)})
    m = ab.negbinomial(["cnt | trunc(ub = 7) ~ x", "shape ~ z"], FRAME, p)
    rows = np.asarray(_rowwise(m)(m.value_point(intercept=0.3, b=[-0.4], shape={"intercept": 0.2, "b": [0.5]})), dtype=float)
    shp = np.exp(0.2 + 0.5 * (FRAME.z.to_numpy() - FRAME.z.mean()))
    np.testing.assert_allclose(rows, truncated_rows("negbinomial", ETA, {"shape": shp}, None, 7), rtol=1e-11)
    # an offset and a varying slope (Weibull), mo() (Poisson): the linear predictor is the term model's
    m = ab.weibull("ypos | cens(c, ypos2) ~ x + offset(log(exposure)) + (1 + x || g)", FRAME,
                   ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                             shape=ab.Gamma(0.01, 0.01)))
    z_int, z_x = np.array([0.4, -0.3, 0.2]), np.array([-0.5, 0.1, 0.6])
    point = m.value_point(intercept=0.3, b=[-0.4], sd={"g": {"Intercept": 0.5, "x": 0.25}}, z={"g": {"Intercept": z_int, "x": z_x}}, shape=1.7)
    j = pd.Categorical(FRAME.g).codes
    eta = ETA + np.log(FRAME.exposure.to_numpy()) + 0.5 * z_int[j] + 0.25 * z_x[j] * FRAME.x.to_numpy()
    np.testing.assert_allclose(np.asarray(_rowwise(m)(point), dtype=float), censored_rows("weibull", eta, {"shape": 1.7}, FRAME.c.to_numpy()),
                               rtol=1e-11)
    m = ab.poisson("cnt | trunc(ub = 6) ~ mo(ord)", FRAME, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), simo={"ord": ab.Dirichlet(1)}))
    point = m.value_point(intercept=0.2, b=[], bsp={"ord": 0.3}, simo={"ord": [0.4, 0.6]})
    eta = 0.2 + 0.3 * 2 * np.array([0.0, 0.4, 1.0])[FRAME.ord.to_numpy() - 1]      # bsp D sum_{i < ord} simo_i, D = 2 steps
    rows = np.asarray(_rowwise(m)(point), dtype=float)
    np.testing.assert_allclose(rows, truncated_rows("poisson", eta, {}, None, 6), rtol=1e-11)


# -- 4. refusals and the parks ------------------------------------------------------------------------------------------------
REFUSALS = [
    ("cnt | cens(c) ~ x", "poisson", r"'cens\(c\)' is not available for the poisson family; brms's cens\(\) is verified here for the gaussian, "
                                     r"gamma, lognormal, weibull, cox families"),   # F05: cox's right censoring
    ("cnt | cens(c) ~ x", "negbinomial", r"'cens\(c\)' is not available for the negbinomial family"),
    ("ypos | cens(c) ~ x", "exponential", r"'cens\(c\)' is not available for the exponential family"),
    ("ypos | cens(c) ~ x", "student", r"'cens\(c\)' is not available for the student family"),
    ("ypos | trunc(ub = 5) ~ x", "weibull", r"'trunc\(ub = 5\)' is not available for the weibull family; brms's trunc\(\) is verified here for "
                                            r"the gaussian, poisson, negbinomial families"),
    ("ypos | trunc(lb = 0.1) ~ x", "lognormal", r"'trunc\(lb = 0.1\)' is not available for the lognormal family"),
    ("ypos | trunc(lb = 0.1) ~ x", "gamma", r"'trunc\(lb = 0.1\)' is not available for the gamma family"),
    ("cnt | trunc(ub = 6.5) ~ x", "poisson", r"the truncation bounds of a count response are integers .*; ub = 6.5"),
    ("cnt | trunc(ub = -1) ~ x", "negbinomial", r"an upper bound below 0 leaves a count response no value"),
    ("cnt | trunc(ub = 1001) ~ x", "poisson", r"a count response's truncation bound is at most 1000 here"),
    ("ypos | cens(c, ypos2, z) ~ x", "weibull", r"cens\(\) takes one column name or, for interval censoring, cens\(col, y2\)"),
    ("ypos | cens(c, 2) ~ x", "weibull", r"cens\(\) takes the censoring column and, for interval censoring, the column of the intervals' "
                                         r"upper bounds"),
    ("ypos | cens(c, c) ~ x", "weibull", r"the upper bounds y2 must be a column other than the censoring codes"),
    ("ypos | cens(c) + trunc(lb = 0) ~ x", "gaussian", r"cens\(\) together with trunc\(\) is not supported"),
    ("cnt | trunc(ub = 6) + mi() ~ x", "gaussian", r"mi\(\) together with trunc\(ub = 6\)"),
]


@pytest.mark.parametrize("formula,family,needle", REFUSALS)
def test_refusals(formula, family, needle):
    with pytest.raises(ab.FormulaError, match=needle):
        ab.parse_formula(formula, family=family)


def test_the_links_multivariate_and_nonlinear_models_refuse_them():
    for family, link, lhs, needle in (("gamma", "inverse", "ypos | cens(c01)", "is not available under the inverse link of the gamma family"),
                                      ("gamma", "identity", "ypos | cens(c01)", "is not available under the identity link of the gamma family"),
                                      ("poisson", "sqrt", "cnt | trunc(ub = 6)", "is not available under the sqrt link of the poisson family"),
                                      ("negbinomial", "identity", "cnt | trunc(ub = 6)",
                                       "is not available under the identity link of the negbinomial family")):
        with pytest.raises(ab.FormulaError, match=needle):
            getattr(ab, family)(f"{lhs} ~ x", FRAME, priors(family, False), link=link)
    from autobayes.multivariate import parse_multivariate
    with pytest.raises(ab.FormulaError, match=r"addition terms \(weights\(\), cens\(\), trunc\(\), \.\.\.\) on the responses of a multivariate"):
        parse_multivariate("mvbind(ypos, ypos2) | cens(c) ~ x", rescor=True)
    with pytest.raises(ab.FormulaError, match=r"response addition terms .* with a non-linear formula"):
        ab.gaussian(["ypos | cens(c01) ~ a * exp(x)", "a ~ 1"], FRAME, ab.Priors(b={"a": ab.Normal(0, 1)}, sigma=ab.HalfNormal(1)), nl=True)


@pytest.mark.parametrize("codes,y2,needle", [
    ([0, 1, -1, 2] + [0] * 6, None, r"code 2 \(interval censoring\) needs brms's cens\(cc, y2\)"),
    ([0, 1, -1, 3] + [0] * 6, "ypos2", r"censoring codes must be -1 \(left\), 0 \(none\), 1 \(right\) or 2 \(interval\); found 3.0"),
    ([0, 1, -1, 2] + [0] * 6, "x", r"an interval-censored row of 'cens\(cc, x\)' lies in \(y, y2\] with y2 > y; row 3 has y = 0.4 and y2 = 0.3"),
    ([0, 1, -1, 2] + [0] * 6, "g", r"column 'g' has dtype \w+; the upper bounds of 'cens\(cc, g\)' must be numeric"),
])
def test_censoring_codes_and_upper_bounds_are_validated(codes, y2, needle):
    lhs = f"ypos | cens(cc, {y2})" if y2 else "ypos | cens(cc)"
    with pytest.raises(ab.FormulaError, match=needle):
        ab.weibull(f"{lhs} ~ x", FRAME.assign(cc=codes), priors("weibull", False))
    m = ab.weibull("ypos | cens(cc, x) ~ x", FRAME.assign(cc=[0] * 10, x=FRAME.x), priors("weibull", False))   # no interval row: any y2
    assert np.array_equal(m.design.rcens, np.zeros(10))


def test_the_parks_carry_their_reasons():
    assert not F.PARKED_TRUNCATION                      # D121: a count response's lower bound offered since its P6 acceptance
    gamma = get_family("gamma")                         # D120: cens() offered since P6's reference of gamma_censored_interval
    assert "cens" in gamma.addition_terms and not gamma.parked_additions
    for link in ("inverse", "identity"):
        assert not get_family("gamma", link).parked_additions and not get_family("gamma", link).addition_terms
    for name, spec in fams.FAMILIES.items():
        assert set(spec.parked_additions) <= spec.addition_terms, name


@pytest.fixture(scope="module")
def parked_bundles():
    """The two cases once parked, now accepted against two references (P6, D120 and D121), built as before with the parks,
    both empty now, lifted."""
    gamma = replace(fams.FAMILIES["gamma"], parked_additions=MappingProxyType({}))
    saved = fams.FAMILIES, F.PARKED_TRUNCATION
    fams.FAMILIES, F.PARKED_TRUNCATION = MappingProxyType({**fams.FAMILIES, "gamma": gamma}), MappingProxyType({})
    try:
        return {cid: gates.Bundle(cid) for cid in ("gamma_censored_interval", "negbinomial_truncated")}
    finally:
        fams.FAMILIES, F.PARKED_TRUNCATION = saved


def _gradient(b, k):
    vp = adapter.value_point(b.model, b.fixture.points[k], b.cmap, b.gmap)
    layout = adapter.value_layout(b.model, vp)
    ga = np.asarray(b.dlogp_nojac(vp), dtype=float)
    return np.array([ga[layout[var].start + idx] for _, var, idx in sorted(adapter.coordinate_map(b.fixture, b.model, b.cmap, b.gmap))])


def _rel(u, v):
    return np.abs(u - v) / np.maximum(1.0, np.maximum(np.abs(u), np.abs(v)))


def _exact_gamma_shape_gradient(b, k):
    """The exact gradient in log(shape), without the Jacobian, of gamma_censored_interval's log density at point k (80 digits,
    from the fixture's own linear predictor): each row's density, log CDF, log CCDF or interval log mass, and the
    gamma(0.01, 0.01) prior."""
    sd, eta = b.fixture.standata, b.fixture.prediction("fitted", "linpred")[k]
    shape = float(np.ravel(b.fixture.points[k].pars["shape"])[0])

    def total(a):
        out = (Decimal("0.01") - 1) * a.ln() - Decimal("0.01") * a
        for y, y2, c, e in zip(sd["Y"], sd["rcens"], sd["cens"].astype(int), eta):
            rate = a / Decimal(float(e)).exp()
            y, y2 = Decimal(float(y)), Decimal(float(y2))
            if c == 0:
                out += a * rate.ln() - _lgamma(a) + (a - 1) * y.ln() - rate * y
            elif c == -1:
                out += _log_p(a, rate * y)
            elif c == 1:
                out += _log_q(a, rate * y)
            else:
                out += (_log_p(a, rate * y2).exp() - _log_p(a, rate * y).exp()).ln()
        return out
    return exact(total, shape)[1] * shape


def test_gamma_censoring_passes_on_the_reference_where_brms_gradient_is_inexact(parked_bundles, lifted):
    """gamma_censored_interval passes G1, the G2 log densities, G3, G4, G5 and G6 against brms, and G2's gradient by
    two-reference acceptance (P6, D120): brms's gradient in log(shape) is off the exact one at points 0, 2, 3, 4 and 6 (at
    point 2 by 2.8e-5 relative), where AutoBayes's is within 2e-10 (PyTensor's digamma in the observed rows' density, 1e-10
    absolute, is the rest), and there the 50-digit reference stands in for brms."""
    b = parked_bundles["gamma_censored_interval"]
    assert b.case["gate"] == {"expect": "pass", "reference": "gamma_censored_interval.json", "divergence": "D120",
                              "fallback": ["gradient"]}
    for check in (gates.check_design, gates.check_logp, gates.check_derived, gates.check_predictions, gates.check_jax,
                  gates.check_default_prior, gates.check_gradient):
        check(b)
    position = [e["par"] for e in b.fixture.upars].index("shape")
    over = []
    for k, p in enumerate(b.fixture.points):
        ours, theirs = _gradient(b, k), np.asarray(p.grad_nojac)
        rel = _rel(ours, theirs)
        if rel.max() > 1e-7:
            over.append(k)
            assert int(np.argmax(rel)) == position
    assert over == [0, 2, 3, 4, 6]
    for k, brms_off in ((2, 2.7e-5), (0, 1.3e-7)):
        ex = _exact_gamma_shape_gradient(b, k)
        ours, theirs = _gradient(b, k)[position], b.fixture.points[k].grad_nojac[position]
        assert abs(ours - ex) / abs(ex) < 2e-10 and abs(theirs - ex) / abs(ex) > brms_off, (k, ours, theirs, ex)


def test_count_lower_bound_passes_brms_and_its_expected_values_the_reference(parked_bundles, lifted):
    """negbinomial_truncated passes G1, G2 (density and gradient), G3, G5 and G6 against brms; G4's expected values pass on the
    two-reference acceptance (P6, D121) at every entry: brms's are E[Y | Y >= 2] (over a finite support), AutoBayes's and the
    50-digit reference's E[Y | Y >= 1] = mu / (1 - P(0)); without the opt-in they fail against brms."""
    b = parked_bundles["negbinomial_truncated"]
    assert b.case["gate"] == {"expect": "pass", "reference": "negbinomial_truncated.json", "divergence": "D121", "fallback": ["epred"]}
    for check in (gates.check_design, gates.check_logp, gates.check_gradient, gates.check_derived, gates.check_jax, gates.check_default_prior):
        check(b)
    with pytest.warns(ReferenceFallback, match="negbinomial_truncated G4 epred: 656 of 656 entries passed on the reference"):
        gates.check_predictions(b)
    unopted = gates.Bundle("negbinomial_truncated", model=b.model)
    unopted.case = {k: v for k, v in unopted.case.items() if k != "gate"}
    with pytest.raises(OracleMismatch, match=r"fitted: 'mu' differs from brms posterior_epred by 0\.49"):
        gates.check_predictions(unopted)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    ours = b.model.predict(idata, kind="mean")["mu"].transpose("chain", "draw", "__obs__").to_numpy()[0]
    eta, brms = b.fixture.prediction("fitted", "linpred"), b.fixture.prediction("fitted", "epred")
    for k in range(len(b.fixture.points)):
        phi = float(np.ravel(b.fixture.points[k].pars["shape"])[0])
        d = stats.nbinom(phi, phi / (phi + np.exp(eta[k])))
        np.testing.assert_allclose(ours[k], np.exp(eta[k]) / d.sf(0), rtol=1e-13)
        if k == 0:                                               # brms: the mean above 1, not above 0
            np.testing.assert_allclose(brms[k], (np.exp(eta[k]) - d.pmf(1)) / d.sf(1), rtol=1e-12)


# -- 5. predictions, log_lik, post-processing -----------------------------------------------------------------------------
def draws(model, family, n_draws=4, intercept=0.3):
    post = {"Intercept": np.full((1, n_draws), intercept), "x": np.full((1, n_draws), -0.4)}
    for name, value in NAT[family].items():
        post[model.names[name]] = np.full((1, n_draws), value)
    return az.from_dict(posterior=post)


@pytest.mark.parametrize("family", TRUNCATED)
@pytest.mark.parametrize("lb,ub", [(None, 6), (1, None), (2, 7)])
def test_count_truncated_means_and_draws(family, lb, ub):
    """The truncated mean E[Y | lb <= Y <= ub] (brms's posterior_epred for an upper bound; for a lower bound brms's differs,
    D121) and draws from the truncated distribution, whose mean is within five standard errors of it on every row."""
    term = ", ".join(f"{s} = {v}" for s, v in (("lb", lb), ("ub", ub)) if v is not None)
    m = build_lifted(family, f"{'cnt' if lb is None else 'ytr' if ub is None else 'ymid'} | trunc({term}) ~ x")
    post = draws(m, family, n_draws=3000)
    out = m.predict(post, kind="response", seed=5)
    eta = 0.3 - 0.4 * FRAME.x.to_numpy()                      # the posterior's Intercept is the raw-scale intercept
    d = scipy_distribution(family, eta, NAT[family])
    lo = 0 if lb is None else lb
    hi = ub if ub is not None else 400
    ks = np.arange(lo, hi + 1)[:, None]
    weights = d.pmf(ks)
    mean = (ks * weights).sum(0) / weights.sum(0)
    np.testing.assert_allclose(out["mu"].to_numpy()[0, 0], mean, rtol=1e-12)
    y = out.y.to_numpy()[0]
    assert y.dtype == np.int64 and y.min() >= lo and (ub is None or y.max() <= ub)
    sd = np.sqrt(((ks - mean) ** 2 * weights).sum(0) / weights.sum(0))
    assert np.all(np.abs(y.mean(0) - mean) < 5 * sd / np.sqrt(3000))


@pytest.mark.parametrize("family", CENSORED)
def test_censoring_changes_no_prediction(family):
    m, plain = build_lifted(family, "ypos | cens(c, ypos2) ~ x"), build_lifted(family, "ypos ~ x")
    post = draws(m, family)
    for kind in ("linear", "mean"):
        a, b = m.predict(post, kind=kind), plain.predict(post, kind=kind)
        for var in a.data_vars:
            np.testing.assert_array_equal(a[var].to_numpy(), b[var].to_numpy())
    new = pd.DataFrame({"x": [0.5, -1.0]})                    # new rows need neither the codes nor the upper bounds
    np.testing.assert_array_equal(m.predict(post, newdata=new, kind="mean")[get_family(family).mean_name].to_numpy(),
                                  plain.predict(post, newdata=new, kind="mean")[get_family(family).mean_name].to_numpy())


@pytest.mark.parametrize("family,lhs", [("weibull", "ypos | cens(c, ypos2)"), ("lognormal", "ypos | cens(c01) + weights(w)"),
                                        ("gamma", "ypos | cens(c, ypos2)"), ("poisson", "cnt | trunc(ub = 6) + weights(w)"),
                                        ("negbinomial", "cnt | trunc(ub = 8)"), ("negbinomial", "ytr | trunc(lb = 1)"),
                                        ("gaussian", "y | cens(c, ypos2)"), ("gaussian", "ypos | trunc(lb = 0.1, ub = 5)")])
def test_log_lik_is_the_models_own_pointwise_density(family, lhs):
    """brms's log_lik() of a censored or truncated model (verified against brms on lognormal_censored_interval_weighted,
    weibull_censored_interval_post and poisson_truncated_weighted_post): each row's own censored or truncated log density,
    times its weight; the Gaussian's too (before E04 log_lik() left out its censoring and truncation)."""
    frame = FRAME.assign(y=FRAME.ypos - 1.2)
    if family == "gaussian":
        m = ab.gaussian(f"{lhs} ~ x", frame, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
        post = az.from_dict(posterior={"Intercept": np.full((1, 2), 0.3), "x": np.full((1, 2), -0.4), "sigma": np.full((1, 2), 0.7)})
        point = m.value_point(intercept=0.3 - 0.4 * XBAR, b=[-0.4], sigma=0.7)
    else:
        m = build_lifted(family, f"{lhs} ~ x", frame=frame)
        post = draws(m, family, n_draws=2)
        point = m.value_point(intercept=0.3 - 0.4 * XBAR, b=[-0.4], **NAT[family])     # the same raw-scale intercept 0.3
    ll = ab.log_lik(m, post).to_numpy()
    rows = np.asarray(_rowwise(m)(point), dtype=float)
    np.testing.assert_allclose(ll, np.broadcast_to(rows, ll.shape), rtol=1e-11)
    assert np.isfinite(ab.loo(m, post).estimates.loc["elpd_loo", "Estimate"])


def test_hypothesis_and_conditional_effects():
    m = ab.weibull("ypos | cens(c, ypos2) ~ x", FRAME, priors("weibull", False))
    post = draws(m, "weibull")
    h = ab.hypothesis(m, post, ["x < 0"])
    assert h.table["Estimate"].iloc[0] == pytest.approx(-0.4)
    with pytest.raises(ValueError, match=r"the addition term\(s\) cens\(c, ypos2\)"):
        ab.conditional_effects(m, post, ["x"])


# -- 6. describe(), identity, record, prior predictive, defaults --------------------------------------------------------------
def test_describe_identity_record_and_prior_predictive():
    m = ab.weibull("ypos | cens(c, ypos2) ~ x + (1 | g)", FRAME, priors("weibull", True))
    text = m.describe()
    assert ("  censoring : cens(c, ypos2): 2 left-censored row(s) (-1: y_i is an upper bound, log P(Y_i <= y_i)), 2 right-censored "
            "(1: a lower bound, log P(Y_i > y_i)), 3 interval-censored (2: y_i < Y_i <= ypos2_i, log P(y_i < Y_i <= ypos2_i)), "
            "3 observed (0)") in text
    assert "cens(..., ypos2): an interval-censored row contributes log P(y_i < Y_i <= ypos2_i)" in text
    other = ab.weibull("ypos | cens(c, ypos2) ~ x + (1 | g)", FRAME.assign(ypos2=FRAME.ypos * 1.7), priors("weibull", True))
    assert other.target_identity != m.target_identity                            # the upper bounds are data of the target
    assert m.spec.canonical_formula == "ypos | cens(c, ypos2) ~ x + (1 | g)" and m.spec.addition("cens").columns == ("c", "ypos2")
    n = ab.poisson("cnt | trunc(ub = 6) ~ x + (1 | g)", FRAME, priors("poisson", True))
    assert ("  truncation: trunc(ub = 6): the counts lie in {0, ..., 6}, both bounds included" in n.describe()
            and "every row's log mass minus log P(lb <= Y_i <= ub)" in n.describe())
    y = n.prior_predictive(draws=40, seed=2).prior_predictive[n.names["y"]].to_numpy()
    assert y.dtype.kind == "i" and y.min() >= 0 and y.max() <= 6 and (y == 6).any()
    tame = ab.weibull("ypos | cens(c, ypos2) ~ x + (1 | g)", FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 0.5),
                                                                                sd=ab.HalfNormal(0.5), shape=ab.Gamma(4, 2)))
    y = tame.prior_predictive(draws=20, seed=2).prior_predictive[tame.names["y"]].to_numpy()
    assert (y > 0).all() and np.isfinite(y).all() and not np.isin(y, FRAME.ypos.to_numpy()).any()   # draws ignore the censoring


@pytest.mark.parametrize("family,lhs", [("weibull", "ypos | cens(c, ypos2)"), ("lognormal", "ypos | cens(c01)"),
                                        ("poisson", "cnt | trunc(ub = 6)"), ("negbinomial", "cnt | trunc(ub = 8)")])
def test_default_priors_build_the_model(family, lhs):
    """brms's defaults read the response as it is, censored or truncated, so they are the uncensored model's (G6 checks every
    row of brms's table on the frozen cases)."""
    plain = f"{lhs.split(' |')[0]} ~ x + (1 | g)"
    m = getattr(ab, family)(f"{lhs} ~ x + (1 | g)", FRAME, ab.default_priors(f"{lhs} ~ x + (1 | g)", FRAME, family=family))
    other = getattr(ab, family)(plain, FRAME, ab.default_priors(plain, FRAME, family=family))
    assert m.prior_table() == other.prior_table() and "intercept" in m.default_prior_labels


# -- the Weibull's cens() with E03's weights() and shape formula (no frozen case of either pair) -----------------------------
def _weibull_combo_frame():
    rng = np.random.default_rng(3)
    n = 60
    y = rng.weibull(1.7, n) * 2.0 + 0.05
    return pd.DataFrame({"y": y, "x": rng.normal(size=n), "z": rng.normal(size=n), "c": rng.choice([0, 1, -1, 2], size=n),
                         "y2": y + rng.uniform(0.1, 1.0, n), "w": rng.uniform(0.5, 2.0, n)})


def _observed_logp_and_parameters(model):
    """The response's log density (no Jacobian, no priors) at a perturbed initial point, with the graph's mean and shape there."""
    pmm = model.pymc_model
    jitter = np.random.default_rng(11)
    point = {k: v + 0.3 * jitter.normal(size=np.shape(v)) for k, v in pmm.initial_point(random_seed=5).items()}
    obs = pmm.compile_logp(vars=pmm.observed_RVs, jacobian=False)(point)
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"], pmm[model.names["shape"]]]), inputs=pmm.value_vars,
                        point_fn=True, on_unused_input="ignore")
    mu, shape = (np.asarray(v, dtype=float) for v in fn(point))
    return float(obs), mu, shape


def _weibull_censored_rows(df, mu, shape):
    from scipy import stats
    from scipy.special import gammaln

    d = stats.weibull_min(c=shape, scale=mu * np.exp(-gammaln(1.0 + 1.0 / shape)))
    y, y2, c = df["y"].to_numpy(), df["y2"].to_numpy(), df["c"].to_numpy()
    interval = d.logsf(y) + np.log1p(-np.exp(d.logsf(y2) - d.logsf(y)))
    return np.where(c == 0, d.logpdf(y), np.where(c == 1, d.logsf(y), np.where(c == -1, d.logcdf(y), interval)))


def test_weibull_cens_with_weights_and_with_a_shape_formula_equal_scipy():
    """E03 gave the Weibull weights() and a shape formula, E04 cens(): the two pairs, which no frozen case covers, equal SciPy's
    censored Weibull row by row (times the weight; a shape per row), all four censoring codes present."""
    df = _weibull_combo_frame()
    weighted = ab.weibull("y | weights(w) + cens(c, y2) ~ x", df,
                          ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(2, 0.5)))
    obs, mu, shape = _observed_logp_and_parameters(weighted)
    assert abs(obs - float(np.sum(df["w"].to_numpy() * _weibull_censored_rows(df, mu, shape)))) <= 1e-12 * abs(obs)
    formula = ab.weibull(["y | cens(c, y2) ~ x", "shape ~ z"], df,
                         ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}))
    obs, mu, shape = _observed_logp_and_parameters(formula)
    assert shape.shape == (len(df),)
    assert abs(obs - float(np.sum(_weibull_censored_rows(df, mu, shape)))) <= 1e-12 * abs(obs)


# -- 7. the branches no frozen case reaches ------------------------------------------------------------------------------------
@pytest.mark.parametrize("a,x", [(0.31, 0.2), (2.24, 2.45), (7.5, 24.6)])
def test_shape_derivative_op_runs_its_python_impl_on_a_scalar_graph(a, x):
    """On a graph of PyTensor scalars (a tensor graph's Elemwise calls the vectorised NumPy function of the op's
    ``nfunc_spec`` instead) the shape derivative op runs its Python ``impl``: a float64 equal to ``dP/da``, from the 80-digit
    reference as ``P d log P/da`` in the series regime (``x < a + 1``) and ``-Q d log Q/da`` in the continued fraction's."""
    av, xv = ps.float64("a"), ps.float64("x")
    got = pytensor.function([av, xv], T._gammainc_da(av, xv))(a, x)
    if x < a + 1.0:
        lp, dlp, _ = exact(_log_p, a, x)
        ref = math.exp(lp) * dlp
    else:
        lq, dlq, _ = exact(_log_q, a, x)
        ref = -math.exp(lq) * dlq
    assert type(got) is np.float64 and abs(got - ref) <= 1e-13 * abs(ref), (a, x, got, ref)


def test_shape_derivative_lowers_to_jax_in_a_compiled_gradient_graph():
    """A gradient graph of ``gammainc_exact`` and ``gammaincc_exact`` compiled to JAX (not JAX's autodiff of the jaxified value,
    which the tests above take and which needs no lowering) carries the shape derivative op, lowered to XLA's
    ``igamma_grad_a``: equal to PyTensor's gradient (this module's series and continued fraction) to 2e-13 relative."""
    pytest.importorskip("jax")
    a, x = (np.array([p[k] for p in GAMMAINC_GRID]) for k in range(2))
    av, xv = pt.dvector("a"), pt.dvector("x")
    grads = [pt.grad(pt.log(T.gammainc_exact(av, xv)).sum(), av), pt.grad(pt.log(T.gammaincc_exact(av, xv)).sum(), av)]
    on_jax, on_pytensor = pytensor.function([av, xv], grads, mode="JAX"), pytensor.function([av, xv], grads)
    assert any(isinstance(getattr(node.op, "scalar_op", None), type(T._gammainc_da)) for node in on_jax.maker.fgraph.toposort())
    for got, ref in zip(on_jax(a, x), on_pytensor(a, x)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-13)


def test_an_unconverged_series_is_nan_not_a_partial_sum():
    """Near ``x = a`` the terms of the series of ``P`` fall off as ``exp(-n^2 / (2 a))``, so it needs about 8 sqrt(a) of them,
    2.6e5 at ``a = 1e9``: beyond the cap of 100,000 steps, that row is NaN rather than a partial sum, while the other rows of
    the same call converge and are exact."""
    assert T._MAX_STEPS == 100_000
    out = T.gammainc_shape_derivative(np.array([1e9, 2.24]), np.array([1e9 - 0.5, 2.45]))
    lp, dlp, _ = exact(_log_p, 2.24, 2.45)
    assert np.isnan(out[0]) and abs(out[1] - math.exp(lp) * dlp) <= 1e-13 * abs(math.exp(lp) * dlp), out


def test_log_lik_of_a_model_without_cens_or_trunc_is_its_density_unchanged():
    """``censored_truncated_log_lik`` returns the family's density of a model with neither term as it is, the same array
    (``log_lik()`` itself calls it only when one of them is present)."""
    m = ab.gaussian("ypos ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    ll = np.arange(20.0).reshape(1, 2, 10)
    assert m.design.cens is None and m.design.lb is None
    assert lk.censored_truncated_log_lik(m, None, None, ll) is ll


@pytest.mark.parametrize("term,side", [("trunc(lb = 0.1)", "lb"), ("trunc(ub = 5)", "ub")])
def test_gaussian_log_lik_under_one_sided_truncation_by_hand(term, side):
    """log_lik() of a Gaussian truncated on one side (a family without ``log_truncation``): each row's Normal log density
    minus the log mass above the lower bound, ``log Phi((mu - lb) / sigma)``, or below the upper bound, ``log Phi((ub - mu) /
    sigma)``."""
    m = ab.gaussian(f"ypos | {term} ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    post = az.from_dict(posterior={"Intercept": np.full((1, 2), 0.3), "x": np.full((1, 2), -0.4), "sigma": np.full((1, 2), 0.7)})
    eta, y = 0.3 - 0.4 * FRAME.x.to_numpy(), FRAME.ypos.to_numpy()
    mass = log_ndtr((eta - 0.1) / 0.7) if side == "lb" else log_ndtr((5.0 - eta) / 0.7)
    np.testing.assert_allclose(ab.log_lik(m, post).to_numpy(), np.broadcast_to(stats.norm.logpdf(y, eta, 0.7) - mass, (1, 2, 10)),
                               rtol=1e-13)
    assert (m.design.lb[0], m.design.ub[0]) == ((0.1, np.inf) if side == "lb" else (-np.inf, 5.0))


@pytest.mark.parametrize("family,lb", [("poisson", 0), ("negbinomial", -2)])
def test_a_count_lower_bound_at_or_below_zero_truncates_nothing(family, lb):
    """With the park lifted (D121), ``trunc(lb = 0)`` or a negative lower bound keeps the whole support, ``P(Y >= lb) = 1``:
    each row's log density and ``log_lik()`` are the untruncated log mass (SciPy's), the truncated mean is the untruncated
    model's ``mu`` itself."""
    m, plain = build_lifted(family, f"cnt | trunc(lb = {lb}) ~ x"), build_lifted(family, "cnt ~ x")
    eta = 0.3 - 0.4 * FRAME.x.to_numpy()
    logpmf = scipy_distribution(family, eta, NAT[family]).logpmf(FRAME.cnt.to_numpy())
    point = m.value_point(intercept=0.3 - 0.4 * XBAR, b=[-0.4], **NAT[family])     # the raw-scale intercept 0.3
    np.testing.assert_allclose(np.asarray(_rowwise(m)(point), dtype=float), logpmf, rtol=1e-12)
    post = draws(m, family, n_draws=2)
    np.testing.assert_allclose(ab.log_lik(m, post).to_numpy(), np.broadcast_to(logpmf, (1, 2, 10)), rtol=1e-12)
    mean = m.predict(post, kind="mean")["mu"].to_numpy()
    np.testing.assert_array_equal(mean, plain.predict(post, kind="mean")["mu"].to_numpy())
    np.testing.assert_allclose(mean[0, 0], np.exp(eta), rtol=1e-14)
