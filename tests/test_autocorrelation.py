"""Autocorrelation terms (brms-parity C04): brms's ``ar()``, ``ma()`` and ``arma()`` in the regression formulation. The
term and its parsing, the refusals (D68-D71), the series of the fitted rows against brms's own Stan data, the recursion in
NumPy and in the graph against an independent transcription of the model, the density and its gradient, the JAX density,
priors and brms's default priors, ``value_point``, predictions (fitted rows, without group effects, draws, the new-data
refusal), ``describe()``, names, the target identity, fit records, prior-predictive draws, and the oracle's extension. The
row orders, ``J_lag`` and default-prior tables below were printed by brms 2.23.0 (R 4.5.2, C collation) for ``FRAME``.
Nothing here samples; the brms comparison is the oracle cases ``gauss_ar1``, ``gauss_arma11`` and ``gauss_arma22_varint``."""
import copy
import json
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytest
from scipy import stats

import autobayes as ab
from autobayes.autocor import Autocor, arma_effect, arma_values, parse_autocor, series_layout
from autobayes.families import FAMILIES
from autobayes.formula import parse_formula

# brms 2.23.0's example frame: set.seed(1); y, x = round(rnorm(12), 3) each; t and g as written
FRAME = pd.DataFrame({
    "y": [-0.626, 0.184, -0.836, 1.595, 0.330, -0.820, 0.487, 0.738, 0.576, -0.305, 1.512, 0.390],
    "x": [-0.621, -2.215, 1.125, -0.045, -0.016, 0.944, 0.821, 0.594, 0.919, 0.782, 0.075, -1.989],
    "t": [3, 1, 2, 5, 1, 2, 4, 3, 7, 1, 2, 9],
    "g": ["b", "a", "a", "b", "b", "c", "a", "c", "c", "d", "d", "d"],
    "h": ["u", "v"] * 6,
})
# make_standata(y ~ x + ar(t, g, p = 2), FRAME): the rows (1-based) in Stan order, and J_lag
BRMS_ORDER = [2, 3, 7, 5, 1, 4, 6, 8, 9, 10, 11, 12]
BRMS_J_LAG = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
# make_standata(y ~ x + ar(gr = g), FRAME): no time, the data order within each series
BRMS_ORDER_NO_TIME, BRMS_J_LAG_NO_TIME = [2, 3, 7, 1, 4, 5, 6, 8, 9, 10, 11, 12], [1, 1, 0] * 4
# make_standata(y ~ x + ar(time = t), FRAME[c(1, 2, 4, 7, 9, 12), ]): no gr, one series
BRMS_ORDER_NO_GR, BRMS_J_LAG_NO_GR = [2, 1, 4, 3, 5, 6], [1, 1, 1, 1, 1, 0]      # 1-based within those six rows


def priors(**extra):
    base = dict(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), ar=ab.Normal(0, 0.5), ma=ab.Normal(0, 0.5))
    return ab.Priors(**{**base, **extra})


def gaussian(formula="y ~ x + arma(time = t, gr = g, p = 1, q = 1)", frame=None, **extra):
    return ab.gaussian(formula, FRAME if frame is None else frame, priors(**extra))


def reference_mu(eta, y, ar, ma, series, time):
    """The linear predictor with an ARMA term written from its definition, series by series in time order (independent of
    ``autobayes.autocor``): the MA part on the lagged residuals, the residual ``e = y - eta - MA part``, the AR part on the
    same lagged residuals; a lag before a series' first row is 0."""
    eta, mu = np.asarray(eta, dtype=float), np.array(eta, dtype=float)
    for s in sorted(set(series)):
        rows = sorted((i for i in range(len(y)) if series[i] == s), key=lambda i: time[i])
        e = []
        for pos, i in enumerate(rows):
            def lag(k):
                return e[pos - k] if pos - k >= 0 else 0.0
            ma_part = sum(ma[k - 1] * lag(k) for k in range(1, len(ma) + 1))
            e.append(y[i] - eta[i] - ma_part)
            mu[i] = eta[i] + ma_part + sum(ar[k - 1] * lag(k) for k in range(1, len(ar) + 1))
    return mu


# -- the term ---------------------------------------------------------------------------------------------------------
def test_the_term():
    s = parse_formula("y ~ x + arma(time = t, gr = g, p = 1, q = 1)", FRAME)
    assert s.autocor == Autocor("arma", "t", "g", 1, 1) and s.autocor.parts == (("ar", 1), ("ma", 1)) and s.autocor.max_lag == 1
    assert s.terms == ("x",) and s.columns == ("x",) and s.pop_columns == ("x",) and not s.groups
    assert s.canonical_formula == "y ~ x + arma(time = t, gr = g, p = 1, q = 1)" and s.bambi_formula == "y ~ x"
    assert s.autocor.columns == ("t", "g")
    with_group = parse_formula("y ~ x + ar(t, g) + (1 | h)", FRAME)        # R's positional arguments; the term is written last
    assert with_group.autocor == Autocor("ar", "t", "g", 1, 0)
    assert with_group.canonical_formula == "y ~ x + (1 | h) + ar(time = t, gr = g, p = 1)"
    assert with_group.bambi_formula == "y ~ x + (1 | h)"
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):         # asdict hands the term back as a mapping
        assert replica == s and replica.autocor == s.autocor and hash(replica) == hash(s)
    assert parse_formula("y ~ x", FRAME).autocor is None


@pytest.mark.parametrize("piece, expected", [
    ("ar(time = t, gr = g, p = 1)", Autocor("ar", "t", "g", 1, 0)),
    ("ar(t, g)", Autocor("ar", "t", "g", 1, 0)),
    ("ar(t, g, 2)", Autocor("ar", "t", "g", 2, 0)),
    ("ar(gr = g, t, 2L)", Autocor("ar", "t", "g", 2, 0)),                  # R: the named argument first, then the others in order
    ("ar(p = 2.0, time = t)", Autocor("ar", "t", None, 2, 0)),
    ("ar()", Autocor("ar", None, None, 1, 0)),                             # brms's defaults: time = NA, gr = NA, p = 1
    ("ar(time = NA, gr = g)", Autocor("ar", None, "g", 1, 0)),
    ("ma(t, g, 2)", Autocor("ma", "t", "g", 0, 2)),
    ("ma(time = t, gr = g, q = 1, cov = FALSE)", Autocor("ma", "t", "g", 0, 1)),
    ("arma(t, g)", Autocor("arma", "t", "g", 1, 1)),
    ("arma(t, g, 2, 0)", Autocor("arma", "t", "g", 2, 0)),
    ("arma(time = t, gr = g, p = 0, q = 3)", Autocor("arma", "t", "g", 0, 3)),
    ("arma( time=t,gr=g,q=2,p=1 )", Autocor("arma", "t", "g", 1, 2)),
])
def test_brms_arguments(piece, expected):
    assert parse_autocor(piece, "f") == expected
    assert parse_autocor(expected.term, "f") == expected                    # the canonical spelling reads back as itself


def test_the_canonical_spelling():
    assert Autocor("ar", "t", "g", 2, 0).term == "ar(time = t, gr = g, p = 2)"
    assert Autocor("ma", None, "g", 0, 1).term == "ma(gr = g, q = 1)"
    assert Autocor("arma", "t", None, 0, 2).term == "arma(time = t, p = 0, q = 2)"
    assert Autocor("ar", None, None, 1, 0).term == "ar(p = 1)"
    assert parse_autocor("x", "f") is None and parse_autocor("log(x)", "f") is None


FORMULA_REFUSALS = [
    ("y ~ x + ar(t, g, cov = TRUE)", r"cov = TRUE, brms's covariance formulation .* divergence D68"),
    ("y ~ x + arma(t, g, 1, 1, TRUE)", r"divergence D68"),
    ("y ~ x + ar(t, g, cov = maybe)", r"cov must be TRUE or FALSE"),
    ("y ~ x + cosy(t, g)", r"brms's compound symmetry, cosy\(\) is not offered here; .* divergence D69"),
    ("y ~ x + car(M, gr = g)", r"spatial conditional autoregression, car\(\) .* divergence D69"),
    ("y ~ x + sar(M)", r"simultaneous spatial autoregression, sar\(\) .* D69"),
    ("y ~ x + fcor(M)", r"fixed residual covariance matrix, fcor\(\) .* D69"),
    ("y ~ x + unstr(t, g)", r"unstructured residual covariance, unstr\(\) .* D69"),
    ("y ~ x + ar(t, g:h)", r"gr = g:h: gr must be one column name here \(brms's gr = g:h groups the series by two columns.*D70"),
    ("y ~ x + ar(time = log(t), gr = g)", r"time = log\(t\): time must be one column name here"),
    ("y ~ x + ar(t, g, p = 1.5)", r"p = 1.5: Autoregressive order must be a non-negative integer"),
    ("y ~ x + ma(t, g, q = -1)", r"q = -1: Moving-average order must be a non-negative integer"),
    ("y ~ x + ar(t, g, p = 0)", r"at least one of p and q must be greater than zero"),
    ("y ~ x + arma(t, g, 0, 0)", r"at least one of p and q must be greater than zero"),
    ("y ~ x + ar(t, g, q = 1)", r"ar\(\) has no argument 'q'; brms's ar\(\) takes time, gr, p, cov"),
    ("y ~ x + ma(t, g, p = 1)", r"ma\(\) has no argument 'p'"),
    ("y ~ x + ar(t, g, p = 1, p = 2)", r"the argument 'p' is given twice"),
    ("y ~ x + ar(t, g, 1, FALSE, 3)", r"too many arguments"),
    ("y ~ x + ar(t, t)", r"the time column cannot also be the series column gr"),
    ("y ~ x + ar(t, g) + ma(t, g)", r"one autocorrelation term per formula \(brms: 'Can only model one term per autocorrelation "
                                    r"class'\); write arma\(time, gr, p, q\)"),
    ("y ~ x + ar(y, g)", r"the response 'y' cannot be the time or series column"),
    ("y ~ x + ar(t, y)", r"the response 'y' cannot be the time or series column"),
    ("y | weights(t) ~ x + ar(t, g)", r"with the response addition term\(s\) weights\(t\) is not supported in this slice .* D70"),
]


@pytest.mark.parametrize("formula, message", FORMULA_REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula)


def test_the_gaussian_family_alone_takes_the_term():
    """D70: brms's ARMA terms are verified here for the Gaussian family only (brms also takes them for Student-t,
    Poisson, ... in its regression formulation)."""
    assert {name for name, spec in FAMILIES.items() if spec.autocorrelation} == {"gaussian"}
    for family, formula in (("poisson", "cnt ~ x + ar(t, g)"), ("student", "y ~ x + ar(t, g)"), ("bernoulli", "yb ~ x + ar(t, g)"),
                            ("lognormal", "cnt ~ x + ar(t, g)")):
        with pytest.raises(ab.FormulaError, match=rf"the autocorrelation term 'ar\(time = t, gr = g, p = 1\)' is not available for the "
                                                  rf"{family} family; brms's ARMA terms are verified here for the gaussian family .* D70"):
            parse_formula(formula, None, family)


def test_distributional_formulas_refuse_the_term():
    """D70: next to a distributional formula (brms accepts bf(y ~ x + ar(...), sigma ~ z)) and inside one (brms refuses
    that too: 'Explicit covariance terms can only be specified on mu')."""
    pr = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)},
                   ar=ab.Normal(0, 0.5))
    with pytest.raises(ab.FormulaError, match=r"the autocorrelation term 'ar\(time = t, gr = g, p = 1\)' with a distributional formula "
                                              r"is not supported in this slice \(brms accepts .*D70"):
        ab.gaussian(["y ~ x + ar(t, g)", "sigma ~ x"], FRAME, pr)
    with pytest.raises(ab.FormulaError, match=r"an autocorrelation term in a distributional formula \(ar\(time = t, gr = g, p = 1\)\) "
                                              r"is not supported: it belongs to the response formula, as in brms"):
        ab.gaussian(["y ~ x", "sigma ~ x + ar(t, g)"], FRAME, pr)


DATA_REFUSALS = [
    (FRAME.assign(t=[3, 1, 2, 5, 1, 2, 4, 3, 7, 1, 1, 9]), r"rows 9 and 10 of series 'd' have the same time 1\.0; time points within "
                                                           r"a series must be unique \(brms's rule too\)"),
    (FRAME.assign(t=[str(v) for v in FRAME.t]), r"time column 't' of 'arma\(time = t, gr = g, p = 1, q = 1\)' has dtype \w+; it must be "
                                                r"numeric \(brms orders any column, text as text"),
    (FRAME.assign(t=[3.0, np.nan] + [1.0] * 10), r"time column 't' .* has missing or non-finite values"),
    (FRAME.assign(t=[True, False] * 6), r"has dtype bool; it must be numeric"),
    (FRAME.assign(g=["b", None] + ["a"] * 10), r"series column 'g' of .* has missing values"),
    (FRAME.drop(columns="t"), r"columns not in data: \['t'\]"),
]


@pytest.mark.parametrize("frame, message", DATA_REFUSALS)
def test_data_refusals(frame, message):
    with pytest.raises(ab.FormulaError, match=message):
        gaussian(frame=frame)
    with pytest.raises(ab.FormulaError, match=message):                    # the same checks when the formula meets the data
        parse_formula("y ~ x + arma(time = t, gr = g, p = 1, q = 1)", frame)


def test_columns_the_term_shares():
    """The time column may also be a predictor (a trend next to the AR term) and the series column a grouping factor or
    a categorical predictor; a derived grouping factor may not be handed to Bambi under a column the term reads."""
    trend = gaussian("y ~ t + x + ar(t, g) + (1 | g)", sd=ab.HalfNormal(1), ma=None)
    assert trend.spec.terms == ("t", "x") and trend.spec.groups == ("g",)
    categorical = ab.gaussian("y ~ g + ar(t, g)", FRAME, priors(ma=None), categorical={"g": "a"})
    assert categorical.spec.categoricals[0].name == "g"
    with pytest.raises(ab.FormulaError, match=r"handed to Bambi as the column 'g_h', which is also a column of the formula"):
        parse_formula("y ~ x + (1 | g:h) + ar(t, g_h)")


# -- the series of the fitted rows ------------------------------------------------------------------------------------
def test_the_series_are_brms_sorted_rows():
    d = series_layout(Autocor("ar", "t", "g", 2, 0), FRAME)
    assert d.series == ("a", "b", "c", "d") and d.T == 3 and d.lengths.tolist() == [3, 3, 3, 3]
    np.testing.assert_array_equal(d.order + 1, BRMS_ORDER)
    np.testing.assert_array_equal(d.J_lag, BRMS_J_LAG)
    np.testing.assert_array_equal(d.series_index, [1, 0, 0, 1, 1, 2, 0, 2, 2, 3, 3, 3])
    np.testing.assert_array_equal(d.position, [1, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(d.cells, [[1, 4, 5, 9], [2, 0, 7, 10], [6, 3, 8, 11]])
    no_time = series_layout(Autocor("ar", None, "g", 1, 0), FRAME)
    np.testing.assert_array_equal(no_time.order + 1, BRMS_ORDER_NO_TIME)
    np.testing.assert_array_equal(no_time.J_lag, BRMS_J_LAG_NO_TIME)
    rows = FRAME.iloc[[0, 1, 3, 6, 8, 11]].reset_index(drop=True)
    no_gr = series_layout(Autocor("ar", "t", None, 1, 0), rows)
    assert no_gr.series == ("1",)
    np.testing.assert_array_equal(no_gr.order + 1, BRMS_ORDER_NO_GR)
    np.testing.assert_array_equal(no_gr.J_lag, BRMS_J_LAG_NO_GR)
    gaps = series_layout(Autocor("ma", "t", "g", 0, 3), FRAME.assign(t=FRAME.t * 10.5 - 100))   # only the order of time counts
    np.testing.assert_array_equal(gaps.order + 1, BRMS_ORDER)
    np.testing.assert_array_equal(gaps.J_lag, [1, 2, 0] * 4)                # min(max(p, q), position of the next row)


# -- the model --------------------------------------------------------------------------------------------------------
ORDERS = [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1), (1, 3), (3, 2)]


@pytest.mark.parametrize("p, q", ORDERS)
def test_the_recursion_is_the_model(p, q):
    """``arma_values`` (brms's row loop with J_lag) and the graph (a scan over positions, every series at once) are the
    model written from its definition, on series of unequal lengths (one of a single row) in shuffled data order."""
    import pytensor.tensor as pt

    rng = np.random.default_rng(11 + 10 * p + q)
    n = 23
    frame = pd.DataFrame({"y": rng.normal(size=n), "t": rng.permutation(n) * 1.5,
                          "g": rng.choice(list("pqrs"), size=n, p=[.45, .3, .2, .05])})
    frame.loc[0, "g"] = "z"                                                # a series of one row
    d = series_layout(Autocor("arma", "t", "g", p, q), frame)
    eta = rng.normal(size=(2, 3, n))
    ar, ma = rng.uniform(-0.9, 0.9, size=(2, 3, p)), rng.uniform(-0.9, 0.9, size=(2, 3, q))
    y = frame.y.to_numpy()
    want = np.array([[reference_mu(eta[c, k], y, ar[c, k], ma[c, k], frame.g.tolist(), frame.t.tolist()) for k in range(3)]
                     for c in range(2)])
    np.testing.assert_allclose(arma_values(eta, y, ar if p else None, ma if q else None, d), want, rtol=1e-13, atol=1e-13)
    eta_v, ar_v, ma_v = pt.vector("eta"), pt.vector("ar"), pt.vector("ma")
    graph = arma_effect(eta_v, y, ar_v if p else None, ma_v if q else None, d)
    fn = pytensor.function([eta_v, ar_v, ma_v], graph, on_unused_input="ignore")
    np.testing.assert_allclose(fn(eta[1, 2], ar[1, 2], ma[1, 2]), want[1, 2], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("p, q", [(1, 0), (2, 1), (1, 3)])
def test_the_density_is_the_one_written_by_hand(p, q):
    m = gaussian(f"y ~ x + arma(time = t, gr = g, p = {p}, q = {q})", ar=ab.Normal(0, 0.5) if p else None,
                 ma=ab.StudentT(3, 0.1, 0.5) if q else None)
    rng = np.random.default_rng(3)
    vp = m.value_point(intercept=0.4, b=[0.7], sigma=0.8, ar=rng.uniform(-0.8, 0.8, p) if p else None,
                       ma=rng.uniform(-0.8, 0.8, q) if q else None)
    x = FRAME.x.to_numpy()
    eta = 0.4 + (x - x.mean()) * 0.7                                      # the centred intercept and slope
    ar, ma = vp.get("ar", np.zeros(0)), vp.get("ma", np.zeros(0))
    mu = reference_mu(eta, FRAME.y.to_numpy(), ar, ma, FRAME.g.tolist(), FRAME.t.tolist())
    want = (stats.norm.logpdf(FRAME.y, mu, 0.8).sum() + stats.norm.logpdf(0.4, 0, 5) + stats.norm.logpdf(0.7, 0, 2)
            + stats.halfnorm.logpdf(0.8, scale=1) + np.log(0.8) + stats.norm.logpdf(ar, 0, 0.5).sum()
            + stats.t.logpdf(ma, 3, 0.1, 0.5).sum())
    logp, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_dlogp()
    assert float(logp(vp)) == pytest.approx(want, rel=1e-12)
    grad, pos, h = np.asarray(dlogp(vp)), 0, 1e-6                         # against central differences (the MA part's through the scan)
    for v in m.pymc_model.value_vars:
        for j in range(np.size(vp[v.name])):
            up, down = ({**vp, v.name: np.array(vp[v.name], dtype=float).copy()} for _ in range(2))
            up[v.name].flat[j] += h
            down[v.name].flat[j] -= h
            assert grad[pos] == pytest.approx((float(logp(up)) - float(logp(down))) / (2 * h), rel=1e-6, abs=1e-6), (v.name, j)
            pos += 1


@pytest.mark.parametrize("p, q", [(1, 1), (2, 3)])
def test_the_jax_density_is_the_pytensor_one(p, q):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    m = gaussian(f"y ~ x + (1 | h) + arma(time = t, gr = g, p = {p}, q = {q})", sd=ab.HalfNormal(1))
    jf = get_jaxified_logp(m.pymc_model)
    names = [v.name for v in m.pymc_model.value_vars]
    logp, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_dlogp()
    rng = np.random.default_rng(5)
    for _ in range(3):
        vp = m.value_point(intercept=0.4, b=[0.7], sigma=0.8, sd=0.6, z=rng.normal(size=2), ar=rng.uniform(-1.2, 1.2, p),
                           ma=rng.uniform(-1.2, 1.2, q))
        args = [np.asarray(vp[n], dtype=float) for n in names]
        assert float(jf(args)) == pytest.approx(float(logp(vp)), rel=1e-12)
        grads = jax.grad(lambda *a: jf(list(a)), argnums=tuple(range(len(args))))(*args)
        np.testing.assert_allclose(np.concatenate([np.ravel(g) for g in grads]), np.asarray(dlogp(vp)), rtol=1e-10, atol=1e-10)


def test_the_term_composes_with_other_constructs():
    """With group-level terms (the residuals are those of the whole predictor; a correlated term's graph is rebuilt after
    Bambi's), a monotonic term, an offset, no intercept; the build-time check compares each graph with brms's recursion."""
    data = FRAME.assign(ord=[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], expo=np.linspace(1, 2, 12))
    slopes = {"h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}
    for formula, extra in (("y ~ x + (1 + x || h) + arma(t, g, 1, 2)", dict(sd=slopes)),
                           ("y ~ x + (1 + x | h) + arma(t, g, 1, 2)", dict(sd=slopes, cor=ab.LKJ(2))),
                           ("y ~ x + mo(ord) + ma(t, g)", dict(simo=ab.Dirichlet(1), ar=None)),
                           ("y ~ x + offset(log(expo)) + ar(t, g, 2)", dict(ma=None)),
                           ("y ~ 0 + x + arma(t, g)", dict(intercept=None))):
        m = ab.gaussian(formula, data, priors(**extra))
        assert m.spec.autocor is not None and m.design.autocor.n == 12


def test_coefficients_no_row_reads_are_named():
    short = FRAME.assign(g=[f"s{k}" for k in range(6)] * 2, t=[1] * 6 + [2] * 6)        # six series of two rows
    m = gaussian("y ~ x + arma(t, g, 2, 3)", frame=short)
    assert m.warnings == ["no series of arma(time = t, gr = g, p = 2, q = 3) has more than 2 row(s): ['ar[2]', 'ma[2]', 'ma[3]'] multiply "
                          "no lagged residual and are informed by their prior alone (brms keeps them too)"]
    single = gaussian("y ~ x + ar(t, g)", frame=FRAME.assign(g=[f"s{k}" for k in range(12)]), ma=None)
    assert single.warnings[0].startswith("no series of ar(time = t, gr = g, p = 1) has more than 1 row(s): ['ar[1]']")
    assert "WARNING: no series of" in single.describe() and gaussian().warnings == []


def test_the_build_checks_the_terms_contribution(monkeypatch):
    import autobayes.autocor as autocor

    honest = autocor.arma_effect
    monkeypatch.setattr(autocor, "arma_effect", lambda eta, y, ar, ma, design: honest(eta, 0.0 * y, ar, ma, design))   # y - eta read as -eta
    with pytest.raises(RuntimeError, match=r"the graph's 'mu' does not add brms's ARMA term 'arma\(time = t, gr = g, p = 1, q = 1\)'"):
        gaussian()
    monkeypatch.setattr(autocor, "arma_effect", lambda eta, y, ar, ma, design: eta + 0.5 * (honest(eta, y, ar, ma, design) - eta))
    with pytest.raises(RuntimeError, match=r"does not add brms's ARMA term"):
        gaussian("y ~ x + (1 | h) + ar(t, g, 2)", sd=ab.HalfNormal(1), ma=None)


# -- priors -----------------------------------------------------------------------------------------------------------
PRIOR_REFUSALS = [
    (dict(ar=None), r"the autocorrelation term 'arma\(time = t, gr = g, p = 1, q = 1\)' has 1 autoregressive coefficient\(s\), which "
                    r"need a prior: pass ar=\.\.\. \(brms class ar, unbounded; brms's default is flat, ab\.Flat\(\)\)"),
    (dict(ma=None), r"has 1 moving-average coefficient\(s\), which need a prior: pass ma="),
    (dict(ar=ab.HalfNormal(1)), r"prior for 'ar' must be a location prior"),
    (dict(ma="normal(0, 1)"), r"prior for 'ma' must be a Prior, got str"),
]


@pytest.mark.parametrize("extra, message", PRIOR_REFUSALS)
def test_prior_refusals(extra, message):
    with pytest.raises(ab.PriorError, match=message):
        gaussian(**extra)


def test_prior_coverage_is_exact():
    with pytest.raises(ab.PriorError, match=r"do not pass ma; 'ar\(time = t, gr = g, p = 1\)' has no moving-average part"):
        gaussian("y ~ x + ar(t, g)")
    with pytest.raises(ab.PriorError, match=r"do not pass ar; the formula has no autocorrelation term ar\(\), ma\(\) or arma\(\)"):
        gaussian("y ~ x", ma=None)
    m = gaussian("y ~ x + arma(t, g, 2, 1)", ar=ab.StudentT(3, 0, 1), ma=ab.Flat())
    assert m.priors.autocor_priors(m.spec) == {"ar": ab.StudentT(3, 0, 1), "ma": ab.Flat()}
    assert m.priors.improper(m.spec) == ["ma"]
    with pytest.raises(ValueError, match=r"the prior of \['ma'\] is flat"):
        m.prior_predictive(draws=5, seed=1)


# -- brms's default priors --------------------------------------------------------------------------------------------
# brms 2.23.0's default_prior() on FRAME: (class, prior, lb); every other column empty but b's coef x, source "default"
BRMS_DEFAULTS = {
    "y ~ x + arma(t, g, p = 2, q = 2)": [("Intercept", "student_t(3, 0.4, 2.5)", ""), ("ar", "", ""), ("b", "", ""), ("b", "", ""),
                                         ("ma", "", ""), ("sigma", "student_t(3, 0, 2.5)", "0")],
    "y ~ x + ar(t, g)": [("Intercept", "student_t(3, 0.4, 2.5)", ""), ("ar", "", ""), ("b", "", ""), ("b", "", ""),
                         ("sigma", "student_t(3, 0, 2.5)", "0")],
}


@pytest.mark.parametrize("formula", sorted(BRMS_DEFAULTS))
def test_the_default_prior_table_is_brms(formula):
    pri = ab.default_priors(formula, FRAME)
    model = ab.gaussian(formula, FRAME, pri)
    table = model.prior_table()
    assert [(r["class"], r["prior"], r["lb"]) for r in table] == BRMS_DEFAULTS[formula]
    assert [r["coef"] for r in table] == ["", "", "", "x", *([""] if "arma" in formula else []), ""]
    assert all(r["ub"] == r["group"] == r["resp"] == r["dpar"] == r["nlpar"] == r["tag"] == "" and r["source"] == "default" for r in table)
    parts = [part for part, _ in model.spec.autocor.parts]
    assert all(getattr(pri, part) == ab.Flat() and getattr(pri, part).source == "default" for part in parts)
    assert [label for label in model.default_prior_labels if label in ("ar", "ma")] == parts
    assert "  ar[k] ~ Flat()   (brms default: flat, improper), k = 1.." in model.describe()


# -- value_point ------------------------------------------------------------------------------------------------------
def test_value_point():
    m = gaussian("y ~ x + arma(t, g, 2, 1)")
    base = dict(intercept=0.1, b=[0.2], sigma=1.0)
    vp = m.value_point(**base, ar=[0.3, -0.2], ma=0.5)
    np.testing.assert_array_equal(vp["ar"], [0.3, -0.2])
    np.testing.assert_array_equal(vp["ma"], [0.5])
    with pytest.raises(ValueError, match=r"ar is required: the term 'arma\(time = t, gr = g, p = 2, q = 1\)' has 2 ar coefficient\(s\)"):
        m.value_point(**base, ma=0.5)
    with pytest.raises(ValueError, match=r"ar must have 2 finite entries"):
        m.value_point(**base, ar=0.3, ma=0.5)
    with pytest.raises(ValueError, match=r"ma must have 1 finite entries"):
        m.value_point(**base, ar=[0.3, 0.1], ma=[np.inf])
    with pytest.raises(ValueError, match=r"this model has no ma coefficients \('ar\(time = t, gr = g, p = 1\)'\); ma does not apply"):
        gaussian("y ~ x + ar(t, g)", ma=None).value_point(**base, ar=0.3, ma=0.5)
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"this model has no ar coefficients \(no autocorrelation term ar\(\), ma\(\) or arma\(\)\)"):
        plain.value_point(**base, ar=0.3)


# -- predictions ------------------------------------------------------------------------------------------------------
def draws(model, a=0.3, bx=0.7, ar=(0.5, -0.2), ma=(0.4,), u=(0.25, -0.15), sigma=0.9):
    """Two chains of three draws (the term's coefficients differ by draw): the raw-scale intercept (as a posterior from
    fit() holds it), the slope, sigma, ``ar`` and ``ma`` and, with a group term, the varying intercepts of h."""
    scale = np.array([[1.0, 0.9, 1.1], [0.8, 1.2, 1.0]])[..., None]
    post = {"Intercept": np.full((2, 3), a), "x": np.full((2, 3), bx), "sigma": np.full((2, 3), sigma)}
    coords, dims = {}, {}
    for part, values in (("ar", ar), ("ma", ma)):
        if part in model.autocor_names:
            post[part] = scale * np.asarray(values)
            coords[f"{part}_dim"], dims[part] = [str(k) for k in range(1, len(values) + 1)], [f"{part}_dim"]
    if model.spec.groups:
        post.update({"1|h": np.broadcast_to(np.array(u), (2, 3, 2)), "1|h_sigma": np.ones((2, 3))})
        coords["h__factor_dim"], dims["1|h"] = ["u", "v"], ["h__factor_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_add_the_term_on_the_fitted_rows():
    m = gaussian("y ~ x + (1 | h) + arma(t, g, 2, 1)", sd=ab.HalfNormal(1))
    post = draws(m)
    y, x = FRAME.y.to_numpy(), FRAME.x.to_numpy()
    fixed = 0.3 + 0.7 * x
    full = fixed + np.where(FRAME.h == "u", 0.25, -0.15)
    ar, ma = post.posterior["ar"].to_numpy(), post.posterior["ma"].to_numpy()

    def want(eta):
        return np.array([[reference_mu(eta, y, ar[c, k], ma[c, k], FRAME.g.tolist(), FRAME.t.tolist()) for k in range(3)]
                         for c in range(2)])
    for kind, var in (("linear", "eta"), ("mean", "mu")):
        np.testing.assert_allclose(m.predict(post, kind=kind)[var], want(full), rtol=1e-13)
        # without group effects the residuals are those of the predictor without them (brms's re_formula = NA)
        np.testing.assert_allclose(m.predict(post, kind=kind, include_group_effects=False)[var], want(fixed), rtol=1e-13)
    out = m.predict(post, kind="response", seed=7)                         # draws around the predictor with the term
    np.testing.assert_allclose(out["y"], np.random.default_rng(7).normal(want(full), 0.9), rtol=1e-13)
    np.testing.assert_allclose(out["mu"], want(full), rtol=1e-13)


def test_new_rows_are_refused():
    m = gaussian()
    post = draws(m, ar=(0.5,))
    for kind in ("linear", "mean", "response"):
        with pytest.raises(ab.FormulaError, match=r"predictions on new data are not offered for a model with the autocorrelation term "
                                                  r"'arma\(time = t, gr = g, p = 1, q = 1\)'.*divergence D71\); predict on the fitted rows"):
            m.predict(post, newdata=FRAME, kind=kind)


# -- describe(), names, identity, records, prior predictive -----------------------------------------------------------
def test_describe_and_names():
    m = gaussian("y ~ x + (1 | h) + arma(t, g, 2, 1)", sd=ab.HalfNormal(1))
    text = m.describe()
    for line in ("AutoBayes M0 model with autocorrelation term arma(time = t, gr = g, p = 2, q = 1)",
                 "  expanded  : y ~ x + (1 | h) + arma(time = t, gr = g, p = 2, q = 1)   (canonical form handed to Bambi)",
                 "  autocor   : arma(time = t, gr = g, p = 2, q = 1): 4 series (the levels of g) of 3 to 3 rows, each ordered by t; brms's Stan "
                 "data list the rows sorted by series and then time, predictions keep the data order",
                 "  mu_i = a_c + (x_x - mean(x_x)) * b_x + u[h_i] + arma_i",
                 "  arma_i = ar[1] * E_i[1] + ar[2] * E_i[2] + ma[1] * E_i[1]: brms's ARMA term arma(time = t, gr = g, p = 2, q = 1) (the "
                 "regression formulation, cov = FALSE),",
                 "  e_i = y_i - eta_i - (ma[1] * E_i[1]): row i's residual after the MA part (brms's err)",
                 "  ar and ma are unbounded (brms's vector[K] ar): |ar| < 1 is not imposed, so the AR part need not be stationary.",
                 "  ar[k] ~ Normal(mu=0, sigma=0.5), k = 1..2   (brms class ar: each autoregressive coefficient, unbounded)",
                 "  ma[k] ~ Normal(mu=0, sigma=0.5), k = 1..1   (brms class ma: each moving-average coefficient, unbounded)",
                 "  ar                       ar[1..2], the autoregressive coefficients (brms's ar; vector, dims ar_dim, unconstrained)"):
        assert line in text, line
    assert m.autocor_names == {"ar": "ar", "ar_coordinate": "ar_dim", "ma": "ma", "ma_coordinate": "ma_dim"}
    assert m.names["ar"] == "ar" and m.names["ma"] == "ma"
    ma_only = gaussian("y ~ x + ma(gr = g)", ar=None)
    assert ma_only.autocor_names == {"ma": "ma", "ma_coordinate": "ma_dim"} and "ar" not in ma_only.names
    assert "  ma is unbounded (brms's vector[K] ma)." in ma_only.describe()
    assert "one series (no gr)" in gaussian("y ~ x + ar(t)", frame=FRAME.assign(t=range(12, 0, -1)), ma=None).describe()
    assert "each in the data order (no time)" in ma_only.describe()
    for column in ("ar", "ma_dim"):
        with pytest.raises(ab.FormulaError, match=rf"generated variable/coordinate name collision: \['{column}'\]"):
            gaussian(f"y ~ x + {column} + arma(t, g)", frame=FRAME.assign(**{column: FRAME.x * 2}))


def test_the_target_identity_holds_the_term():
    base = gaussian().target_identity
    assert gaussian().target_identity == base
    assert gaussian("y ~ x + arma(t, g, 1, 2)").target_identity != base
    assert gaussian(ar=ab.Normal(0, 1)).target_identity != base
    assert gaussian(frame=FRAME.assign(t=-FRAME.t)).target_identity != base           # time reversed: other lags
    assert gaussian(frame=FRAME.assign(t=FRAME.t * 2 + 1)).target_identity == base     # the same order
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    assert plain.target_identity != base


def test_fit_records_and_summary_rows(monkeypatch):
    m = gaussian("y ~ x + arma(t, g, 2, 1)")
    post = draws(m)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = session._summary_rows(m, post)
    assert {"ar[1]", "ar[2]", "ma[1]"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=3,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["ar"] == rec["model"]["priors"]["ma"] == "Normal(mu=0, sigma=0.5)"
    assert rec["model"]["autocorrelation"] == {"term": "arma(time = t, gr = g, p = 2, q = 1)", "time": "t", "gr": "g", "p": 2, "q": 1,
                                               "series": 4, "names": m.autocor_names}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_prior_predictive_draws_the_coefficients():
    pp = gaussian("y ~ x + arma(t, g, 2, 1)").prior_predictive(draws=40, seed=3)
    assert pp.prior["ar"].shape == (1, 40, 2) and pp.prior["ma"].shape == (1, 40, 1)
    assert np.isfinite(pp.prior_predictive["y"]).all()


# -- the oracle's extension to autocorrelation terms ------------------------------------------------------------------
def _bundle(case="gauss_arma11", model=None):
    from tests.oracle import gates

    return gates.Bundle(case, model=model)


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_oracle_compares_brms_sorted_rows_through_the_derived_permutation():
    """brms's Stan rows are the data sorted by (g, t); the adapter derives that order from the data, checks brms's Y against
    it, and puts every per-row array back in data order (J_lag stays on brms's rows)."""
    from tests.oracle import adapter
    from tests.oracle import fixture as fx

    b = _bundle("gauss_arma22_varint")
    raw = fx.Fixture.load("gauss_arma22_varint").standata
    order = adapter.brms_row_order(b.case, b.data, b.model.spec.autocor)
    np.testing.assert_array_equal(b.fixture.stan_order, order)
    np.testing.assert_array_equal(order, b.model.design.autocor.order)        # AutoBayes sorts the series as brms does here
    assert not np.array_equal(order, np.arange(order.size))                    # the data are not in Stan order
    np.testing.assert_array_equal(b.fixture.standata["Y"], b.model.design.y)
    np.testing.assert_array_equal(b.fixture.standata["Y"][order], raw["Y"])
    np.testing.assert_array_equal(b.fixture.standata["J_1"][order], raw["J_1"])
    np.testing.assert_array_equal(b.fixture.standata["J_lag"], raw["J_lag"])
    np.testing.assert_array_equal(b.model.design.autocor.J_lag, raw["J_lag"])


def test_the_gates_see_a_corrupted_autocorrelation_fixture():
    from tests.oracle import adapter, gates
    from tests.oracle.adapter import OracleMismatch

    b = _bundle()
    for gate in gates.ALL:
        gate(b)

    def J_lag(raw):
        J = raw["standata"]["J_lag"]["values"]
        J[3], J[4] = J[4], J[3]                                                 # a series ends one row early
    with pytest.raises(OracleMismatch, match=r"brms's J_lag .* differs from the lags of AutoBayes's series positions"):
        gates.check_design(_corrupted(b, J_lag))

    def order(raw):
        Y = raw["standata"]["Y"]["values"]
        Y[0], Y[1] = Y[1], Y[0]                                                 # two rows of a series swapped
    with pytest.raises(OracleMismatch, match=r"brms's Y is not the response sorted by \('t', 'g'\) \(brms's order_data\(\)\)"):
        _corrupted(b, order)

    def Kma(raw):
        raw["standata"]["Kma"]["values"][0] = 2
    with pytest.raises(OracleMismatch, match=r"brms has Kar = 1, Kma = 2; arma\(time = t, gr = g, p = 1, q = 1\) has p = 1, q = 1"):
        gates.check_design(_corrupted(b, Kma))

    def gradient(raw):
        i = [k for k, e in enumerate(raw["upars"]) if e["par"] == "ma"][0]
        raw["points"][3]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 3: gradient of ma\[0\]"):
        gates.check_gradient(_corrupted(b, gradient))

    def coefficient(raw):
        raw["points"][6]["pars"]["ar"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 6: log density"):
        gates.check_logp(_corrupted(b, coefficient))

    def linpred(raw):
        raw["predictions"]["fitted"]["linpred"]["values"][17] += 1e-6
    with pytest.raises(OracleMismatch, match="fitted: linear predictor differs from brms posterior_linpred"):
        gates.check_predictions(_corrupted(b, linpred))

    def default(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "ma"]
        row["prior"] = "normal(0, 1)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(b, default))

    reversed_time = adapter.build_model(b.case, b.data.assign(t=-b.data.t))   # another model: each series read backwards
    with pytest.raises(OracleMismatch, match=r"Stan row 1 starts brms's series 'a'; AutoBayes puts it at position 4 of series 'a'"):
        gates.check_design(_bundle(model=reversed_time))
    with pytest.raises(OracleMismatch, match=r"point 0: log density"):
        gates.check_logp(_bundle(model=reversed_time))
    no_ma = ab.gaussian("y ~ x + ar(time = t, gr = g, p = 1)", b.data, priors(ma=None))
    with pytest.raises(OracleMismatch, match=r"brms has Kar = 1, Kma = 1; ar\(time = t, gr = g, p = 1\) has p = 1, q = 0"):
        gates.check_design(_bundle(model=no_ma))
    plain = ab.gaussian("y ~ x", b.data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(OracleMismatch, match=r"design column 'x' matches 0 brms columns"):     # brms's sorted rows, unmatched
        _bundle(model=plain)
