"""brms's gp() beyond one input (brms-parity G09, docs/gaussian-processes.md "Several inputs, by, other kernels and
scale = FALSE"): several inputs (one kernel of the Euclidean distance, or with iso = FALSE one length-scale each), by = a factor
of the model (one process per level, each level's inputs scaled by its own dmax) or a numeric column (the process times it),
the Matern 3/2, Matern 5/2 and exponential kernels (exact, and approximate with their spectral densities) and scale = FALSE.

Checked here against computations written out independently of autobayes.gaussian_process (the kernels of the Stan Functions
Reference, the spectral densities as numerical Fourier transforms of the kernels, the log density and predictions by hand),
against brms 2.23.0's default_prior() on probe frames (printed in the comments), and the refusals of every combination no frozen
case verifies (D195, D196). The brms comparison of each construct is the oracle cases gauss_gp_two_inputs,
gauss_gp_two_inputs_aniso, gauss_gp_by_factor, gauss_gp_by_numeric, gauss_gp_matern32, gauss_gp_matern52, gauss_gp_exponential,
gauss_gp_approx_matern32, gauss_gp_approx_matern52, gauss_gp_approx_exponential, gauss_gp_scale_false and
gauss_gp_matern32_default_prior. Nothing here samples."""
import copy
import json
import math
import warnings
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import autobayes as ab
from autobayes import gaussian_process as gpm
from autobayes.formula import GaussianProcess, gaussian_process_coding, parse_formula

FRAME = pd.DataFrame({
    "y": [0.31, -1.2, 2.05, 0.77, -0.4, 1.66, 0.12, -0.95, 2.41, 1.08, 0.5, -0.33],
    "ord": [1, 2, 3, 4, 2, 3, 1, 4, 3, 2, 1, 4],
    "o2": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    "b2": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],                       # every (o2, b2) pair, the first six rows in order
    "z": ["p", "q", "r"] * 4,
    "w": [0.2, -1.1, 0.7, 1.5, -0.3, 0.05, -2.0, 0.8, 1.2, -0.6, 0.35, -0.1],
    "x": [0.7, -0.2, 1.3, 1.9, -0.5, 3.1, 0.4, 2.4, -0.9, 1.1, 2.0, 0.3],   # twelve distinct values
    "flag": [True, False] * 6,
    "g": list("aabbccddeeff"),
    "cnt": [0, 3, 1, 1, 7, 2, 0, 0, 12, 4, 1, 0],
})
CAT = {"z": "p"}
PAIRS = [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)]          # the distinct (o2, b2) rows in first-appearance order
PAIR_INDEX = np.array([0, 1, 2, 3, 4, 5, 0, 4, 2, 3, 1, 5])
ROOT5 = math.sqrt(5.0)
# z's levels: p on rows 0, 3, 6, 9 (ord 1, 4, 1, 2: distinct 1, 4, 2, dmax 3), q on 1, 4, 7, 10 (ord 2, 2, 4, 1: distinct 2, 4,
# 1, dmax 3), r on 2, 5, 8, 11 (ord 3, 3, 3, 4: two distinct inputs, dmax 1)
LEVELS = {"p": ([0, 3, 6, 9], [1.0, 4.0, 2.0], [0, 1, 0, 2], 3.0), "q": ([1, 4, 7, 10], [2.0, 4.0, 1.0], [0, 0, 1, 2], 3.0),
          "r": ([2, 5, 8, 11], [3.0, 4.0], [0, 0, 0, 1], 1.0)}


def priors(**extra):
    base = dict(intercept=ab.Normal(0, 5), sigma=ab.HalfNormal(1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    return ab.Priors(**{**base, **extra})


def gaussian(formula, frame=None, categorical=None, **extra):
    return ab.gaussian(formula, FRAME if frame is None else frame, priors(**extra), categorical=categorical)


# -- the kernels, written out from the Stan Functions Reference -------------------------------------------------------
def _k(r, cov):
    r = np.asarray(r, dtype=float)
    if cov == "exp_quad":
        return np.exp(-0.5 * r ** 2)
    if cov == "matern32":
        return (1 + math.sqrt(3) * r) * np.exp(-math.sqrt(3) * r)
    if cov == "matern52":
        return (1 + math.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-math.sqrt(5) * r)
    return np.exp(-r)


def _cov(X, sdgp, lscale, cov="exp_quad"):
    """K + 1e-12 I: one length-scale (the Euclidean distance) or one per column (the product of one-dimensional kernels)."""
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    if np.ndim(lscale) == 0:
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        K = sdgp ** 2 * _k(d / lscale, cov)
    else:
        K = sdgp ** 2 * np.prod([_k(np.abs(X[:, None, j] - X[None, :, j]) / l, cov) for j, l in enumerate(lscale)], axis=0)
    return K + 1e-12 * np.eye(len(X))


def _f(X, sdgp, lscale, z, cov="exp_quad"):
    return np.linalg.cholesky(_cov(X, sdgp, lscale, cov)) @ np.asarray(z, dtype=float)


def _spd(w, lscale, cov):
    """The one-dimensional spectral density of the kernel (sdgp = 1) as its Fourier transform, 2 int_0^inf k(t) cos(w t) dt,
    integrated numerically: an independent check of the closed forms of autobayes.gaussian_process."""
    val, _ = integrate.quad(lambda t: _k(t / lscale, cov), 0, np.inf, weight="cos", wvar=w, limlst=200)
    return 2 * val


def _priors_logp(sdgps, lscales, zs, intercept=0.0, sigma=1.0):
    return (stats.norm.logpdf(intercept, 0, 5) + stats.halfnorm.logpdf(sigma) + sum(stats.halfnorm.logpdf(s) for s in sdgps)
            + sum(stats.invgamma.logpdf(l, 3, scale=1) for l in lscales) + sum(stats.norm.logpdf(z).sum() for z in zs))


def _logp(m, point):
    return float(m.pymc_model.compile_logp(jacobian=False)(point))


# -- the term ---------------------------------------------------------------------------------------------------------
TERMS = [
    ("y ~ gp(o2, b2)", "gp(o2, b2)", "o2b2", ("o2b2",)),
    ("y ~ gp(o2, b2, iso = FALSE)", "gp(o2, b2, iso = FALSE)", "o2b2", ("o2b2o2", "o2b2b2")),
    ("y ~ gp(o2, b2, iso = TRUE)", "gp(o2, b2)", "o2b2", ("o2b2",)),
    ("y ~ gp(ord, by = w)", "gp(ord, by = w)", "ordw", ("ordw",)),
    ("y ~ z + gp(ord, by = z, cmc = TRUE)", "gp(ord, by = z)", "ordz", ("ordz",)),
    ('y ~ gp(x, cov = "matern32")', 'gp(x, cov = "matern32")', "x", ("x",)),
    ("y ~ gp(x, cov = 'matern52')", 'gp(x, cov = "matern52")', "x", ("x",)),
    ('y ~ gp(x, k = 6, cov = "exponential")', 'gp(x, k = 6, c = 1.25, cov = "exponential")', "x", ("x",)),
    ("y ~ gp(ord, scale = F)", "gp(ord, scale = FALSE)", "ord", ("ord",)),
    ('y ~ gp(x, cov = "exp_quad", scale = TRUE)', "gp(x)", "x", ("x",)),
]


@pytest.mark.parametrize("formula, term, key, lscale_keys", TERMS)
def test_the_term_of_each_form(formula, term, key, lscale_keys):
    g = parse_formula(formula, None, categorical=CAT if "by = z" in formula else None).gp_terms[0]
    assert (g.term, g.key, g.lscale_keys) == (term, key, lscale_keys)
    assert g.plain == (term in ("gp(x)",)) and not g.known


def test_the_dataclass_checks_its_fields():
    with pytest.raises(ValueError, match=r"the first input of a Gaussian process is its column 'x', got \('z', 'x'\)"):
        GaussianProcess("x", columns=("z", "x"))
    with pytest.raises(ValueError, match=r"the kernel of a Gaussian process is one of .*'matern72'"):
        GaussianProcess("x", cov="matern72")
    two = GaussianProcess("o2", columns=("o2", "b2"), inputs=[[1, 0], [2, 0]], dmax=1.0, cmean=[1.5, 0.0])
    assert two.inputs == ((1.0, 0.0), (2.0, 0.0)) and two.cmean == (1.5, 0.0) and two.ndim == 2


REFUSALS = [
    ("y ~ gp(o2, o2)", r"the column 'o2' is given twice as an input of one Gaussian process"),
    ("y ~ gp(o2, log(b2))", r"gp\(\) takes a column name here, not an expression \('log\(b2\)'"),
    ("y ~ gp(ord, by = log(w))", r"by takes a column name here, not an expression \('log\(w\)'\).*D196"),
    ("y ~ gp(ord, by = ord)", r"'ord' is an input of the Gaussian process and its by variable; one column cannot be both"),
    ("y ~ gp(ord, by = y)", r"the response 'y' cannot be the by variable of a Gaussian process"),
    ('y ~ gp(x, cov = "matern72")', r"cov = \"matern72\" is not offered here; cov is one of brms's kernels 'exp_quad', 'matern32', "
                                    r"'matern52', 'exponential', quoted"),
    ("y ~ gp(x, cov = matern32)", r"cov = matern32 is not offered here"),
    ("y ~ gp(o2, b2, iso = maybe)", r"iso takes TRUE or FALSE, got 'maybe'"),
    ("y ~ gp(x, gr = maybe)", r"gr takes TRUE or FALSE, got 'maybe'"),
    ("y ~ gp(x, iso = TRUE)", r"gp\(\)'s iso decides whether several covariates share one length-scale.*drop it.*D64"),
    ("y ~ gp(x, cmc = TRUE)", r"gp\(\)'s cmc only matters with a factor by =: drop it.*D64"),
    ("y ~ gp(ord, by = w, cmc = maybe)", r"cmc takes TRUE or FALSE, got 'maybe'"),
    ("y ~ gp(ord, by = w, cmc = FALSE)", r"cmc = FALSE \(contrast Gaussian processes of a factor by.*is not supported.*D196"),
    ('y ~ gp(o2, b2, iso = FALSE, cov = "matern32")', r"iso = FALSE with cov = \"matern32\" is not supported: brms's exact Gaussian "
                                                      r"process then multiplies one one-dimensional matern32 kernel per input.*D195"),
    ('y ~ gp(o2, b2, k = 5, iso = FALSE, cov = "exponential")', r"iso = FALSE with cov = \"exponential\" is not supported.*D195"),
    ("y ~ gp(o2, b2, by = w)", r"gp\(\) with several inputs \(o2, b2\) and by = w is not supported: each is verified against brms on its "
                               r"own.*D196"),
    ('y ~ gp(o2, b2, cov = "matern32")', r"several inputs \(o2, b2\) and cov = \"matern32\" is not supported.*D196"),
    ("y ~ gp(o2, b2, k = 5)", r"several inputs \(o2, b2\) and the approximate form \(k\) is not supported.*D196"),
    ('y ~ gp(ord, by = w, cov = "matern52")', r"by = w and cov = \"matern52\" is not supported.*D196"),
    ("y ~ gp(ord, by = w, k = 4)", r"by = w and the approximate form \(k\) is not supported.*D196"),
    ("y ~ gp(ord, by = w, scale = FALSE)", r"by = w and scale = FALSE is not supported.*D196"),
    ('y ~ gp(x, cov = "matern32", scale = FALSE)', r"cov = \"matern32\" and scale = FALSE is not supported.*D196"),
    ('y ~ gp(x, k = 5, cov = "matern32", scale = FALSE)', r"the approximate form \(k\) and cov = \"matern32\" and scale = FALSE"),
    ("y ~ gp(o2, b2) + gp(b2)", r"'gp\(b2\)': a second Gaussian process of 'b2' \(after 'gp\(o2, b2\)'\); one gp\(\) per column"),
    ("y ~ gp(o2, b2) + gp(o2b2)", r"'gp\(o2b2\)' and 'gp\(o2, b2\)' would share the name 'o2b2' of their parameters"),
]


@pytest.mark.parametrize("formula, message", REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula, None)


def test_other_refusals_name_the_construct():
    with pytest.raises(ab.FormulaError, match=r"the trials column 'cnt' cannot be the by variable of a Gaussian process"):
        parse_formula("y | trials(cnt) ~ gp(x, by = cnt)", None, "binomial")
    with pytest.raises(ab.FormulaError, match=r"the trials column 'cnt' cannot be the input of a Gaussian process"):
        parse_formula("y | trials(cnt) ~ gp(x, cnt)", None, "binomial")
    with pytest.raises(ab.FormulaError, match=r"'gp\(o2, b2\)': 'b2' is declared categorical; the input of a Gaussian process"):
        parse_formula("y ~ b2 + gp(o2, b2)", FRAME, categorical={"b2": 0})


def test_coding_refusals():
    """Read off the data: a level of a factor by with one distinct input (brms: 'Could not scale GP covariates'), a logical or
    ordered factor by (D196), a by that is neither numeric nor a categorical predictor, a by or input with missing values, a
    process of several inputs with one distinct input row, and two parameters that would share a name."""
    with pytest.raises(ab.FormulaError, match=r"column 'ord' of the Gaussian process 'gp\(ord, by = z\)' at level 'r' of 'z' takes the "
                                              r"single value 3\.0 in the fitted rows; a Gaussian process needs at least two distinct"):
        parse_formula("y ~ z + gp(ord, by = z)", FRAME.assign(ord=np.where(FRAME.z == "r", 3, FRAME.ord)), categorical=CAT)
    with pytest.raises(ab.FormulaError, match=r"by = flag is a logical column; a factor by is verified for an unordered factor only.*D196"):
        gaussian("y ~ flag + gp(ord, by = flag)", b=ab.Normal(0, 1))
    ordered = FRAME.assign(z=pd.Categorical(FRAME.z, categories=["p", "q", "r"], ordered=True))
    with pytest.raises(ab.FormulaError, match=r"by = z is an ordered factor \(polynomial contrasts\); a factor by is verified"):
        gaussian("y ~ z + gp(ord, by = z)", ordered, b=ab.Normal(0, 1))
    with pytest.raises(ab.FormulaError, match=r"column 'g' \(the by variable of the Gaussian process 'gp\(ord, by = g\)'\) has dtype "
                                              r".*a by variable is a numeric column or a categorical predictor of the formula \(g \+ "):
        parse_formula("y ~ gp(ord, by = g)", FRAME)
    with pytest.raises(ab.FormulaError, match=r"by variable of the Gaussian process 'gp\(ord, by = w\)'\) has missing or non-finite"):
        gaussian_process_coding(parse_formula("y ~ gp(ord, by = w)", None).gp_terms[0], FRAME.assign(w=np.inf))
    with pytest.raises(ab.FormulaError, match=r"the columns \['o2', 'b2'\] of the Gaussian process 'gp\(o2, b2\)' take the single input "
                                              r"row \(1\.0, 0\.0\)"):
        parse_formula("y ~ gp(o2, b2)", FRAME.assign(o2=1, b2=0))
    with pytest.raises(ab.FormulaError, match=r"column 'b2' of the Gaussian process 'gp\(o2, b2\)' has dtype"):
        parse_formula("y ~ gp(o2, b2)", FRAME.assign(b2=FRAME.flag))
    with pytest.raises(ab.FormulaError, match=r"has missing or non-finite values"):
        parse_formula("y ~ gp(o2, b2)", FRAME.assign(b2=np.where(FRAME.index == 3, np.inf, FRAME.b2)))
    clash = FRAME.assign(ordzp=FRAME.x)                       # gp(ordzp)'s parameters and level p's of gp(ord, by = z)
    with pytest.raises(ab.FormulaError, match=r"two parameters of the Gaussian processes .* would share the name 'ordzp'"):
        parse_formula("y ~ z + gp(ord, by = z) + gp(ordzp)", clash, categorical=CAT)


def test_the_coding_of_each_form():
    two = parse_formula("y ~ gp(o2, b2)", FRAME).gp("o2b2")
    assert two.inputs == tuple((float(a), float(b)) for a, b in PAIRS) and two.dmax == ROOT5
    np.testing.assert_array_equal(two.index(FRAME), PAIR_INDEX)
    np.testing.assert_allclose(two.scaled_inputs, np.array(PAIRS) / ROOT5, rtol=1e-15)
    assert two.cmeans == pytest.approx(np.mean(np.array(PAIRS) / ROOT5, axis=0), rel=1e-15)
    with pytest.raises(ab.FormulaError, match=r"has the input rows \[\(4\.0, 0\.0\)\], which are not inputs of the exact Gaussian "
                                              r"process 'gp\(o2, b2\)': it has values at its 6 fitted input rows only.*D65"):
        two.index(pd.DataFrame({"o2": [1, 4], "b2": [0, 0]}), where="newdata column")
    by = parse_formula("y ~ z + gp(ord, by = z)", FRAME, categorical=CAT).gp("ordz")
    assert by.is_by_factor and not by.by_numeric and [p.key for p in by.subprocesses] == ["ordzp", "ordzq", "ordzr"]
    for p in by.parts:
        rows, inputs, index, dmax = LEVELS[p.level]
        assert p.inputs == tuple(inputs) and p.dmax == dmax and p.lscale_keys == (p.key,)
        np.testing.assert_array_equal(by.level_rows(FRAME)[p.level], rows)
        np.testing.assert_array_equal(p.index(FRAME.iloc[rows]), index)
    with pytest.raises(ab.FormulaError, match=r"newdata column 'z' has the levels \['s'\], which the fit did not see"):
        by.level_rows(pd.DataFrame({"z": ["p", "s"]}), where="newdata column")
    with pytest.raises(ab.FormulaError, match=r"at level 'q' of 'z': it has values"):
        by.parts[1].index(pd.Series([3.0]))
    numeric = parse_formula("y ~ gp(ord, by = w)", FRAME).gp("ordw")
    assert numeric.by_numeric and numeric.subprocesses == (numeric,)
    np.testing.assert_array_equal(numeric.multiplier(FRAME), FRAME.w.to_numpy())
    unscaled = parse_formula("y ~ gp(ord, scale = FALSE)", FRAME).gp("ord")
    assert unscaled.dmax == 1.0 and unscaled.cmean == 2.5
    np.testing.assert_array_equal(unscaled.scaled_inputs, [1.0, 2.0, 3.0, 4.0])
    assert ab.gaussian("y ~ gp(ord)", FRAME, priors()).spec.pop_columns == ("ord",)
    assert parse_formula("y ~ z + gp(ord, by = z)", FRAME, categorical=CAT).pop_columns == ("z", "ord")
    with pytest.raises(KeyError):
        parse_formula("y ~ gp(ord)", FRAME).gp_owner("x")


# -- the mathematics ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cov", ["exp_quad", "matern32", "matern52", "exponential"])
def test_the_kernels_are_the_functions_reference(cov):
    X = np.array(PAIRS) / ROOT5
    np.testing.assert_allclose(gpm.covariance(X, 1.3, 0.4, cov), _cov(X, 1.3, 0.4, cov), rtol=1e-14, atol=1e-15)
    one = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(gpm.covariance(one, 0.7, 0.3, cov), _cov(one, 0.7, 0.3, cov), rtol=1e-14, atol=1e-15)
    draws = gpm.covariance(X, np.array([1.0, 2.0]), np.array([0.4, 0.8]), cov)        # draws broadcast
    np.testing.assert_allclose(draws[1], _cov(X, 2.0, 0.8, cov), rtol=1e-14, atol=1e-15)
    z = np.linspace(-1, 1, 6)
    np.testing.assert_allclose(gpm.exact_values(X, 1.3, 0.4, z, cov), _f(X, 1.3, 0.4, z, cov), rtol=1e-12, atol=1e-14)


def test_one_length_scale_per_input():
    X = np.array(PAIRS) / ROOT5
    np.testing.assert_allclose(gpm.covariance(X, 1.3, [0.4, 0.9]), _cov(X, 1.3, [0.4, 0.9]), rtol=1e-14, atol=1e-15)
    z = np.linspace(-1, 1, 6)
    np.testing.assert_allclose(gpm.exact_values(X, 1.3, [0.4, 0.9], z), _f(X, 1.3, [0.4, 0.9], z), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match=r"one length-scale per input is offered for the squared-exponential kernel only"):
        gpm.covariance(X, 1.0, [0.4, 0.9], "matern32")
    import pytensor.tensor as pt
    with pytest.raises(ValueError, match=r"one length-scale per input is offered for the squared-exponential kernel only"):
        gpm.exact_effect(X, pt.as_tensor(1.0), [pt.as_tensor(0.4), pt.as_tensor(0.9)], pt.as_tensor(z), cov="matern32")


@pytest.mark.parametrize("cov", ["matern32", "matern52", "exponential"])
def test_the_spectral_densities_are_the_kernels_fourier_transforms(cov):
    w = np.array([0.3, 1.7, 4.2])
    for lscale in (0.2, 0.9):
        want = np.sqrt([_spd(v, lscale, cov) for v in w])
        np.testing.assert_allclose(gpm.sqrt_spectral_density(w, 1.0, lscale, cov), want, rtol=1e-7)
        np.testing.assert_allclose(gpm.sqrt_spectral_density(w, 2.5, lscale, cov), 2.5 * want, rtol=1e-7)


@pytest.mark.parametrize("cov", ["exp_quad", "matern32", "matern52", "exponential"])
def test_the_graph_forms_are_the_numpy_forms(cov):
    import pytensor
    import pytensor.tensor as pt

    s, l, z6, z4 = pt.dscalar("s"), pt.dscalar("l"), pt.dvector("z6"), pt.dvector("z4")
    X = np.array(PAIRS) / ROOT5
    f = pytensor.function([s, l, z6], gpm.exact_effect(X, s, l, z6, cov=cov))
    zv = np.linspace(-1, 1, 6)
    np.testing.assert_allclose(f(1.3, 0.4, zv), _f(X, 1.3, 0.4, zv, cov), rtol=1e-12, atol=1e-14)
    one = np.array([0.0, 0.3, 0.55, 1.0])
    f1 = pytensor.function([s, l, z4], gpm.exact_effect(one, s, l, z4, cov=cov))
    np.testing.assert_allclose(f1(0.7, 0.35, zv[:4]), _f(one, 0.7, 0.35, zv[:4], cov), rtol=1e-12, atol=1e-14)
    phi, w = gpm.basis(one - 0.5, 1.25, 4), gpm.sqrt_eigenvalues(1.25, 4)
    fa = pytensor.function([s, l, z4], gpm.approximate_effect(phi, w, s, l, z4, cov=cov))
    want = phi @ (np.array([math.sqrt(_spd(v, 0.35, cov)) if cov != "exp_quad" else
                            math.sqrt(math.sqrt(2 * math.pi) * 0.35 * math.exp(-0.5 * 0.35 ** 2 * v ** 2)) for v in w]) * 0.7 * zv[:4])
    np.testing.assert_allclose(fa(0.7, 0.35, zv[:4]), want, rtol=1e-7)
    np.testing.assert_allclose(gpm.approximate_values(phi, w, 0.7, 0.35, zv[:4], cov), fa(0.7, 0.35, zv[:4]), rtol=1e-13)
    if cov == "exp_quad":
        l2 = pt.dscalar("l2")
        fan = pytensor.function([s, l, l2, z6], gpm.exact_effect(X, s, [l, l2], z6))
        np.testing.assert_allclose(fan(1.3, 0.4, 0.9, zv), _f(X, 1.3, [0.4, 0.9], zv), rtol=1e-12, atol=1e-14)


def test_the_distances():
    X = np.array(PAIRS, dtype=float)
    assert gpm.smallest_distance(X) == 1.0 and gpm.largest_distance(X) == ROOT5
    assert gpm.smallest_distance([3.0, 1.0, 1.5]) == 0.5 and gpm.largest_distance([3.0, 1.0, 1.5]) == 2.0


# -- the log density, written out ----------------------------------------------------------------------------------------
Z6 = np.array([0.3, -1.1, 0.8, 0.05, -0.6, 1.4])


def test_several_inputs_share_one_kernel_of_their_distance():
    m = gaussian("y ~ gp(o2, b2)")
    point = m.value_point(intercept=0.2, b=[], sigma=0.9, sdgp=1.2, lscale=0.45, zgp=Z6)
    assert set(point) == {"Intercept", "sigma_log__", "sdgp_o2b2_log__", "lscale_o2b2_log__", "zgp_o2b2"}
    mu = 0.2 + _f(np.array(PAIRS) / ROOT5, 1.2, 0.45, Z6)[PAIR_INDEX]
    want = stats.norm.logpdf(FRAME.y, mu, 0.9).sum() + _priors_logp([1.2], [0.45], [Z6], 0.2, 0.9)
    assert _logp(m, point) == pytest.approx(want, rel=1e-12)


def test_iso_false_gives_each_input_its_length_scale():
    m = gaussian("y ~ gp(o2, b2, iso = FALSE)", lscale={"o2b2o2": ab.InverseGamma(3, 1), "o2b2b2": ab.Gamma(2, 3)})
    assert m.gp_names == {"o2b2": {"sdgp": "sdgp_o2b2", "lscales": {"o2b2o2": "lscale_o2b2o2", "o2b2b2": "lscale_o2b2b2"},
                                   "zgp": "zgp_o2b2", "coordinate": "zgp_o2b2_dim"}}
    assert m.gp_parameter_names == ["sdgp_o2b2", "lscale_o2b2o2", "lscale_o2b2b2"]
    assert {m.names[k] for k in ("sdgp:o2b2", "lscale:o2b2o2", "lscale:o2b2b2", "zgp:o2b2")} == {
        "sdgp_o2b2", "lscale_o2b2o2", "lscale_o2b2b2", "zgp_o2b2"}
    point = m.value_point(intercept=0.2, b=[], sigma=0.9, sdgp=1.2, lscale={"o2b2o2": 0.45, "o2b2b2": 1.7}, zgp=Z6)
    mu = 0.2 + _f(np.array(PAIRS) / ROOT5, 1.2, [0.45, 1.7], Z6)[PAIR_INDEX]
    want = (stats.norm.logpdf(FRAME.y, mu, 0.9).sum() + stats.norm.logpdf(0.2, 0, 5) + stats.halfnorm.logpdf(0.9)
            + stats.halfnorm.logpdf(1.2) + stats.invgamma.logpdf(0.45, 3, scale=1) + stats.gamma.logpdf(1.7, 2, scale=1 / 3)
            + stats.norm.logpdf(Z6).sum())
    assert _logp(m, point) == pytest.approx(want, rel=1e-12)
    with pytest.raises(ValueError, match=r"lscale requires an exact-key mapping over the Gaussian processes' columns "
                                         r"\['o2b2o2', 'o2b2b2'\]"):
        m.value_point(intercept=0.2, b=[], sigma=0.9, sdgp=1.2, lscale=0.45, zgp=Z6)
    with pytest.raises(ValueError, match=r"lscale for 'gp\(o2, b2, iso = FALSE\)' \(o2b2b2\) must be a finite positive scalar"):
        m.value_point(intercept=0.2, b=[], sigma=0.9, sdgp=1.2, lscale={"o2b2o2": 0.45, "o2b2b2": -1.0}, zgp=Z6)


def _by_factor_model(**extra):
    return gaussian("y ~ z + gp(ord, by = z)", categorical=CAT, b=ab.Normal(0, 2), **extra)


ZP, ZQ, ZR = np.array([0.4, -0.7, 1.1]), np.array([-0.2, 0.9, 0.3]), np.array([0.6, -1.3])
BY_VALUES = dict(sdgp={"ordzp": 1.2, "ordzq": 0.6, "ordzr": 0.9}, lscale={"ordzp": 0.4, "ordzq": 0.8, "ordzr": 0.5},
                 zgp={"ordzp": ZP, "ordzq": ZQ, "ordzr": ZR})


def _by_factor_gp(values=BY_VALUES):
    out = np.zeros(len(FRAME))
    for level, z in zip("pqr", (ZP, ZQ, ZR)):
        rows, inputs, index, dmax = LEVELS[level]
        key = f"ordz{level}"
        out[rows] = _f(np.array(inputs) / dmax, values["sdgp"][key], values["lscale"][key], values["zgp"][key])[index]
    return out


def test_a_factor_by_gives_each_level_its_process():
    m = _by_factor_model()
    assert list(m.gp_names) == ["ordzp", "ordzq", "ordzr"] and m.design.gp["ordzq"].rows.tolist() == LEVELS["q"][0]
    point = m.value_point(intercept=0.1, b=[0.5, -0.3], sigma=1.1, **BY_VALUES)
    X = np.column_stack([FRAME.z == "q", FRAME.z == "r"]).astype(float)
    mu = 0.1 + (X - X.mean(0)) @ np.array([0.5, -0.3]) + _by_factor_gp()
    want = (stats.norm.logpdf(FRAME.y, mu, 1.1).sum() + stats.norm.logpdf([0.5, -0.3], 0, 2).sum()
            + _priors_logp(BY_VALUES["sdgp"].values(), BY_VALUES["lscale"].values(), (ZP, ZQ, ZR), 0.1, 1.1))
    assert _logp(m, point) == pytest.approx(want, rel=1e-12)
    with pytest.raises(ValueError, match=r"sdgp must cover exactly the columns of the Gaussian processes \['ordzp', 'ordzq', 'ordzr'\]"):
        m.value_point(intercept=0.1, b=[0.5, -0.3], sigma=1.1, **{**BY_VALUES, "sdgp": {"ordz": 1.0}})


def test_a_numeric_by_multiplies_the_process():
    m = gaussian("y ~ gp(ord, by = w)")
    z4 = np.array([0.3, -0.8, 1.2, 0.1])
    point = m.value_point(intercept=-0.1, b=[], sigma=0.8, sdgp=1.4, lscale=0.6, zgp=z4)
    inputs = [1.0, 2.0, 3.0, 4.0]
    f = _f(np.array(inputs) / 3, 1.4, 0.6, z4)[FRAME.ord.to_numpy() - 1]
    want = stats.norm.logpdf(FRAME.y, -0.1 + FRAME.w * f, 0.8).sum() + _priors_logp([1.4], [0.6], [z4], -0.1, 0.8)
    assert _logp(m, point) == pytest.approx(want, rel=1e-12)
    assert m.design.gp["ordw"].multiplier.tolist() == FRAME.w.tolist()


@pytest.mark.parametrize("cov", ["matern32", "matern52", "exponential"])
def test_the_other_kernels_exact_and_approximate(cov):
    exact = gaussian(f'y ~ gp(x, cov = "{cov}")')
    z12 = np.linspace(-1.2, 1.1, 12)
    point = exact.value_point(intercept=0.3, b=[], sigma=1.0, sdgp=0.9, lscale=0.3, zgp=z12)
    x = FRAME.x.to_numpy()
    mu = 0.3 + _f(x / 4.0, 0.9, 0.3, z12, cov)                              # twelve distinct inputs in data order, dmax 4
    want = stats.norm.logpdf(FRAME.y, mu, 1.0).sum() + _priors_logp([0.9], [0.3], [z12], 0.3, 1.0)
    assert _logp(exact, point) == pytest.approx(want, rel=1e-11)
    approx = gaussian(f'y ~ gp(x, k = 5, c = 1.5, cov = "{cov}")')
    z5 = np.array([0.5, -0.4, 1.2, -0.9, 0.25])
    point = approx.value_point(intercept=0.3, b=[], sigma=1.0, sdgp=0.9, lscale=0.3, zgp=z5)
    u = x / 4.0 - np.mean(x / 4.0)
    j = np.arange(1, 6)
    phi = np.sin(np.pi * j * (u[:, None] + 1.5) / 3.0) / math.sqrt(1.5)
    mu = 0.3 + phi @ (0.9 * np.sqrt([_spd(v, 0.3, cov) for v in np.pi * j / 3.0]) * z5)
    want = stats.norm.logpdf(FRAME.y, mu, 1.0).sum() + _priors_logp([0.9], [0.3], [z5], 0.3, 1.0)
    assert _logp(approx, point) == pytest.approx(want, rel=1e-7)


def test_scale_false_leaves_the_inputs_as_they_are():
    m = gaussian("y ~ gp(ord, scale = FALSE)")
    z4 = np.array([0.3, -0.8, 1.2, 0.1])
    point = m.value_point(intercept=0.0, b=[], sigma=1.0, sdgp=1.0, lscale=2.0, zgp=z4)
    f = _f(np.array([1.0, 2.0, 3.0, 4.0]), 1.0, 2.0, z4)[FRAME.ord.to_numpy() - 1]
    want = stats.norm.logpdf(FRAME.y, f, 1.0).sum() + _priors_logp([1.0], [2.0], [z4])
    assert _logp(m, point) == pytest.approx(want, rel=1e-12)


def test_the_build_checks_each_levels_contribution(monkeypatch):
    honest = gpm.exact_effect
    monkeypatch.setattr(gpm, "exact_effect", lambda x, s, l, z, **kw: honest(x[::-1], s, l, z, **kw))
    with pytest.raises(RuntimeError, match=r"the graph's 'mu' does not move by the values of 'gp\(ord, by = z\)' at its rows' inputs"):
        _by_factor_model()


def test_the_design_checks_the_graph(monkeypatch):
    """Design.gp is read off the graph: a length-scale that is not a free positive scalar on its log, or a zgp without its
    coordinate, is refused (here simulated by pointing the names elsewhere)."""
    m = gaussian("y ~ gp(o2, b2, iso = FALSE)")
    roles = m.gp_names["o2b2"]
    monkeypatch.setattr(type(m), "gp_names", property(lambda self: {"o2b2": {**roles, "lscales": {"o2b2o2": "Intercept",
                                                                                                 "o2b2b2": "lscale_o2b2b2"}}}))
    with pytest.raises(RuntimeError, match=r"'Intercept' is not a free positive scalar of the graph sampled on its log"):
        m._gp_design()
    monkeypatch.setattr(type(m), "gp_names", property(lambda self: {"o2b2": {**roles, "coordinate": "elsewhere"}}))
    with pytest.raises(RuntimeError, match=r"'zgp_o2b2' is not a free vector of 6 entries with the coordinate 'elsewhere'"):
        m._gp_design()


# -- priors and brms's defaults ------------------------------------------------------------------------------------------
def test_priors_cover_each_process_and_length_scale():
    m = _by_factor_model(sdgp={"ordzp": ab.HalfNormal(1), "ordzq": ab.HalfNormal(2), "ordzr": ab.Exponential(1)})
    pr = m.priors.sdgp_priors(m.spec)
    assert list(pr) == ["ordzp", "ordzq", "ordzr"] and pr["ordzr"] == ab.Exponential(1)
    assert set(m.priors.lscale_priors(m.spec)) == {"ordzp", "ordzq", "ordzr"}                # one bare prior, each level's
    with pytest.raises(ab.PriorError, match=r"sdgp must cover exactly the columns of the Gaussian processes \['ordzp', 'ordzq', 'ordzr'\]; "
                                            r"got \['ordzp'\]"):
        _by_factor_model(sdgp={"ordzp": ab.HalfNormal(1)})
    with pytest.raises(ab.PriorError, match=r"lscale must cover exactly the columns of the Gaussian processes \['o2b2o2', 'o2b2b2'\]"):
        gaussian("y ~ gp(o2, b2, iso = FALSE)", lscale={"o2b2": ab.InverseGamma(3, 1)})
    table = _by_factor_model(sdgp={"ordzp": ab.HalfNormal(1), "ordzq": ab.HalfNormal(2), "ordzr": ab.HalfNormal(1)}).prior_table()
    assert [(r["class"], r["coef"], r["prior"]) for r in table if r["class"] == "sdgp"] == [
        ("sdgp", "", ""), ("sdgp", "gpordzp", "normal(0, 1)"), ("sdgp", "gpordzq", "normal(0, 2)"), ("sdgp", "gpordzr", "normal(0, 1)")]
    aniso = gaussian("y ~ gp(o2, b2, iso = FALSE)").prior_table()
    assert [(r["class"], r["coef"], r["prior"]) for r in aniso if r["class"] == "lscale"] == [
        ("lscale", "", ""), ("lscale", "gpo2b2b2", "inv_gamma(3, 1)"), ("lscale", "gpo2b2o2", "inv_gamma(3, 1)")]


def _lscale_rows(formula, frame, **kw):
    pri = ab.default_priors(formula, frame, **kw)
    model = ab.gaussian(formula, frame, pri, categorical=kw.get("categorical"))
    return {r["coef"]: r["prior"] for r in model.prior_table() if r["class"] == "lscale" and r["coef"]}


def test_the_default_lscale_prior_is_brms():
    """brms 2.23.0's default_prior() of each probe frame (printed in the comments): the inverse gamma of D67 per level, per input
    with iso = FALSE, on the Euclidean distances of several inputs, and with scale = FALSE on the inputs scaled by dmax with the
    scale times dmax (the floor 0.01 relative to dmax); normal(0, 0.5) with two distinct values, which scale = FALSE leaves as is."""
    y = [0.3, -1.2, 0.5, 1.1, -0.4, 0.9, 0.2, -0.7]
    # default_prior(y ~ gp(x, scale = FALSE), x = (0, 0.001, 1, 2)): inv_gamma(1.494197, 0.113215); (0, 10, 30): (18.506437, 299.544337)
    assert _lscale_rows("y ~ gp(x, scale = FALSE)", pd.DataFrame({"y": y[:4], "x": [0, 0.001, 1, 2]})) == {"gpx": "inv_gamma(1.494197, 0.113215)"}
    assert _lscale_rows("y ~ gp(x, scale = FALSE)", pd.DataFrame({"y": y[:3], "x": [0, 10, 30]})) == {"gpx": "inv_gamma(18.506437, 299.544337)"}
    # (0, 0.005, 0.5): inv_gamma(1.494197, 0.028304); two distinct inputs (0, 10): normal(0, 0.5)
    assert _lscale_rows("y ~ gp(x, scale = FALSE)", pd.DataFrame({"y": y[:3], "x": [0, 0.005, 0.5]})) == {"gpx": "inv_gamma(1.494197, 0.028304)"}
    assert _lscale_rows("y ~ gp(x, scale = FALSE)", pd.DataFrame({"y": y[:6], "x": [0, 10] * 3})) == {"gpx": "normal(0, 0.5)"}
    # gp(x, by = f), f = a on x = (0, 1, 3, 4), b on (0, 2, 0, 2): inv_gamma(11.830309, 5.315674) and normal(0, 0.5)
    by = pd.DataFrame({"y": y, "x": [0, 1, 3, 4, 0, 2, 0, 2], "f": ["a"] * 4 + ["b"] * 4})
    assert _lscale_rows("y ~ f + gp(x, by = f)", by, categorical={"f": "a"}) == {"gpxfa": "inv_gamma(11.830309, 5.315674)",
                                                                                "gpxfb": "normal(0, 0.5)"}
    # gp(x, z, w): inv_gamma(10.212745, 4.265136); gp(x, z, by = w) is refused here (D196)
    three = pd.DataFrame({"y": y, "x": [0, 1, 3, 4, 0, 2, 0, 2], "z": [1, 0, 1, 0, 2, 2, 1, 1], "w": [0.5, 1, 1, 2, 2, 3, 3, 1]})
    assert _lscale_rows("y ~ gp(x, z, w)", three) == {"gpxzw": "inv_gamma(10.212745, 4.265136)"}
    # the frozen fixtures: gauss_gp_two_inputs_aniso's per input, and its binary input's normal(0, 0.5)
    assert _lscale_rows("y ~ gp(o2, b2, iso = FALSE)", FRAME) == {"gpo2b2o2": "inv_gamma(45.631905, 28.084743)", "gpo2b2b2": "normal(0, 0.5)"}
    pri = ab.default_priors("y ~ z + gp(ord, by = z)", FRAME, categorical=CAT)
    assert set(pri.sdgp) == set(pri.lscale) == {"ordzp", "ordzq", "ordzr"} and pri.sdgp["ordzp"] == ab.HalfStudentT(3, 2.5)
    assert pri.lscale["ordzr"] == ab.HalfNormal(0.5)                                            # level r: two distinct inputs
    for cov in ("matern32", "matern52", "exponential"):                                       # the rule does not see the kernel
        assert _lscale_rows(f'y ~ gp(x, cov = "{cov}")', FRAME) == _lscale_rows("y ~ gp(x)", FRAME)


# -- predictions ------------------------------------------------------------------------------------------------------
def _posterior(model, values):
    """One chain of two draws of every variable in ``values`` (scalars broadcast to both draws, vectors shared), the raw
    intercept 0.3 and sigma 1, with each zgp's coordinate."""
    post, coords, dims = {"Intercept": np.full((1, 2), 0.3), "sigma": np.ones((1, 2))}, {}, {}
    for name, v in values.items():
        v = np.asarray(v, dtype=float)
        if v.ndim == 1 and name.startswith("zgp_"):
            post[name] = np.broadcast_to(v, (1, 2, len(v)))
            coords[f"{name}_dim"], dims[name] = [str(j) for j in range(1, len(v) + 1)], [f"{name}_dim"]
        else:
            post[name] = np.broadcast_to(v, (1, 2)) if v.ndim == 0 else v.reshape(1, 2)
    start = model.pymc_model.initial_point()
    for t in model.spec.pop_terms:                                       # the coefficients at 0 (a factor's: a vector)
        shape = np.shape(start[t.name])
        post[t.name] = np.zeros((1, 2, *shape))
        if shape:
            coords[f"{t.name}_dim"], dims[t.name] = list(model.pymc_model.coords[f"{t.name}_dim"]), [f"{t.name}_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_of_a_factor_by():
    m = gaussian("y ~ z + gp(ord, by = z)", categorical=CAT, b=ab.Normal(0, 2))
    values = {f"{cls}_{k}": BY_VALUES[cls][k] for cls in ("sdgp", "lscale", "zgp") for k in BY_VALUES[cls]}
    post = _posterior(m, values)
    eta = m.predict(post, kind="linear")["eta"][0].to_numpy()
    np.testing.assert_allclose(eta, np.broadcast_to(0.3 + _by_factor_gp(), (2, 12)), rtol=1e-12)
    new = pd.DataFrame({"z": ["r", "p", "q"], "ord": [4, 2, 1]})          # each at a fitted input of its own level
    want = 0.3 + np.array([_by_factor_gp()[11], _by_factor_gp()[9], _by_factor_gp()[10]])
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="mean").mu[0, 0], want, rtol=1e-12)
    with pytest.raises(ab.FormulaError, match=r"has the values \[3\.0\], which are not inputs of the exact Gaussian process "
                                              r"'gp\(ord, by = z\)' at level 'p' of 'z'"):
        m.predict(post, newdata=new.assign(ord=[4, 3, 1]), kind="mean")


def test_predictions_of_a_numeric_by_and_of_several_inputs():
    m = gaussian("y ~ gp(ord, by = w)")
    z4 = np.array([0.3, -0.8, 1.2, 0.1])
    post = _posterior(m, {"sdgp_ordw": [[1.4, 0.7]], "lscale_ordw": [[0.6, 0.9]], "zgp_ordw": z4})
    new = pd.DataFrame({"ord": [2, 4], "w": [10.0, -0.5]})
    got = m.predict(post, newdata=new, kind="linear")["eta"][0].to_numpy()
    for d, (s, l) in enumerate(((1.4, 0.6), (0.7, 0.9))):
        f = _f(np.array([1.0, 2.0, 3.0, 4.0]) / 3, s, l, z4)
        np.testing.assert_allclose(got[d], 0.3 + new.w.to_numpy() * f[[1, 3]], rtol=1e-12)
    a = gaussian("y ~ gp(o2, b2, iso = FALSE)")
    post = _posterior(a, {"sdgp_o2b2": 1.2, "lscale_o2b2o2": 0.45, "lscale_o2b2b2": 1.7, "zgp_o2b2": Z6})
    got = a.predict(post, newdata=pd.DataFrame({"o2": [3, 1], "b2": [1, 0]}), kind="linear")["eta"][0, 0].to_numpy()
    np.testing.assert_allclose(got, 0.3 + _f(np.array(PAIRS) / ROOT5, 1.2, [0.45, 1.7], Z6)[[5, 0]], rtol=1e-12)


def test_predictions_of_an_approximate_matern_process_anywhere_inside_its_boundary():
    m = gaussian('y ~ gp(x, k = 5, c = 1.5, cov = "matern52")')
    z5 = np.array([0.5, -0.4, 1.2, -0.9, 0.25])
    post = _posterior(m, {"sdgp_x": 0.9, "lscale_x": 0.3, "zgp_x": z5})
    new = pd.DataFrame({"x": [0.0, 4.5]})
    x = FRAME.x.to_numpy()
    u = new.x.to_numpy() / 4.0 - np.mean(x / 4.0)
    j = np.arange(1, 6)
    phi = np.sin(np.pi * j * (u[:, None] + 1.5) / 3.0) / math.sqrt(1.5)
    want = 0.3 + phi @ (0.9 * np.sqrt([_spd(v, 0.3, "matern52") for v in np.pi * j / 3.0]) * z5)
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="mean").mu[0, 0], want, rtol=1e-7)
    with pytest.raises(ab.FormulaError, match=r"outside the boundary of the approximate Gaussian process"):
        m.predict(post, newdata=pd.DataFrame({"x": [40.0]}), kind="mean")


# -- describe(), the target identity and fit records ------------------------------------------------------------------
def test_describe_writes_each_form():
    by = _by_factor_model().describe()
    for line in ("  gp(ord, by = z): exact Gaussian process of 'ord', squared-exponential kernel; one process per level of the factor 'z', "
                 "on that level's rows (brms's by = z, cmc = TRUE)",
                 "    level 'q': 3 distinct inputs from 1 to 4, scaled by dmax = 3, their largest distance, so that it is 1 (brms's scale = TRUE)",
                 " + gp(ord, by = z)_i",
                 "  sdgp_ordzq ~ HalfNormal(sigma=1)   (brms class sdgp, coefficient gpordzq)",
                 "  lscale_ordzr ~ InverseGamma(alpha=3, beta=1)   (brms class lscale, coefficient gpordzr; on the scaled input)",
                 "  zgp_ordzp[j] ~ Normal(0, 1)   (brms's zgp)",
                 "log(sdgp_ordzq), the marginal SD of gp(ord, by = z) at level 'q' of z (brms's sdgp)",
                 "  gp(ord, by = z)_i on the rows where z = 'r' = f(u_i) at the 2 distinct inputs",
                 "  New rows take the process's value at a fitted input of their level; a row of a level the fit did not see is refused"):
        assert line in by, line
    aniso = gaussian("y ~ gp(o2, b2, iso = FALSE)").describe()
    for line in ("  gp(o2, b2, iso = FALSE): exact Gaussian process of 'o2', 'b2', squared-exponential kernel, one length-scale per input "
                 "(iso = FALSE: the product of one-dimensional kernels); 6 distinct input rows, scaled by dmax = 2.23607, their largest "
                 "Euclidean distance",
                 "K[j, m] = sdgp_o2b2^2 exp(-(u_j[o2] - u_m[o2])^2 / (2 lscale_o2b2o2^2)) * exp(-(u_j[b2] - u_m[b2])^2 / (2 lscale_o2b2b2^2))",
                 "lscale_o2b2o2, lscale_o2b2b2 its length-scales, one per input.",
                 "log(lscale_o2b2b2), the length-scale in 'b2' of gp(o2, b2, iso = FALSE) (brms's lscale)"):
        assert line in aniso, line
    iso = gaussian("y ~ gp(o2, b2)").describe()
    assert "K[j, m] = sdgp_o2b2^2 k(r), r = |u_j - u_m| / lscale_o2b2 (Euclidean), k(r) = exp(-r^2 / 2)" in iso
    numeric = gaussian("y ~ gp(ord, by = w)").describe()
    assert "multiplied row by row by 'w' (brms's by = w, Cgp)" in numeric and "(brms's zgp) times w_i" not in numeric
    assert "zgp_ordw[j] ~ Normal(0, 1): brms's exact Gaussian process, non-centred times w_i; u_i" in numeric
    kern = gaussian('y ~ gp(x, k = 5, cov = "matern32")').describe()
    for line in ("approximate (Hilbert-space) Gaussian process of 'x', Matern 3/2 kernel",
                 "S(w) = sdgp_x^2 12 sqrt(3) lscale_x (3 + lscale_x^2 w^2)^-2 the Matern 3/2 kernel's spectral density",
                 "  New rows: the basis at their inputs, which must lie inside the boundary",
                 "inputs lscale_x * 4 apart are correlated as the kernel at r = 1"):
        assert line in kern, line
    unscaled = gaussian("y ~ gp(ord, scale = FALSE)").describe()
    for line in ("unscaled (brms's scale = FALSE: lscale is on the inputs' own scale)", "(brms class lscale, coefficient gpord; on the "
                 "unscaled input)", "u_i, the input as it is (scale = FALSE).", "inputs lscale_ord apart are correlated exp(-1/2) = 0.61"):
        assert line in unscaled, line


def test_the_target_identity_and_fit_records():
    base = _by_factor_model().target_identity
    assert _by_factor_model().target_identity == base
    assert _by_factor_model(lscale={"ordzp": ab.InverseGamma(3, 1), "ordzq": ab.InverseGamma(3, 1),
                                    "ordzr": ab.InverseGamma(3, 2)}).target_identity != base
    assert gaussian("y ~ z + gp(ord, by = z)", frame=FRAME.assign(ord=FRAME.ord * 2), categorical=CAT,
                    b=ab.Normal(0, 2)).target_identity != base
    assert gaussian("y ~ gp(o2, b2)").target_identity != gaussian("y ~ gp(o2, b2, iso = FALSE)").target_identity
    assert gaussian("y ~ gp(ord, by = w)").target_identity != gaussian("y ~ gp(ord, by = w)", frame=FRAME.assign(w=FRAME.w * 2)).target_identity
    assert gaussian('y ~ gp(x, cov = "matern32")').target_identity != gaussian('y ~ gp(x, cov = "matern52")').target_identity
    assert gaussian('y ~ gp(x, k = 5, cov = "matern32")').target_identity != gaussian('y ~ gp(x, k = 6, cov = "matern32")').target_identity
    import autobayes.session as session

    m = gaussian("y ~ gp(o2, b2, iso = FALSE)")
    post = _posterior(m, {"sdgp_o2b2": 1.2, "lscale_o2b2o2": 0.45, "lscale_o2b2b2": 1.7, "zgp_o2b2": Z6})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = session._summary_rows(m, post)
    assert {"sdgp_o2b2", "lscale_o2b2o2", "lscale_o2b2b2"} <= set(rows)
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=2,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    session_summary, session_env = session._summary_rows, session.environment
    try:
        session._summary_rows, session.environment = (lambda *args: {}), (lambda: {})
        rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    finally:
        session._summary_rows, session.environment = session_summary, session_env
    assert rec["model"]["priors"]["lscale:o2b2b2"] == "InverseGamma(alpha=3, beta=1)"
    entry = rec["model"]["gaussian_processes"]["gp(o2, b2, iso = FALSE)"]
    assert entry["columns"] == ["o2", "b2"] and not entry["iso"] and entry["processes"]["o2b2"]["names"] == m.gp_names["o2b2"]
    assert entry["processes"]["o2b2"]["dmax"] == ROOT5
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_prior_predictive_draws_each_level():
    pp = _by_factor_model().prior_predictive(draws=30, seed=4)
    assert pp.prior["zgp_ordzr"].shape == (1, 30, 2) and (pp.prior["sdgp_ordzq"] > 0).all()
    assert pp.prior_predictive["y"].shape == (1, 30, 12)


# -- the oracle's extension ------------------------------------------------------------------------------------------------
def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_a_corrupted_by_variable():
    """check_gp_by and match_gp_levels fail on a changed Cgp_k, Cgp_k_l, Igp_k_l or Ngp_k, and check_design on scale = FALSE's
    dmax and on several inputs' cmeans."""
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    numeric = gates.Bundle("gauss_gp_by_numeric")
    gates.check_design(numeric)

    def cgp(raw):
        raw["standata"]["Cgp_1"]["values"][4] += 1e-6
    with pytest.raises(OracleMismatch, match=r"brms Cgp_1 is"):
        gates.check_design(_corrupted(numeric, cgp))

    factor = gates.Bundle("gauss_gp_by_factor")
    gates.check_design(factor)

    def cgp_level(raw):
        raw["standata"]["Cgp_1_2"]["values"][0] = 2
    with pytest.raises(OracleMismatch, match=r"brms Cgp_1_2 is"):
        gates.check_design(_corrupted(factor, cgp_level))

    def rows(raw):
        raw["standata"]["Igp_1_3"]["values"][0] = 1
    with pytest.raises(OracleMismatch, match=r"0 brms Gaussian processes have its 3 levels' rows"):
        _corrupted(factor, rows)

    def counts(raw):
        raw["standata"]["Ngp_1"]["values"][0] = 21
    with pytest.raises(OracleMismatch, match=r"brms Kgp_1, Ngp_1 are"):
        gates.check_design(_corrupted(factor, counts))

    unscaled = gates.Bundle("gauss_gp_scale_false")
    gates.check_design(unscaled)

    def dmax(raw):
        raw["standata"]["dmax_1"] = {"dims": [], "values": [3]}
    with pytest.raises(OracleMismatch, match=r"brms's Gaussian process 1 has dmax_1; 'gp\(ord, scale = FALSE\)' leaves its inputs unscaled"):
        gates.check_design(_corrupted(unscaled, dmax))

    two = gates.Bundle("gauss_gp_two_inputs")

    def cmeans(raw):
        raw["standata"]["cmeans_1"]["values"][1] += 1e-6
    with pytest.raises(OracleMismatch, match=r"brms cmeans_1 is"):
        gates.check_design(_corrupted(two, cmeans))

    def extra(raw):
        raw["standata"]["Cgp_1"] = {"dims": [36], "values": [1.0] * 36}
    with pytest.raises(OracleMismatch, match=r"brms's Gaussian process 1 has Cgp_1; 'gp\(o2, b2\)' has no by variable"):
        gates.check_design(_corrupted(two, extra))
