"""Gaussian processes (brms-parity C03): brms's exact ``gp(x)`` and its Hilbert-space approximation ``gp(x, k, c)``. The term
model and the scaling of the input, the refusals, the graph against a density written by hand (both forms), the build-time
check of the term's contribution, priors, ``value_point``, by-hand predictions, ``describe()``, names, the target identity,
fit records, prior-predictive draws, brms's default priors (the tuned inverse gamma of ``lscale``), and the oracle's extension
to Gaussian processes. The Stan data and default-prior tables below were printed by brms 2.23.0 (R 4.5.2, C collation) for
``FRAME``. Nothing here samples; the brms comparison is the oracle cases ``gauss_gp_exact`` and ``gauss_gp_approx``."""
import copy
import json
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import autobayes as ab
from autobayes.formula import GaussianProcess, parse_formula
from autobayes.gaussian_process import inverse_gamma_tails

FRAME = pd.DataFrame({
    "y": [0.31, -1.2, 2.05, 0.77, -0.4, 1.66, 0.12, -0.95, 2.41, 1.08, 0.5, -0.33],
    "x": [0.7, -0.2, 0.7, 1.9, -0.2, 3.1, 0.7, 2.4, -0.9, 1.1, 2.4, 0.3],       # 8 distinct inputs, three repeated
    "t": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    "two": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    "w": [0.2, -1.1, 0.7, 1.5, -0.3, 0.05, -2.0, 0.8, 1.2, -0.6, 0.35, -0.1],
    "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
    "cnt": [0, 3, 1, 1, 7, 2, 0, 0, 12, 4, 1, 0],
    "yb": [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
})
FRAME["big"] = FRAME.y * 20
FRAME["pos"] = np.exp(FRAME.y)                                       # positive and proportion responses for the family checks
FRAME["prop"] = 1 / (1 + np.exp(-FRAME.y))
# brms 2.23.0's standata(y ~ gp(x), FRAME): Nsubgp_1 = 8, Jgp_1, and Xgp_prior_1 = Xgp_1 (the exact GP), the distinct inputs in
# the order they first appear divided by dmax_1 = 4; and standata(y ~ gp(x, k = 5, c = 1.5)): Lgp_1 = 1.5, NBgp_1 = 5, Xgp_1
BRMS_JGP_X = [1, 2, 1, 3, 2, 4, 1, 5, 6, 7, 5, 8]
BRMS_XGP_PRIOR_X = [0.175, -0.05, 0.475, 0.775, 0.6, -0.225, 0.275, 0.075]
BRMS_XGP_K5 = np.reshape([   # column-major, as brms prints it
    0.81307131863569326, 0.7731652131102027, 0.79636366992931507, 0.70170239661830269, 0.76603001803672421, 0.71239000664803187,
    0.81642662968482305, 0.80080782807347883, 0.14879468348790609, 0.49705162538629727, -0.35151083859541776, -0.71755116774927097,
    -0.53027211095177929, 0.69617777895677102, -0.021373388792499913, 0.31245971410378265, -0.7858414098603892, -0.45362119572572512,
    -0.64120858900878419, 0.032055505335131962, -0.39895757305092483, -0.032055505335131698, -0.81586709165425719,
    -0.67889209559033026, -0.29260620551781796, -0.78867513459481309, 0.63453702041140647, 0.68477165125215322, 0.80644415311810924,
    -0.72750378058409926, 0.042732129331946041, -0.57735026918962595, 0.73229352698871653, -0.053401431398385824,
    0.36112720334681764, -0.73229352698871653, -0.15929058089825698, -0.67889209559033037, 0.81474839907254204, 0.45362119572572557],
    (8, 5), order="F")
# standata(y ~ gp(t) + gp(x, k = 3)): GP 1 is gp(t) (Xgp_1 = 1/3, 2/3, 1, 4/3), GP 2 the approximate gp(x) with brms's default
# c = 5/4 (Lgp_2 = 1.25)
BRMS_XGP_K3 = np.reshape([
    0.88902570535136882, 0.82634297508638566, 0.86272638234933885, 0.71525970277633544, 0.8151838725891607, 0.73177535357383861,
    0.89431684773702691, 0.86971409651410558, 0.1951132466379259, 0.63245553203367588, -0.45530048359222453, -0.85891278381498426,
    -0.67091973694751394, 0.84154934324156883, -0.028094637009425126, 0.4060614474226909, -0.84620446994226883, -0.34228246745251345,
    -0.62244332783695966, 0.31615751161061462, -0.26299766213823161, 0.23601523086110923, -0.89343426498049205, -0.68012777183121398],
    (8, 3), order="F")
Z8 = np.array([0.3, -1.1, 0.8, 0.05, -0.6, 1.4, -0.2, 0.9])
Z5 = np.array([0.5, -0.4, 1.2, -0.9, 0.25])


def priors(**extra):
    base = dict(intercept=ab.Normal(0, 5), sigma=ab.HalfNormal(1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    return ab.Priors(**{**base, **extra})


def gaussian(formula="y ~ gp(x)", frame=None, **extra):
    return ab.gaussian(formula, FRAME if frame is None else frame, priors(**extra))


def _exact(u, sdgp, lscale, z):
    """The exact GP at the distinct scaled inputs ``u``, written out: chol(sdgp^2 exp(-d^2 / (2 lscale^2)) + 1e-12 I) z."""
    d2 = (np.asarray(u)[:, None] - np.asarray(u)[None, :]) ** 2
    return np.linalg.cholesky(sdgp ** 2 * np.exp(-d2 / (2 * lscale ** 2)) + 1e-12 * np.eye(len(u))) @ z


def _approximate(u, L, sdgp, lscale, z):
    """The Hilbert-space approximation at the scaled, centred inputs ``u``, written out from Solin and Sarkka (2020)."""
    j = np.arange(1, len(z) + 1)
    phi = np.sin(np.pi * j * (np.asarray(u)[:, None] + L) / (2 * L)) / np.sqrt(L)
    w = np.pi * j / (2 * L)
    return phi @ (np.sqrt(sdgp ** 2 * np.sqrt(2 * np.pi) * lscale * np.exp(-0.5 * lscale ** 2 * w ** 2)) * z)


SCALED_X = FRAME.x.to_numpy() / 4.0
CMEAN_X = np.mean(BRMS_XGP_PRIOR_X)


# -- the term model and the scaling -----------------------------------------------------------------------------------
def test_the_term_model():
    s = parse_formula("y ~ w + gp(x) + (1 | g)", FRAME)
    assert s.terms == ("w",) and s.columns == ("w",) and s.coefficient_labels == ("w",) and s.pop_columns == ("w", "x")
    assert s.canonical_formula == "y ~ w + gp(x) + (1 | g)" and s.bambi_formula == "y ~ w + (1 | g)"
    (g,) = s.gp_terms
    assert g == s.gp("x") and g.term == "gp(x)" and not g.approximate and g.known and g.size == 8
    assert g.inputs == (0.7, -0.2, 1.9, 3.1, 2.4, -0.9, 1.1, 0.3) and g.dmax == 4.0
    np.testing.assert_array_equal(g.scaled_inputs, BRMS_XGP_PRIOR_X)
    approx = parse_formula("y ~ gp(x, k = 10, c = 5/4) + gp(t, k = 3)", FRAME)
    assert [h.term for h in approx.gp_terms] == ["gp(x, k = 10, c = 1.25)", "gp(t, k = 3, c = 1.25)"]   # brms's default c, written out
    assert approx.rhs == "gp(x, k = 10, c = 1.25) + gp(t, k = 3, c = 1.25)" and approx.bambi_formula == "y ~ 1"
    assert approx.gp("x").L == 1.25 and approx.gp("t").size == 3 and approx.gp("x").cmean == pytest.approx(CMEAN_X, abs=1e-16)
    unseen = parse_formula("y ~ gp(x, c = 1.5, k = 4)")
    assert unseen.gp_terms == (GaussianProcess("x", k=4, c=1.5),) and not unseen.gp("x").known and unseen.gp("x").size == 4
    with pytest.raises(ValueError, match="known once the data are seen"):
        parse_formula("y ~ gp(x)").gp("x").size
    with pytest.raises(ValueError, match="is an exact Gaussian process; L belongs to the approximate one"):
        g.L
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):                   # asdict hands the processes back as mappings
        assert replica == s and hash(replica) == hash(s)
    spelled = parse_formula("y ~ gp( x , cov = 'exp_quad', gr = TRUE, scale = T)", FRAME)   # brms's defaults written out
    assert spelled.gp_terms == s.gp_terms and spelled.rhs == "gp(x)"
    with pytest.raises(ValueError, match="is exact \\(no k, no c\\) or approximate \\(k and c\\)"):
        GaussianProcess("x", k=4)
    with pytest.raises(ValueError, match="gp_terms must name each column once"):
        ab.ModelSpec(formula="", response="y", gp_terms=(GaussianProcess("x"), GaussianProcess("x", k=3, c=1.25)))


def test_brms_scales_and_orders_the_inputs():
    """The Stan data brms printed for FRAME: equal inputs share one latent value, the distinct inputs in the order they first
    appear (Jgp), divided by their largest distance (Xgp_prior; the exact GP's Xgp), and the approximate GP's basis at them less
    the mean of the distinct scaled inputs (Xgp, Lgp = c)."""
    exact = gaussian()
    block = exact.design.gp["x"]
    assert (block.index + 1).tolist() == BRMS_JGP_X and block.basis is None and block.slambda is None and block.term == "gp(x)"
    np.testing.assert_array_equal(block.inputs, BRMS_XGP_PRIOR_X)
    approx = gaussian("y ~ gp(x, k = 5, c = 1.5)")
    block = approx.design.gp["x"]
    assert (block.index + 1).tolist() == BRMS_JGP_X
    np.testing.assert_allclose(block.basis, BRMS_XGP_K5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(block.slambda, np.pi * np.arange(1, 6) / 3.0, rtol=1e-15)
    two = gaussian("y ~ gp(t) + gp(x, k = 3)", lscale={"t": ab.InverseGamma(3, 1), "x": ab.HalfNormal(1)},
                   sdgp={"t": ab.HalfNormal(1), "x": ab.HalfNormal(2)})
    np.testing.assert_allclose(two.design.gp["t"].inputs, [1 / 3, 2 / 3, 1.0, 4 / 3], rtol=1e-16)
    np.testing.assert_allclose(two.design.gp["x"].basis, BRMS_XGP_K3, rtol=0, atol=1e-15)
    assert two.spec.gp("x").L == 1.25 and list(two.design.gp) == ["t", "x"]
    signed = GaussianProcess("v")                                       # R's unique(): 0 and -0 are one input
    from autobayes.formula import gaussian_process_coding

    assert gaussian_process_coding(signed, pd.Series([0.0, -0.0, 1.0, 2.0])).inputs == (0.0, 1.0, 2.0)


CODING_REFUSALS = [
    (FRAME.g, r"has dtype (object|str); the input of a Gaussian process is a numeric column"),
    (pd.Series([True, False] * 6), r"has dtype bool"),
    (pd.Series([1 + 1j] * 12), r"is complex-valued"),
    (pd.Series([2.0] * 12), r"takes the single value 2\.0 in the fitted rows; a Gaussian process needs at least two distinct inputs"),
    (pd.Series([1.0, np.inf] * 6), r"has missing or non-finite values"),
]


@pytest.mark.parametrize("column, message", CODING_REFUSALS)
def test_coding_refusals(column, message):
    from autobayes.formula import gaussian_process_coding

    with pytest.raises(ab.FormulaError, match=message):
        gaussian_process_coding(GaussianProcess("v"), column)


def test_a_boundary_inside_the_inputs_is_refused():
    """With c >= 1 every fitted input lies inside the approximate GP's boundary (the centred, scaled inputs are strictly inside
    (-1, 1)); a smaller c that leaves one outside is refused (brms fits it anyway, D66)."""
    assert gaussian("y ~ gp(x, k = 3, c = 1)").spec.gp("x").L == 1.0
    with pytest.raises(ab.FormulaError, match=r"c = 0\.5 puts fitted inputs outside the approximate Gaussian process's boundary.*D66"):
        parse_formula("y ~ gp(x, k = 3, c = 0.5)", FRAME)


FORMULA_REFUSALS = [
    ("y ~ gp(x, w)", r"one covariate per gp\(\) here; brms's gp\(x, w, \.\.\.\) is one Gaussian process of several covariates.*D64"),
    ("y ~ gp(x, by = g)", r"brms's by = g .* is not part of this slice.*D64"),
    ("y ~ gp(x, cov = \"matern32\")", r"cov = \"matern32\" is not offered here; only brms's default squared-exponential kernel.*D64"),
    ("y ~ gp(x, cov = exp_quad)", r"cov = exp_quad is not offered here"),
    ("y ~ gp(x, iso = FALSE)", r"gp\(\)'s iso decides whether several covariates share one length-scale.*drop it.*D64"),
    ("y ~ gp(x, cmc = TRUE)", r"gp\(\)'s cmc only matters with a factor by =.*D64"),
    ("y ~ gp(x, gr = FALSE)", r"gr = FALSE gives every row its own latent value.*D64"),
    ("y ~ gp(x, scale = FALSE)", r"scale = FALSE keeps the inputs unscaled.*D64"),
    ("y ~ gp(x, scale = maybe)", r"scale takes TRUE or FALSE, got 'maybe'"),
    ("y ~ gp(x, k = 0)", r"k is the number of basis functions of the approximate Gaussian process, a positive whole number, got '0'"),
    ("y ~ gp(x, k = 2.5)", r"a positive whole number, got '2\.5'"),
    ("y ~ gp(x, k = NA)", r"k = NA is brms's exact Gaussian process: drop k"),
    ("y ~ gp(x, c = 1.5)", r"c is the boundary factor of the approximate Gaussian process, which k = <number of basis functions> asks for"),
    ("y ~ gp(x, k = 4, c = 0)", r"c is the approximate Gaussian process's boundary factor, a positive number"),
    ("y ~ gp(x, k = 4, c = sqrt(2))", r"a positive number such as 1\.25 or 5/4, got 'sqrt\(2\)'"),
    ("y ~ gp(x, k = 4, k = 5)", r"gp\(\)'s argument 'k' is given twice"),
    ("y ~ gp(x, m = 3)", r"gp\(\) has no argument 'm'; brms's gp\(\) takes by, k, cov, iso, gr, cmc, scale and c"),
    ("y ~ gp(log(x))", r"gp\(\) takes a column name here, not an expression \('log\(x\)'.*D64"),
    ("y ~ gp()", r"gp\(\) takes the column of its input first"),
    ("y ~ gp(k = 4)", r"gp\(\) takes the column of its input first"),
    ("y ~ gp(x):w", r"an interaction with a Gaussian process \('gp\(x\)'\) is not supported.*D64"),
    ("y ~ w * gp(x)", r"an interaction with a Gaussian process"),
    ("y ~ gp(x) + gp(x, k = 4)", r"'gp\(x, k = 4\)': a second Gaussian process of 'x' \(after 'gp\(x\)'\); one gp\(\) per column"),
    ("y ~ gp(y)", r"the response 'y' cannot be the input of a Gaussian process"),
    ("y ~ gp(t) + (1 | t)", r"'t': the grouping factor cannot also be a predictor or the response"),
    ("y ~ (1 + gp(x) | g)", r"function calls and transformations inside a group term are not supported"),
]


@pytest.mark.parametrize("formula, message", FORMULA_REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula, FRAME)


def test_other_refusals_name_the_construct():
    with pytest.raises(ab.FormulaError, match=r"the trials column 'cnt' cannot be the input of a Gaussian process"):
        parse_formula("yb | trials(cnt) ~ gp(cnt)", None, "binomial")
    with pytest.raises(ab.FormulaError, match=r"'gp\(t\)': 't' is declared categorical; the input of a Gaussian process is a numeric"):
        parse_formula("y ~ t + gp(t)", FRAME, categorical={"t": 1})
    with pytest.raises(ab.FormulaError, match=r"a Gaussian process in a distributional formula \(gp\(x\)\) is not supported.*D64"):
        parse_formula(["y ~ w", "sigma ~ gp(x)"], FRAME)
    with pytest.raises(ab.FormulaError, match=r"a Gaussian process \(gp\(x\)\) is not available for the categorical family.*D64"):
        parse_formula("g ~ gp(x)", None, "categorical")


def test_without_an_intercept_or_another_term():
    """``y ~ gp(t)`` and ``y ~ 0 + gp(t)``: Bambi is handed ``y ~ 1`` or ``y ~ 0``, and the process is the rest of the model."""
    m = ab.gaussian("y ~ 0 + gp(t)", FRAME, ab.Priors(sigma=ab.HalfNormal(1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1)))
    assert m.spec.bambi_formula == "y ~ 0" and m.design.X_raw.shape == (12, 0) and not m.spec.intercept
    z = np.array([0.4, -0.3, 1.1, -0.8])
    point = m.value_point(b=[], sigma=1.1, sdgp=0.9, lscale=0.4, zgp=z)
    mu = _exact([1 / 3, 2 / 3, 1.0, 4 / 3], 0.9, 0.4, z)[FRAME.t.to_numpy() - 1]
    want = (stats.norm.logpdf(FRAME.y, mu, 1.1).sum() + stats.halfnorm.logpdf(1.1) + stats.halfnorm.logpdf(0.9)
            + stats.invgamma.logpdf(0.4, 3, scale=1) + stats.norm.logpdf(z).sum())
    assert float(m.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(want, rel=1e-12)


# -- the graph --------------------------------------------------------------------------------------------------------
def _check_against_hand(m, point, mu, nojac, jac):
    """The model's log density with and without the Jacobian, its linear predictor, every gradient coordinate (against
    central differences of the density) and the JAX density."""
    assert float(m.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(nojac, rel=1e-12)
    assert float(m.pymc_model.compile_logp()(point)) == pytest.approx(jac, rel=1e-12)
    fn = m.pymc_model.compile_fn(m.pymc_model.replace_rvs_by_values([m.pymc_model["mu"]]), inputs=m.pymc_model.value_vars,
                                 point_fn=True, on_unused_input="ignore")
    np.testing.assert_allclose(np.asarray(fn(point)[0]), mu, rtol=1e-12, atol=1e-13)
    logp, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_dlogp()
    names = [v.name for v in m.pymc_model.value_vars]
    flat = np.concatenate([np.atleast_1d(point[n]) for n in names])
    sizes = np.cumsum([0] + [np.atleast_1d(point[n]).size for n in names])

    def unflatten(v):
        return {n: v[sizes[i]:sizes[i + 1]].reshape(np.shape(point[n])) for i, n in enumerate(names)}
    grad = np.asarray(dlogp(point))
    for k in range(flat.size):
        e = np.zeros_like(flat)
        e[k] = 1e-6
        assert grad[k] == pytest.approx((float(logp(unflatten(flat + e))) - float(logp(unflatten(flat - e)))) / 2e-6, abs=1e-5)
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    assert float(get_jaxified_logp(m.pymc_model)([point[n] for n in names])) == pytest.approx(jac, rel=1e-12)


def test_the_exact_density_is_the_one_written_by_hand():
    m = gaussian("y ~ w + gp(x)", b=ab.Normal(0, 2))
    a, bw, sigma, sdgp, lscale = 0.3, 0.7, 1.3, 0.8, 0.35
    point = m.value_point(intercept=a, b=[bw], sigma=sigma, sdgp=sdgp, lscale=lscale, zgp=Z8)
    assert set(point) == {"Intercept", "w", "sigma_log__", "sdgp_x_log__", "lscale_x_log__", "zgp_x"}
    w = FRAME.w.to_numpy()
    mu = a + (w - w.mean()) * bw + _exact(BRMS_XGP_PRIOR_X, sdgp, lscale, Z8)[np.array(BRMS_JGP_X) - 1]
    nojac = (stats.norm.logpdf(FRAME.y, mu, sigma).sum() + stats.norm.logpdf(a, 0, 5) + stats.norm.logpdf(bw, 0, 2)
             + stats.halfnorm.logpdf(sigma) + stats.halfnorm.logpdf(sdgp) + stats.invgamma.logpdf(lscale, 3, scale=1)
             + stats.norm.logpdf(Z8).sum())
    _check_against_hand(m, point, mu, nojac, nojac + np.log(sigma) + np.log(sdgp) + np.log(lscale))


def test_the_approximate_density_is_the_one_written_by_hand():
    m = gaussian("y ~ gp(x, k = 5, c = 1.5)", sdgp=ab.Exponential(1.5), lscale=ab.Gamma(2, 4))
    a, sigma, sdgp, lscale = -0.2, 0.9, 1.4, 0.45
    point = m.value_point(intercept=a, sigma=sigma, b=[], sdgp=sdgp, lscale=lscale, zgp=Z5)
    mu = a + _approximate(SCALED_X - CMEAN_X, 1.5, sdgp, lscale, Z5)
    nojac = (stats.norm.logpdf(FRAME.y, mu, sigma).sum() + stats.norm.logpdf(a, 0, 5) + stats.halfnorm.logpdf(sigma)
             + stats.expon.logpdf(sdgp, scale=1 / 1.5) + stats.gamma.logpdf(lscale, 2, scale=1 / 4) + stats.norm.logpdf(Z5).sum())
    _check_against_hand(m, point, mu, nojac, nojac + np.log(sigma) + np.log(sdgp) + np.log(lscale))


def test_the_spectral_density_keeps_a_finite_gradient_where_it_underflows():
    """sqrt(S) is formed as a product, so a large length-scale, where S underflows to 0 at the highest frequencies (and the
    gradient of sqrt(S) itself would be inf * 0), keeps a finite density and gradient."""
    m = gaussian("y ~ gp(x, k = 40)")
    point = m.value_point(intercept=0.1, b=[], sigma=1.0, sdgp=1.0, lscale=40.0, zgp=np.linspace(-1, 1, 40))
    w = np.pi * 40 / 2.5
    assert np.exp(-0.5 * 40.0 ** 2 * w ** 2) == 0.0                     # S underflows at the highest frequency
    assert np.isfinite(float(m.pymc_model.compile_logp()(point))) and np.isfinite(np.asarray(m.pymc_model.compile_dlogp()(point))).all()


def test_the_term_composes_with_other_constructs():
    """With weights() (B02) each row's log density is weighted; with a distributional formula (B01), a monotonic term (B03),
    correlated slopes (A02, whose post-build edit rebuilds the graph) and under a log or logit link the process is built and
    checked against the data as everywhere."""
    w = ab.gaussian("y | weights(t) ~ gp(x, k = 5, c = 1.5)", FRAME, priors())
    point = w.value_point(intercept=0.2, b=[], sigma=1.1, sdgp=0.7, lscale=0.3, zgp=Z5)
    rows = stats.norm.logpdf(FRAME.y, 0.2 + _approximate(SCALED_X - CMEAN_X, 1.5, 0.7, 0.3, Z5), 1.1)
    want = ((FRAME.t.to_numpy() * rows).sum() + stats.norm.logpdf(0.2, 0, 5) + stats.halfnorm.logpdf(1.1) + stats.halfnorm.logpdf(0.7)
            + stats.invgamma.logpdf(0.3, 3, scale=1) + stats.norm.logpdf(Z5).sum())
    assert float(w.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(want, rel=1e-12)
    d = ab.gaussian(["y ~ gp(x)", "sigma ~ w"], FRAME, priors(sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}))
    assert set(d.design.gp) == {"x"} and "sigma_w" in d.pymc_model.named_vars
    mo = ab.gaussian("y ~ mo(t) + gp(x)", FRAME, priors(b=ab.Normal(0, 1), simo=ab.Dirichlet(1)))
    assert {"simo_t_stickbreaking__", "mo(t)", "zgp_x"} <= {v.name for v in mo.pymc_model.value_vars}
    c = ab.gaussian("y ~ w + gp(x) + (1 + w | g)", FRAME, priors(b=ab.Normal(0, 1), cor=ab.LKJ(2),
                                                                 sd={"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfNormal(1)}}))
    assert {"zgp_x", "L|g_cholesky_corr__"} <= {v.name for v in c.pymc_model.value_vars}
    p = ab.poisson("cnt ~ gp(x) + gp(t, k = 4)", FRAME, ab.Priors(intercept=ab.Normal(0, 1), sdgp={"x": ab.HalfNormal(1), "t": ab.HalfNormal(1)},
                                                                 lscale={"x": ab.InverseGamma(3, 1), "t": ab.InverseGamma(3, 1)}))
    assert set(p.design.gp) == {"x", "t"}
    b = ab.bernoulli("yb ~ gp(t)", FRAME, ab.Priors(intercept=ab.Normal(0, 1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1)),
                     link="probit")
    assert set(b.design.gp) == {"t"}


FAMILIES = [("student", "y ~ gp(x, k = 4)", dict(sigma=ab.HalfNormal(1), nu=ab.Gamma(2, 0.1)), {}),
            ("poisson", "cnt ~ gp(x)", {}, {}), ("negbinomial", "cnt ~ gp(x, k = 4)", dict(shape=ab.Gamma(0.01, 0.01)), {}),
            ("gamma", "pos ~ gp(x)", dict(shape=ab.Gamma(0.01, 0.01)), {}), ("lognormal", "pos ~ gp(t)", dict(sigma=ab.HalfNormal(1)), {}),
            ("beta", "prop ~ gp(x)", dict(phi=ab.Gamma(0.01, 0.01)), {}), ("bernoulli", "yb ~ gp(x)", {}, {}),
            ("binomial", "yb | trials(t) ~ gp(x, k = 5)", {}, {"link": "probit"}), ("zero_inflated_poisson", "cnt ~ gp(x)", dict(zi=ab.Beta(1, 1)), {}),
            ("hurdle_poisson", "cnt ~ gp(x)", dict(hu=ab.Beta(1, 1)), {}), ("cumulative", "t ~ gp(x)", {}, {})]


@pytest.mark.parametrize("family, formula, extra, kwargs", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_every_family_but_the_categorical_takes_a_gaussian_process(family, formula, extra, kwargs):
    """The process enters the parent's linear predictor before the link, so every family's graph takes it; each is built and
    its contribution checked against the data at build (``_check_gp_effects``, under Bambi's identity or log link)."""
    m = getattr(ab, family)(formula, FRAME, ab.Priors(intercept=ab.Normal(0, 1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1),
                                                      **extra), **kwargs)
    assert len(m.design.gp) == 1 and ab.default_priors(formula, FRAME, family=family, **kwargs).lscale is not None


def test_the_build_checks_the_terms_contribution(monkeypatch):
    import autobayes.gaussian_process as gp

    honest_exact, honest_approximate = gp.exact_effect, gp.approximate_effect
    monkeypatch.setattr(gp, "exact_effect", lambda x, s, l, z: honest_exact(x[::-1], s, l, z))
    with pytest.raises(RuntimeError, match=r"the graph's 'mu' does not move by the values of 'gp\(x\)' at its rows' inputs"):
        gaussian()
    monkeypatch.setattr(gp, "approximate_effect", lambda phi, w, s, l, z: 2.0 * honest_approximate(phi, w, s, l, z))
    with pytest.raises(RuntimeError, match=r"does not move by the values of 'gp\(t, k = 4, c = 1\.25\)'"):   # under Bambi's log link too
        ab.poisson("cnt ~ gp(t, k = 4)", FRAME, ab.Priors(intercept=ab.Normal(0, 1), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1)))


# -- priors -----------------------------------------------------------------------------------------------------------
PRIOR_REFUSALS = [
    (dict(sdgp=None), r"the Gaussian process\(es\) \['gp\(x\)'\] need a prior on their marginal SD: pass sdgp=\{'x': \.\.\.\}"),
    (dict(lscale=None), r"need a prior on their length-scale: pass lscale=\{'x': \.\.\.\} \(brms class lscale"),
    (dict(lscale={"gpx": ab.InverseGamma(3, 1)}), r"lscale must cover exactly the columns of the Gaussian processes \['x'\]; got "
                                                  r"\['gpx'\]; key the prior by the column, lscale=\{'x': \.\.\.\}"),
    (dict(sdgp={"gp(x)": ab.HalfNormal(1)}), r"key the prior by the column, sdgp=\{'x': \.\.\.\}"),
    (dict(lscale=ab.Normal(0, 1)), r"prior for 'lscale:x' must be a positive-support prior .*got Normal"),
    (dict(sdgp="normal(0, 1)"), r"prior for 'sdgp:x' must be a Prior, got str"),
]


@pytest.mark.parametrize("extra, message", PRIOR_REFUSALS)
def test_prior_refusals(extra, message):
    with pytest.raises(ab.PriorError, match=message):
        gaussian(**extra)


def test_prior_forms():
    by_column = gaussian(sdgp={"x": ab.Exponential(2)}, lscale={"x": ab.HalfCauchy(1)})
    assert by_column.priors.sdgp_priors(by_column.spec) == {"x": ab.Exponential(2)}
    assert by_column.priors.lscale_priors(by_column.spec) == {"x": ab.HalfCauchy(1)}
    with pytest.raises(ab.PriorError, match=r"several Gaussian processes \['gp\(x\)', 'gp\(t\)'\] need an exact-key sdgp mapping"):
        gaussian("y ~ gp(x) + gp(t)")
    with pytest.raises(ab.PriorError, match=r"the formula has no Gaussian process gp\(x\); do not pass sdgp"):
        gaussian("y ~ w", b=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"the formula has no Gaussian process gp\(x\); do not pass lscale"):
        gaussian("y ~ w", b=ab.Normal(0, 1), sdgp=None)


def test_value_point_refusals():
    m = gaussian()
    base = dict(intercept=0.1, b=[], sigma=1.0)
    with pytest.raises(ValueError, match=r"sdgp, lscale and zgp are required for the Gaussian processes \['gp\(x\)'\]"):
        m.value_point(**base, sdgp=1.0, lscale=0.3)
    with pytest.raises(ValueError, match=r"zgp for 'gp\(x\)' must have 8 finite entries, one per distinct input"):
        m.value_point(**base, sdgp=1.0, lscale=0.3, zgp=Z5)
    with pytest.raises(ValueError, match=r"lscale for 'gp\(x\)' must be a finite positive scalar"):
        m.value_point(**base, sdgp=1.0, lscale=0.0, zgp=Z8)
    with pytest.raises(ValueError, match=r"sdgp must cover exactly the columns of the Gaussian processes \['x'\]; got \['gp\(x\)'\]"):
        m.value_point(**base, sdgp={"gp(x)": 1.0}, lscale=0.3, zgp=Z8)
    approx = gaussian("y ~ gp(x, k = 5)")
    with pytest.raises(ValueError, match=r"must have 5 finite entries, one per basis function"):
        approx.value_point(**base, sdgp=1.0, lscale=0.3, zgp=Z8)
    keyed = m.value_point(**base, sdgp={"x": 1.0}, lscale={"x": 0.3}, zgp={"x": Z8})
    assert keyed.keys() == m.value_point(**base, sdgp=1.0, lscale=0.3, zgp=Z8).keys()
    plain = ab.gaussian("y ~ w", FRAME, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"this model has no Gaussian process gp\(x\); sdgp, lscale and zgp do not apply"):
        plain.value_point(intercept=0.1, b=[0.2], sigma=1.0, zgp=Z8)


# -- predictions ------------------------------------------------------------------------------------------------------
def draws(model, column="x", z=Z8, sdgp=(0.8, 1.1), lscale=(0.35, 0.6), u=(0.2, -0.1, 0.05, 0.3)):
    """One chain of two draws: the raw-scale intercept, the process's sdgp, lscale and zgp (the same z in both draws) and, with
    a group term, the varying intercepts."""
    post = {"Intercept": np.full((1, 2), 0.3), "sigma": np.ones((1, 2)), f"sdgp_{column}": np.array([sdgp]),
            f"lscale_{column}": np.array([lscale]), f"zgp_{column}": np.broadcast_to(z, (1, 2, len(z)))}
    coords, dims = {f"zgp_{column}_dim": [str(j) for j in range(1, len(z) + 1)]}, {f"zgp_{column}": [f"zgp_{column}_dim"]}
    if model.spec.groups:
        post.update({"1|g": np.broadcast_to(np.array(u), (1, 2, 4)), "1|g_sigma": np.ones((1, 2))})
        coords["g__factor_dim"], dims["1|g"] = ["a", "b", "c", "d"], ["g__factor_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_of_the_exact_process():
    m = gaussian("y ~ gp(x) + (1 | g)", sd=ab.HalfNormal(1))
    post = draws(m)
    u = dict(zip("abcd", (0.2, -0.1, 0.05, 0.3)))
    f = np.stack([_exact(BRMS_XGP_PRIOR_X, s, l, Z8) for s, l in ((0.8, 0.35), (1.1, 0.6))])        # (draw, distinct input)
    fixed = 0.3 + f[:, np.array(BRMS_JGP_X) - 1]
    full = fixed + np.array([u[g] for g in FRAME.g])
    for kind, var in (("linear", "eta"), ("mean", "mu")):
        np.testing.assert_allclose(m.predict(post, kind=kind)[var][0], full, rtol=1e-13)
        np.testing.assert_allclose(m.predict(post, kind=kind, include_group_effects=False)[var][0], fixed, rtol=1e-13)
    new = pd.DataFrame({"x": [3.1, -0.9, 0.7], "g": ["b", "a", "d"]})               # fitted inputs: the process's own values
    want = 0.3 + f[:, [3, 5, 0]] + np.array([-0.1, 0.2, 0.3])
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="mean").mu[0], want, rtol=1e-13)
    with pytest.raises(ab.FormulaError, match=r"newdata column 'x' has the values \[0\.5\], which are not inputs of the exact "
                                              r"Gaussian process 'gp\(x\)'.*conditional normal.*D65"):
        m.predict(post, newdata=new.assign(x=[3.1, 0.5, 0.7]), kind="mean")
    with pytest.raises(ab.FormulaError, match=r"newdata lacks columns \['x'\]"):
        m.predict(post, newdata=new.drop(columns="x"), kind="mean")


def test_predictions_of_the_approximate_process():
    m = gaussian("y ~ gp(x, k = 5, c = 1.5)")
    post = draws(m, z=Z5)
    for (s, l), row in zip(((0.8, 0.35), (1.1, 0.6)), m.predict(post, kind="linear")["eta"][0].to_numpy()):
        np.testing.assert_allclose(row, 0.3 + _approximate(SCALED_X - CMEAN_X, 1.5, s, l, Z5), rtol=1e-13)
    new = pd.DataFrame({"x": [0.0, 4.5, -2.0]})                                      # unfitted inputs inside the boundary
    got = m.predict(post, newdata=new, kind="mean").mu[0].to_numpy()
    for (s, l), row in zip(((0.8, 0.35), (1.1, 0.6)), got):
        np.testing.assert_allclose(row, 0.3 + _approximate(new.x.to_numpy() / 4 - CMEAN_X, 1.5, s, l, Z5), rtol=1e-13)
    y = m.predict(post, newdata=new, kind="response", seed=1)["y"]                    # new observations around that mean
    assert y.shape == (1, 2, 3) and np.isfinite(y).all()
    lo, hi = (CMEAN_X - 1.5) * 4, (CMEAN_X + 1.5) * 4
    for bad in (hi + 0.01, lo - 0.01, 9.0):
        with pytest.raises(ab.FormulaError, match=r"outside the boundary of the approximate Gaussian process 'gp\(x, k = 5, c = 1\.5\)'"
                                                  r".*refit with a larger c"):
            m.predict(post, newdata=pd.DataFrame({"x": [0.0, bad]}), kind="mean")


# -- describe(), names, identity, records, prior predictive -----------------------------------------------------------
def test_describe_and_names():
    m = gaussian("y ~ w + gp(x) + (1 | g)", b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
    text = m.describe()
    for line in ("AutoBayes M0 model with Gaussian process(es)",
                 "  gp(x)     : exact Gaussian process of 'x' (brms gp(x)), squared-exponential kernel; 8 distinct inputs from -0.9 to 3.1, "
                 "scaled by dmax = 4, their largest distance, so that it is 1 (brms's scale = TRUE)",
                 "  mu_i = a_c + (x_w - mean(x_w)) * b_w + gp(x)_i + u[g_i]",
                 "f = chol(K + 1e-12 I) zgp_x, K[j, m] = sdgp_x^2 exp(-(u_j - u_m)^2 / (2 lscale_x^2))",
                 "inputs lscale_x * 4 apart in x are correlated exp(-1/2) = 0.61",
                 "  New rows take the process's value at a fitted input of x; any other input is refused",
                 "  sdgp_x ~ HalfNormal(sigma=1)   (brms class sdgp, coefficient gpx)",
                 "  lscale_x ~ InverseGamma(alpha=3, beta=1)   (brms class lscale, coefficient gpx; on the scaled input)",
                 "  zgp_x                    zgp_x[1..8], the standardised values of gp(x), one per distinct input (brms's zgp); dims zgp_x_dim"):
        assert line in text, line
    assert m.gp_names == {"x": {"sdgp": "sdgp_x", "lscale": "lscale_x", "zgp": "zgp_x", "coordinate": "zgp_x_dim"}}
    assert m.names["sdgp:x"] == "sdgp_x" and m.names["lscale:x"] == "lscale_x" and m.names["zgp:x"] == "zgp_x"
    assert m.coefficients == (("w", "w", None),) and "b:gp(x)" not in m.names
    approx = gaussian("y ~ gp(x, k = 5, c = 1.5)").describe()
    for line in ("k = 5 basis functions on [-L, L], L = c = 1.5, of the scaled input less the mean of its distinct values, 0.2625",
                 "phi_j(u) = sin(pi j (u + L) / (2 L)) / sqrt(L)", "S(w) = sdgp_x^2 sqrt(2 pi) lscale_x exp(-lscale_x^2 w^2 / 2)",
                 "  New rows: the basis at their inputs, which must lie inside the boundary, x in (-4.95, 7.05)"):
        assert line in approx, line
    with pytest.raises(ab.FormulaError, match=r"generated variable/coordinate name collision: \['sdgp_x'\]"):
        gaussian("y ~ gp(x) + sdgp_x", FRAME.assign(sdgp_x=FRAME.w), b=ab.Normal(0, 1))


def test_the_target_identity_holds_the_term():
    base = gaussian().target_identity
    assert gaussian().target_identity == base
    assert gaussian(lscale=ab.InverseGamma(3, 2)).target_identity != base
    assert gaussian(frame=FRAME.assign(x=FRAME.x * 2)).target_identity != base        # the same scaled inputs from other values
    assert gaussian("y ~ gp(x, k = 5)").target_identity != gaussian("y ~ gp(x, k = 5, c = 1.5)").target_identity
    assert gaussian(frame=FRAME.assign(x=FRAME.x + 1.0)).target_identity != base          # the same distances, other inputs
    assert gaussian(sdgp=ab.HalfNormal(2)).target_identity != base


def test_fit_records_and_summary_rows(monkeypatch):
    m = gaussian()
    post = draws(m)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert {"sdgp_x", "lscale_x"} <= set(rows) and not any(r.startswith("zgp_x") for r in rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=2,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["sdgp:x"] == "HalfNormal(sigma=1)" and rec["model"]["priors"]["lscale:x"] == "InverseGamma(alpha=3, beta=1)"
    assert rec["model"]["gaussian_processes"] == {"gp(x)": {"column": "x", "approximate": False, "k": None, "c": None,
                                                            "distinct_inputs": 8, "dmax": 4.0, "cmean": m.spec.gp("x").cmean,
                                                            "names": m.gp_names["x"]}}
    assert rec["model"]["terms"] == [] and rec["model"]["canonical_formula"] == "y ~ gp(x)"
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_prior_predictive_draws_the_process():
    pp = gaussian("y ~ gp(x, k = 5)").prior_predictive(draws=40, seed=3)
    assert pp.prior["zgp_x"].shape == (1, 40, 5) and (pp.prior["lscale_x"] > 0).all() and (pp.prior["sdgp_x"] > 0).all()
    assert pp.prior_predictive["y"].shape == (1, 40, 12)


# -- brms's default priors ------------------------------------------------------------------------------------------------
def test_the_tuned_inverse_gamma_has_one_percent_tails():
    """The exact solution, checked against 50-digit values (mpmath) and its own tails; brms's floor of 0.01 on the smallest
    distance; the parameters rounded to 6 decimals reproduce what brms printed for the two frozen fixtures."""
    for lower, (a50, b50) in ((1 / 3, (18.506436825829016228, 9.9848112399774948779)), (0.01, (1.4941966870247053117, 0.05660740814638979045)),
                              (0.5, (45.631904608849696562, 31.399697537362833312))):
        a, b = inverse_gamma_tails(lower, 1.0)
        assert a == pytest.approx(a50, rel=1e-12) and b == pytest.approx(b50, rel=1e-12)
        assert special.gammaincc(a, b / lower) == pytest.approx(0.01, rel=1e-12)       # P(lscale < lower)
        assert special.gammainc(a, b / 1.0) == pytest.approx(0.01, rel=1e-12)          # P(lscale > 1)
    a, b = inverse_gamma_tails(0.2, 2.0)                                              # the scale family: b scales with the bounds
    assert (a, b) == pytest.approx((inverse_gamma_tails(0.1, 1.0)[0], 2 * inverse_gamma_tails(0.1, 1.0)[1]), rel=1e-12)
    with pytest.raises(ValueError, match="need 0 < lower < upper"):
        inverse_gamma_tails(1.0, 1.0)


# brms 2.23.0's default_prior() for y ~ gp(x) on x = (0, l, 1) and on (0, 0.005, 0.3, 0.6, 1): the smallest distance l (scaled; at
# least 0.01) and what brms printed. At l = 0.5 and l = 0.48188060200668892 brms's iterative solution stops just below a rounding
# boundary that the exact one crosses (50-digit values 45.6319046088 / 31.3996975374 and 27.7431735383): divergence D67.
BRMS_LSCALE = [((0, 0.05, 0.3, 0.6, 1), (2.938537, 0.415087)), ((0, 0.005, 0.3, 0.6, 1), (1.494197, 0.056607)),
               ((0, 0.011, 1), (1.541023, 0.063302)), ((0, 0.02, 1), (1.91136, 0.129447)), ((0, 0.1, 1), (4.62909, 1.103366)),
               ((0, 0.25, 0.5, 1), (11.830309, 5.315674)), ((0, 1 / 3, 1), (18.506437, 9.984811))]
BRMS_LSCALE_STOPS_SHORT = [((0, 0.5, 1), (45.631904, 31.399697), (45.631905, 31.399698)),
                           ((0, 0.48188060200668892, 1), (41.190626, 27.743173), (41.190626, 27.743174))]


def _default_lscale(x):
    frame = pd.DataFrame({"y": np.linspace(0.1, 0.5, len(x)), "x": x})
    return ab.default_priors("y ~ gp(x)", frame).lscale["x"]


@pytest.mark.parametrize("x, printed", BRMS_LSCALE)
def test_the_default_lscale_is_brms(x, printed):
    assert _default_lscale(x) == ab.InverseGamma(*printed)


@pytest.mark.parametrize("x, brms, ours", BRMS_LSCALE_STOPS_SHORT)
def test_where_brms_stops_short_of_the_exact_solution(x, brms, ours):
    assert _default_lscale(x) == ab.InverseGamma(*ours) != ab.InverseGamma(*brms)


# brms 2.23.0's default_prior() on FRAME: (class, coef, group, prior, lb); every other column empty, source "default"
BRMS = {
    "gaussian, gp(x)": ("y ~ gp(x)", "gaussian", [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("lscale", "", "", "", "0"), ("lscale", "gpx", "", "inv_gamma(4.62909, 1.103366)", ""),
        ("sdgp", "", "", "student_t(3, 0, 2.5)", "0"), ("sdgp", "gpx", "", "", ""), ("sigma", "", "", "student_t(3, 0, 2.5)", "0")]),
    "gaussian, w + gp(t) + (1 | g)": ("y ~ w + gp(t) + (1 | g)", "gaussian", [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", ""), ("b", "w", "", "", ""), ("lscale", "", "", "", "0"),
        ("lscale", "gpt", "", "inv_gamma(18.506437, 9.984811)", ""), ("sd", "", "", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g", "", ""),
        ("sd", "Intercept", "g", "", ""), ("sdgp", "", "", "student_t(3, 0, 2.5)", "0"), ("sdgp", "gpt", "", "", ""),
        ("sigma", "", "", "student_t(3, 0, 2.5)", "0")]),
    "poisson, two processes": ("cnt ~ gp(x) + gp(t, k = 4)", "poisson", [
        ("Intercept", "", "", "student_t(3, 0, 2.5)", ""), ("lscale", "", "", "", "0"), ("lscale", "gpt", "", "inv_gamma(18.506437, 9.984811)", ""),
        ("lscale", "gpx", "", "inv_gamma(4.62909, 1.103366)", ""), ("sdgp", "", "", "student_t(3, 0, 2.5)", "0"), ("sdgp", "gpt", "", "", ""),
        ("sdgp", "gpx", "", "", "")]),
    "two distinct inputs": ("y ~ gp(two)", "gaussian", [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("lscale", "", "", "", "0"), ("lscale", "gptwo", "", "normal(0, 0.5)", ""),
        ("sdgp", "", "", "student_t(3, 0, 2.5)", "0"), ("sdgp", "gptwo", "", "", ""), ("sigma", "", "", "student_t(3, 0, 2.5)", "0")]),
    "a large response scale": ("big ~ gp(x, k = 6)", "gaussian", [
        ("Intercept", "", "", "student_t(3, 8.1, 22.8)", ""), ("lscale", "", "", "", "0"), ("lscale", "gpx", "", "inv_gamma(4.62909, 1.103366)", ""),
        ("sdgp", "", "", "student_t(3, 0, 22.8)", "0"), ("sdgp", "gpx", "", "", ""), ("sigma", "", "", "student_t(3, 0, 22.8)", "0")]),
    "no intercept": ("y ~ 0 + gp(t)", "gaussian", [
        ("lscale", "", "", "", "0"), ("lscale", "gpt", "", "inv_gamma(18.506437, 9.984811)", ""), ("sdgp", "", "", "student_t(3, 0, 2.5)", "0"),
        ("sdgp", "gpt", "", "", ""), ("sigma", "", "", "student_t(3, 0, 2.5)", "0")]),
}


@pytest.mark.parametrize("label", sorted(BRMS))
def test_the_default_prior_table_is_brms(label):
    formula, family, rows = BRMS[label]
    pri = ab.default_priors(formula, FRAME, family=family)
    model = getattr(ab, family)(formula, FRAME, pri)
    table = model.prior_table()
    assert [(r["class"], r["coef"], r["group"], r["prior"], r["lb"]) for r in table] == rows
    assert all(r["ub"] == r["resp"] == r["dpar"] == r["nlpar"] == r["tag"] == "" and r["source"] == "default" for r in table)
    columns = [g.column for g in model.spec.gp_terms]
    assert set(pri.sdgp) == set(pri.lscale) == set(columns) and all(p.source == "default" for p in (*pri.sdgp.values(), *pri.lscale.values()))
    assert [lab for lab in model.default_prior_labels if lab.startswith(("sdgp:", "lscale:"))] == (
        [f"sdgp:{c}" for c in columns] + [f"lscale:{c}" for c in columns])


def test_the_user_prior_table_puts_lscale_on_its_coefficient_row():
    """brms's prior_summary of the frozen case: a user's lscale prior sits on the coefficient row and the class row stays empty."""
    table = gaussian().prior_table()
    assert [(r["class"], r["coef"], r["prior"], r["lb"], r["source"]) for r in table if r["class"] in ("lscale", "sdgp")] == [
        ("lscale", "", "", "0", "default"), ("lscale", "gpx", "inv_gamma(3, 1)", "", "user"),
        ("sdgp", "", "normal(0, 1)", "0", "user"), ("sdgp", "gpx", "", "", "user")]
    two = gaussian("y ~ gp(x) + gp(t)", sdgp={"x": ab.HalfNormal(1), "t": ab.HalfNormal(2)},
                   lscale={"x": ab.InverseGamma(3, 1), "t": ab.InverseGamma(3, 1)}).prior_table()
    assert [(r["class"], r["coef"], r["prior"]) for r in two if r["class"] == "sdgp"] == [
        ("sdgp", "", ""), ("sdgp", "gpt", "normal(0, 2)"), ("sdgp", "gpx", "normal(0, 1)")]


# -- the oracle's extension to Gaussian processes ---------------------------------------------------------------------------
def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


@pytest.mark.parametrize("case", ["gauss_gp_exact", "gauss_gp_approx"])
def test_the_gates_see_a_corrupted_gp_fixture(case):
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    b = gates.Bundle(case)
    for gate in gates.ALL:
        gate(b)

    def index(raw):                                                      # a row moved to another distinct input
        values = raw["standata"]["Jgp_1"]["values"]
        values[0] = values[0] % int(raw["standata"]["Nsubgp_1"]["values"][0]) + 1
    with pytest.raises(OracleMismatch, match=r"0 brms Gaussian processes carry its row index"):
        _corrupted(b, index)

    def order(raw):                                                      # the distinct inputs in another order
        x = raw["standata"]["Xgp_1"]
        n = int(raw["standata"]["Nsubgp_1"]["values"][0])
        x["values"][:n] = x["values"][:n][::-1]
    with pytest.raises(OracleMismatch, match=r"0 brms Gaussian processes carry"):
        _corrupted(b, order)

    def scaling(raw):
        raw["standata"]["dmax_1"]["values"][0] *= 1.0 + 1e-9
    with pytest.raises(OracleMismatch, match=r"brms dmax_1 is"):
        gates.check_design(_corrupted(b, scaling))

    def latent(raw):                                                     # brms's density at a moved zgp
        raw["points"][3]["pars"]["zgp_1"]["values"][1] += 1e-3
    with pytest.raises(OracleMismatch, match=r"point 3: log density"):
        gates.check_logp(_corrupted(b, latent))
