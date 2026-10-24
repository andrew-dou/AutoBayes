"""brms's ``se()`` and ``rate()`` response addition terms (brms-parity feature F06a; docs/se-rate.md, divergences D156, D157).

The grammar and every refusal by message; the data's validation; the likelihood of each form against SciPy at chosen natural
values (``se(col)``: ``Normal(mu, se_i)`` and the Student-t's with no sigma; ``se(col, sigma = TRUE)``: the scale
``sqrt(sigma^2 + se_i^2)``; ``rate(col)``: ``Poisson(exp(eta_i) d_i)`` and ``NegBinomial2(exp(eta_i) d_i, shape d_i)``), the graph's
mean, compositions with other terms, the family spec without sigma, priors and brms's defaults, predictions (linear, mean and
seeded draws, on fitted and new rows), ``log_lik()``, prior-predictive draws, ``describe()``, the target identity and the fit
record, and the oracle's G1 failing when the terms' data are dropped or misread. The numbers against brms are the oracle's
(tests/oracle: ``gauss_se_meta``, ``gauss_se_sigma``, ``student_se_sigma``, ``poisson_rate``, ``negbinomial_rate``). Nothing
samples."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import autobayes as ab
import autobayes.se_rate as sr
from autobayes.families import family_of, get_family
from autobayes.formula import AdditionTerm, addition_data
from tests.oracle import adapter, datasets, gates
from tests.oracle.adapter import OracleMismatch
from tests.test_population_only import PANEL_STUB

DATA = datasets.tier_f2(961, 36)            # y, sdx (standard errors), cnt and ynb (counts), exposure (denominators), x, z, g, ...
GAUSS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
META = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1))           # se(col): no sigma
STUDENT = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1), nu=ab.Gamma(2, 0.1))
COUNTS = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
NB = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), shape=ab.Gamma(0.01, 0.01))
NAT = {"intercept": 0.3, "b": np.array([0.4]), "sd": 0.7, "sigma": 0.9, "nu": 5.0, "shape": 2.5}


def _z(model):
    return np.linspace(-1.2, 1.1, model.design.J)


def _eta(model, nat):
    """The linear predictor of ``y ~ x + (1 | g)`` at the natural values, written out: the centred intercept, the slope of the
    centred x and the varying intercept sd * z of each row's level."""
    d = model.design
    return nat["intercept"] + (d.X_raw[:, 0] - d.x_means[0]) * nat["b"][0] + nat["sd"] * _z(model)[d.group_index]


def _value_point(model, **extra):
    nat = {k: NAT[k] for k in ("intercept", "b", "sd")}
    return model.value_point(**nat, z=_z(model), **extra)


def _rows(model, point):
    """The model's log likelihood per row, and its parent deterministic (Bambi's mu), at a value point."""
    pmm = model.pymc_model
    rv = pmm[model.names["y"]]
    rows = pmm.compile_fn(pmm.logp(vars=[rv], sum=False)[0], inputs=pmm.value_vars, on_unused_input="ignore")(point)
    mu = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[model.names["mean"]]])[0], inputs=pmm.value_vars,
                        on_unused_input="ignore")(point)
    return np.asarray(rows, dtype=float), np.asarray(mu, dtype=float).ravel()


@pytest.fixture(scope="module")
def models():
    return {
        "meta": ab.gaussian("y | se(sdx) ~ x + (1 | g)", DATA, META),
        "sigma": ab.gaussian("y | se(sdx, sigma = TRUE) ~ x + (1 | g)", DATA, GAUSS),
        "student_meta": ab.student("y | se(sdx) ~ x + (1 | g)", DATA, META, nu=4),
        "student_sigma": ab.student("y | se(sdx, sigma = TRUE) ~ x + (1 | g)", DATA, STUDENT),
        "poisson": ab.poisson("cnt | rate(exposure) ~ x + (1 | g)", DATA, COUNTS),
        "negbinomial": ab.negbinomial("ynb | rate(exposure) ~ x + (1 | g)", DATA, NB),
    }


# -- 1. the grammar and the term model ------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,sigma,term", [
    ("se(sdx)", False, "se(sdx)"), ("se( sdx )", False, "se(sdx)"), ("se(sdx, sigma = TRUE)", True, "se(sdx, sigma = TRUE)"),
    ("se(sdx,sigma=T)", True, "se(sdx, sigma = TRUE)"), ("se(sdx, TRUE)", True, "se(sdx, sigma = TRUE)"),
    ("se(sdx, sigma = FALSE)", False, "se(sdx)"), ("se(sdx, F)", False, "se(sdx)"),
])
def test_se_grammar_and_canonical_form(text, sigma, term):
    s = ab.parse_formula(f"y | {text} ~ x + (1 | g)")
    a = s.addition("se")
    assert a == AdditionTerm("se", column="sdx", sigma=sigma) and (a.term, a.columns) == (term, ("sdx",))
    assert s.canonical_formula == f"y | {term} ~ x + (1 | g)" and s.bambi_formula == "y ~ x + (1 | g)"


def test_rate_grammar_and_canonical_form():
    s = ab.parse_formula("cnt|rate( exposure ) ~ x", family="poisson")
    assert s.additions == (AdditionTerm("rate", column="exposure"),) and s.addition("rate").term == "rate(exposure)"
    assert s.canonical_formula == "cnt | rate(exposure) ~ x" and s.bambi_formula == "cnt ~ x"
    assert ab.parse_formula("ynb | rate(exposure) ~ x", family="negbinomial").addition("rate").columns == ("exposure",)


@pytest.mark.parametrize("formula,family,needle", [
    ("y | se(sdx, sigma = maybe) ~ x", "gaussian", r"se\(\) takes the column of the standard errors and, to learn sigma as well"),
    ("y | se(sdx, TRUE, 2) ~ x", "gaussian", r"se\(\) takes one column name of standard errors"),
    ("y | se(2 * sdx) ~ x", "gaussian", r"'se\(2 \* sdx\)': se\(\) takes one column name.*compute the column; docs/se-rate.md, divergence D157"),
    ("y | se() ~ x", "gaussian", r"se\(\) takes one column name"),
    ("cnt | rate(2 * exposure) ~ x", "poisson", r"rate\(\) takes one column name of denominators.*divergence D157"),
    ("cnt | rate(exposure, x) ~ x", "poisson", r"rate\(\) takes one column name"),
    ("cnt | se(sdx) ~ x", "poisson", r"'se\(sdx\)' is not available for the poisson family; brms's se\(\) is verified here for the "
                                     r"gaussian, student families"),
    ("y | rate(exposure) ~ x", "gaussian", r"'rate\(exposure\)' is not available for the gaussian family; brms's rate\(\) is verified "
                                           r"here for the poisson, negbinomial families"),
    ("ypos | se(sdx) ~ x", "lognormal", r"'se\(sdx\)' is not available for the lognormal family"),
    ("y | se(sdx) + weights(w) ~ x", "gaussian", r"'se\(sdx\)' together with 'weights\(w\)' is not supported in this slice.*D157"),
    ("y | weights(w) + se(sdx, sigma = TRUE) ~ x", "gaussian", r"'se\(sdx, sigma = TRUE\)' together with 'weights\(w\)'"),
    ("y | se(sdx) + cens(cens_ind) ~ x", "gaussian", r"'se\(sdx\)' together with 'cens\(cens_ind\)'"),
    ("cnt | rate(exposure) + trunc(ub = 30) ~ x", "poisson", r"'rate\(exposure\)' together with 'trunc\(ub = 30\)'"),
    ("y | se(sdx) + se(w) ~ x", "gaussian", r"se\(\) appears more than once on the left-hand side"),
])
def test_refusals_name_the_construct(formula, family, needle):
    with pytest.raises(ab.FormulaError, match=needle):
        ab.parse_formula(formula, family=family)


def test_refused_contexts_name_the_construct():
    """Distributional formulas (D157), the E06 links (D133) and the model structures that take no addition term."""
    with pytest.raises(ab.FormulaError, match=r"'se\(sdx, sigma = TRUE\)' with the distributional formula 'sigma ~ x' is not supported.*D157"):
        ab.gaussian(["y | se(sdx, sigma = TRUE) ~ x", "sigma ~ x"], DATA, GAUSS)
    with pytest.raises(ab.FormulaError, match=r"'se\(sdx\)' with the distributional formula 'sigma ~ x'"):
        ab.parse_formula(["y | se(sdx) ~ x", "sigma ~ x"])
    with pytest.raises(ab.FormulaError, match=r"'rate\(exposure\)' with the distributional formula 'shape ~ x'"):
        ab.parse_formula(["ynb | rate(exposure) ~ x", "shape ~ x"], family="negbinomial")
    assert ab.parse_formula(["y | weights(w) ~ x", "sigma ~ x"]).dpar_names == ("sigma",)   # other addition terms keep their formulas
    with pytest.raises(ab.FormulaError, match=r"the addition term 'rate\(exposure\)' is not available under the sqrt link.*D133"):
        ab.poisson("cnt | rate(exposure) ~ x", DATA, COUNTS, link="sqrt")
    with pytest.raises(ab.FormulaError, match=r"the addition term 'rate\(exposure\)' is not available under the identity link.*D133"):
        ab.negbinomial("ynb | rate(exposure) ~ x", DATA, NB, link="identity")
    with pytest.raises(ab.FormulaError, match=r"the autocorrelation term 'ar\(time = t, p = 1\)' with the response addition term\(s\) se\(sdx\)"):
        ab.parse_formula("y | se(sdx) ~ x + ar(t)")
    with pytest.raises(ab.FormulaError, match=r"response addition terms .* with a non-linear formula"):
        ab.gaussian(["y | se(sdx) ~ a * exp(x)", "a ~ 1"], DATA, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)),
                    nl=True)


@pytest.mark.parametrize("formula,family,needle", [
    ("y | se(sdx) ~ x + fcor(M)", "gaussian", r"fcor\(M\) with the response addition term\(s\) se\(sdx\) is not supported.*D144"),
    ("y | se(sdx) ~ x + unstr(tt, subj)", "gaussian", r"'unstr\(time = tt, gr = subj\)' with the response addition term\(s\) se\(sdx\)"),
    ("y | se(sdx, sigma = TRUE) ~ x + cosy(tt, subj)", "student", r"'cosy\(time = tt, gr = subj\)' with the response addition term\(s\) "
                                                                  r"se\(sdx, sigma = TRUE\)"),
    ("cnt | rate(exposure) ~ x + car(W, gr = site, type = 'icar')", "poisson", r"next to the addition term 'rate\(exposure\)' is not supported"),
    ("y | se(sdx) ~ x + sar(M)", "gaussian", r"the SAR term 'sar\(M, type = \"lag\"\)' next to the response addition term 'se\(sdx\)'"),
])
def test_residual_structures_refuse_them(formula, family, needle):
    """The residual-correlation and spatial terms (F02a, F02b, F03a, F03b) take no addition term: each names se() or rate()."""
    with pytest.raises(ab.FormulaError, match=needle):
        ab.parse_formula(formula, family=family)


# -- 2. the data -----------------------------------------------------------------------------------------------------------
def test_addition_data_and_design():
    spec = ab.parse_formula("y | se(sdx) ~ x")
    np.testing.assert_array_equal(addition_data(spec, DATA)["se"], DATA["sdx"].to_numpy())
    spec = ab.parse_formula("cnt | rate(exposure) ~ x", family="poisson")
    assert set(addition_data(spec, DATA)) == {"denom"}
    np.testing.assert_array_equal(addition_data(spec, DATA)["denom"], DATA["exposure"].to_numpy())


@pytest.mark.parametrize("formula,family,column,value,needle", [
    ("y | se(sdx) ~ x", "gaussian", "sdx", -0.1, r"the standard errors of 'se\(sdx\)' must be non-negative \(brms: 'Standard errors must"),
    ("y | se(sdx, sigma = TRUE) ~ x", "gaussian", "sdx", -0.1, r"'se\(sdx, sigma = TRUE\)' must be non-negative.*row 3 has -0.1"),
    ("y | se(sdx) ~ x", "gaussian", "sdx", 0.0, r"must be positive: without sigma = TRUE row 3's scale is its standard error.*D156"),
    ("y | se(sdx) ~ x", "gaussian", "sdx", np.nan, r"column 'sdx' of 'se\(sdx\)' has missing or non-finite values"),
    ("y | se(sdx) ~ x", "gaussian", "sdx", np.inf, r"column 'sdx' of 'se\(sdx\)' has missing or non-finite values"),
    ("cnt | rate(exposure) ~ x", "poisson", "exposure", 0.0, r"the denominators of 'rate\(exposure\)' must be positive \(brms refuses a zero or negative one too\)"),
    ("cnt | rate(exposure) ~ x", "poisson", "exposure", -2.0, r"row 3 has -2.0"),
    ("cnt | rate(exposure) ~ x", "poisson", "exposure", np.nan, r"'rate\(exposure\)' has missing or non-finite values"),
])
def test_invalid_values_are_refused(formula, family, column, value, needle):
    spec = ab.parse_formula(formula, family=family)
    bad = DATA.copy()
    bad.loc[3, column] = value
    with pytest.raises(ab.FormulaError, match=needle):
        addition_data(spec, bad)


def test_non_numeric_and_complex_columns_are_refused():
    spec = ab.parse_formula("y | se(sdx) ~ x")
    with pytest.raises(ab.FormulaError, match=r"has dtype .*; the standard errors of 'se\(sdx\)' must be numeric"):
        addition_data(spec, DATA.assign(sdx=DATA["g"]))
    with pytest.raises(ab.FormulaError, match=r"the standard errors of 'se\(sdx\)' must be numeric"):
        addition_data(spec, DATA.assign(sdx=DATA["sdx"] > 0.3))
    with pytest.raises(ab.FormulaError, match=r"complex-valued"):
        addition_data(spec, DATA.assign(sdx=DATA["sdx"] + 0j))
    spec = ab.parse_formula("cnt | rate(exposure) ~ x", family="poisson")
    with pytest.raises(ab.FormulaError, match=r"the denominators of 'rate\(exposure\)' must be numeric"):
        addition_data(spec, DATA.assign(exposure=DATA["g"]))
    assert sr.se_values("s", pd.Series([0.0, 0.2]), "column", sigma=True).tolist() == [0.0, 0.2]   # a zero is fine with sigma


# -- 3. the likelihood against SciPy at chosen natural values -----------------------------------------------------------------
def test_gaussian_se_without_sigma(models):
    m = models["meta"]
    rows, mu = _rows(m, _value_point(m))
    eta = _eta(m, NAT)
    np.testing.assert_allclose(mu, eta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(rows, stats.norm.logpdf(m.design.y, eta, DATA["sdx"].to_numpy()), rtol=1e-12, atol=1e-12)
    assert "sigma" not in m.names and m.family_spec.scale_parameter is None and m.family == "gaussian"
    assert family_of(m) is m.family_spec is sr.se_only_spec(get_family("gaussian")) and m.family_spec.se_parameter == "sigma"
    assert "sigma" not in {v.name for v in m.pymc_model.free_RVs} and "sigma" not in m.pymc_model.named_vars


def test_gaussian_se_with_sigma(models):
    m = models["sigma"]
    rows, mu = _rows(m, _value_point(m, sigma=NAT["sigma"]))
    scale = np.sqrt(NAT["sigma"] ** 2 + DATA["sdx"].to_numpy() ** 2)
    np.testing.assert_allclose(rows, stats.norm.logpdf(m.design.y, _eta(m, NAT), scale), rtol=1e-12, atol=1e-12)
    assert m.names["sigma"] == "sigma" and family_of(m) is get_family("gaussian")


def test_student_se_with_and_without_sigma(models):
    m = models["student_sigma"]
    rows, _ = _rows(m, _value_point(m, sigma=NAT["sigma"], nu=NAT["nu"]))
    scale = np.sqrt(NAT["sigma"] ** 2 + DATA["sdx"].to_numpy() ** 2)
    np.testing.assert_allclose(rows, stats.t.logpdf(m.design.y, NAT["nu"], _eta(m, NAT), scale), rtol=1e-12, atol=1e-12)
    m = models["student_meta"]                                           # a fixed nu, no sigma
    rows, _ = _rows(m, _value_point(m))
    np.testing.assert_allclose(rows, stats.t.logpdf(m.design.y, 4.0, _eta(m, NAT), DATA["sdx"].to_numpy()), rtol=1e-12, atol=1e-12)
    assert family_of(m) is sr.se_only_spec(get_family("student")) and "sigma" not in m.names and m.nu == 4.0


def test_poisson_rate(models):
    m = models["poisson"]
    rows, mu = _rows(m, _value_point(m))
    d, eta = DATA["exposure"].to_numpy(), _eta(m, NAT)
    np.testing.assert_allclose(mu, np.exp(eta) * d, rtol=1e-13)          # the graph's mu is brms's posterior_epred
    np.testing.assert_allclose(rows, stats.poisson.logpmf(m.design.y, np.exp(eta) * d), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(m.design.denom, d)
    assert m.design.se is None and models["meta"].design.denom is None


def test_negbinomial_rate(models):
    m = models["negbinomial"]
    rows, mu = _rows(m, _value_point(m, shape=NAT["shape"]))
    d, eta = DATA["exposure"].to_numpy(), _eta(m, NAT)
    n = NAT["shape"] * d                                                 # brms's shape * denom
    np.testing.assert_allclose(mu, np.exp(eta) * d, rtol=1e-13)
    np.testing.assert_allclose(rows, stats.nbinom.logpmf(m.design.y, n, n / (n + np.exp(eta) * d)), rtol=1e-11, atol=1e-11)
    assert get_family("negbinomial").rate_parameters == ("alpha",) and get_family("poisson").rate_parameters == ()


def test_the_jax_path_equals_pytensor(models):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    for name, extra in (("student_meta", {}), ("negbinomial", {"shape": 2.5}), ("meta", {})):
        m = models[name]
        point = _value_point(m, **extra)
        names = [v.name for v in m.pymc_model.value_vars]
        a, b = float(get_jaxified_logp(m.pymc_model)([point[k] for k in names])), float(m.pymc_model.compile_logp()(point))
        assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), name


@pytest.mark.parametrize("family,formula,priors,kwargs,plain", [
    ("poisson", "cnt | rate(exposure) ~ x + offset(log(xpos)) + (1 + x | g)",            # an offset and correlated slopes
     ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
               cor={"g": ab.LKJ(2)}), {}, "cnt ~ x + offset(log(xpos)) + (1 + x | g)"),
    ("negbinomial", "ynb | rate(exposure) ~ x + mo(ord)",                                # a monotonic term
     ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(0.01, 0.01), simo={"ord": ab.Dirichlet(1)}), {},
     "ynb ~ x + mo(ord)"),
    ("gaussian", "y | se(sdx) ~ x + z + (1 | g:h)", META, {"categorical": {"z": "p"}}, "y ~ x + z + (1 | g:h)"),   # a factor, g:h
    ("gaussian", "y | se(sdx, sigma = TRUE) ~ x + mo(ord)",
     ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), simo={"ord": ab.Dirichlet(1)}), {}, "y ~ x + mo(ord)"),
    ("gaussian", "y | se(sdx, sigma = TRUE) ~ me(xobs, xpos)",                           # a measurement-error term
     ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), meanme=ab.Normal(0, 1), sdme=ab.HalfNormal(1)), {},
     "y ~ me(xobs, xpos)"),
    ("poisson", "cnt | rate(exposure) ~ gp(x)",                                          # a Gaussian process
     ab.Priors(intercept=ab.Normal(0, 2), sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1)), {}, "cnt ~ gp(x)"),
])
def test_the_terms_compose_with_the_linear_predictor(family, formula, priors, kwargs, plain):
    """Whatever builds the linear predictor, the term acts on the likelihood alone: at the same point the model's rows are the
    family's density at the parameters the term gives, from the same model without the term's own linear predictor."""
    m = getattr(ab, family)(formula, DATA, priors, **kwargs)
    p0 = priors if "se(sdx)" not in formula else ab.Priors(**{**{k: getattr(priors, k) for k in ("intercept", "b", "sd", "simo")},
                                                            "sigma": ab.HalfNormal(1)})       # se(col)'s model has no sigma
    m0 = getattr(ab, family)(plain, DATA, p0, **kwargs)
    point = {k: np.asarray(v) + 0.1 for k, v in m.pymc_model.initial_point().items()}
    point0 = {**{k: np.asarray(v) + 0.1 for k, v in m0.pymc_model.initial_point().items()}, **point}
    rows, mu = _rows(m, point)
    _, mu0 = _rows(m0, point0)
    y = m.design.y
    if family == "gaussian":
        se = DATA["sdx"].to_numpy()
        scale = np.sqrt(np.exp(point["sigma_log__"]) ** 2 + se ** 2) if "sigma_log__" in point else se
        np.testing.assert_allclose(mu, mu0, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(rows, stats.norm.logpdf(y, mu0, scale), rtol=1e-12, atol=1e-12)
    else:
        d = DATA["exposure"].to_numpy()
        np.testing.assert_allclose(mu, mu0 * d, rtol=1e-12)              # exp(eta + log d) = exp(eta) d
        if family == "poisson":
            np.testing.assert_allclose(rows, stats.poisson.logpmf(y, mu0 * d), rtol=1e-12, atol=1e-12)
        else:
            n = np.exp(point["alpha_log__"]) * d
            np.testing.assert_allclose(rows, stats.nbinom.logpmf(y, n, n / (n + mu0 * d)), rtol=1e-11, atol=1e-11)


def test_the_helpers_on_each_form(models):
    plain = ab.gaussian("y ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    se = np.array([0.3, 0.4])
    assert sr.row_scale(plain, 0.5, None) == 0.5 and sr.scale_text(plain, "sigma") == "sigma"
    np.testing.assert_array_equal(sr.row_scale(models["meta"], None, se), se)
    np.testing.assert_allclose(sr.row_scale(models["sigma"], 0.4, se), [0.5, np.sqrt(0.32)], rtol=1e-15)
    assert sr.scale_text(models["meta"], None) == "sdx_i" and sr.scale_text(models["sigma"], "sigma") == "sqrt(sigma^2 + sdx_i^2)"
    kw = {"mu": 1.0}
    assert sr.likelihood_parameters(kw, {"weights": se}, None) is kw          # other kinds' data change nothing
    assert sr.denominators(plain, None) is None and sr.standard_errors(plain, None) is None and sr.rate_mean(plain, 2.0, None) == 2.0
    assert sr.newdata_columns(models["poisson"], "linear") == [] and sr.newdata_columns(models["sigma"], "mean") == []
    assert sr.newdata_columns(models["sigma"], "response") == ["sdx"] and sr.newdata_columns(models["poisson"], "mean") == ["exposure"]
    sr.check_additions((AdditionTerm("weights", column="w"), AdditionTerm("cens", column="c")), "f")   # not this module's terms
    sr.check_distributional((AdditionTerm("weights", column="w"),), "sigma ~ x", "f")
    sr.check_priors(ab.parse_formula("y | se(s, sigma = TRUE) ~ x"), GAUSS)    # sigma's prior is the learned sigma's


# -- 4. the family without sigma, the priors and brms's defaults --------------------------------------------------------
def test_the_family_without_sigma():
    gauss, student = get_family("gaussian"), get_family("student")
    spec = sr.se_only_spec(gauss)
    assert spec is sr.se_only_spec(gauss) and spec.name == "gaussian" and spec.link == "identity"     # one cached spec
    assert (spec.scale_parameter, spec.bambi_scale_parameter, spec.scale_missing_error, spec.default_scale_prior) == (None,) * 4
    assert spec.distributional_link is None and spec.addition_terms == gauss.addition_terms and "fixes the gaussian family's sigma at 0" in spec.scale_rule
    family = spec.bambi_family()
    assert family.likelihood.params == ["mu"] and family.likelihood.parent == "mu" and family.name == "gaussian_se"
    assert sr.se_only_spec(student).bambi_family().likelihood.params == ["mu", "nu"]
    assert sr.family_for(ab.parse_formula("y | se(s, sigma = TRUE) ~ x"), gauss) is gauss
    assert sr.family_for(ab.parse_formula("y ~ x"), gauss) is gauss and sr.family_for(ab.parse_formula("y | se(s) ~ x"), gauss) is spec
    assert {n for n, f in ab.families.FAMILIES.items() if "se" in f.addition_terms} == {"gaussian", "student"}
    assert {n for n, f in ab.families.FAMILIES.items() if "rate" in f.addition_terms} == {"poisson", "negbinomial"}
    assert {n for n, f in ab.families.FAMILIES.items() if f.se_parameter} == {"gaussian", "student"}


def test_priors_and_value_points(models):
    with pytest.raises(ab.PriorError, match=r"'se\(sdx\)' without sigma = TRUE fixes sigma at 0.*write se\(sdx, sigma = TRUE\)"):
        ab.gaussian("y | se(sdx) ~ x + (1 | g)", DATA, GAUSS)
    with pytest.raises(ab.PriorError, match=r"needs a prior on the residual SD"):
        ab.gaussian("y | se(sdx, sigma = TRUE) ~ x + (1 | g)", DATA, META)
    with pytest.raises(ValueError, match=r"sigma does not apply: se\(col\) without sigma = TRUE fixes the gaussian family's sigma at 0"):
        _value_point(models["meta"], sigma=1.0)
    with pytest.raises(ValueError, match=r"sigma is required for the Gaussian family"):
        _value_point(models["sigma"])
    assert [r["class"] for r in models["meta"].prior_table()] == ["Intercept", "b", "b", "sd", "sd", "sd"]
    assert "sigma" in [r["class"] for r in models["sigma"].prior_table()]


def test_default_priors():
    meta = ab.default_priors("y | se(sdx) ~ x + (1 | g)", DATA)
    assert meta.sigma is None and meta.intercept == ab.default_priors("y ~ x + (1 | g)", DATA).intercept   # se() leaves y's reading
    assert ab.default_priors("y | se(sdx, sigma = TRUE) ~ x", DATA).sigma == ab.default_priors("y ~ x", DATA).sigma
    rate = ab.default_priors("cnt | rate(exposure) ~ x", DATA, family="poisson")
    assert rate.intercept == ab.default_priors("cnt ~ x", DATA, family="poisson").intercept       # brms reads y, not y / denom
    nb = ab.default_priors("ynb | rate(exposure) ~ x", DATA, family="negbinomial")
    assert repr(nb.shape) == repr(ab.default_priors("ynb ~ x", DATA, family="negbinomial").shape)
    m = ab.gaussian("y | se(sdx) ~ x + (1 | g)", DATA, meta)
    assert [r["class"] for r in m.prior_table()] == ["Intercept", "b", "b", "sd", "sd", "sd"]


# -- 5. predictions, log_lik() and prior-predictive draws ---------------------------------------------------------------
@pytest.fixture(scope="module")
def bundles():
    out = {}
    for cid in ("gauss_se_meta", "student_se_sigma", "poisson_rate", "negbinomial_rate"):
        b = gates.Bundle(cid)
        out[cid] = (b, adapter.draws(b.fixture, b.model, b.cmap, b.gmap))
    return out


def test_se_predictions(bundles):
    b, idata = bundles["student_se_sigma"]
    m, nd = b.model, b.newdata
    for newdata, se in ((None, m.design.se), (nd, nd["sdx"].to_numpy())):
        lin = m.predict(idata, kind="linear", newdata=newdata)["eta"].to_numpy()
        mean = m.predict(idata, kind="mean", newdata=newdata)["mu"].to_numpy()
        np.testing.assert_array_equal(mean, lin)                         # brms's posterior_epred is mu
        resp = m.predict(idata, kind="response", newdata=newdata, seed=5)
        post = idata.posterior
        scale = np.sqrt(post["sigma"].to_numpy()[..., None] ** 2 + se ** 2)
        rng = np.random.default_rng(5)
        want = mean + scale * rng.standard_t(post["nu"].to_numpy()[..., None], size=mean.shape)
        np.testing.assert_allclose(resp["y"].to_numpy(), want, rtol=1e-13, atol=1e-13)
    nd0 = nd.drop(columns=["sdx"])
    assert m.predict(idata, kind="mean", newdata=nd0)["mu"].shape == (1, 8, len(nd))   # new rows need no se for the mean
    with pytest.raises(ab.FormulaError, match=r"newdata lacks the column 'sdx' of 'se\(sdx, sigma = TRUE\)', which kind='response' needs: "
                                              r"each new row's response is drawn with its own standard error"):
        m.predict(idata, kind="response", newdata=nd0)
    with pytest.raises(ab.FormulaError, match=r"'se\(sdx, sigma = TRUE\)' must be non-negative"):
        m.predict(idata, kind="response", newdata=nd.assign(sdx=-nd["sdx"]))
    b, idata = bundles["gauss_se_meta"]
    resp = b.model.predict(idata, kind="response", newdata=b.newdata, seed=3)
    np.testing.assert_array_equal(resp["y"].to_numpy(), np.random.default_rng(3).normal(resp["mu"].to_numpy(), b.newdata["sdx"].to_numpy()))
    with pytest.raises(ab.FormulaError, match=r"must be positive: without sigma = TRUE"):
        b.model.predict(idata, kind="response", newdata=b.newdata.assign(sdx=0.0))


def test_rate_predictions(bundles):
    for cid in ("poisson_rate", "negbinomial_rate"):
        b, idata = bundles[cid]
        m, nd = b.model, b.newdata
        for newdata, d in ((None, m.design.denom), (nd, nd["exposure"].to_numpy())):
            lin = m.predict(idata, kind="linear", newdata=newdata)["eta"].to_numpy()
            mean = m.predict(idata, kind="mean", newdata=newdata)["mu"].to_numpy()
            np.testing.assert_allclose(mean, np.exp(lin) * d, rtol=1e-13)       # brms's posterior_epred, exp(eta) denom
            resp = m.predict(idata, kind="response", newdata=newdata, seed=11)
            rng = np.random.default_rng(11)
            if cid == "poisson_rate":
                want = rng.poisson(mean)
            else:
                n = idata.posterior["alpha"].to_numpy()[..., None] * d
                want = rng.negative_binomial(n, n / (n + mean))
            np.testing.assert_array_equal(resp["y"].to_numpy(), want)
        nd0 = nd.drop(columns=["exposure"])
        assert m.predict(idata, kind="linear", newdata=nd0)["eta"].shape == (1, 8, len(nd))   # the linear predictor needs none
        for kind in ("mean", "response"):
            with pytest.raises(ab.FormulaError, match=rf"newdata lacks the column 'exposure' of 'rate\(exposure\)', which kind='{kind}' "
                                                      r"needs: each new row's expected count is exp\(eta\) times its denominator"):
                m.predict(idata, kind=kind, newdata=nd0)
        with pytest.raises(ab.FormulaError, match=r"the denominators of 'rate\(exposure\)' must be positive"):
            m.predict(idata, kind="mean", newdata=nd.assign(exposure=0.0))


def test_log_lik_and_bayes_r2(bundles):
    b, idata = bundles["gauss_se_meta"]
    mu = b.model.predict(idata, kind="mean")["mu"].to_numpy()
    np.testing.assert_allclose(ab.log_lik(b.model, idata).to_numpy(), stats.norm.logpdf(b.model.design.y, mu, b.model.design.se),
                               rtol=1e-12, atol=1e-12)
    b, idata = bundles["student_se_sigma"]
    post = idata.posterior
    scale = np.sqrt(post["sigma"].to_numpy()[..., None] ** 2 + b.model.design.se ** 2)
    mu = b.model.predict(idata, kind="mean")["mu"].to_numpy()
    np.testing.assert_allclose(ab.log_lik(b.model, idata).to_numpy(),
                               stats.t.logpdf(b.model.design.y, post["nu"].to_numpy()[..., None], mu, scale), rtol=1e-12, atol=1e-12)
    b, idata = bundles["poisson_rate"]
    mean = b.model.predict(idata, kind="mean")["mu"].to_numpy()
    np.testing.assert_allclose(ab.log_lik(b.model, idata).to_numpy(), stats.poisson.logpmf(b.model.design.y, mean), rtol=1e-12, atol=1e-12)
    yhat = mean[0]
    r2 = np.var(yhat, axis=-1, ddof=1) / (np.var(yhat, axis=-1, ddof=1) + np.var(b.model.design.y - yhat, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(b.model, idata).to_numpy()[0], r2, rtol=1e-12)   # the mean with its denominator
    b, idata = bundles["negbinomial_rate"]
    mean = b.model.predict(idata, kind="mean")["mu"].to_numpy()
    n = idata.posterior["alpha"].to_numpy()[..., None] * b.model.design.denom
    np.testing.assert_allclose(ab.log_lik(b.model, idata).to_numpy(), stats.nbinom.logpmf(b.model.design.y, n, n / (n + mean)),
                               rtol=1e-11, atol=1e-11)


def test_post_processing_contexts(bundles):
    """conditional_effects() refuses every addition term (no frozen case checks how brms holds one on its grid, D99);
    prior_draws() has the priors the model has: none of sigma under se(col)."""
    for cid, term in (("gauss_se_meta", "se\\(sdx\\)"), ("poisson_rate", "rate\\(exposure\\)")):
        b, idata = bundles[cid]
        with pytest.raises(ValueError, match=rf"conditional_effects\(\) of a model with the addition term\(s\) {term} is not offered"):
            ab.conditional_effects(b.model, idata, "x")
    b, _ = bundles["gauss_se_meta"]
    assert list(ab.prior_draws(b.model, draws=10, seed=1).columns) == ["Intercept", "sd_g"]


def test_prior_predictive_draws_with_each_row_s_scale(models):
    m = models["meta"]
    pp = m.prior_predictive(draws=4000, seed=4)
    resid = pp.prior_predictive["y"].to_numpy()[0] - pp.prior["mu"].to_numpy()[0]
    np.testing.assert_allclose(resid.std(axis=0), m.design.se, rtol=0.06)     # Normal(mu, se_i): no sigma of 1 by default
    nb = ab.negbinomial("ynb | rate(exposure) ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 0.5), b=ab.Normal(0, 0.5), shape=ab.Gamma(20, 2)))
    pp = nb.prior_predictive(draws=4000, seed=1)
    y, mu = pp.prior_predictive["ynb"].to_numpy()[0], pp.prior["mu"].to_numpy()[0]
    shape = pp.prior["alpha"].to_numpy()[0][:, None] * nb.design.denom          # each row's shape * denom
    np.testing.assert_allclose(y.mean(axis=0) / mu.mean(axis=0), 1.0, atol=0.08)          # mean exp(eta) * denom
    np.testing.assert_allclose(((y - mu) ** 2).mean(axis=0) / (mu + mu ** 2 / shape).mean(axis=0), 1.0, atol=0.15)


# -- 6. describe(), identity, the fit record ----------------------------------------------------------------------------------
def test_describe(models):
    text = models["meta"].describe()
    for s in ["AutoBayes M0 model with response addition term(s) se(sdx)",
              "  se        : se(sdx): each row's known standard error sdx_i, 0.2011 to 0.5877 (brms se); row i's scale is sdx_i itself, "
              "and the model has no sigma (brms fixes it at 0)",
              "  y_i ~ Normal(mu_i, sdx_i)",
              "  se(): row i's scale is its known standard error sdx_i; the model has no sigma (brms's Stan data sigma = 0)"]:
        assert s in text, s
    assert "\n  sigma ~" not in text and "\n  sigma_log__" not in text
    text = models["student_sigma"].describe()
    for s in ["row i's scale is sqrt(sigma^2 + sdx_i^2)", "  sigma ~ HalfNormal(sigma=1)",
              "  y_i ~ StudentT(nu, mu_i, sqrt(sigma^2 + sdx_i^2)); E[y_i] = mu_i",
              "  se(): row i's scale is sigma and its known standard error sdx_i in quadrature, sqrt(sigma^2 + sdx_i^2)"]:
        assert s in text, s
    text = models["student_meta"].describe()
    assert "  nu = 4 (fixed, not sampled)" in text and "  y_i ~ StudentT(nu, mu_i, sdx_i); E[y_i] = mu_i" in text and "sigma ~" not in text
    text = models["poisson"].describe()
    for s in ["  rate      : rate(exposure): each row's denominator exposure_i, ", "(the graph's mu holds it); predict(kind='linear')",
              "  E[y_i] = exp(eta_i) * exposure_i; eta is the log expected count per unit of the denominator (brms's rate()).",
              "  y_i ~ Poisson(mu_i * exposure_i); mu_i = exp(eta_i), exposure_i the row's denominator (brms's rate())",
              "  No overdispersion or zero-inflation is modeled.",
              "  rate(): the log likelihood is the family's at the mean exp(eta_i) * exposure_i (brms's mu + log_denom)\n"]:
        assert s in text, s
    text = models["negbinomial"].describe()
    for s in ["on the log scale (the graph's mu holds it), and the shape is shape * exposure_i;",
              "  y_i ~ NegativeBinomial(mu_i * exposure_i, shape * exposure_i); mu_i = exp(eta_i)", "  No zero-inflation is modeled.",
              "(brms's mu + log_denom) and the shape shape * exposure_i"]:
        assert s in text, s
    plain = ab.poisson("cnt ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1))).describe()
    assert "rate" not in plain and "  No exposure/offset, overdispersion or zero-inflation is modeled." in plain
    plain = ab.negbinomial("ynb ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(0.01, 0.01))).describe()
    assert "rate" not in plain and "  E[y_i] = mu_i = exp(eta_i); eta is log expected count;" in plain


def test_identity_and_record(models, bundles):
    m = models["meta"]
    same = ab.gaussian("y | se(sdx) ~ x + (1 | g)", DATA.copy(), META)
    other = ab.gaussian("y | se(sdx) ~ x + (1 | g)", DATA.assign(sdx=DATA["sdx"] * 2), META)
    assert same.target_identity == m.target_identity
    assert len({m.target_identity["sha256"], other.target_identity["sha256"], models["sigma"].target_identity["sha256"]}) == 3
    r = models["poisson"]
    other = ab.poisson("cnt | rate(exposure) ~ x + (1 | g)", DATA.assign(exposure=DATA["exposure"] + 1), COUNTS)
    plain = ab.poisson("cnt ~ x + (1 | g)", DATA, COUNTS)
    assert len({r.target_identity["sha256"], other.target_identity["sha256"], plain.target_identity["sha256"]}) == 3
    b, idata = bundles["gauss_se_meta"]
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 8, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(b.model.target_identity))
    rec = ab.fit_record(b.model, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert rec["model"]["addition_terms"] == ["se(sdx)"] and "sigma" not in rec["model"]["priors"]
    assert "sigma" not in rec["posterior"] and "Intercept" in rec["posterior"]


# -- 7. the oracle discriminates ------------------------------------------------------------------------------------------
def _bundle(case_id, formula, priors=None):
    b = gates.Bundle(case_id)
    case = copy.deepcopy(b.case)
    case["autobayes"]["formula"] = formula
    if priors is not None:
        case["autobayes"]["priors"] = priors
    return gates.Bundle(case_id, model=adapter.build_model(case, b.data))


def test_gate_g1_discriminates():
    """G1 fails when the model drops the term, reads another column, or learns the sigma brms fixes at 0."""
    sigma = {"intercept": ["Normal", 0, 5], "sd": ["HalfNormal", 1], "sigma": ["HalfNormal", 1]}
    with pytest.raises(OracleMismatch, match="brms has se but AutoBayes does not"):
        gates.check_design(_bundle("gauss_se_meta", "y ~ 1 + (1 | g)", sigma))
    with pytest.raises(OracleMismatch, match="se differ by"):
        gates.check_design(_bundle("gauss_se_meta", "y | se(xpos) ~ 1 + (1 | g)"))
    with pytest.raises(OracleMismatch, match=r"brms's Stan data holds sigma = 0.0 \(se\(\) without sigma = TRUE\); the AutoBayes model "
                                             r"learns sigma"):
        gates.check_design(_bundle("gauss_se_meta", "y | se(sdx, sigma = TRUE) ~ 1 + (1 | g)", sigma))
    b = gates.Bundle("gauss_se_sigma")
    case = copy.deepcopy(b.case)
    case["autobayes"]["formula"] = "y | se(sdx) ~ x + (1 | g)"
    case["autobayes"]["priors"] = {k: v for k, v in case["autobayes"]["priors"].items() if k != "sigma"}
    with pytest.raises(OracleMismatch, match="brms reports sigma = .*; the AutoBayes model has no sigma"):
        gates.Bundle("gauss_se_sigma", model=adapter.build_model(case, b.data)).points().__next__()
    with pytest.raises(OracleMismatch, match="fixes sigma at 0 under se\\(\\) without sigma = TRUE; brms's Stan data has no sigma"):
        gates.check_design(gates.Bundle("gauss_se_sigma", model=adapter.build_model(case, b.data)))
    with pytest.raises(OracleMismatch, match="brms has denom but AutoBayes does not"):
        gates.check_design(_bundle("poisson_rate", "cnt ~ x + (1 | g)"))
    with pytest.raises(OracleMismatch, match="denom differ by"):
        gates.check_design(_bundle("poisson_rate", "cnt | rate(xpos) ~ x + (1 | g)"))
    with pytest.raises(OracleMismatch, match="log density"):
        gates.check_logp(_bundle("negbinomial_rate", "ynb | rate(xpos) ~ x + (1 | g)"))
