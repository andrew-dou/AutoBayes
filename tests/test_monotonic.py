"""Monotonic effects (brms-parity B03): brms's ``mo(x)`` of an ordinal predictor. The term model and the coding of the
predictor, the refusals, Stan's stick-breaking simplex transform, the graph against a density written by hand, the build-time
check of the term's contribution, priors, ``value_point``, by-hand predictions, ``describe()``, names, the target identity,
fit records, prior-predictive draws, brms's default priors, and the oracle's extension to monotonic terms. The codes and
default-prior tables below were printed by brms 2.23.0 (R 4.5.2, C collation) for ``FRAME``, with ``lik`` an R ordered
factor. Nothing here samples; the brms comparison is the oracle cases ``gauss_monotonic``, ``gauss_monotonic_two`` and one
``<family>_monotonic`` case per further family."""
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
from autobayes.families import FAMILIES
from autobayes.formula import Monotonic, PopTerm, monotonic_coding, parse_formula, pop_term
from autobayes.monotonic import StickBreakingTransform, check_simplex, stick_breaking_constrain, stick_breaking_free

LIK = ["never", "rarely", "often", "always"]                         # declared order; 'often' has no row (brms drops it)
FRAME = pd.DataFrame({
    "y": [0.31, -1.2, 2.05, 0.77, -0.4, 1.66, 0.12, -0.95, 2.41, 1.08, 0.5, -0.33],
    "x": [0.2, -1.1, 0.7, 1.5, -0.3, 0.05, -2.0, 0.8, 1.2, -0.6, 0.35, -0.1],
    "ord": [1, 2, 4, 2, 1, 4, 2, 1, 4, 4, 2, 1],                      # 3 has no row (brms keeps it: Jmo = 3)
    "neg": [-2, 0, -1, 1, -2, 0, 1, -1, 0, 1, -2, -1],               # coded from its minimum, -2
    "lik": pd.Categorical(["never", "rarely", "always", "rarely", "never", "always", "rarely", "never", "always", "always",
                           "rarely", "never"], categories=LIK, ordered=True),
    "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
    "cnt": [0, 3, 1, 1, 7, 2, 0, 0, 12, 4, 1, 0],
    "w": [0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0],
})
# brms 2.23.0's standata(y ~ mo(ord) + mo(neg) + mo(lik), FRAME): Ksp = Imo = 3, Jmo = (3, 3, 2), con_simo_k all ones
BRMS_XMO = {"ord": [0, 1, 3, 1, 0, 3, 1, 0, 3, 3, 1, 0], "neg": [0, 2, 1, 3, 0, 2, 3, 1, 2, 3, 0, 1],
            "lik": [0, 1, 2, 1, 0, 2, 1, 0, 2, 2, 1, 0]}
BRMS_JMO = {"ord": 3, "neg": 3, "lik": 2}
SIMPLEX = np.array([0.2, 0.5, 0.3])


def priors(**extra):
    base = dict(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), simo=ab.Dirichlet(2, 2, 2))
    return ab.Priors(**{**base, **extra})


def gaussian(formula="y ~ x + mo(ord)", frame=None, **extra):
    return ab.gaussian(formula, FRAME if frame is None else frame, priors(**extra))


def _mo(codes, simplex):
    """brms's mo(simo, i) = D * sum(simo[1:i]) per row, written out."""
    return np.array([len(simplex) * float(np.sum(simplex[:c])) for c in codes])


def _value(expression):
    """A PyTensor expression's value (compiled with no inputs)."""
    return np.asarray(pytensor.function([], expression)())


# -- the term model and the coding ------------------------------------------------------------------------------------
def test_the_term_model():
    s = parse_formula("y ~ x + mo(ord) + (1 | g)", FRAME)
    kinds = [(t.name, t.kind, t.columns) for t in s.pop_terms]
    assert kinds == [("x", "numeric", ("x",)), ("mo(ord)", "monotonic", ())]
    assert s.terms == ("x", "mo(ord)") and s.columns == ("x",) and s.coefficient_labels == ("x", "mo(ord)")
    assert s.column_terms == (("x", "x", None),) and s.predictors == ("x",) and s.pop_columns == ("x", "ord")
    assert [t.name for t in s.mo_terms] == ["mo(ord)"] and [t.name for t in s.bambi_terms] == ["x"]
    assert s.mo_terms[0].is_monotonic and not s.mo_terms[0].is_vector and s.mo_terms[0].operands == ("ord",)
    assert s.canonical_formula == "y ~ x + mo(ord) + (1 | g)" and s.bambi_formula == "y ~ x + (1 | g)"
    assert s.monotonics == (Monotonic("ord", minimum=1, maximum=4),) and s.monotonic("ord").D == 3
    unseen = parse_formula("y ~ x + mo(ord) + (1 | g)")                 # the coding is known once the data are seen
    assert unseen.monotonics == (Monotonic("ord"),) and not unseen.monotonic("ord").known
    with pytest.raises(ValueError, match="known once the data are seen"):
        unseen.monotonic("ord").D
    assert parse_formula("y ~ mo( ord ) + x", FRAME).terms == ("mo(ord)", "x")       # the canonical spelling
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):                   # asdict hands the codings back as mappings
        assert replica == s and hash(replica) == hash(s)
    assert pop_term("mo(ord)", "monotonic") == PopTerm("mo(ord)", "monotonic", ())
    with pytest.raises(ValueError, match=r"a monotonic term is spelled 'mo\(col\)' and has no design column"):
        PopTerm("mo(ord)", "monotonic", ("mo(ord)",))
    with pytest.raises(ValueError, match="monotonics must give the coding of exactly"):
        ab.ModelSpec(formula="", response="y", pop_terms=(pop_term("mo(ord)", "monotonic"),), monotonics=())


def test_brms_codes_the_predictor():
    """The codes and number of steps brms printed for FRAME: whole numbers less their minimum (a value with no row stays
    a value), an ordered factor by its levels after brms drops the unused ones."""
    m = ab.gaussian("y ~ mo(ord) + mo(neg) + mo(lik)", FRAME, priors(simo={"ord": ab.Dirichlet(1), "neg": ab.Dirichlet(1),
                                                                            "lik": ab.Dirichlet(1)}))
    for column in ("ord", "neg", "lik"):
        block = m.design.monotonic[column]
        assert block.codes.tolist() == BRMS_XMO[column] and block.D == BRMS_JMO[column] and block.term == f"mo({column})"
    assert m.spec.monotonic("neg") == Monotonic("neg", minimum=-2, maximum=1)
    assert m.spec.monotonic("lik") == Monotonic("lik", categories=("never", "rarely", "always"))
    assert m.spec.monotonic("lik").step_labels() == ("never -> rarely", "rarely -> always")
    assert any("values ['3'] of the monotonic predictor 'ord' have no fitted rows" in w for w in m.warnings)
    assert any("declared categories ['often'] of the monotonic predictor 'lik' do not occur in the fitted rows" in w
               for w in m.warnings)
    assert monotonic_coding("ord", FRAME.ord.astype(float)) == Monotonic("ord", minimum=1, maximum=4)   # any numeric dtype


CODING_REFUSALS = [
    (FRAME.x, r"has the value 0\.2; a monotonic predictor holds whole numbers.*Monotonic predictors must be integers"),
    (FRAME.g, r"has dtype (object|str); a monotonic predictor holds whole numbers or an ordered pandas Categorical"),
    (pd.Series(pd.Categorical(FRAME.g)), r"is an unordered pandas Categorical"),
    (pd.Series([True, False] * 6), r"is boolean"),
    (pd.Series([2] * 12), r"takes the values \[2\] in the fitted rows; a monotonic effect needs at least three values"),
    (pd.Series([0, 1] * 6), r"with two values a monotonic effect has a single step and no simplex to estimate.*D42"),
    (pd.Series([1.0, np.nan] * 6), r"has missing values"),
    (pd.Series([0, 2_000_001] * 6), r"spans 0 to 2000001, 2000001 steps"),
]


@pytest.mark.parametrize("column, message", CODING_REFUSALS)
def test_coding_refusals(column, message):
    with pytest.raises(ab.FormulaError, match=message):
        monotonic_coding("v", column)


def test_new_rows_are_coded_as_the_fitted_rows():
    m = monotonic_coding("ord", FRAME.ord)
    assert m.codes(pd.Series([4, 1, 3])).tolist() == [3, 0, 2]          # 3 is a value although no fitted row takes it
    for bad, message in ((pd.Series([5]), r"has the value 5\.0, which is not a whole number from 1 to 4"),
                         (pd.Series([0]), r"has the value 0\.0"), (pd.Series([2.5]), r"has the value 2\.5"),
                         (pd.Series(["2"]), r"has dtype (object|str); it was fitted as whole numbers 1\.\.4")):
        with pytest.raises(ab.FormulaError, match=message):
            m.codes(bad, where="newdata column")
    lik = monotonic_coding("lik", FRAME.lik)
    assert lik.codes(pd.Series(["always", "never"])).tolist() == [2, 0]  # labels, a Categorical or not
    with pytest.raises(ab.FormulaError, match=r"has values \['often'\] that are not among its ordered categories"):
        lik.codes(pd.Series(["often"]), where="newdata column")


FORMULA_REFUSALS = [
    ("y ~ x + mo(ord) + x:mo(ord)", r"an interaction with a monotonic term \('mo\(ord\)'\) is not supported.*D40"),
    ("y ~ mo(ord) * x", r"an interaction with a monotonic term.*D40"),
    ("y ~ mo(ord):mo(neg)", r"an interaction with a monotonic term.*D40"),
    ("y ~ mo(ord) + (1 + mo(ord) | g)", r"a monotonic effect cannot vary by 'g' in this slice.*D41"),
    ("y ~ mo(ord, id = 'a')", r"mo\(\) takes one column here; brms's id argument.*D42"),
    ("y ~ mo(log(ord))", r"mo\(\) takes a column name here, not an expression \('log\(ord\)'.*D42"),
    ("y ~ mo(ord + 1)", r"not an expression"),
    ("y ~ mo(y)", r"the response 'y' cannot be a monotonic predictor"),
    ("y ~ mo(ord) + mo( ord )", r"'mo\(ord\)': duplicated predictor \('mo\( ord \)' spells 'mo\(ord\)' again\)"),
    ("y ~ mo(g) + (1 | g)", r"'g': the grouping factor cannot also be a predictor"),
]


@pytest.mark.parametrize("formula, message", FORMULA_REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula, FRAME)


def test_other_refusals_name_the_construct():
    with pytest.raises(ab.FormulaError, match=r"the trials column 'cnt' cannot be a monotonic predictor"):
        parse_formula("ord | trials(cnt) ~ mo(cnt)", None, "binomial")
    with pytest.raises(ab.FormulaError, match=r"'mo\(ord\)': 'ord' is declared categorical; mo\(\) reads it as an ordinal predictor"):
        parse_formula("y ~ ord + mo(ord)", FRAME, categorical={"ord": 1})
    with pytest.raises(ab.FormulaError, match=r"a monotonic term in a distributional formula \(mo\(ord\)\) is not supported.*D42"):
        parse_formula(["y ~ x", "sigma ~ mo(ord)"], FRAME)
    with pytest.raises(ab.FormulaError, match=r"the monotonic term 'mo\(ord\)' is not available for the categorical family; "
                                              r"brms's mo\(\) is verified here for the gaussian, student.*D42"):
        parse_formula("lik ~ mo(ord)", None, "categorical")


def test_the_families_that_take_monotonic_terms():
    # C05: refused for mixtures; D01: refused for the five continuous families, which no frozen mo() case verifies (D79); D02:
    # likewise for the beta-binomial, geometric and negbinomial2 families (D81)
    assert {name for name, spec in FAMILIES.items() if spec.monotonic} == set(FAMILIES) - {
        "categorical", "mixture", "weibull", "exponential", "skew_normal", "shifted_lognormal", "exgaussian",
        "beta_binomial", "geometric", "negbinomial2"}
    probit = ab.bernoulli("cnt ~ mo(ord)", FRAME.assign(cnt=[0, 1] * 6), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1),
                          simo=ab.Dirichlet(1)), link="probit")           # the probit link takes it as the logit does
    assert set(probit.design.monotonic) == {"ord"}


# -- Stan's simplex transform -----------------------------------------------------------------------------------------
def test_the_stick_breaking_transform_is_stans():
    from tests.oracle import fixture as fx

    for y in (np.zeros(3), np.array([-0.7, 1.3, 0.2]), np.array([3.0, -4.0, 5.0])):
        x, lp = stick_breaking_constrain(y, 4)
        assert x.shape == (4,) and abs(x.sum() - 1.0) < 1e-15 and (x > 0).all()
        np.testing.assert_allclose(stick_breaking_free(x), y, rtol=1e-12, atol=1e-12)
        # Stan's Jacobian: the log determinant of d(x_1..x_{D-1}) / dy, by central differences
        h, J = 1e-6, np.empty((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            J[:, j] = (stick_breaking_constrain(y + e, 4)[0][:3] - stick_breaking_constrain(y - e, 4)[0][:3]) / (2 * h)
        assert lp == pytest.approx(np.linalg.slogdet(J)[1], abs=1e-6)
    np.testing.assert_allclose(stick_breaking_constrain(np.zeros(3), 4)[0], np.full(4, 0.25), rtol=1e-15)   # Stan's shift
    f = fx.Fixture.load("gauss_monotonic")                                  # brms's own coordinates of its simplex
    for p in f.points:
        x, _ = stick_breaking_constrain(p.u[1:3], 3)
        np.testing.assert_allclose(x, p.pars["simo_1"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(stick_breaking_free(p.pars["simo_1"]), p.u[1:3], rtol=0, atol=1e-15)
    t = StickBreakingTransform(4)                                           # the PyTensor map is the NumPy one
    y = np.array([-0.7, 1.3, 0.2])
    x, lp = stick_breaking_constrain(y, 4)
    np.testing.assert_allclose(_value(t.backward(pytensor.shared(y))), x, rtol=1e-15)
    np.testing.assert_allclose(_value(t.forward(pytensor.shared(x))), y, rtol=1e-14)
    assert float(_value(t.log_jac_det(pytensor.shared(y)))) == pytest.approx(lp, rel=1e-14)
    from pymc.distributions.transforms import simplex                       # PyMC's own transform is another map
    assert not np.allclose(_value(simplex.forward(pytensor.shared(x))), y)
    with pytest.raises(ValueError, match="at least two entries"):
        StickBreakingTransform(1)
    for bad, message in (([0.5, 0.5], "a simplex of 3 finite entries"), ([0.5, 0.5, 0.0], "strictly positive"),
                         ([0.5, 0.5, 0.1], "must sum to 1")):
        with pytest.raises(ValueError, match=message):
            check_simplex(bad, 3, "simo")


# -- the graph --------------------------------------------------------------------------------------------------------
def test_the_density_is_the_one_written_by_hand():
    m = gaussian()
    a, bx, bsp, sigma = 0.3, 0.7, -0.45, 1.3
    point = m.value_point(intercept=a, b=[bx], sigma=sigma, bsp=bsp, simo=SIMPLEX)
    assert set(point) == {"Intercept", "x", "sigma_log__", "mo(ord)", "simo_ord_stickbreaking__"}
    np.testing.assert_allclose(point["simo_ord_stickbreaking__"], stick_breaking_free(SIMPLEX), rtol=1e-15)
    x = FRAME.x.to_numpy()
    mu = a + (x - x.mean()) * bx + bsp * _mo(BRMS_XMO["ord"], SIMPLEX)
    nojac = (stats.norm.logpdf(FRAME.y, mu, sigma).sum() + stats.norm.logpdf(a, 0, 5) + stats.norm.logpdf(bx, 0, 2)
             + stats.norm.logpdf(bsp, 0, 2) + stats.halfnorm.logpdf(sigma) + stats.dirichlet.logpdf(SIMPLEX, [2, 2, 2]))
    jac = nojac + np.log(sigma) + stick_breaking_constrain(point["simo_ord_stickbreaking__"], 3)[1]
    assert float(m.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(nojac, rel=1e-12)
    assert float(m.pymc_model.compile_logp()(point)) == pytest.approx(jac, rel=1e-12)
    fn = m.pymc_model.compile_fn(m.pymc_model.replace_rvs_by_values([m.pymc_model["mu"]]), inputs=m.pymc_model.value_vars,
                                 point_fn=True, on_unused_input="ignore")
    np.testing.assert_allclose(np.asarray(fn(point)[0]), mu, rtol=1e-13)
    # every coordinate of the gradient, against central differences of the density
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

    jf = get_jaxified_logp(m.pymc_model)
    assert float(jf([point[n] for n in names])) == pytest.approx(jac, rel=1e-12)


def test_the_term_composes_with_other_constructs():
    """With weights() (B02) each row's log density is weighted; with a distributional formula (B01) and with correlated
    slopes (A02, whose post-build edit rebuilds the graph) the term is built and checked against the data as everywhere."""
    w = gaussian("y | weights(w) ~ x + mo(ord)")
    a, bx, bsp, sigma = 0.3, 0.7, -0.45, 1.3
    point = w.value_point(intercept=a, b=[bx], sigma=sigma, bsp=bsp, simo=SIMPLEX)
    x = FRAME.x.to_numpy()
    rows = stats.norm.logpdf(FRAME.y, a + (x - x.mean()) * bx + bsp * _mo(BRMS_XMO["ord"], SIMPLEX), sigma)
    want = ((FRAME.w.to_numpy() * rows).sum() + stats.norm.logpdf(a, 0, 5) + stats.norm.logpdf(bx, 0, 2)
            + stats.norm.logpdf(bsp, 0, 2) + stats.halfnorm.logpdf(sigma) + stats.dirichlet.logpdf(SIMPLEX, [2, 2, 2]))
    assert float(w.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(want, rel=1e-12)
    d = ab.gaussian(["y ~ mo(ord)", "sigma ~ x"], FRAME, priors(sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}))
    assert set(d.design.monotonic) == {"ord"} and "sigma_x" in d.pymc_model.named_vars
    c = ab.gaussian("y ~ x + mo(ord) + (1 + x | g)", FRAME, priors(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                                                                   cor=ab.LKJ(2)))
    assert {"simo_ord_stickbreaking__", "mo(ord)", "L|g_cholesky_corr__"} <= {v.name for v in c.pymc_model.value_vars}


def test_without_an_intercept_or_another_term():
    """``y ~ 0 + mo(ord)``: Bambi is handed ``y ~ 0`` and builds no population-level design at all."""
    m = ab.gaussian("y ~ 0 + mo(ord)", FRAME, ab.Priors(b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), simo=ab.Dirichlet(2, 2, 2)))
    assert m.spec.bambi_formula == "y ~ 0" and m.design.X_raw.shape == (12, 0) and not m.spec.intercept
    point = m.value_point(b=[], sigma=1.1, bsp=0.8, simo=SIMPLEX)
    want = (stats.norm.logpdf(FRAME.y, 0.8 * _mo(BRMS_XMO["ord"], SIMPLEX), 1.1).sum() + stats.norm.logpdf(0.8, 0, 2)
            + stats.halfnorm.logpdf(1.1) + stats.dirichlet.logpdf(SIMPLEX, [2, 2, 2]))
    assert float(m.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(want, rel=1e-12)


def test_the_build_checks_the_terms_contribution(monkeypatch):
    import autobayes.monotonic as monotonic

    honest = monotonic.monotonic_effect
    monkeypatch.setattr(monotonic, "monotonic_effect", lambda simplex, codes, D: honest(simplex, np.minimum(codes, 1), D))
    with pytest.raises(RuntimeError, match=r"the graph's 'mu' does not move by D \* sum\(simplex\[:Xmo\]\) per row"):
        gaussian()
    monkeypatch.setattr(monotonic, "monotonic_effect", lambda simplex, codes, D: 2.0 * honest(simplex, codes, D))
    with pytest.raises(RuntimeError, match=r"does not move by D \* sum\(simplex\[:Xmo\]\)"):      # under Bambi's log link too
        ab.poisson("cnt ~ mo(ord)", FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), simo=ab.Dirichlet(1)))


# -- priors -----------------------------------------------------------------------------------------------------------
def test_the_dirichlet_prior():
    assert ab.Dirichlet(2, 2, 2) == ab.Dirichlet([2, 2, 2]) == ab.Dirichlet(np.array([2.0, 2.0, 2.0]))
    assert str(ab.Dirichlet(2, 1, 3)) == "Dirichlet(a[1]=2, a[2]=1, a[3]=3)" and str(ab.Dirichlet(1)) == "Dirichlet(a=1)"
    assert ab.Dirichlet([1.5]) == ab.Dirichlet(1.5)                         # one number: brms's shorthand
    from autobayes.priors import brms_prior, dirichlet_concentration

    assert brms_prior(ab.Dirichlet(2, 1, 3)) == "dirichlet(2, 1, 3)" and brms_prior(ab.Dirichlet(1)) == "dirichlet(1)"
    np.testing.assert_array_equal(dirichlet_concentration(ab.Dirichlet(1.5), 4), [1.5] * 4)
    for bad in ((), (1, 0), (-1,), (np.inf,)):
        with pytest.raises(ab.PriorError):
            ab.Dirichlet(*bad)
    with pytest.raises(ab.PriorError, match="Invalid Dirichlet prior. Expected input of length 4"):
        dirichlet_concentration(ab.Dirichlet(1, 2), 4)


PRIOR_REFUSALS = [
    (dict(simo=None), r"the monotonic term\(s\) \['mo\(ord\)'\] need a prior on their simplex: pass simo=\{'ord': ab.Dirichlet"),
    (dict(simo={"mo(ord)": ab.Dirichlet(1)}), r"simo must cover exactly the columns of the monotonic terms \['ord'\]; got "
                                              r"\['mo\(ord\)'\]; key the prior by the column, simo=\{'ord': \.\.\.\}"),
    (dict(simo={"ord": ab.Normal(0, 1)}), r"prior for 'simo:ord' must be a prior on a simplex \(\['Dirichlet'\]"),
    (dict(simo={"ord": "dirichlet(1)"}), r"prior for 'simo:ord' must be a Prior, got str"),
    (dict(simo=ab.Dirichlet(1, 2)), r"prior for 'simo:ord' \(mo\(ord\), 3 steps from 1 to 4\): .*Expected input of length 3"),
    (dict(b={"x": ab.Normal(0, 1), "ord": ab.Normal(0, 1)}), r"missing \['mo\(ord\)'\], unexpected \['ord'\]; the monotonic "
                                                             r"term's coefficient is keyed by the term, b=\{'mo\(ord\)': prior\}"),
    (dict(b={"x": ab.Normal(0, 1), "moord": ab.Normal(0, 1)}), r"keyed by the term"),
    (dict(b={"x": ab.Normal(0, 1), "mo(ord)": ab.HalfNormal(1)}), r"prior for 'mo\(ord\)' must be a location prior"),
]


@pytest.mark.parametrize("extra, message", PRIOR_REFUSALS)
def test_prior_refusals(extra, message):
    with pytest.raises(ab.PriorError, match=message):
        gaussian(**extra)


def test_prior_forms():
    by_term = gaussian(b={"x": ab.Normal(0, 1), "mo(ord)": ab.StudentT(3, 0, 1)}, simo={"ord": ab.Dirichlet(1)})
    assert by_term.priors.coefficient_priors(by_term.spec) == {"x": ab.Normal(0, 1), "mo(ord)": ab.StudentT(3, 0, 1)}
    two = ab.gaussian("y ~ mo(ord) + mo(neg)", FRAME, priors(simo={"ord": ab.Dirichlet(1), "neg": ab.Dirichlet(2, 1, 1)}))
    assert two.priors.simo_priors(two.spec) == {"ord": ab.Dirichlet(1), "neg": ab.Dirichlet(2, 1, 1)}
    with pytest.raises(ab.PriorError, match=r"several monotonic terms \['mo\(ord\)', 'mo\(neg\)'\] need an exact-key simo mapping"):
        ab.gaussian("y ~ mo(ord) + mo(neg)", FRAME, priors())
    with pytest.raises(ab.PriorError, match=r"the formula has no monotonic term mo\(x\); do not pass simo"):
        ab.gaussian("y ~ x", FRAME, priors())
    flat = gaussian(b={"x": ab.Normal(0, 1), "mo(ord)": ab.Flat()})
    with pytest.raises(ValueError, match=r"the prior of \['b:mo\(ord\)'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)


def test_value_point_refusals():
    m = gaussian()
    base = dict(intercept=0.1, b=[0.2], sigma=1.0)
    with pytest.raises(ValueError, match=r"bsp \(the coefficient\) and simo \(the simplex\) are required for the monotonic terms"):
        m.value_point(**base)
    with pytest.raises(ValueError, match=r"simo for 'mo\(ord\)' must be a simplex of 3 finite entries"):
        m.value_point(**base, bsp=0.3, simo=[0.5, 0.5])
    with pytest.raises(ValueError, match="must sum to 1"):
        m.value_point(**base, bsp=0.3, simo=[0.5, 0.4, 0.2])
    with pytest.raises(ValueError, match=r"bsp must cover exactly the columns of the monotonic terms \['ord'\]; got \['mo\(ord\)'\]"):
        m.value_point(**base, bsp={"mo(ord)": 0.3}, simo=SIMPLEX)
    with pytest.raises(ValueError, match=r"bsp for 'mo\(ord\)' must be a finite scalar"):
        m.value_point(**base, bsp=[0.3], simo=SIMPLEX)
    with pytest.raises(ValueError, match="b must have 1 entries, one per design column"):  # the term has no entry in b
        m.value_point(intercept=0.1, b=[0.2, 0.3], sigma=1.0, bsp=0.3, simo=SIMPLEX)
    assert m.value_point(**base, bsp={"ord": 0.3}, simo={"ord": SIMPLEX}).keys() == m.value_point(**base, bsp=0.3, simo=SIMPLEX).keys()
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"this model has no monotonic term mo\(x\); bsp and simo do not apply"):
        plain.value_point(**base, bsp=0.3)


# -- predictions ------------------------------------------------------------------------------------------------------
def draws(model, a=0.3, bx=0.7, bsp=-0.45, simplex=SIMPLEX, u=(0.2, -0.1, 0.05, 0.3)):
    """Two chains of three constant draws: the raw-scale intercept (as a posterior from fit() holds it), the coefficients,
    the monotonic term's coefficient and simplex and, with a group term, the varying intercepts."""
    post = {"Intercept": np.full((2, 3), a), "x": np.full((2, 3), bx), "mo(ord)": np.full((2, 3), bsp),
            "simo_ord": np.broadcast_to(simplex, (2, 3, 3)), "sigma": np.ones((2, 3))}
    coords, dims = {"simo_ord_dim": ["1", "2", "3"]}, {"simo_ord": ["simo_ord_dim"]}
    if model.spec.groups:
        post.update({"1|g": np.broadcast_to(np.array(u), (2, 3, 4)), "1|g_sigma": np.ones((2, 3))})
        coords["g__factor_dim"], dims["1|g"] = ["a", "b", "c", "d"], ["g__factor_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_add_the_monotonic_term():
    m = gaussian("y ~ x + mo(ord) + (1 | g)", sd=ab.HalfNormal(1))
    post = draws(m)
    u = dict(zip("abcd", (0.2, -0.1, 0.05, 0.3)))
    fixed = 0.3 + 0.7 * FRAME.x.to_numpy() - 0.45 * _mo(BRMS_XMO["ord"], SIMPLEX)
    full = fixed + np.array([u[g] for g in FRAME.g])
    for kind, var in (("linear", "eta"), ("mean", "mu")):
        np.testing.assert_allclose(m.predict(post, kind=kind)[var], np.broadcast_to(full, (2, 3, 12)), rtol=1e-14)
        np.testing.assert_allclose(m.predict(post, kind=kind, include_group_effects=False)[var], np.broadcast_to(fixed, (2, 3, 12)),
                                   rtol=1e-14)
    new = pd.DataFrame({"x": [0.5, -1.0, 0.0], "ord": [3, 1, 4], "g": ["b", "a", "d"]})
    want = 0.3 + 0.7 * new.x.to_numpy() - 0.45 * np.array([3 * 0.7, 0.0, 3.0]) + np.array([-0.1, 0.2, 0.3])
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="mean").mu[0, 0], want, rtol=1e-14)
    for bad, message in ((new.assign(ord=[3, 1, 5]), r"newdata column 'ord' of the monotonic term 'mo\(ord\)' has the value 5\.0"),
                         (new.assign(ord=[3, 1.5, 4]), r"has the value 1\.5"), (new.drop(columns="ord"), r"newdata lacks columns \['ord'\]")):
        with pytest.raises(ab.FormulaError, match=message):
            m.predict(post, newdata=bad, kind="mean")
    lik = ab.gaussian("y ~ x + mo(lik)", FRAME, priors(simo=ab.Dirichlet(1)))
    post_lik = az.from_dict(posterior={"Intercept": np.full((1, 2), 0.3), "x": np.full((1, 2), 0.7), "mo(lik)": np.full((1, 2), -0.45),
                                       "simo_lik": np.broadcast_to(np.array([0.25, 0.75]), (1, 2, 2))},
                            coords={"simo_lik_dim": ["1", "2"]}, dims={"simo_lik": ["simo_lik_dim"]})
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "lik": ["always", "never", "rarely"]})          # labels, read by the fitted coding
    np.testing.assert_allclose(lik.predict(post_lik, newdata=new, kind="mean").mu[0, 0], 0.3 - 0.45 * np.array([2.0, 0.0, 0.5]),
                               rtol=1e-14)
    with pytest.raises(ab.FormulaError, match=r"has values \['often'\] that are not among its ordered categories"):
        lik.predict(post_lik, newdata=new.assign(lik=["often", "never", "rarely"]), kind="mean")


# -- describe(), names, identity, records, prior predictive -----------------------------------------------------------
def test_describe_and_names():
    m = gaussian("y ~ x + mo(ord) + (1 | g)", sd=ab.HalfNormal(1))
    text = m.describe()
    for line in ("AutoBayes M0 model with monotonic effect(s)",
                 "  mo(ord)   : monotonic effect of 'ord' (brms mo()): whole numbers 1..4 coded as the value minus 1, 0..3 (brms's Xmo), "
                 "D = 3 steps (brms's Jmo)",
                 "  mu_i = a_c + (x_x - mean(x_x)) * b_x + b_mo(ord) * mo(ord)_i + u[g_i]",
                 "  mo(ord)_i = 3 * (simo_ord[1] + ... + simo_ord[Xmo_i]) with Xmo_i = ord_i - 1 in 0..3",
                 "one per step 1 -> 2, 2 -> 3, 3 -> 4: b_mo(ord) (brms's bsp) is the average change of mu per step",
                 "  mo(ord) is not centred (brms does not centre it either): the intercept refers to rows at ord = 1.",
                 "  b_mo(ord) ~ Normal(mu=0, sigma=2)   (brms's bsp_moord: class b, coefficient moord)",
                 "  simo_ord ~ Dirichlet(a[1]=2, a[2]=2, a[3]=2)   (brms class simo, coefficient moord1; concentrations 2, 2, 2)",
                 "  simo_ord_stickbreaking__ simo_ord[1..3] on Stan's stick-breaking simplex coordinates (2 entries; brms's simo)",
                 "WARNING: values ['3'] of the monotonic predictor 'ord' have no fitted rows"):
        assert line in text, line
    assert m.mo_names == {"ord": {"b": "mo(ord)", "simplex": "simo_ord", "coordinate": "simo_ord_dim"}}
    assert m.names["b:mo(ord)"] == "mo(ord)" and m.names["simo:ord"] == "simo_ord"
    assert m.coefficients == (("x", "x", None),)
    lik = ab.gaussian("y ~ mo(lik)", FRAME, priors(simo=ab.Dirichlet(1)))
    assert "ordered categories never < rarely < always coded 0..2" in lik.describe()
    assert "Xmo_i = the position of lik_i among the ordered categories in 0..2" in lik.describe()
    with pytest.raises(ab.FormulaError, match=r"generated variable/coordinate name collision: \['simo_ord'\]"):
        ab.gaussian("y ~ mo(ord) + simo_ord", FRAME.assign(simo_ord=FRAME.x), priors())


def test_the_target_identity_holds_the_term():
    base = gaussian().target_identity
    assert gaussian().target_identity == base
    assert gaussian(simo=ab.Dirichlet(2)).target_identity != base           # the same concentrations, another prior as written
    assert gaussian(simo=ab.Dirichlet(1, 2, 2)).target_identity != base
    assert gaussian(frame=FRAME.assign(ord=FRAME.ord + 10)).target_identity != base   # the same codes from other values
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    assert plain.target_identity != base


def test_fit_records_and_summary_rows(monkeypatch):
    m = gaussian()
    post = draws(m)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert {"mo(ord)", "simo_ord[1]", "simo_ord[2]", "simo_ord[3]"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=3,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["b:mo(ord)"] == "Normal(mu=0, sigma=2)"
    assert rec["model"]["priors"]["simo:ord"] == "Dirichlet(a[1]=2, a[2]=2, a[3]=2)"
    assert rec["model"]["monotonic"] == {"mo(ord)": {"column": "ord", "coding": "integer", "values": ["1", "4"], "D": 3,
                                                     "names": m.mo_names["ord"]}}
    assert rec["model"]["terms"] == ["x", "mo(ord)"] and rec["model"]["design_columns"] == ["x"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_prior_predictive_draws_simplexes():
    pp = gaussian().prior_predictive(draws=50, seed=3)
    simplex = pp.prior["simo_ord"].to_numpy()
    assert simplex.shape == (1, 50, 3) and (simplex > 0).all()
    np.testing.assert_allclose(simplex.sum(-1), 1.0, rtol=1e-12)


# -- brms's default priors ------------------------------------------------------------------------------------------------
# brms 2.23.0's default_prior() on FRAME: (class, coef, group, prior, lb); every other column empty, source "default"
BRMS = {
    "gaussian, x + mo(ord) + (1 | g)": ("y ~ x + mo(ord) + (1 | g)", "gaussian", [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", ""), ("b", "moord", "", "", ""), ("b", "x", "", "", ""),
        ("sd", "", "", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g", "", ""), ("sd", "Intercept", "g", "", ""),
        ("sigma", "", "", "student_t(3, 0, 2.5)", "0"), ("simo", "moord1", "", "dirichlet(1)", "")]),
    "gaussian, an ordered factor": ("y ~ mo(lik)", "gaussian", [
        ("Intercept", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", ""), ("b", "molik", "", "", ""),
        ("sigma", "", "", "student_t(3, 0, 2.5)", "0"), ("simo", "molik1", "", "dirichlet(1)", "")]),
    "poisson, two terms": ("cnt ~ mo(neg) + mo(ord)", "poisson", [
        ("Intercept", "", "", "student_t(3, 0, 2.5)", ""), ("b", "", "", "", ""), ("b", "moneg", "", "", ""), ("b", "moord", "", "", ""),
        ("simo", "moneg1", "", "dirichlet(1)", ""), ("simo", "moord1", "", "dirichlet(1)", "")]),
    "gaussian, no intercept": ("y ~ 0 + mo(ord)", "gaussian", [
        ("b", "", "", "", ""), ("b", "moord", "", "", ""), ("sigma", "", "", "student_t(3, 0, 2.5)", "0"),
        ("simo", "moord1", "", "dirichlet(1)", "")]),
    "cumulative": ("ord ~ x + mo(neg)", "cumulative", [
        ("Intercept", "", "", "student_t(3, 0, 2.5)", ""), ("Intercept", "1", "", "", ""), ("Intercept", "2", "", "", ""),
        ("Intercept", "3", "", "", ""), ("b", "", "", "", ""), ("b", "moneg", "", "", ""), ("b", "x", "", "", ""),
        ("simo", "moneg1", "", "dirichlet(1)", "")]),
}


@pytest.mark.parametrize("label", sorted(BRMS))
def test_the_default_prior_table_is_brms(label):
    formula, family, rows = BRMS[label]
    pri = ab.default_priors(formula, FRAME, family=family)
    model = getattr(ab, family)(formula, FRAME, pri)
    table = model.prior_table()
    assert [(r["class"], r["coef"], r["group"], r["prior"], r["lb"]) for r in table] == rows
    assert all(r["ub"] == r["resp"] == r["dpar"] == r["nlpar"] == r["tag"] == "" and r["source"] == "default" for r in table)
    columns = [t.operands[0] for t in model.spec.mo_terms]
    assert pri.simo == {c: ab.Dirichlet(1) for c in columns} and all(p.source == "default" for p in pri.simo.values())
    assert [label for label in model.default_prior_labels if label.startswith("simo:")] == [f"simo:{c}" for c in columns]


# -- the oracle's extension to monotonic terms ----------------------------------------------------------------------------
def _bundle(model=None):
    from tests.oracle import gates

    return gates.Bundle("gauss_monotonic", model=model)


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_an_ordered_categorical_or_shifted_predictor_meets_the_frozen_fixture():
    """The ordered-Categorical and the shifted whole-number codings give brms's codes of the frozen case, so the model
    built from either passes every gate against it: the coding, not the column's type, is what brms compares."""
    from tests.oracle import adapter, gates

    b = _bundle()
    labels = {1: "low", 2: "mid", 3: "high", 4: "top"}

    def ordered(ord):
        return pd.Categorical(ord.map(labels), categories=list(labels.values()), ordered=True)
    for recode in (ordered, lambda ord: ord + 10, lambda ord: ord.astype(float)):
        model = adapter.build_model(b.case, b.data.assign(ord=recode(b.data.ord)))
        bundle = gates.Bundle("gauss_monotonic", model=model)
        bundle.newdata = bundle.newdata.assign(ord=recode(bundle.newdata.ord))     # new rows in the same coding
        for gate in gates.ALL:
            gate(bundle)


def test_the_gates_see_a_corrupted_monotonic_fixture():
    from tests.oracle import adapter, gates
    from tests.oracle.adapter import OracleMismatch

    b = _bundle()
    for gate in gates.ALL:
        gate(b)

    def codes(raw):
        raw["standata"]["Xmo_1"]["values"][4] -= 1                       # a row of value 4 coded as 3
    with pytest.raises(OracleMismatch, match=r"monotonic term 'mo\(ord\)': 0 brms monotonic variables carry its codes"):
        _corrupted(b, codes)

    def concentration(raw):
        raw["standata"]["con_simo_1"]["values"][1] = 2.5
    with pytest.raises(OracleMismatch, match=r"brms con_simo_1 is .*; the prior of 'mo\(ord\)' gives"):
        gates.check_design(_corrupted(b, concentration))

    def coordinate(raw):
        i = [k for k, e in enumerate(raw["upars"]) if e["transform"] == "unreached"][0]
        raw["points"][5]["u"]["values"][i] += 1e-9
    with pytest.raises(OracleMismatch, match=r"point 5: Stan's stick-breaking coordinates of simo_1"):
        gates.check_gradient(_corrupted(b, coordinate))

    def gradient(raw):
        i = [k for k, e in enumerate(raw["upars"]) if e["transform"] == "unreached"][1]
        raw["points"][2]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 2: gradient of simo_ord_stickbreaking__\[1\]"):
        gates.check_gradient(_corrupted(b, gradient))

    def coefficient(raw):
        raw["points"][6]["pars"]["bsp"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 6: log density"):
        gates.check_logp(_corrupted(b, coefficient))

    def default(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "simo"]
        row["prior"] = "dirichlet(2)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(b, default))

    plain = ab.gaussian("y ~ 1 + (1 | g)", b.data, ab.Priors(intercept=ab.Normal(0, 5), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    with pytest.raises(OracleMismatch, match=r"brms has special terms \(Ksp, Xmo_\[1\]\); the AutoBayes model has no monotonic term"):
        gates.Bundle("gauss_monotonic", model=plain)
    other = adapter.build_model(b.case, b.data.assign(ord=b.data.ord.replace({2: 3, 3: 2})))   # 2 and 3 swapped: other codes
    with pytest.raises(OracleMismatch, match=r"0 brms monotonic variables carry its codes and its 3 steps"):
        gates.Bundle("gauss_monotonic", model=other)
