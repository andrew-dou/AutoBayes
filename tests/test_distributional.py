"""Distributional formulas (brms-parity B01): an auxiliary parameter with its own formula, brms's ``bf(y ~ x, sigma ~ z)``,
written ``["y ~ x", "sigma ~ z"]``. The term model of the second formula, its refusals, its priors and their checks, the
graph against a density written by hand, ``value_point``, by-hand predictions with each row's own ``sigma`` (and
``dpar="sigma"``), ``describe()``, the target identity, fit records, prior-predictive draws, and brms's default priors
for such formulas (the tables below were printed by brms 2.23.0's ``default_prior()``, R 4.5.2, C collation, on
``FRAME``). Nothing here samples; the brms comparison of the six claimed families is the oracle cases
``gauss_distributional_sigma``, ``student_distributional_sigma``, ``lognormal_distributional_sigma``,
``negbinomial_distributional_shape``, ``gamma_distributional_shape`` and ``beta_distributional_phi``."""
import copy
import json
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import autobayes as ab
from autobayes.families import FAMILIES
from autobayes.formula import parse_formula, pop_term_design

FRAME = pd.DataFrame({
    "y": [0.01, 0.5, 3.0, 40.0, 200.0, 1500.0, 2.2, 0.03, 7.5, 0.0, 61.0, 0.9],     # positive with a zero (0.1 on the log scale)
    "yn": [-1.2, 0.4, 2.2, -0.3, 1.9, 0.8, -2.5, 1.1, 0.05, -0.7, 3.3, 0.6],       # a negative response: log y is NaN for brms
    "x": [0.12, -1.3, 0.85, 1.7, -0.44, 0.03, -2.1, 0.9, 1.15, -0.6, 0.31, -0.05],
    "z": ["p", "q", "r", "p", "q", "r", "p", "q", "r", "p", "q", "r"],
    "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
    "cnt": [0, 3, 0, 1, 7, 2, 0, 0, 12, 4, 1, 0],
    "ypos": [0.82, 3.5, 1.27, 0.06, 12.4, 2.2, 0.95, 5.75, 0.31, 1.6, 8.9, 0.47],
    "prop": [0.12, 0.55, 0.31, 0.08, 0.77, 0.43, 0.26, 0.64, 0.19, 0.38, 0.91, 0.5],
})
Z = {"z": "p"}
SIGMA = {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}


def priors(**extra):
    return ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), **extra)


def gaussian(formula=("yn ~ x", "sigma ~ x + z"), *, frame=None, categorical=Z, **extra):
    return ab.gaussian(list(formula), FRAME if frame is None else frame, priors(**{"sigma": SIGMA, **extra}), categorical=categorical)


# -- the term model ---------------------------------------------------------------------------------------------------
def test_a_list_of_formulas_parses_into_the_response_formula_and_its_distributional_ones():
    s = parse_formula(["yn ~ x + (1 | g)", "sigma ~ x * z"], FRAME, "gaussian", Z)
    alone = parse_formula("yn ~ x + (1 | g)", FRAME, "gaussian")
    assert s.formula == alone.formula and s.pop_terms == alone.pop_terms and s.group_terms == alone.group_terms
    assert s.canonical_formula == alone.canonical_formula and s.bambi_formula == alone.bambi_formula
    assert s.dpar_names == ("sigma",) and s.formulas == ("yn ~ x + (1 | g)", "sigma ~ x * z")
    d = s.dpar("sigma")
    assert d.response == "sigma" and d.intercept and d.terms == ("x", "z", "x:z") and not d.group_terms
    assert d.columns == ("x", "z[q]", "z[r]", "x:z[q]", "x:z[r]") and d.categorical("z").levels == ("p", "q", "r")
    assert d.canonical_formula == "sigma ~ x + z + x:z" and d.bambi_rhs == "x + z + x:z"
    assert s.categoricals == () and [c.name for c in s.all_categoricals] == ["z"] and s.model_columns == ("x", "z")
    with pytest.raises(KeyError):
        s.dpar("shape")
    assert parse_formula(["yn ~ x"], FRAME) == parse_formula("yn ~ x", FRAME)        # one formula is the string itself
    assert parse_formula(("yn ~ x", "sigma ~ x"), FRAME).dpar_names == ("sigma",)    # a tuple too
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):                   # asdict hands dpars back as mappings
        assert replica == s and hash(replica) == hash(s) and replica.dpars == s.dpars
    # a column declared categorical is categorical in every formula that names it as a main effect
    both = parse_formula(["yn ~ z", "sigma ~ z"], FRAME, "gaussian", Z)
    assert both.categorical("z") == both.dpar("sigma").categorical("z")
    assert [c.name for c in both.all_categoricals] == ["z"]


REFUSALS = [
    (["yn ~ x", "sigma ~ x + (1 | g)"], "gaussian", r"group-level terms in a distributional formula \(\(1 \| g\)\) are not supported.*D32"),
    (["yn ~ x", "sigma ~ x + offset(ypos)"], "gaussian", r"'offset\(ypos\)' in a distributional formula is not supported.*D32"),
    (["yn ~ x", "nu ~ x"], "student", r"brms's nu ~ \.\.\. is not offered here; a distributional formula is offered for the auxiliary "
                                      r"parameter of the gaussian \(sigma\), student \(sigma\).*divergence D33"),
    (["cnt ~ x", "zi ~ x"], "zero_inflated_poisson", r"brms's zi ~ \.\.\. is not offered here.*D33"),
    (["yn ~ x", "shape ~ x"], "gaussian", r"the gaussian family has no parameter 'shape' \(its parameters besides the mean are \['sigma'\]\)"),
    (["cnt ~ x", "sigma ~ x"], "poisson", r"the poisson family has no parameter 'sigma' \(it has no parameter besides the mean\)"),
    (["yn ~ x", "mu ~ x"], "gaussian", r"'mu' is the response formula's; the first formula of the list is the response's"),
    (["yn ~ x", "yn ~ z"], "gaussian", r"'yn' is the response formula's"),
    (["yn ~ x", "sigma ~ x", "sigma ~ z"], "gaussian", r"'sigma' has two formulas"),
    (["yn ~ x", "sigma | weights(x) ~ z"], "gaussian", r"a distributional formula is '<parameter> ~ <terms>'"),
    (["yn ~ x", "~ z"], "gaussian", r"a distributional formula is '<parameter> ~ <terms>'"),
    (["yn ~ x + (1 | g)", "sigma ~ g"], "gaussian", r"'g' is a grouping factor of the response formula; it cannot also be a predictor of sigma.*D35"),
    (["yn ~ x", "sigma ~ yn"], "gaussian", r"'yn' is the response; it cannot also be a predictor of sigma.*D35"),
    (["k | trials(n) ~ x", "sigma ~ x"], "binomial", r"the binomial family has no parameter 'sigma'"),
    ([], "gaussian", r"a formula is a string, or a list of formulas"),
    (["yn ~ x", 3], "gaussian", r"a formula is a string, or a list of formulas"),
    (["yn ~ x", "sigma ~ nope"], "gaussian", r"columns not in data: \['nope'\]"),
    (["yn ~ x", "sigma ~ g"], "gaussian", r"column 'g' has dtype \w+; a predictor is numeric unless it is declared categorical"),
]


@pytest.mark.parametrize("formula, family, message", REFUSALS)
def test_refusals_name_the_construct(formula, family, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula, FRAME, family)


def test_declared_categorical_columns_are_categorical_everywhere():
    with pytest.raises(ab.FormulaError, match=r"'log\(x\)': 'x' is declared categorical; an in-formula transform applies to a numeric column"):
        parse_formula(["yn ~ x", "sigma ~ log(x)"], FRAME, "gaussian", {"x": "0.12"})
    with pytest.raises(ab.FormulaError, match=r"'weights\(ypos\)': 'ypos' is declared categorical; the response formula reads it as a number"):
        parse_formula(["yn | weights(ypos) ~ x", "sigma ~ ypos"], FRAME, "gaussian", {"ypos": "0.82"})
    with pytest.raises(ab.FormulaError, match=r"'nope' is declared categorical but is not a term of"):
        parse_formula(["yn ~ x", "sigma ~ z"], FRAME, "gaussian", {"z": "p", "nope": "a"})
    with pytest.raises(ab.FormulaError, match=r"reference level 'zz' of 'z' does not occur in the fitted rows"):
        parse_formula(["yn ~ x", "sigma ~ z"], FRAME, "gaussian", {"z": "zz"})


def test_center_predictors_false_is_refused_where_brms_would_still_centre():
    with pytest.raises(ab.FormulaError, match=r"center_predictors=False with the formula 'sigma ~ x' is not supported: brms centres.*D34"):
        ab.gaussian(["yn ~ x", "sigma ~ x"], FRAME, priors(sigma=SIGMA), center_predictors=False)
    for formula, sigma in ((["yn ~ x", "sigma ~ 0 + x"], {"b": ab.Normal(0, 0.5)}), (["yn ~ x", "sigma ~ 1"], {"intercept": ab.Normal(0, 1)})):
        m = ab.gaussian(formula, FRAME, priors(sigma=sigma), center_predictors=False)     # nothing for brms to centre here
        assert not m.design.dpars["sigma"].centered and not m.design.centered


# -- the design and the graph -----------------------------------------------------------------------------------------
def test_every_term_kind_has_its_own_centred_design_and_variables():
    frame = FRAME.assign(xpos=np.exp(FRAME.x))
    m = gaussian(("yn ~ x + (1 | g)", "sigma ~ x * z + log(xpos)"), frame=frame, sd=ab.HalfNormal(1))
    dd, d = m.design.dpars["sigma"], m.spec.dpar("sigma")
    raw = np.concatenate([pop_term_design(t, m.data, d.categoricals) for t in d.pop_terms], axis=1)
    assert np.array_equal(dd.X_raw, raw) and dd.columns == ("x", "z[q]", "z[r]", "log(xpos)", "x:z[q]", "x:z[r]")
    assert dd.centered and np.allclose(dd.x_means, raw.mean(0)) and np.allclose(dd.X_model, raw - raw.mean(0)) and dd.link == "log"
    assert np.allclose(dd.X_raw[:, 3], FRAME.x)                                         # log(exp(x))
    assert m.dpar_names == {"sigma": {"value": "sigma", "intercept": "sigma_Intercept", "b:x": "sigma_x", "b:z": "sigma_z",
                                      "b:log(xpos)": "sigma_log(xpos)", "b:x:z": "sigma_x:z"}}
    assert m.names["sigma"] == "sigma" and m.pymc_model["sigma"] in m.pymc_model.deterministics
    values = {v.name for v in m.pymc_model.value_vars}
    assert {"sigma_Intercept", "sigma_x", "sigma_z", "sigma_log(xpos)", "sigma_x:z"} <= values and "sigma_log__" not in values
    assert [str(c) for c in m.pymc_model.coords["sigma_z_dim"]] == ["q", "r"]
    assert m.design.X_raw.shape == (12, 1) and m.spec.columns == ("x",)                # the response formula is untouched


def test_the_density_is_the_one_written_by_hand():
    m = gaussian(("yn ~ x", "sigma ~ x + z"))
    dd = m.design.dpars["sigma"]
    a, b, a_s, b_s = 0.3, 0.7, -0.2, np.array([0.4, 0.25, -0.5])
    point = m.value_point(intercept=a, b=[b], sigma={"intercept": a_s, "b": b_s})
    assert set(point) == {"Intercept", "x", "sigma_Intercept", "sigma_x", "sigma_z"}
    mu = a + (FRAME.x.to_numpy() - FRAME.x.mean()) * b
    sigma = np.exp(a_s + dd.X_model @ b_s)
    want = (stats.norm.logpdf(FRAME.yn, mu, sigma).sum() + stats.norm.logpdf(a, 0, 5) + stats.norm.logpdf(b, 0, 2)
            + stats.norm.logpdf(a_s, 0, 1) + stats.norm.logpdf(b_s, 0, 0.5).sum())
    for jacobian in (True, False):                                  # every variable is unconstrained
        assert float(m.pymc_model.compile_logp(jacobian=jacobian)(point)) == pytest.approx(want, rel=1e-12)
    fn = m.pymc_model.compile_fn(m.pymc_model.replace_rvs_by_values([m.pymc_model["sigma"]]), inputs=m.pymc_model.value_vars,
                                 point_fn=True, on_unused_input="ignore")
    np.testing.assert_allclose(np.asarray(fn(point)[0]), sigma, rtol=1e-14)
    # with weights() (B02) each row's log density, its own sigma included, is multiplied by its weight
    w = gaussian(("yn | weights(ypos) ~ x", "sigma ~ x + z"))
    rows = stats.norm.logpdf(FRAME.yn, mu, sigma)
    weighted = want - rows.sum() + (FRAME.ypos.to_numpy() * rows).sum()
    wpoint = w.value_point(intercept=a, b=[b], sigma={"intercept": a_s, "b": b_s})
    assert float(w.pymc_model.compile_logp()(wpoint)) == pytest.approx(weighted, rel=1e-12)
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(m.pymc_model)
    assert float(jf([point[v.name] for v in m.pymc_model.value_vars])) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("family, formula, parameter, bambi", [
    ("student", ("yn ~ x", "sigma ~ x"), "sigma", "sigma"), ("lognormal", ("ypos ~ x", "sigma ~ z"), "sigma", "sigma"),
    ("negbinomial", ("cnt ~ x", "shape ~ z"), "shape", "alpha"), ("gamma", ("ypos ~ x", "shape ~ x"), "shape", "alpha"),
    ("beta", ("prop ~ x", "phi ~ z"), "phi", "phi")])
def test_each_family_offers_a_formula_on_its_auxiliary_parameter(family, formula, parameter, bambi):
    kwargs = {"nu": 4} if family == "student" else {}
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **{parameter: SIGMA})
    m = getattr(ab, family)(list(formula), FRAME, pri, categorical=Z if "z" in formula[1] else None, **kwargs)
    assert FAMILIES[family].distributional_link == "log" and FAMILIES[family].scale_parameter == parameter
    assert m.names[parameter] == bambi and m.dpar_names[parameter]["value"] == bambi
    assert m.dpar_names[parameter]["intercept"] == f"{bambi}_Intercept"
    assert m.bambi_model.family.link[bambi].name == "log"
    point = m.value_point(intercept=0.1, b=[0.2], **{parameter: {"intercept": 0.3, "b": np.full(len(m.design.dpars[parameter].columns), 0.1)}})
    assert np.isfinite(float(m.pymc_model.compile_logp()(point)))
    assert {name for name, spec in FAMILIES.items() if spec.distributional_link is not None} == {
        "gaussian", "student", "lognormal", "negbinomial", "gamma", "beta"}


# -- priors -----------------------------------------------------------------------------------------------------------
PRIOR_REFUSALS = [
    (None, r"the formula 'sigma ~ x \+ z' gives sigma its own linear predictor, which needs priors on its intercept and "
           r"coefficients: pass sigma=\{'intercept': \.\.\., 'b': \.\.\.\}"),
    (ab.HalfNormal(1), r"sigma has its own formula 'sigma ~ x \+ z': its priors are a mapping \{'intercept': prior, 'b': prior\}"),
    ({"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), "sd": ab.HalfNormal(1)}, r"take the keys 'intercept' and 'b'.*unexpected \['sd'\]"),
    ({"b": ab.Normal(0, 1)}, r"'sigma ~ x \+ z' has an intercept, which needs a prior"),
    ({"intercept": ab.Normal(0, 1)}, r"'sigma ~ x \+ z' has the population-level term\(s\) \['x', 'z'\], which need priors"),
    ({"intercept": ab.HalfNormal(1), "b": ab.Normal(0, 1)}, r"prior for 'sigma:intercept' must be a location prior"),
    ({"intercept": ab.Normal(0, 1), "b": {"x": ab.Normal(0, 1)}}, r"coefficient priors must cover exactly the predictor terms \['x', 'z'\]"),
    ({"intercept": ab.Normal(0, 1), "b": {"x": ab.Normal(0, 1), "z": {"q": ab.Normal(0, 1), "r": ab.HalfNormal(1)}}},
     r"must share one distribution family"),
]


@pytest.mark.parametrize("sigma, message", PRIOR_REFUSALS)
def test_prior_refusals(sigma, message):
    with pytest.raises(ab.PriorError, match=message):
        ab.gaussian(["yn ~ x", "sigma ~ x + z"], FRAME, priors(sigma=sigma), categorical=Z)


def test_prior_forms():
    with pytest.raises(ab.PriorError, match=r"prior for 'sigma' is a mapping, the form of the priors of sigma's own formula"):
        ab.gaussian("yn ~ x", FRAME, priors(sigma=SIGMA))
    with pytest.raises(ab.PriorError, match=r"'sigma ~ 0 \+ z' has no intercept; omit 'intercept'"):
        ab.gaussian(["yn ~ x", "sigma ~ 0 + z"], FRAME, priors(sigma=SIGMA), categorical=Z)
    per_term = {"intercept": ab.StudentT(3, 0, 1), "b": {"x": ab.Normal(0, 1), "z": {"q": ab.Normal(0, 2), "r": ab.Normal(0, 3)}}}
    m = gaussian(sigma=per_term)
    assert m.priors.dpar_priors(m.spec, "sigma").coefficient_priors(m.spec.dpar("sigma")) == {
        "x": ab.Normal(0, 1), "z[q]": ab.Normal(0, 2), "z[r]": ab.Normal(0, 3)}
    rows = {(r["class"], r["coef"], r["dpar"]): r["prior"] for r in m.prior_table()}
    assert rows[("Intercept", "", "sigma")] == "student_t(3, 0, 1)" and rows[("b", "zr", "sigma")] == "normal(0, 3)"
    assert rows[("b", "", "sigma")] == "" and ("sigma", "", "") not in rows
    only = ab.gaussian(["yn ~ x", "sigma ~ 1"], FRAME, priors(sigma={"intercept": ab.Normal(0, 1)}))   # intercept-only: no b needed
    assert not only.design.dpars["sigma"].columns and only.dpar_names["sigma"] == {"value": "sigma", "intercept": "sigma_Intercept"}
    with pytest.raises(ab.PriorError, match="must be a positive-support prior"):               # without a formula, as always
        ab.gaussian("yn ~ x", FRAME, priors(sigma=ab.Normal(0, 1)))


def test_value_point_refusals():
    m = gaussian()
    with pytest.raises(ValueError, match=r"sigma has its own formula 'sigma ~ x \+ z': give sigma=\{'intercept': float, 'b': array\}"):
        m.value_point(intercept=0.0, b=[0.0], sigma=0.5)
    with pytest.raises(ValueError, match=r"sigma\['intercept'\] is required"):
        m.value_point(intercept=0.0, b=[0.0], sigma={"b": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match=r"sigma\['b'\] must have 3 finite entries, one per design column \['x', 'z\[q\]', 'z\[r\]'\]"):
        m.value_point(intercept=0.0, b=[0.0], sigma={"intercept": 0.0, "b": [0.0]})
    with pytest.raises(ValueError, match="sigma is required for the Gaussian family"):
        m.value_point(intercept=0.0, b=[0.0])


# -- predictions --------------------------------------------------------------------------------------------------------
def draws(model, a=0.3, b=0.7, a_s=-0.2, b_s=(0.4, 0.25, -0.5)):
    """Two chains of four constant draws: the raw-scale intercepts (as a posterior from fit() holds them) and coefficients."""
    roles = model.dpar_names["sigma"]
    post = {"Intercept": np.full((2, 4), a), "x": np.full((2, 4), b), roles["intercept"]: np.full((2, 4), a_s),
            roles["b:x"]: np.full((2, 4), b_s[0]), roles["b:z"]: np.broadcast_to(np.array(b_s[1:]), (2, 4, 2))}
    return az.from_dict(posterior=post, coords={"sigma_z_dim": ["q", "r"]}, dims={"sigma_z": ["sigma_z_dim"]})


def test_predictions_use_each_rows_sigma():
    m = gaussian()
    post = draws(m)
    z = FRAME.z.to_numpy()
    mu = 0.3 + 0.7 * FRAME.x.to_numpy()                                            # raw-scale intercept
    sigma = np.exp(-0.2 + 0.4 * FRAME.x.to_numpy() + 0.25 * (z == "q") - 0.5 * (z == "r"))
    np.testing.assert_allclose(m.predict(post, kind="linear").eta, np.broadcast_to(mu, (2, 4, 12)), rtol=1e-14)
    np.testing.assert_allclose(m.predict(post, kind="mean").mu, np.broadcast_to(mu, (2, 4, 12)), rtol=1e-14)
    out = m.predict(post, kind="response", seed=5)
    rng = np.random.default_rng(5)
    np.testing.assert_allclose(out.y, rng.normal(np.broadcast_to(mu, (2, 4, 12)), np.broadcast_to(sigma, (2, 4, 12))), rtol=1e-14)
    lin = m.predict(post, kind="linear", dpar="sigma")
    np.testing.assert_allclose(lin.eta, np.broadcast_to(np.log(sigma), (2, 4, 12)), rtol=1e-13)
    assert lin.attrs["dpar"] == "sigma" and lin.attrs["link"] == "log" and lin.attrs["prediction_scale"] == "linear"
    np.testing.assert_allclose(m.predict(post, kind="mean", dpar="sigma").sigma, np.broadcast_to(sigma, (2, 4, 12)), rtol=1e-13)
    new = pd.DataFrame({"x": [0.5, -1.0], "z": ["r", "p"]})
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="mean", dpar="sigma").sigma[0, 0],
                               np.exp(-0.2 + 0.4 * np.array([0.5, -1.0]) + np.array([-0.5, 0.0])), rtol=1e-13)
    with pytest.raises(ValueError, match=r"dpar='sigma' takes kind='linear' \(its linear predictor\) or kind='mean'"):
        m.predict(post, kind="response", dpar="sigma")
    with pytest.raises(ValueError, match=r"dpar must name a distributional parameter with its own formula, \['sigma'\]; got 'nu'"):
        m.predict(post, kind="mean", dpar="nu")
    with pytest.raises(ab.FormulaError, match=r"newdata lacks columns \['z'\]"):
        m.predict(post, newdata=new[["x"]], kind="mean")
    with pytest.raises(ab.FormulaError, match=r"newdata has levels of the categorical predictor 'z' that were not in the fitted rows: \['s'\]"):
        m.predict(post, newdata=new.assign(z=["s", "p"]), kind="mean")
    plain = ab.gaussian("yn ~ x", FRAME, priors(sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match="this model has no distributional formula; dpar does not apply"):
        plain.predict(az.from_dict(posterior={"Intercept": np.zeros((1, 2)), "x": np.zeros((1, 2)), "sigma": np.ones((1, 2))}),
                      kind="mean", dpar="sigma")


def test_the_lognormal_mean_and_every_familys_draws_take_each_rows_parameter():
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=SIGMA)
    m = ab.lognormal(["ypos ~ x", "sigma ~ x + z"], FRAME, pri, categorical=Z)
    post = draws(m)
    z = FRAME.z.to_numpy()
    mu = 0.3 + 0.7 * FRAME.x.to_numpy()
    sigma = np.exp(-0.2 + 0.4 * FRAME.x.to_numpy() + 0.25 * (z == "q") - 0.5 * (z == "r"))
    np.testing.assert_allclose(m.predict(post, kind="mean")["mean"], np.broadcast_to(np.exp(mu + sigma ** 2 / 2), (2, 4, 12)), rtol=1e-13)
    out = m.predict(post, kind="response", seed=3)
    np.testing.assert_allclose(out.y, np.random.default_rng(3).lognormal(np.broadcast_to(mu, (2, 4, 12)), np.broadcast_to(sigma, (2, 4, 12))),
                               rtol=1e-13)
    for family, formula, parameter in [("student", ["yn ~ x", "sigma ~ x + z"], "sigma"), ("negbinomial", ["cnt ~ x", "shape ~ x + z"], "shape"),
                                       ("gamma", ["ypos ~ x", "shape ~ x + z"], "shape"), ("beta", ["prop ~ x", "phi ~ x + z"], "phi")]:
        fm = getattr(ab, family)(formula, FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **{parameter: SIGMA}),
                                 categorical=Z, **({"nu": 4} if family == "student" else {}))
        roles = fm.dpar_names[parameter]
        p = az.from_dict(posterior={"Intercept": np.full((2, 4), 0.1), "x": np.full((2, 4), 0.2), roles["intercept"]: np.full((2, 4), 0.5),
                                    roles["b:x"]: np.full((2, 4), 0.3), roles["b:z"]: np.full((2, 4, 2), -0.2)},
                         coords={f"{roles['b:z']}_dim": ["q", "r"]}, dims={roles["b:z"]: [f"{roles['b:z']}_dim"]})
        y = fm.predict(p, kind="response", seed=1).y.to_numpy()
        assert y.shape == (2, 4, 12) and np.isfinite(y).all(), family
        np.testing.assert_allclose(fm.predict(p, kind="mean", dpar=parameter)[parameter][0, 0],
                                   np.exp(0.5 + 0.3 * FRAME.x.to_numpy() - 0.2 * (z != "p")), rtol=1e-13)


# -- describe, identity, records, prior predictive --------------------------------------------------------------------
def test_describe_writes_the_formula_out():
    text = gaussian(("yn ~ x + (1 | g)", "sigma ~ x + z"), sd=ab.HalfNormal(1)).describe()
    assert text.splitlines()[0] == "AutoBayes M0 model with distributional formula(s) for sigma"     # z is sigma's only
    for line in ["  sigma     : sigma ~ x + z   (distributional formula: sigma has its own linear predictor, log link)",
                 "  z         : categorical, treatment coding, reference 'p'; levels (sorted) p, q, r; coefficients b_sigma_z[q, r]",
                 "Distributional parameter sigma (its own formula 'sigma ~ x + z', log link; brms's bf(..., sigma ~ x + z))",
                 "  log(sigma_i) = eta_sigma,i = a_sigma_c + (x_x - mean(x_x)) * b_sigma_x + (1[z_i = q] - p_z[q]) * b_sigma_z[q] "
                 "+ (1[z_i = r] - p_z[r]) * b_sigma_z[r]",
                 "  a_sigma_c ~ Normal(mu=0, sigma=1)   (on the log scale of sigma)", "  b_sigma_z[r] ~ Normal(mu=0, sigma=0.5)",
                 "  sigma_i = exp(eta_sigma,i), one per row: the formula 'sigma ~ x + z' (Distributional parameter sigma, above)",
                 "  y_i ~ Normal(mu_i, sigma_i)", "  sigma_Intercept          a_sigma_c (the intercept of sigma's formula)",
                 "  sigma_z                  b_sigma_z[q, r] (vector; dims sigma_z_dim, posterior rows sigma_z[q], sigma_z[r])",
                 "  sigma_x                  b_sigma_x"]:
        assert line in text.splitlines(), line
    assert "\n  sigma_log__" not in text and "  sigma ~ HalfNormal" not in text
    assert text.splitlines()[-1].endswith("; sigma = exp(eta_sigma), one sigma per row (B01)")
    beta = ab.beta(["prop ~ x", "phi ~ x"], FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), phi=SIGMA)).describe()
    assert "  y_i ~ Beta(mu_i phi_i, (1 - mu_i) phi_i); mu_i = expit(eta_i); E[y_i] = mu_i, Var[y_i] = mu_i (1 - mu_i) / (1 + phi_i)" in beta


def test_the_target_identity_holds_the_formula_and_its_priors():
    base = gaussian().target_identity
    assert gaussian().target_identity == base
    assert gaussian(("yn ~ x", "sigma ~ x"), categorical=None).target_identity != base
    assert gaussian(sigma={"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 0.5)}).target_identity != base
    assert gaussian(sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.6)}).target_identity != base
    plain = ab.gaussian("yn ~ x", FRAME, priors(sigma=ab.HalfNormal(1))).target_identity
    assert plain != base and ab.gaussian(["yn ~ x"], FRAME, priors(sigma=ab.HalfNormal(1))).target_identity == plain


def test_fit_records_and_summary_rows(monkeypatch):
    m = gaussian()
    post = draws(m)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "sigma_Intercept", "sigma_x", "sigma_z[q]", "sigma_z[r]"}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    pri = rec["model"]["priors"]
    assert pri["sigma:intercept"] == "Normal(mu=0, sigma=1)" and pri["sigma:b:z[q]"] == "Normal(mu=0, sigma=0.5)" and "sigma" not in pri
    dist = rec["model"]["distributional"]["sigma"]
    assert dist["formula"] == "sigma ~ x + z" and dist["link"] == "log" and dist["centered"] and dist["design_columns"] == ["x", "z[q]", "z[r]"]
    assert dist["names"]["intercept"] == "sigma_Intercept"
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
    other = json.loads(json.dumps(rec))
    other["model"]["distributional"]["sigma"]["formula"] = "sigma ~ x"
    with pytest.raises(ValueError, match="different models"):
        ab.compare_records(rec, other)


def test_prior_predictive_draws_and_flat_priors():
    pp = gaussian().prior_predictive(draws=20, seed=4)
    assert pp.prior["sigma"].shape == (1, 20, 12) and (pp.prior["sigma"] > 0).all()
    flat = gaussian(sigma={"intercept": ab.Normal(0, 1), "b": ab.Flat()})
    with pytest.raises(ValueError, match=r"the prior of \['sigma:b:x', 'sigma:b:z\[q\]', 'sigma:b:z\[r\]'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)


# -- brms's default priors -----------------------------------------------------------------------------------------------
# (formulas, family, keyword arguments, brms's rows as (class, coef, group, dpar, prior, lb)); every row's ub, resp, nlpar
# and tag are empty and its source "default"
BRMS = {
    "gaussian, sigma ~ z, a zero response": (["y ~ x + (1 | g)", "sigma ~ z"], "gaussian", {"categorical": Z}, [
        ("Intercept", "", "", "", "student_t(3, 2.6, 3.8)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("sd", "", "", "", "student_t(3, 0, 3.8)", "0"), ("sd", "", "g", "", "", ""), ("sd", "Intercept", "g", "", "", ""),
        ("Intercept", "", "", "sigma", "student_t(3, 0, 4.4)", ""), ("b", "", "", "sigma", "", ""),
        ("b", "zq", "", "sigma", "", ""), ("b", "zr", "", "sigma", "", "")]),
    "gaussian, sigma ~ x * z, a negative response": (["yn ~ x", "sigma ~ x * z"], "gaussian", {"categorical": Z}, [
        ("Intercept", "", "", "", "student_t(3, 0.5, 2.5)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("Intercept", "", "", "sigma", "student_t(3, 0, 2.5)", ""), ("b", "", "", "sigma", "", ""), ("b", "x", "", "sigma", "", ""),
        ("b", "x:zq", "", "sigma", "", ""), ("b", "x:zr", "", "sigma", "", ""), ("b", "zq", "", "sigma", "", ""),
        ("b", "zr", "", "sigma", "", "")]),
    "gaussian, no intercepts": (["y ~ 0 + x", "sigma ~ 0 + z"], "gaussian", {"categorical": Z}, [
        ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""), ("b", "", "", "sigma", "", ""), ("b", "zp", "", "sigma", "", ""),
        ("b", "zq", "", "sigma", "", ""), ("b", "zr", "", "sigma", "", "")]),
    "gaussian, sigma ~ 1": (["y ~ x", "sigma ~ 1"], "gaussian", {}, [
        ("Intercept", "", "", "", "student_t(3, 2.6, 3.8)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("Intercept", "", "", "sigma", "student_t(3, 0, 4.4)", "")]),
    "student, sigma ~ log(ypos)": (["y ~ x", "sigma ~ log(ypos)"], "student", {}, [
        ("Intercept", "", "", "", "student_t(3, 2.6, 3.8)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("nu", "", "", "", "gamma(2, 0.1)", "1"), ("Intercept", "", "", "sigma", "student_t(3, 0, 4.4)", ""),
        ("b", "", "", "sigma", "", ""), ("b", "logypos", "", "sigma", "", "")]),
    "lognormal, sigma ~ z": (["ypos ~ x", "sigma ~ z"], "lognormal", {"categorical": Z}, [
        ("Intercept", "", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("Intercept", "", "", "sigma", "student_t(3, 0, 2.5)", ""), ("b", "", "", "sigma", "", ""), ("b", "zq", "", "sigma", "", ""),
        ("b", "zr", "", "sigma", "", "")]),
    "negbinomial, shape ~ z": (["cnt ~ x", "shape ~ z"], "negbinomial", {"categorical": Z}, [
        ("Intercept", "", "", "", "student_t(3, 0, 3.1)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("Intercept", "", "", "shape", "student_t(3, 0, 2.5)", ""), ("b", "", "", "shape", "", ""), ("b", "zq", "", "shape", "", ""),
        ("b", "zr", "", "shape", "", "")]),
    "Gamma, shape ~ x": (["ypos ~ x", "shape ~ x"], "gamma", {}, [
        ("Intercept", "", "", "", "student_t(3, 0.4, 2.5)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("Intercept", "", "", "shape", "student_t(3, 0, 2.5)", ""), ("b", "", "", "shape", "", ""), ("b", "x", "", "shape", "", "")]),
    "Beta, phi ~ z": (["prop ~ x", "phi ~ z"], "beta", {"categorical": Z}, [
        ("Intercept", "", "", "", "student_t(3, 0, 2.5)", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("Intercept", "", "", "phi", "student_t(3, 0, 2.5)", ""), ("b", "", "", "phi", "", ""), ("b", "zq", "", "phi", "", ""),
        ("b", "zr", "", "phi", "", "")]),
}


@pytest.mark.parametrize("label", sorted(BRMS))
def test_the_default_prior_table_is_brms(label):
    formula, family, kwargs, rows = BRMS[label]
    pri = ab.default_priors(formula, FRAME, family=family, **kwargs)
    model = getattr(ab, family)(formula, FRAME, pri, **kwargs)
    table = model.prior_table()
    assert [(r["class"], r["coef"], r["group"], r["dpar"], r["prior"], r["lb"]) for r in table] == rows
    assert all(r["ub"] == r["resp"] == r["nlpar"] == r["tag"] == "" and r["source"] == "default" for r in table)
    scale, ds = FAMILIES[family].scale_parameter, model.spec.dpar(FAMILIES[family].scale_parameter)
    want = ([f"{scale}:intercept"] if ds.intercept else []) + [f"{scale}:b:{col}" for col in ds.columns]
    assert [label for label in model.default_prior_labels if label.startswith(f"{scale}:")] == want and scale not in model.default_prior_labels
    if ds.intercept:
        assert f"  a_{scale}{'_c' if model.design.dpars[scale].centered else ''} ~ StudentT(" in model.describe()


def test_the_default_intercept_of_a_distributional_formula_reads_the_response_on_the_log_scale():
    pri = ab.default_priors(["y ~ x", "sigma ~ x"], FRAME)
    assert pri.sigma == {"intercept": ab.StudentT(3, 0, 4.4), "b": ab.Flat()}
    assert pri.sigma["intercept"].source == "default" and pri.sigma["b"].source == "default"
    assert ab.default_priors(["y ~ x", "sigma ~ 1"], FRAME).sigma == {"intercept": ab.StudentT(3, 0, 4.4)}
    assert ab.default_priors(["y ~ 0 + x", "sigma ~ 0 + x"], FRAME).sigma == {"b": ab.Flat()}
    assert ab.default_priors(["cnt ~ x", "shape ~ x"], FRAME, family="negbinomial").shape == {
        "intercept": ab.StudentT(3, 0, 2.5), "b": ab.Flat()}                      # brms's student_t(3, 0, 2.5) for shape and phi


# -- the oracle's extension to distributional formulas fails on a corrupted fixture -------------------------------------
def _sigma_bundle():
    from tests.oracle import fixture as fx
    from tests.oracle import gates

    return fx, gates, gates.Bundle("gauss_distributional_sigma")


def _corrupted(fx, bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_a_corrupted_distributional_fixture():
    from tests.oracle.adapter import OracleMismatch

    fx, gates, b = _sigma_bundle()
    for gate in gates.ALL:
        gate(b)

    def design(raw):
        raw["standata"]["X_sigma"]["values"][-1] += 1e-11                        # the last entry of zr
    with pytest.raises(OracleMismatch, match=r"sigma's design column 'z\[r\]' matches 0 brms X_sigma columns"):
        _corrupted(fx, b, design)

    def coefficient(raw):
        raw["points"][3]["pars"]["b_sigma"]["values"][0] += 1e-6                 # the density then differs at point 3
    with pytest.raises(OracleMismatch, match="point 3: log density"):
        gates.check_logp(_corrupted(fx, b, coefficient))

    def gradient(raw):
        (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "Intercept_sigma"]
        raw["points"][1]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 1: gradient of sigma_Intercept\[0\]"):
        gates.check_gradient(_corrupted(fx, b, gradient))

    def raw_intercept(raw):
        raw["points"][2]["pars"]["b_sigma_Intercept"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 2: raw-scale intercept of sigma"):
        gates.check_derived(_corrupted(fx, b, raw_intercept))

    def default(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "Intercept" and r["dpar"] == "sigma"]
        row["prior"] = "student_t(3, 0, 2.6)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(fx, b, default))

    def no_rows(raw):
        raw["default_prior"] = [r for r in raw["default_prior"] if r["dpar"] != "sigma"]
    with pytest.raises(OracleMismatch, match=r"brms default_prior lacks the rows \[\('Intercept', 'sigma'\), \('b', 'sigma'\)\]"):
        gates.check_default_prior(_corrupted(fx, b, no_rows))

    plain = ab.gaussian("y ~ x + (1 | g)", b.data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                                                             sigma=ab.HalfNormal(1)))
    with pytest.raises(OracleMismatch, match=r"brms gives sigma its own formula \(standata X_sigma\); the AutoBayes model has none"):
        gates.Bundle("gauss_distributional_sigma", model=plain)
