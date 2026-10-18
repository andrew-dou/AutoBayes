"""Non-linear formulas (brms-parity C02): ``ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], data, priors,
nl=True)``, brms's ``bf(y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE)``. The expression grammar and its parser
(R's precedence; nothing is evaluated), the model's term model and refusals (D60-D63), the priors per non-linear
parameter, the graph against a density written by hand, ``value_point``, by-hand predictions (``nlpar=``, new rows, group
policies), ``describe()``, the target identity, fit plumbing with a stubbed sampler, fit records, prior-predictive draws,
brms's default priors (the tables below were printed by brms 2.23.0's ``default_prior()``, R 4.5.2, C collation, on
``FRAME``), the combinations with the other features refused (a multivariate formula, C01; a mixture, C05; ``gp()``, C03;
their priors, and ``nlpars`` in theirs) and C07's ``log_lik``, ``bayes_R2`` and ``loo`` against the graph's own density.
Nothing here samples; the brms comparison is the oracle cases ``gauss_nonlinear_exp`` and ``gauss_nonlinear_terms``."""
import ast
import copy
import dataclasses
import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import autobayes as ab
from autobayes import nonlinear as nlm
from autobayes.families import FAMILIES
from autobayes.formula import FormulaError
from autobayes.nonlinear import Binary, Call, Name, Negate, Number, parse_expression, parse_nonlinear, render
from autobayes.priors import PriorError

FRAME = pd.DataFrame({
    "y": [12.5, -30.2, 4.1, 55.0, -8.3, 21.9, -41.0, 3.3, 17.4, -12.6, 60.1, -2.2],     # mad(y) = 22.239: brms's scale 22.2
    "ypos": [0.82, 3.5, 1.27, 0.06, 12.4, 2.2, 0.95, 5.75, 0.31, 1.6, 8.9, 0.47],
    "x": [0.12, -1.3, 0.85, 1.7, -0.44, 0.03, -2.1, 0.9, 1.15, -0.6, 0.31, -0.05],
    "w": [1.4, 0.6, 1.9, 0.8, 1.1, 0.5, 1.6, 0.7, 1.3, 0.9, 1.2, 1.0],
    "xpos": [1.1, 0.45, 2.3, 0.9, 1.7, 0.62, 1.25, 3.1, 0.8, 1.9, 0.55, 1.4],
    "z": ["p", "q", "r", "p", "q", "r", "p", "q", "r", "p", "q", "r"],
    "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
    "h": ["h1", "h2", "h1", "h2", "h1", "h2", "h1", "h2", "h1", "h2", "h1", "h2"],
})
Z = {"z": "p"}
EXP = ["ypos ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"]
EXP_PRIORS = ab.Priors(nlpars={"a": {"b": ab.Normal(1, 1), "sd": ab.HalfNormal(1)}, "b": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1))
# a model with every construct: two covariates, a categorical term and a correlated slope in a's formula, no intercept in c's
RICH = ["y ~ a * exp(-b * x) + c / (1 + w^2)", "a ~ 1 + z + w + (1 + w | g)", "b ~ 1", "c ~ 0 + z + (1 | h)"]


def rich_priors(**changes):
    nl = {"a": {"b": {"Intercept": ab.Normal(1, 2), "z": ab.Normal(0, 1), "w": ab.StudentT(3, 0, 1)},
                "sd": {"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfStudentT(3, 0.5)}}},
          "b": {"b": ab.Normal(0, 0.5)},
          "c": {"b": ab.Normal(0, 1), "sd": ab.HalfCauchy(1)}}
    base = dict(nlpars=nl, cor={"g": ab.LKJ(2)}, sigma=ab.HalfNormal(10))
    base.update(changes)
    return ab.Priors(**base)


def rich(**changes):
    return ab.gaussian(RICH, FRAME, rich_priors(**changes), nl=True, categorical=Z)


def exp_model(frame=FRAME, priors=EXP_PRIORS):
    return ab.gaussian(EXP, frame, priors, nl=True)


# -- the grammar --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, tree", [
    ("-a^2", Negate(Binary("^", Name("a"), Number(2.0)))),                                   # R: ^ binds tighter than unary minus
    ("a^-b", Binary("^", Name("a"), Negate(Name("b")))),
    ("2^3^2", Binary("^", Number(2.0), Binary("^", Number(3.0), Number(2.0)))),             # right-associative
    ("-b * x", Binary("*", Negate(Name("b")), Name("x"))),                                  # unary minus before *
    ("a - b - c", Binary("-", Binary("-", Name("a"), Name("b")), Name("c"))),
    ("a / b * c", Binary("*", Binary("/", Name("a"), Name("b")), Name("c"))),
    ("a + b * c", Binary("+", Name("a"), Binary("*", Name("b"), Name("c")))),
    ("a * -b", Binary("*", Name("a"), Negate(Name("b")))),
    ("x**2", Binary("^", Name("x"), Number(2.0))),                                          # R reads ** as ^
    ("plogis(x)", Call("inv_logit", (Name("x"),))),                                          # R's name of inv_logit
    ("pow(x, .5e1)", Call("pow", (Name("x"), Number(5.0)))),
    ("((a))", Name("a")),
])
def test_precedence_and_associativity_are_rs(text, tree):
    assert parse_expression(text) == tree
    assert parse_expression(render(tree)) == tree


def test_numbers_follow_r_arithmetic():
    """The values R gives for the same text (``-2^2`` is -4, ``2^-1`` 0.5, ``2^3^2`` 512, ``1 / 2`` 0.5: every literal is a
    double, never Stan's integer division)."""
    for text, value in (("-2^2", -4.0), ("2^-1", 0.5), ("2^3^2", 512.0), ("8 / 2 / 2", 2.0), ("1 / 2", 0.5), ("-3 - -2", -1.0),
                        ("2 * 3 + 4", 10.0), ("exp(0) + log(1) + sqrt(4)", 3.0), ("inv_logit(0) * pow(2, 3)", 4.0)):
        assert float(nlm.to_numpy(parse_expression(text), {})) == value, text


@pytest.mark.parametrize("text", ["a * exp(-b * x) + c / (1 + w^2)", "-(a + b) * c", "a - (b - c)", "a / (b / c)", "(a^b)^c",
                                  "a^(b^c)", "-(-a)", "(-a)^2", "pow(a + b, -c)", "a - (-b)", "1e-05 * x + 0.25", "(a * b) * c"])
def test_render_round_trips(text):
    tree = parse_expression(text)
    assert parse_expression(render(tree)) == tree


def test_names_in_order_is_brms_covariate_order():
    assert nlm.names_in_order(parse_expression("c * w + a * exp(b * x) + log(w)")) == ("c", "w", "a", "b", "x")


@pytest.mark.parametrize("text, message", [
    ("a %% b", r"'%' at position 2 is outside the grammar"),
    ("a == b", r"'=' at position 2 is outside the grammar"),
    ("a > b", r"'>' at position 2"),
    ("a$b", r"'\$' at position 1"),
    ("a[1]", r"'\[' at position 1"),
    ("`a b`", r"'`' at position 0"),
    ("__import__('os')", r"\"'\" at position 11"),
    ("abs(a)", r"abs\(\) at position 0 is outside the grammar"),
    ("ifelse(a, b, c)", r"ifelse\(\) at position 0 is outside the grammar"),
    ("log(a, 2)", r"log\(\) takes 1 argument, got 2"),
    ("pow(a)", r"pow\(\) takes 2 arguments, got 1"),
    ("nlf(a)", r"brms's nlf\(\) is not supported here \(divergence D60\)"),
    ("a + lf(b)", r"brms's lf\(\) is not supported here \(divergence D60\)"),
    ("+a", r"unary plus is outside the grammar"),
    ("2L * a", r"R's integer, hexadecimal and complex literals"),
    ("0x10 * a", r"R's integer, hexadecimal and complex literals"),
    ("1e999 * a", r"the constant '1e999' is not a finite double"),
    ("a *", r"the expression ends early"),
    ("(a + b", r"expected '\)' at the end"),
    ("a b", r"unexpected token at position 2 \('b'\)"),
    ("", r"the non-linear expression is empty"),
    ("a + , b", r"expected a number, a name, a function or '\('"),
])
def test_grammar_refusals_name_the_construct(text, message):
    with pytest.raises(FormulaError, match=message):
        parse_expression(text)


def test_nothing_is_evaluated():
    """The module calls none of Python's code-evaluating builtins, and a name that is one is only a name: it resolves to a
    non-linear parameter or a column, or is refused as a missing column."""
    tree = ast.parse(Path(nlm.__file__).read_text(encoding="utf-8"))
    forbidden = {"eval", "exec", "compile", "__import__", "breakpoint"}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & forbidden
    assert parse_expression("eval * exec") == Binary("*", Name("eval"), Name("exec"))
    with pytest.raises(FormulaError, match=r"columns not in data: \['exec'\]"):
        ab.gaussian(["ypos ~ a * exec", "a ~ 1"], FRAME, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)), nl=True)


# -- the term model -----------------------------------------------------------------------------------------------------
def test_parameters_covariates_and_their_formulas():
    s = parse_nonlinear(RICH, FRAME, "gaussian", Z)
    assert s.response == "y" and s.nlpar_names == ("a", "b", "c") and s.covariates == ("x", "w")
    assert s.formula == RICH[0] and s.formulas == tuple(RICH) and s.rhs == "a * exp(-b * x) + c / (1 + w^2)"
    a = s.nlpar("a")
    assert a.response == "a" and a.intercept and a.terms == ("z", "w") and a.columns == ("z[q]", "z[r]", "w")
    assert [t.canonical for t in a.group_terms] == ["(1 + w | g)"] and s.nlpar("c").columns == ("z[p]", "z[q]", "z[r]")
    assert not s.nlpar("c").intercept and s.nlpar("b").terms == ()
    assert s.factors == ("g", "h") and s.correlated_factors == ("g",) and [c.name for c in s.all_categoricals] == ["z"]
    assert s.model_columns == ("x", "w", "z") and s.data_columns == ("y", "x", "w", "z", "g", "h")
    assert s.intercept is False and s.terms == () and s.group_terms == () and s.dpar_names == () and s.addition("weights") is None
    assert hash(s) == hash(parse_nonlinear(RICH, FRAME, "gaussian", Z))
    with pytest.raises(KeyError):
        s.nlpar("sigma")


@pytest.mark.parametrize("formulas, message", [
    ("ypos ~ a * exp(b * x)", r"nl=True takes a list of formulas"),
    (["ypos ~ a * exp(b * x)"], r"has no formula of a non-linear parameter"),
    ([["ypos ~ a", "a ~ 1"], ["y ~ b", "b ~ 1"]], r"a list of response formulas with nl=True: a non-linear model in a multivariate "
                                                  r"formula is not supported \(divergence D62"),
    ("mvbind(ypos, xpos) ~ a * exp(b * x)", r"'mvbind\(ypos, xpos\) ~ a \* exp\(b \* x\)': mvbind\(\) with nl=True: a non-linear "
                                            r"model in a multivariate formula is not supported \(divergence D62"),
    (["mvbind(ypos, xpos) ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], r"mvbind\(\) with nl=True: a non-linear model in a multivariate "
                                                                 r"formula is not supported \(divergence D62"),
    (["ypos ~ a * exp(b * x)", "nlf(a ~ c * d)", "b ~ 1"], r"brms's nlf\(\) is not supported here \(divergence D60\)"),
    (["ypos ~ a * exp(b * x)", "a ~ 1 + b", "b ~ 1"], r"'b' is a non-linear parameter; a non-linear parameter inside another's formula "
                                                      r"is not supported \(divergence D60\)"),
    (["ypos ~ a * exp(b * x)", "a ~ 1 + (1 | b)", "b ~ 1"], r"'b' is a non-linear parameter"),
    (["ypos ~ a * exp(b * x)", "a ~ 1", "b ~ 1", "c ~ 1"], r"'c' does not appear in the non-linear formula"),
    (["ypos ~ a * exp(b * x)", "a ~ 1", "a ~ x"], r"'a' has two formulas"),
    (["ypos ~ a * exp(b * x)", "a ~ 1", "b ~ 1", "sigma ~ x"], r"a formula of the distributional parameter 'sigma' with a non-linear "
                                                              r"formula is not supported in this slice \(divergence D62\)"),
    (["ypos ~ a * exp(mu * x)", "a ~ 1", "mu ~ 1"], r"distributional parameter 'mu'"),
    (["ypos ~ a * exp(b * x)", "ypos ~ 1", "b ~ 1"], r"'ypos' is the response"),
    (["ypos ~ a * exp(b * ypos)", "a ~ 1", "b ~ 1"], r"the response 'ypos' cannot be a covariate of its own non-linear predictor"),
    (["ypos ~ a * exp(b * x)", "a ~ 1 + ypos", "b ~ 1"], r"the response 'ypos' cannot be a predictor of a non-linear parameter"),
    (["ypos | weights(w) ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], r"response addition terms .* with a non-linear formula are not supported "
                                                              r"in this slice \(divergence D61\)"),
    (["ypos ~ a * exp(b * x)", "a ~ 1 + offset(w)", "b ~ 1"], r"'offset\(w\)' in a non-linear parameter's formula is not supported in "
                                                             r"this slice \(divergence D61\)"),
    (["ypos ~ a * exp(b * x)", "a ~ 1 + mo(w)", "b ~ 1"], r"the monotonic term 'mo\(w\)' in a non-linear parameter's formula"),
    (["ypos ~ a * exp(b * x)", "a ~ 1 + gp(w)", "b ~ 1"], r"the Gaussian process 'gp\(w\)' in a non-linear parameter's formula is not "
                                                         r"supported in this slice \(divergence D61\)"),
    (["ypos ~ a * exp(b * x)", "a ~ 1", "b ~ 1 + gp(w, k = 5, c = 1.5)"], r"the Gaussian process 'gp\(w, k = 5, c = 1.5\)' in a "
                                                                          r"non-linear parameter's formula is not supported"),
    (["ypos ~ a * gp(x)", "a ~ 1"], r"gp\(\) at position 4 is brms's Gaussian process \(C03\), a term of a linear formula, not a "
                                    r"function of the non-linear expression .* \(divergence D61\)"),
    (["ypos ~ a * exp(b * mo(w))", "a ~ 1", "b ~ 1"], r"mo\(\) at position 12 is brms's monotonic term \(B03\), a term of a linear "
                                                      r"formula"),
    (["ypos ~ a * abs(b * x)", "a ~ 1", "b ~ 1"], r"abs\(\) at position 4 is outside the grammar"),
    (["ypos ~ a * exp(b * x)", "a | weights(w) ~ 1", "b ~ 1"], r"a non-linear parameter's formula is '<parameter> ~ <terms>'"),
    (["ypos ~ a * exp(b * qq)", "a ~ 1", "b ~ 1"], r"columns not in data: \['qq'\] \(\['qq'\] appear in the non-linear formula"),
    (["ypos ~ a * exp(b * z)", "a ~ 1", "b ~ 1"], r"covariate 'z' of the non-linear formula has dtype \w+; a covariate is a numeric"),
])
def test_refusals_name_the_construct(formulas, message):
    with pytest.raises(FormulaError, match=message):
        parse_nonlinear(formulas, FRAME, "gaussian")


def test_declared_categorical_columns_and_covariates():
    with pytest.raises(FormulaError, match=r"'x' is declared categorical, but it is a covariate of the non-linear formula"):
        parse_nonlinear(EXP, FRAME, "gaussian", {"x": 0.12})
    with pytest.raises(FormulaError, match=r"\['h'\] declared categorical, but no non-linear parameter's formula names it"):
        parse_nonlinear(EXP, FRAME, "gaussian", {"h": "h1"})
    s = parse_nonlinear(["ypos ~ a + b * x", "a ~ 1 + z", "b ~ 0 + z"], FRAME, "gaussian", Z)
    assert s.nlpar("a").columns == ("z[q]", "z[r]") and s.nlpar("b").columns == ("z[p]", "z[q]", "z[r]")   # 0 + z: every level
    with pytest.raises(FormulaError, match=r"missing values in columns \['x'\]"):
        parse_nonlinear(EXP, FRAME.assign(x=[np.nan] + list(FRAME.x[1:])), "gaussian")
    with pytest.raises(FormulaError, match=r"covariate 'x' has non-finite values"):
        parse_nonlinear(EXP, FRAME.assign(x=[np.inf] + list(FRAME.x[1:])), "gaussian")


def test_nl_options_and_families():
    with pytest.raises(FormulaError, match=r"loop=False is not supported \(divergence D63\)"):
        ab.gaussian(EXP, FRAME, EXP_PRIORS, nl=True, loop=False)
    with pytest.raises(ValueError, match=r"loop is brms's option of a non-linear formula; it applies with nl=True only"):
        ab.gaussian("ypos ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)), loop=False)
    with pytest.raises(FormulaError, match=r"non-linear formulas are verified for the gaussian family only; the poisson family is not "
                                           r"offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel(EXP, FRAME, EXP_PRIORS, family="poisson")
    with pytest.raises(FormulaError, match=r"the poisson family is not offered with nl=True \(divergence D62\)"):
        ab.default_priors(EXP, FRAME, family="poisson", nl=True)
    with pytest.raises(FormulaError, match=r"If 'a' is a non-linear parameter of the first formula, build the model with nl=True"):
        ab.gaussian(EXP, FRAME, EXP_PRIORS)                            # nl=True forgotten: brms's 'Did you forget to set nl = TRUE?'
    assert {name for name, spec in FAMILIES.items() if spec.nonlinear_likelihood is not None} == {"gaussian"}
    m = exp_model()
    assert isinstance(m, ab.NonlinearModel) and isinstance(m, ab.Model) and m.bambi_model is None and m.effective_center_predictors is False
    assert ab.gaussian(EXP, FRAME, EXP_PRIORS, nl=True, center_predictors=False).target_identity == m.target_identity   # never centred


# -- priors -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("changes, message", [
    ({"nlpars": None}, r"the non-linear parameters \['a', 'b', 'c'\] need priors: pass nlpars="),
    ({"nlpars": {"a": {"b": ab.Normal(0, 1)}}}, r"nlpars must cover exactly the non-linear parameters \['a', 'b', 'c'\]; missing \['b', 'c'\]"),
    ({"b": ab.Normal(0, 1)}, r"a non-linear model has no b prior of its own"),
    ({"intercept": ab.Normal(0, 1)}, r"a non-linear model has no intercept prior of its own"),
    ({"sd": ab.HalfNormal(1)}, r"a non-linear model has no sd prior of its own"),
    ({"sigma": None}, r"the Gaussian family needs a prior on the residual SD"),
    ({"shape": ab.Gamma(2, 1)}, r"the gaussian family has no shape parameter"),
    ({"sigma": {"intercept": ab.Normal(0, 1)}}, r"prior for 'sigma' is a mapping, the form of a distributional formula's priors"),
    ({"cor": None}, r"the correlated group term\(s\) on \['g'\] need an LKJ prior"),
    ({"cor": {"h": ab.LKJ(2)}}, r"cor must cover exactly the factors of the correlated group terms \['g'\]"),
])
def test_prior_refusals(changes, message):
    with pytest.raises(PriorError, match=message):
        rich(**changes)


@pytest.mark.parametrize("entry, message", [
    ({"b": ab.Normal(0, 1), "sd": {"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfNormal(1)}}, "cor": ab.LKJ(2)},
     r"unexpected \['cor'\] \(the LKJ prior of a correlated term is Priors\(cor=\{factor: ab.LKJ\(eta\)\}\)"),
    ({"sd": {"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfNormal(1)}}}, r"'a ~ 1 \+ z \+ w \+ \(1 \+ w \| g\)' has coefficients"),
    ({"b": ab.Normal(0, 1)}, r"whose SDs need priors: pass nlpars=\{'a': \{..., 'sd': ...\}\}"),
    ({"b": {"z": ab.Normal(0, 1), "w": ab.Normal(0, 1)}, "sd": {"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfNormal(1)}}},
     r"must cover its intercept"),
    ({"b": {"Intercept": ab.Normal(0, 1), "z": ab.Normal(0, 1)}, "sd": {"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfNormal(1)}}},
     r"non-linear parameter 'a' \('a ~ 1 \+ z \+ w \+ \(1 \+ w \| g\)'\): coefficient priors must cover exactly the predictor terms"),
    ({"b": ab.HalfNormal(1), "sd": {"g": {"Intercept": ab.HalfNormal(1), "w": ab.HalfNormal(1)}}},
     r"non-linear parameter 'a' .*: prior for 'intercept' must be a location prior"),
    ({"b": ab.Normal(0, 1), "sd": ab.HalfNormal(1)}, r"sd for 'g' must be a mapping over its coefficients"),
    ({"b": ab.Normal(0, 1), "sd": {"g": {"Intercept": ab.Normal(0, 1), "w": ab.HalfNormal(1)}}},
     r"prior for 'sd:g' must be a positive-support prior"),
])
def test_priors_of_one_parameter(entry, message):
    nl = {"a": entry, "b": {"b": ab.Normal(0, 0.5)}, "c": {"b": ab.Normal(0, 1), "sd": ab.HalfCauchy(1)}}
    with pytest.raises(PriorError, match=message):
        rich(nlpars=nl)


def test_prior_forms():
    m = rich()
    labels = dict(nlm.nonlinear_prior_labels(m.priors, m.spec, m.family_spec))
    assert list(labels) == ["a:b:Intercept", "a:b:z[q]", "a:b:z[r]", "a:b:w", "a:sd:g", "a:sd:w|g", "b:b:Intercept", "c:b:z[p]",
                            "c:b:z[q]", "c:b:z[r]", "c:sd:h", "cor:g", "sigma"]
    assert labels["a:b:Intercept"] == ab.Normal(1, 2) and labels["a:b:w"] == ab.StudentT(3, 0, 1) and labels["cor:g"] == ab.LKJ(2)
    with pytest.raises(PriorError, match=r"'c ~ 0 \+ z \+ \(1 \| h\)' has no intercept; omit 'Intercept'"):
        rich(nlpars={**m.priors.nlpars, "c": {"b": {"Intercept": ab.Normal(0, 1), "z": ab.Normal(0, 1)}, "sd": ab.HalfCauchy(1)}})
    with pytest.raises(PriorError, match=r"nlpars gives the priors of non-linear parameters, which a model has only with nl=True"):
        ab.gaussian("ypos ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                                                 nlpars={"a": {"b": ab.Normal(0, 1)}}))


# -- the graph ----------------------------------------------------------------------------------------------------------
def natural(seed=3):
    """Natural-scale values of every parameter of ``rich()``, for ``value_point``."""
    rng = np.random.default_rng(seed)
    r = 0.6
    return dict(nlpars={"a": {"b": rng.normal(size=4), "sd": {"g": {"Intercept": 0.8, "w": 0.4}},
                              "z": {"g": {"Intercept": rng.normal(size=4), "w": rng.normal(size=4)}},
                              "L": np.array([[1.0, 0.0], [r, np.sqrt(1 - r * r)]])},
                        "b": {"b": [0.3]},
                        "c": {"b": rng.normal(size=3), "sd": 1.3, "z": rng.normal(size=2)}},
                sigma=7.5)


def test_variables_and_names():
    m = rich()
    assert [v.name for v in m.pymc_model.value_vars] == [
        "a_Intercept", "a_z", "a_w", "a_1|g_sigma_log__", "a_w|g_sigma_log__", "a_1|g_offset", "a_w|g_offset", "a_L|g_cholesky_corr__",
        "b_Intercept", "c_z", "c_1|h_sigma_log__", "c_1|h_offset", "sigma_log__"]
    assert m.names == {"linpred": "mu", "mean": "mu", "y": "y", "sigma": "sigma", "mu": "mu", "nlpar:a": "a", "nlpar:b": "b", "nlpar:c": "c"}
    assert m.nlpar_names["a"] == {"value": "a", "b:Intercept": "a_Intercept", "b:z": "a_z", "b:w": "a_w"}
    assert m.nlpar_coefficients["a"] == (("Intercept", "a_Intercept", None), ("z[q]", "a_z", 0), ("z[r]", "a_z", 1), ("w", "a_w", None))
    assert m.nlpar_group_names[("a", "g", "w")] == {"sd": "a_w|g_sigma", "offset": "a_w|g_offset", "group_effect": "a_w|g",
                                                     "coordinate": "g__factor_dim"}
    assert m.nlpar_correlation_names == {("a", "g"): {"L": "a_L|g", "cor": "a_cor|g", "coordinate": "a_g__cor_dim",
                                                      "coordinate_T": "a_g__cor_dim_T"}}
    assert m.design.nlpars["a"].columns == ("Intercept", "z[q]", "z[r]", "w") and m.design.covariates.keys() == {"x", "w"}
    np.testing.assert_array_equal(m.design.nlpars["c"].X, (FRAME.z.to_numpy()[:, None] == np.array(["p", "q", "r"])).astype(float))
    assert m.group_coefficient_names == {} and m.coefficients == () and m.dpar_names == {}       # the linear views are empty
    with pytest.raises(FormulaError, match=r"generated variable/coordinate name collision: \['a_x'\]"):
        ab.gaussian(["ypos ~ a + a_x", "a ~ 1 + x", "a_x ~ 1"], FRAME,
                    ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}, "a_x": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)), nl=True)


def test_the_density_is_the_one_written_by_hand():
    """The graph's log density at a point equals the model written out with scipy: every prior, the standardised group
    effects, the Gaussian likelihood of the non-linear predictor, and the Jacobians of the log and cholesky_corr transforms
    (the LKJ density and Stan's cholesky_corr map are the library's, oracle-verified in A02)."""
    import pytensor
    import pytensor.tensor as pt

    from autobayes.correlation import cholesky_corr_constrain, lkj_cholesky_logp

    Lsym = pt.dmatrix("L")
    m, nat = rich(), natural()
    vp = m.value_point(**nat)
    a, c = nat["nlpars"]["a"], nat["nlpars"]["c"]
    L, ljac = cholesky_corr_constrain(vp["a_L|g_cholesky_corr__"], 2)
    np.testing.assert_allclose(L, a["L"], atol=1e-15)
    gi = np.searchsorted(["a", "b", "c", "d"], FRAME.g)
    hi = np.searchsorted(["h1", "h2"], FRAME.h)
    zq, zr = (FRAME.z == "q").to_numpy(float), (FRAME.z == "r").to_numpy(float)
    w, x, y = FRAME.w.to_numpy(), FRAME.x.to_numpy(), FRAME.y.to_numpy()
    sd0, sd1 = a["sd"]["g"]["Intercept"], a["sd"]["g"]["w"]
    z0, z1 = a["z"]["g"]["Intercept"], a["z"]["g"]["w"]
    r0, r1 = sd0 * z0, sd1 * (L[1, 0] * z0 + L[1, 1] * z1)                     # brms's scale_r_cor
    av = a["b"][0] + zq * a["b"][1] + zr * a["b"][2] + w * a["b"][3] + r0[gi] + r1[gi] * w
    bv = nat["nlpars"]["b"]["b"][0]
    cv = (FRAME.z == "p").to_numpy(float) * c["b"][0] + zq * c["b"][1] + zr * c["b"][2] + (c["sd"] * c["z"])[hi]
    mu = av * np.exp(-bv * x) + cv / (1 + w ** 2)
    lprior = (stats.norm.logpdf(a["b"][0], 1, 2) + stats.norm.logpdf(a["b"][1:3], 0, 1).sum() + stats.t.logpdf(a["b"][3], 3, 0, 1)
              + np.log(2) + stats.norm.logpdf(sd0, 0, 1) + np.log(2) + stats.t.logpdf(sd1, 3, 0, 0.5)
              + float(pytensor.function([Lsym], lkj_cholesky_logp(Lsym, 2.0, 2))(a["L"]))
              + stats.norm.logpdf(bv, 0, 0.5) + stats.norm.logpdf(c["b"], 0, 1).sum() + stats.halfcauchy.logpdf(c["sd"], 0, 1)
              + np.log(2) + stats.norm.logpdf(nat["sigma"], 0, 10))
    rest = stats.norm.logpdf(np.r_[z0, z1, c["z"]]).sum() + stats.norm.logpdf(y, mu, nat["sigma"]).sum()
    jacobian = np.log([sd0, sd1, c["sd"], nat["sigma"]]).sum() + ljac
    # PyMC adds the log 2 of HalfStudentT and HalfCauchy in single precision (1.9e-9 each; docs/default-priors.md)
    assert abs(float(m.pymc_model.compile_logp(jacobian=False)(vp)) - (lprior + rest)) < 1e-8
    assert abs(float(m.pymc_model.compile_logp()(vp)) - (lprior + rest + jacobian)) < 1e-8
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"], pmm["a"], pmm["c"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    got_mu, got_a, got_c = fn(vp)
    np.testing.assert_allclose(got_mu, mu, rtol=1e-13)
    np.testing.assert_allclose(got_a, av, rtol=1e-13)
    np.testing.assert_allclose(got_c, cv, rtol=1e-13)


def test_value_point_refusals():
    m = rich()
    nat = natural()
    with pytest.raises(ValueError, match=r"nlpars must be a mapping over exactly the non-linear parameters \['a', 'b', 'c'\]"):
        m.value_point(nlpars={"a": nat["nlpars"]["a"]}, sigma=1.0)
    with pytest.raises(ValueError, match=r"nlpars\['b'\]\['b'\] must have 1 finite entries, one per design column \['Intercept'\]"):
        m.value_point(nlpars={**nat["nlpars"], "b": {"b": [0.1, 0.2]}}, sigma=1.0)
    with pytest.raises(ValueError, match=r"'b ~ 1' has no group-level term; sd, z and L do not apply to 'b'"):
        m.value_point(nlpars={**nat["nlpars"], "b": {"b": [0.1], "sd": 1.0}}, sigma=1.0)
    with pytest.raises(ValueError, match=r"L is required for the correlated term\(s\) of 'a'"):
        m.value_point(nlpars={**nat["nlpars"], "a": {k: v for k, v in nat["nlpars"]["a"].items() if k != "L"}}, sigma=1.0)
    with pytest.raises(ValueError, match=r"sd of 'g' in 'a' must be a mapping over its coefficients"):
        m.value_point(nlpars={**nat["nlpars"], "a": {**nat["nlpars"]["a"], "sd": 0.5}}, sigma=1.0)
    with pytest.raises(ValueError, match=r"sigma is required for the Gaussian family"):
        m.value_point(nlpars=nat["nlpars"])
    with pytest.raises(ValueError, match=r"sd:h of 'c' must be a finite positive scalar"):
        m.value_point(nlpars={**nat["nlpars"], "c": {**nat["nlpars"]["c"], "sd": -1.0}}, sigma=1.0)
    with pytest.raises(ValueError, match=r"unexpected \['b'\]"):
        m.value_point(nlpars=nat["nlpars"], sigma=1.0, b=[1.0])


# -- predictions ----------------------------------------------------------------------------------------------------------
def draws(m, seed=0, chains=2, n=3):
    """Synthetic posterior draws under the model's names (a correlated term's ``L`` a valid Cholesky factor)."""
    rng = np.random.default_rng(seed)
    data, coords = {}, {"chain": np.arange(chains), "draw": np.arange(n)}
    for d in m.spec.nlpars:
        roles = m.nlpar_names[d.response]
        if d.intercept:
            data[roles["b:Intercept"]] = (("chain", "draw"), rng.normal(size=(chains, n)))
        for t in d.pop_terms:
            var = roles[f"b:{t.name}"]
            if t.is_vector:
                data[var] = (("chain", "draw", f"{var}_dim"), rng.normal(size=(chains, n, len(t.levels))))
                coords[f"{var}_dim"] = list(t.levels)
            else:
                data[var] = (("chain", "draw"), 0.3 * rng.normal(size=(chains, n)))
    for (name, g, c), r in m.nlpar_group_names.items():
        levels = m.design.nlpars[name].group_blocks[(g, c)].levels
        data[r["sd"]] = (("chain", "draw"), rng.uniform(0.2, 1.0, (chains, n)))
        data[r["group_effect"]] = (("chain", "draw", r["coordinate"]), rng.normal(size=(chains, n, len(levels))))
        coords[r["coordinate"]] = list(levels)
    for (name, g), r in m.nlpar_correlation_names.items():
        rho = rng.uniform(-0.8, 0.8, (chains, n))
        L = np.zeros((chains, n, 2, 2))
        L[..., 0, 0], L[..., 1, 0], L[..., 1, 1] = 1.0, rho, np.sqrt(1 - rho ** 2)
        data[r["L"]] = (("chain", "draw", r["coordinate"], r["coordinate_T"]), L)
        coords[r["coordinate"]] = coords[r["coordinate_T"]] = ["Intercept", "w"]
    data["sigma"] = (("chain", "draw"), rng.uniform(1.0, 2.0, (chains, n)))
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def by_hand(m, post, frame, include=True):
    """The parameters' values and the non-linear predictor of ``rich()`` from the draws, written out."""
    p = {k: post[k].to_numpy() for k in post.data_vars}
    zq, zr, zp = ((frame.z == lvl).to_numpy(float) for lvl in ("q", "r", "p"))
    w, x = frame.w.to_numpy(float), frame.x.to_numpy(float)
    a = (p["a_Intercept"][..., None] + p["a_z"][..., 0, None] * zq + p["a_z"][..., 1, None] * zr + p["a_w"][..., None] * w)
    c = p["c_z"][..., 0, None] * zp + p["c_z"][..., 1, None] * zq + p["c_z"][..., 2, None] * zr
    if include:
        gi, hi = np.searchsorted(["a", "b", "c", "d"], frame.g), np.searchsorted(["h1", "h2"], frame.h)
        a = a + p["a_1|g"][..., gi] + p["a_w|g"][..., gi] * w
        c = c + p["c_1|h"][..., hi]
    b = np.repeat(p["b_Intercept"][..., None], len(frame), axis=-1)
    return {"a": a, "b": b, "c": c, "mu": a * np.exp(-b * x) + c / (1 + w ** 2)}


NEW = pd.DataFrame({"x": [0.5, -1.0, 2.0], "w": [0.7, 1.5, 1.1], "z": ["r", "p", "q"], "g": ["b", "d", "a"], "h": ["h2", "h1", "h2"]})


def test_predictions_are_the_model_written_out():
    m = rich()
    post = draws(m)
    for frame, newdata in ((FRAME, None), (NEW, NEW)):
        for include in (True, False):
            want = by_hand(m, post.posterior, frame, include)
            lin = m.predict(post, newdata=newdata, kind="linear", include_group_effects=include)
            assert lin.attrs["prediction_scale"] == "linear" and lin.attrs["link"] == "identity"
            np.testing.assert_allclose(lin["eta"].to_numpy(), want["mu"], rtol=1e-13)
            np.testing.assert_allclose(m.predict(post, newdata=newdata, kind="mean", include_group_effects=include)["mu"].to_numpy(),
                                       want["mu"], rtol=1e-13)
            for name in ("a", "b", "c"):
                got = m.predict(post, newdata=newdata, kind="linear", include_group_effects=include, nlpar=name)
                assert got.attrs["nlpar"] == name
                np.testing.assert_allclose(got["eta"].to_numpy(), want[name], rtol=1e-13)
                np.testing.assert_allclose(m.predict(post, newdata=newdata, kind="mean", include_group_effects=include, nlpar=name)[name]
                                           .to_numpy(), want[name], rtol=1e-13)
    y = m.predict(post, kind="response", seed=4)
    assert y["y"].shape == (2, 3, 12) and np.isfinite(y["y"]).all()
    np.testing.assert_array_equal(y["y"].to_numpy(), ab.predict(m, post, kind="response", seed=4)["y"].to_numpy())
    assert y.attrs["new_groups"] == "in_sample" and y.attrs["include_group_effects"] == 1


def test_unseen_levels_follow_the_policies():
    m = rich()
    post = draws(m)
    unseen = NEW.assign(g=["b", "zz", "zz"], h=["h9", "h1", "h2"])
    with pytest.raises(FormulaError, match=r"newdata has levels of 'g' not seen in training: \['zz'\]"):
        m.predict(post, newdata=unseen, kind="mean")
    for policy in ("population", "resample_observed"):
        out = m.predict(post, newdata=unseen, kind="mean", new_groups=policy, seed=1)
        assert out["mu"].shape == (2, 3, 3) and np.isfinite(out["mu"]).all()
        assert json.loads(out.attrs["new_levels_by_factor_json"]) == {"g": ["zz"], "h": ["h9"]}
        np.testing.assert_array_equal(out["mu"].to_numpy(), m.predict(post, newdata=unseen, kind="mean", new_groups=policy, seed=1)["mu"]
                                      .to_numpy())
    # resample_observed reuses one observed level's whole effect vector: rows 1 and 2 share 'zz', so their a-effects agree
    a = m.predict(post, newdata=unseen.assign(w=[1.0, 1.0, 1.0], z=["p", "p", "p"]), kind="linear", nlpar="a",
                  new_groups="resample_observed", seed=2)["eta"].to_numpy()
    np.testing.assert_allclose(a[..., 1], a[..., 2], rtol=1e-14)
    with pytest.raises(ValueError, match=r"new_groups must cover exactly the grouping factors \['g', 'h'\]"):
        m.predict(post, newdata=unseen, kind="mean", new_groups={"g": "population"})


def test_prediction_refusals():
    m = rich()
    post = draws(m)
    with pytest.raises(ValueError, match=r"nlpar='a' takes kind='linear' or kind='mean'"):
        m.predict(post, kind="response", nlpar="a")
    with pytest.raises(ValueError, match=r"nlpar must name a non-linear parameter, \['a', 'b', 'c'\]; got 'sigma'"):
        m.predict(post, kind="linear", nlpar="sigma")
    with pytest.raises(ValueError, match=r"a non-linear model has no distributional formula; dpar does not apply"):
        m.predict(post, kind="linear", dpar="sigma")
    with pytest.raises(FormulaError, match=r"newdata lacks columns \['x'\] \(the covariates of the non-linear formula are needed"):
        m.predict(post, newdata=NEW.drop(columns="x"), kind="mean")
    with pytest.raises(FormulaError, match=r"newdata has levels of the categorical predictor 'z' that were not in the fitted rows"):
        m.predict(post, newdata=NEW.assign(z=["s", "p", "q"]), kind="mean")
    with pytest.raises(ValueError, match=r"new_groups must stay at its default when group effects are excluded"):
        m.predict(post, kind="mean", include_group_effects=False, new_groups="population")
    linear = ab.gaussian("ypos ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"nlpar applies to a non-linear model \(nl=True"):
        linear.predict(post, kind="linear", nlpar="a")
    logm = ab.gaussian(["ypos ~ log(a) * x", "a ~ 1"], FRAME, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)),
                       nl=True)
    neg = az.InferenceData(posterior=xr.Dataset({"a_Intercept": (("chain", "draw"), [[-1.0]]), "sigma": (("chain", "draw"), [[1.0]])}))
    with pytest.raises(ValueError, match=r"the non-linear predictor contains nonfinite draws"):
        logm.predict(neg, kind="linear")


# -- describe, identity, records, prior predictive -------------------------------------------------------------------------
def test_describe_writes_the_model_out():
    text = rich().describe()
    for line in ("AutoBayes M0 model with a non-linear formula: 3 non-linear parameter(s) (a, b, c), covariate(s) x (brms's C_1), "
                 "w (brms's C_2)",
                 "  mu_i = a_i * exp(-b_i * x_i) + c_i / (1 + w_i^2)",
                 "  a_i = b_a[Intercept] + 1[z_i = q] * b_a[zq] + 1[z_i = r] * b_a[zr] + w_i * b_a[w] + u_a_g[g_i] + v_a_w|g[g_i] * w_i",
                 "  c_i = 1[z_i = p] * b_c[zp] + 1[z_i = q] * b_c[zq] + 1[z_i = r] * b_c[zr] + u_c_h[h_i]",
                 "  b_a[w] ~ StudentT(nu=3, mu=0, sigma=1)", "  sd_a_w|g ~ HalfStudentT(nu=3, sigma=0.5)", "  y_i ~ Normal(mu_i, sigma)",
                 "  a_L|g_cholesky_corr__    L_a_g on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)",
                 "  c_z                      b_c[zp, zq, zr] (vector; dims c_z_dim, coordinates p, q, r)"):
        assert line in text.split("\n"), line
    assert "Cor of every correlated term on g ~ LKJ(eta=2)" in text and "never centred" in text
    assert "(R's plogis)" in ab.gaussian(["ypos ~ plogis(a) * x", "a ~ 1"], FRAME,
                                         ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)), nl=True).describe()


def test_the_target_identity_holds_the_formulas_and_priors():
    base = rich().target_identity
    assert rich().target_identity == base
    assert rich(sigma=ab.HalfNormal(11)).target_identity != base
    assert rich(cor={"g": ab.LKJ(3)}).target_identity != base
    other = ab.gaussian(["y ~ a * exp(-b * x) + c / (1 + w^3)", *RICH[1:]], FRAME, rich_priors(), nl=True, categorical=Z)
    assert other.target_identity != base
    assert exp_model().target_identity != exp_model(priors=dataclasses.replace(EXP_PRIORS, sigma=ab.HalfNormal(2))).target_identity
    plogis = ab.gaussian(["ypos ~ plogis(a) * x", "a ~ 1"], FRAME, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)), nl=True)
    inv = ab.gaussian(["ypos ~ inv_logit(a) * x", "a ~ 1"], FRAME, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)), nl=True)
    assert plogis.target_identity == inv.target_identity                  # the same model, two spellings


@pytest.mark.nosampling
def test_fit_samples_the_graph_and_records_it(monkeypatch):
    """``ab.fit`` hands a non-linear model to its own ``sample_posterior`` (PyMC's ``sample`` on the graph, stubbed here: nothing
    samples); the fit is bound to the model, predicts, and makes a fit record that ``compare_records`` accepts."""
    import pymc as pm

    import autobayes.session as session

    m = rich()
    fake = draws(m)
    calls = {}

    def sample(**kwargs):
        calls.update(kwargs)
        out = copy.deepcopy(fake)
        out.posterior["a_1|g_offset"] = out.posterior["a_1|g"]          # dropped from the posterior, as Bambi drops offsets
        return out

    monkeypatch.setattr(pm, "sample", sample)
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = ab.fit(m, chains=2, draws=3, tune=0, seed=5)
    assert calls["model"] is m.pymc_model and calls["nuts_sampler"] == "pymc" and calls["random_seed"] == 5
    assert "mu" not in calls["var_names"] and "a" not in calls["var_names"] and "a_1|g" in calls["var_names"]
    assert "a_1|g_offset" not in fit.idata.posterior and fit.target_identity == m.target_identity
    assert m.predict(fit, kind="mean").attrs["fit_binding_status"] == "matched"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # synthetic draws: ArviZ's diagnostics divide by zero
        rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["formula"] == RICH and rec["model"]["fit_binding_status"] == "matched"
    assert rec["model"]["nonlinear"]["covariates"] == ["x", "w"] and rec["model"]["priors"]["a:sd:w|g"] == "HalfStudentT(nu=3, sigma=0.5)"
    assert rec["model"]["nonlinear"]["nlpars"]["a"]["groups"][0]["correlation_names"]["L"] == "a_L|g"
    assert set(rec["posterior"]) == {"a_Intercept", "a_z[q]", "a_z[r]", "a_w", "b_Intercept", "c_z[p]", "c_z[q]", "c_z[r]",
                                     "a_1|g_sigma", "a_w|g_sigma", "c_1|h_sigma", "sigma"}
    same = ab.compare_records(rec, json.loads(json.dumps(rec)))      # synthetic draws have no MCSE: agreement is not the point
    assert same["identity_status"] == "matched" and set(same["rows"]) == set(rec["posterior"])
    other = json.loads(json.dumps(rec))
    other["model"]["formula"][0] = "y ~ a"
    with pytest.raises(ValueError, match="different models"):
        ab.compare_records(rec, other)
    with pytest.raises(ValueError, match="different models or data"):
        exp_model().predict(fit, kind="mean")


def test_prior_predictive_draws_and_flat_priors():
    pp = rich().prior_predictive(draws=15, seed=2)
    assert pp.prior_predictive["y"].shape == (1, 15, 12) and pp.prior["a_L|g"].shape == (1, 15, 2, 2)
    assert np.isfinite(pp.prior["mu"]).all()
    flat = ab.default_priors(EXP, FRAME, nl=True)
    with pytest.raises(ValueError, match=r"the prior of \['a:b:Intercept', 'b:b:Intercept'\] is flat"):
        exp_model(priors=flat).prior_predictive(draws=5, seed=1)


# -- brms's default priors -----------------------------------------------------------------------------------------------
# (formulas, keyword arguments, brms's rows as (class, coef, group, nlpar, prior, lb)); every row's ub, resp, dpar and tag are
# empty and its source "default"
BRMS = {
    "exp, the response's scale 22.2": (["y ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], {}, [
        ("sigma", "", "", "", "student_t(3, 0, 22.2)", "0"), ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""),
        ("sd", "", "", "a", "student_t(3, 0, 22.2)", "0"), ("sd", "", "g", "a", "", ""), ("sd", "Intercept", "g", "a", "", ""),
        ("b", "", "", "b", "", ""), ("b", "Intercept", "", "b", "", "")]),
    "exp, scale 2.5": (EXP, {}, [
        ("sigma", "", "", "", "student_t(3, 0, 2.5)", "0"), ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""),
        ("sd", "", "", "a", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g", "a", "", ""), ("sd", "Intercept", "g", "a", "", ""),
        ("b", "", "", "b", "", ""), ("b", "Intercept", "", "b", "", "")]),
    "categorical, correlated slope, no intercept": (["y ~ a + b * x", "a ~ 1 + z + w + (1 + w | g)", "b ~ 0 + z"], {"categorical": Z}, [
        ("cor", "", "", "", "lkj(1)", ""), ("cor", "", "g", "", "", ""), ("sigma", "", "", "", "student_t(3, 0, 22.2)", "0"),
        ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""), ("b", "w", "", "a", "", ""), ("b", "zq", "", "a", "", ""),
        ("b", "zr", "", "a", "", ""), ("sd", "", "", "a", "student_t(3, 0, 22.2)", "0"), ("sd", "", "g", "a", "", ""),
        ("sd", "Intercept", "g", "a", "", ""), ("sd", "w", "g", "a", "", ""), ("b", "", "", "b", "", ""), ("b", "zp", "", "b", "", ""),
        ("b", "zq", "", "b", "", ""), ("b", "zr", "", "b", "", "")]),
    "two parameters on g, an independent slope": (["y ~ a * inv_logit(c * x) + b", "a ~ 1 + w + (1 + w || g)", "b ~ 1 + (1 | g)",
                                                   "c ~ 1 + w + (1 | h)"], {}, [
        ("sigma", "", "", "", "student_t(3, 0, 22.2)", "0"), ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""),
        ("b", "w", "", "a", "", ""), ("sd", "", "", "a", "student_t(3, 0, 22.2)", "0"), ("sd", "", "g", "a", "", ""),
        ("sd", "Intercept", "g", "a", "", ""), ("sd", "w", "g", "a", "", ""), ("b", "", "", "b", "", ""), ("b", "Intercept", "", "b", "", ""),
        ("sd", "", "", "b", "student_t(3, 0, 22.2)", "0"), ("sd", "", "g", "b", "", ""), ("sd", "Intercept", "g", "b", "", ""),
        ("b", "", "", "c", "", ""), ("b", "Intercept", "", "c", "", ""), ("b", "w", "", "c", "", ""),
        ("sd", "", "", "c", "student_t(3, 0, 22.2)", "0"), ("sd", "", "h", "c", "", ""), ("sd", "Intercept", "h", "c", "", "")]),
    "interaction and transform": (["ypos ~ a / (1 + exp(-b * x))", "a ~ 1 + w * z + log(xpos)", "b ~ 1"], {"categorical": Z}, [
        ("sigma", "", "", "", "student_t(3, 0, 2.5)", "0"), ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""),
        ("b", "logxpos", "", "a", "", ""), ("b", "w", "", "a", "", ""), ("b", "w:zq", "", "a", "", ""), ("b", "w:zr", "", "a", "", ""),
        ("b", "zq", "", "a", "", ""), ("b", "zr", "", "a", "", ""), ("b", "", "", "b", "", ""), ("b", "Intercept", "", "b", "", "")]),
    "two correlated factors": (["y ~ a + b * x", "a ~ 1 + w + (1 + w | g)", "b ~ 1 + w + (1 + w | h)"], {}, [
        ("cor", "", "", "", "lkj(1)", ""), ("cor", "", "g", "", "", ""), ("cor", "", "h", "", "", ""),
        ("sigma", "", "", "", "student_t(3, 0, 22.2)", "0"), ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""),
        ("b", "w", "", "a", "", ""), ("sd", "", "", "a", "student_t(3, 0, 22.2)", "0"), ("sd", "", "g", "a", "", ""),
        ("sd", "Intercept", "g", "a", "", ""), ("sd", "w", "g", "a", "", ""), ("b", "", "", "b", "", ""), ("b", "Intercept", "", "b", "", ""),
        ("b", "w", "", "b", "", ""), ("sd", "", "", "b", "student_t(3, 0, 22.2)", "0"), ("sd", "", "h", "b", "", ""),
        ("sd", "Intercept", "h", "b", "", ""), ("sd", "w", "h", "b", "", "")]),
    "g:h and g/h": (["ypos ~ a * exp(b * x)", "a ~ 1 + (1 | g/h)", "b ~ 1 + (1 | g:h)"], {}, [
        ("sigma", "", "", "", "student_t(3, 0, 2.5)", "0"), ("b", "", "", "a", "", ""), ("b", "Intercept", "", "a", "", ""),
        ("sd", "", "", "a", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g", "a", "", ""), ("sd", "Intercept", "g", "a", "", ""),
        ("sd", "", "g:h", "a", "", ""), ("sd", "Intercept", "g:h", "a", "", ""), ("b", "", "", "b", "", ""),
        ("b", "Intercept", "", "b", "", ""), ("sd", "", "", "b", "student_t(3, 0, 2.5)", "0"), ("sd", "", "g:h", "b", "", ""),
        ("sd", "Intercept", "g:h", "b", "", "")]),
}


@pytest.mark.parametrize("label", sorted(BRMS))
def test_the_default_prior_table_is_brms(label):
    formula, kwargs, rows = BRMS[label]
    pri = ab.default_priors(formula, FRAME, nl=True, **kwargs)
    model = ab.gaussian(formula, FRAME, pri, nl=True, **kwargs)
    table = model.prior_table()
    assert [(r["class"], r["coef"], r["group"], r["nlpar"], r["prior"], r["lb"]) for r in table] == rows
    assert all(r["ub"] == r["resp"] == r["dpar"] == r["tag"] == "" and r["source"] == "default" for r in table)
    assert model.default_prior_labels == [label for label, _ in nlm.nonlinear_prior_labels(pri, model.spec, model.family_spec)]
    assert "(brms default: flat, improper)" in model.describe()


def test_the_default_priors_mirror_brms():
    pri = ab.default_priors(["y ~ a + b * x", "a ~ 1 + z + w + (1 + w | g)", "b ~ 0 + z"], FRAME, nl=True, categorical=Z)
    assert pri.nlpars == {"a": {"b": ab.Flat(), "sd": {"g": {"Intercept": ab.HalfStudentT(3, 22.2), "w": ab.HalfStudentT(3, 22.2)}}},
                          "b": {"b": ab.Flat()}}
    assert pri.cor == {"g": ab.LKJ(1)} and pri.sigma == ab.HalfStudentT(3, 22.2) and pri.intercept is None and pri.b is None
    assert pri.sigma.source == "default" and pri.nlpars["a"]["b"].source == "default"


# -- the oracle's extension to non-linear models fails on a corrupted fixture -----------------------------------------------
def _bundle(case="gauss_nonlinear_exp"):
    from tests.oracle import fixture as fx
    from tests.oracle import gates

    return fx, gates, gates.Bundle(case)


def _corrupted(fx, bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_a_corrupted_nonlinear_fixture():
    from tests.oracle.adapter import OracleMismatch

    fx, gates, b = _bundle()
    for gate in gates.ALL:
        gate(b)

    def covariate(raw):
        raw["standata"]["C_1"]["values"][0] += 1e-11
    with pytest.raises(OracleMismatch, match=r"covariate 'x' of the non-linear formula matches 0 brms covariates C_k"):
        _corrupted(fx, b, covariate)

    def design(raw):
        raw["standata"]["X_a"]["values"][3] += 1e-11
    with pytest.raises(OracleMismatch, match=r"a's design column 'Intercept' matches 0 brms X_a columns"):
        _corrupted(fx, b, design)

    def group(raw):
        raw["standata"]["Z_1_a_1"]["values"][5] = 2.0
    with pytest.raises(OracleMismatch, match=r"group-level coefficient 'Intercept' of 'g' in 'a': 0 brms group terms of a"):
        _corrupted(fx, b, group)

    def coefficient(raw):
        raw["points"][3]["pars"]["b_b"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 3: log density"):
        gates.check_logp(_corrupted(fx, b, coefficient))

    def gradient(raw):
        (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "b_b"]
        raw["points"][1]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 1: gradient of b_Intercept\[0\]"):
        gates.check_gradient(_corrupted(fx, b, gradient))

    def linpred(raw):
        raw["predictions"]["fitted"]["linpred"]["values"][7] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 7: the graph's non-linear predictor 'mu' differs from brms"):
        gates.check_derived(_corrupted(fx, b, linpred))
    with pytest.raises(OracleMismatch, match=r"fitted: linear predictor differs from brms posterior_linpred"):
        gates.check_predictions(_corrupted(fx, b, linpred))

    def effects(raw):
        raw["points"][2]["pars"]["r_1_a_1"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 2: the graph's values of the non-linear parameter 'a' differ from brms's linear "
                                             r"predictor of a"):
        gates.check_derived(_corrupted(fx, b, effects))

    def default(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "sd" and r["nlpar"] == "a" and r["group"] == ""]
        row["prior"] = "student_t(3, 0, 2.6)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(fx, b, default))

    def no_rows(raw):
        raw["default_prior"] = [r for r in raw["default_prior"] if r["nlpar"] != "b"]
    with pytest.raises(OracleMismatch, match=r"brms default_prior lacks the non-linear parameters' rows \[\('b', 'b'\)\]"):
        gates.check_default_prior(_corrupted(fx, b, no_rows))

    def unscoped(raw):
        for grp in raw["groups"]:
            del grp["nlpar"]
    with pytest.raises(OracleMismatch, match=r"brms's group table has no nlpar column"):
        _corrupted(fx, b, unscoped)

    linear = ab.gaussian("ypos ~ x + (1 | g)", b.data, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1),
                                                                sigma=ab.HalfNormal(1)))
    with pytest.raises(KeyError, match="'X'"):                        # brms has no X: a linear model cannot be matched
        gates.Bundle("gauss_nonlinear_exp", model=linear)
    other = ab.gaussian(["ypos ~ a * exp(b * w)", "a ~ 1 + (1 | g)", "b ~ 1"], b.data, EXP_PRIORS, nl=True)
    with pytest.raises(OracleMismatch, match=r"covariate 'w' of the non-linear formula matches 0 brms covariates"):
        gates.Bundle("gauss_nonlinear_exp", model=other)


def test_the_gates_see_a_corrupted_correlated_nonlinear_fixture():
    from tests.oracle.adapter import OracleMismatch

    fx, gates, b = _bundle("gauss_nonlinear_terms")

    def cor(raw):
        raw["points"][4]["pars"]["Cor_3"]["values"][1] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 4: the correlation matrix of 'h' in 'c' differs from brms Cor_3"):
        gates.check_derived(_corrupted(fx, b, cor))

    def cholesky(raw):
        (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "L_3"]
        raw["points"][0]["u"]["values"][i] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 0: Stan's cholesky_corr coordinate 0 of L_3"):
        gates.check_gradient(_corrupted(fx, b, cholesky))


# -- with the other features: multivariate models (C01), mixtures (C05), Gaussian processes (C03), post-processing (C07) ------
def test_nl_is_refused_with_a_multivariate_model_or_a_mixture():
    """brms takes both (``bf(mvbind(y, y2) ~ a * exp(b * x), ..., nl = TRUE)`` gives each response its own parameters,
    ``X_y_a``; ``mixture(gaussian, gaussian)`` puts the expression in every component's mean with shared parameters), and
    neither is verified here (divergence D62): refused by every entry point, naming the construct."""
    with pytest.raises(FormulaError, match=r"rescor= belongs to a multivariate model, 'mvbind\(y, y2\) ~ \.\.\.'; a non-linear model "
                                           r"in a multivariate formula is not supported \(docs/nonlinear.md, divergence D62\)"):
        ab.gaussian(EXP, FRAME, EXP_PRIORS, nl=True, rescor=False)
    with pytest.raises(FormulaError, match=r"mvbind\(\) with nl=True: a non-linear model in a multivariate formula is not supported"):
        ab.gaussian(["mvbind(ypos, xpos) ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], FRAME, EXP_PRIORS, nl=True)
    with pytest.raises(FormulaError, match=r"nl=True with a mixture is not supported \(docs/nonlinear.md, divergence D62\)"):
        ab.mixture(EXP, FRAME, EXP_PRIORS, families=["gaussian", "gaussian"], nl=True)
    with pytest.raises(FormulaError, match=r"families= and order= are a mixture's \(ab.mixture, docs/mixtures.md\); a non-linear model "
                                           r"in a mixture is not supported \(divergence D62\)"):
        ab.default_priors(EXP, FRAME, family="mixture", families=["gaussian", "gaussian"], nl=True)
    with pytest.raises(FormulaError, match=r"the mixture family is not offered with nl=True \(divergence D62\)"):
        ab.default_priors(EXP, FRAME, family="mixture", nl=True)
    with pytest.raises(FormulaError, match=r"rescor= belongs to a multivariate model"):
        ab.default_priors(EXP, FRAME, rescor=True, nl=True)
    with pytest.raises(FormulaError, match=r"mvbind\(\) with nl=True"):
        ab.default_priors(["mvbind(ypos, xpos) ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], FRAME, nl=True)
    with pytest.raises(ValueError, match=r"nu applies only to the student or frechet family, which is not offered with nl=True"):
        ab.default_priors(EXP, FRAME, nu=4, nl=True)


LINEAR = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
MIXTURE = ab.Priors(components=[LINEAR, LINEAR], theta=ab.Dirichlet(1))
RESPONSES = {"y": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(1)},
             "xpos": {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(1)}}
NLPARS = {"a": {"b": ab.Normal(0, 1)}}


@pytest.mark.parametrize("field, value, message", [
    ("sdgp", ab.HalfNormal(1), r"sdgp is the prior of a Gaussian process gp\(x\) \(docs/gaussian-processes.md\); a non-linear model takes "
                               r"no Gaussian process \(divergence D61\)"),
    ("lscale", ab.InverseGamma(3, 1), r"lscale is the prior of a Gaussian process gp\(x\)"),
    ("components", [LINEAR, LINEAR], r"components is the prior of a mixture \(ab.mixture, docs/mixtures.md\); a non-linear model in a "
                                     r"mixture is not supported \(divergence D62\)"),
    ("theta", ab.Dirichlet(1), r"theta is the prior of a mixture"),
    ("resp", RESPONSES, r"resp is the prior of a multivariate model \(docs/multivariate.md\); a non-linear model in a multivariate "
                        r"formula is not supported \(divergence D62\)"),
    ("rescor", ab.LKJ(1), r"rescor is the prior of a multivariate model"),
])
def test_a_nonlinear_model_refuses_the_priors_of_the_other_features(field, value, message):
    with pytest.raises(PriorError, match=message):
        ab.gaussian(EXP, FRAME, dataclasses.replace(EXP_PRIORS, **{field: value}), nl=True)


def test_the_other_models_refuse_nlpars():
    linear = r"nlpars gives the priors of non-linear parameters, which a model has only with nl=True \(docs/nonlinear.md\); this formula is linear"
    with pytest.raises(PriorError, match=linear):
        ab.gaussian("y ~ x", FRAME, dataclasses.replace(LINEAR, nlpars=NLPARS))
    mix = dict(families=["gaussian", "gaussian"])
    ab.mixture("y ~ x", FRAME, MIXTURE, **mix)
    with pytest.raises(PriorError, match=linear):
        ab.mixture("y ~ x", FRAME, dataclasses.replace(MIXTURE, nlpars=NLPARS), **mix)
    with pytest.raises(PriorError, match=r"component 2 \(brms's mu2 and sigma2\): " + linear):
        ab.mixture("y ~ x", FRAME, dataclasses.replace(MIXTURE, components=[LINEAR, dataclasses.replace(LINEAR, nlpars=NLPARS)]), **mix)
    mv = ab.Priors(resp=RESPONSES, rescor=ab.LKJ(1))
    ab.gaussian("mvbind(y, xpos) ~ x", FRAME, mv, rescor=True)
    with pytest.raises(PriorError, match=r"nlpars gives the priors of non-linear parameters \(ab.gaussian\(..., nl=True\), "
                                         r"docs/nonlinear.md\); a multivariate model takes none: a non-linear model in a multivariate "
                                         r"formula is not supported \(divergence D62\)"):
        ab.gaussian("mvbind(y, xpos) ~ x", FRAME, dataclasses.replace(mv, nlpars=NLPARS), rescor=True)
    with pytest.raises(PriorError, match=r"response 'y': the priors of the response 'y' take the classes .*; unexpected \['nlpars'\]"):
        ab.gaussian("mvbind(y, xpos) ~ x", FRAME, dataclasses.replace(mv, resp={**RESPONSES, "y": {**RESPONSES["y"], "nlpars": NLPARS}}),
                    rescor=True)
    with pytest.raises(PriorError, match=r"response 'y': " + linear):
        ab.gaussian("mvbind(y, xpos) ~ x", FRAME, dataclasses.replace(mv, resp={**RESPONSES, "y": ab.Priors(**RESPONSES["y"], nlpars=NLPARS)}),
                    rescor=True)


@pytest.mark.parametrize("case", ["gauss_nonlinear_exp", "gauss_nonlinear_terms"])
def test_log_lik_bayes_R2_and_loo_are_the_graphs(case):
    """C07's post-processing of a non-linear model, at the frozen brms points held as draws: ``log_lik`` row by row is the
    graph's own elementwise log density of the response (the per-row terms whose sum gate G2 checks against brms's
    ``log_prob``), as C07 checks it for every other model; ``bayes_R2`` is ``var(mu) / (var(mu) + var(y - mu))`` with the
    graph's ``mu`` (brms's ``posterior_epred`` of a Gaussian non-linear model); ``loo`` is PSIS-LOO of that log-likelihood."""
    from autobayes.post_processing import loo_from_log_lik
    from tests.oracle import adapter
    from tests.oracle.tolerances import rel_err

    _, _, b = _bundle(case)
    pmm = b.model.pymc_model
    pointwise = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, point_fn=True,
                               on_unused_input="ignore")
    mu_fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[b.model.names["mu"]]]), inputs=pmm.value_vars, point_fn=True,
                           on_unused_input="ignore")
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    ll, r2 = ab.log_lik(b.model, idata), ab.bayes_R2(b.model, idata)
    assert ll.dims == ("chain", "draw", "__obs__") and ll.shape == (1, len(b.fixture.points), b.model.design.n)
    assert r2.dims == ("chain", "draw") and r2.shape == (1, len(b.fixture.points))
    y = np.asarray(b.model.design.y, dtype=float)
    for k, _, vp in b.points():
        assert rel_err(ll.to_numpy()[0, k], np.asarray(pointwise(vp)[0], dtype=float).ravel()) <= 1e-12, k
        (mu,) = mu_fn(vp)
        mu = np.asarray(mu, dtype=float)
        assert rel_err(r2.to_numpy()[0, k], np.var(mu, ddof=1) / (np.var(mu, ddof=1) + np.var(y - mu, ddof=1))) <= 1e-12, k
    res, reference = ab.loo(b.model, idata), loo_from_log_lik(ll)
    np.testing.assert_array_equal(res.pointwise.to_numpy(), reference.pointwise.to_numpy())
    np.testing.assert_array_equal(res.estimates.to_numpy(), reference.estimates.to_numpy())
