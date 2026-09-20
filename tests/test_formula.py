import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from tests.conftest import PRIORS, simulate


@pytest.mark.parametrize("formula,needle", [
    ("y ~ x + (1 + x | g)", "varying slopes"),
    ("y ~ x + (x | g)", "varying slopes"),
    ("y ~ x + (1 | g/h)", "nested"),
    ("y ~ x + (1 | g:h)", "nested"),
    ("y ~ x + (1 | g) + (1 | h)", "more than one grouping"),
    ("y ~ x + (1 || g)", "'||'"),
    ("y ~ x * w + (1 | g)", "two numeric terms"),          # M3a-2: only numeric-by-declared-categorical interactions
    ("y ~ x:w + (1 | g)", "main effect 'x' is missing"),
    ("y ~ log(x) + (1 | g)", "function calls"),
    ("y ~ (1 | g)", "no population-level predictor"),
    ("y x + (1 | g)", "expected 'response ~ terms'"),
    ("y ~ x + x + (1 | g)", "duplicated"),
    ("y ~ y + (1 | g)", "response cannot also be"),
    ("y ~ x + (1 | x)", "grouping factor cannot also be"),
])
def test_rejects_unsupported_syntax(formula, needle):
    with pytest.raises(ab.FormulaError, match=needle):
        ab.parse_formula(formula)


def test_accepts_supported_syntax():
    s = ab.parse_formula("y ~ 1 + x + w + (1 | g)")
    assert s.response == "y" and s.predictors == ("x", "w") and s.group == "g"
    s = ab.parse_formula("y ~ 1 + (1|g)")
    assert s.predictors == ()
    s = ab.parse_formula("y~x+(1|g)")
    assert s.predictors == ("x",)
    s = ab.parse_formula("y ~ x")                                     # M3c-1: the group term is optional
    assert s.predictors == ("x",) and s.group is None and s.rhs == "x"


def test_data_validation():
    df = simulate()
    with pytest.raises(ab.FormulaError, match="not in data"):
        ab.parse_formula("y ~ z + (1 | g)", df)
    with pytest.raises(ab.FormulaError, match="declared categorical with its reference level"):
        ab.parse_formula("y ~ g + (1 | g2)", df.assign(g2=df.g))
    with pytest.raises(ab.FormulaError, match="missing values"):
        ab.parse_formula("y ~ x + (1 | g)", df.assign(x=df.x.where(df.index > 0)))
    with pytest.raises(ab.FormulaError, match="single level"):
        ab.parse_formula("y ~ x + (1 | g)", df.assign(g="only"))
    s = ab.parse_formula("y ~ x + (1 | g)", df)
    assert s.levels == tuple(sorted(df.g.unique()))


def test_priors_required():
    df = simulate()
    with pytest.raises(ab.PriorError, match="location prior"):
        ab.gaussian("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.HalfNormal(1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="positive-support"):
        ab.gaussian("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="missing \\['w'\\]"):
        ab.gaussian("y ~ x + w + (1 | g)", simulate(extra=True), ab.Priors(intercept=ab.Normal(0, 1), b={"x": ab.Normal(0, 1)}, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="residual SD"):   # sigma missing for the Gaussian family
        ab.gaussian("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="positive"):
        ab.Normal(0, -1)


def test_model_builds_and_describes():
    m = ab.gaussian("y ~ x + (1 | g)", simulate(), PRIORS)
    text = m.describe()
    assert "mu_i = a_c + (x_x - mean(x_x)) * b_x + u[g_i]" in text
    assert "sigma_log__" in text and "1|g_offset" in text
    assert {v.name for v in m.pymc_model.free_RVs} == {"Intercept", "x", "sigma", "1|g_sigma", "1|g_offset"}


def test_complex_columns_are_refused_gaussian():
    import warnings
    df = simulate()
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(ab.FormulaError, match="response 'y' is complex"):
            ab.gaussian("y ~ x + (1 | g)", df.assign(y=df.y.astype(complex) + 4j), PRIORS)
        with pytest.raises(ab.FormulaError, match="column 'x' is complex"):
            ab.gaussian("y ~ x + (1 | g)", df.assign(x=df.x.astype(complex)), PRIORS)
        with pytest.raises(ab.FormulaError, match="is complex"):
            ab.parse_formula("y ~ x + (1 | g)", df.assign(y=df.y.astype(complex)))
    assert ab.parse_formula("y ~ x + (1 | g)", df.assign(y=df.y.astype(np.float32), x=df.x.astype(int))).levels
