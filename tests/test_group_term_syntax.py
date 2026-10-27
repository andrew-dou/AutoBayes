"""Group-level term syntax (brms-parity feature G05a; docs/group-term-syntax.md): the varying slopes alone, ``(0 + x | g)`` and
``(x - 1 | g)``; one factor in several terms, ``(1 | g) + (0 + x | g)``; factor, transform, product and ``mo()`` slopes; three
grouping columns, ``g:h:f2`` and ``g/h/f2``; and a varying slope without its population-level term.

The grammar and every refusal by message, the term model and the data-dependent expansion of a factor slope, SD-prior coverage,
the design blocks, PyMC names and the graph's linear predictor against a hand computation, predictions by hand on fitted and new
rows (and their agreement with Bambi's), ``describe()``, the target identity, the fit record, the summaries' refusals and the
refusals of the multivariate, non-linear and count-matrix models. Build-only; nothing samples. The numbers against brms are the
oracle's (tests/oracle, the eight G05a cases)."""
import json
import re
from dataclasses import replace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import autobayes as ab
from autobayes import formula as fm
from autobayes.formula import FormulaError, GroupTerm, parse_formula
from autobayes.priors import PriorError
from tests.test_population_only import PANEL_STUB

G, H, F2 = ("a", "b", "c", "d"), ("h1", "h2"), ("u", "v")


def frame(n=32, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.3, 1.0, n), "w": rng.normal(size=n),
                       "z": [("p", "q", "r")[i % 3] for i in range(n)], "ord": [1 + (i // 2) % 4 for i in range(n)],
                       "g": [G[i % 4] for i in range(n)], "h": [H[(i // 4) % 2] for i in range(n)],
                       "f2": [F2[(i // 8) % 2] for i in range(n)], "c": rng.integers(0, 6, n), "s": np.full(n, 0.3),
                       "y2": rng.normal(size=n), "k": rng.integers(0, 5, n), "n": np.full(n, 5)})
    return df


def priors(sd, *, cor=None, simo=None, b=ab.Normal(0, 2)):
    return ab.Priors(intercept=ab.Normal(0, 5), b=b, sd=sd, cor=cor, simo=simo, sigma=ab.HalfNormal(1))


def spec(formula, data=None, **kw):
    return parse_formula(formula, data, **kw)


# -- 1. the grammar ---------------------------------------------------------------------------------------------------
def test_slopes_alone_and_minus_one():
    for written in ("y ~ x + (0 + x | g)", "y ~ x + (x - 1 | g)", "y ~ x + (x + 0 | g)"):
        (t,) = spec(written).group_terms
        assert t == GroupTerm("g", ("x",), True) and not t.has_intercept and t.slopes == ("x",)
        assert t.canonical == "(0 + x | g)" and t.bambi_canonical == "(0 + x | g)" and not t.has_correlation
    (t,) = spec("y ~ x + w + (0 + x + w || g)").group_terms
    assert t.canonical == "(0 + x + w || g)" and not t.has_correlation and t.bambi_canonical == "(0 + x + w | g)"
    (t,) = spec("y ~ x + w + (0 + x + w | g)").group_terms
    assert t.has_correlation and t.canonical == "(0 + x + w | g)"
    assert spec("y ~ x + (0 + x | g)").rhs == "x + (0 + x | g)" and spec("y ~ x + (x - 1 | g)").rhs == "x + (0 + x | g)"


def test_one_factor_in_several_terms_is_one_uncorrelated_term():
    s = spec("y ~ x + (1 | g) + (0 + x | g)")
    (t,) = s.group_terms
    assert t.coefs == ("Intercept", "x") and t.blocks == (("Intercept",), ("x",)) and not t.correlated and not t.has_correlation
    assert t.canonical == "(1 | g) + (0 + x | g)" and t.bambi_canonical == "(1 + x | g)"
    assert s.rhs == "x + (1 | g) + (0 + x | g)" and s.bambi_formula == "y ~ x + (1 + x | g)"
    # the order of the terms is kept in blocks, the intercept first in the term's coefficients
    (u,) = spec("y ~ x + w + (0 + x | g) + (1 + w || g)").group_terms
    assert u.coefs == ("Intercept", "x", "w") and u.blocks == (("x",), ("Intercept", "w"))
    assert u.canonical == "(0 + x | g) + (1 + w || g)"
    (v,) = spec("y ~ x + w + (0 + x | g) + (0 + w | g)").group_terms        # slopes alone, in two terms
    assert v.coefs == ("x", "w") and not v.has_intercept and v.bambi_canonical == "(0 + x + w | g)"
    two = spec("y ~ x + (1 | g/h) + (0 + x | g)")                           # with a nested factor's terms
    assert [t.factor for t in two.group_terms] == ["g", "g:h"] and two.group_terms[0].blocks == (("Intercept",), ("x",))


def test_three_grouping_columns():
    s = spec("y ~ x + (1 | g/h/f2)")
    assert [t.factor for t in s.group_terms] == ["g", "g:h", "g:h:f2"]
    assert s.rhs == "x + (1 | g) + (1 | g:h) + (1 | g:h:f2)"
    t = s.group_terms[2]
    assert t.operands == ("g", "h", "f2") and t.derived and t.column == "g_h_f2" and t.bambi_canonical == "(1 | g_h_f2)"
    (u,) = spec("y ~ x + (1 + x | g:h:f2)").group_terms
    assert u.coefs == ("Intercept", "x") and u.has_correlation


def test_transform_product_monotonic_and_column_slopes():
    s = spec("y ~ x * w + I(x^2) + (1 + I( x ^ 2 ) + w:x | g)")
    (t,) = s.group_terms
    assert t.coefs == ("Intercept", "I(x^2)", "x:w")                       # the population-level terms' spellings
    assert t.canonical == "(1 + I(x^2) + x:w | g)" and s.rhs.endswith("(1 + I(x^2) + x:w | g)")
    assert t.bambi_canonical == "(1 + `I(x^2)` + `x:w` | g)" and t.slope_columns == ()
    (m,) = spec("y ~ mo(ord) + (1 + mo( ord ) | g)").group_terms
    assert m.coefs == ("Intercept", "mo(ord)") and m.monotonic_slopes == ("mo(ord)",) and m.bambi_coefs == ("Intercept",)
    assert m.bambi_canonical == "(1 | g)"
    (c,) = spec("y ~ w + (1 + x | g)").group_terms                          # no population-level x
    assert c.slope_columns == ("x",) and c.data_columns == ("g", "x") and c.coefs == ("Intercept", "x")
    implied = spec("y ~ (1 + x | g)")                                       # R's formula implies the population intercept
    assert implied.intercept and implied.pop_terms == () and implied.group_terms[0].slope_columns == ("x",)
    assert spec("y ~ (1 | g)").intercept
    assert spec("y ~ x + (1 + x | g)").group_terms[0].slope_columns == ()


REFUSALS = [
    ("y ~ x + (1 | g:h:f2:k)", {}, "at most 3 columns can be combined in a grouping factor ('g:h:f2:k' names 4 columns)"),
    ("y ~ x + (1 | g/h/g)", {}, "a grouping factor cannot be combined with itself ('g/h/g')"),
    ("y ~ x + (0 + 1 + x | g)", {}, "'0' removes the varying intercept and '1' adds it; write one of them"),
    ("y ~ x + (0 + x | g)", {}, None),
    ("y ~ x + (0 + x |p| g)", {}, "a term without the varying intercept is not supported with a correlation ID ('p'); write '(1 + x |p| g)'"),
    ("y ~ x + z + (1 + x |p| g) + (0 + z |p| g)", {}, "share the correlation ID 'p' as one block of correlated coefficients, which is "
                                                     "one term here: write '(1 + x + z | g)'"),
    ("y ~ x + (1 + offset(x) | g)", {}, "offset() cannot be placed inside a group term"),
    ("y ~ x + (1 + mo(x + 1) | g)", {}, "'mo(x + 1)': mo() takes one column name here"),
    ("y ~ x + me(x, s) + (1 + me(x, s) | g)", {}, "a measurement-error term cannot vary by 'g'"),
    ("y ~ x + (1 + sin(x) | g)", {}, "function calls and transformations inside a group term are not supported ('sin(x)'); a varying slope "
                                     "is a numeric column"),
    ("y ~ x * w + (1 + x*w | g)", {}, "'*' inside a group term is not supported: an interaction cannot be a varying slope as 'x*w'"),
    ("y ~ x + (1 + x:2w | g)", {}, "'2w' is not a column name"),
    ("y ~ x + (1 + x:g | g)", {}, "the grouping factor 'g' cannot also be a varying slope"),
    ("y ~ x + (1 + x:y | g)", {}, "the response 'y' cannot be a varying slope"),
    ("k | trials(n) ~ x + (1 + x:n | g)", {"family": "binomial"}, "the trials column 'n' cannot be a varying slope"),
    ("y ~ x + (1 + x:w:c | g)", {}, "'x:w:c': an interaction varying slope is a product of two different columns, x:w"),
    ("y ~ x + (1 + x:x | g)", {}, "'x:x': an interaction varying slope is a product of two different columns"),
    ("y ~ x * w + (1 + x:w + w:x | g)", {}, "duplicated varying slope 'x:w'"),
    ("y ~ x + (1 | g) + (1 | g)", {}, "duplicate grouping factor 'g': its coefficient 'Intercept' varies in '(1 | g)' and in '(1 | g)'; "
                                      "each coefficient of a factor varies in one of its terms"),
    ("y ~ x + (1 | g/h/f2) + (1 | g:h)", {}, "its coefficient 'Intercept' varies in '(1 | g/h/f2)' and in '(1 | g:h)' ('(1 | g/h/f2)' "
                                             "expands to '(1 | g) + (1 | g:h) + (1 | g:h:f2)')"),
    ("y ~ x + (1 | gr(g, by = h)) + (0 + x | g)", {}, "'(1 | gr(g, by = h))' is one of several terms of 'g'; a factor in several terms "
                                                      "with mm() or gr(by = ) is not supported"),
    ("y ~ x + (1 |p| g) + (0 + x | g)", {}, "a factor in several terms with a correlation ID is not supported"),
    ("y ~ x + w + (1 + x | g) + (0 + w | g)", {}, "'(1 + x | g)' is one of several terms of 'g' and correlates its coefficients (brms "
                                                  "gives each term of a factor its own correlation matrix), which no frozen case verifies"),
    ("y ~ x + mo(ord) + (1 + mo(ord) || g) + (0 + x | g)", {}, "has the monotonic slope 'mo(ord)', which no frozen case verifies"),
    ("y ~ x + (1 + I(x^2) | g)", {}, "the varying slope 'I(x^2)' must also be a population-level term of the formula"),
    ("y ~ x + w + (1 + x:w | g)", {}, "the varying slope 'x:w' must also be a population-level term of the formula"),
    ("y ~ x + (1 + mo(ord) | g)", {}, "the varying slope 'mo(ord)' must also be a population-level term of the formula"),
    ("y ~ x:w + (1 + x:w | g)", {}, "the interaction 'x:w', coded by R's rule (an operand without its main effect, or three operands), "
                                    "as a varying slope is not supported"),
    ("y ~ x * z + (1 + x:z | g)", {"categorical": {"z": "p"}}, "the interaction 'x:z' with a factor as a varying slope (one coefficient "
                                                               "per level) is not supported"),
    ("y ~ z + (1 + z || g)", {"categorical": {"z": "p"}}, "the factor 'z' varies by 'g' here only correlated with the varying intercept"),
    ("y ~ z + (0 + z | g)", {"categorical": {"z": "p"}}, "the factor 'z' varies by 'g' here only correlated"),
    ("y ~ 0 + z + (1 + z | g)", {"categorical": {"z": "p"}}, "next to a population intercept, as the frozen cases verify"),
    ("y ~ mo(ord) + (1 + mo(ord) || g)", {}, "the monotonic term 'mo(ord)' varies by 'g' here only correlated with the varying intercept"),
    ("y ~ mo(ord) + (0 + mo(ord) | g)", {}, "the monotonic term 'mo(ord)' varies by 'g' here only correlated"),
    ("y ~ x + (1 + g_h | g:h)", {}, "the derived grouping factor 'g:h' is handed to Bambi as the column 'g_h', which is also a column of "
                                    "the formula"),
    ("y ~ offset(x)", {}, "no population-level predictor"),                   # a group-level term implies the intercept; nothing else
]


@pytest.mark.parametrize("formula,kwargs,message", [r for r in REFUSALS if r[2] is not None], ids=[r[0] for r in REFUSALS if r[2]])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        parse_formula(formula, **kwargs)


def test_extended_group_syntax_names_each_construct():
    cases = [("y ~ x + (1 | g) + (0 + x | g)", "the factor 'g' written in several terms ('(1 | g) + (0 + x | g)')"),
             ("y ~ x + (0 + x | g)", "a term without its varying intercept ('(0 + x | g)')"),
             ("y ~ x + (1 | g:h:f2)", "the grouping factor of three columns 'g:h:f2' ('(1 | g:h:f2)')"),
             ("y ~ w + (1 + x | g)", "the varying slope 'x' without its population-level term ('(1 + x | g)')"),
             ("y ~ x + I(x^2) + (1 + I(x^2) | g)", "the varying slope 'I(x^2)' ('(1 + I(x^2) | g)')"),
             ("y ~ mo(ord) + (1 + mo(ord) | g)", "the varying slope 'mo(ord)' ('(1 + mo(ord) | g)')")]
    for formula, what in cases:
        assert fm.extended_group_syntax(spec(formula)) == what
    z = spec("y ~ z + (1 + z | g)", categorical={"z": "p"})
    assert fm.extended_group_syntax(z) == "the varying slope 'z' ('(1 + z | g)')"
    seen = spec("y ~ z + (1 + z | g)", frame(), categorical={"z": "p"})        # expanded: written
    assert fm.extended_group_syntax(seen) == "the varying slope 'z' ('(1 + z | g)')"
    for plain in ("y ~ x + (1 + x | g)", "y ~ x + (1 | g:h)", "y ~ x + (1 | mm(g, h))"):
        assert fm.extended_group_syntax(spec(plain)) is None


# -- 2. once the data are seen ------------------------------------------------------------------------------------------
def test_factor_slope_expands_to_its_levels():
    s = spec("y ~ z + (1 + z | g)", frame(), categorical={"z": "p"})
    (t,) = s.group_terms
    assert t.coefs == ("Intercept", "z[q]", "z[r]") and t.written == ("Intercept", "z") and t.canonical == "(1 + z | g)"
    assert [t.term_of(c) for c in t.coefs] == ["Intercept", "z", "z"]
    assert t.bambi_canonical == "(1 + `z[q]` + `z[r]` | g)"
    assert fm.derived_group_columns(s) == ("z[q]", "z[r]")
    df = frame()
    assert np.array_equal(fm.group_covariate(s, "z[q]", df), (df.z == "q").to_numpy(dtype=float))
    assert np.array_equal(fm.group_covariate(s, "Intercept", df), np.ones(len(df)))
    assert fm.validate_data(s, df).group_terms == s.group_terms              # seen once: not expanded again
    ordered = df.assign(z=pd.Categorical(df.z, categories=["p", "q", "r"], ordered=True))
    with pytest.raises(FormulaError, match=re.escape("the ordered factor 'z' as a varying slope (one coefficient per polynomial contrast) "
                                                     "is not supported")):
        spec("y ~ z + (1 + z | g)", ordered, categorical={"z": "p"})


def test_covariates_of_transforms_products_and_columns():
    df = frame()
    s = spec("y ~ x * w + I(x^2) + (1 + I(x^2) + x:w | g)", df)
    assert fm.derived_group_columns(s) == ("x:w",)
    assert np.allclose(fm.group_covariate(s, "I(x^2)", df), df.x ** 2) and np.allclose(fm.group_covariate(s, "x:w", df), df.x * df.w)
    c = spec("y ~ w + (1 + x | g)", df)
    assert np.array_equal(fm.group_covariate(c, "x", df), df.x.to_numpy()) and fm.derived_group_columns(c) == ()


def test_three_column_levels_are_the_observed_combinations():
    df = frame()
    s = spec("y ~ x + (1 | g/h/f2)", df)
    labels = sorted({f"{a}_{b}_{c}" for a, b, c in zip(df.g, df.h, df.f2)})
    assert s.levels_by_group["g:h:f2"] == tuple(labels) and len(labels) == 16
    assert s.levels_by_group["g:h"] == tuple(sorted({f"{a}_{b}" for a, b in zip(df.g, df.h)}))


def test_slope_columns_are_checked_on_the_data():
    df = frame()
    with pytest.raises(FormulaError, match=re.escape("column 'z' has dtype")):
        spec("y ~ w + (1 + z | g)", df)
    with pytest.raises(FormulaError, match=re.escape("the varying slope 's' of '(1 + s | g)' is constant in the fitted rows")):
        spec("y ~ w + (1 + s | g)", df)
    with pytest.raises(FormulaError, match="complex"):
        spec("y ~ w + (1 + x | g)", df.assign(x=df.x + 1j))


# -- 3. priors ----------------------------------------------------------------------------------------------------------
def test_sd_priors_of_a_factor_slope_are_keyed_by_the_factor():
    s = spec("y ~ z + (1 + z | g)", frame(), categorical={"z": "p"})
    p = priors({"g": {"Intercept": ab.HalfNormal(1), "z": ab.HalfNormal(0.5)}}, cor={"g": ab.LKJ(2)})
    sds = p.sd_priors(s)
    assert sds == {("g", "Intercept"): ab.HalfNormal(1), ("g", "z[q]"): ab.HalfNormal(0.5), ("g", "z[r]"): ab.HalfNormal(0.5)}
    with pytest.raises(PriorError, match=re.escape("'zq' names a level's coefficient of the factor slope 'z': its SD prior is keyed by "
                                                   "the factor, 'z': <prior>")):
        priors({"g": {"Intercept": ab.HalfNormal(1), "zq": ab.HalfNormal(1), "zr": ab.HalfNormal(1)}}).sd_priors(s)
    per_level = priors({"g": {"Intercept": ab.HalfNormal(1), "z": {"q": ab.HalfNormal(1), "r": ab.HalfNormal(1)}}}, cor={"g": ab.LKJ(2)})
    message = "the SD prior of the factor slope 'z' of 'g' is one Prior for the SDs of all its coefficients (one per non-reference level), got dict"
    for seen in (s, spec("y ~ z + (1 + z | g)", categorical={"z": "p"})):       # with the data and before them
        with pytest.raises(PriorError, match=re.escape(message)):
            per_level.sd_priors(seen)
    defaults = ab.default_priors("y ~ z + (1 + z | g)", frame(), categorical={"z": "p"})
    assert set(defaults.sd["g"]) == {"Intercept", "z"}
    split = ab.default_priors("y ~ x + (1 | g) + (0 + x | g)", frame())
    assert set(split.sd["g"]) == {"Intercept", "x"}


# -- 4. the models: blocks, names and the graph ---------------------------------------------------------------------------
SPLIT = "y ~ w + (1 | g) + (0 + x | g)"
SPLIT_PRIORS = priors({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}})
SLOPE_ONLY = "y ~ x + (0 + x | g) + (1 | h)"
SLOPE_ONLY_PRIORS = priors({"g": {"x": ab.HalfNormal(0.5)}, "h": ab.HalfNormal(1)})
FACTOR = "y ~ z + (1 + z | g)"
FACTOR_PRIORS = priors({"g": {"Intercept": ab.HalfNormal(1), "z": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)})
TRANSFORM = "y ~ x * w + I(x^2) + (1 + I(x^2) + x:w | g)"
TRANSFORM_PRIORS = priors({"g": {"Intercept": ab.HalfNormal(1), "I(x^2)": ab.HalfNormal(1), "x:w": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)})
MO = "y ~ mo(ord) + (1 + mo(ord) | g)"
MO_PRIORS = priors({"g": {"Intercept": ab.HalfNormal(1), "mo(ord)": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)}, simo={"ord": ab.Dirichlet(2, 2, 2)})
THREE = "y ~ x + (1 | g/h/f2)"
THREE_PRIORS = priors({"g": ab.HalfNormal(1), "g:h": ab.HalfNormal(0.5), "g:h:f2": ab.HalfNormal(0.3)})
COLUMN = "y ~ w + (1 + x | g)"
COLUMN_PRIORS = priors({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}, cor={"g": ab.LKJ(2)})


@pytest.fixture(scope="module")
def models():
    df = frame()
    return {"split": ab.gaussian(SPLIT, df, SPLIT_PRIORS), "slope_only": ab.gaussian(SLOPE_ONLY, df, SLOPE_ONLY_PRIORS),
            "factor": ab.gaussian(FACTOR, df, FACTOR_PRIORS, categorical={"z": "p"}),
            "transform": ab.gaussian(TRANSFORM, df, TRANSFORM_PRIORS), "mo": ab.gaussian(MO, df, MO_PRIORS),
            "three": ab.gaussian(THREE, df, THREE_PRIORS), "column": ab.gaussian(COLUMN, df, COLUMN_PRIORS)}


def indicator(labels, levels):
    return np.column_stack([np.asarray(labels) == l for l in levels]).astype(float)


def test_blocks_and_names(models):
    df = frame()
    m = models["slope_only"]
    assert list(m.design.group_blocks) == [("g", "x"), ("h", "Intercept")] and "g" not in m.design.group_designs
    assert m.design.factor_blocks["g"] is m.design.group_blocks[("g", "x")]
    assert np.array_equal(m.design.group_blocks[("g", "x")].Z, indicator(df.g, G) * df.x.to_numpy()[:, None])
    assert m.group_names["g"] == {"coordinate": "g__factor_dim", "sd:x": "x|g_sigma", "offset:x": "x|g_offset", "group_effect:x": "x|g"}
    assert {"x|g_sigma", "x|g_offset", "1|h_sigma"} <= {v.name for v in m.pymc_model.free_RVs}
    one = ab.gaussian("y ~ x + (0 + x | g)", df, priors({"g": {"x": ab.HalfNormal(0.5)}}))
    assert one.design.J == 4 and one.design.levels == G and np.array_equal(one.design.group_index, np.arange(32) % 4)
    assert one.names["sd:x"] == "x|g_sigma" and "sd" not in one.names
    split = models["split"]
    assert list(split.design.group_blocks) == [("g", "Intercept"), ("g", "x")] and split.correlation_names == {}
    assert np.array_equal(split.design.group_blocks[("g", "x")].Z, indicator(df.g, G) * df.x.to_numpy()[:, None])
    f = models["factor"]
    assert list(f.design.group_blocks) == [("g", "Intercept"), ("g", "z[q]"), ("g", "z[r]")]
    assert np.array_equal(f.design.group_blocks[("g", "z[r]")].Z, indicator(df.g, G) * (df.z == "r").to_numpy()[:, None])
    assert f.correlation_coefs("g") == ("Intercept", "z[q]", "z[r]") and f.data["z[q]"].tolist() == (df.z == "q").astype(float).tolist()
    t = models["transform"]
    assert np.allclose(t.design.group_blocks[("g", "x:w")].Z, indicator(df.g, G) * (df.x * df.w).to_numpy()[:, None])
    mo = models["mo"]
    block = mo.design.group_blocks[("g", "mo(ord)")]
    assert block.monotonic == "ord" and np.array_equal(block.Z, indicator(df.g, G)) and block.column_slice == slice(4, 8)
    assert mo.design.Z.shape == (32, 8)
    assert {"mo(ord)|g_sigma", "mo(ord)|g_offset"} <= {v.name for v in mo.pymc_model.free_RVs}
    assert mo.pymc_model["mo(ord)|g"] in mo.pymc_model.deterministics
    three = models["three"]
    assert list(three.design.factor_blocks) == ["g", "g:h", "g:h:f2"] and three.design.factor_blocks["g:h:f2"].J == 16
    assert three.group_names["g:h:f2"]["sd"] == "1|g_h_f2_sigma"


def mu_at(m, point):
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    return np.asarray(fn(point)[0], dtype=float)


def test_the_graphs_linear_predictor_by_hand(models):
    """The graph's ``mu`` at a value point against the model written out: brms's Z of each coefficient, the slope of a column
    without its population term multiplying the raw column, and a monotonic slope multiplying the term's mo(simo, Xmo)."""
    df = frame()
    gi = np.array([G.index(v) for v in df.g])
    rng = np.random.default_rng(2)
    u, v = rng.normal(size=4), rng.normal(size=4)
    m = models["split"]
    point = m.value_point(intercept=0.4, b=[0.7], sd={"g": {"Intercept": 0.8, "x": 0.5}}, z={"g": {"Intercept": u, "x": v}}, sigma=1.0)
    a = 0.4 - 0.7 * df.w.mean()                                              # the raw-scale intercept of the centred design
    assert np.allclose(mu_at(m, point), a + 0.7 * df.w + 0.8 * u[gi] + df.x * 0.5 * v[gi], atol=1e-12)
    mo = models["mo"]
    simplex = np.array([0.2, 0.5, 0.3])
    L = np.array([[1.0, 0.0], [0.6, 0.8]])
    point = mo.value_point(intercept=0.1, b=[], bsp={"ord": 0.9}, simo={"ord": simplex}, sd={"g": {"Intercept": 0.7, "mo(ord)": 0.4}},
                           z={"g": {"Intercept": u, "mo(ord)": v}}, L={"g": L}, sigma=1.0)
    mo_values = 3 * np.concatenate([[0.0], np.cumsum(simplex)])[df.ord.to_numpy() - 1]
    r_slope = 0.4 * (0.6 * u + 0.8 * v)                                      # brms's scale_r_cor: diag(sd) L z
    assert np.allclose(mu_at(mo, point), 0.1 + 0.9 * mo_values + 0.7 * u[gi] + r_slope[gi] * mo_values, atol=1e-12)
    with pytest.raises(RuntimeError, match=re.escape("does not move by r[g] * mo(ord) per row with the varying slope 'mo(ord)' of 'g'")):
        bad = replace(mo.design.group_blocks[("g", "mo(ord)")], group_index=(gi + 1) % 4)
        mo._check_monotonic_slopes({("g", "mo(ord)"): bad})


def test_a_monotonic_slope_next_to_another_factor():
    df = frame()
    m = ab.gaussian("y ~ mo(ord) + (1 + mo(ord) | g) + (1 | h)", df, priors(
        {"g": {"Intercept": ab.HalfNormal(1), "mo(ord)": ab.HalfNormal(1)}, "h": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)},
        simo={"ord": ab.Dirichlet(2, 2, 2)}))
    assert [k for k, b in m.design.group_blocks.items() if b.monotonic] == [("g", "mo(ord)")]
    assert "mo(ord)|h" not in m.pymc_model.named_vars


def test_a_monotonic_slope_under_a_log_link():
    df = frame()
    m = ab.poisson("c ~ mo(ord) + (1 + mo(ord) | g)", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), cor=ab.LKJ(2),
                   sd={"g": {"Intercept": ab.HalfNormal(1), "mo(ord)": ab.HalfNormal(1)}}, simo={"ord": ab.Dirichlet(2, 2, 2)}))
    assert m.design.group_blocks[("g", "mo(ord)")].monotonic == "ord" and m.link == "log"


# -- 5. predictions -------------------------------------------------------------------------------------------------------
def posterior(m, draws=4, seed=9):
    """Distinct draws of every free variable and deterministic the predictions read, under the model's names and dims."""
    rng = np.random.default_rng(seed)
    pmm, data = m.pymc_model, {}
    coords = {"chain": [0], "draw": np.arange(draws)}
    for var in (*pmm.free_RVs, *pmm.deterministics):
        name = var.name
        if name == "mu":
            continue
        dims = tuple(pmm.named_vars_to_dims.get(name, ()))
        shape = (1, draws, *(len(pmm.coords[d]) for d in dims))
        coords.update({d: list(pmm.coords[d]) for d in dims})
        if name.startswith(("L|", "cor|")):
            value = np.broadcast_to(np.eye(shape[-1]), shape).copy()
        elif name.startswith("simo_"):
            value = rng.dirichlet(np.ones(shape[-1]), size=shape[:2])
        elif name.endswith("sigma"):
            value = rng.uniform(0.3, 0.8, shape)
        else:
            value = rng.normal(0.0, 0.6, shape)
        data[name] = (("chain", "draw", *dims), value)
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def by_hand(m, idata, df):
    """The linear predictor written out per construct (the draws' names as the model's)."""
    P = idata.posterior
    val = {k: P[k].to_numpy() for k in P.data_vars}
    a = val["Intercept"][..., None]
    gi = np.array([G.index(v) for v in df.g])
    eta = a + 0.0
    f = m.spec.formula
    if f == SPLIT:
        eta = eta + val["w"][..., None] * df.w.to_numpy() + val["1|g"][..., gi] + val["x|g"][..., gi] * df.x.to_numpy()
    elif f == SLOPE_ONLY:
        hi = np.array([H.index(v) for v in df.h])
        eta = eta + val["x"][..., None] * df.x.to_numpy() + val["x|g"][..., gi] * df.x.to_numpy() + val["1|h"][..., hi]
    elif f == FACTOR:
        q, r = (df.z == "q").to_numpy(float), (df.z == "r").to_numpy(float)
        eta = eta + val["z"][..., 0][..., None] * q + val["z"][..., 1][..., None] * r + val["1|g"][..., gi]
        eta = eta + val["z[q]|g"][..., gi] * q + val["z[r]|g"][..., gi] * r
    elif f == TRANSFORM:
        x, w = df.x.to_numpy(), df.w.to_numpy()
        eta = eta + val["x"][..., None] * x + val["w"][..., None] * w + val["I(x^2)"][..., None] * x ** 2 + val["x:w"][..., None] * x * w
        eta = eta + val["1|g"][..., gi] + val["I(x^2)|g"][..., gi] * x ** 2 + val["x:w|g"][..., gi] * x * w
    elif f == MO:
        s = val["simo_ord"]
        mo = 3 * np.concatenate([np.zeros(s.shape[:-1] + (1,)), np.cumsum(s, -1)], -1)[..., df.ord.to_numpy() - 1]
        eta = eta + val["mo(ord)"][..., None] * mo + val["1|g"][..., gi] + val["mo(ord)|g"][..., gi] * mo
    elif f == COLUMN:
        eta = eta + val["w"][..., None] * df.w.to_numpy() + val["1|g"][..., gi] + val["x|g"][..., gi] * df.x.to_numpy()
    else:                                                                    # THREE
        lv = {k: list(m.spec.levels_by_group[k]) for k in ("g:h", "g:h:f2")}
        gh = [lv["g:h"].index(f"{a}_{b}") for a, b in zip(df.g, df.h)]
        ghf = [lv["g:h:f2"].index(f"{a}_{b}_{c}") for a, b, c in zip(df.g, df.h, df.f2)]
        eta = eta + val["x"][..., None] * df.x.to_numpy() + val["1|g"][..., gi] + val["1|g_h"][..., gh] + val["1|g_h_f2"][..., ghf]
    return eta


@pytest.mark.parametrize("name", ["split", "slope_only", "factor", "transform", "mo", "three", "column"])
def test_predictions_on_fitted_and_new_rows(models, name):
    m = models[name]
    idata = posterior(m)
    df = frame()
    lin = m.predict(idata, kind="linear")["eta"].to_numpy()
    assert np.allclose(lin, by_hand(m, idata, df), atol=1e-10)
    assert np.allclose(m.predict(idata, kind="mean")["mu"].to_numpy(), lin, atol=1e-9)   # the Bambi path (or by hand), the same
    new = frame(n=12, seed=11)                                                # new rows of fitted levels: the covariates recomputed
    new = new[[tuple(r) in set(zip(df.g, df.h, df.f2)) for r in zip(new.g, new.h, new.f2)]].reset_index(drop=True)
    out = m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy()
    assert np.allclose(out, by_hand(m, idata, new), atol=1e-10)
    assert np.allclose(m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy(), out, atol=1e-9)
    if name == "column":                                                      # without group effects x is not read at all
        alone = m.predict(idata, newdata=new.drop(columns=["x", "g"]), kind="linear", include_group_effects=False)["eta"]
        P = idata.posterior
        assert np.allclose(alone.to_numpy(), P["Intercept"].to_numpy()[..., None] + P["w"].to_numpy()[..., None] * new.w.to_numpy())
        with pytest.raises(FormulaError, match=re.escape("newdata column 'x' must be numeric (a varying slope)")):
            m.predict(idata, newdata=new.assign(x="text"), kind="linear")
        with pytest.raises(FormulaError, match="complex"):
            m.predict(idata, newdata=new.assign(x=new.x + 1j), kind="linear")


def test_new_levels_of_a_slope_only_factor_and_a_monotonic_slope(models):
    for name in ("slope_only", "mo"):
        m = models[name]
        idata = posterior(m)
        new = frame(n=4, seed=3).assign(g="new")
        for policy in ("population", "resample_observed"):
            out = m.predict(idata, newdata=new, kind="linear", new_groups=policy, seed=1)
            levels = (json.loads(out.attrs["new_levels_by_factor_json"])["g"] if "new_levels_by_factor_json" in out.attrs
                      else list(out.attrs["new_levels"]))
            assert levels == ["new"] and np.isfinite(out["eta"].to_numpy()).all()
        with pytest.raises(FormulaError, match="not seen in training"):
            m.predict(idata, newdata=new, kind="linear")


# -- 6. describe(), the identity, the fit record, the summaries ---------------------------------------------------------
def test_describe(models):
    text = models["slope_only"].describe()
    for s in ["  g         : (0 + x | g) lets the slope(s) alone vary: the intercept does not vary by g (brms's term without its varying "
              "intercept)", "mu_i = a_c + (x_x - mean(x_x)) * b_x + x_x,i * v_x|g[g_i] + u_h[h_i]",
              "v_x|g[j] = sd_x|g * z_x|g[j], z_x|g[j] ~ Normal(0, 1), j = 0..3: the slope of x varies by g; the intercept does not",
              "J = 4 levels of g; J = 2 levels of h"]:
        assert s in text, s
    text = models["split"].describe()
    assert ("  g         : written as 2 terms, (1 | g) + (0 + x | g): brms's 2 group IDs of g, each with its own SD(s) and no correlation "
            "with the other(s), so the coefficients Intercept, x vary independently") in text
    assert ("  x         : varies by g without a population-level term: no b_x (its population-level slope is 0); each level's effect "
            "multiplies the raw column x (brms's Z)") in text
    text = models["factor"].describe()
    assert ("  z         : a factor slope of g: one varying coefficient per non-reference level, z[q], z[r], each multiplying the "
            "population design's indicator 1[z_i = l] (treatment contrasts against 'p')") in text
    assert "+ 1[z_i = q] * v_z[q]|g[g_i] + 1[z_i = r] * v_z[r]|g[g_i]" in text
    text = models["transform"].describe()
    assert "  I(x^2)    : the slope of the population-level term I(x^2) varies by g, multiplying its design column I(x^2)_i" in text
    assert "  x:w       : the slope of the population-level term x:w varies by g, multiplying its design column x_x,i * x_w,i" in text
    text = models["mo"].describe()
    assert ("  mo(ord)   : the monotonic term's slope varies by g: row i adds (b_mo(ord) + v_mo(ord)|g[g_i]) * mo(simo_ord, ord_i), one "
            "simplex shared with the population-level term and no design column") in text
    assert "+ u[g_i] + mo(ord)_i * v_mo(ord)|g[g_i]" in text
    text = models["three"].describe()
    assert ("g:h:f2    : derived grouping factor, one level per observed (g, h, f2) combination, labelled g_h_f2 as brms does "
            "(paste(g, h, f2, sep = \"_\"))") in text
    corr = ab.gaussian("y ~ x + w + (0 + x + w | g)", frame(), priors({"g": {"x": ab.HalfNormal(1), "w": ab.HalfNormal(1)}},
                                                                       cor={"g": ab.LKJ(2)})).describe()
    assert ("(v_x|g[j], v_w|g[j]) = diag(sd_x|g, sd_w|g) L_g (z_x|g[j], z_w|g[j]), z ~ Normal(0, 1), j = 0..3: the slopes vary by g with "
            "the correlation matrix Cor_g = L_g L_g^T of (x, w)") in corr
    two = ab.gaussian("y ~ x + w + (0 + x + w || g)", frame(), priors({"g": {"x": ab.HalfNormal(1), "w": ab.HalfNormal(1)}})).describe()
    assert "the slope of w varies by g independently of the other slopes; the intercept does not" in two


def test_target_identity(models):
    df = frame()
    same = ab.gaussian("y ~ w + (1 + x || g)", df, SPLIT_PRIORS)              # the same model written in one term
    assert models["split"].target_identity != same.target_identity     # the formula as written differs
    assert models["slope_only"].target_identity != ab.gaussian("y ~ x + (1 + x | g) + (1 | h)", df, priors(
        {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "h": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)})).target_identity
    relabelled = ab.gaussian(SLOPE_ONLY, df.assign(g=df.g.str.upper()), SLOPE_ONLY_PRIORS)   # the levels enter the identity
    assert relabelled.target_identity != models["slope_only"].target_identity
    assert models["mo"].target_identity != ab.gaussian(MO, df, priors(
        {"g": {"Intercept": ab.HalfNormal(1), "mo(ord)": ab.HalfNormal(2)}}, cor={"g": ab.LKJ(2)},
        simo={"ord": ab.Dirichlet(2, 2, 2)})).target_identity


def test_fit_record(models):
    m = models["slope_only"]
    fit = ab.FitResult(posterior(m, draws=50), "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 50, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(m.target_identity))
    from autobayes.session import fit_record
    rec = fit_record(m, fit, panel=PANEL_STUB)
    json.dumps(rec)
    g, h = rec["model"]["groups"]
    assert g["factor"] == "g" and "term" not in g and "sd_prior" not in g and [s["coef"] for s in g["slopes"]] == ["x"]
    assert h["term"] == "1|h" and h["sd_prior"] == "HalfNormal(sigma=1)"


def test_summaries_refuse_what_no_case_verifies(models):
    from autobayes import summaries

    for model, what in [(models["slope_only"], "the varying intercept absent"), (models["transform"], "the varying slope 'I(x^2)'"),
                        (models["factor"], "the group-level term (1 + z | g) (written)"),
                        (models["three"], "a grouping factor of two columns")]:
        with pytest.raises(ValueError, match=re.escape(what)):
            summaries.refuse_unsupported(model, "fixef()")
    later = ab.gaussian("y ~ w + (0 + x | g) + (1 | g)", frame(), SPLIT_PRIORS)
    with pytest.raises(ValueError, match=re.escape("the varying intercept after a slope")):
        summaries.refuse_unsupported(later, "fixef()")
    m = models["split"]
    summaries.refuse_unsupported(m, "coef()")
    idata = posterior(m, draws=6)
    out = ab.coef(m, idata)["g"]
    P = idata.posterior
    assert list(out.columns.get_level_values(0).unique()) == ["Intercept", "x", "w"]
    assert np.allclose(out[("x", "Estimate")].to_numpy(), P["x|g"].to_numpy()[0].mean(0))    # its group effect alone
    assert np.allclose(out[("w", "Estimate")].to_numpy(), P["w"].to_numpy().mean())


# -- 7. the other kinds of models -----------------------------------------------------------------------------------------
def test_refused_in_multivariate_nonlinear_and_count_models():
    from autobayes.multinomial import parse_multinomial
    from autobayes.multivariate import parse_multivariate
    from autobayes.nonlinear import parse_nonlinear

    df = frame()
    with pytest.raises(FormulaError, match=re.escape("a term without its varying intercept ('(0 + x | g)') in a multivariate formula is "
                                                     "not supported")):
        parse_multivariate("mvbind(y, y2) ~ x + (0 + x | g)", df, rescor=False)
    with pytest.raises(FormulaError, match=re.escape("the grouping factor of three columns 'g:h:f2' ('(1 | g:h:f2)') in a multivariate "
                                                     "model is not supported")):
        parse_multivariate(["y ~ x + (1 | g:h:f2)", "y2 ~ w"], df, rescor=False)
    with pytest.raises(FormulaError, match=re.escape("the varying slope 'x' without its population-level term ('(1 + x | g)') in a "
                                                     "multivariate model is not supported")):
        parse_multivariate(["y ~ mi(xm) + w + (1 + x | g)", "xm | mi() ~ w"], df.assign(xm=df.x.where(df.index % 5 > 0)), rescor=False)
    with pytest.raises(FormulaError, match=re.escape("a term without its varying intercept ('(0 + w | g)') in a non-linear parameter's "
                                                     "formula is not supported")):
        parse_nonlinear(["y ~ a * exp(c * x)", "a ~ 1 + w + (0 + w | g)", "c ~ 1"], df)
    with pytest.raises(FormulaError, match=re.escape("the grouping factor of three columns '(1 | g:h:f2)' is not supported for the "
                                                     "multinomial families")):
        parse_multinomial("cbind(m1, m2, m3) | trials(size) ~ x + (1 | g:h:f2)")
    with pytest.raises(FormulaError, match=re.escape("a grouping factor of three columns ('(1 | g:h:f2)') is not offered for the "
                                                     "dirichlet family")):
        parse_formula("cbind(a, b, c) ~ x + (1 | g:h:f2)", None, "dirichlet")


def test_prior_draws_refuse_the_new_terms(models):
    with pytest.raises(ValueError, match=re.escape("prior_draws() of a model with the factor 'g' written in several terms "
                                                   "('(1 | g) + (0 + x | g)') is not offered")):
        ab.prior_draws(models["split"], draws=10)


# -- 8. the oracle adapter's extension is strict ----------------------------------------------------------------------------
def test_the_adapter_tells_one_brms_group_from_two():
    """A factor written in several terms meets brms's several groups only (one per term), and one term meets one group only:
    ``(1 + x || g)`` against brms's ``(1 | g) + (0 + x | g)`` and the reverse are mismatches, as is a model without a monotonic
    slope against brms's ``(1 + mo(ord) | g)``; brms's labels are the AutoBayes labels' rename()."""
    from tests.oracle import adapter
    from tests.oracle import fixture as fx

    def load(case_id):
        case = fx.load_case(case_id)
        return fx.Fixture.load(case_id), fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")

    split, split_data = load("gauss_split_same_factor")
    one, one_data = load("gauss_slope_independent")
    sd = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
    gmap = adapter.match_groups(split, ab.gaussian("y ~ x + (1 | g) + (0 + x | g)", split_data, priors(sd)))
    assert gmap.coef_id == {("g", "Intercept"): 1, ("g", "x"): 2} and adapter.group_id(gmap, "g", "x") == 2
    with pytest.raises(adapter.OracleMismatch, match="its coefficients match different brms groups"):
        adapter.match_groups(split, ab.gaussian("y ~ x + (1 + x || g)", split_data, priors(sd)))
    with pytest.raises(adapter.OracleMismatch, match="its coefficients match different brms groups"):
        adapter.match_groups(one, ab.gaussian("y ~ x + (1 | g) + (0 + x | g)", one_data, priors(sd)))
    mo, mo_data = load("gauss_mo_varying_slope")
    with pytest.raises(adapter.OracleMismatch, match=re.escape("brms group-level terms without an AutoBayes counterpart: [('g', 'moord')]")):
        adapter.match_groups(mo, ab.gaussian("y ~ mo(ord) + (1 | g)", mo_data, priors({"g": ab.HalfNormal(1)},
                                                                                       simo={"ord": ab.Dirichlet(2, 2, 2)})))
    assert [adapter.brms_coef_label(c) for c in ("Intercept", "x", "I(x^2)", "z[q]", "mo(ord)", "x:w")] == \
        ["Intercept", "x", "IxE2", "zq", "moord", "x:w"]
