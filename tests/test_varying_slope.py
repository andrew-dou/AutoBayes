"""Independent varying slopes, ``(1 + x || g)`` (brms-parity feature A01; docs/varying-slopes.md).

Refusals with their messages, the term model and design blocks, SD-prior coverage, ``value_point``, the
density against an explicit reference, by-hand predictions on fitted and new rows under both ``new_groups``
policies (and their agreement with the Bambi path), and ``describe()``. Build-only; nothing samples. The
numbers against brms are the oracle's business (tests/oracle, the five ``*slope*`` cases)."""
import re

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes.formula import GroupTerm, PopTerm, bambi_group_name, sd_tag

J_G, J_H = 4, 3
LEVELS_G, LEVELS_H = tuple(f"g{i}" for i in range(J_G)), tuple(f"h{i}" for i in range(J_H))


def frame(n=24, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.4, 1.0, n), "w": rng.normal(size=n),
                       "z": rng.choice(["p", "q", "r"], size=n), "g": [LEVELS_G[i % J_G] for i in range(n)],
                       "h": [LEVELS_H[(i // 2) % J_H] for i in range(n)], "yb": (rng.uniform(size=n) < 0.5).astype(int),
                       "k": rng.integers(0, 4, n), "n": np.full(n, 5)})
    assert set(df.z) == {"p", "q", "r"} and set(df.yb) == {0, 1}
    return df


def indicator(df, factor, levels):
    return np.column_stack([df[factor].to_numpy() == l for l in levels]).astype(float)


SLOPE_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5),
                         sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}})
TWO_FACTOR_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5),
                              sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.Exponential(3.0)}, "h": ab.HalfNormal(0.7)})
INTERCEPT_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))


@pytest.fixture(scope="module")
def single():
    """``y ~ x + (1 + x || g)``: one factor, one slope."""
    return ab.gaussian("y ~ x + (1 + x || g)", frame(), SLOPE_PRIORS)


@pytest.fixture(scope="module")
def two_factor():
    """``y ~ x + w + (1 + x || g) + (1 | h)``: a slope factor next to an intercept-only factor."""
    return ab.gaussian("y ~ x + w + (1 + x || g) + (1 | h)", frame(), TWO_FACTOR_PRIORS)


# -- 1. the grammar: what is accepted and every refusal by message -------------------------------------------
REFUSALS = [                                   # '(1 + x | g)', once refused here, is brms's correlated form (A02, tests/test_correlated_slopes.py)
    ("y ~ x + (0 + x || g)", {}, "varying slopes without the varying intercept are not supported; write '(1 + x || g)' so that the intercept varies by 'g' as well"),
    ("y ~ x + (x - 1 || g)", {}, "varying slopes without the varying intercept are not supported; write '(1 + x || g)'"),   # F08:
    ("y ~ x + (0 + x | g)", {}, "varying slopes without the varying intercept are not supported"),   # '(x || g)' is '(1 + x || g)'
    ("y ~ x + (0 | g)", {}, "'0' removes the varying intercept and leaves no group-level coefficient; a group term is '(1 | g)', or '(1 + x || g)' for varying slopes"),
    ("y ~ w + (1 + x || g)", {}, "the varying slope 'x' must also be a population-level term of the formula, as in 'y ~ x + (1 + x || g)'"),
    ("y ~ (1 + x || g)", {}, "the varying slope 'x' must also be a population-level term of the formula"),
    ("y ~ x + z + (1 + z || g)", {"categorical": {"z": "p"}}, "'z' is declared categorical; a categorical varying slope (one varying coefficient per level of 'z') is not supported"),
    ("y ~ x + z + x:z + (1 + x:z || g)", {"categorical": {"z": "p"}}, "an interaction cannot be a varying slope in this subset"),
    ("y ~ x + (1 + x*z || g)", {}, "an interaction cannot be a varying slope in this subset"),
    ("y ~ x + (1 + g || g)", {}, "the grouping factor 'g' cannot also be a varying slope"),
    ("y ~ x + (1 + y || g)", {}, "the response 'y' cannot be a varying slope"),
    ("k | trials(n) ~ x + (1 + n || g)", {"family": "binomial"}, "the trials column 'n' cannot be a varying slope"),
    ("y ~ x + (1 + x || g) + (1 | g)", {}, "duplicate grouping factor 'g'; a factor appears in one group term, '(1 | g)' or '(1 + x || g)'"),
    ("y ~ x + (1 + x || g) + (1 + x || g)", {}, "duplicate grouping factor 'g'"),
    ("y ~ x + (1 + x + x || g)", {}, "duplicated varying slope 'x'"),
    ("y ~ x + (1 + log(x) || g)", {}, "function calls and transformations inside a group term are not supported"),
    ("y ~ x + (1 + 1 || g)", {}, "'1' appears more than once in the group term"),
    ("y ~ x + (1 + x || g:h:k)", {}, "only two factors can be combined in a grouping factor"),   # A03 accepts '(1 + x || g:h)' and '(1 + x || g/h)'
    ("y ~ x + (1 + x || g/h:k)", {}, "mixing '/' and ':' in a grouping factor"),
    ("y ~ x + (1 + x || mm(g, h))", {}, "only the varying intercept '(1 | mm(...))' is supported for a multi-membership term"),   # D08
    ("y ~ x + (1 + x || log(g))", {}, "transformed grouping factors are not supported"),
    ("y ~ x + (1 + x | id | g)", {}, "in a formula on its own it is not supported here"),   # D08: an ID is shared with sigma ~ only
    ("y ~ x + (1 + x |)", {}, "parenthesised term is not a '(1 | group)' or '(1 + x || group)' term"),
    ("y ~ x + (1 + x || g) + x|g", {}, "a '|' term must be written '(1 | group)' or '(1 + x || group)'"),
    ("y ~ x + (1 + x || g", {}, "unbalanced parentheses"),
    ("y ~ x + (1 + x || g) + (1 | x)", {}, "the grouping factor cannot also be a predictor or the response"),
    ("y ~ x + ( || g)", {}, "parenthesised term is not a '(1 | group)' or '(1 + x || group)' term"),
    ("y ~ x + (1 + || g)", {}, "empty term inside the group term"),
    ("y ~ x + (1 + 2x || g)", {}, "'2x' is not a column name"),
    ("y ~ x + (0 + log(x) || g)", {}, "function calls and transformations inside a group term are not supported"),
]


@pytest.mark.parametrize("formula,kwargs,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, **kwargs)


def test_slope_only_refusals_still_mention_varying_slopes():
    """The needles of the pre-existing tests (``"varying slopes"``) keep matching the kept refusals; ``(1 + x | g)`` is
    no longer among them (A02), nor ``(x | g)``, brms's ``(1 + x | g)`` (F08)."""
    for formula in ("y ~ x + (0 + x | g)", "y ~ x + (0 + x || g)"):
        with pytest.raises(ab.FormulaError, match="varying slopes"):
            ab.parse_formula(formula)
    assert ab.parse_formula("y ~ x + (1 + x | g)").group_terms == (GroupTerm("g", ("Intercept", "x"), True),)


def test_accepted_forms_and_the_term_model():
    s = ab.parse_formula("y ~ x + w + (1 + x || g) + (1 | h)")
    assert s.pop_terms == (PopTerm("x", "numeric", ("x",), ()), PopTerm("w", "numeric", ("w",), ()))
    assert s.group_terms == (GroupTerm("g", ("Intercept", "x"), False), GroupTerm("h", ("Intercept",), True))
    assert s.group_coefficients == (("g", "Intercept"), ("g", "x"), ("h", "Intercept")) and s.has_slopes
    assert s.group_term_of("g").slopes == ("x",) and s.group_term_of("h").slopes == ()
    assert s.group_term_of("g").canonical == "(1 + x || g)" and s.group_term_of("h").canonical == "(1 | h)"
    assert s.canonical_formula == "y ~ x + w + (1 + x || g) + (1 | h)"
    assert s.bambi_formula == "y ~ x + w + (1 + x | g) + (1 | h)"          # Bambi has no '||'
    # order inside the term and spacing do not matter; several slopes, and the same slope in two factors, are allowed
    assert ab.parse_formula("y ~ x + w + (x+1||g)").group_terms == (GroupTerm("g", ("Intercept", "x"), False),)
    assert ab.parse_formula("y ~ x + w + (1 + x + w || g)").group_terms == (GroupTerm("g", ("Intercept", "x", "w"), False),)
    assert ab.parse_formula("y ~ x + w + (1 + w + x || g)").rhs == "x + w + (1 + w + x || g)"
    both = ab.parse_formula("y ~ x + (1 + x || g) + (1 + x || h)")
    assert both.group_coefficients == (("g", "Intercept"), ("g", "x"), ("h", "Intercept"), ("h", "x"))
    # '(1 || g)' is the varying intercept: one coefficient has nothing to correlate with
    t = ab.parse_formula("y ~ x + (1 || g)")
    assert t.group_terms == (GroupTerm("g", ("Intercept",), True),) and t.rhs == "x + (1 | g)" and not t.has_slopes
    # a slope-only formula without a population intercept keeps the varying intercept
    u = ab.parse_formula("y ~ 0 + x + (1 + x || g)")
    assert u.intercept is False and u.group_terms == (GroupTerm("g", ("Intercept", "x"), False),)
    # binomial: the trials column is not a slope, but another numeric term is
    b = ab.parse_formula("k | trials(n) ~ x + (1 + x || g)", family="binomial")
    assert b.group_terms == (GroupTerm("g", ("Intercept", "x"), False),) and b.bambi_formula == "p(k, n) ~ x + (1 + x | g)"
    assert bambi_group_name("g", "Intercept") == "1|g" and bambi_group_name("g", "x") == "x|g"
    assert sd_tag("g", "Intercept") == "g" and sd_tag("g", "x") == "x|g"


def test_double_bar_intercept_only_is_the_same_model_as_single_bar():
    df = frame()
    a = ab.gaussian("y ~ x + (1 || g)", df, INTERCEPT_PRIORS)
    b = ab.gaussian("y ~ x + (1 | g)", df, INTERCEPT_PRIORS)
    assert a.target_identity == b.target_identity and a.spec.canonical_formula == "y ~ x + (1 | g)"
    assert a.group_names == b.group_names == {"g": {"sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "coordinate": "g__factor_dim"}}
    assert not any(k.startswith(("sd:", "offset:", "group_effect:")) for k in b.names)
    assert "varying slope" not in b.describe() and "v_" not in b.describe()


# -- 2. the design blocks and the graph ----------------------------------------------------------------------
def test_design_blocks_graph_and_names(two_factor):
    m, df = two_factor, two_factor.data
    d = m.design
    x = df.x.to_numpy(dtype=float)
    ind_g, ind_h = indicator(df, "g", LEVELS_G), indicator(df, "h", LEVELS_H)
    assert list(d.group_blocks) == [("g", "Intercept"), ("g", "x"), ("h", "Intercept")]
    bg, bx, bh = d.group_blocks[("g", "Intercept")], d.group_blocks[("g", "x")], d.group_blocks[("h", "Intercept")]
    assert np.array_equal(bg.Z, ind_g) and np.array_equal(bx.Z, ind_g * x[:, None]) and np.array_equal(bh.Z, ind_h)
    assert bx.levels == bg.levels == LEVELS_G and bx.J == J_G and np.array_equal(bx.group_index, bg.group_index)
    assert (bg.column_slice, bx.column_slice, bh.column_slice) == (slice(0, 4), slice(4, 8), slice(8, 11))
    assert np.array_equal(d.Z, np.column_stack([ind_g, ind_g * x[:, None], ind_h])) and d.Z.shape == (24, 11)
    assert d.group_designs == {"g": bg, "h": bh} and d.group_slices == {"g": slice(0, 4), "h": slice(8, 11)}
    # the population design and its centring are untouched by the slope; the slope block multiplies the RAW x
    assert d.columns == ("x", "w") and np.allclose(d.X_model[:, 0], x - x.mean()) and d.centered
    # PyMC: the slope's variables sit on the factor's level dimension, next to the intercept's
    pmm = m.pymc_model
    assert [v.name for v in pmm.value_vars if "|" in v.name] == ["1|g_sigma_log__", "1|g_offset", "x|g_sigma_log__", "x|g_offset", "1|h_sigma_log__", "1|h_offset"]
    assert tuple(pmm.named_vars_to_dims["x|g_offset"]) == tuple(pmm.named_vars_to_dims["x|g"]) == ("g__factor_dim",)
    assert list(pmm.coords["g__factor_dim"]) == list(LEVELS_G)
    # names: the intercept-only factor is exactly as before, the slope factor adds the slope's roles
    slope = {"sd": "x|g_sigma", "offset": "x|g_offset", "group_effect": "x|g", "coordinate": "g__factor_dim"}
    assert m.group_coefficient_names == {("g", "Intercept"): {"sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "coordinate": "g__factor_dim"},
                                         ("g", "x"): slope,
                                         ("h", "Intercept"): {"sd": "1|h_sigma", "offset": "1|h_offset", "group_effect": "1|h", "coordinate": "h__factor_dim"}}
    assert m.group_names["g"] == {"sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "coordinate": "g__factor_dim",
                                  "sd:x": "x|g_sigma", "offset:x": "x|g_offset", "group_effect:x": "x|g"}
    assert m.group_names["h"] == {"sd": "1|h_sigma", "offset": "1|h_offset", "group_effect": "1|h", "coordinate": "h__factor_dim"}
    assert not {"sd", "sd:x"} & set(m.names)                                   # two factors: no single-factor roles in names
    assert m.coefficients == (("x", "x", None), ("w", "w", None))
    # priors: one SD prior per coefficient; the per-factor view and Bambi's dict
    assert m.priors.sd_priors(m.spec) == {("g", "Intercept"): ab.HalfNormal(1), ("g", "x"): ab.Exponential(3.0), ("h", "Intercept"): ab.HalfNormal(0.7)}
    assert m.priors.group_priors(m.spec) == {"g": ab.HalfNormal(1), "h": ab.HalfNormal(0.7)}
    assert {"1|g", "x|g", "1|h"} <= set(m.priors.to_bambi(m.spec, "gaussian"))


def test_single_factor_names_include_the_slope(single):
    m = single
    assert {k: m.names[k] for k in ("sd", "offset", "group_effect", "sd:x", "offset:x", "group_effect:x")} == {
        "sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "sd:x": "x|g_sigma", "offset:x": "x|g_offset", "group_effect:x": "x|g"}
    assert m.spec.group_term == "1|g" and m.design.J == J_G and m.design.levels == LEVELS_G


# -- 3. SD priors: forms accepted and coverage errors ------------------------------------------------------------
def test_sd_prior_forms_and_coverage_errors():
    spec = ab.parse_formula("y ~ x + w + (1 + x || g)")
    two = ab.parse_formula("y ~ x + (1 + x || g) + (1 | h)")

    def validate(sd, s=spec):
        ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=sd, sigma=ab.HalfNormal(1)).validate(s, "gaussian")

    def refused(sd, message, s=spec):
        with pytest.raises(ab.PriorError, match=re.escape(message)):
            validate(sd, s)

    validate({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}})
    validate({"g": {"x": ab.HalfNormal(0.5), "Intercept": ab.HalfNormal(1)}})                         # order is free
    refused(ab.HalfNormal(1), "sd for 'g' must be a mapping over its coefficients ['Intercept', 'x'] ('(1 + x || g)' has varying slopes), got a single Prior")
    refused({"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)},
            "sd must cover exactly the grouping factors ['g']; got ['Intercept', 'x']; ['Intercept', 'x'] look like the coefficients of 'g': "
            "nest them under the factor, sd={'g': {'Intercept': <prior>, 'x': <prior>}}")
    refused({"g": {"Intercept": ab.HalfNormal(1)}}, "SD priors for 'g' must cover exactly its coefficients ['Intercept', 'x']; missing ['x'], unexpected []")
    refused({"g": {"x": ab.HalfNormal(1)}}, "missing ['Intercept'], unexpected []")
    refused({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1), "w": ab.HalfNormal(1)}},
            "missing [], unexpected ['w'] ('w' is a population-level term, but '(1 + x || g)' has no varying slope on it; write '(1 + w || g)' to add one)")
    refused({"g": {"Intercept": ab.HalfNormal(1), "x": ab.Normal(0, 1)}}, "prior for 'sd:x|g' must be a positive-support prior")
    refused({"g": {"Intercept": ab.Normal(0, 1), "x": ab.HalfNormal(1)}}, "prior for 'sd:g' must be a positive-support prior")
    refused({"g": {"Intercept": ab.HalfNormal(1), "x": 0.5}}, "prior for 'sd:x|g' must be a Prior, got float")
    refused({"g": [ab.HalfNormal(1), ab.HalfNormal(1)]}, "sd for 'g' must be a mapping over its coefficients ['Intercept', 'x'] ('(1 + x || g)' has varying slopes), got a single list")
    refused(None, "the group term '(1 + x || g)' needs a prior on the SD of each of its coefficients ['Intercept', 'x']: pass sd={'g': {'Intercept': <prior>, 'x': <prior>}}")
    # two factors: the slope factor takes the mapping, the intercept-only factor a prior or the one-entry mapping
    validate({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "h": ab.HalfNormal(1)}, two)
    validate({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "h": {"Intercept": ab.HalfNormal(1)}}, two)
    refused({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}, "sd must cover exactly the grouping factors ['g', 'h']; got ['g']", two)
    refused({"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}, "sd for 'g' must be a mapping over its coefficients", two)
    refused({"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
            "SD priors for 'h' must cover exactly its coefficients ['Intercept']; missing [], unexpected ['x'] ('x' is a population-level term, but '(1 | h)' has no varying slope on it; write '(1 + x || h)' to add one)", two)
    refused(None, "each varying intercept needs a prior on the SD: pass sd for ['g', 'h']", two)
    # intercept-only models keep their forms
    one = ab.parse_formula("y ~ x + (1 | g)")
    validate(ab.HalfNormal(1), one)
    validate({"g": {"Intercept": ab.HalfNormal(1)}}, one)
    pr = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1)}}, sigma=ab.HalfNormal(1))
    assert pr.sd_priors(one) == {("g", "Intercept"): ab.HalfNormal(1)} and pr.group_priors(one) == {"g": ab.HalfNormal(1)}
    refused({"Intercept": ab.HalfNormal(1)}, "sd must cover exactly the grouping factors ['g']; got ['Intercept']; ['Intercept'] look like the coefficients of 'g'", one)


def test_one_entry_intercept_mapping_builds_the_same_model():
    df = frame()
    a = ab.gaussian("y ~ x + (1 | g)", df, INTERCEPT_PRIORS)
    b = ab.gaussian("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd={"g": {"Intercept": ab.HalfNormal(1)}}, sigma=ab.HalfNormal(1.5)))
    assert a.target_identity == b.target_identity and a.describe() == b.describe()


# -- 4. value_point and the density ------------------------------------------------------------------------------
def reference_logp(m, nat, jacobian):
    """Explicit density of the two-factor model at natural-scale arguments: slope effect times the RAW x."""
    df, d = m.data, m.design
    x = df.x.to_numpy(dtype=float)
    ind_g, ind_h = indicator(df, "g", LEVELS_G), indicator(df, "h", LEVELS_H)
    sd_g, sd_x, sd_h = nat["sd"]["g"]["Intercept"], nat["sd"]["g"]["x"], nat["sd"]["h"]
    z_g, z_x, z_h = nat["z"]["g"]["Intercept"], nat["z"]["g"]["x"], nat["z"]["h"]
    eta = nat["intercept"] + d.X_model @ nat["b"] + ind_g @ (sd_g * z_g) + x * (ind_g @ (sd_x * z_x)) + ind_h @ (sd_h * z_h)
    lp = (norm.logpdf(nat["intercept"], 0.2, 5) + norm.logpdf(nat["b"], 0.1, 2).sum()
          + norm.logpdf(z_g).sum() + norm.logpdf(z_x).sum() + norm.logpdf(z_h).sum()
          + halfnorm.logpdf(sd_g, scale=1) + np.log(3.0) - 3.0 * sd_x + halfnorm.logpdf(sd_h, scale=0.7)
          + halfnorm.logpdf(nat["sigma"], scale=1.5) + norm.logpdf(df.y.to_numpy(), eta, nat["sigma"]).sum())
    if jacobian:
        lp += np.log(sd_g) + np.log(sd_x) + np.log(sd_h) + np.log(nat["sigma"])
    return float(lp), eta


def natural(k=0):
    return dict(intercept=0.3 + 0.1 * k, b=np.array([0.5, -0.2]) + 0.05 * k,
                sd={"g": {"Intercept": 0.8 - 0.1 * k, "x": 0.3 + 0.05 * k}, "h": 0.5},
                z={"g": {"Intercept": np.array([-0.4, 0.2, 0.9, -0.1]) + 0.1 * k, "x": np.array([0.7, -0.5, 0.1, 0.3]) - 0.1 * k},
                   "h": np.array([0.2, -0.6, 0.4])},
                sigma=0.9 + 0.1 * k)


def test_value_point_places_every_coefficient_and_the_density_matches(two_factor):
    m = two_factor
    pmm = m.pymc_model
    logp, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    linpred = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    for k in range(3):
        nat = natural(k)
        pt = m.value_point(**nat)
        assert set(pt) == {v.name for v in pmm.value_vars}
        assert float(pt["x|g_sigma_log__"]) == np.log(nat["sd"]["g"]["x"]) and float(pt["1|g_sigma_log__"]) == np.log(nat["sd"]["g"]["Intercept"])
        assert np.array_equal(pt["x|g_offset"], nat["z"]["g"]["x"]) and np.array_equal(pt["1|g_offset"], nat["z"]["g"]["Intercept"])
        assert np.array_equal(pt["1|h_offset"], nat["z"]["h"]) and float(pt["1|h_sigma_log__"]) == np.log(0.5)
        ref, eta = reference_logp(m, nat, True)
        assert abs(float(logp(pt)) - ref) < 1e-8 and abs(float(nojac(pt)) - reference_logp(m, nat, False)[0]) < 1e-8
        assert np.allclose(np.asarray(linpred(pt)[0]).ravel(), eta, atol=1e-12)
    # the one-entry mapping for an intercept-only factor is the bare form
    nat = natural()
    alt = m.value_point(**{**nat, "sd": {**nat["sd"], "h": {"Intercept": 0.5}}, "z": {**nat["z"], "h": {"Intercept": nat["z"]["h"]}}})
    assert all(np.array_equal(alt[k], v) for k, v in m.value_point(**nat).items())


def test_value_point_refusals(two_factor, single):
    m, nat = two_factor, natural()

    def refused(message, **override):
        with pytest.raises(ValueError, match=re.escape(message)):
            m.value_point(**{**nat, **override})

    refused("sd for 'g' must be a mapping over its coefficients ['Intercept', 'x'] ('(1 + x || g)' has varying slopes)", sd={"g": 0.8, "h": 0.5})
    refused("z for 'g' must be a mapping over its coefficients ['Intercept', 'x']", z={**nat["z"], "g": np.zeros(J_G)})
    refused("sd for 'g' must cover exactly its coefficients ['Intercept', 'x']; got ['Intercept']", sd={"g": {"Intercept": 0.8}, "h": 0.5})
    refused("z for 'g' must cover exactly its coefficients ['Intercept', 'x']; got ['Intercept', 'x', 'w']",
            z={**nat["z"], "g": {**nat["z"]["g"], "w": np.zeros(J_G)}})
    refused("z for the slope 'x' of 'g' must have 4 finite entries", z={**nat["z"], "g": {**nat["z"]["g"], "x": np.zeros(J_G + 1)}})
    refused("z for 'g' must have 4 finite entries", z={**nat["z"], "g": {**nat["z"]["g"], "Intercept": np.array([0.0, np.nan, 0.0, 0.0])}})
    refused("sd:x|g must be a finite positive scalar", sd={"g": {"Intercept": 0.8, "x": 0.0}, "h": 0.5})
    refused("sd:g must be a finite positive scalar", sd={"g": {"Intercept": -0.8, "x": 0.3}, "h": 0.5})
    refused("sd for 'h' must cover exactly its coefficients ['Intercept']; got ['Intercept', 'x']", sd={**nat["sd"], "h": {"Intercept": 0.5, "x": 0.1}})
    refused("sd must cover exactly the grouping factors ['g', 'h']", sd={"g": nat["sd"]["g"]})
    refused("sd and z are required for every group-level coefficient", sd=None)
    # a single-factor model must still name the factor; the coefficient-keyed form without it gets a hint
    with pytest.raises(ValueError, match=re.escape("sd must cover exactly the grouping factors ['g'] (['Intercept', 'x'] look like the coefficients of 'g': "
                                                   "nest them under the factor, sd={'g': {'Intercept': ..., 'x': ...}})")):
        single.value_point(intercept=0.1, b=[0.2], sd={"Intercept": 0.8, "x": 0.3}, z={"g": {"Intercept": np.zeros(J_G), "x": np.zeros(J_G)}}, sigma=0.5)
    with pytest.raises(ValueError, match=re.escape("z must cover exactly the grouping factors ['g'] (['Intercept', 'x'] look like the coefficients of 'g'")):
        single.value_point(intercept=0.1, b=[0.2], sd={"g": {"Intercept": 0.8, "x": 0.3}}, z={"Intercept": np.zeros(J_G), "x": np.zeros(J_G)}, sigma=0.5)
    with pytest.raises(ValueError, match=re.escape("sd for 'g' must be a mapping over its coefficients")):
        single.value_point(intercept=0.1, b=[0.2], sd=0.8, z={"g": {"Intercept": np.zeros(J_G), "x": np.zeros(J_G)}}, sigma=0.5)
    pt = single.value_point(intercept=0.1, b=[0.2], sd={"g": {"Intercept": 0.8, "x": 0.3}}, z={"g": {"Intercept": np.zeros(J_G), "x": np.ones(J_G)}}, sigma=0.5)
    assert set(pt) == {v.name for v in single.pymc_model.value_vars} and np.array_equal(pt["x|g_offset"], np.ones(J_G))


# -- 5. predictions by hand, both policies, and the Bambi path -----------------------------------------------------
def synthetic(model, draws=5, seed=8):
    """A posterior whose draws are distinct constants, so that the pairing of (u, v) per level is checkable."""
    rng = np.random.default_rng(seed)
    J = model.design.J
    shape = (1, draws)
    var = {"Intercept": (("chain", "draw"), rng.normal(1.0, 0.1, shape)), "x": (("chain", "draw"), rng.normal(0.5, 0.1, shape)),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)), "x|g_sigma": (("chain", "draw"), rng.uniform(0.2, 0.4, shape)),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (J,))), "x|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.3, shape + (J,))),
           "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(model.design.levels)}))


def test_predictions_fitted_rows_and_without_group_effects(single):
    m, idata = single, synthetic(single)
    P = idata.posterior
    a, b, U, V = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy(), P["x|g"].to_numpy()
    x, gi = m.data.x.to_numpy(dtype=float), m.design.group_index
    eta = a[..., None] + b[..., None] * x + U[..., gi] + x * V[..., gi]                   # the slope multiplies the raw x
    lin = m.predict(idata, kind="linear")
    assert np.allclose(lin["eta"].to_numpy(), eta, atol=1e-12) and lin.attrs["include_group_effects"] == 1
    assert np.allclose(m.predict(idata, kind="mean")["mu"].to_numpy(), eta, atol=1e-10)   # the Bambi path agrees
    fixed = a[..., None] + b[..., None] * x
    out = m.predict(idata, kind="mean", include_group_effects=False)
    assert np.allclose(out["mu"].to_numpy(), fixed, atol=1e-12) and out.attrs["new_groups"] == "not_applicable"
    assert np.allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"].to_numpy(), fixed, atol=1e-12)
    resp = m.predict(idata, kind="response", seed=4)
    assert np.allclose(resp["mu"].to_numpy(), eta, atol=1e-10) and resp["y"].shape == eta.shape


@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_predictions_on_new_rows_and_unseen_levels(single, policy):
    m, idata = single, synthetic(single)
    P = idata.posterior
    a, b, U, V = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy(), P["x|g"].to_numpy()
    sd_u, sd_v = P["1|g_sigma"].to_numpy(), P["x|g_sigma"].to_numpy()
    levels = list(m.design.levels)
    new = pd.DataFrame({"x": [0.0, 1.0, -2.0, 1.0, 1.0, 3.0, 1.0, 2.0],
                        "g": [levels[2], levels[2], levels[0], "newA", "newA", "newA", "newB", "newB"]})
    seed = 11
    out = m.predict(idata, newdata=new, kind="linear", new_groups=policy, seed=seed)
    eta = out["eta"].to_numpy()
    assert out.attrs["new_groups"] == policy and list(out.attrs["new_levels"]) == ["newA", "newB"]
    for i in range(3):                                                                    # known levels: exact label identity
        j = levels.index(new.g[i])
        assert np.allclose(eta[..., i], a + b * new.x[i] + U[..., j] + new.x[i] * V[..., j], atol=1e-12)
    # unseen labels: eta = a + b x + u* + x v* with one (u*, v*) pair per label and draw, reproduced from the seed:
    # RNG order is factor, then sorted unseen labels, then coefficient (intercept, slope)
    rng = np.random.default_rng(seed)
    pairs = {}
    for lvl in ("newA", "newB"):
        if policy == "population":
            pairs[lvl] = (rng.normal(0.0, 1.0, size=sd_u.shape) * sd_u, rng.normal(0.0, 1.0, size=sd_v.shape) * sd_v)
        else:
            chosen = rng.integers(0, len(levels), size=sd_u.shape)                        # ONE observed level for both coefficients
            pairs[lvl] = (np.take_along_axis(U, chosen[..., None], -1)[..., 0], np.take_along_axis(V, chosen[..., None], -1)[..., 0])
    for i in range(3, 8):
        u_star, v_star = pairs[new.g[i]]
        assert np.allclose(eta[..., i], a + b * new.x[i] + u_star + new.x[i] * v_star, atol=1e-12)
    # the pairs solved from the rows agree: rows 3, 4 share x = 1 and the label; row 5 has x = 3
    assert np.allclose(eta[..., 3], eta[..., 4])
    v_solved = (eta[..., 5] - eta[..., 3] - 2 * b) / 2.0
    u_solved = eta[..., 3] - a - b - v_solved
    observed_pair = [any(np.isclose(u_solved[0, d], U[0, d, j]) and np.isclose(v_solved[0, d], V[0, d, j]) for j in range(len(levels)))
                     for d in range(eta.shape[1])]
    assert all(observed_pair) if policy == "resample_observed" else not any(observed_pair)
    assert not np.allclose(eta[..., 6] - a - b, eta[..., 3] - a - b)                      # newB has its own pair
    # kind="mean" goes by hand too (unseen levels) and the response draws sit on top of it
    mean = m.predict(idata, newdata=new, kind="mean", new_groups=policy, seed=seed)
    assert np.allclose(mean["mu"].to_numpy(), eta, atol=1e-12)
    # the default refuses unseen levels, and the Bambi path agrees on new rows with fitted levels only
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(idata, newdata=new, kind="mean")
    known = new.iloc[:3]
    assert np.allclose(m.predict(idata, newdata=known, kind="mean")["mu"].to_numpy(),
                       m.predict(idata, newdata=known, kind="linear")["eta"].to_numpy(), atol=1e-10)
    # without group effects the slope's column is still needed (it is a population-level term) but not the factor
    fixed = m.predict(idata, newdata=new.drop(columns="g"), kind="linear", include_group_effects=False)["eta"].to_numpy()
    assert np.allclose(fixed, a[..., None] + b[..., None] * new.x.to_numpy(), atol=1e-12)
    with pytest.raises(ab.FormulaError, match="newdata lacks columns \\['x'\\]"):
        m.predict(idata, newdata=new.drop(columns="x"), kind="linear", new_groups=policy)


def test_intercept_only_factor_consumes_the_same_random_numbers(two_factor):
    """An intercept-only factor next to a slope factor: one draw per unseen label, as before the feature."""
    m = two_factor
    rng = np.random.default_rng(1)
    J = {"g": J_G, "h": J_H}
    shape = (1, 4)
    var = {"Intercept": (("chain", "draw"), rng.normal(size=shape)), "x": (("chain", "draw"), rng.normal(size=shape)), "w": (("chain", "draw"), rng.normal(size=shape)),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)), "x|g_sigma": (("chain", "draw"), rng.uniform(0.2, 0.4, shape)), "1|h_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=shape + (J["g"],))), "x|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=shape + (J["g"],))),
           "1|h": (("chain", "draw", "h__factor_dim"), rng.normal(size=shape + (J["h"],))), "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    idata = az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(4), "g__factor_dim": list(LEVELS_G), "h__factor_dim": list(LEVELS_H)}))
    P = idata.posterior
    new = pd.DataFrame({"x": [0.5, -1.0], "w": [0.1, 0.2], "g": ["newG", LEVELS_G[1]], "h": [LEVELS_H[0], "newH"]})
    eta = m.predict(idata, newdata=new, kind="linear", new_groups="population", seed=21)["eta"].to_numpy()
    rng = np.random.default_rng(21)
    u_g = rng.normal(0.0, 1.0, size=shape) * P["1|g_sigma"].to_numpy()                   # factor g first: its unseen label, intercept then slope
    v_g = rng.normal(0.0, 1.0, size=shape) * P["x|g_sigma"].to_numpy()
    u_h = rng.normal(0.0, 1.0, size=shape) * P["1|h_sigma"].to_numpy()                   # then factor h: one draw for its unseen label
    a, bx, bw = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["w"].to_numpy()
    row0 = a + bx * 0.5 + bw * 0.1 + u_g + 0.5 * v_g + P["1|h"].to_numpy()[..., 0]
    row1 = a + bx * -1.0 + bw * 0.2 + P["1|g"].to_numpy()[..., 1] + -1.0 * P["x|g"].to_numpy()[..., 1] + u_h
    assert np.allclose(eta[..., 0], row0, atol=1e-12) and np.allclose(eta[..., 1], row1, atol=1e-12)


# -- 6. describe() -------------------------------------------------------------------------------------------------
def test_describe_states_the_slope(single, two_factor):
    text = single.describe()
    for s in ["AutoBayes M0 model with independent varying slope(s)",
              "mu_i = a_c + (x_x - mean(x_x)) * b_x + u[g_i] + x_x,i * v_x|g[g_i]",
              "u_g[j] = sd_g * z_g[j], z_g[j] ~ Normal(0, 1), j = 0..3",
              "v_x|g[j] = sd_x|g * z_x|g[j], z_x|g[j] ~ Normal(0, 1), j = 0..3: the slope of x varies by g ('||': no correlation with u_g)",
              "multiplies the raw predictor (never centred, as brms's Z does)", "that independence is a statement at x = 0",
              "sd_g ~ HalfNormal(sigma=1)", "sd_x|g ~ HalfNormal(sigma=0.5)",
              "x|g_sigma_log__          log(sd_x|g)", "x|g_offset               z_x|g", "1|g_offset               z",
              "Deterministic: 1|g = u, x|g = v_x|g, mu = linear predictor"]:
        assert s in text, s
    text2 = two_factor.describe()
    assert "mu_i = a_c + (x_x - mean(x_x)) * b_x + (x_w - mean(x_w)) * b_w + u_g[g_i] + x_x,i * v_x|g[g_i] + u_h[h_i]" in text2
    assert "sd_x|g ~ Exponential(lam=3)" in text2 and "sd_h ~ HalfNormal(sigma=0.7)" in text2
    assert "1|g_offset               z_g" in text2 and "x|g_offset               z_x|g" in text2 and "1|h_offset               z_h" in text2
    assert "Deterministic: 1|g = u_g, x|g = v_x|g, 1|h = u_h, mu = linear predictor" in text2
    assert "J = 4 levels of g; J = 3 levels of h" in text2
