"""The term model (brms-parity loop, P4): a parsed formula is an explicit tuple of ``PopTerm`` and ``GroupTerm``,
the design holds one ``GroupBlock`` per (factor, coefficient), and every flat view of the formula (predictors,
terms, columns, groups, ...) is derived from the terms. Build-only; nothing samples."""
import copy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from autobayes.formula import GroupTerm, PopTerm, pop_term, pop_term_design
from autobayes.model import GroupBlock
from autobayes.predict import raw_design

PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)},
                   sigma=ab.HalfNormal(1))


def frame(n=24, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.choice(["p", "q", "r"], size=n),
                       "g": [f"g{i % 4}" for i in range(n)], "h": [f"h{i % 3}" for i in range(n)],
                       "k": rng.integers(0, 5, n), "size": np.full(n, 5)})
    assert set(df.z) == {"p", "q", "r"}
    return df


def full_model() -> ab.Model:
    return ab.gaussian("y ~ x * z + (1 | g) + (1 | h)", frame(), PRIORS, categorical={"z": "p"})


# -- the terms of a few formulas ---------------------------------------------------------------------------
def test_terms_of_a_varying_intercept_formula():
    s = ab.parse_formula("y ~ x + (1 | g)")
    assert s.pop_terms == (PopTerm("x", "numeric", ("x",), ()),)
    assert s.group_terms == (GroupTerm("g", ("Intercept",), True),)
    m = ab.gaussian("y ~ x + (1 | g)", frame(), ab.Priors(ab.Normal(0, 5), ab.Normal(0, 2), ab.HalfNormal(1), ab.HalfNormal(1)))
    assert m.spec.pop_terms == s.pop_terms and m.spec.group_terms == s.group_terms
    assert m.spec.levels_by_group == {"g": ("g0", "g1", "g2", "g3")}


def test_terms_of_an_interaction_formula_with_two_factors():
    before = ab.parse_formula("y ~ x * z + (1 | g) + (1 | h)", categorical={"z": "p"})
    assert before.pop_terms == (PopTerm("x", "numeric", ("x",), ()),               # levels unknown until the data are seen
                                PopTerm("z", "categorical", ("z",), ()),
                                PopTerm("x:z", "interaction", ("x:z",), ()))
    assert before.group_terms == (GroupTerm("g", ("Intercept",), True), GroupTerm("h", ("Intercept",), True))
    m = full_model()
    assert m.spec.pop_terms == (PopTerm("x", "numeric", ("x",), ()),
                                PopTerm("z", "categorical", ("z[q]", "z[r]"), ("q", "r")),
                                PopTerm("x:z", "interaction", ("x:z[q]", "x:z[r]"), ("q", "r")))
    assert m.spec.group_terms == before.group_terms
    assert m.spec.pop_term("x:z").operands == ("x", "z") and m.spec.pop_term("z").is_vector and not m.spec.pop_term("x").is_vector


def test_terms_of_a_no_intercept_formula_code_the_first_categorical_full_rank():
    m = ab.gaussian("y ~ 0 + x + z", frame(), ab.Priors(intercept=None, b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)),
                    categorical={"z": "p"})
    assert m.spec.intercept is False and m.spec.group_terms == ()
    assert m.spec.pop_terms == (PopTerm("x", "numeric", ("x",), ()),
                                PopTerm("z", "categorical", ("z[p]", "z[q]", "z[r]"), ("p", "q", "r")))
    assert m.design.group_blocks == {} and m.design.Z.shape == (24, 0) and m.design.group_designs == {}


def test_terms_of_a_binomial_formula():
    s = ab.parse_formula("k | trials(size) ~ x", family="binomial")
    assert s.pop_terms == (PopTerm("x", "numeric", ("x",), ()),) and s.group_terms == () and s.trials == "size"
    m = ab.binomial("k | trials(size) ~ x", frame(), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1)))
    assert m.spec.pop_terms == s.pop_terms and m.names["b:x"] == "x" and m.names["y"] == "p(k, size)"


# -- the group blocks --------------------------------------------------------------------------------------
def test_group_blocks_are_the_one_hot_indicators_in_formula_order():
    m = full_model()
    d, df = m.design, m.data
    assert list(d.group_blocks) == [("g", "Intercept"), ("h", "Intercept")]
    start = 0
    for (factor, coef), block in d.group_blocks.items():
        assert isinstance(block, GroupBlock) and coef == "Intercept"
        levels = m.spec.levels_by_group[factor]
        indicator = np.column_stack([df[factor].to_numpy() == level for level in levels]).astype(float)
        assert block.levels == levels and block.J == len(levels) and block.Z.shape == (d.n, block.J)
        assert np.array_equal(block.Z, indicator)
        assert np.array_equal(block.group_index, indicator.argmax(1))
        assert block.column_slice == slice(start, start + block.J)
        assert np.array_equal(d.Z[:, block.column_slice], block.Z)
        start += block.J
    assert d.Z.shape == (d.n, start)
    assert d.group_designs == {"g": d.group_blocks[("g", "Intercept")], "h": d.group_blocks[("h", "Intercept")]}
    assert d.group_slices == {"g": slice(0, 4), "h": slice(4, 7)}
    with pytest.raises(ValueError, match="ambiguous"):
        d.levels


# -- the derived views agree with the terms ------------------------------------------------------------------
def test_flat_views_are_derived_from_the_terms():
    m = full_model()
    s = m.spec
    assert s.terms == tuple(t.name for t in s.pop_terms) == ("x", "z", "x:z")
    assert s.predictors == tuple(t.name for t in s.pop_terms if t.kind == "numeric") == ("x",)
    assert s.categorical_names == ("z",) and s.interaction_names == ("x:z",) and s.interactions == (("x", "z"),)
    assert s.interaction_of("x:z") == ("x", "z")
    assert s.columns == tuple(c for t in s.pop_terms for c in t.columns) == ("x", "z[q]", "z[r]", "x:z[q]", "x:z[r]")
    assert s.column_terms == (("x", "x", None), ("z[q]", "z", 0), ("z[r]", "z", 1), ("x:z[q]", "x:z", 0), ("x:z[r]", "x:z", 1))
    for t in s.pop_terms:
        assert s.levels_of(t.name) == t.levels and s.term_columns(t.name) == t.columns
        assert s.is_vector_term(t.name) is t.is_vector
    assert s.levels_of("absent") == () and s.term_columns("absent") == ("absent",) and not s.is_vector_term("absent")
    assert s.groups == tuple(t.factor for t in s.group_terms) == ("g", "h") and s.has_group
    assert m.coefficients == s.column_terms
    assert {k: v for k, v in m.names.items() if k.startswith("b:")} == {f"b:{t.name}": t.name for t in s.pop_terms}
    assert list(m.group_names) == ["g", "h"]
    assert m.group_names["g"] == {"sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "coordinate": "g__factor_dim"}
    assert m.priors.coefficient_priors(s).keys() == set(s.columns)


def test_population_design_is_the_terms_blocks_side_by_side():
    m = full_model()
    blocks = [pop_term_design(t, m.data) for t in m.spec.pop_terms]
    assert [b.shape for b in blocks] == [(24, 1), (24, 2), (24, 2)]
    assert np.array_equal(np.concatenate(blocks, axis=1), m.design.X_raw)
    assert np.array_equal(raw_design(m, m.data), m.design.X_raw)
    labels = m.data["z"].astype(str).to_numpy()
    assert np.array_equal(blocks[1], np.column_stack([labels == "q", labels == "r"]).astype(float))
    assert np.array_equal(blocks[2], m.data["x"].to_numpy()[:, None] * blocks[1])


def test_value_point_places_b_term_by_term():
    m = full_model()
    b = np.arange(1.0, 6.0)
    point = m.value_point(intercept=0.3, b=b, sd={"g": 0.5, "h": 0.7}, z={"g": np.zeros(4), "h": np.zeros(3)}, sigma=0.4)
    assert point["x"].shape == () and float(point["x"]) == 1.0
    assert np.array_equal(point["z"], [2.0, 3.0]) and np.array_equal(point["x:z"], [4.0, 5.0])
    with pytest.raises(ValueError, match="5 entries"):
        m.value_point(intercept=0.3, b=b[:3], sd={"g": 0.5, "h": 0.7}, z={"g": np.zeros(4), "h": np.zeros(3)}, sigma=0.4)


# -- the specification object ------------------------------------------------------------------------------
def test_spec_round_trips_and_keeps_the_original_constructor():
    s = ab.parse_formula("y ~ x * z + (1 | g) + (1 | h)", frame(), categorical={"z": "p"})
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):        # asdict hands the terms back as mappings
        assert replica == s and hash(replica) == hash(s) and replica.pop_terms == s.pop_terms
    assert replace(s, intercept=False).pop_terms == s.pop_terms and replace(s, intercept=False).groups == ("g", "h")
    legacy = ab.ModelSpec("y ~ x + w + (1 | g)", "y", ("x", "w"), "g", True, ("A", "B"))
    assert legacy.pop_terms == (PopTerm("x", "numeric", ("x",), ()), PopTerm("w", "numeric", ("w",), ()))
    assert legacy.group_terms == (GroupTerm("g", ("Intercept",), True),) and legacy.levels == ("A", "B")
    with pytest.raises(ValueError, match="not both"):
        ab.ModelSpec("y ~ x", "y", ("x",), pop_terms=legacy.pop_terms)
    with pytest.raises(ValueError, match="not both"):
        ab.ModelSpec("y ~ x", "y", group="g", group_terms=legacy.group_terms)
    with pytest.raises(ValueError, match="kind must be one of"):
        PopTerm("x", "slope", ("x",))
    cat = ab.Categorical("z", "p", levels=("p", "q", "r"))
    assert pop_term("z", "categorical", (cat,)) == PopTerm("z", "categorical", ("z[q]", "z[r]"), ("q", "r"))
    assert pop_term("x:z", "interaction", (cat,)) == PopTerm("x:z", "interaction", ("x:z[q]", "x:z[r]"), ("q", "r"))
    assert pop_term("x", "numeric") == PopTerm("x", "numeric", ("x",), ())
