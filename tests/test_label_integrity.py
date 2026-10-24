"""M3a-1 repair record: categorical labels keep their identity (R1-R6 of docs/history/m3a1-decision.md)."""
import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from autobayes.formula import categorical_labels
from tests.test_categorical import G_PRIORS, L_PRIORS

BIG = [2**53, 2**53 + 1, 2**53 + 2]
UNS = [0, 2**63, 2**64 - 1]


def frame(codes, dtype):
    z = pd.Series(codes, dtype=dtype) if dtype != "category" else pd.Series(pd.Categorical(codes))
    return pd.DataFrame({"y": np.arange(len(codes), dtype=float), "z": z, "g": ["a", "b"] * (len(codes) // 2)})


# -- R2: exact integer labels in every container -----------------------------------------------------
@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64", "UInt64", "object", "category"])
def test_large_integer_labels_are_exact_in_every_container(dtype):
    labels, unused = categorical_labels("z", frame(BIG * 2, dtype)["z"])
    assert labels.tolist() == [str(v) for v in BIG * 2] and unused == ()


@pytest.mark.parametrize("dtype", ["uint64", "UInt64", "object", "category"])
def test_unsigned_labels_at_and_above_2_63_are_exact(dtype):
    labels, _ = categorical_labels("z", frame(UNS * 2, dtype)["z"])
    assert labels.tolist() == [str(v) for v in UNS * 2]


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "object", "category"])
@pytest.mark.parametrize("reference", [2**53, "9007199254740992", np.uint64(2**53)])
def test_public_model_keeps_three_large_codes_and_original_design(dtype, reference):
    raw = BIG * 4
    df = frame(raw, dtype)
    m = ab.gaussian("y ~ z + (1 | g)", df, G_PRIORS, categorical={"z": reference})
    cat = m.spec.categorical("z")
    assert cat.levels == tuple(str(v) for v in BIG) and cat.reference == str(BIG[0]) and cat.coefficients == tuple(str(v) for v in BIG[1:])
    expected = np.column_stack([np.asarray(raw, dtype=object) == v for v in BIG[1:]]).astype(float)   # exact integer comparison on the raw values
    assert np.array_equal(m.design.X_raw, expected)
    assert m.original_labels["z"].tolist() == [str(v) for v in raw]
    assert list(m.pymc_model.coords["z_dim"]) == [str(v) for v in BIG[1:]]


# -- R3: floats only inside the exact-integer domain -------------------------------------------------
def test_integral_floats_inside_the_exact_domain_are_labels_and_others_are_refused():
    labels, _ = categorical_labels("z", pd.Series([2.0, 1.0, float(2**53), -3.0]))
    assert labels.tolist() == ["2", "1", "9007199254740992", "-3"]
    for bad, msg in [(float(2**53) + 2, "exceeds 2\\*\\*53"), (1e300, "exceeds 2\\*\\*53"), (float("inf"), "not a finite"),
                     (1.5, "integral")]:
        with pytest.raises(ab.FormulaError, match=msg):
            categorical_labels("z", pd.Series([1.0, bad, 3.0]))
    with pytest.raises(ab.FormulaError, match="missing values"):
        categorical_labels("z", pd.Series([1.0, float("nan"), 3.0]))
    df = frame([1.0, 2.0, 3.0] * 4, "float64")
    with pytest.raises(ab.FormulaError, match="exceeds 2\\*\\*53"):
        ab.gaussian("y ~ z + (1 | g)", df.assign(z=df.z.replace(3.0, float(2**53) + 2)), G_PRIORS, categorical={"z": 1.0})
    with pytest.raises(ab.FormulaError, match="reference level 1e+300 refused|exceeds 2\\*\\*53"):
        ab.gaussian("y ~ z + (1 | g)", df, G_PRIORS, categorical={"z": 1e300})


# -- R4: scalar rules in every container, and for references, categories and prior keys -------------
@pytest.mark.parametrize("dtype", ["complex128", "object", "category"])
def test_complex_values_are_refused_in_every_container(dtype):
    with pytest.raises(ab.FormulaError, match="complex"):
        categorical_labels("z", pd.Series([1 + 0j, 2 + 1j, 3 + 0j], dtype=dtype))
    df = frame([1, 2, 3] * 4, "int64")
    with pytest.raises(ab.FormulaError, match="complex"):
        ab.gaussian("y ~ z + (1 | g)", df.assign(z=pd.Series([1 + 0j, 2 + 1j, 3 + 0j] * 4, dtype=dtype)), G_PRIORS, categorical={"z": 1})


def test_reference_declared_category_and_prior_keys_follow_the_same_scalar_rule():
    df = frame([1, 2, 3] * 4, "int64")
    for bad, msg in [(1 + 0j, "complex"), (True, "boolean"), (1.5, "integral"), (b"1", "unsupported"), (None, "explicit reference")]:
        with pytest.raises(ab.FormulaError, match=msg):
            ab.gaussian("y ~ z + (1 | g)", df, G_PRIORS, categorical={"z": bad})
    with pytest.raises(ab.FormulaError, match="declared category .* complex"):
        categorical_labels("z", pd.Series(pd.Categorical([1, 2, 1], categories=[1, 2, 3 + 1j])))
    with pytest.raises(ab.FormulaError, match="mixes text labels and numeric codes"):
        categorical_labels("z", pd.Series(pd.Categorical(["a", 1, "a"], categories=["a", 1])))

    def pri(keys):
        return ab.Priors(intercept=ab.Normal(0, 5), b={"z": {k: ab.Normal(0, 1) for k in keys}}, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    for keys, msg in [((2 + 0j, 3), "complex"), ((True, 3), "boolean"), ((2.5, 3), "integral"), ((2, "2", 3), "ambiguous")]:
        with pytest.raises(ab.PriorError, match=msg):
            ab.gaussian("y ~ z + (1 | g)", df, pri(keys), categorical={"z": 1})
    m = ab.gaussian("y ~ z + (1 | g)", df, pri((2.0, "3")), categorical={"z": "1"})          # intentional equivalences: 2.0 == "2", "1" == 1
    assert [str(p) for p in m.priors.coefficient_priors(m.spec).values()] == ["Normal(mu=0, sigma=1)"] * 2


@pytest.mark.parametrize("dtype", ["bool", "object", "category"])
def test_booleans_and_missing_values_are_refused_in_every_container(dtype):
    with pytest.raises(ab.FormulaError, match="boolean"):
        categorical_labels("z", pd.Series([True, False, True], dtype=dtype))
    for missing in [None, float("nan"), pd.NA]:
        with pytest.raises(ab.FormulaError, match="missing values"):
            categorical_labels("z", pd.Series(["a", missing, "b"], dtype="object" if dtype != "category" else "category"))
    with pytest.raises(ab.FormulaError, match="missing values"):
        categorical_labels("z", pd.Series([1, None, 3], dtype="Int64"))


def test_mixed_text_and_numeric_columns_are_refused_as_ambiguous():
    with pytest.raises(ab.FormulaError, match="mixes text labels and numeric codes"):
        categorical_labels("z", pd.Series(["1", 1, 2], dtype="object"))
    df = frame(["1", 1, 2] * 4, "object")
    with pytest.raises(ab.FormulaError, match="mixes text labels and numeric codes"):
        ab.gaussian("y ~ z + (1 | g)", df, G_PRIORS, categorical={"z": 1})


# -- R5: predictions keep the identity of newdata codes --------------------------------------------------
def test_newdata_codes_keep_identity_across_containers():
    raw = BIG * 4
    m = ab.gaussian("y ~ z + (1 | g)", frame(raw, "int64"), G_PRIORS, categorical={"z": 2**53})
    fit = m.fit(backend="pymc", chains=1, draws=40, tune=40, seed=1, cores=1)
    base = m.predict(fit, newdata=pd.DataFrame({"z": BIG, "g": ["a"] * 3}), kind="mean")["mu"].to_numpy()
    assert not np.allclose(base[..., 0], base[..., 1]) and not np.allclose(base[..., 1], base[..., 2])   # three distinct levels
    for dtype in ["uint64", "Int64", "UInt64", "object", "category"]:
        nd = pd.DataFrame({"z": pd.Series(BIG, dtype=dtype) if dtype != "category" else pd.Series(pd.Categorical(BIG)), "g": ["a"] * 3})
        assert np.array_equal(m.predict(fit, newdata=nd, kind="mean")["mu"].to_numpy(), base)
    strings = pd.DataFrame({"z": [str(v) for v in BIG], "g": ["a"] * 3})
    assert np.array_equal(m.predict(fit, newdata=strings, kind="mean")["mu"].to_numpy(), base)
    with pytest.raises(ab.FormulaError, match="not in the fitted rows: \\['9007199254740995'\\]"):
        m.predict(fit, newdata=pd.DataFrame({"z": [2**53 + 3], "g": ["a"]}), kind="mean")
    with pytest.raises(ab.FormulaError, match="exceeds 2\\*\\*53"):
        m.predict(fit, newdata=pd.DataFrame({"z": [float(2**53) + 2], "g": ["a"]}), kind="mean")
    with pytest.raises(ab.FormulaError, match="complex"):
        m.predict(fit, newdata=pd.DataFrame({"z": pd.Series([2**53 + 0j], dtype=object), "g": ["a"]}), kind="mean")


def test_binomial_and_bernoulli_paths_share_the_contract():
    raw = UNS * 4
    df = frame(raw, "uint64").assign(k=[1, 2, 3] * 4, n=5, y01=[0, 1] * 6)
    mb = ab.binomial("k | trials(n) ~ z + (1 | g)", df, L_PRIORS, categorical={"z": 2**64 - 1})
    assert mb.spec.categorical("z").levels == ("0", "18446744073709551615", "9223372036854775808") and mb.spec.columns == ("z[0]", "z[9223372036854775808]")
    expected = np.column_stack([np.asarray(raw, dtype=object) == v for v in (0, 2**63)]).astype(float)
    assert np.array_equal(mb.design.X_raw, expected)
    me = ab.bernoulli("y01 ~ z + (1 | g)", df, L_PRIORS, categorical={"z": "0"})
    assert np.array_equal(me.design.X_raw, np.column_stack([np.asarray(raw, dtype=object) == v for v in (2**64 - 1, 2**63)]).astype(float))
