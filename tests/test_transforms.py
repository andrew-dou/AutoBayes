"""In-formula transforms ``log(x)``, ``exp(x)``, ``sqrt(x)``, ``I(x^k)``, ``I(x * k)``, ``I(x + k)``, ``I(x - k)`` (brms-parity
feature A05; docs/transforms.md).

Refusals with their messages, the term model and the canonical spelling, every supported form against NumPy, the
design and the hidden columns Bambi is handed, the graph's linear predictor and density at a point, by-hand and
Bambi-path predictions on fitted and new rows (recomputed from the raw columns), the priors, the fit record, the
target identity and ``describe()``. Build-only; nothing samples. The numbers against brms are the oracle's business
(tests/oracle, case ``gauss_transforms``)."""
import copy
import json
import re
from dataclasses import asdict

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes.formula import Offset, PopTerm, Transform, parse_transform, pop_term_design
from tests.test_population_only import PANEL_STUB

LEVELS = ("g0", "g1", "g2", "g3")


def frame(n=24, seed=6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.3, 1.0, n), "xpos": np.round(np.exp(rng.normal(0.0, 0.5, n)), 4),
                       "w": rng.normal(size=n), "z": rng.choice(["p", "q", "r"], size=n), "g": [LEVELS[i % 4] for i in range(n)],
                       "k": rng.integers(0, 4, n), "n": np.full(n, 5), "cnt": rng.poisson(2.0, n).astype(np.int64)})
    assert (df.xpos > 0).all() and set(df.z) == {"p", "q", "r"} and (df.x != 0).all()
    return df


PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))
POP_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5))
UNSUPPORTED = ("only log(col), exp(col), sqrt(col), I(col^k) with integer k, and I(col * k), I(col + k), I(col - k) with a numeric "
               "constant k are supported as in-formula transforms of one numeric column (docs/transforms.md, "
               "divergence D16); other function calls and transformations inside the formula are not supported: compute "
               "the column in the data frame and name it instead.")


@pytest.fixture(scope="module")
def model():
    """``y ~ log(xpos) + I(x^2) + (1 | g)``: the frozen oracle case's shape."""
    return ab.gaussian("y ~ log(xpos) + I(x^2) + (1 | g)", frame(), PRIORS)


# -- 1. the grammar: what is accepted and every refusal by message ----------------------------------------------
REFUSALS = [
    ("y ~ sin(x)", {}, f"'sin(x)': {UNSUPPORTED}"),
    ("y ~ scale(x) + (1 | g)", {}, UNSUPPORTED),
    ("y ~ poly(x, 2)", {}, UNSUPPORTED),
    ("y ~ log(x + 1)", {}, UNSUPPORTED),
    ("y ~ log(x, 2)", {}, UNSUPPORTED),
    ("y ~ log1p(x)", {}, UNSUPPORTED),
    ("y ~ I(x * w)", {}, UNSUPPORTED),
    ("y ~ I(x^2.5)", {}, UNSUPPORTED),
    ("y ~ I(x^w)", {}, UNSUPPORTED),
    ("y ~ I(2^x)", {}, UNSUPPORTED),
    ("y ~ I(x)", {}, UNSUPPORTED),
    ("y ~ I(1 - x)", {}, UNSUPPORTED),
    ("y ~ I(x / 2)", {}, UNSUPPORTED),
    ("y ~ exp(-x)", {}, UNSUPPORTED),
    ("y ~ np.log(x)", {}, UNSUPPORTED),
    ("y ~ log()", {}, UNSUPPORTED),
    ("y ~ log(log(x))", {}, UNSUPPORTED),
    ("y ~ log(y)", {}, "'log(y)': the response 'y' cannot be transformed into a predictor"),
    ("k | trials(n) ~ x + log(n)", {"family": "binomial"}, "'log(n)': the trials column 'n' cannot be transformed into a predictor"),
    ("y ~ log(g) + (1 | g)", {}, "'g': the grouping factor cannot also be a predictor or the response"),
    ("y ~ x + I(h^2) + (1 | g:h)", {}, "'h': the grouping factor cannot also be a predictor or the response"),
    ("y ~ log(z)", {"categorical": {"z": "p"}}, "'log(z)': 'z' is declared categorical; an in-formula transform applies to a numeric column"),
    ("y ~ x + z + log(x):z", {"categorical": {"z": "p"}}, "'log(x):z': an interaction with a transformed term ('log(x)') is not supported; compute the column in the data frame and interact the column instead."),
    ("y ~ x + z + sin(x):z", {"categorical": {"z": "p"}}, f"'sin(x)': {UNSUPPORTED}"),
    ("y ~ log(xpos) * z", {"categorical": {"z": "p"}}, "an interaction with a transformed term ('log(xpos)') is not supported"),
    ("y ~ log(xpos) + log( xpos )", {}, "'log(xpos)': duplicated predictor"),
    ("y ~ I(x^2) + I( x ^ 2 )", {}, "'I(x^2)': duplicated predictor"),
    ("y ~ x + (1 + log(x) || g)", {}, "function calls and transformations inside a group term are not supported"),
    ("y ~ x + (1 | log(g))", {}, "transformed grouping factors are not supported"),
    ("y ~ log(xpos) + (1 + xpos || g)", {}, "the varying slope 'xpos' must also be a population-level term of the formula"),
]


@pytest.mark.parametrize("formula,kwargs,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, **kwargs)


def test_data_refusals():
    df = frame()
    with pytest.raises(ab.FormulaError, match=re.escape("columns not in data: ['xpos']")):
        ab.parse_formula("y ~ log(xpos) + (1 | g)", df.drop(columns="xpos"))
    with pytest.raises(ab.FormulaError, match=re.escape("column 'xpos' must be positive for the term 'log(xpos)'; row 3 has 0.0")):
        ab.parse_formula("y ~ log(xpos)", df.assign(xpos=df.xpos.where(df.index != 3, 0.0)))
    with pytest.raises(ab.FormulaError, match=re.escape("column 'xpos' must be positive for the term 'sqrt(xpos)'; row 5 has -1.0")):
        ab.gaussian("y ~ sqrt(xpos)", df.assign(xpos=df.xpos.where(df.index != 5, -1.0)), POP_PRIORS)
    with pytest.raises(ab.FormulaError, match=re.escape("the term 'exp(x)' is not finite at row 0 (x = 800.0)")):
        ab.parse_formula("y ~ exp(x)", df.assign(x=df.x.where(df.index != 0, 800.0)))
    with pytest.raises(ab.FormulaError, match=re.escape("the term 'I(x^-1)' is not finite at row 2 (x = 0.0)")):
        ab.parse_formula("y ~ I(x^-1)", df.assign(x=df.x.where(df.index != 2, 0.0)))
    with pytest.raises(ab.FormulaError, match=r"column 'xpos' has dtype \w+; the term 'log\(xpos\)' needs a numeric column"):
        ab.parse_formula("y ~ log(xpos)", df.assign(xpos=df.xpos.astype(str)))
    with pytest.raises(ab.FormulaError, match=re.escape("column 'x' of the term 'I(x^2)' has missing or non-finite values")):
        ab.parse_formula("y ~ I(x^2)", df.assign(x=df.x.where(df.index > 0)))
    with pytest.raises(ab.FormulaError, match="is complex-valued"):
        ab.parse_formula("y ~ exp(x)", df.assign(x=df.x.astype(complex)))
    for formula in ("y ~ I(x^0)", "y ~ I(x * 0)", "y ~ I(0 * x)"):
        with pytest.raises(ab.FormulaError, match=re.escape(f"term '{formula[4:]}' is constant in the fitted rows")):
            ab.parse_formula(formula, df)
    # the raw column need not itself be a predictor, and negative values are fine where the function allows them
    assert ab.parse_formula("y ~ exp(x) + I(x^3) + I(x - 1)", df.assign(x=-df.x.abs())).columns == ("exp(x)", "I(x^3)", "I(x - 1)")


# -- 2. the term model and the canonical spelling ----------------------------------------------------------------
def test_term_model_and_canonical_spelling():
    s = ab.parse_formula("y ~ log(xpos) + I(x^2) + (1 | g)")
    lx, x2 = s.pop_terms
    assert s.pop_terms == (PopTerm("log(xpos)", "transform", ("log(xpos)",), (), ()), PopTerm("I(x^2)", "transform", ("I(x^2)",), (), ()))
    assert s.terms == s.columns == ("log(xpos)", "I(x^2)") and s.predictors == () and s.pop_columns == ("xpos", "x")
    assert s.transform_terms == s.pop_terms and lx.operands == ("xpos",) and x2.operands == ("x",) and lx.categorical == () and not lx.is_vector
    assert s.column_terms == (("log(xpos)", "log(xpos)", None), ("I(x^2)", "I(x^2)", None)) and s.interactions == ()
    assert s.rhs == "log(xpos) + I(x^2) + (1 | g)" and s.canonical_formula == "y ~ log(xpos) + I(x^2) + (1 | g)"
    assert s.bambi_formula == "y ~ `log(xpos)` + `I(x^2)` + (1 | g)"            # the evaluated columns, named as written
    assert lx.transform == Transform("log", "xpos") and lx.transform.term == "log(xpos)" and lx.transform.expression("xpos_i") == "log(xpos_i)"
    assert x2.transform == Transform("^", "x", "2") and x2.transform.expression("x_i") == "x_i^2"
    # spacing inside the call does not matter; the term is spelled canonically, constants as written
    many = ab.parse_formula("y ~ log( xpos ) + I( x ^ 2 ) + I(2*x) + I(x+1) + I( x - 1 ) + I(x*2.5) + sqrt(xpos) + exp(x) + I(1+x) + I(x^-1)")
    assert many.columns == ("log(xpos)", "I(x^2)", "I(2 * x)", "I(x + 1)", "I(x - 1)", "I(x * 2.5)", "sqrt(xpos)", "exp(x)", "I(1 + x)", "I(x^-1)")
    assert many.pop_columns == ("xpos", "x") and many.bambi_formula.startswith("y ~ `log(xpos)` + `I(x^2)` + `I(2 * x)` + `I(x + 1)`")
    assert [t.transform.expression("v") for t in many.pop_terms] == ["log(v)", "v^2", "2 * v", "v + 1", "v - 1", "v * 2.5", "sqrt(v)", "exp(v)", "1 + v", "v^-1"]
    # brms names the coefficient with its rename() of the term (checked against brms 2.23's rename)
    for written, name in [("log(xpos)", "logxpos"), ("I(x^2)", "IxE2"), ("exp(x)", "expx"), ("sqrt(xpos)", "sqrtxpos"), ("I(x * 2)", "IxMU2"),
                          ("I(x + 1)", "IxP1"), ("I(x - 1)", "IxM1"), ("I(2 * x)", "I2MUx"), ("I(x^-1)", "IxEM1"), ("I(x * 2.5)", "IxMU2.5")]:
        assert parse_transform(written).brms_name == name, written
    assert parse_transform("x") is None and parse_transform("x:z") is None and parse_transform("offset(x)") is None and parse_transform("sin(x)") is None
    with pytest.raises(ValueError, match="'sin\\(x\\)' is not a supported in-formula transform"):
        PopTerm("sin(x)", "transform", ("sin(x)",))
    # the raw column may also be a term, and the offset may share the column
    both = ab.parse_formula("y ~ x + I(x^2)")
    assert both.predictors == ("x",) and both.pop_columns == ("x",) and both.columns == ("x", "I(x^2)")
    shared = ab.parse_formula("cnt ~ log(exposure) + offset(log(exposure)) + (1 | g)", family="poisson")
    assert shared.offset == Offset("exposure", log=True) and shared.columns == ("log(exposure)",) and shared.rhs == "log(exposure) + offset(log(exposure)) + (1 | g)"
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):
        assert replica == s and hash(replica) == hash(s) and replica.pop_terms == s.pop_terms
    assert hash(s) != hash(ab.parse_formula("y ~ log(xpos) + I(x^3) + (1 | g)"))


def test_every_supported_form_against_numpy():
    df = frame()
    x, xpos = df.x.to_numpy(), df.xpos.to_numpy()
    for written, expected in [("log(xpos)", np.log(xpos)), ("exp(x)", np.exp(x)), ("sqrt(xpos)", np.sqrt(xpos)), ("I(x^2)", x ** 2), ("I(x^3)", x ** 3),
                              ("I(xpos^-1)", xpos ** -1.0), ("I(x * 2)", x * 2), ("I(2 * x)", 2 * x), ("I(x * -1.5)", x * -1.5), ("I(x + 1)", x + 1),
                              ("I(1 + x)", 1 + x), ("I(x - 1)", x - 1), ("I(x + .5)", x + 0.5), ("I(x * 1e-1)", x * 0.1)]:
        s = ab.parse_formula(f"y ~ {written}", df)
        (t,) = s.pop_terms
        assert t.kind == "transform" and t.name == written, written
        assert np.array_equal(pop_term_design(t, df)[:, 0], expected) and np.array_equal(t.transform.values(df), expected), written


# -- 3. the design, the hidden columns and the names -----------------------------------------------------------------
def test_design_and_hidden_columns(model):
    m, d, df = model, model.design, frame()
    lx, x2 = np.log(df.xpos.to_numpy()), df.x.to_numpy() ** 2
    assert d.columns == ("log(xpos)", "I(x^2)") and d.X_raw.shape == (24, 2) and d.predictors == ()
    assert np.array_equal(d.X_raw[:, 0], lx) and np.array_equal(d.X_raw[:, 1], x2)
    assert np.allclose(d.X_model, d.X_raw - d.X_raw.mean(0)) and np.allclose(d.x_means, [lx.mean(), x2.mean()]) and d.centered
    # the fitted frame carries the evaluated columns under the terms' names next to the raw columns; Bambi coded them as numeric
    assert np.array_equal(m.data["log(xpos)"].to_numpy(), lx) and np.array_equal(m.data["I(x^2)"].to_numpy(), x2) and {"x", "xpos"} <= set(m.data.columns)
    comp = m.bambi_model.components["mu"].design.common
    assert list(comp.terms) == ["Intercept", "log(xpos)", "I(x^2)"] and comp.terms["log(xpos)"].kind == "numeric" and list(comp.terms["I(x^2)"].labels) == ["I(x^2)"]
    assert {v.name for v in m.pymc_model.free_RVs} == {"Intercept", "log(xpos)", "I(x^2)", "sigma", "1|g_sigma", "1|g_offset"}
    assert m.names["b:log(xpos)"] == "log(xpos)" and m.names["b:I(x^2)"] == "I(x^2)"
    assert m.coefficients == (("log(xpos)", "log(xpos)", None), ("I(x^2)", "I(x^2)", None))
    assert m.priors.coefficient_priors(m.spec) == {"log(xpos)": PRIORS.b, "I(x^2)": PRIORS.b}
    assert set(m.priors.to_bambi(m.spec, "gaussian")) == {"Intercept", "sigma", "log(xpos)", "I(x^2)", "1|g"}
    assert np.array_equal(np.concatenate([pop_term_design(t, m.data) for t in m.spec.pop_terms], axis=1), d.X_raw)
    # a model on precomputed columns is a different target (different column names), with the same numbers
    pre = ab.gaussian("y ~ lx + x2 + (1 | g)", df.assign(lx=lx, x2=x2), PRIORS)
    assert np.array_equal(pre.design.X_raw, d.X_raw) and pre.target_identity != m.target_identity
    # a transform next to its own raw column, and a transform of the offset column
    m2 = ab.gaussian("y ~ x + I(x^2) + (1 | g)", df, PRIORS)
    assert m2.spec.columns == ("x", "I(x^2)") and np.array_equal(m2.design.X_raw[:, 1], x2)
    m3 = ab.poisson("cnt ~ log(xpos) + offset(log(xpos)) + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    assert np.array_equal(m3.design.offset, lx) and np.array_equal(m3.design.X_raw[:, 0], lx)


def natural(k=0):
    return dict(intercept=0.3 + 0.1 * k, b=np.array([0.5 - 0.05 * k, -0.2 + 0.05 * k]), sd=0.8 - 0.1 * k,
                z=np.array([-0.4, 0.2, 0.9, -0.1]) + 0.1 * k, sigma=0.9 + 0.1 * k)


def test_graph_linear_predictor_and_density_at_a_point(model):
    m, d = model, model.design
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    logp, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    for k in range(3):
        nat = natural(k)
        pt = m.value_point(**nat)
        assert float(pt["log(xpos)"]) == nat["b"][0] and float(pt["I(x^2)"]) == nat["b"][1]
        eta = nat["intercept"] + d.X_model @ nat["b"] + d.group_designs["g"].Z @ (nat["sd"] * nat["z"])
        assert np.allclose(np.asarray(fn(pt)[0]).ravel(), eta, atol=1e-12)
        ref = (norm.logpdf(nat["intercept"], 0.2, 5) + norm.logpdf(nat["b"], 0.1, 2).sum() + norm.logpdf(nat["z"]).sum()
               + halfnorm.logpdf(nat["sd"], scale=1) + halfnorm.logpdf(nat["sigma"], scale=1.5) + norm.logpdf(d.y, eta, nat["sigma"]).sum())
        assert abs(float(nojac(pt)) - ref) < 1e-8 and abs(float(logp(pt)) - ref - np.log(nat["sd"]) - np.log(nat["sigma"])) < 1e-8


# -- 4. predictions on fitted and new rows ----------------------------------------------------------------------
def synthetic(draws=4, seed=8):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    var = {"Intercept": (("chain", "draw"), rng.normal(0.2, 0.1, shape)), "log(xpos)": (("chain", "draw"), rng.normal(0.4, 0.1, shape)),
           "I(x^2)": (("chain", "draw"), rng.normal(-0.2, 0.1, shape)), "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (4,))), "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(LEVELS)}))


def hand_eta(post, df, with_groups=True):
    eta = (post["Intercept"].to_numpy()[..., None] + post["log(xpos)"].to_numpy()[..., None] * np.log(df.xpos.to_numpy(dtype=float))
           + post["I(x^2)"].to_numpy()[..., None] * df.x.to_numpy(dtype=float) ** 2)
    return eta + post["1|g"].to_numpy()[..., [LEVELS.index(g) for g in df.g]] if with_groups else eta


def test_predictions_recompute_the_transforms_from_the_raw_columns(model):
    m, idata = model, synthetic()
    P = idata.posterior
    eta = hand_eta(P, frame())
    assert np.allclose(m.predict(idata, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
    assert np.allclose(m.predict(idata, kind="mean")["mu"].to_numpy(), eta, atol=1e-10)                      # Bambi's path agrees
    assert np.allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"].to_numpy(), hand_eta(P, frame(), False), atol=1e-12)
    new = pd.DataFrame({"x": [0.0, 1.5, -2.0, 0.5], "xpos": [1.0, 2.5, 0.5, 4.0], "g": ["g2", "g0", "g3", "g1"]})
    eta_new = hand_eta(P, new)
    assert np.allclose(m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy(), eta_new, atol=1e-12)
    assert np.allclose(m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy(), eta_new, atol=1e-10)   # Bambi evaluates the hidden columns
    assert np.allclose(m.predict(idata, newdata=new.drop(columns="g"), kind="linear", include_group_effects=False)["eta"].to_numpy(), hand_eta(P, new, False), atol=1e-12)
    # row 0 has xpos = 1 and x = 0: both transforms vanish, so eta is the raw intercept plus the group effect
    assert np.allclose(m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy()[..., 0], P["Intercept"].to_numpy() + P["1|g"].to_numpy()[..., 2], atol=1e-12)
    # a stale evaluated column in newdata is ignored: the transforms are recomputed from xpos and x
    stale = new.assign(**{"log(xpos)": 99.0, "I(x^2)": -99.0})
    assert np.allclose(m.predict(idata, newdata=stale, kind="linear")["eta"].to_numpy(), eta_new, atol=1e-12)
    assert np.allclose(m.predict(idata, newdata=stale, kind="mean")["mu"].to_numpy(), eta_new, atol=1e-10)
    resp = m.predict(idata, newdata=new, kind="response", seed=3)
    assert resp["y"].shape == eta_new.shape and np.allclose(resp["mu"].to_numpy(), eta_new, atol=1e-10)
    # the raw columns are required on new rows and must be valid there
    with pytest.raises(ab.FormulaError, match=re.escape("newdata lacks columns ['xpos']")):
        m.predict(idata, newdata=new.drop(columns="xpos"), kind="linear")
    with pytest.raises(ab.FormulaError, match=re.escape("newdata lacks columns ['x']")):
        m.predict(idata, newdata=new.drop(columns="x").assign(**{"I(x^2)": 1.0}), kind="linear")
    with pytest.raises(ab.FormulaError, match=re.escape("newdata column 'xpos' must be positive for the term 'log(xpos)'; row 1 has 0.0")):
        m.predict(idata, newdata=new.assign(xpos=[1.0, 0.0, 0.5, 4.0]), kind="linear")
    with pytest.raises(ab.FormulaError, match="newdata column 'x' must be numeric"):
        m.predict(idata, newdata=new.assign(x=["a", "b", "c", "d"]), kind="linear")
    with pytest.raises(ab.FormulaError, match="newdata contains missing values"):
        m.predict(idata, newdata=new.assign(xpos=[1.0, np.nan, 0.5, 4.0]), kind="linear")


# -- 5. priors, records and describe() ---------------------------------------------------------------------------
def test_prior_forms_and_refusals():
    df = frame()
    def pri(b):
        return ab.Priors(intercept=ab.Normal(0.2, 5), b=b, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))
    per_term = pri({"log(xpos)": ab.Normal(0, 2), "I(x^2)": ab.StudentT(4, 0, 1)})
    m = ab.gaussian("y ~ log(xpos) + I(x^2) + (1 | g)", df, per_term)
    assert per_term.coefficient_priors(m.spec) == {"log(xpos)": ab.Normal(0, 2), "I(x^2)": ab.StudentT(4, 0, 1)}
    assert "b_I(x^2) ~ StudentT(nu=4, mu=0, sigma=1)" in m.describe()
    for b, msg in [({"log(xpos)": ab.Normal(0, 2)}, "coefficient priors must cover exactly the predictor terms ['log(xpos)', 'I(x^2)']; missing ['I(x^2)'], unexpected []"),
                   ({"log(xpos)": ab.Normal(0, 2), "I(x^2)": ab.Normal(0, 1), "xpos": ab.Normal(0, 1)}, "unexpected ['xpos']"),
                   ({"log(xpos)": {"a": ab.Normal(0, 2)}, "I(x^2)": ab.Normal(0, 1)}, "predictor 'log(xpos)' takes one prior, got dict"),
                   ({"log(xpos)": ab.HalfNormal(1), "I(x^2)": ab.Normal(0, 1)}, "prior for 'log(xpos)' must be a location prior")]:
        with pytest.raises(ab.PriorError, match=re.escape(msg)):
            ab.gaussian("y ~ log(xpos) + I(x^2) + (1 | g)", df, pri(b))


def test_fit_record_and_describe(model):
    m = model
    idata = synthetic(draws=50)
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 50, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(m.target_identity))
    rec = ab.fit_record(m, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert rec["model"]["transforms"] == ["log(xpos)", "I(x^2)"] and rec["model"]["terms"] == rec["model"]["design_columns"] == ["log(xpos)", "I(x^2)"]
    assert rec["model"]["interactions"] == [] and rec["model"]["priors"]["b:log(xpos)"] == "Normal(mu=0.1, sigma=2)"
    assert {"log(xpos)", "I(x^2)"} <= set(rec["posterior"]) and np.isclose(rec["posterior"]["I(x^2)"]["mean"], idata.posterior["I(x^2)"].to_numpy().mean())
    text = m.describe()
    lx = np.log(frame().xpos.to_numpy())
    for s in ["AutoBayes Gaussian model with in-formula transform(s)",
              f"  log(xpos) : in-formula transform of 'xpos', log(xpos_i), range {lx.min():.4f} to {lx.max():.4f} over the fitted rows; recomputed from 'xpos' "
              "on new rows; centred like any column; brms names its coefficient logxpos",
              "  I(x^2)    : in-formula transform of 'x', x_i^2,",
              "brms names its coefficient IxE2",
              "mu_i = a_c + (log(xpos_i) - mean(log(xpos))) * b_log(xpos) + (x_i^2 - mean(I(x^2))) * b_I(x^2) + u[g_i]",
              "log(xpos): b_log(xpos) is the slope in log(xpos), not in xpos; a unit change of log(xpos) changes mu by b_log(xpos), other predictors and group effects fixed.",
              "I(x^2): b_I(x^2) is the slope in I(x^2), not in x;",
              "a_c is the expected response at the predictor means (group effect zero)",
              "b_log(xpos) ~ Normal(mu=0.1, sigma=2)", "b_I(x^2) ~ Normal(mu=0.1, sigma=2)",
              f"  {'log(xpos)':<24} b_log(xpos)", f"  {'I(x^2)':<24} b_I(x^2)"]:
        assert s in text, s
    assert "with offset" not in text and "interaction" not in text
    uncentred = ab.gaussian("y ~ log(xpos) + I(x^2) + (1 | g)", frame(), PRIORS, center_predictors=False).describe()
    assert "mu_i = a + log(xpos_i) * b_log(xpos) + x_i^2 * b_I(x^2) + u[g_i]" in uncentred
    plain = ab.gaussian("y ~ x + (1 | g)", frame(), PRIORS).describe()
    assert "transform" not in plain
