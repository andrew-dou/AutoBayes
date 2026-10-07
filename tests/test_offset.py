"""The offset term, ``offset(col)`` and ``offset(log(col))`` (brms-parity feature A06; docs/offset.md).

Refusals with their messages, the term model (no design column, no coefficient, no prior), the design's offset
vector against the data, the graph's linear predictor equal to ``X b + offset + Z u`` at a point, by-hand and
Bambi-path predictions on fitted and new rows with and without group effects, newdata without the column refused,
the target identity, the fit record and ``describe()``. Build-only; nothing samples. The numbers against brms
are the oracle's business (tests/oracle, case ``poisson_offset``)."""
import copy
import json
import re
from dataclasses import asdict

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import autobayes as ab
from autobayes.formula import Offset, PopTerm
from tests.test_population_only import PANEL_STUB

LEVELS = ("g0", "g1", "g2", "g3")


def frame(n=24, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.3, 1.0, n), "w": rng.normal(size=n),
                       "exposure": np.round(np.exp(rng.normal(1.0, 0.4, n)), 4),
                       "z": rng.choice(["p", "q", "r"], size=n), "g": [LEVELS[i % 4] for i in range(n)],
                       "k": rng.integers(0, 4, n), "n": np.full(n, 5)})
    df["cnt"] = rng.poisson(df.exposure * np.exp(0.2 + 0.4 * df.x)).astype(np.int64)
    assert set(df.z) == {"p", "q", "r"} and (df.exposure > 0).all()
    return df


POISSON_PRIORS = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
GAUSS_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))


@pytest.fixture(scope="module")
def poisson():
    """``cnt ~ x + offset(log(exposure)) + (1 | g)``: the frozen oracle case's shape."""
    return ab.poisson("cnt ~ x + offset(log(exposure)) + (1 | g)", frame(), POISSON_PRIORS)


@pytest.fixture(scope="module")
def gauss():
    """``y ~ x + offset(exposure) + (1 | g)``: an identity link, so the graph's ``mu`` is the linear predictor."""
    return ab.gaussian("y ~ x + offset(exposure) + (1 | g)", frame(), GAUSS_PRIORS)


# -- 1. the grammar: what is accepted and every refusal by message -------------------------------------------
REFUSALS = [
    ("cnt ~ x + offset(exp(exposure)) + (1 | g)", {}, "only 'offset(col)' and 'offset(log(col))' are supported inside offset()"),
    ("cnt ~ x + offset(exposure + 1)", {}, "only 'offset(col)' and 'offset(log(col))' are supported inside offset()"),
    ("cnt ~ x + offset(log(exposure) + 1)", {}, "only 'offset(col)' and 'offset(log(col))'"),
    ("cnt ~ x + offset(log(exposure, 2))", {}, "only 'offset(col)' and 'offset(log(col))'"),
    ("cnt ~ x + offset(2 * exposure)", {}, "only 'offset(col)' and 'offset(log(col))'"),
    ("cnt ~ x + offset()", {}, "only 'offset(col)' and 'offset(log(col))'"),
    ("cnt ~ x + offset(exposure) + offset(w)", {}, "more than one offset term ('offset(exposure)' and 'offset(w)'); add the offsets into one column"),
    ("cnt ~ x + offset(log(exposure)) + offset(log(w))", {}, "more than one offset term"),
    ("cnt ~ x + offset(cnt)", {}, "the response 'cnt' cannot be the offset"),
    ("cnt ~ x + offset(log(cnt))", {}, "the response 'cnt' cannot be the offset"),
    ("k | trials(n) ~ x + offset(n)", {"family": "binomial"}, "the trials column 'n' cannot be the offset"),
    ("cnt ~ x + offset(g) + (1 | g)", {}, "the grouping factor 'g' cannot be the offset; an offset is a known numeric column"),
    ("cnt ~ x + offset(log(g)) + (1 | g)", {}, "the grouping factor 'g' cannot be the offset"),
    ("cnt ~ x + z + offset(z)", {"categorical": {"z": "p"}}, "'z' is declared categorical; an offset is a known numeric column"),
    ("cnt ~ x + (1 + offset(exposure) || g)", {}, "offset() cannot be placed inside a group term; write it as a population-level term, 'cnt ~ ... + offset(exposure) + (1 || g)'"),
    ("cnt ~ x + (1 + offset(log(exposure)) | g)", {}, "offset() cannot be placed inside a group term"),
    ("cnt ~ x + (1 | offset(g))", {}, "transformed grouping factors are not supported"),
    ("cnt ~ offset(exposure)", {}, "no population-level predictor"),
    ("cnt ~ offset(exposure) + (1 | g)", {}, "no population-level predictor"),
    ("cnt ~ x + offset(log(exposure)) + log(exposure, 2)", {}, "function calls and transformations inside the formula are not supported"),   # A05 accepts log(exposure) next to the offset
    ("cnt ~ x + offset(log(exposure)):z", {}, "offset() cannot enter an interaction; an offset is a known column added to the linear predictor with no coefficient"),
]


@pytest.mark.parametrize("formula,kwargs,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, **kwargs)


def test_data_refusals():
    df = frame()
    spec = ab.parse_formula("cnt ~ x + offset(log(exposure)) + (1 | g)", family="poisson")
    with pytest.raises(ab.FormulaError, match=re.escape("columns not in data: ['exposure']")):
        ab.parse_formula("cnt ~ x + offset(log(exposure)) + (1 | g)", df.drop(columns="exposure"), "poisson")
    with pytest.raises(ab.FormulaError, match=re.escape("column 'exposure' must be positive for the offset term 'offset(log(exposure))'; row 3 has 0.0")):
        ab.parse_formula("cnt ~ x + offset(log(exposure)) + (1 | g)", df.assign(exposure=df.exposure.where(df.index != 3, 0.0)), "poisson")
    with pytest.raises(ab.FormulaError, match="must be positive for the offset term"):
        ab.poisson("cnt ~ x + offset(log(exposure)) + (1 | g)", df.assign(exposure=-df.exposure), POISSON_PRIORS)
    with pytest.raises(ab.FormulaError, match=re.escape("the offset term 'offset(log(exposure))' needs a numeric column")):
        ab.poisson("cnt ~ x + offset(log(exposure)) + (1 | g)", df.assign(exposure=df.exposure.astype(str)), POISSON_PRIORS)
    with pytest.raises(ab.FormulaError, match="has missing or non-finite values"):
        ab.poisson("cnt ~ x + offset(log(exposure)) + (1 | g)", df.assign(exposure=df.exposure.where(df.index > 0)), POISSON_PRIORS)
    with pytest.raises(ab.FormulaError, match="is complex-valued"):
        ab.poisson("cnt ~ x + offset(exposure) + (1 | g)", df.assign(exposure=df.exposure.astype(complex)), POISSON_PRIORS)
    # a plain offset(col) accepts zero and negative values; only log() needs positive ones
    m = ab.gaussian("y ~ x + offset(w) + (1 | g)", df, GAUSS_PRIORS)
    assert np.array_equal(m.design.offset, df.w.to_numpy(dtype=float)) and (df.w <= 0).any()
    assert spec.offset == Offset("exposure", log=True)


# -- 2. the term model ----------------------------------------------------------------------------------------
def test_offset_is_not_a_population_level_term():
    s = ab.parse_formula("cnt ~ x + offset(log(exposure)) + (1 | g)", family="poisson")
    assert s.offset == Offset("exposure", log=True) and s.offset.term == "offset(log(exposure))" and s.offset.expression == "log(exposure)"
    assert s.pop_terms == (PopTerm("x", "numeric", ("x",), ()),) and s.terms == ("x",) and s.columns == ("x",)
    assert s.predictors == ("x",) and s.column_terms == (("x", "x", None),)
    assert s.rhs == "x + offset(log(exposure)) + (1 | g)" and s.canonical_formula == "cnt ~ x + offset(log(exposure)) + (1 | g)"
    assert s.bambi_formula == "cnt ~ x + offset(log(exposure)) + (1 | g)"             # Bambi evaluates log() itself
    # order and spacing as written do not matter; the canonical form puts the offset after the population terms
    assert ab.parse_formula("cnt ~ offset( log( exposure ) ) + x + (1|g)", family="poisson").rhs == s.rhs
    assert ab.parse_formula("cnt ~ x + offset (exposure)", family="poisson").offset == Offset("exposure")
    # with and without a population intercept, with interactions, with a binomial response
    assert ab.parse_formula("cnt ~ 1 + offset(exposure)", family="poisson").rhs == "offset(exposure)"
    assert ab.parse_formula("cnt ~ 0 + x + offset(exposure)", family="poisson").rhs == "0 + x + offset(exposure)"
    assert ab.parse_formula("cnt ~ x * z + offset(exposure)", family="poisson", categorical={"z": "p"}).rhs == "x + z + x:z + offset(exposure)"
    assert ab.parse_formula("k | trials(n) ~ x + offset(log(exposure))", family="binomial").bambi_formula == "p(k, n) ~ x + offset(log(exposure))"
    # the offset column may also be a predictor (brms allows it); the offset is still not a term
    both = ab.parse_formula("cnt ~ exposure + offset(log(exposure))", family="poisson")
    assert both.predictors == ("exposure",) and both.offset == Offset("exposure", log=True)
    assert ab.parse_formula("cnt ~ x", family="poisson").offset is None
    # the specification round-trips with the offset
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):
        assert replica == s and hash(replica) == hash(s) and replica.offset == s.offset
    assert hash(s) != hash(ab.parse_formula("cnt ~ x + offset(exposure) + (1 | g)", family="poisson"))


# -- 3. the design and the graph ---------------------------------------------------------------------------------
def test_design_records_the_offset_outside_the_coefficient_design(poisson, gauss):
    df = poisson.data
    assert np.array_equal(poisson.design.offset, np.log(df.exposure.to_numpy(dtype=float)))
    assert np.array_equal(gauss.design.offset, gauss.data.exposure.to_numpy(dtype=float))
    for m in (poisson, gauss):
        d = m.design
        assert d.columns == ("x",) and d.X_raw.shape == (24, 1) and np.array_equal(d.X_raw[:, 0], m.data.x.to_numpy(dtype=float))
        assert np.allclose(d.X_model[:, 0], d.X_raw[:, 0] - d.X_raw[:, 0].mean()) and d.centered
        assert m.coefficients == (("x", "x", None),) and m.names["b:x"] == "x" and not any(k.startswith("b:offset") for k in m.names)
        assert m.priors.coefficient_priors(m.spec) == {"x": m.priors.b}
        assert "exposure" in m.data.columns
    assert {v.name for v in poisson.pymc_model.free_RVs} == {"Intercept", "x", "1|g_sigma", "1|g_offset"}
    assert {v.name for v in gauss.pymc_model.free_RVs} == {"Intercept", "x", "sigma", "1|g_sigma", "1|g_offset"}
    # Bambi's prior dict and PyMC's value variables know nothing of the offset
    assert set(poisson.priors.to_bambi(poisson.spec, "poisson")) == {"Intercept", "x", "1|g"}
    assert {v.name for v in poisson.pymc_model.value_vars} == {"Intercept", "x", "1|g_sigma_log__", "1|g_offset"}
    # a model without the offset on the same data is a different target; so is the untransformed offset
    plain = ab.poisson("cnt ~ x + (1 | g)", frame(), POISSON_PRIORS)
    raw = ab.poisson("cnt ~ x + offset(exposure) + (1 | g)", frame(), POISSON_PRIORS)
    assert len({poisson.target_identity["sha256"], plain.target_identity["sha256"], raw.target_identity["sha256"]}) == 3
    assert plain.design.offset is None and plain.spec.offset is None


def natural(k=0):
    return dict(intercept=0.3 + 0.1 * k, b=np.array([0.5 - 0.05 * k]), sd=0.8 - 0.1 * k,
                z=np.array([-0.4, 0.2, 0.9, -0.1]) + 0.1 * k)


def test_graph_linear_predictor_equals_Xb_plus_offset_plus_Zu(poisson, gauss):
    for m, link in ((gauss, lambda eta: eta), (poisson, np.exp)):
        pmm = m.pymc_model
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
        d = m.design
        for k in range(3):
            nat = natural(k)
            pt = m.value_point(**nat, **({"sigma": 0.9} if m.family == "gaussian" else {}))
            eta = nat["intercept"] + d.X_model @ nat["b"] + d.offset + d.group_designs["g"].Z @ (nat["sd"] * nat["z"])
            assert np.allclose(np.asarray(fn(pt)[0]).ravel(), link(eta), rtol=1e-12, atol=1e-12)
    # the Poisson density at a point: offset inside the log mean, no prior term for it
    from scipy.stats import halfnorm, norm, poisson as pois
    m, d = poisson, poisson.design
    nat = natural()
    eta = nat["intercept"] + d.X_model @ nat["b"] + d.offset + d.group_designs["g"].Z @ (nat["sd"] * nat["z"])
    ref = (norm.logpdf(nat["intercept"], 0, 2) + norm.logpdf(nat["b"], 0, 1).sum() + halfnorm.logpdf(nat["sd"], scale=1)
           + norm.logpdf(nat["z"]).sum() + pois.logpmf(d.y, np.exp(eta)).sum())
    assert abs(float(m.pymc_model.compile_logp(jacobian=False)(m.value_point(**nat))) - ref) < 1e-8


# -- 4. predictions on fitted and new rows ----------------------------------------------------------------------
def synthetic(model, draws=4, seed=8):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    var = {"Intercept": (("chain", "draw"), rng.normal(0.2, 0.1, shape)), "x": (("chain", "draw"), rng.normal(0.4, 0.1, shape)),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (len(LEVELS),)))}
    if model.family == "gaussian":
        var["sigma"] = (("chain", "draw"), rng.uniform(0.3, 0.5, shape))
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(LEVELS)}))


def test_predictions_add_the_offset_on_fitted_rows(poisson, gauss):
    for m in (poisson, gauss):
        idata = synthetic(m)
        P = idata.posterior
        a, b, U = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy()
        x, off, gi = m.data.x.to_numpy(dtype=float), m.design.offset, m.design.group_index
        eta = a[..., None] + b[..., None] * x + off + U[..., gi]
        fixed = a[..., None] + b[..., None] * x + off                                          # the offset stays without group effects
        assert np.allclose(m.predict(idata, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
        assert np.allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"].to_numpy(), fixed, atol=1e-12)
        mean = m.predict(idata, kind="mean")["mu"].to_numpy()
        assert np.allclose(mean, np.exp(eta) if m.family == "poisson" else eta, rtol=1e-10, atol=1e-10)   # Gaussian: Bambi's path
        assert np.allclose(m.predict(idata, kind="mean", include_group_effects=False)["mu"].to_numpy(),
                           np.exp(fixed) if m.family == "poisson" else fixed, rtol=1e-10, atol=1e-10)
        resp = m.predict(idata, kind="response", seed=3)
        assert resp["y"].shape == eta.shape and np.allclose(resp["mu"].to_numpy(), mean, rtol=1e-10, atol=1e-10)


def test_predictions_on_new_rows_use_their_own_offset_column(poisson, gauss):
    for m in (poisson, gauss):
        idata = synthetic(m)
        P = idata.posterior
        a, b, U, sd = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy(), P["1|g_sigma"].to_numpy()
        new = pd.DataFrame({"x": [0.0, 1.0, -2.0, 0.5], "exposure": [1.0, 2.5, 0.5, 4.0], "g": [LEVELS[2], LEVELS[0], LEVELS[3], LEVELS[2]]})
        off = np.log(new.exposure.to_numpy()) if m.spec.offset.log else new.exposure.to_numpy()
        eta = a[..., None] + b[..., None] * new.x.to_numpy() + off + U[..., [LEVELS.index(l) for l in new.g]]
        assert np.allclose(m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
        mean = m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy()                      # Gaussian: Bambi's path on new rows
        assert np.allclose(mean, np.exp(eta) if m.family == "poisson" else eta, rtol=1e-10, atol=1e-10)
        fixed = m.predict(idata, newdata=new.drop(columns="g"), kind="linear", include_group_effects=False)["eta"].to_numpy()
        assert np.allclose(fixed, a[..., None] + b[..., None] * new.x.to_numpy() + off, atol=1e-12)
        # an unseen level: its effect comes from the policy, the offset from the row
        unseen = new.assign(g=["new", LEVELS[0], "new", LEVELS[2]])
        out = m.predict(idata, newdata=unseen, kind="linear", new_groups="population", seed=11)["eta"].to_numpy()
        rng = np.random.default_rng(11)
        u_new = rng.normal(0.0, 1.0, size=sd.shape) * sd
        expect = a[..., None] + b[..., None] * new.x.to_numpy() + off
        expect[..., [1, 3]] += U[..., [LEVELS.index(LEVELS[0]), LEVELS.index(LEVELS[2])]]
        expect[..., [0, 2]] += u_new[..., None]
        assert np.allclose(out, expect, atol=1e-12)
        # the offset column is required on new rows, and must be valid there
        with pytest.raises(ab.FormulaError, match=re.escape(f"newdata lacks columns ['exposure'] (the offset term {m.spec.offset.term!r} needs 'exposure' on every row to predict)")):
            m.predict(idata, newdata=new.drop(columns="exposure"), kind="linear")
        with pytest.raises(ab.FormulaError, match=re.escape("newdata lacks columns ['exposure']")):
            m.predict(idata, newdata=new.drop(columns=["exposure", "g"]), kind="linear", include_group_effects=False)
        if m.spec.offset.log:
            with pytest.raises(ab.FormulaError, match=re.escape("newdata column 'exposure' must be positive for the offset term 'offset(log(exposure))'; row 1 has 0.0")):
                m.predict(idata, newdata=new.assign(exposure=[1.0, 0.0, 0.5, 4.0]), kind="linear")
        with pytest.raises(ab.FormulaError, match="needs a numeric column"):
            m.predict(idata, newdata=new.assign(exposure=["1", "2", "3", "4"]), kind="linear")


# -- 5. records and describe() ----------------------------------------------------------------------------------
def test_fit_record_and_describe(poisson, gauss):
    idata = synthetic(poisson, draws=50)
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 50, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(poisson.target_identity))
    rec = ab.fit_record(poisson, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert rec["model"]["offset"] == "offset(log(exposure))" and rec["model"]["terms"] == ["x"] and rec["model"]["design_columns"] == ["x"]
    assert set(rec["model"]["priors"]) == {"intercept", "b:x", "sd"}
    text = poisson.describe()
    for s in ["AutoBayes Poisson model with offset",
              "offset    : log(exposure), added to the linear predictor as written (no coefficient, no prior, never centred; brms offset())",
              "eta_i = a_c + (x_x - mean(x_x)) * b_x + offset_i + u[g_i]",
              "offset_i = log(exposure_i): the offset term, a known quantity added to every row's eta_i; it has no coefficient and no prior and is not centred.",
              "The exposure enters through the offset term (mu_i = exp(offset_i) * exp(eta_i - offset_i)); no overdispersion or zero-inflation is modeled.",
              "b_x ~ Normal(mu=0, sigma=1)"]:
        assert s in text, s
    assert "b_offset" not in text and "No exposure/offset" not in text
    text2 = gauss.describe()
    assert "AutoBayes M0 model with offset" in text2 and "mu_i = a_c + (x_x - mean(x_x)) * b_x + offset_i + u[g_i]" in text2
    assert "offset_i = exposure_i: the offset term" in text2
    plain = ab.poisson("cnt ~ x + (1 | g)", frame(), POISSON_PRIORS).describe()
    assert "offset_i" not in plain and "with offset" not in plain and "offset    :" not in plain
    assert "No exposure/offset, overdispersion or zero-inflation is modeled." in plain
