"""M3c-1: population-only models, no group-level term (docs/history/m3c1-decision.md). Deterministic checks only: no
sampling. Predictions and records use synthetic fixed draws; densities and gradients are compared with the
independent references of tests/reference_logp.py and tests/reference_binomial.py extended to ``group=None``."""
import copy
import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from scipy.special import expit

import autobayes as ab
from tests.conftest import PRIORS as GROUPED_PRIORS, simulate as simulate_grouped
from tests.reference_binomial import reference_logp_binomial
from tests.reference_logp import build_design, conditional_gaussian_posterior, design_columns, reference_logp, reference_logp_bernoulli

LEVELS, PROPS = ("a", "b", "c"), (0.5, 0.3, 0.2)
XMEAN = {"a": 0.0, "b": 1.0, "c": -0.5}                      # different within-level x means: product-column means are not products of means
BETA = {"a": 0.0, "b": -0.8, "c": 0.6}
DELTA = {"a": 0.0, "b": 0.4, "c": -0.7}
TRUTH = dict(a=1.0, bx=0.5, bw=-0.3, sigma=0.4)
G = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
G_LEVEL = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": {"a": ab.Normal(0, 1), "c": ab.Normal(0.5, 2)}, "x:z": ab.Normal(0, 1)},
                    sigma=ab.HalfNormal(1))
L = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1))
L_LEVEL = ab.Priors(intercept=ab.Normal(0, 1.5), b={"x": ab.Normal(0, 1), "z": ab.StudentT(5, 0, 1),
                                                      "x:z": {"a": ab.StudentT(4, 0, 1), "c": ab.StudentT(6, 0.2, 2)}})
LHS = {"gaussian": "y", "bernoulli": "y01", "binomial": "k | trials(n)"}
RHS = {"intercept": "1", "numeric": "x + w", "categorical": "x + z", "interaction": "x + z + x:z"}
PANEL_STUB = {"values": {"divergences": 0.0, "divergence_fraction": 0.0, "max_rhat": 1.0, "min_ess_bulk": 1000.0, "min_ess_tail": 1000.0,
                         "min_bfmi": 1.0, "treedepth_saturation": 0.0, "n_draws": 80.0},
              "status": {k: "pass" for k in ("divergence_fraction", "max_rhat", "min_ess_bulk", "min_ess_tail", "min_bfmi", "treedepth_saturation")},
              "overall": "pass"}                                  # a stub handed to fit_record: no sampler statistics exist here


def simulate_pop(n=90, seed=11, trials=20):
    """No grouping column anywhere: numeric x and w, categorical z with unequal proportions and different x means."""
    rng = np.random.default_rng(seed)
    z = rng.choice(LEVELS, size=n, p=PROPS)
    x = rng.normal(size=n) + np.array([XMEAN[v] for v in z]); w = rng.normal(size=n)
    eta = TRUTH["a"] + TRUTH["bx"] * x + TRUTH["bw"] * w + np.array([BETA[v] for v in z]) + np.array([DELTA[v] for v in z]) * x
    df = pd.DataFrame({"y": eta + rng.normal(0, TRUTH["sigma"], n), "x": x, "w": w, "z": z,
                       "y01": (rng.uniform(size=n) < expit(eta - 1.0)).astype(int), "k": rng.binomial(trials, expit(eta - 1.0)), "n": trials})
    assert set(df.z) == set(LEVELS) and df.y01.nunique() == 2 and "g" not in df.columns
    return df


def build(family, df, priors=None, kind="numeric", center=True, formula=None, categorical=None):
    if priors is None:
        priors = G if family == "gaussian" else L
    if categorical is None and kind in ("categorical", "interaction"):
        categorical = {"z": "b"}
    return getattr(ab, family)(formula or f"{LHS[family]} ~ {RHS[kind]}", df, priors, center_predictors=center, categorical=categorical)


def ref_priors(priors, spec):
    coef = priors.coefficient_priors(spec)
    out = {"intercept": (priors.intercept.name, dict(priors.intercept.params)), "b": {c: (p.name, dict(p.params)) for c, p in coef.items()}}
    if priors.sigma is not None:
        out["sigma"] = (priors.sigma.name, dict(priors.sigma.params))
    return out


def ref_point(model, point):
    b = np.concatenate([np.atleast_1d(point[t]) for t in model.spec.terms]) if model.spec.terms else np.zeros(0)
    out = {"a_c": float(point["Intercept"]), "b": b}
    if model.family == "gaussian":
        out["log_sigma"] = float(point["sigma_log__"])
    return out


def reference(model, df, pd_, point, jacobian):
    mains = [t for t in model.spec.terms if t not in model.spec.interaction_names]
    kw = dict(response=model.spec.response, predictors=mains, group=None, priors=pd_, center=model.center_predictors, jacobian=jacobian,
              categorical={c.name: c.reference for c in model.spec.categoricals}, interactions=[tuple(p) for p in model.spec.interactions])
    pt = ref_point(model, point)
    if model.family == "gaussian":
        return reference_logp(pt, df, **kw)
    if model.family == "bernoulli":
        return reference_logp_bernoulli(pt, df, **kw)
    return reference_logp_binomial(pt, df, trials="n", **kw)


def random_nat(model, rng):
    nat = dict(intercept=rng.normal(0.5, 0.7), b=rng.normal(0, 0.6, len(model.spec.columns)))
    if model.family == "gaussian":
        nat["sigma"] = np.exp(rng.normal(-0.8, 0.3))
    return nat


def synthetic_posterior(model, draws=200, seed=3):
    """Fixed random draws for every population parameter of the model, two chains, no group variables."""
    rng = np.random.default_rng(seed); shape = (2, draws)
    data = {"Intercept": (("chain", "draw"), rng.normal(1.0, 0.2, shape))}
    coords = {"chain": [0, 1], "draw": np.arange(draws)}
    for term in model.spec.terms:
        levels = model.spec.levels_of(term)
        if levels:
            data[term] = (("chain", "draw", f"{term}_dim"), rng.normal(0, 0.5, shape + (len(levels),))); coords[f"{term}_dim"] = list(levels)
        else:
            data[term] = (("chain", "draw"), rng.normal(0.5, 0.2, shape))
    if model.family == "gaussian":
        data["sigma"] = (("chain", "draw"), np.full(shape, 0.3))
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def synthetic_fit(model, draws=40):
    return ab.FitResult(idata=synthetic_posterior(model, draws=draws), backend="pymc", device_requested="cpu", device_actual="cpu", platform="cpu",
                        dtype="float64", chain_method=None, chains=2, draws=draws, tune=0, target_accept=0.9, wall_seconds=0.0, notes=[], seed=0, _target_identity_json=json.dumps(model.target_identity))


def hand_linpred(model, post, df):
    """Raw intercept + coefficients on the raw design, per draw; no group effect exists."""
    from autobayes.predict import raw_design
    a = post["Intercept"].to_numpy()[..., None]
    parts = [post[t].to_numpy() if model.spec.levels_of(t) else post[t].to_numpy()[..., None] for t in model.spec.terms]
    if not parts:
        return np.repeat(a, len(df), axis=-1)
    return a + np.einsum("cdp,np->cdn", np.concatenate(parts, axis=-1), raw_design(model, df))


# -- P1 grammar ------------------------------------------------------------------------------------------
def test_grammar_accepts_population_only_forms_and_keeps_refusals():
    for formula, terms, rhs in [("y ~ 1", (), "1"), ("y ~ x", ("x",), "x"), ("y ~ 1 + x", ("x",), "x"), ("y ~ x + w", ("x", "w"), "x + w"), ("y~x", ("x",), "x")]:
        s = ab.parse_formula(formula)
        assert s.group is None and not s.has_group and s.levels == () and s.terms == terms and s.predictors == terms and s.rhs == rhs
        assert s.canonical_formula == f"y ~ {rhs}" and s.bambi_formula == f"y ~ {rhs}"
    s = ab.parse_formula("y ~ x + z + x:z", categorical={"z": "b"})
    assert s.terms == ("x", "z", "x:z") and s.rhs == "x + z + x:z" and s.interactions == (("x", "z"),) and s.group is None
    assert ab.parse_formula("y ~ x*z", categorical={"z": "b"}).rhs == "x + z + x:z"
    s = ab.parse_formula("k | trials(n) ~ x", family="binomial")
    assert s.group is None and s.trials == "n" and s.bambi_formula == "p(k, n) ~ x"
    g = ab.parse_formula("y ~ x + (1 | g)")
    assert g.group == "g" and g.has_group and g.group_term == "1|g" and g.rhs == "x + (1 | g)"       # grouped forms unchanged
    assert ab.parse_formula("y ~ 1 + (1 | g)").rhs == "(1 | g)"
    with pytest.raises(ValueError, match="no group-level term"):
        ab.parse_formula("y ~ x").group_term
    for formula, needle in [("y ~ x + (1 | g) + (1 | g)", "duplicate grouping"),
                            ("y ~ x + (x | g)", "varying slopes"), ("y ~ (1 | g)", "no population-level predictor"),   # A02 accepts '(1 + x | g)'
                            ("y ~ x + (x || g)", "without the varying intercept"),
                            ("y ~ sin(x)", "function calls"), ("y ~ x + x", "duplicated"), ("y ~ y", "response cannot also be"),   # A05 accepts log(x)
                            ("y ~ x:z", "main effect 'x' is missing"), ("y ~ x + (1 | g/h/k)", "only two factors"), ("y x", "expected 'response ~ terms'")]:   # A03 accepts '(1 | g/h)'
        with pytest.raises(ab.FormulaError, match=needle):
            ab.parse_formula(formula)


# -- P2 data ----------------------------------------------------------------------------------------------
def test_data_validation_without_a_grouping_column():
    df = simulate_pop()
    m = build("gaussian", df)
    assert m.spec.group is None and m.design.J == 0 and list(m.data.columns) == ["y", "x", "w"]
    with pytest.raises(ab.FormulaError, match="not in data"):
        ab.parse_formula("y ~ q", df)
    with pytest.raises(ab.FormulaError, match="missing values"):
        ab.gaussian("y ~ x", df.assign(x=df.x.where(df.index > 0)), G)
    with pytest.raises(ab.FormulaError, match="constant"):
        ab.gaussian("y ~ x", df.assign(x=1.0), G)
    with pytest.raises(ab.FormulaError, match="declared categorical"):
        ab.gaussian("y ~ z", df, G)
    with pytest.raises(ab.FormulaError, match="outside \\{0, 1\\}"):
        ab.bernoulli("y01 ~ x", df.assign(y01=df.y01 * 2), L)
    with pytest.raises(ab.FormulaError, match="exceed trials"):
        ab.binomial("k | trials(n) ~ x", df.assign(k=df.n + 1), L)
    with pytest.raises(ab.FormulaError, match="needs 'successes"):
        ab.binomial("k ~ x", df, L)
    assert build("gaussian", df.assign(g="extra")).spec.group is None                   # a column named like a group is just a column
    s = ab.parse_formula("y ~ x + z", df, categorical={"z": "b"})
    assert s.categorical("z").levels == LEVELS and s.levels == ()


# -- P3 priors --------------------------------------------------------------------------------------------
def test_prior_refusals_and_coverage():
    df = simulate_pop()
    with pytest.raises(ab.PriorError, match="no group-level term"):
        ab.gaussian("y ~ x", df, GROUPED_PRIORS)                                           # sd given, no group term
    with pytest.raises(ab.PriorError, match="needs a prior on the SD of its varying intercept"):
        ab.gaussian("y ~ x + (1 | g)", simulate_grouped(), ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="residual SD"):
        ab.gaussian("y ~ x", df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2)))
    with pytest.raises(ab.PriorError, match="no residual SD"):
        ab.bernoulli("y01 ~ x", df, G)
    with pytest.raises(ab.PriorError, match="missing \\['w'\\]"):
        ab.gaussian("y ~ x + w", df, ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2)}, sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="non-reference levels"):
        ab.gaussian("y ~ x + z", df, ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": {"a": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1)),
                    categorical={"z": "b"})
    m = build("gaussian", df, G_LEVEL, kind="interaction")
    assert m.priors.sd is None and set(m.priors.to_bambi(m.spec, "gaussian")) == {"Intercept", "sigma", "x", "z", "x:z"}
    gm = ab.gaussian("y ~ x + (1 | g)", simulate_grouped(), GROUPED_PRIORS)
    assert "1|g" in gm.priors.to_bambi(gm.spec, "gaussian")


# -- P4 design against an independent construction --------------------------------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("kind", ["numeric", "categorical", "interaction"])
@pytest.mark.parametrize("center", [True, False])
def test_design_matches_independent_construction(family, kind, center):
    df = simulate_pop()
    m = build(family, df, kind=kind, center=center)
    d = m.design
    mains = [t for t in m.spec.terms if t not in m.spec.interaction_names]
    inter = [tuple(p) for p in m.spec.interactions]
    cat = {"z": "b"} if kind != "numeric" else {}
    _, X_raw, _, gidx, levels = build_design(m.data, m.spec.response, mains, None, False, cat, inter)
    _, Xc, xbar, _, _ = build_design(m.data, m.spec.response, mains, None, center, cat, inter)
    assert gidx is None and levels == []
    assert list(m.spec.columns) == design_columns(m.data, mains, cat, inter)
    assert np.array_equal(d.X_raw, X_raw) and np.allclose(d.X_model, Xc) and np.allclose(d.x_means, xbar)
    assert d.Z.shape == (d.n, 0) and d.J == 0 and d.levels == () and d.group_index.shape == (0,)
    assert np.allclose(d.X_model, d.X_raw - d.x_means)
    if kind != "numeric":
        assert list(m.pymc_model.coords["z_dim"]) == ["a", "c"]
        if center:
            assert np.allclose(d.x_means[list(m.spec.columns).index("z[a]")], (df.z == "a").mean())
    if kind == "interaction" and center:
        col = list(m.spec.columns).index("x:z[c]")
        assert np.allclose(d.x_means[col], (df.x * (df.z == "c")).mean())
    if family == "binomial":
        assert np.array_equal(d.trials, df.n.to_numpy())


# -- P5 density, Jacobian and gradients on both paths ------------------------------------------------------
@pytest.mark.parametrize("family,priors,kind", [("gaussian", G, "intercept"), ("gaussian", G, "numeric"), ("gaussian", G, "categorical"),
                                                ("gaussian", G_LEVEL, "interaction"), ("bernoulli", L, "intercept"), ("bernoulli", L, "numeric"),
                                                ("bernoulli", L_LEVEL, "interaction"), ("binomial", L, "categorical"), ("binomial", L_LEVEL, "interaction")])
@pytest.mark.parametrize("center", [True, False])
def test_density_jacobian_and_gradients_both_paths(family, priors, kind, center):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    df = simulate_pop()
    m = build(family, df, priors, kind=kind, center=center)
    pd_ = ref_priors(priors, m.spec)
    assert set(pd_["b"]) == set(m.spec.columns)
    logp, logp_nojac, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_logp(jacobian=False), m.pymc_model.compile_dlogp()
    names = [v.name for v in m.pymc_model.value_vars]
    assert not any(n.startswith("1|") for n in names)
    jf = get_jaxified_logp(m.pymc_model)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    rng = np.random.default_rng(5)
    for _ in range(3):
        nat = random_nat(m, rng)
        point = m.value_point(**nat)
        ref, ref_nojac = reference(m, m.data, pd_, point, True), reference(m, m.data, pd_, point, False)
        assert np.isfinite(ref)
        assert abs(float(logp(point)) - ref) < 1e-8, (family, kind, float(logp(point)), ref)
        assert abs(float(logp_nojac(point)) - ref_nojac) < 1e-8
        assert abs(ref - ref_nojac - (np.log(nat["sigma"]) if family == "gaussian" else 0.0)) < 1e-12
        assert abs(float(jf([point[n] for n in names])) - ref) < 1e-8
        grad_pt = np.asarray(dlogp(point), dtype=float)
        grad_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)).ravel() for g in jgrad(*[point[n] for n in names])])
        flat = np.concatenate([np.atleast_1d(np.asarray(point[n], dtype=float)).ravel() for n in names])
        def f(vec):
            q, i = {}, 0
            for n_ in names:
                size = int(np.prod(np.shape(point[n_])))
                q[n_] = vec[i:i + size].reshape(np.shape(point[n_])) if np.ndim(point[n_]) else np.asarray(vec[i])
                i += size
            return reference(m, m.data, pd_, q, True)
        h = 1e-5
        fd = np.array([(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in np.eye(len(flat))])
        assert np.allclose(grad_pt, fd, rtol=1e-6, atol=1e-5), (family, kind, grad_pt, fd)
        assert np.allclose(grad_jx, grad_pt, rtol=1e-9, atol=1e-9)


# -- P6 the closed-form Gaussian posterior with sigma fixed ------------------------------------------------
def test_fixed_sigma_posterior_is_closed_form_gaussian():
    df = simulate_pop()
    m = build("gaussian", df, kind="numeric")                                                # y ~ x + w
    pmm = m.pymc_model; sigma = 0.3
    fixed = pm.do(pmm, {pmm["sigma"]: sigma})
    names = [v.name for v in fixed.value_vars]
    assert set(names) == {"Intercept", "x", "w"}
    mean, cov, P = conditional_gaussian_posterior(m.data, response="y", predictors=["x", "w"], group=None, center=True, sigma=sigma,
                                                  prior_intercept={"mu": 0.0, "sigma": 5.0},
                                                  prior_b={"x": {"mu": 0.0, "sigma": 2.0}, "w": {"mu": 0.0, "sigma": 2.0}})
    order = [{"Intercept": 0, "x": 1, "w": 2}[n] for n in names]
    d2 = fixed.compile_d2logp()
    for pt in ({"Intercept": np.asarray(0.7), "x": np.asarray(0.2), "w": np.asarray(-0.1)}, {"Intercept": np.asarray(-3.0), "x": np.asarray(5.0), "w": np.asarray(2.0)}):
        assert np.allclose(np.asarray(d2(pt)), P[np.ix_(order, order)], atol=1e-8)          # constant precision: a Gaussian
    from scipy.optimize import minimize
    logp = fixed.compile_logp()
    res = minimize(lambda v: -float(logp(dict(zip(names, v)))), np.zeros(3), method="BFGS")
    hand = np.empty(3); hand[order] = res.x
    assert np.allclose(hand, mean, atol=1e-4)


# -- P7 the graph, names and value_point ----------------------------------------------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_graph_names_and_value_point(family):
    df = simulate_pop()
    m = build(family, df, kind="categorical")                                                # x + z
    pmm = m.pymc_model
    assert {v.name for v in pmm.free_RVs} == {"Intercept", "x", "z"} | ({"sigma"} if family == "gaussian" else set())
    assert {v.name for v in pmm.deterministics} == {m.names["linpred"]}
    n = m.names
    assert not ({"sd", "offset", "group_effect"} & set(n)) and n["b:z"] == "z" and n["intercept"] == "Intercept"
    nat = random_nat(m, np.random.default_rng(1))
    point = m.value_point(**nat)
    assert set(point) == {v.name for v in pmm.value_vars}
    lin_v, = pmm.replace_rvs_by_values([pmm[n["linpred"]]])
    fn = pmm.compile_fn([lin_v], inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    graph = np.asarray(fn(point)[0]).ravel()
    assert np.allclose(graph, nat["intercept"] + m.design.X_model @ nat["b"], atol=1e-12)
    with pytest.raises(ValueError, match="no group-level term"):
        m.value_point(**dict(nat, sd=0.5))
    with pytest.raises(ValueError, match="no group-level term"):
        m.value_point(**dict(nat, z=np.zeros(3)))
    gm = ab.gaussian("y ~ x + (1 | g)", simulate_grouped(), GROUPED_PRIORS)
    with pytest.raises(ValueError, match="sd and z are required"):
        gm.value_point(intercept=0.0, b=[0.0], sigma=1.0)
    assert {"sd", "offset", "group_effect"} <= set(gm.names) and gm.design.J == 6


# -- P8 describe() and the fit record (synthetic posterior, stub panel; nothing sampled) ---------------------
def test_describe_and_fit_record_without_sampling():
    df = simulate_pop()
    m = build("gaussian", df, kind="numeric")
    text = m.describe()
    assert "no group-level term (population-level model)" in text and "(population-level only)" in text
    assert "levels    :" not in text and "u[" not in text and "sd_" not in text and "Group effects (non-centred)" not in text and "1|" not in text
    assert "mu_i = a_c + (x_x - mean(x_x)) * b_x + (x_w - mean(x_w)) * b_w" in text and "sigma_log__" in text
    assert "a_c is the expected response at the predictor means; the" in text
    rec = ab.fit_record(m, synthetic_fit(m), panel=PANEL_STUB)
    json.dumps(rec)
    assert rec["model"]["group"] is None and rec["model"]["J"] == 0 and rec["model"]["noncentered"] is None and rec["model"]["formula"] == "y ~ x + w"
    assert rec["model"]["priors"] == {"intercept": "Normal(mu=0, sigma=5)", "b:x": "Normal(mu=0, sigma=2)", "b:w": "Normal(mu=0, sigma=2)",
                                      "sigma": "HalfNormal(sigma=1)"}
    assert set(rec["posterior"]) == {"Intercept", "x", "w", "sigma"} and rec["model"]["intercept_prior_applies_to"].startswith("centred intercept")
    assert ab.compare_records(rec, copy.deepcopy(rec))["agree"]
    b = build("bernoulli", df, kind="numeric")
    brec = ab.fit_record(b, synthetic_fit(b), panel=PANEL_STUB)
    assert set(brec["posterior"]) == {"Intercept", "x", "w"} and "sd" not in brec["model"]["priors"] and brec["model"]["link"] == "logit"
    assert "P(y_i = 1) = p_i = expit(eta_i)" in b.describe() and "eta_i = a_c + (x_x - mean(x_x)) * b_x" in b.describe()
    c = build("binomial", df, kind="numeric")
    assert "no group-level term (population-level model); 0 zero-trial row(s) dropped" in c.describe()


# -- P9 predictions computed by hand from fixed draws, no grouping column ----------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_predictions_by_hand_from_fixed_draws(family):
    df = simulate_pop()
    m = build(family, df, kind="interaction")
    idata = synthetic_posterior(m); post = idata.posterior
    mean_name = "mu" if family == "gaussian" else "p"
    link = (lambda v: v) if family == "gaussian" else expit
    ins = m.predict(idata, kind="mean")
    assert ins.attrs["new_groups"] == "not_applicable" and ins.attrs["new_levels"] == [] and ins.sizes["__obs__"] == len(m.data)
    assert np.allclose(ins[mean_name].to_numpy(), link(hand_linpred(m, post, m.data)), atol=1e-10)
    new = pd.DataFrame({"x": [0.3, -1.0, 0.0, 2.0, 0.3], "z": ["c", "a", "b", "a", "c"], "n": [10, 0, 5, 7, 10]})     # no grouping column at all
    out = m.predict(idata, newdata=new, kind="mean", seed=1)
    hand = link(hand_linpred(m, post, new))
    assert np.allclose(out[mean_name].to_numpy(), hand, atol=1e-10) and out.attrs["new_groups"] == "not_applicable" and out.attrs["new_levels"] == []
    assert np.allclose(out[mean_name].to_numpy()[..., 0], out[mean_name].to_numpy()[..., 4])                          # identical rows, identical predictions
    if family == "binomial":
        assert np.allclose(out["expected"].to_numpy(), hand * new.n.to_numpy(), atol=1e-10)
        assert np.allclose(m.predict(idata, newdata=new.drop(columns="n"), kind="probability")["p"].to_numpy(), hand, atol=1e-10)
    resp = m.predict(idata, newdata=new, kind="response", seed=4)
    if family == "gaussian":
        resid = (resp["y"] - resp["mu"]).to_numpy()
        assert abs(resid.std() - 0.3) < 0.03 and abs(resid.mean()) < 0.03
    elif family == "bernoulli":
        y = resp["y"].to_numpy()
        assert set(np.unique(y)) <= {0, 1} and abs(y.mean() - hand.mean()) < 0.05
    else:
        y = resp["y"].to_numpy()
        assert y.dtype.kind == "i" and (y <= new.n.to_numpy()).all() and (y[..., 1] == 0).all()
    extra = m.predict(idata, newdata=new.assign(g=["p", "q", "p", "q", "p"]), kind="mean", seed=1)                  # a stray column is not a group
    assert np.allclose(extra[mean_name].to_numpy(), hand, atol=1e-10)
    for ng in ("population", "resample_observed"):
        with pytest.raises(ValueError, match="no group-level term"):
            m.predict(idata, newdata=new, kind="mean", new_groups=ng)
    with pytest.raises(ab.FormulaError, match="not in the fitted rows: \\['d'\\]"):
        m.predict(idata, newdata=new.assign(z=["d", "a", "b", "a", "c"]), kind="mean")


# -- P10, P11 grouped models unchanged; the label contract without a group ----------------------------------
def test_grouped_models_unchanged_and_label_contract_without_group():
    gdf = simulate_grouped()
    gm = ab.gaussian("y ~ x + (1 | g)", gdf, GROUPED_PRIORS)
    assert gm.spec.group == "g" and gm.design.J == 6 and gm.design.Z.shape == (len(gdf), 6) and "Group effects (non-centred)" in gm.describe()
    assert {v.name for v in gm.pymc_model.free_RVs} == {"Intercept", "x", "sigma", "1|g_sigma", "1|g_offset"}
    df = simulate_pop(); codes = {"a": 1, "b": 2, "c": 3}
    m_int = ab.gaussian("y ~ x + z", df.assign(z=df.z.map(codes)), G, categorical={"z": 2})
    m_str = ab.gaussian("y ~ x + z", df.assign(z=df.z.map(codes).astype(str)), G, categorical={"z": "2"})
    assert m_int.spec.columns == m_str.spec.columns == ("x", "z[1]", "z[3]") and np.array_equal(m_int.design.X_raw, m_str.design.X_raw)
    m1 = build("gaussian", df, kind="intercept")
    assert m1.spec.columns == () and m1.design.X_raw.shape == (len(df), 0) and set(m1.value_point(intercept=1.0, b=[], sigma=0.4)) == {"Intercept", "sigma_log__"}
