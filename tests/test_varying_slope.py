"""M3b-1: independent varying intercept and one numeric varying slope, ``(1 + x || g)``.
Checks F1-F6 and F10 of docs/m3b1-decision.md."""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import arviz as az
from scipy import stats
from scipy.special import expit

import autobayes as ab
from tests.reference_binomial import reference_logp_binomial
from tests.reference_logp import reference_logp, reference_logp_bernoulli

TRUTH = dict(a=1.0, b=0.5, sd0=0.5, sd1=0.3, sigma=0.4)
G_PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd={"Intercept": ab.HalfNormal(1), "x": ab.HalfStudentT(4, 0.7)},
                     sigma=ab.HalfNormal(1.5))
L_PRIORS = ab.Priors(intercept=ab.Normal(0.1, 1.5), b=ab.Normal(-0.2, 1), sd={"Intercept": ab.Exponential(2.0), "x": ab.HalfCauchy(0.5)})
LHS = {"gaussian": "y", "bernoulli": "y01", "binomial": "k | trials(n)"}


def simulate_slope(n=96, J=8, seed=5, trials=15):
    rng = np.random.default_rng(seed)
    g = np.array([f"g{i % J:02d}" for i in range(n)]); rng.shuffle(g)
    x = rng.normal(size=n) + 0.7                                        # raw x not centred at zero
    u = rng.normal(0, TRUTH["sd0"], J); v = rng.normal(0, TRUTH["sd1"], J)
    gi = np.array([int(s[1:]) for s in g])
    eta = TRUTH["a"] + TRUTH["b"] * x + u[gi] + x * v[gi]
    df = pd.DataFrame({"y": eta + rng.normal(0, TRUTH["sigma"], n), "x": x, "g": g,
                       "y01": (rng.uniform(size=n) < expit(eta)).astype(int), "k": rng.binomial(trials, expit(eta)), "n": trials})
    assert (df.groupby("g").x.std() > 0).all()
    return df


def build(family, df, priors=None, formula=None, **kw):
    priors = priors or (G_PRIORS if family == "gaussian" else L_PRIORS)
    return getattr(ab, family)(formula or f"{LHS[family]} ~ x + (1 + x || g)", df, priors, **kw)


def ref_priors(priors, spec):
    coef = priors.coefficient_priors(spec); sds = priors.sd_priors(spec)
    out = {"intercept": (priors.intercept.name, dict(priors.intercept.params)),
           "b": {c: (p.name, dict(p.params)) for c, p in coef.items()},
           "sd": (sds["Intercept"].name, dict(sds["Intercept"].params))}
    if spec.varying_slope:
        out["sd_slope"] = (sds[spec.varying_slope].name, dict(sds[spec.varying_slope].params))
    if priors.sigma is not None:
        out["sigma"] = (priors.sigma.name, dict(priors.sigma.params))
    return out


# -- F1: grammar -------------------------------------------------------------------------------------
def test_grammar_canonical_forms_and_refusals():
    df = simulate_slope()
    m = build("gaussian", df)
    assert m.spec.varying_slope == "x" and m.spec.group_spec == "(1 + x || g)" and m.spec.slope_term == "x|g"
    assert m.spec.canonical_formula == "y ~ x + (1 + x || g)" and m.spec.bambi_formula == "y ~ x + (1 + x | g)"
    m2 = build("gaussian", df, formula="y ~ x + (x + 1 || g)")
    assert m2.spec.canonical_formula == m.spec.canonical_formula and np.array_equal(m2.design.Z_slope, m.design.Z_slope)
    m0 = build("gaussian", df, priors=ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)), formula="y ~ x + (1 || g)")
    assert m0.spec.varying_slope is None and m0.spec.canonical_formula == "y ~ x + (1 | g)" and m0.design.Z_slope is None
    m1 = build("gaussian", df, priors=ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"Intercept": ab.HalfNormal(1)}, sigma=ab.HalfNormal(1)), formula="y ~ x + (1 | g)")
    assert m1.spec.varying_slope is None and "sd_slope" not in m1.names
    cases = [("y ~ x + (1 + x | g)", "CORRELATED varying intercept and slope"), ("y ~ x + (x || g)", "write the varying intercept explicitly"),
             ("y ~ x + (x | g)", "write the varying intercept explicitly"), ("y ~ x + (0 + x || g)", "without a varying intercept"),
             ("y ~ x + (1 + x + w || g)", "more than one varying slope"), ("y ~ (1 + x || g)", "must also be a numeric fixed-effect term"),
             ("y ~ w + (1 + x || g)", "must also be a numeric fixed-effect term"), ("y ~ x + (1 + g || g)", "cannot be the grouping factor"),
             ("y ~ x + (1 + x || g) + (1 | h)", "more than one grouping factor"), ("y ~ x + (1 + x:w || g)", "single numeric column name"),
             ("y ~ x + (1 + x || g/h)", "nested or crossed"), ("y ~ x + (1 + x |)", "grouping factor must be a single column name"),
             ("y ~ x + (1 + x || g", "unbalanced")]
    d = df.assign(w=df.x ** 2, h=df.g)
    for formula, msg in cases:
        with pytest.raises(ab.FormulaError, match=msg):
            ab.gaussian(formula, d, G_PRIORS)
    dz = df.assign(z=["a", "b", "c"] * (len(df) // 3))
    with pytest.raises(ab.FormulaError, match="categorical varying slope is not supported"):
        ab.gaussian("y ~ x + z + (1 + z || g)", dz, G_PRIORS, categorical={"z": "a"})
    old = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    for formula in ["y ~ x + (1 | g)", "y ~ 1 + (1 | g)", "y ~ x + z + x:z + (1 | g)"]:
        mm = ab.gaussian(formula, dz, old, categorical={"z": "a"} if "z" in formula else None)   # unchanged earlier models
        assert mm.spec.varying_slope is None and mm.spec.group_spec == "(1 | g)"


# -- F2: the group design from the original rows and labels ---------------------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("center", [True, False])
def test_group_design_from_original_rows(family, center):
    df = simulate_slope().sample(frac=1.0, random_state=2).reset_index(drop=True)
    if family == "binomial":
        df = df.copy(); df.loc[df.index[[3, 11, 40]], ["n", "k"]] = 0
    m = build(family, df, center_predictors=center)
    fitted = df[df.n > 0].reset_index(drop=True) if family == "binomial" else df          # chosen here
    levels = sorted(fitted.g.unique())
    Z = np.array([[float(lab == l) for l in levels] for lab in fitted.g])
    x_raw = fitted.x.to_numpy()
    assert m.design.n == len(fitted) and list(m.design.levels) == levels
    assert np.array_equal(m.design.Z, Z) and np.array_equal(m.design.Z_slope, x_raw[:, None] * Z)      # raw x, never centred
    assert m.design.slope == "x" and np.allclose(m.design.x_means, [x_raw.mean()] if center else [0.0])
    assert [v.name for v in m.pymc_model.value_vars if "|" in v.name] == ["1|g_sigma_log__", "1|g_offset", "x|g_sigma_log__", "x|g_offset"]
    assert m.pymc_model.named_vars_to_dims["x|g_offset"] == ("g__factor_dim",) and list(m.pymc_model.coords["g__factor_dim"]) == levels
    assert m.names["sd_slope"] == "x|g_sigma" and m.names["group_slope"] == "x|g" and m.names["offset_slope"] == "x|g_offset"
    if center:                                                                                      # fixed centring unaffected
        assert np.allclose(m.design.X_model[:, 0], x_raw - x_raw.mean())


# -- F3: reference density, Jacobian and gradients on both paths ------------------------------------------
def _ref_point(model, point):
    out = {"a_c": float(point["Intercept"]), "b": np.concatenate([np.atleast_1d(point[t]) for t in model.spec.terms]),
           "log_sd": float(point["1|g_sigma_log__"]), "z": np.asarray(point["1|g_offset"]),
           "log_sd_slope": float(point["x|g_sigma_log__"]), "z_slope": np.asarray(point["x|g_offset"])}
    if model.family == "gaussian":
        out["log_sigma"] = float(point["sigma_log__"])
    return out


def _reference(model, data, pd_, point, jacobian):
    kw = dict(response=model.spec.response, predictors=list(model.spec.predictors), group="g", priors=pd_, center=model.center_predictors,
              jacobian=jacobian, varying_slope="x")
    if model.family == "gaussian":
        return reference_logp(_ref_point(model, point), data, **kw)
    if model.family == "bernoulli":
        return reference_logp_bernoulli(_ref_point(model, point), data, **kw)
    return reference_logp_binomial(_ref_point(model, point), data, trials="n", **kw)


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("center", [True, False])
def test_density_jacobian_and_gradients_both_paths(family, center):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    df = simulate_slope()
    if family == "binomial":
        df = df.copy(); df.loc[df.index[[5, 20]], ["n", "k"]] = 0
    m = build(family, df, center_predictors=center)
    fitted = df[df.n > 0].reset_index(drop=True) if family == "binomial" else df
    pd_ = ref_priors(m.priors, m.spec)
    logp, logp_nojac, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_logp(jacobian=False), m.pymc_model.compile_dlogp()
    names = [v.name for v in m.pymc_model.value_vars]
    jf = get_jaxified_logp(m.pymc_model)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    rng = np.random.default_rng(3)
    for _ in range(3):
        nat = dict(intercept=rng.normal(0.5, 0.7), b=rng.normal(0, 0.5, 1), sd=np.exp(rng.normal(-0.6, 0.3)), z=rng.normal(size=m.design.J),
                   sd_slope=np.exp(rng.normal(-1.0, 0.3)), z_slope=rng.normal(size=m.design.J))
        if family == "gaussian":
            nat["sigma"] = np.exp(rng.normal(-0.8, 0.3))
        point = m.value_point(**nat)
        ref, ref_nojac = _reference(m, fitted, pd_, point, True), _reference(m, fitted, pd_, point, False)
        assert np.isfinite(ref)
        assert abs(float(logp(point)) - ref) < 1e-8, (family, float(logp(point)), ref)
        assert abs(float(logp_nojac(point)) - ref_nojac) < 1e-8
        assert abs(ref - ref_nojac - (np.log(nat["sd"]) + np.log(nat["sd_slope"]) + (np.log(nat["sigma"]) if family == "gaussian" else 0.0))) < 1e-12
        assert abs(float(jf([point[n] for n in names])) - ref) < 1e-8
        grad_pt = np.asarray(dlogp(point), dtype=float)
        grad_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)).ravel() for g in jgrad(*[point[n] for n in names])])
        flat = np.concatenate([np.atleast_1d(np.asarray(point[n], dtype=float)).ravel() for n in names])
        def f(vec):
            q, i = {}, 0
            for n_ in names:
                size = int(np.prod(np.shape(point[n_])))
                q[n_] = vec[i:i + size].reshape(np.shape(point[n_])) if np.ndim(point[n_]) else np.asarray(vec[i]); i += size
            return _reference(m, fitted, pd_, q, True)
        h = 1e-5
        fd = np.array([(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in np.eye(len(flat))])
        assert np.allclose(grad_pt, fd, rtol=1e-6, atol=1e-5), (family, grad_pt, fd)
        assert np.allclose(grad_jx, grad_pt, rtol=1e-9, atol=1e-9)


# -- F4: deterministic synthetic-posterior predictions ------------------------------------------------------
def synthetic(model, draws=4, seed=8):
    """A synthetic posterior with distinct values per draw so joint (u, v) pairing is testable."""
    rng = np.random.default_rng(seed); J = model.design.J
    var = {"Intercept": (("chain", "draw"), rng.normal(1.0, 0.1, (1, draws))), "x": (("chain", "draw"), rng.normal(0.5, 0.1, (1, draws))),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, (1, draws))), "x|g_sigma": (("chain", "draw"), rng.uniform(0.2, 0.4, (1, draws))),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, (1, draws, J))), "x|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.3, (1, draws, J)))}
    if model.family == "gaussian":
        var["sigma"] = (("chain", "draw"), rng.uniform(0.3, 0.5, (1, draws)))
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": list(range(draws)), "g__factor_dim": list(model.design.levels)}))


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_predictions_known_and_unseen_groups(family):
    df = simulate_slope()
    m = build(family, df)
    post = synthetic(m).posterior
    link = (lambda e: e) if family == "gaussian" else expit
    mean_name = "mu" if family == "gaussian" else "p"
    a, bx = post["Intercept"].to_numpy(), post["x"].to_numpy()
    U, V = post["1|g"].to_numpy(), post["x|g"].to_numpy()
    levels = list(m.design.levels)
    new = pd.DataFrame({"x": [0.0, 1.0, -2.0, 1.0, 1.0, 3.0, 1.0], "g": [levels[2], levels[2], levels[0], "new_a", "new_a", "new_a", "new_b"], "n": [5, 5, 5, 0, 5, 5, 5]})
    for mode in ("population", "resample_observed"):
        out = m.predict(synthetic(m), newdata=new, kind="mean", new_groups=mode, seed=11)
        eta = out[mean_name].to_numpy() if family == "gaussian" else np.log(out[mean_name].to_numpy() / (1 - out[mean_name].to_numpy()))
        # known groups: a + b x + u_j + x v_j, exact label identity and row order
        for i, (xv, lvl) in enumerate(zip(new.x[:3], new.g[:3])):
            j = levels.index(lvl)
            assert np.allclose(eta[..., i], a + bx * xv + U[..., j] + xv * V[..., j], atol=1e-10)
        assert np.isfinite(eta).all()
        # unseen label: eta = a + b x + u* + x v* with one (u*, v*) pair per label per draw; solve from rows 3, 4, 5 (x = 1, 1, 3)
        assert np.allclose(eta[..., 3], eta[..., 4])                                   # repeated rows share the pair
        v_star = (eta[..., 5] - eta[..., 3] - 2 * bx) / 2.0
        u_star = eta[..., 3] - a - bx - v_star
        sd0, sd1 = post["1|g_sigma"].to_numpy(), post["x|g_sigma"].to_numpy()
        if mode == "resample_observed":                                                  # both effects from ONE observed group per draw
            found = [any(np.isclose(u_star[0, d], U[0, d, j]) and np.isclose(v_star[0, d], V[0, d, j]) for j in range(len(levels))) for d in range(eta.shape[1])]
            assert all(found), (u_star, v_star)
        else:                                                                            # a fresh independent pair: not an observed group's pair
            found = [any(np.isclose(u_star[0, d], U[0, d, j]) and np.isclose(v_star[0, d], V[0, d, j]) for j in range(len(levels))) for d in range(eta.shape[1])]
            assert not any(found)
            assert np.isfinite(u_star).all() and np.isfinite(v_star).all()
        v_b = None
        eta_b = eta[..., 6]                                                              # a second label: its own pair
        assert not np.allclose(eta_b, eta[..., 3])
        if family == "binomial":
            assert np.allclose(out["expected"].to_numpy(), link(eta) * new.n.to_numpy(), atol=1e-10)
            resp = m.predict(synthetic(m), newdata=new, kind="response", new_groups=mode, seed=1)
            assert resp["y"].dtype.kind == "i" and int(resp["y"][..., 3].max()) == 0 and np.all(resp["y"].to_numpy() <= new.n.to_numpy())
            assert np.allclose(m.predict(synthetic(m), newdata=new.drop(columns="n"), kind="probability", new_groups=mode, seed=11)["p"].to_numpy(), link(eta), atol=1e-10)
        elif family == "bernoulli":
            resp = m.predict(synthetic(m), newdata=new, kind="response", new_groups=mode, seed=11)   # same seed: same new-label pairs
            assert set(np.unique(resp["y"].to_numpy())) <= {0, 1} and np.allclose(resp["p"].to_numpy(), link(eta), atol=1e-10)
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(synthetic(m), newdata=new, kind="mean")
    ins = m.predict(synthetic(m), kind="mean")
    gi = m.design.group_index; xv = m.data.x.to_numpy()
    eta_in = a[..., None] + bx[..., None] * xv + U[..., gi] + xv * V[..., gi]
    assert np.allclose(ins[mean_name].to_numpy(), link(eta_in), atol=1e-10)
    # a varying-intercept-only model is unchanged: one effect per new label
    m0 = build(family, df, priors=ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1) if family == "gaussian" else None),
               formula=f"{LHS[family]} ~ x + (1 | g)")
    post0 = synthetic(m0).posterior.drop_vars(["x|g_sigma", "x|g"])
    out0 = m0.predict(az.InferenceData(posterior=post0), newdata=new, kind="mean", new_groups="resample_observed", seed=2)
    eta0 = out0[mean_name].to_numpy() if family == "gaussian" else np.log(out0[mean_name].to_numpy() / (1 - out0[mean_name].to_numpy()))
    u0 = eta0[..., 3] - post0["Intercept"].to_numpy() - post0["x"].to_numpy()
    assert all(any(np.isclose(u0[0, d], post0["1|g"].to_numpy()[0, d, j]) for j in range(len(levels))) for d in range(4))


# -- F5: SD prior forms --------------------------------------------------------------------------------------
def test_sd_prior_forms_and_refusals():
    df = simulate_slope()
    def pri(sd):
        return ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=sd, sigma=ab.HalfNormal(1))
    with pytest.raises(ab.PriorError, match="needs one SD prior per varying effect"):
        build("gaussian", df, priors=pri(ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="missing \\['x'\\]"):
        build("gaussian", df, priors=pri({"Intercept": ab.HalfNormal(1)}))
    with pytest.raises(ab.PriorError, match="unexpected \\['w'\\]"):
        build("gaussian", df, priors=pri({"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1), "w": ab.HalfNormal(1)}))
    with pytest.raises(ab.PriorError, match="missing \\['Intercept'\\]"):
        build("gaussian", df, priors=pri({"x": ab.HalfNormal(1)}))
    with pytest.raises(ab.PriorError, match="positive-support prior"):
        build("gaussian", df, priors=pri({"Intercept": ab.HalfNormal(1), "x": ab.Normal(0, 1)}))
    with pytest.raises(ab.PriorError, match="sd must be a Prior or a mapping"):
        build("gaussian", df, priors=pri([ab.HalfNormal(1)]))
    with pytest.raises(ab.PriorError, match="unexpected \\['x'\\]"):
        ab.gaussian("y ~ x + (1 | g)", df, pri({"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}))
    m = build("gaussian", df, priors=pri({"Intercept": ab.HalfNormal(1), "x": ab.Exponential(3.0)}))
    assert m.priors.sd_priors(m.spec) == {"Intercept": ab.HalfNormal(1), "x": ab.Exponential(3.0)}
    bp = m.priors.to_bambi(m.spec, "g", "gaussian")
    assert set(bp) >= {"1|g", "x|g"}


# -- F6/F10: describe(), fit record, value_point -------------------------------------------------------------
def test_describe_record_and_value_point():
    df = simulate_slope()
    m = build("bernoulli", df)
    text = m.describe()
    for s in ["independent varying slope", "u[j] = sd_g * z0[j]", "v[j] = sd_x|g * z1[j]", "independent by construction ('||')",
              "multiplies the RAW x_x", "independence is stated for the intercept at x_x = 0", "u' = u + c v correlated with v",
              "sd_g ~ Exponential(lam=2)", "sd_x|g ~ HalfCauchy(beta=0.5)", "x|g_sigma_log__", "z1 (varying-slope offsets)", "x|g = v",
              "+ u[g_i] + x_x,i * v[g_i]"]:
        assert s in text, s
    fit = m.fit(backend="pymc", chains=1, draws=40, tune=40, seed=2, cores=1)
    rec = ab.fit_record(m, fit)
    assert rec["model"]["varying_slope"] == "x" and rec["model"]["group_effects"] == ["Intercept", "x"]
    assert rec["model"]["priors"]["sd:Intercept"] == "Exponential(lam=2)" and rec["model"]["priors"]["sd:x"] == "HalfCauchy(beta=0.5)"
    assert list(rec["posterior"]) == ["Intercept", "x", "1|g_sigma", "x|g_sigma"]
    json.dumps(rec)
    pt = m.value_point(intercept=0.1, b=[0.2], sd=0.5, z=np.zeros(m.design.J), sd_slope=0.3, z_slope=np.ones(m.design.J))
    assert set(pt) == {v.name for v in m.pymc_model.value_vars} and float(pt["x|g_sigma_log__"]) == np.log(0.3)
    with pytest.raises(ValueError, match="sd_slope and z_slope missing"):
        m.value_point(intercept=0.1, b=[0.2], sd=0.5, z=np.zeros(m.design.J))
    m0 = ab.gaussian("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    rec0 = ab.fit_record(m0, m0.fit(backend="pymc", chains=1, draws=30, tune=30, seed=2, cores=1))
    assert rec0["model"]["priors"]["sd"] == "HalfNormal(sigma=1)" and rec0["model"]["varying_slope"] is None and "sd_slope" not in m0.names
    with pytest.raises(ValueError, match="sd_slope and z_slope given, but this model has no varying slope"):
        m0.value_point(intercept=0.1, b=[0.2], sd=0.5, z=np.zeros(m0.design.J), sigma=0.4, sd_slope=0.3, z_slope=np.zeros(m0.design.J))
