"""M1: Bernoulli varying-intercept model with a logit link. Checks A1-A8 of docs/history/m1-decision.md."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import autobayes as ab
from tests.reference_logp import reference_logp_bernoulli

PRIORS = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
PRIORS_DICT = {
    "intercept": ("Normal", {"mu": 0.0, "sigma": 1.5}),
    "b": {"x": ("Normal", {"mu": 0.0, "sigma": 1.0}), "w": ("Normal", {"mu": 0.0, "sigma": 1.0})},
    "sd": ("HalfNormal", {"sigma": 1.0}),
}


def simulate_binary(n=120, J=6, seed=0, a=-0.3, b=0.8, sd=0.7, extra=False):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, J, n)
    x = rng.normal(size=n)
    u = rng.normal(0, sd, J)
    eta = a + b * x + u[g]
    df = pd.DataFrame({"x": x, "g": [f"lvl{k:02d}" for k in g]})
    if extra:
        df["w"] = rng.normal(size=n)
        eta = eta - 0.5 * df["w"].to_numpy()
    df["y"] = rng.binomial(1, expit(eta))
    return df


@pytest.fixture(scope="module")
def bdata():
    return simulate_binary()


@pytest.fixture(scope="module")
def bmodel(bdata):
    return ab.bernoulli("y ~ x + (1 | g)", bdata, PRIORS)


@pytest.fixture(scope="module")
def bfit(bmodel):
    return bmodel.fit(backend="pymc", chains=2, draws=200, tune=200, seed=1, cores=1)


def _random_points(model, k=5, seed=3, scale=1.0):
    """Points with moderate logits: |eta| stays well below 15 for these draws."""
    rng = np.random.default_rng(seed)
    return [dict(intercept=rng.normal(0, scale), b=rng.normal(0, scale, len(model.spec.predictors)),
                 sd=np.exp(rng.normal(-0.5, 0.5)), z=rng.normal(size=model.design.J)) for _ in range(k)]


def _ref_point(nat):
    return {"a_c": nat["intercept"], "b": nat["b"], "log_sd": np.log(nat["sd"]), "z": nat["z"]}


def _eta(model, nat):
    d = model.design
    return nat["intercept"] + d.X_model @ nat["b"] + nat["sd"] * nat["z"][d.group_index]


# -- A1, A2 ------------------------------------------------------------------------------
@pytest.mark.parametrize("formula,extra,center", [
    ("y ~ x + (1 | g)", False, True),
    ("y ~ x + (1 | g)", False, False),
    ("y ~ x + w + (1 | g)", True, True),
    ("y ~ 1 + (1 | g)", False, True),
])
def test_logp_matches_reference(formula, extra, center):
    df = simulate_binary(extra=extra)
    m = ab.bernoulli(formula, df, PRIORS, center_predictors=center)
    preds = list(m.spec.predictors)
    logp = m.pymc_model.compile_logp()
    logp_nojac = m.pymc_model.compile_logp(jacobian=False)
    for nat in _random_points(m):
        assert np.max(np.abs(_eta(m, nat))) < 15
        point = m.value_point(**nat)
        kw = dict(response="y", predictors=preds, group="g", priors=PRIORS_DICT, center=center)
        ref = reference_logp_bernoulli(_ref_point(nat), df, jacobian=True, **kw)
        ref_nojac = reference_logp_bernoulli(_ref_point(nat), df, jacobian=False, **kw)
        assert np.isfinite(ref)
        assert abs(float(logp(point)) - ref) < 1e-8, (float(logp(point)), ref)          # A1
        assert abs(float(logp_nojac(point)) - ref_nojac) < 1e-8                          # A2
        assert abs(ref - ref_nojac - np.log(nat["sd"])) < 1e-12


# -- A3 ----------------------------------------------------------------------------------
def test_gradient_matches_finite_differences(bmodel, bdata):
    dlogp = bmodel.pymc_model.compile_dlogp()
    names = [v.name for v in bmodel.pymc_model.value_vars]
    for nat in _random_points(bmodel, k=3):
        point = bmodel.value_point(**nat)
        grad = np.asarray(dlogp(point), dtype=float)
        sizes = [int(np.asarray(point[n]).size) for n in names]

        def f(vec):
            q, i = {}, 0
            for n_, s in zip(names, sizes):
                q[n_] = vec[i:i + s] if s > 1 else float(vec[i])
                i += s
            nat2 = dict(intercept=q["Intercept"], b=[q[p] for p in bmodel.spec.predictors],
                        sd=np.exp(q["1|g_sigma_log__"]), z=np.asarray(q["1|g_offset"]))
            return reference_logp_bernoulli(_ref_point(nat2), bdata, response="y", predictors=["x"],
                                            group="g", priors=PRIORS_DICT, center=True, jacobian=True)
        vec = np.concatenate([np.atleast_1d(np.asarray(point[n], dtype=float)).ravel() for n in names])
        fd = np.zeros_like(vec)
        h = 1e-5
        for i in range(len(vec)):
            e = np.zeros_like(vec); e[i] = h
            fd[i] = (f(vec + e) - f(vec - e)) / (2 * h)
        assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6), (grad, fd)


# -- A4 ----------------------------------------------------------------------------------
def test_jax_logp_matches_pytensor(bmodel):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(bmodel.pymc_model)
    logp = bmodel.pymc_model.compile_logp()
    names = [v.name for v in bmodel.pymc_model.value_vars]
    for nat in _random_points(bmodel, k=3):
        point = bmodel.value_point(**nat)
        assert abs(float(jf([point[n] for n in names])) - float(logp(point))) < 1e-8


# -- A5 (replaced by R1/R2 after the review; see docs/history/m1-decision.md, repair record) --------
EXTREME = [0.0, 0.5, -0.5, 12.0, -12.0, 40.0, -40.0, 300.0, -300.0, 1000.0, -1000.0, 1e6, -1e6]


def _analytic_gradient(model, data, nat):
    """d log p / d (value variables) from the equations, with the priors of PRIORS."""
    d = model.design
    eta = _eta(model, nat)
    r = data.y.to_numpy() - expit(eta)                      # d loglik / d eta_i, stable
    a, sd, z = nat["intercept"], nat["sd"], nat["z"]
    out = {"Intercept": r.sum() - a / 1.5 ** 2,
           "1|g_sigma_log__": (r * z[d.group_index]).sum() * sd - sd ** 2 + 1.0,   # HalfNormal(1) + Jacobian
           "1|g_offset": np.array([r[d.group_index == j].sum() * sd - z[j] for j in range(d.J)])}
    for i, pname in enumerate(model.spec.predictors):
        out[pname] = (r * d.X_model[:, i]).sum() - nat["b"][i]
    return out


def test_extreme_logits_density_and_gradient_both_paths(bmodel, bdata):
    """R1/R2: value and gradient equal the stable reference at every logit, PyTensor and JAX."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp

    pmm = bmodel.pymc_model
    logp, dlogp = pmm.compile_logp(), pmm.compile_dlogp()
    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    kw = dict(response="y", predictors=["x"], group="g", priors=PRIORS_DICT, center=True, jacobian=True)
    rng = np.random.default_rng(5)
    assert set(np.unique(bdata.y)) == {0, 1}
    for a in EXTREME:
        for b, sd in [(0.0, 0.1), (0.7, 0.5)]:
            nat = dict(intercept=a, b=np.array([b]), sd=sd, z=rng.normal(size=bmodel.design.J))
            point = bmodel.value_point(**nat)
            ref = reference_logp_bernoulli(_ref_point(nat), bdata, **kw)
            gref = _analytic_gradient(bmodel, bdata, nat)
            gref_vec = np.concatenate([np.atleast_1d(gref[n]) for n in names])
            tol = 1e-8 * max(1.0, abs(ref))
            v_pt, g_pt = float(logp(point)), np.asarray(dlogp(point), dtype=float)
            v_jx = float(jf([point[n] for n in names]))
            g_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float))
                                   for g in jgrad(*[jnp.asarray(point[n]) for n in names])])
            assert np.isfinite(v_pt) and np.isfinite(v_jx)
            assert abs(v_pt - ref) < tol, (a, b, sd, v_pt, ref)
            assert abs(v_jx - ref) < tol, (a, b, sd, v_jx, ref)
            assert np.all(np.isfinite(g_pt)) and np.all(np.isfinite(g_jx))
            assert np.allclose(g_pt, gref_vec, atol=1e-8, rtol=1e-10), (a, b, sd, g_pt, gref_vec)
            assert np.allclose(g_jx, gref_vec, atol=1e-8, rtol=1e-10), (a, b, sd, g_jx, gref_vec)


def test_reviewer_case_two_zeros_two_ones():
    """The likelihood term alone: -80 at eta = 40 and -600 at eta = 300 (was -72.087 for both)."""
    df = pd.DataFrame({"y": [0, 1, 0, 1], "g": ["a", "a", "b", "b"]})
    m = ab.bernoulli("y ~ 1 + (1 | g)", df, PRIORS)
    import pytensor
    pmm = m.pymc_model
    lik_term = pmm.logp(vars=[pmm["y"]], jacobian=False)              # the likelihood term alone
    lik = pmm.compile_fn(lik_term, inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    dlik = pmm.compile_fn(pytensor.grad(lik_term, wrt=pmm.value_vars), inputs=pmm.value_vars,
                          point_fn=True, on_unused_input="ignore")
    i_a = [v.name for v in pmm.value_vars].index("Intercept")
    for eta, expected in [(40.0, -80.0), (300.0, -600.0), (-40.0, -80.0), (-300.0, -600.0), (1000.0, -2000.0)]:
        point = m.value_point(intercept=eta, b=[], sd=1e-12, z=np.zeros(2))
        assert abs(float(lik(point)) - expected) < 1e-8 * abs(expected)
        g = [np.asarray(x, dtype=float) for x in dlik(point)]
        assert all(np.all(np.isfinite(x)) for x in g)
        assert abs(float(g[i_a]) - (2.0 - 4.0 * expit(eta))) < 1e-8     # d/da sum_i (y_i - expit(eta))


# -- A6 ----------------------------------------------------------------------------------
def test_design_and_probability_from_graph(bmodel, bdata):
    d = bmodel.design
    assert d.levels == tuple(sorted(bdata.g.unique()))
    assert np.array_equal(d.group_index, np.array([d.levels.index(v) for v in bdata.g]))
    assert np.allclose(d.X_model[:, 0], bdata.x.to_numpy() - bdata.x.mean())
    assert np.array_equal(d.y, bdata.y.to_numpy())
    nat = _random_points(bmodel, k=1)[0]
    point = bmodel.value_point(**nat)
    pmm = bmodel.pymc_model
    eta_v, u_v = pmm.replace_rvs_by_values([pmm["logit_p"], pmm["1|g"]])
    fn = pmm.compile_fn([eta_v, u_v], inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    eta_graph, u_graph = (np.asarray(a).ravel() for a in fn(point))
    assert np.allclose(eta_graph, _eta(bmodel, nat), atol=1e-12)
    assert np.allclose(expit(eta_graph), expit(_eta(bmodel, nat)), atol=1e-12)
    assert np.allclose(u_graph, nat["sd"] * nat["z"], atol=1e-12)
    assert "logit link" in bmodel.describe() and "P(y_i = 1) = p_i = expit(eta_i)" in bmodel.describe()
    assert "softplus" in bmodel.describe() and {v.name for v in pmm.deterministics} == {"1|g", "logit_p"}
    assert {v.name for v in pmm.free_RVs} == {"Intercept", "x", "1|g_sigma", "1|g_offset"}


# -- A7 ----------------------------------------------------------------------------------
def test_invalid_inputs(bdata):
    with pytest.raises(ab.FormulaError, match="outside \\{0, 1\\}"):
        ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y * 2), PRIORS)
    with pytest.raises(ab.FormulaError, match="outside \\{0, 1\\}"):
        ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y + 0.5), PRIORS)
    with pytest.raises(ab.FormulaError, match="boolean"):
        ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y.astype(bool)), PRIORS)
    with pytest.raises(ab.FormulaError, match="numeric 0/1"):
        ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y.map({0: "no", 1: "yes"})), PRIORS)
    with pytest.raises(ab.FormulaError, match="constant"):
        ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=1), PRIORS)
    with pytest.raises(ab.FormulaError, match="missing values"):
        ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y.where(bdata.index > 0)), PRIORS)
    with pytest.raises(ab.PriorError, match="no residual SD"):
        ab.bernoulli("y ~ x + (1 | g)", bdata, ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1),
                                                          sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="needs a prior on the SD of its varying intercept"):   # sd missing (M3c-1: refused at build)
        ab.bernoulli("y ~ x + (1 | g)", bdata, ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="positive-support"):
        ab.bernoulli("y ~ x + (1 | g)", bdata, ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="needs a prior on the residual SD"):
        ab.gaussian("y ~ x + (1 | g)", bdata, PRIORS)
    with pytest.raises(ValueError, match="family must be"):
        ab.Model("y ~ x + (1 | g)", bdata, PRIORS, family="not_a_family")
    for bad, needle in [("y ~ x + (0 + x | g)", "varying slopes"), ("y ~ x + (1 | g/h/k)", "only two factors")]:   # A03 accepts '(1 | g/h)', A02 '(1 + x | g)', F08 '(x | g)'
        with pytest.raises(ab.FormulaError, match=needle):
            ab.bernoulli(bad, bdata, PRIORS)
    m = ab.bernoulli("y ~ x + (1 | g)", bdata, PRIORS)
    with pytest.raises(ValueError, match="sigma"):
        m.value_point(intercept=0.0, b=[0.0], sd=1.0, z=np.zeros(m.design.J), sigma=1.0)


# -- A8 ----------------------------------------------------------------------------------
def test_prediction_meaning(bmodel, bfit, bdata):
    post = bfit.idata.posterior
    a = post["Intercept"].to_numpy(); b = post["x"].to_numpy(); u = post["1|g"].to_numpy()
    p_hand = expit(a[..., None] + b[..., None] * bdata.x.to_numpy() + u[..., bmodel.design.group_index])

    mean = bmodel.predict(bfit, kind="mean")
    assert set(mean.data_vars) == {"p"} and mean["p"].dims == ("chain", "draw", "__obs__")
    assert mean.sizes["__obs__"] == len(bdata)
    assert float(mean["p"].min()) > 0.0 and float(mean["p"].max()) < 1.0
    assert np.allclose(mean["p"].to_numpy(), p_hand, atol=1e-10)

    resp = bmodel.predict(bfit, kind="response", seed=0)
    assert set(resp.data_vars) == {"p", "y"}
    yv = resp["y"].to_numpy()
    assert set(np.unique(yv)) <= {0, 1} and np.issubdtype(yv.dtype, np.integer)
    p_bar = resp["p"].to_numpy().mean(axis=(0, 1))
    y_bar = yv.mean(axis=(0, 1))
    n_draws = yv.shape[0] * yv.shape[1]
    se = np.sqrt(p_bar * (1 - p_bar) / n_draws)
    assert np.all(np.abs(y_bar - p_bar) < 4 * se + 1e-12)
    assert abs(y_bar.mean() - p_bar.mean()) < 0.01

    new = pd.DataFrame({"x": [0.0, 1.0, 0.0], "g": ["lvl00", "lvl03", "lvl00"]})
    obs = bmodel.predict(bfit, newdata=new, kind="mean")
    lv = list(bmodel.design.levels)
    for i, (xv, gv) in enumerate(zip(new.x, new.g)):
        assert np.allclose(obs["p"][..., i].to_numpy(), expit(a + b * xv + u[..., lv.index(gv)]), atol=1e-10)


def test_new_group_semantics(bmodel, bfit):
    from scipy.special import logit

    post = bfit.idata.posterior
    a = post["Intercept"].to_numpy(); u = post["1|g"].to_numpy(); sd = post["1|g_sigma"].to_numpy()
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["lvl00", "NEW_A", "NEW_B"]})
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        bmodel.predict(bfit, newdata=new)

    pop = bmodel.predict(bfit, newdata=new, kind="response", new_groups="population", seed=1)
    assert set(pop.data_vars) == {"p", "y"} and pop.attrs["new_levels"] == ["NEW_A", "NEW_B"]
    assert np.allclose(pop["p"][..., 0].to_numpy(), expit(a + u[..., 0]), atol=1e-10)   # observed level untouched
    dev_a = logit(pop["p"][..., 1].to_numpy()) - a                                        # fresh logit-scale effect
    dev_b = logit(pop["p"][..., 2].to_numpy()) - a
    target = np.sqrt(np.mean(sd ** 2))
    assert abs(dev_a.std() - target) < 0.15 * target
    assert abs(np.corrcoef(dev_a.ravel(), dev_b.ravel())[0, 1]) < 0.15
    assert set(np.unique(pop["y"].to_numpy())) <= {0, 1}

    res = bmodel.predict(bfit, newdata=new, kind="mean", new_groups="resample_observed", seed=1)
    dev = logit(res["p"][..., 1].to_numpy()) - a
    assert np.isclose(dev[..., None], u, atol=1e-8).any(-1).all()


def test_fit_record_and_report(bmodel, bfit):
    import json
    rec = ab.fit_record(bmodel, bfit)
    json.dumps(rec)
    assert rec["model"]["family"] == "bernoulli" and rec["model"]["link"] == "logit"
    assert "sigma" not in rec["model"]["priors"] and set(rec["posterior"]) == {"Intercept", "x", "1|g_sigma"}
    assert "logit scale" in rec["model"]["intercept_prior_applies_to"]
    assert bfit.backend == "pymc" and bfit.dtype == "float64"
    g = ab.gaussian("y ~ x + (1 | g)", bmodel.data.assign(y=bmodel.data.y.astype(float)),
                    ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    grec = ab.fit_record(g, g.fit(backend="pymc", chains=2, draws=20, tune=20, seed=1, cores=1))
    with pytest.raises(ValueError, match="different models"):
        ab.compare_records(rec, grec)


def test_gpu_unavailable_is_error_for_this_family(bmodel):
    jax = pytest.importorskip("jax")
    if any(d.platform in {"gpu", "cuda", "rocm"} for d in jax.devices()):
        pytest.skip("a GPU is visible")
    with pytest.raises(ab.DeviceError, match="does not fall back"):
        bmodel.fit(backend="numpyro", device="gpu", chains=1, draws=5, tune=5)


# -- U1-U4 (docs/history/m1-decision.md, follow-up unit): unseen-label identity, Bernoulli ------
def test_resample_observed_keeps_distinct_unseen_labels_bernoulli(bmodel, bfit):
    from tests.test_predict import check_resample_observed_identity, synthetic_posterior

    idata = synthetic_posterior(bmodel, effects=np.linspace(-2.5, 2.5, bmodel.design.J))
    check_resample_observed_identity(bmodel, idata, "p", expit)
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["NEW_A", "NEW_B", "NEW_A"]})
    ds = bmodel.predict(idata, newdata=new, kind="response", new_groups="resample_observed", seed=1)
    yv = ds["y"].to_numpy()
    assert set(np.unique(yv)) <= {0, 1} and np.issubdtype(yv.dtype, np.integer)
    assert abs(yv.mean() - ds["p"].to_numpy().mean()) < 0.02
    # on the real fit: same label shares, distinct labels are not collapsed
    ds = bmodel.predict(bfit, newdata=new, kind="mean", new_groups="resample_observed", seed=2)
    v = ds["p"].to_numpy()
    assert np.allclose(v[..., 0], v[..., 2]) and np.isclose(v[..., 0], v[..., 1]).mean() < 0.5


def test_complex_inputs_are_refused_before_casting(bdata):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)     # a cast would raise here
        with pytest.raises(ab.FormulaError, match="response 'y' is complex"):
            ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y.astype(complex) + 4j), PRIORS)
        with pytest.raises(ab.FormulaError, match="response 'y' is complex"):
            ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y.astype(complex)), PRIORS)
        with pytest.raises(ab.FormulaError, match="column 'x' is complex"):
            ab.bernoulli("y ~ x + (1 | g)", bdata.assign(x=bdata.x.astype(complex)), PRIORS)
    m = ab.bernoulli("y ~ x + (1 | g)", bdata.assign(y=bdata.y.astype(float)), PRIORS)   # real float 0/1 still fine
    fit = m.fit(backend="pymc", chains=1, draws=10, tune=10, seed=1, cores=1)
    with pytest.raises(ab.FormulaError, match="newdata column 'x' is complex"):
        m.predict(fit, newdata=pd.DataFrame({"x": np.array([0.5 + 0j]), "g": ["lvl00"]}), kind="mean")
