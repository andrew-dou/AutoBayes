"""A1-A6: the Bambi-built PyMC model means what the reference says it means."""
import numpy as np
import pymc as pm
import pytest

import autobayes as ab
from tests.conftest import PRIORS, PRIORS_DICT, simulate
from tests.reference_logp import conditional_gaussian_posterior, reference_logp


def _random_points(model, k=5, seed=3):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(k):
        nat = dict(intercept=rng.normal(1, 1), b=rng.normal(0, 1, len(model.spec.predictors)),
                   sigma=np.exp(rng.normal(-1, 0.5)), sd=np.exp(rng.normal(-0.5, 0.5)),
                   z=rng.normal(size=model.design.J))
        pts.append(nat)
    return pts


def _ref_point(nat):
    return {"a_c": nat["intercept"], "b": nat["b"], "log_sigma": np.log(nat["sigma"]),
            "log_sd": np.log(nat["sd"]), "z": nat["z"]}


@pytest.mark.parametrize("formula,extra,center", [
    ("y ~ x + (1 | g)", False, True),
    ("y ~ x + (1 | g)", False, False),
    ("y ~ x + w + (1 | g)", True, True),
    ("y ~ 1 + (1 | g)", False, True),
])
def test_logp_matches_reference(formula, extra, center):
    df = simulate(extra=extra)
    m = ab.gaussian(formula, df, PRIORS, center_predictors=center)
    preds = list(m.spec.predictors)
    logp = m.pymc_model.compile_logp()
    logp_nojac = m.pymc_model.compile_logp(jacobian=False)
    for nat in _random_points(m):
        point = m.value_point(**nat)
        ref = reference_logp(_ref_point(nat), df, response="y", predictors=preds, group="g",
                             priors=PRIORS_DICT, center=center, jacobian=True)
        ref_nojac = reference_logp(_ref_point(nat), df, response="y", predictors=preds, group="g",
                                   priors=PRIORS_DICT, center=center, jacobian=False)
        assert np.isfinite(ref)
        assert abs(float(logp(point)) - ref) < 1e-8, (float(logp(point)), ref)            # A1
        assert abs(float(logp_nojac(point)) - ref_nojac) < 1e-8                            # A2
        assert abs(ref - ref_nojac - (np.log(nat["sigma"]) + np.log(nat["sd"]))) < 1e-12


def test_gradient_matches_finite_differences(model, data):
    """A3: PyMC's gradient vs central differences of the reference."""
    dlogp = model.pymc_model.compile_dlogp()
    names = [v.name for v in model.pymc_model.value_vars]
    for nat in _random_points(model, k=3):
        point = model.value_point(**nat)
        grad = np.asarray(dlogp(point), dtype=float)
        # flatten the reference gradient in PyMC's value-var order
        def f(vec):
            q = _unflatten(vec, names, point)
            nat2 = dict(intercept=q["Intercept"], b=[q[p] for p in model.spec.predictors],
                        sigma=np.exp(q["sigma_log__"]), sd=np.exp(q["1|g_sigma_log__"]), z=q["1|g_offset"])
            return reference_logp(_ref_point(nat2), data, response="y", predictors=list(model.spec.predictors),
                                  group="g", priors=PRIORS_DICT, center=True, jacobian=True)
        vec = _flatten(point, names)
        fd = np.zeros_like(vec)
        h = 1e-5
        for i in range(len(vec)):
            e = np.zeros_like(vec); e[i] = h
            fd[i] = (f(vec + e) - f(vec - e)) / (2 * h)
        assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6), (grad, fd)


def _flatten(point, names):
    return np.concatenate([np.atleast_1d(np.asarray(point[n], dtype=float)).ravel() for n in names])


def _unflatten(vec, names, template):
    out, i = {}, 0
    for n in names:
        size = int(np.asarray(template[n]).size)
        chunk = vec[i:i + size]
        out[n] = chunk.reshape(np.asarray(template[n]).shape) if size > 1 else float(chunk[0])
        i += size
    return out


def test_jax_logp_matches_pytensor(model):
    """A4: the JAX-compiled density (what NumPyro samples) equals the C/NumPy path."""
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    # PyMC 5.28.5 quirk: the flag is named backwards. The default (negative_logp=True)
    # returns model.logp() itself; passing False negates it. Verified in source
    # (pymc/sampling/jax.py, get_jaxified_logp) and by this test.
    jf = get_jaxified_logp(model.pymc_model)
    logp = model.pymc_model.compile_logp()
    names = [v.name for v in model.pymc_model.value_vars]
    for nat in _random_points(model, k=3):
        point = model.value_point(**nat)
        a = float(jf([point[n] for n in names]))
        b = float(logp(point))
        assert abs(a - b) < 1e-8, (a, b)


def test_design_and_group_coding(model, data):
    """A6: design matrices, level order, and mu from the graph vs by hand."""
    d = model.design
    assert d.levels == tuple(sorted(data.g.unique()))
    assert np.array_equal(d.group_index, np.array([d.levels.index(v) for v in data.g]))
    assert np.allclose(d.Z[np.arange(d.n), d.group_index], 1.0) and np.allclose(d.Z.sum(1), 1.0)
    assert np.allclose(d.X_raw[:, 0], data.x.to_numpy())
    assert np.allclose(d.X_model[:, 0], data.x.to_numpy() - data.x.mean())
    assert np.allclose(d.x_means, [data.x.mean()])
    # mu from the PyMC graph at a point
    nat = _random_points(model, k=1)[0]
    point = model.value_point(**nat)
    mu_fn = model.pymc_model.compile_fn(model.pymc_model["mu"], point_fn=True)
    mu_graph = np.asarray(mu_fn(point)).ravel()
    mu_hand = nat["intercept"] + d.X_model @ nat["b"] + nat["sd"] * nat["z"][d.group_index]
    assert np.allclose(mu_graph, mu_hand, atol=1e-12)
    u_fn = model.pymc_model.compile_fn(model.pymc_model["1|g"], point_fn=True)
    assert np.allclose(np.asarray(u_fn(point)).ravel(), nat["sd"] * nat["z"], atol=1e-12)


def test_hessian_is_analytic_precision(model, data):
    """A5: with the scales fixed, the log density is the closed-form Gaussian."""
    pmm = model.pymc_model
    sigma, sd = 0.3, 0.5
    fixed = pm.do(pmm, {pmm["sigma"]: sigma, pmm["1|g_sigma"]: sd})
    names = [v.name for v in fixed.value_vars]
    assert set(names) == {"Intercept", "x", "1|g_offset"}
    mean, cov, P = conditional_gaussian_posterior(
        data, response="y", predictors=["x"], group="g", center=True, sigma=sigma, sd=sd,
        prior_intercept={"mu": 0.0, "sigma": 5.0}, prior_b={"x": {"mu": 0.0, "sigma": 2.0}})
    d2 = fixed.compile_d2logp()
    point = {"Intercept": np.asarray(0.7), "x": np.asarray(0.2), "1|g_offset": np.zeros(model.design.J)}
    H = np.asarray(d2(point))
    # order in fixed.value_vars may differ from [a, b, z]; map it
    order = []
    for n in names:
        if n == "Intercept": order.append([0])
        elif n == "x": order.append([1])
        else: order.append(list(range(2, 2 + model.design.J)))
    idx = np.concatenate(order)
    assert np.allclose(H, -P[np.ix_(idx, idx)], atol=1e-8)
    # and the maximiser of the log density is the exact posterior mean
    from scipy.optimize import minimize
    logp = fixed.compile_logp()
    def neg(vec):
        return -float(logp({"Intercept": vec[0], "x": vec[1], "1|g_offset": vec[2:]}))
    res = minimize(neg, np.zeros(2 + model.design.J), method="BFGS")
    assert np.allclose(res.x, mean, atol=1e-4)
