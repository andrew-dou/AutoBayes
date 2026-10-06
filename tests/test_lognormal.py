"""Lognormal responses (brms-parity A10): response validation, the sigma prior and ``value_point``, the density and
gradient by hand on both paths, by-hand predictions (the mean is ``exp(mu + sigma^2 / 2)``, not ``exp(mu)``),
``describe()``, prior-predictive shapes and the fit record. No MCMC; the brms comparison is the oracle case
``lognormal_varint``."""
from types import SimpleNamespace
import json
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.stats import halfnorm, lognorm, norm

import autobayes as ab


def lognormal_logpdf(y, mu, sigma):
    """brms's ``lognormal_lpdf(y | mu, sigma)``: the Normal density of ``log y`` minus ``log y``."""
    return norm.logpdf(np.log(y), mu, sigma) - np.log(y)


def data():
    return pd.DataFrame({"x": [-1.0, 0.5, 2.0, 3.0, 0.2, -0.7], "y": [0.4, 1.3, 2.2, 7.5, 0.9, 0.2],
                         "g": ["a", "a", "b", "b", "c", "c"]})


def priors(group=False, sigma=ab.HalfNormal(1), **extra):
    return ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=sigma,
                     **({"sd": ab.HalfNormal(1)} if group else {}), **extra)


# -- response validation --------------------------------------------------------------------------------------
@pytest.mark.parametrize("values, message", [
    ([0.0, 1.0], "non-positive value"), ([-0.5, 1.0], "non-positive value"), ([np.inf, 1.0], "non-finite value"),
    ([True, False], "numeric response"), (["1", "2"], "numeric response"), ([1 + 1j, 1.0], "complex-valued"),
    ([np.nan, 1.0], "missing values"),
])
def test_invalid_responses(values, message):
    with pytest.raises(ab.FormulaError, match=message):
        ab.lognormal("y ~ 1", pd.DataFrame({"y": values}), ab.Priors(intercept=ab.Normal(0, 2), b={}, sigma=ab.HalfNormal(1)))


def test_positive_responses_are_kept_as_given():
    m = ab.lognormal("y ~ x", data(), priors())
    assert m.design.n == 6 and np.array_equal(m.design.y, data().y) and m.design.y.dtype.kind == "f"
    assert m.warnings == [] and not m.dropped_rows
    tiny = ab.lognormal("y ~ x", data().assign(y=[1e-8, 1.3, 2.2, 7.5, 0.9, 1e8]), priors())
    assert tiny.design.y[0] == 1e-8 and tiny.design.y[-1] == 1e8


# -- the sigma prior ------------------------------------------------------------------------------------------
def test_prior_coverage():
    with pytest.raises(ab.PriorError, match="the lognormal family needs a prior on the SD of log y: pass sigma="):
        ab.lognormal("y ~ x", data(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="the lognormal family has no shape parameter; do not pass shape. Its auxiliary parameter is sigma: pass sigma=... Its intercept and slope priors are on the log scale"):
        ab.lognormal("y ~ x", data(), priors(shape=ab.Gamma(2, 0.5)))
    with pytest.raises(ab.PriorError, match="prior for 'sigma' must be a positive-support prior"):
        ab.lognormal("y ~ x", data(), priors(sigma=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="prior for 'sigma' must be a Prior"):
        ab.lognormal("y ~ x", data(), priors(sigma=1.0))
    with pytest.raises(ValueError, match="nu"):
        ab.Model("y ~ x", data(), priors(), family="lognormal", nu=4)
    with pytest.raises(ab.FormulaError, match="binomial"):
        ab.lognormal("y | trials(n) ~ x", data().assign(n=10), priors())
    for column in ("sigma", "mu"):                                     # the scale variable and the location deterministic
        with pytest.raises(ab.FormulaError, match="collision"):
            ab.lognormal(f"y ~ {column}", data().rename(columns={"x": column}), priors())
    for prior in (ab.Gamma(2, 0.5), ab.Exponential(1), ab.HalfCauchy(1), ab.HalfStudentT(3, 1)):   # any positive-support class
        assert ab.lognormal("y ~ x", data(), priors(sigma=prior)).priors.sigma is prior


# -- names and value_point ------------------------------------------------------------------------------------
def test_names_and_value_point():
    m = ab.lognormal("y ~ x + (1 | g)", data(), priors(True))
    assert m.names["sigma"] == "sigma" and "shape" not in m.names and m.names["linpred"] == "mu" and m.names["mu"] == "mu"
    assert m.names["mean"] == "mean" and m.link == "identity" and m.family_spec.continuous and m.family_spec.link_scale == "log"
    assert "sigma_log__" in {v.name for v in m.pymc_model.value_vars}
    p = m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), sigma=0.7)
    assert set(p) == {v.name for v in m.pymc_model.value_vars}
    np.testing.assert_allclose(p["sigma_log__"], np.log(0.7))
    with pytest.raises(ValueError, match="sigma is required for the lognormal family; no other auxiliary parameter applies"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), shape=0.7)
    with pytest.raises(ValueError, match="sigma is required"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3))
    with pytest.raises(ValueError, match="sigma is required"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), sigma=0.7, shape=0.7)
    with pytest.raises(ValueError, match="sigma must be a finite positive scalar"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), sigma=-1.0)


# -- density and gradient -------------------------------------------------------------------------------------
def test_hierarchical_density_by_hand_both_paths():
    df = data()
    m = ab.lognormal("y ~ x + (1 | g)", df, priors(True))
    nat = dict(intercept=0.3, b=[-0.2], sd=0.5, z=np.array([0.2, -0.4, 0.1]), sigma=0.7)
    p = m.value_point(**nat)
    x, y = df.x.to_numpy(), df.y.to_numpy()
    mu = 0.3 - 0.2 * (x - x.mean()) + 0.5 * nat["z"][pd.Categorical(df.g).codes]
    lik = lognormal_logpdf(y, mu, 0.7).sum()
    np.testing.assert_allclose(lik, lognorm.logpdf(y, s=0.7, scale=np.exp(mu)).sum())        # scipy agrees here
    prior = (norm.logpdf(0.3, 0, 2) + norm.logpdf(-0.2, 0, 1) + halfnorm.logpdf(0.5, scale=1) + norm.logpdf(nat["z"]).sum()
             + halfnorm.logpdf(0.7, scale=1))
    jac = np.log(0.5) + np.log(0.7)                              # sd and sigma are sampled on the log scale
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), lik + prior + jac, rtol=1e-10)
    np.testing.assert_allclose(m.pymc_model.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-10)
    np.testing.assert_allclose(m.pymc_model.compile_logp(vars=[m.pymc_model["sigma"]], jacobian=False)(p),
                               halfnorm.logpdf(0.7, scale=1), rtol=1e-10)
    from pymc.sampling.jax import get_jaxified_logp
    import jax
    names = [v.name for v in m.pymc_model.value_vars]
    jf = get_jaxified_logp(m.pymc_model)
    args = [p[n] for n in names]
    np.testing.assert_allclose(jf(args), lik + prior + jac, rtol=1e-10)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    g_pt = np.asarray(m.pymc_model.compile_dlogp()(p), dtype=float)
    np.testing.assert_allclose(g_jax, g_pt, rtol=1e-8, atol=1e-10)
    # d/d log(sigma) by hand: (sum_i [-1/sigma + (log y_i - mu_i)^2 / sigma^3] - sigma) * sigma + 1
    r = np.log(y) - mu
    d_sigma = (-1 / 0.7 + r ** 2 / 0.7 ** 3).sum() - 0.7
    np.testing.assert_allclose(g_pt[names.index("sigma_log__")], d_sigma * 0.7 + 1.0, rtol=1e-8)
    # d/d Intercept by hand: sum_i (log y_i - mu_i) / sigma^2 - a / 4
    np.testing.assert_allclose(g_pt[names.index("Intercept")], (r / 0.7 ** 2).sum() - 0.3 / 4, rtol=1e-8)
    assert m.design.centered and m.link == "identity"


@pytest.mark.parametrize("mu", [-30.0, 0.0, 30.0])
@pytest.mark.parametrize("sigma", [0.05, 1.0, 1e4])
def test_response_boundaries_density_both_paths(mu, sigma):
    y = np.array([1e-6, 1.0, 1e6])
    m = ab.lognormal("y ~ 1", pd.DataFrame({"y": y}), ab.Priors(intercept=ab.Normal(0, 2), b={}, sigma=ab.HalfNormal(1)))
    p = m.value_point(intercept=mu, b=[], sigma=sigma)
    ref = lognormal_logpdf(y, mu, sigma).sum() + norm.logpdf(mu, 0, 2) + halfnorm.logpdf(sigma, scale=1) + np.log(sigma)
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), ref, rtol=1e-9)
    from pymc.sampling.jax import get_jaxified_logp
    jf = get_jaxified_logp(m.pymc_model)
    np.testing.assert_allclose(jf([p[v.name] for v in m.pymc_model.value_vars]), ref, rtol=1e-9)


# -- predictions ----------------------------------------------------------------------------------------------
def draws(group=False, a=0.3, sigma=1.3):
    p = {"Intercept": np.full((2, 4), a), "x": np.full((2, 4), -0.2), "sigma": np.full((2, 4), sigma)}
    if group:
        p.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.1, -0.1, 0.3], (2, 4, 3))})
    return az.from_dict(posterior=p, coords={"g__factor_dim": ["a", "b", "c"]}, dims={"1|g": ["g__factor_dim"]})


def test_fitted_row_predictions_by_hand():
    m = ab.lognormal("y ~ x + (1 | g)", data(), priors(True))
    post = draws(True)
    u = np.array([0.1, -0.1, 0.3])[pd.Categorical(data().g).codes]
    mu = np.broadcast_to(0.3 - 0.2 * data().x.to_numpy() + u, (2, 4, 6))
    lin = m.predict(post, kind="linear")
    np.testing.assert_allclose(lin.eta, mu)
    assert lin.attrs["prediction_scale"] == "linear" and lin.attrs["link"] == "identity"
    out = m.predict(post, kind="response", seed=9)
    np.testing.assert_allclose(out["mean"], np.exp(mu + 1.3 ** 2 / 2))                  # brms's epred, not exp(mu)
    assert not np.allclose(out["mean"], np.exp(mu))
    np.testing.assert_allclose(out.y, np.random.default_rng(9).lognormal(mu, 1.3))
    assert out.y.dtype.kind == "f" and (out.y > 0).all() and set(out.data_vars) == {"mean", "y"}
    nogroup = m.predict(post, kind="mean", include_group_effects=False)
    np.testing.assert_allclose(nogroup["mean"], np.broadcast_to(np.exp(0.3 - 0.2 * data().x.to_numpy() + 1.3 ** 2 / 2), (2, 4, 6)))
    with pytest.raises(ValueError, match="kind"):
        m.predict(post, kind="probability")


@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_seeded_new_group_predictions(policy):
    m = ab.lognormal("y ~ x + (1 | g)", data(), priors(True))
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["new", "new"]})
    rng = np.random.default_rng(8)
    effect = rng.normal(size=(2, 4)) * 0.5 if policy == "population" else np.take([0.1, -0.1, 0.3], rng.integers(0, 3, size=(2, 4)))
    mu = 0.3 - 0.2 * new.x.to_numpy() + effect[..., None]
    out = m.predict(draws(True), newdata=new, kind="response", new_groups=policy, seed=8)
    np.testing.assert_allclose(out["mean"], np.exp(mu + 1.3 ** 2 / 2))
    np.testing.assert_allclose(out.y, rng.lognormal(mu, 1.3))
    assert out.attrs["new_groups"] == policy and list(out.attrs["new_levels"]) == ["new"]
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(draws(True), newdata=new, kind="mean")


def test_prediction_limits_and_moments():
    m = ab.lognormal("y ~ x", data(), priors())
    with pytest.raises(ValueError, match="overflow"):
        m.predict(draws(a=1000.0), kind="mean")
    with pytest.raises(ValueError, match="overflow"):                 # sigma^2 / 2 = 800 overflows the mean at mu = 0.3
        m.predict(draws(sigma=40.0), kind="mean")
    assert np.isfinite(m.predict(draws(sigma=40.0), kind="linear").eta).all()
    big = m.predict(draws(a=50.0), kind="response", seed=2)         # continuous: no integer-generator limit
    assert np.isfinite(big["mean"]).all() and np.isfinite(big.y).all()
    post = {"Intercept": np.full((1, 4000), np.log(5.0)), "x": np.zeros((1, 4000)), "sigma": np.full((1, 4000), 0.5)}
    y = m.predict(az.from_dict(posterior=post), kind="response", seed=1).y.to_numpy()[0, :, 0]
    assert abs(y.mean() - 5.0 * np.exp(0.125)) < 0.3 and abs(np.median(y) - 5.0) < 0.3     # mean exp(mu + sigma^2/2), median exp(mu)
    assert 0.7 * (np.exp(0.25) - 1) * 25 * np.exp(0.25) < y.var() < 1.3 * (np.exp(0.25) - 1) * 25 * np.exp(0.25)


# -- describe, prior predictive, record -----------------------------------------------------------------------
def test_describe_and_registry_constructor():
    m = ab.lognormal("y ~ x + (1 | g)", data(), priors(True))
    text = m.describe()
    assert text.startswith("AutoBayes lognormal model") and "family    : lognormal, identity link" in text
    assert "sigma ~ HalfNormal(sigma=1)" in text and "y_i ~ LogNormal(mu_i, sigma)" in text
    assert "E[y_i] = exp(mu_i + sigma^2 / 2)" in text and "median(y_i) = exp(mu_i)" in text
    assert "sigma_log__" in text and "log(sigma)" in text and "mu_i = a_c" in text
    assert "a_c is the expected log response at the predictor means" in text
    assert "location coefficients on the log scale" in text and "shape ~" not in text and "log(shape)" not in text
    assert "exp(mu + sigma^2 / 2) is computed by predict()" in text
    same = ab.Model("y ~ x + (1 | g)", data(), priors(True), family="lognormal")
    assert same.describe() == text and same.target_identity == m.target_identity
    other = ab.lognormal("y ~ x + (1 | g)", data(), priors(True, sigma=ab.Exponential(1)))
    assert other.target_identity != m.target_identity                 # the sigma prior is part of the frozen target
    gaussian = ab.gaussian("y ~ x + (1 | g)", data(), priors(True))
    assert gaussian.target_identity != m.target_identity              # same priors and data, a different family


def test_prior_predictive_shapes():
    m = ab.lognormal("y ~ x + (1 | g)", data(), priors(True))
    out = m.prior_predictive(draws=6, seed=3)
    assert out.prior["sigma"].shape == (1, 6) and (out.prior["sigma"].values > 0).all()
    y = out.prior_predictive["y"].values
    assert y.shape == (1, 6, 6) and (y > 0).all() and y.dtype.kind == "f"
    assert out.attrs["autobayes_generation_kind"] == "prior_predictive"


def test_fit_record_and_summary_rows(monkeypatch):
    m = ab.lognormal("y ~ x + (1 | g)", data(), priors(True))
    post = draws(True)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "1|g_sigma", "sigma"}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "lognormal" and rec["model"]["link"] == "identity"
    assert rec["model"]["priors"]["sigma"] == "HalfNormal(sigma=1)" and "shape" not in rec["model"]["priors"]
    assert "log scale" in rec["model"]["intercept_prior_applies_to"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
