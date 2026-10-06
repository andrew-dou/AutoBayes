"""Beta responses with a logit link (brms-parity A11): response validation, the phi prior and ``value_point``, the
density and gradient by hand on both paths, by-hand predictions, ``describe()``, prior-predictive shapes and the fit
record. No MCMC; the brms comparison is the oracle case ``beta_varint``."""
from types import SimpleNamespace
import json
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma, expit, logit
from scipy.stats import beta as beta_dist, gamma as gamma_dist, halfnorm, norm

import autobayes as ab


def beta_logpdf(y, eta, phi):
    """brms's ``beta_lpdf(y | mu * phi, (1 - mu) * phi)`` with ``mu = expit(eta)`` and ``1 - mu`` as ``expit(-eta)``."""
    return beta_dist.logpdf(y, expit(eta) * phi, expit(-eta) * phi)


def data():
    return pd.DataFrame({"x": [-1.0, 0.5, 2.0, 3.0, 0.2, -0.7], "y": [0.2, 0.55, 0.7, 0.9, 0.4, 0.1],
                         "g": ["a", "a", "b", "b", "c", "c"]})


def priors(group=False, phi=ab.Gamma(2, 0.5), **extra):
    return ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), phi=phi,
                     **({"sd": ab.HalfNormal(1)} if group else {}), **extra)


# -- response validation --------------------------------------------------------------------------------------
@pytest.mark.parametrize("values, message", [
    ([0.0, 0.5], "outside the open interval"), ([1.0, 0.5], "outside the open interval"),
    ([-0.1, 0.5], "outside the open interval"), ([1.5, 0.5], "outside the open interval"), ([np.inf, 0.5], "non-finite value"),
    ([True, False], "numeric response"), (["0.1", "0.2"], "numeric response"), ([0.1 + 0.1j, 0.5], "complex-valued"),
    ([np.nan, 0.5], "missing values"),
])
def test_invalid_responses(values, message):
    with pytest.raises(ab.FormulaError, match=message):
        ab.beta("y ~ 1", pd.DataFrame({"y": values}), ab.Priors(intercept=ab.Normal(0, 2), b={}, phi=ab.Gamma(2, 0.5)))


def test_proportions_are_kept_as_given():
    m = ab.beta("y ~ x", data(), priors())
    assert m.design.n == 6 and np.array_equal(m.design.y, data().y) and m.design.y.dtype.kind == "f"
    assert m.warnings == [] and not m.dropped_rows
    edge = ab.beta("y ~ x", data().assign(y=[1e-8, 0.55, 0.7, 1 - 1e-8, 0.4, 0.1]), priors())
    assert edge.design.y[0] == 1e-8 and edge.design.y[3] == 1 - 1e-8


# -- the phi prior --------------------------------------------------------------------------------------------
def test_prior_coverage():
    with pytest.raises(ab.PriorError, match="the Beta family needs a prior on the precision parameter phi: pass phi="):
        ab.beta("y ~ x", data(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="the beta family has no residual SD; do not pass sigma. Its auxiliary parameter is phi: pass phi=... Its intercept and slope priors are on the logit scale"):
        ab.beta("y ~ x", data(), priors(sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="the beta family has no shape parameter; do not pass shape. Its auxiliary parameter is phi"):
        ab.beta("y ~ x", data(), priors(shape=ab.Gamma(2, 0.5)))
    with pytest.raises(ab.PriorError, match="prior for 'phi' must be a positive-support prior"):
        ab.beta("y ~ x", data(), priors(phi=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="prior for 'phi' must be a Prior"):
        ab.beta("y ~ x", data(), priors(phi=2.0))
    with_phi = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), phi=ab.Gamma(2, 0.5))
    with pytest.raises(ab.PriorError, match="the gaussian family has no precision parameter; do not pass phi. Its auxiliary parameter is sigma"):
        ab.gaussian("y ~ x", data(), ab.Priors(sigma=ab.HalfNormal(1), **with_phi))
    with pytest.raises(ab.PriorError, match="the gamma family has no precision parameter; do not pass phi. Its auxiliary parameter is shape: pass shape=... Its intercept and slope priors are on the log scale"):
        ab.gamma("y ~ x", data(), ab.Priors(shape=ab.Gamma(2, 0.5), **with_phi))
    with pytest.raises(ab.PriorError, match="the bernoulli family has no precision parameter; do not pass phi. Its intercept and slope priors are on the logit scale"):
        ab.bernoulli("yb ~ x", data().assign(yb=[0, 1, 0, 1, 1, 0]), ab.Priors(**with_phi))
    with pytest.raises(ValueError, match="nu"):
        ab.Model("y ~ x", data(), priors(), family="beta", nu=4)
    with pytest.raises(ab.FormulaError, match="binomial"):
        ab.beta("y | trials(n) ~ x", data().assign(n=10), priors())
    for column in ("phi", "logit_mu"):                           # the precision variable and the linear-predictor deterministic
        with pytest.raises(ab.FormulaError, match="collision"):
            ab.beta(f"y ~ {column}", data().rename(columns={"x": column}), priors())
    for prior in (ab.HalfNormal(1), ab.Exponential(1), ab.HalfCauchy(1), ab.HalfStudentT(3, 1)):      # any positive-support class
        assert ab.beta("y ~ x", data(), priors(phi=prior)).priors.phi is prior


# -- names and value_point ------------------------------------------------------------------------------------
def test_names_and_value_point():
    m = ab.beta("y ~ x + (1 | g)", data(), priors(True))
    assert m.names["phi"] == "phi" and "sigma" not in m.names and "shape" not in m.names
    assert m.names["linpred"] == "logit_mu" and m.names["mean"] == "mu" and "mu" not in m.names   # mu = expit(eta) is not a graph variable
    assert m.link == "logit" and m.family_spec.continuous and m.family_spec.link_scale == "logit"
    assert "phi_log__" in {v.name for v in m.pymc_model.value_vars} and "logit_mu" in {d.name for d in m.pymc_model.deterministics}
    p = m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), phi=0.7)
    assert set(p) == {v.name for v in m.pymc_model.value_vars}
    np.testing.assert_allclose(p["phi_log__"], np.log(0.7))
    with pytest.raises(ValueError, match="phi is required for the Beta family; sigma and shape do not apply"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), sigma=0.7)
    with pytest.raises(ValueError, match="phi is required"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3))
    with pytest.raises(ValueError, match="phi is required"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), phi=0.7, shape=0.7)
    with pytest.raises(ValueError, match="phi must be a finite positive scalar"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), phi=0.0)
    g = ab.gaussian("y ~ x", data(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match="sigma is required for the Gaussian family"):
        g.value_point(intercept=0.3, b=[-0.2], sigma=0.5, phi=0.7)


# -- density and gradient -------------------------------------------------------------------------------------
def test_hierarchical_density_by_hand_both_paths():
    df = data()
    m = ab.beta("y ~ x + (1 | g)", df, priors(True))
    nat = dict(intercept=0.3, b=[-0.2], sd=0.5, z=np.array([0.2, -0.4, 0.1]), phi=0.7)
    p = m.value_point(**nat)
    x, y = df.x.to_numpy(), df.y.to_numpy()
    eta = 0.3 - 0.2 * (x - x.mean()) + 0.5 * nat["z"][pd.Categorical(df.g).codes]
    lik = beta_logpdf(y, eta, 0.7).sum()
    prior = (norm.logpdf(0.3, 0, 2) + norm.logpdf(-0.2, 0, 1) + halfnorm.logpdf(0.5, scale=1) + norm.logpdf(nat["z"]).sum()
             + gamma_dist.logpdf(0.7, 2, scale=1 / 0.5))
    jac = np.log(0.5) + np.log(0.7)                              # sd and phi are sampled on the log scale
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), lik + prior + jac, rtol=1e-10)
    np.testing.assert_allclose(m.pymc_model.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-10)
    np.testing.assert_allclose(m.pymc_model.compile_logp(vars=[m.pymc_model["phi"]], jacobian=False)(p),
                               gamma_dist.logpdf(0.7, 2, scale=2.0), rtol=1e-10)
    from pymc.sampling.jax import get_jaxified_logp
    import jax
    names = [v.name for v in m.pymc_model.value_vars]
    jf = get_jaxified_logp(m.pymc_model)
    args = [p[n] for n in names]
    np.testing.assert_allclose(jf(args), lik + prior + jac, rtol=1e-10)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    g_pt = np.asarray(m.pymc_model.compile_dlogp()(p), dtype=float)
    np.testing.assert_allclose(g_jax, g_pt, rtol=1e-8, atol=1e-10)
    # d/d log(phi) by hand: (sum_i [mu_i log y_i + (1 - mu_i) log(1 - y_i) - mu_i psi(mu_i phi) - (1 - mu_i) psi((1 - mu_i) phi)
    #                        + psi(phi)] + (2 - 1) / phi - 0.5) * phi + 1
    mu, phi = expit(eta), 0.7
    d_phi = (mu * np.log(y) + (1 - mu) * np.log1p(-y) - mu * digamma(mu * phi) - (1 - mu) * digamma((1 - mu) * phi)
             + digamma(phi)).sum() + (2 - 1) / phi - 0.5
    np.testing.assert_allclose(g_pt[names.index("phi_log__")], d_phi * phi + 1.0, rtol=1e-8)
    # d/d Intercept by hand: sum_i phi mu_i (1 - mu_i) [logit(y_i) - psi(mu_i phi) + psi((1 - mu_i) phi)] - a / 4
    d_a = (phi * mu * (1 - mu) * (logit(y) - digamma(mu * phi) + digamma((1 - mu) * phi))).sum() - 0.3 / 4
    np.testing.assert_allclose(g_pt[names.index("Intercept")], d_a, rtol=1e-8)
    assert m.design.centered and m.link == "logit"


@pytest.mark.parametrize("eta", [-30.0, 0.0, 30.0])
@pytest.mark.parametrize("phi", [0.05, 1.0, 1e4])
def test_response_boundaries_density_both_paths(eta, phi):
    y = np.array([1e-6, 0.5, 1 - 1e-6])
    m = ab.beta("y ~ 1", pd.DataFrame({"y": y}), ab.Priors(intercept=ab.Normal(0, 2), b={}, phi=ab.Gamma(2, 0.5)))
    p = m.value_point(intercept=eta, b=[], phi=phi)
    ref = beta_logpdf(y, eta, phi).sum() + norm.logpdf(eta, 0, 2) + gamma_dist.logpdf(phi, 2, scale=2.0) + np.log(phi)
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), ref, rtol=1e-9)
    from pymc.sampling.jax import get_jaxified_logp
    jf = get_jaxified_logp(m.pymc_model)
    np.testing.assert_allclose(jf([p[v.name] for v in m.pymc_model.value_vars]), ref, rtol=1e-9)


# -- predictions ----------------------------------------------------------------------------------------------
def draws(group=False, a=0.3, phi=2.5):
    p = {"Intercept": np.full((2, 4), a), "x": np.full((2, 4), -0.2), "phi": np.full((2, 4), phi)}
    if group:
        p.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.1, -0.1, 0.3], (2, 4, 3))})
    return az.from_dict(posterior=p, coords={"g__factor_dim": ["a", "b", "c"]}, dims={"1|g": ["g__factor_dim"]})


def test_fitted_row_predictions_by_hand():
    m = ab.beta("y ~ x + (1 | g)", data(), priors(True))
    post = draws(True)
    u = np.array([0.1, -0.1, 0.3])[pd.Categorical(data().g).codes]
    eta = np.broadcast_to(0.3 - 0.2 * data().x.to_numpy() + u, (2, 4, 6))
    lin = m.predict(post, kind="linear")
    np.testing.assert_allclose(lin.eta, eta)
    assert lin.attrs["prediction_scale"] == "linear" and lin.attrs["link"] == "logit"
    out = m.predict(post, kind="response", seed=9)
    np.testing.assert_allclose(out.mu, expit(eta))
    np.testing.assert_allclose(out.y, np.random.default_rng(9).beta(expit(eta) * 2.5, expit(-eta) * 2.5))
    assert out.y.dtype.kind == "f" and ((out.y > 0) & (out.y < 1)).all() and set(out.data_vars) == {"mu", "y"}
    nogroup = m.predict(post, kind="mean", include_group_effects=False)
    np.testing.assert_allclose(nogroup.mu, np.broadcast_to(expit(0.3 - 0.2 * data().x.to_numpy()), (2, 4, 6)))
    with pytest.raises(ValueError, match="kind"):
        m.predict(post, kind="probability")


@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_seeded_new_group_predictions(policy):
    m = ab.beta("y ~ x + (1 | g)", data(), priors(True))
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["new", "new"]})
    rng = np.random.default_rng(8)
    effect = rng.normal(size=(2, 4)) * 0.5 if policy == "population" else np.take([0.1, -0.1, 0.3], rng.integers(0, 3, size=(2, 4)))
    eta = 0.3 - 0.2 * new.x.to_numpy() + effect[..., None]
    out = m.predict(draws(True), newdata=new, kind="response", new_groups=policy, seed=8)
    np.testing.assert_allclose(out.mu, expit(eta))
    np.testing.assert_allclose(out.y, rng.beta(expit(eta) * 2.5, expit(-eta) * 2.5))
    assert out.attrs["new_groups"] == policy and list(out.attrs["new_levels"]) == ["new"]
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(draws(True), newdata=new, kind="mean")


def test_prediction_limits_and_dispersion():
    m = ab.beta("y ~ x", data(), priors())
    far = m.predict(draws(a=1000.0), kind="mean")                     # expit saturates at 1.0 in float64, as brms's epred does
    assert (far.mu == 1.0).all()
    with pytest.raises(ValueError, match="underflowed"):             # (1 - mu) phi is exactly zero there: no Beta draw exists
        m.predict(draws(a=1000.0), kind="response", seed=2)
    big = m.predict(draws(a=30.0), kind="response", seed=2)         # (1 - mu) phi = 2.3e-13 > 0: draws exist, at the edge
    assert np.isfinite(big.y).all() and (big.y <= 1).all() and (big.y > 0.999).all()
    post = {"Intercept": np.full((1, 4000), logit(0.3)), "x": np.zeros((1, 4000)), "phi": np.full((1, 4000), 5.0)}
    y = m.predict(az.from_dict(posterior=post), kind="response", seed=1).y.to_numpy()[0, :, 0]
    assert abs(y.mean() - 0.3) < 0.015 and 0.7 * 0.3 * 0.7 / 6 < y.var() < 1.3 * 0.3 * 0.7 / 6      # Var = mu (1 - mu) / (1 + phi)


# -- describe, prior predictive, record -----------------------------------------------------------------------
def test_describe_and_registry_constructor():
    m = ab.beta("y ~ x + (1 | g)", data(), priors(True))
    text = m.describe()
    assert text.startswith("AutoBayes Beta model") and "family    : beta, logit link" in text
    assert "phi ~ Gamma(alpha=2, beta=0.5)" in text and "y_i ~ Beta(mu_i phi, (1 - mu_i) phi)" in text
    assert "Var[y_i] = mu_i (1 - mu_i) / (1 + phi)" in text and "phi_log__" in text and "log(phi)" in text
    assert "1 - mu computed as expit(-eta)" in text and "mu = expit(logit_mu), the expected proportion" in text
    assert "a_c is the logit of the expected proportion at the predictor means" in text
    assert "location coefficients on the logit scale" in text and "sigma ~" not in text and "shape ~" not in text
    same = ab.Model("y ~ x + (1 | g)", data(), priors(True), family="beta")
    assert same.describe() == text and same.target_identity == m.target_identity
    other = ab.beta("y ~ x + (1 | g)", data(), priors(True, phi=ab.Gamma(0.01, 0.01)))
    assert other.target_identity != m.target_identity                 # the phi prior is part of the frozen target
    cat = ab.beta("y ~ x + z", data().assign(z=["p", "q", "p", "q", "p", "q"]), priors(), categorical={"z": "p"})
    assert "expit(a_c) is the expected proportion at that proportion-weighted predictor" in cat.describe()


def test_prior_predictive_shapes():
    m = ab.beta("y ~ x + (1 | g)", data(), priors(True))
    out = m.prior_predictive(draws=6, seed=3)
    assert out.prior["phi"].shape == (1, 6) and (out.prior["phi"].values > 0).all()
    y = out.prior_predictive["y"].values
    assert y.shape == (1, 6, 6) and ((y >= 0) & (y <= 1)).all() and y.dtype.kind == "f"   # extreme prior draws saturate at 0 or 1 in float64
    assert out.attrs["autobayes_generation_kind"] == "prior_predictive"


def test_fit_record_and_summary_rows(monkeypatch):
    m = ab.beta("y ~ x + (1 | g)", data(), priors(True))
    post = draws(True)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "1|g_sigma", "phi"}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "beta" and rec["model"]["link"] == "logit"
    assert rec["model"]["priors"]["phi"] == "Gamma(alpha=2, beta=0.5)" and "sigma" not in rec["model"]["priors"] and "shape" not in rec["model"]["priors"]
    assert "logit scale" in rec["model"]["intercept_prior_applies_to"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
