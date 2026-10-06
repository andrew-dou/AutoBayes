"""Gamma responses with a log link (brms-parity A09): response validation, the shape prior and ``value_point``, the
density and gradient by hand on both paths, by-hand predictions, ``describe()``, prior-predictive shapes and the
fit record. No MCMC; the brms comparison is the oracle case ``gamma_varint``."""
from types import SimpleNamespace
import json
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist, halfnorm, norm

import autobayes as ab


def gamma_logpdf(y, eta, shape):
    """brms's ``gamma_lpdf(y | shape, shape / mu)`` with ``mu = exp(eta)``, written in ``eta``."""
    return shape * (np.log(shape) - eta) - gammaln(shape) + (shape - 1) * np.log(y) - shape * y * np.exp(-eta)


def data():
    return pd.DataFrame({"x": [-1.0, 0.5, 2.0, 3.0, 0.2, -0.7], "y": [0.4, 1.3, 2.2, 7.5, 0.9, 0.2],
                         "g": ["a", "a", "b", "b", "c", "c"]})


def priors(group=False, shape=ab.Gamma(2, 0.5), **extra):
    return ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=shape,
                     **({"sd": ab.HalfNormal(1)} if group else {}), **extra)


# -- response validation --------------------------------------------------------------------------------------
@pytest.mark.parametrize("values, message", [
    ([0.0, 1.0], "non-positive value"), ([-0.5, 1.0], "non-positive value"), ([np.inf, 1.0], "non-finite value"),
    ([True, False], "numeric response"), (["1", "2"], "numeric response"), ([1 + 1j, 1.0], "complex-valued"),
    ([np.nan, 1.0], "missing values"),
])
def test_invalid_responses(values, message):
    with pytest.raises(ab.FormulaError, match=message):
        ab.gamma("y ~ 1", pd.DataFrame({"y": values}), ab.Priors(intercept=ab.Normal(0, 2), b={}, shape=ab.Gamma(2, 0.5)))


def test_positive_responses_are_kept_as_given():
    m = ab.gamma("y ~ x", data(), priors())
    assert m.design.n == 6 and np.array_equal(m.design.y, data().y) and m.design.y.dtype.kind == "f"
    assert m.warnings == [] and not m.dropped_rows
    tiny = ab.gamma("y ~ x", data().assign(y=[1e-8, 1.3, 2.2, 7.5, 0.9, 1e8]), priors())
    assert tiny.design.y[0] == 1e-8 and tiny.design.y[-1] == 1e8


# -- the shape prior ------------------------------------------------------------------------------------------
def test_prior_coverage():
    with pytest.raises(ab.PriorError, match="the Gamma family needs a prior on the shape parameter: pass shape="):
        ab.gamma("y ~ x", data(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="the gamma family has no residual SD; do not pass sigma. Its auxiliary parameter is shape"):
        ab.gamma("y ~ x", data(), priors(sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="prior for 'shape' must be a positive-support prior"):
        ab.gamma("y ~ x", data(), priors(shape=ab.StudentT(3, 0, 1)))
    with pytest.raises(ab.PriorError, match="prior for 'shape' must be a Prior"):
        ab.gamma("y ~ x", data(), priors(shape="Gamma"))
    with pytest.raises(ab.PriorError, match="the student family has no shape parameter; do not pass shape. Its auxiliary parameter is sigma"):
        ab.student("y ~ x", data(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), shape=ab.Gamma(2, 0.5)), nu=4)
    with pytest.raises(ValueError, match="nu"):
        ab.Model("y ~ x", data(), priors(), family="gamma", nu=4)
    with pytest.raises(ab.FormulaError, match="binomial"):
        ab.gamma("y | trials(n) ~ x", data().assign(n=10), priors())
    with pytest.raises(ab.FormulaError, match="collision"):          # 'alpha' is Bambi's name for the shape variable
        ab.gamma("y ~ alpha", data().rename(columns={"x": "alpha"}), priors())
    for prior in (ab.HalfNormal(1), ab.Exponential(1), ab.HalfCauchy(1), ab.HalfStudentT(3, 1)):      # any positive-support class
        assert ab.gamma("y ~ x", data(), priors(shape=prior)).priors.shape is prior


# -- names and value_point ------------------------------------------------------------------------------------
def test_names_and_value_point():
    m = ab.gamma("y ~ x + (1 | g)", data(), priors(True))
    assert m.names["shape"] == "alpha" and "sigma" not in m.names and m.names["mean"] == "mu" and m.names["mu"] == "mu"
    assert m.link == "log" and m.family_spec.continuous
    assert "alpha_log__" in {v.name for v in m.pymc_model.value_vars}
    p = m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), shape=0.7)
    assert set(p) == {v.name for v in m.pymc_model.value_vars}
    np.testing.assert_allclose(p["alpha_log__"], np.log(0.7))
    with pytest.raises(ValueError, match="shape is required for the Gamma family; sigma does not apply"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), sigma=0.7)
    with pytest.raises(ValueError, match="shape is required"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3))
    with pytest.raises(ValueError, match="shape must be a finite positive scalar"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), shape=-1.0)


# -- density and gradient -------------------------------------------------------------------------------------
def test_hierarchical_density_by_hand_both_paths():
    df = data()
    m = ab.gamma("y ~ x + (1 | g)", df, priors(True))
    nat = dict(intercept=0.3, b=[-0.2], sd=0.5, z=np.array([0.2, -0.4, 0.1]), shape=0.7)
    p = m.value_point(**nat)
    x = df.x.to_numpy()
    eta = 0.3 - 0.2 * (x - x.mean()) + 0.5 * nat["z"][pd.Categorical(df.g).codes]
    lik = gamma_logpdf(df.y.to_numpy(), eta, 0.7).sum()
    np.testing.assert_allclose(lik, gamma_dist.logpdf(df.y.to_numpy(), 0.7, scale=np.exp(eta) / 0.7).sum())   # scipy agrees here
    prior = (norm.logpdf(0.3, 0, 2) + norm.logpdf(-0.2, 0, 1) + halfnorm.logpdf(0.5, scale=1) + norm.logpdf(nat["z"]).sum()
             + gamma_dist.logpdf(0.7, 2, scale=1 / 0.5))
    jac = np.log(0.5) + np.log(0.7)                              # sd and shape are sampled on the log scale
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), lik + prior + jac, rtol=1e-10)
    np.testing.assert_allclose(m.pymc_model.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-10)
    np.testing.assert_allclose(m.pymc_model.compile_logp(vars=[m.pymc_model["alpha"]], jacobian=False)(p),
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
    # d/d log(shape) by hand: sum_i [log(shape) + 1 - eta_i - digamma(shape) + log(y_i) - y_i / mu_i] * shape + prior + 1
    from scipy.special import digamma
    y = df.y.to_numpy()
    d_shape = (np.log(0.7) + 1 - eta - digamma(0.7) + np.log(y) - y * np.exp(-eta)).sum() + (2 - 1) / 0.7 - 0.5
    np.testing.assert_allclose(g_pt[names.index("alpha_log__")], d_shape * 0.7 + 1.0, rtol=1e-8)
    assert m.design.centered and m.link == "log"


@pytest.mark.parametrize("eta", [-30.0, 0.0, 30.0])
@pytest.mark.parametrize("shape", [0.05, 1.0, 1e4])
def test_response_boundaries_density_both_paths(eta, shape):
    y = np.array([1e-6, 1.0, 1e6])
    m = ab.gamma("y ~ 1", pd.DataFrame({"y": y}), ab.Priors(intercept=ab.Normal(0, 2), b={}, shape=ab.Gamma(2, 0.5)))
    p = m.value_point(intercept=eta, b=[], shape=shape)
    ref = gamma_logpdf(y, eta, shape).sum() + norm.logpdf(eta, 0, 2) + gamma_dist.logpdf(shape, 2, scale=2.0) + np.log(shape)
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), ref, rtol=1e-9)
    from pymc.sampling.jax import get_jaxified_logp
    jf = get_jaxified_logp(m.pymc_model)
    np.testing.assert_allclose(jf([p[v.name] for v in m.pymc_model.value_vars]), ref, rtol=1e-9)


# -- predictions ----------------------------------------------------------------------------------------------
def draws(group=False, a=0.3, shape=2.5):
    p = {"Intercept": np.full((2, 4), a), "x": np.full((2, 4), -0.2), "alpha": np.full((2, 4), shape)}
    if group:
        p.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.1, -0.1, 0.3], (2, 4, 3))})
    return az.from_dict(posterior=p, coords={"g__factor_dim": ["a", "b", "c"]}, dims={"1|g": ["g__factor_dim"]})


def test_fitted_row_predictions_by_hand():
    m = ab.gamma("y ~ x + (1 | g)", data(), priors(True))
    post = draws(True)
    u = np.array([0.1, -0.1, 0.3])[pd.Categorical(data().g).codes]
    eta = 0.3 - 0.2 * data().x.to_numpy() + u
    expected = np.broadcast_to(np.exp(eta), (2, 4, 6))
    lin = m.predict(post, kind="linear")
    np.testing.assert_allclose(lin.eta, np.broadcast_to(eta, (2, 4, 6)))
    assert lin.attrs["prediction_scale"] == "linear" and lin.attrs["link"] == "log"
    out = m.predict(post, kind="response", seed=9)
    np.testing.assert_allclose(out.mu, expected)
    np.testing.assert_allclose(out.y, np.random.default_rng(9).gamma(2.5, expected / 2.5))
    assert out.y.dtype.kind == "f" and (out.y > 0).all() and set(out.data_vars) == {"mu", "y"}
    nogroup = m.predict(post, kind="mean", include_group_effects=False)
    np.testing.assert_allclose(nogroup.mu, np.broadcast_to(np.exp(0.3 - 0.2 * data().x.to_numpy()), (2, 4, 6)))
    with pytest.raises(ValueError, match="kind"):
        m.predict(post, kind="probability")


@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_seeded_new_group_predictions(policy):
    m = ab.gamma("y ~ x + (1 | g)", data(), priors(True))
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["new", "new"]})
    rng = np.random.default_rng(8)
    effect = rng.normal(size=(2, 4)) * 0.5 if policy == "population" else np.take([0.1, -0.1, 0.3], rng.integers(0, 3, size=(2, 4)))
    expected = np.exp(0.3 - 0.2 * new.x.to_numpy() + effect[..., None])
    out = m.predict(draws(True), newdata=new, kind="response", new_groups=policy, seed=8)
    np.testing.assert_allclose(out.mu, expected)
    np.testing.assert_allclose(out.y, rng.gamma(2.5, expected / 2.5))
    assert out.attrs["new_groups"] == policy and list(out.attrs["new_levels"]) == ["new"]
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(draws(True), newdata=new, kind="mean")


def test_prediction_limits_and_dispersion():
    m = ab.gamma("y ~ x", data(), priors())
    with pytest.raises(ValueError, match="overflow"):
        m.predict(draws(a=1000.0), kind="mean")
    big = m.predict(draws(a=50.0), kind="response", seed=2)         # continuous: no integer-generator limit
    assert np.isfinite(big.mu).all() and np.isfinite(big.y).all()
    post = {"Intercept": np.full((1, 4000), np.log(5.0)), "x": np.zeros((1, 4000)), "alpha": np.full((1, 4000), 0.5)}
    y = m.predict(az.from_dict(posterior=post), kind="response", seed=1).y.to_numpy()[0, :, 0]
    assert abs(y.mean() - 5.0) < 0.5 and 0.7 * 5.0 ** 2 / 0.5 < y.var() < 1.3 * 5.0 ** 2 / 0.5    # Var = mu^2 / shape


# -- describe, prior predictive, record -----------------------------------------------------------------------
def test_describe_and_registry_constructor():
    m = ab.gamma("y ~ x + (1 | g)", data(), priors(True))
    text = m.describe()
    assert text.startswith("AutoBayes Gamma model") and "family    : gamma, log link" in text
    assert "shape ~ Gamma(alpha=2, beta=0.5)" in text and "y_i ~ Gamma(shape, rate = shape / mu_i)" in text
    assert "Var[y_i] = mu_i^2 / shape" in text and "alpha_log__" in text and "log(shape)" in text
    assert "brms's default Gamma link is inverse" in text and "mu = exp(eta), expected response" in text
    assert "location coefficients on the log scale" in text and "sigma ~" not in text and "log(sigma)" not in text
    same = ab.Model("y ~ x + (1 | g)", data(), priors(True), family="gamma")
    assert same.describe() == text and same.target_identity == m.target_identity
    other = ab.gamma("y ~ x + (1 | g)", data(), priors(True, shape=ab.Gamma(0.01, 0.01)))
    assert other.target_identity != m.target_identity                 # the shape prior is part of the frozen target


def test_prior_predictive_shapes():
    m = ab.gamma("y ~ x + (1 | g)", data(), priors(True))
    out = m.prior_predictive(draws=6, seed=3)
    assert out.prior["alpha"].shape == (1, 6) and (out.prior["alpha"].values > 0).all()
    y = out.prior_predictive["y"].values
    assert y.shape == (1, 6, 6) and (y > 0).all() and y.dtype.kind == "f"
    assert out.attrs["autobayes_generation_kind"] == "prior_predictive"


def test_fit_record_and_summary_rows(monkeypatch):
    m = ab.gamma("y ~ x + (1 | g)", data(), priors(True))
    post = draws(True)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "1|g_sigma", "alpha"}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "gamma" and rec["model"]["link"] == "log"
    assert rec["model"]["priors"]["shape"] == "Gamma(alpha=2, beta=0.5)" and "sigma" not in rec["model"]["priors"]
    assert "log scale" in rec["model"]["intercept_prior_applies_to"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
