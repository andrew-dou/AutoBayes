"""Every prior class the API accepts, checked end to end: build, log density against the
independent reference, and a few draws. Normal/HalfNormal are covered elsewhere; these are
the ones a wrong parameter name in ``Priors.to_bambi`` would only reveal at build time."""
import numpy as np
import pytest

import autobayes as ab
from tests.conftest import simulate
from tests.reference_logp import reference_logp, reference_logp_bernoulli
from tests.test_bernoulli import simulate_binary

ALT_GAUSSIAN = ab.Priors(intercept=ab.StudentT(3, 0, 2.5), b=ab.StudentT(5, 0, 1), sd=ab.HalfStudentT(3, 1),
                         sigma=ab.Exponential(2))
ALT_GAUSSIAN_DICT = {"intercept": ("StudentT", {"nu": 3.0, "mu": 0.0, "sigma": 2.5}),
                     "b": {"x": ("StudentT", {"nu": 5.0, "mu": 0.0, "sigma": 1.0})},
                     "sd": ("HalfStudentT", {"nu": 3.0, "sigma": 1.0}), "sigma": ("Exponential", {"lam": 2.0})}
ALT_BERNOULLI = ab.Priors(intercept=ab.StudentT(4, 0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfCauchy(1))
ALT_BERNOULLI_DICT = {"intercept": ("StudentT", {"nu": 4.0, "mu": 0.0, "sigma": 1.5}),
                      "b": {"x": ("Normal", {"mu": 0.0, "sigma": 1.0})}, "sd": ("HalfCauchy", {"beta": 1.0})}


def test_gaussian_alternative_prior_families():
    df = simulate()
    m = ab.gaussian("y ~ x + (1 | g)", df, ALT_GAUSSIAN)
    text = m.describe()
    assert "StudentT(nu=3, mu=0, sigma=2.5)" in text and "HalfStudentT(nu=3, sigma=1)" in text and "Exponential(lam=2)" in text
    logp = m.pymc_model.compile_logp()
    rng = np.random.default_rng(1)
    for _ in range(4):
        nat = dict(intercept=rng.normal(1, 1), b=rng.normal(0, 1, 1), sigma=np.exp(rng.normal(-1, 0.4)),
                   sd=np.exp(rng.normal(-0.5, 0.4)), z=rng.normal(size=m.design.J))
        ref = reference_logp({"a_c": nat["intercept"], "b": nat["b"], "log_sigma": np.log(nat["sigma"]),
                              "log_sd": np.log(nat["sd"]), "z": nat["z"]}, df, response="y", predictors=["x"],
                             group="g", priors=ALT_GAUSSIAN_DICT, center=True, jacobian=True)
        assert abs(float(logp(m.value_point(**nat))) - ref) < 1e-8
    fit = m.fit(backend="pymc", chains=2, draws=20, tune=20, seed=1, cores=1)
    assert set(fit.idata.posterior.data_vars) >= {"Intercept", "x", "1|g_sigma", "sigma"}
    rec = ab.fit_record(m, fit)
    assert rec["model"]["priors"]["sigma"] == "Exponential(lam=2)"


def test_bernoulli_alternative_prior_families():
    df = simulate_binary()
    m = ab.bernoulli("y ~ x + (1 | g)", df, ALT_BERNOULLI)
    assert "HalfCauchy(beta=1)" in m.describe()
    logp = m.pymc_model.compile_logp()
    rng = np.random.default_rng(2)
    for _ in range(4):
        nat = dict(intercept=rng.normal(0, 1), b=rng.normal(0, 1, 1), sd=np.exp(rng.normal(-0.5, 0.4)),
                   z=rng.normal(size=m.design.J))
        ref = reference_logp_bernoulli({"a_c": nat["intercept"], "b": nat["b"], "log_sd": np.log(nat["sd"]),
                                        "z": nat["z"]}, df, response="y", predictors=["x"], group="g",
                                       priors=ALT_BERNOULLI_DICT, center=True, jacobian=True)
        assert abs(float(logp(m.value_point(**nat))) - ref) < 1e-8
    fit = m.fit(backend="pymc", chains=2, draws=20, tune=20, seed=1, cores=1)
    assert "1|g_sigma" in fit.idata.posterior


@pytest.mark.parametrize("bad", [
    lambda: ab.StudentT(0, 0, 1), lambda: ab.HalfStudentT(3, -1), lambda: ab.Exponential(0), lambda: ab.HalfCauchy(-2),
])
def test_prior_parameters_must_be_positive(bad):
    with pytest.raises(ab.PriorError, match="positive"):
        bad()
