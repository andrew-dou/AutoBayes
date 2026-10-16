"""brms-parity A13: the Student-t degrees of freedom learned under their own prior (docs/student-learned-nu.md).

brms declares ``real<lower=1> nu``, adds ``gamma_lpdf(nu | 2, 0.1) - gamma_lccdf(1 | 2, 0.1)`` to ``lprior`` and samples
``log(nu - 1)``. Here: the refusals; the density against a closed-form reference that includes the truncation constant,
with and without the Jacobian; every accepted prior family against its own closed-form truncation constant; the
gradient of the ``log(nu - 1)`` coordinate on both paths against the analytic derivative; ``value_point``; predictions
with each draw's own nu; ``describe()``, the same under two hash seeds; the target identity and the fit record; the prior
predictive. Nothing samples.
"""
import json
import os
import subprocess
import sys
from dataclasses import replace
from math import factorial
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma, erfc, gammaln
from scipy.stats import halfnorm, norm
from scipy.stats import t as student_t

import autobayes as ab

REPO = Path(__file__).resolve().parent.parent
FORMULA = "y ~ x + (1 | g)"
PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
                   nu=ab.Gamma(2, 0.1))
FIXED = replace(PRIORS, nu=None)


def frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = np.array([f"l{i % 5}" for i in range(n)], dtype=object)
    rng.shuffle(g)
    u = dict(zip(sorted(set(g)), rng.normal(0, 0.5, 5)))
    y = 0.4 + 0.8 * x + np.array([u[v] for v in g]) + 0.7 * rng.standard_t(4, size=n)
    return pd.DataFrame({"x": x, "g": g, "y": y})


@pytest.fixture(scope="module")
def df():
    return frame()


@pytest.fixture(scope="module")
def model(df):
    return ab.student(FORMULA, df, PRIORS)


# -- references written from the mathematics ------------------------------------------------------------------
def log_prior_nu(prior, nu):
    """The untruncated prior log density of nu, in closed form."""
    p = prior.params
    if prior.name == "Gamma":
        a, b = p["alpha"], p["beta"]
        return a * np.log(b) - gammaln(a) + (a - 1) * np.log(nu) - b * nu
    if prior.name == "Exponential":
        return np.log(p["lam"]) - p["lam"] * nu
    if prior.name == "HalfNormal":
        return halfnorm.logpdf(nu, scale=p["sigma"])
    return np.log(2 / (np.pi * p["beta"])) - np.log1p((nu / p["beta"]) ** 2)                  # HalfCauchy


def log_truncation(prior):
    """log P(nu > 1) under the prior, in closed form: e^-b sum_{j < a} b^j / j! for an integer Gamma shape a (rate b),
    e^-lam, erfc(1 / (s sqrt 2)) for HalfNormal(s), 1 - 2 atan(1 / s) / pi for HalfCauchy(s)."""
    p = prior.params
    if prior.name == "Gamma":
        a, b = int(p["alpha"]), p["beta"]
        assert a == p["alpha"]
        return -b + np.log(sum(b ** j / factorial(j) for j in range(a)))
    if prior.name == "Exponential":
        return -p["lam"]
    if prior.name == "HalfNormal":
        return np.log(erfc(1 / (p["sigma"] * np.sqrt(2))))
    return np.log1p(-2 / np.pi * np.arctan(1 / p["beta"]))


def mu_of(model, nat):
    d = model.design
    return nat["intercept"] + d.X_model @ nat["b"] + nat["sd"] * nat["z"][d.group_index]


def reference(model, nat, *, jacobian=True):
    """The target in PyMC's value space: the Student-t likelihood, the priors of PRIORS with nu's truncated to nu > 1 and
    renormalised there, z ~ Normal(0, 1), and log(sd) + log(sigma) + log(nu - 1) with the Jacobian."""
    nu, sigma, sd = nat["nu"], nat["sigma"], nat["sd"]
    lik = student_t.logpdf(model.design.y, df=nu, loc=mu_of(model, nat), scale=sigma).sum()
    prior = (norm.logpdf(nat["intercept"], 0, 5) + norm.logpdf(nat["b"], 0, 2).sum() + halfnorm.logpdf(sd)
             + halfnorm.logpdf(sigma) + norm.logpdf(nat["z"]).sum()
             + log_prior_nu(model.priors.nu, nu) - log_truncation(model.priors.nu))
    return lik + prior + ((np.log(sd) + np.log(sigma) + np.log(nu - 1)) if jacobian else 0.0)


def d_log_nu_minus_one(model, nat):
    """d log p / d log(nu - 1) = (nu - 1) (d/dnu of the likelihood + (2 - 1) / nu - 0.1) + 1 (the Jacobian), with, per
    row, d/dnu log t(y | nu, mu, sigma) = (psi((nu + 1) / 2) - psi(nu / 2)) / 2 - 1 / (2 nu) - log(1 + z^2 / nu) / 2
    + (nu + 1) z^2 / (2 nu (nu + z^2)) and z = (y - mu) / sigma; Gamma(2, 0.1)'s truncation constant does not move."""
    nu = nat["nu"]
    z = (model.design.y - mu_of(model, nat)) / nat["sigma"]
    dlik = np.sum(0.5 * (digamma((nu + 1) / 2) - digamma(nu / 2)) - 1 / (2 * nu) - 0.5 * np.log1p(z * z / nu)
                  + (nu + 1) * z * z / (2 * nu * (nu + z * z)))
    return (nu - 1) * (dlik + 1 / nu - 0.1) + 1.0


def points(model, k=4, seed=3):
    rng = np.random.default_rng(seed)
    return [dict(intercept=rng.normal(0.3, 0.5), b=rng.normal(0.7, 0.3, 1), sd=np.exp(rng.normal(-0.7, 0.3)),
                 z=rng.normal(size=model.design.J), sigma=np.exp(rng.normal(-0.3, 0.2)), nu=1.0 + np.exp(rng.normal(1.0, 1.0)))
            for _ in range(k)]


# -- refusals and names ---------------------------------------------------------------------------------------
def test_refusals(df):
    with pytest.raises(ValueError, match=r"nu=4 fixes the degrees of freedom and Priors\(nu=Gamma\(alpha=2, beta=0.1\)\) learns them"):
        ab.student(FORMULA, df, PRIORS, nu=4)
    with pytest.raises(ab.PriorError, match=r"the student family needs nu: a fixed nu > 1 \(nu=4\), or nu=None with a "
                                            r"positive-support prior Priors\(nu=\.\.\.\) to learn it"):
        ab.student(FORMULA, df, FIXED)
    with pytest.raises(ab.PriorError, match=r"prior for 'nu' must be a positive-support prior \['Exponential', 'Gamma', "
                                            r"'HalfCauchy', 'HalfNormal'\]; got Normal\(mu=0, sigma=10\)"):
        ab.student(FORMULA, df, replace(PRIORS, nu=ab.Normal(0, 10)))
    with pytest.raises(ab.PriorError, match=r"prior for 'nu' cannot be HalfStudentT: PyMC has no CDF for it"):
        ab.student(FORMULA, df, replace(PRIORS, nu=ab.HalfStudentT(3, 10)))
    with pytest.raises(ab.PriorError, match="prior for 'nu' must be a Prior, got float"):
        ab.student(FORMULA, df, replace(PRIORS, nu=4.0))
    with pytest.raises(ab.PriorError, match="the gaussian family has no degrees of freedom; do not pass nu"):
        ab.gaussian(FORMULA, df, PRIORS)
    counts = df.assign(k=np.arange(len(df)) % 4)
    with pytest.raises(ab.PriorError, match="the poisson family has no degrees of freedom; do not pass nu"):
        ab.poisson("k ~ x", counts, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), nu=ab.Gamma(2, 0.1)))
    with pytest.raises(ValueError, match="nu applies only to the student family"):
        ab.Model(FORMULA, df, FIXED, family="gaussian", nu=4)
    fixed = ab.student(FORMULA, df, FIXED, nu=4)
    with pytest.raises(ValueError, match=r"nu is fixed at 4 in this model, not sampled; omit it"):
        fixed.value_point(intercept=0.0, b=[0.0], sd=0.5, z=np.zeros(fixed.design.J), sigma=1.0, nu=4.0)
    gauss = ab.gaussian(FORMULA, df, FIXED)
    with pytest.raises(ValueError, match="the gaussian family has no nu; omit it"):
        gauss.value_point(intercept=0.0, b=[0.0], sd=0.5, z=np.zeros(gauss.design.J), sigma=1.0, nu=4.0)


def test_value_point_and_names(df, model):
    point = model.value_point(intercept=0.1, b=[0.5], sd=0.4, z=np.zeros(model.design.J), sigma=0.8, nu=3.5)
    assert set(point) == {v.name for v in model.pymc_model.value_vars} == {
        "Intercept", "x", "1|g_sigma_log__", "1|g_offset", "sigma_log__", "nu_interval__"}
    assert point["nu_interval__"] == np.log(2.5)
    assert model.names["nu"] == "nu" and model.nu is None and model.learned_parameters == {"nu": 1.0}
    assert {v.name for v in model.pymc_model.free_RVs} == {"Intercept", "x", "1|g_sigma", "1|g_offset", "sigma", "nu"}
    for bad in (None, 1.0, 0.5, np.inf, np.nan, [3.0]):
        with pytest.raises(ValueError, match="nu"):
            model.value_point(intercept=0.1, b=[0.5], sd=0.4, z=np.zeros(model.design.J), sigma=0.8, nu=bad)
    fixed = ab.student(FORMULA, df, FIXED, nu=4)
    assert "nu" not in fixed.names and fixed.learned_parameters == {} and fixed.nu == 4.0
    assert ab.Model(FORMULA, df, PRIORS, family="student").target_identity == model.target_identity


# -- the density and its gradient --------------------------------------------------------------------------------
def test_density_matches_the_closed_form_reference_with_the_truncation_constant(model):
    pmm = model.pymc_model
    logp, logp_nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    nu_prior = pmm.compile_logp(vars=[pmm["nu"]], jacobian=False)
    for nat in points(model):
        point = model.value_point(**nat)
        assert abs(float(logp(point)) - reference(model, nat)) < 1e-9
        assert abs(float(logp_nojac(point)) - reference(model, nat, jacobian=False)) < 1e-9
        # Gamma(2, 0.1) at nu minus log P(nu > 1) = log(e^-0.1 * 1.1), the constant brms subtracts (gamma_lccdf(1 | 2, 0.1))
        want = 2 * np.log(0.1) + np.log(nat["nu"]) - 0.1 * nat["nu"] - (np.log(1.1) - 0.1)
        assert abs(float(nu_prior(point)) - want) < 1e-12


@pytest.mark.parametrize("prior", [ab.Gamma(3, 0.5), ab.Exponential(0.2), ab.HalfNormal(5), ab.HalfCauchy(4)])
def test_every_accepted_prior_is_truncated_and_renormalised(df, prior):
    m = ab.student(FORMULA, df, replace(PRIORS, nu=prior))
    nu_prior = m.pymc_model.compile_logp(vars=[m.pymc_model["nu"]], jacobian=False)
    # PyMC's HalfCauchy log density adds pt.log(2) and its HalfNormal log CDF divides by pt.sqrt(2.0), both evaluated in
    # float32 because PyTensor narrows exactly representable Python constants: errors of about 2e-9 and 3e-9 here
    tol = 1e-8 if prior.name in ("HalfNormal", "HalfCauchy") else 1e-12
    for nu in (1.01, 2.3, 7.0, 40.0):
        point = m.value_point(intercept=0.0, b=[0.0], sd=0.5, z=np.zeros(m.design.J), sigma=1.0, nu=nu)
        assert abs(float(nu_prior(point)) - (log_prior_nu(prior, nu) - log_truncation(prior))) < tol, (prior, nu)


def test_gradient_of_the_log_nu_minus_one_coordinate_on_both_paths(model):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp

    pmm = model.pymc_model
    names = [v.name for v in pmm.value_vars]
    dlogp = pmm.compile_dlogp()
    jf = get_jaxified_logp(pmm)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    base = points(model, k=1)[0]
    for nat in points(model) + [dict(base, nu=1.0 + 1e-6), dict(base, nu=300.0)]:
        point = model.value_point(**nat)
        at = sum(int(np.asarray(point[n]).size) for n in names[:names.index("nu_interval__")])
        want = d_log_nu_minus_one(model, nat)
        g_pt = float(np.asarray(dlogp(point), dtype=float)[at])
        g_jx = float(np.asarray(jgrad(*[jnp.asarray(point[n]) for n in names])[names.index("nu_interval__")]))
        tol = 1e-8 * max(1.0, abs(want))
        assert abs(g_pt - want) <= tol and abs(g_jx - want) <= tol, (nat["nu"], g_pt, g_jx, want)
        assert abs(float(jf([point[n] for n in names])) - reference(model, nat)) < 1e-8


# -- predictions -------------------------------------------------------------------------------------------------
def posterior(model, nu, draws=500, seed=1):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    post = {"Intercept": rng.normal(0.4, 0.1, shape), "x": rng.normal(0.8, 0.1, shape),
            "1|g_sigma": np.exp(rng.normal(-0.7, 0.2, shape)), "sigma": np.exp(rng.normal(-0.4, 0.1, shape)),
            "nu": np.broadcast_to(nu, shape).copy()}
    post["1|g"] = rng.normal(size=shape + (model.design.J,)) * post["1|g_sigma"][..., None]
    return az.from_dict(posterior=post, coords={"g__factor_dim": list(model.design.levels)}, dims={"1|g": ["g__factor_dim"]})


def test_predictions_draw_student_t_with_each_draws_nu(df, model):
    nu = np.where(np.arange(500) < 250, 1.2, 200.0)                    # heavy tails in the first half, near-Normal after
    idata = posterior(model, nu)
    post = idata.posterior
    a, b, u, sigma = (post[k].to_numpy() for k in ("Intercept", "x", "1|g", "sigma"))
    mu = a[..., None] + b[..., None] * df.x.to_numpy() + u[..., model.design.group_index]
    np.testing.assert_allclose(model.predict(idata, kind="mean")["mu"].to_numpy(), mu, atol=1e-12)
    np.testing.assert_allclose(model.predict(idata, kind="linear")["eta"].to_numpy(), mu, atol=1e-12)
    # by hand (population level only): mu + sigma * standard_t(that draw's nu), from the seeded generator
    nog = model.predict(idata, kind="response", include_group_effects=False, seed=7)
    mu0 = a[..., None] + b[..., None] * df.x.to_numpy()
    rng = np.random.default_rng(7)
    np.testing.assert_allclose(nog["y"].to_numpy(), mu0 + sigma[..., None] * rng.standard_t(nu[None, :, None], size=mu0.shape))
    # through Bambi (the fitted rows): each draw's nu sets its tails
    resp = model.predict(idata, kind="response", seed=3)["y"].to_numpy()
    scaled = np.abs(resp - mu) / sigma[..., None]
    assert resp.shape == mu.shape and np.isfinite(resp).all()
    assert np.mean(scaled[:, :250] > 10) > 0.01 and np.mean(scaled[:, 250:] > 10) == 0.0
    # new rows of an unseen level, by hand
    new = pd.DataFrame({"x": [0.0, 1.0], "g": ["NEW", "NEW"]})
    out = model.predict(idata, newdata=new, kind="response", new_groups="population", seed=2)
    assert out["y"].shape == (1, 500, 2) and np.isfinite(out["y"].to_numpy()).all()


# -- describe(), identity, records, prior predictive ---------------------------------------------------------------
def test_describe(df, model):
    text = model.describe()
    assert "family    : student, identity link" in text
    assert "  nu ~ Gamma(alpha=2, beta=0.1), truncated to nu > 1 and renormalised there (brms's real<lower=1> nu:" in text
    assert "nu is learned, sampled as log(nu - 1)" in text
    assert "Learned nu is not supported" not in text and "fixed, not sampled" not in text
    assert f"  {'nu_interval__':<24} log(nu - 1)" in text
    fixed = ab.student(FORMULA, df, FIXED, nu=4).describe()
    assert "  nu = 4 (fixed, not sampled)" in fixed and "nu_interval__" not in fixed and "Learned nu is not supported" in fixed
    assert text.index(f"  {'sigma_log__':<24} log(sigma)") < text.index(f"  {'nu_interval__':<24} log(nu - 1)")


def _fresh(seed: int, script: str) -> dict:
    """``script`` run by this interpreter in a fresh process under ``PYTHONHASHSEED=seed``: its last JSON line."""
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    return json.loads(lines[-1])


SET_ORDER = 'import json; print(json.dumps({"order": list(set(("mu", "sigma", "nu")) - {"mu"})}))'   # as Bambi's auxiliary_parameters
BUILD = """
import json
import autobayes as ab
from tests.test_student_learned_nu import FORMULA, PRIORS, frame
m = ab.student(FORMULA, frame(), PRIORS)
print(json.dumps({"describe": m.describe(), "identity": m.target_identity, "value_vars": [v.name for v in m.pymc_model.value_vars]}))
"""


def test_describe_does_not_depend_on_the_hash_seed():
    """Bambi creates sigma and nu in the iteration order of the set of their names (``Family.auxiliary_parameters``), so
    PyMC's order of ``sigma_log__`` and ``nu_interval__`` follows ``PYTHONHASHSEED``. Built under two hash seeds that order
    that set differently, the PyMC orders differ and ``describe()`` (sigma, then nu) and the target identity do not."""
    seeds = {}
    for seed in range(32):
        seeds.setdefault(tuple(_fresh(seed, SET_ORDER)["order"]), seed)
        if len(seeds) == 2:
            break
    assert len(seeds) == 2, "no two hash seeds order {'sigma', 'nu'} differently"
    first, second = (_fresh(seed, BUILD) for seed in seeds.values())
    assert first["value_vars"] != second["value_vars"] and sorted(first["value_vars"]) == sorted(second["value_vars"])
    assert first["describe"] == second["describe"] and first["identity"] == second["identity"]
    text = first["describe"]
    assert text.index("  sigma_log__ ") < text.index("  nu_interval__ ")


def test_target_identity_and_fit_record(df, model, monkeypatch):
    assert ab.student(FORMULA, df, PRIORS, nu=None).target_identity == model.target_identity
    assert ab.student(FORMULA, df, replace(PRIORS, nu=ab.Gamma(2, 0.2))).target_identity != model.target_identity
    fixed = ab.student(FORMULA, df, FIXED, nu=4)
    assert fixed.target_identity != model.target_identity
    import autobayes.session as session
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})

    def record(m):
        fit = SimpleNamespace(target_identity=m.target_identity, idata=None, backend="synthetic", device_requested="cpu",
                              device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=0, draws=0,
                              tune=0, target_accept=0.9, seed=1, wall_seconds=0, notes=["synthetic record; not a fit"])
        return json.loads(json.dumps(ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})))

    rec, frec = record(model), record(fixed)
    assert rec["model"]["priors"]["nu"] == "Gamma(alpha=2, beta=0.1)" and "family_parameters" not in rec["model"]
    assert frec["model"]["family_parameters"] == {"nu": 4.0} and "nu" not in frec["model"]["priors"]
    with pytest.raises(ValueError, match="different models"):
        ab.compare_records(rec, frec)


def test_summary_rows_include_nu(model):
    from autobayes.session import _summary_rows

    rng = np.random.default_rng(0)
    post = {k: rng.normal(size=(2, 50)) for k in ("Intercept", "x")}
    post.update({"1|g_sigma": np.exp(rng.normal(size=(2, 50))), "sigma": np.exp(rng.normal(size=(2, 50))),
                 "nu": 1 + np.exp(rng.normal(size=(2, 50)))})
    rows = _summary_rows(model, az.from_dict(posterior=post))
    assert set(rows) == {"Intercept", "x", "1|g_sigma", "sigma", "nu"}


def test_prior_predictive(model):
    pp = model.prior_predictive(draws=40, seed=5)
    assert (pp.prior["nu"].to_numpy() > 1).all() and np.isfinite(pp.prior_predictive["y"].to_numpy()).all()
