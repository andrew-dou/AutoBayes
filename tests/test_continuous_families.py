"""Weibull, exponential, skew-normal, shifted lognormal and ex-Gaussian responses (brms-parity D01,
docs/continuous-families.md): each family's density in brms's parameterisation against SciPy's own distribution (whose
moments confirm the parameterisation) on both paths, the skew normal's and ex-Gaussian's tails where brms's form is -inf,
the extra parameters alpha, beta and ndt (priors, refusals, value_point, ndt's prior truncated to (0, min y)), the term
model composing with the new likelihoods, by-hand predictions and response draws, prior-predictive draws, brms's default
priors, describe() under two hash seeds and the fit record. No MCMC; the brms comparison is the oracle cases
weibull_varint, exponential_varint, skew_normal_varint, shifted_lognormal_varint and exgaussian_varint."""
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import stats
from scipy.special import expit, gamma as gamma_fn

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import get_family
from autobayes.priors import BOUNDABLE, bounded_log_mass

REPO = Path(__file__).resolve().parent.parent
FAMILIES = ("weibull", "exponential", "skew_normal", "shifted_lognormal", "exgaussian")
RESPONSE = {"weibull": "ypos", "exponential": "ypos", "shifted_lognormal": "ypos", "skew_normal": "y", "exgaussian": "y"}
OWN = {"weibull": {"shape": ab.Gamma(2, 0.5)}, "exponential": {},
       "skew_normal": {"sigma": ab.HalfNormal(1), "alpha": ab.Normal(0, 4)},
       "shifted_lognormal": {"sigma": ab.HalfNormal(1), "ndt": ab.Normal(0, 0.3)},
       "exgaussian": {"sigma": ab.HalfNormal(1), "beta": ab.Gamma(1, 0.1)}}
VALUES = {"weibull": {"shape": 1.7}, "exponential": {}, "skew_normal": {"sigma": 0.8, "alpha": -2.5},
          "shifted_lognormal": {"sigma": 0.6, "ndt": 0.15}, "exgaussian": {"sigma": 0.7, "beta": 1.3}}


def frame(n=24, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "w": rng.uniform(-1, 1, n), "o": rng.normal(0, 0.2, n),
                         "z": np.array(["p", "q", "r"])[np.arange(n) % 3], "g": [f"l{k}" for k in np.arange(n) % 4],
                         "h": np.array(["u", "v"])[(np.arange(n) // 4) % 2],
                         "y": rng.normal(0.5, 1.5, n) + rng.exponential(1.0, n), "ypos": 0.3 + rng.gamma(2.0, 0.8, n)})


def build(family, formula=None, frame_=None, **extra):
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **{**OWN[family], **extra})
    return getattr(ab, family)(formula or f"{RESPONSE[family]} ~ x", frame() if frame_ is None else frame_, priors)


def scipy_distribution(family, mu, values):
    """SciPy's distribution of each row, from brms's parameterisation, derived without the library's formulas: the
    skew normal's location and scale from SciPy's own standard moments, the ex-Gaussian's from its definition."""
    v = values
    if family == "weibull":
        return stats.weibull_min(v["shape"], scale=mu / gamma_fn(1 + 1 / v["shape"]))
    if family == "exponential":
        return stats.expon(scale=mu)
    if family == "skew_normal":
        m0, s0 = stats.skewnorm.mean(v["alpha"]), stats.skewnorm.std(v["alpha"])
        omega = v["sigma"] / s0
        return stats.skewnorm(v["alpha"], loc=mu - omega * m0, scale=omega)
    if family == "shifted_lognormal":
        return stats.lognorm(v["sigma"], loc=v["ndt"], scale=np.exp(mu))
    return stats.exponnorm(v["beta"] / v["sigma"], loc=mu - v["beta"], scale=v["sigma"])


def mean_of(family, eta, values):
    """brms's posterior_epred of each family at the linear predictor ``eta``."""
    if family == "shifted_lognormal":
        return np.exp(eta + values["sigma"] ** 2 / 2) + values["ndt"]
    return np.exp(eta) if get_family(family).link == "log" else eta


def graph(m, point, names):
    pmm = m.pymc_model
    return [np.asarray(a, dtype=float) for a in
            pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n] for n in names]), inputs=pmm.value_vars, on_unused_input="ignore")(point)]


def rows_logp(m, point):
    pmm = m.pymc_model
    f = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, on_unused_input="ignore")
    return np.asarray(f(point)[0], dtype=float).ravel()


# -- the densities: SciPy, the parameterisation by moments, both paths -----------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_density_is_brms_parameterisation_on_both_paths(family):
    m = build(family)
    df, v = frame(), VALUES[family]
    p = m.value_point(intercept=0.4, b=[0.3], **v)
    x = df.x.to_numpy()
    eta = 0.4 + 0.3 * (x - x.mean())
    mu = np.exp(eta) if m.link == "log" else eta
    dist = scipy_distribution(family, mu, v)
    np.testing.assert_allclose(dist.mean(), mean_of(family, eta, v), rtol=1e-12)    # the parameterisation, by moments
    if family in ("skew_normal", "exgaussian"):
        sd = v["sigma"] if family == "skew_normal" else math.hypot(v["sigma"], v["beta"])
        np.testing.assert_allclose(dist.std(), sd, rtol=1e-12)
    y = df[RESPONSE[family]].to_numpy()
    np.testing.assert_allclose(rows_logp(m, p), dist.logpdf(y), rtol=1e-12)
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    pmm = m.pymc_model
    names = [w.name for w in pmm.value_vars]
    args = [p[n] for n in names]
    jf = get_jaxified_logp(pmm)
    np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(p)), rtol=1e-13)
    g_jax = np.concatenate([np.ravel(a) for a in jax.grad(lambda q: jf(q))(args)])
    # PyTensor's digamma (the gradient of the Weibull's gamma(1 + 1 / shape)) is off by about 1e-10 absolute (SciPy's
    # digamma at 1.59: 1.4e-10), so the shape coordinate agrees to 1e-9 there; brms's gradient tolerance (G2) is 1e-7
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float), rtol=1e-8 if family == "weibull" else 1e-10,
                               atol=1e-12)


def _rows(logp, mode, value, *params):
    """Per-row value and gradient with respect to the response and every parameter, on PyTensor's or JAX's path."""
    vs = [pt.dvector(f"a{k}") for k in range(1 + len(params))]
    rows = logp(*vs)
    if mode == "pytensor":
        f = pytensor.function(vs, [rows, *pt.grad(rows.sum(), vs)])
        return [np.asarray(a, dtype=float) for a in f(value, *params)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=vs, outputs=[rows])
    args = [np.asarray(a, dtype=float) for a in (value, *params)]
    grads = jax.grad(lambda *a: jf(*a)[0].sum(), argnums=tuple(range(len(args))))(*args)
    return [np.asarray(jf(*args)[0], dtype=float), *(np.asarray(g, dtype=float) for g in grads)]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_skew_normal_and_exgaussian_tails_stay_finite_where_brms_form_is_minus_inf(mode):
    """brms writes log Phi as log(erfc(-x / sqrt 2)) - log 2, which is -inf once erfc underflows (x below about -37.5): a
    skew-normal row far on the short side, an ex-Gaussian row far below its mean. Here both are SciPy's value (log_ndtr) with a
    finite gradient, and equal it in the bulk too."""
    y = np.array([-60.0, -9.0, -2.0, 0.0, 1.5, 12.0, 80.0])
    mu, sigma, alpha = np.zeros(7), np.full(7, 0.9), np.full(7, 6.0)
    value, *grads = _rows(lk.skew_normal_logp, mode, y, mu, sigma, alpha)
    xi, omega, _ = lk.skew_normal_location_scale(0.0, 0.9, 6.0)
    np.testing.assert_allclose(value, stats.skewnorm(6.0, loc=xi, scale=omega).logpdf(y), rtol=1e-12)
    assert all(np.isfinite(g).all() for g in grads) and value[0] < -1e4
    y = np.array([-80.0, -20.0, -3.0, 0.0, 2.0, 30.0])
    mu, sigma, beta = np.zeros(6), np.full(6, 0.5), np.full(6, 2.0)
    value, *grads = _rows(lk.exgaussian_logp, mode, y, mu, sigma, beta)
    np.testing.assert_allclose(value, stats.exponnorm(4.0, loc=-2.0, scale=0.5).logpdf(y), rtol=1e-12)
    assert all(np.isfinite(g).all() for g in grads) and value[0] < -1e4


def test_tails_equal_stans_functions_where_they_are_finite():
    """Stan 2.32's skew_normal_lpdf and exp_mod_normal_lpdf (brms's forms), evaluated through rstan::expose_stan_functions at
    the distributions of the test above (printed with 17 digits): finite and equal to these where log Phi's argument is -36.25
    and above, -inf at -46.25 and -242, where these stay finite (divergence D77)."""
    from autobayes.families import pointwise

    stan_sn = {-60.0: -np.inf, -9.0: -540.89529403339941, -6.0: -209.13959180535016, -5.0: -133.2545510139955,
               -2.0: -9.152687918083231, 0.0: -0.91316025442954452}
    stan_eg = {-80.0: -np.inf, -25.0: -np.inf, -20.0: -653.2032846514843, -19.0: -583.14662293880406,
               -3.0: -4.5662125617132174, 0.0: -1.6619856017541847}
    for table, ours in ((stan_sn, lambda y: pointwise.skew_normal(y, 0.0, 0.9, 6.0)),
                        (stan_eg, lambda y: pointwise.exgaussian(y, 0.0, 0.5, 2.0))):
        y = np.array(list(table))
        got, want = ours(y), np.array(list(table.values()))
        finite = np.isfinite(want)
        assert np.isfinite(got).all() and (got[~finite] < -1000).all()
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-15)


@pytest.mark.parametrize("family", ["skew_normal", "exgaussian", "shifted_lognormal"])
def test_likelihood_objects_draw_their_distribution(family):
    import pymc as pm

    rng = np.random.default_rng(11)
    v = VALUES[family]
    eta = np.array([-0.5, 0.2, 1.0])
    if family == "skew_normal":
        dist = pm.CustomDist.dist(eta, v["sigma"], v["alpha"], logp=lk.skew_normal_logp, random=lk.skew_normal_random)
    elif family == "exgaussian":
        dist = pm.CustomDist.dist(eta, v["sigma"], v["beta"], logp=lk.exgaussian_logp, random=lk.exgaussian_random)
    else:
        dist = pm.CustomDist.dist(eta, v["sigma"], v["ndt"], logp=lk.shifted_lognormal_logp, random=lk.shifted_lognormal_random)
    draws = pm.draw(dist, draws=40000, random_seed=rng)
    ref = scipy_distribution(family, eta, v)
    for k in range(3):                                          # the mean and the quartiles, within sampling error
        sd = ref.std()[k]
        assert abs(draws[:, k].mean() - ref.mean()[k]) < 4.5 * sd / math.sqrt(40000)
        for q in (0.25, 0.5, 0.75):
            assert abs(np.mean(draws[:, k] <= ref.ppf(q)[k]) - q) < 4.5 * math.sqrt(q * (1 - q) / 40000)


# -- the extra parameters: priors, refusals, value_point ------------------------------------------------------------------
def test_extra_parameters_are_required_exactly_for_their_family():
    df = frame()
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"the skew_normal family needs a prior on its skewness alpha: pass alpha=..., such as "
                                            r"ab.Normal\(0, 4\) \(brms's default is normal\(0, 4\)\)"):
        ab.skew_normal("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r"needs a prior on its mean of the exponential component beta.*gamma\(1, 0.1\)"):
        ab.exgaussian("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r'needs a prior on its non-decision time ndt.*ab.Uniform\(0, "min_Y"\).*uniform\(0, min_Y\)'):
        ab.shifted_lognormal("ypos ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r"the gaussian family has no skewness; do not pass alpha"):
        ab.gaussian("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), alpha=ab.Normal(0, 4)))
    with pytest.raises(ab.PriorError, match=r"the skew_normal family has no non-decision time; do not pass ndt.*Its skewness is "
                                            r"alpha: pass alpha=..."):
        ab.skew_normal("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), alpha=ab.Normal(0, 4), ndt=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match="the exponential family has no mean of the exponential component; do not pass beta"):
        ab.exponential("ypos ~ x", df, ab.Priors(**base, beta=ab.Gamma(1, 0.1)))
    with pytest.raises(ab.PriorError, match="the weibull family has no residual SD; do not pass sigma"):
        ab.weibull("ypos ~ x", df, ab.Priors(**base, shape=ab.Gamma(2, 1), sigma=ab.HalfNormal(1)))


def test_each_support_takes_its_own_priors():
    df = frame()
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
    with pytest.raises(ab.PriorError, match=r"prior for 'alpha' must be a location prior \(brms's real alpha\), one of "
                                            r"\['Normal', 'StudentT'\]; got HalfNormal"):
        ab.skew_normal("y ~ x", df, ab.Priors(**base, alpha=ab.HalfNormal(4)))
    with pytest.raises(ab.PriorError, match=r"prior for 'beta' must be a positive-support prior \(brms's real<lower=0> beta\)"):
        ab.exgaussian("y ~ x", df, ab.Priors(**base, beta=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'ndt' must be a prior truncated to \(0, min\(y\)\).*HalfStudentT\(nu, s\) is "
                                            r"StudentT\(nu, 0, s\) there\), one of .*; got HalfStudentT"):
        ab.shifted_lognormal("ypos ~ x", df, ab.Priors(**base, ndt=ab.HalfStudentT(3, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'alpha' must be a Prior, got dict \(.*no formula of its own, divergence D79\)"):
        ab.skew_normal("y ~ x", df, ab.Priors(**base, alpha={"intercept": ab.Normal(0, 1)}))
    with pytest.raises(ab.PriorError, match=r'Uniform is brms\'s uniform\(0, min_Y\), the default prior of ndt: write Uniform\(0, "min_Y"\); '
                                            r"got Uniform\(0, 1\).*D78"):
        ab.Uniform(0, 1)
    with pytest.raises(ab.PriorError, match=r"prior for 'sigma' must be a positive-support prior"):
        ab.shifted_lognormal("ypos ~ x", df, ab.Priors(**{**base, "sigma": ab.Uniform(0, "min_Y")}, ndt=ab.Uniform(0, "min_Y")))
    for prior in (ab.Normal(0, 0.3), ab.StudentT(3, 0, 1), ab.HalfNormal(0.3), ab.HalfCauchy(1), ab.Exponential(2), ab.Gamma(2, 5),
                  ab.InverseGamma(3, 0.5), ab.Uniform(0, "min_Y")):
        assert ab.shifted_lognormal("ypos ~ x", df, ab.Priors(**base, ndt=prior)).priors.ndt == prior


def test_the_extra_parameters_are_refused_in_mixtures_and_multivariate_models():
    df = frame()
    comp = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
    with pytest.raises(ab.PriorError, match=r"\['alpha'\] at the top level do not apply"):
        ab.mixture("y ~ x", df, ab.Priors(components=[comp, comp], theta=ab.Dirichlet(2), alpha=ab.Normal(0, 4)),
                   families=["gaussian", "gaussian"])
    with pytest.raises(ab.PriorError, match="the gaussian family has no skewness; do not pass alpha"):
        ab.mixture("y ~ x", df, ab.Priors(components=[comp, replace(comp, alpha=ab.Normal(0, 4))], theta=ab.Dirichlet(2)),
                   families=["gaussian", "gaussian"])
    resp = {"y": {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)},
            "w": {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)}}
    with pytest.raises(ab.PriorError, match=r"takes its priors by response.*got \['beta'\] at the top level"):
        ab.gaussian("mvbind(y, w) ~ x", df, ab.Priors(resp=resp, beta=ab.Gamma(1, 0.1)), rescor=False)
    with pytest.raises(ab.PriorError, match=r"response 'y': the gaussian family has no non-decision time"):
        ab.gaussian("mvbind(y, w) ~ x", df, ab.Priors(resp={**resp, "y": {**resp["y"], "ndt": ab.Normal(0, 1)}}), rescor=False)
    with pytest.raises(ValueError, match="alpha, beta and ndt are the skew_normal, exgaussian and shifted_lognormal families' "
                                         "parameters; a mixture has none"):
        ab.mixture("y ~ x", df, ab.Priors(components=[comp, comp], theta=ab.Dirichlet(2)), families=["gaussian", "gaussian"]).value_point(
            intercept=[0.0, 1.0], b=np.zeros((1, 2)), sigma=[1.0, 1.0], theta=[0.5, 0.5], alpha=1.0)


def test_value_point_coordinates_and_refusals():
    df = frame()
    m = build("shifted_lognormal")
    top = df.ypos.min()
    p = m.value_point(intercept=0.1, b=[0.2], sigma=0.5, ndt=0.25 * top)
    np.testing.assert_allclose(p["ndt_interval__"], math.log(0.25 / 0.75), rtol=1e-14)    # Stan's log((ndt - 0) / (min_Y - ndt))
    for bad in (0.0, top, -0.1, 2 * top, np.nan):
        with pytest.raises(ValueError, match="ndt .* must (lie strictly between 0 and min|be a finite)"):
            m.value_point(intercept=0.1, b=[0.2], sigma=0.5, ndt=bad)
    with pytest.raises(ValueError, match=r"ndt \(the non-decision time\) is required for the shifted_lognormal family"):
        m.value_point(intercept=0.1, b=[0.2], sigma=0.5)
    m = build("exgaussian")
    np.testing.assert_allclose(m.value_point(intercept=0.1, b=[0.2], sigma=0.5, beta=1.5)["beta_log__"], math.log(1.5), rtol=1e-15)
    with pytest.raises(ValueError, match=r"beta \(the mean of the exponential component of the exgaussian family\) must be positive"):
        m.value_point(intercept=0.1, b=[0.2], sigma=0.5, beta=0.0)
    with pytest.raises(ValueError, match=r"the exgaussian family has no skewness; omit alpha \(its mean of the exponential component is beta\)"):
        m.value_point(intercept=0.1, b=[0.2], sigma=0.5, beta=1.5, alpha=0.3)
    m = build("skew_normal")
    assert m.value_point(intercept=0.1, b=[0.2], sigma=0.5, alpha=-3.25)["alpha"] == -3.25
    assert m.names["alpha"] == "alpha" and "alpha" in {w.name for w in m.pymc_model.value_vars}


# -- ndt's prior: truncated to (0, min y) and renormalised, as brms's lprior ---------------------------------------------------
@pytest.mark.parametrize("prior", [ab.Normal(0, 0.3), ab.StudentT(3, 0.1, 0.2), ab.HalfNormal(0.3), ab.HalfCauchy(0.5),
                                   ab.Exponential(4), ab.Gamma(2, 8), ab.InverseGamma(3, 0.5), ab.Uniform(0, "min_Y")],
                         ids=lambda p: p.name)
@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_ndt_prior_is_truncated_and_renormalised(prior, mode):
    m = build("shifted_lognormal", ndt=prior)
    top = frame().ypos.min()
    pmm = m.pymc_model
    for u in (-3.0, 0.3, 2.5):
        p = m.value_point(intercept=0.1, b=[0.2], sigma=0.5, ndt=top * expit(u))
        if mode == "pytensor":
            got = float(pmm.compile_logp(vars=[pmm["ndt"]], jacobian=False)(p))
        else:
            from pymc.sampling.jax import get_jaxified_graph

            value = pmm.rvs_to_values[pmm["ndt"]]
            jf = get_jaxified_graph(inputs=[value], outputs=[pmm.logp(vars=[pmm["ndt"]], jacobian=False)])
            got = float(jf(p["ndt_interval__"])[0])
        x = top * expit(u)
        tol = 1e-8 if prior.name == "HalfCauchy" else 1e-13     # PyMC's HalfCauchy density adds log(2) as a float32 constant (1.9e-9)
        if prior.name == "Uniform":
            want = -math.log(top)
        else:
            from autobayes.priors import _scipy_distribution

            d = _scipy_distribution(prior)
            want = d.logpdf(x) - math.log(d.cdf(top) - d.cdf(0.0))
        assert abs(got - want) <= tol * max(1.0, abs(want)), (u, got, want)


def test_ndt_draws_lie_below_the_smallest_response_and_follow_the_truncated_prior():
    import pymc as pm

    m = build("shifted_lognormal", ndt=ab.Normal(0, 0.3))
    top = frame().ypos.min()
    draws = pm.draw(m.pymc_model["ndt"], draws=20000, random_seed=2)
    assert draws.min() > 0 and draws.max() < top
    ref = stats.truncnorm(0.0, top / 0.3, loc=0.0, scale=0.3)
    assert abs(draws.mean() - ref.mean()) < 4.5 * ref.std() / math.sqrt(20000)
    assert np.isfinite(m.pymc_model.initial_point()["ndt_interval__"])           # the support point: the interval's middle
    assert m.pymc_model.initial_point()["ndt_interval__"] == 0.0


def test_bounded_log_mass_from_either_tail():
    for prior, lo, hi in ((ab.Normal(0, 1), 0.0, 0.3), (ab.Normal(0, 1), 6.0, 7.0), (ab.Gamma(3, 1), 20.0, 21.0),
                          (ab.StudentT(3, 0, 1), -2.0, 1.0)):
        from autobayes.priors import _scipy_distribution

        d = _scipy_distribution(prior)
        want = math.log(d.sf(lo) - d.sf(hi)) if d.cdf(lo) > 0.5 else math.log(d.cdf(hi) - d.cdf(lo))
        assert abs(bounded_log_mass(prior, lo, hi) - want) <= 1e-12 * abs(want), prior
    assert "Uniform" in BOUNDABLE and "HalfStudentT" not in BOUNDABLE


# -- the term model composes with the new likelihoods ---------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_slopes_categorical_interactions_transforms_offset_and_gp_compose(family):
    """Correlated varying slopes, a combined grouping factor g:h, a declared categorical predictor and its interaction, a
    transform, an offset and a Gaussian process all reach the likelihood through the linear predictor: the observed rows'
    density at a point is SciPy's at the graph's linear predictor."""
    y = RESPONSE[family]
    extra = dict(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "g:h": ab.HalfNormal(0.5)}, cor={"g": ab.LKJ(2)},
                 sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[family], **extra)
    m = getattr(ab, family)(f"{y} ~ x * z + I(w^2) + offset(o) + gp(w) + (1 + x | g) + (1 | g:h)", frame(), priors,
                            categorical={"z": "p"})
    pmm = m.pymc_model
    rng = np.random.default_rng(7)
    p = {k: v + 0.2 * rng.standard_normal(np.shape(v)) for k, v in pmm.initial_point().items()}
    (parent,) = graph(m, p, [m.names["linpred"]])
    values = {"shape": math.exp(p["alpha_log__"])} if family == "weibull" else {}
    if "sigma_log__" in p:
        values["sigma"] = math.exp(p["sigma_log__"])
    values.update(alpha=p.get("alpha"), beta=math.exp(p["beta_log__"]) if "beta_log__" in p else None,
                  ndt=frame().ypos.min() * expit(p["ndt_interval__"]) if "ndt_interval__" in p else None)
    np.testing.assert_allclose(rows_logp(m, p), scipy_distribution(family, parent.ravel(), values).logpdf(frame()[y].to_numpy()),
                               rtol=1e-11)
    assert m.spec.gp_terms and m.design.offset is not None and "g" in m.correlation_names and {"x:z", "I(w^2)"} <= set(m.spec.terms)
    assert "g:h" in m.spec.groups


# -- predictions ------------------------------------------------------------------------------------------------------------
def _posterior(m, values, n_draw=3):
    """Synthetic posterior draws: one chain, the same values in every draw but the intercept."""
    post = {m.names["intercept"]: np.linspace(0.2, 0.6, n_draw)[None], "x": np.full((1, n_draw), 0.3)}
    for name, v in values.items():
        post[m.names[name]] = np.full((1, n_draw), v)
    return az.from_dict(posterior=post)


@pytest.mark.parametrize("family", FAMILIES)
def test_predictions_by_hand_and_response_draws(family):
    m = build(family)
    v = VALUES[family]
    idata = _posterior(m, v)
    nd = pd.DataFrame({"x": [-1.0, 0.0, 2.0]})
    eta = np.linspace(0.2, 0.6, 3)[:, None] + 0.3 * nd.x.to_numpy()[None]   # the intercept draws are on the raw scale
    lin = m.predict(idata, newdata=nd, kind="linear")["eta"].to_numpy()[0]
    np.testing.assert_allclose(lin, eta, rtol=1e-14)
    mean = m.predict(idata, newdata=nd, kind="mean")[m.names["mean"]].to_numpy()[0]
    np.testing.assert_allclose(mean, scipy_distribution(family, np.exp(eta) if m.link == "log" else eta, v).mean(), rtol=1e-12)
    big = pd.DataFrame({"x": np.zeros(20000)})
    draws = m.predict(_posterior(m, v, 1), newdata=big, kind="response", seed=5)["y"].to_numpy().ravel()
    ref = scipy_distribution(family, np.exp(0.2) if m.link == "log" else 0.2, v)
    assert abs(draws.mean() - ref.mean()) < 4.5 * ref.std() / math.sqrt(draws.size)
    assert abs(np.mean(draws <= ref.median()) - 0.5) < 4.5 * 0.5 / math.sqrt(draws.size)
    if family == "shifted_lognormal":
        assert draws.min() > v["ndt"]


@pytest.mark.parametrize("family", FAMILIES)
def test_prior_predictive(family):
    m = build(family, f"{RESPONSE[family]} ~ x + (1 | g)", sd=ab.HalfNormal(0.5))
    out = m.prior_predictive(draws=30, seed=3)
    y = out.prior_predictive[RESPONSE[family]].values
    assert y.shape == (1, 30, 24) and np.isfinite(y).all()
    if family in ("weibull", "exponential", "shifted_lognormal"):
        assert (y > 0).all()
    if family == "shifted_lognormal":
        ndt = out.prior["ndt"].values
        assert ((ndt > 0) & (ndt < frame().ypos.min())).all() and (y > ndt[..., None]).all()


# -- brms's default priors (A07) ----------------------------------------------------------------------------------------
# brms 2.23.0's default_prior() of y ~ x + (1 | g) on frame() (written to CSV at 17 digits and read by R 4.5.2 with LC_ALL=C,
# the oracle's collation), its class rows with a prior, as (class, prior, lb, ub) in brms's order
BRMS_CLASS_ROWS = {
    "weibull": [("Intercept", "student_t(3, 0.4, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                ("shape", "gamma(0.01, 0.01)", "0", "")],
    "exponential": [("Intercept", "student_t(3, 0.4, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", "")],
    "skew_normal": [("Intercept", "student_t(3, 1.7, 2.5)", "", ""), ("alpha", "normal(0, 4)", "", ""),
                    ("sd", "student_t(3, 0, 2.5)", "0", ""), ("sigma", "student_t(3, 0, 2.5)", "0", "")],
    "shifted_lognormal": [("Intercept", "student_t(3, 0.4, 2.5)", "", ""), ("ndt", "uniform(0, min_Y)", "0", "min_Y"),
                          ("sd", "student_t(3, 0, 2.5)", "0", ""), ("sigma", "student_t(3, 0, 2.5)", "0", "")],
    "exgaussian": [("Intercept", "student_t(3, 1.7, 2.5)", "", ""), ("beta", "gamma(1, 0.1)", "0", ""),
                   ("sd", "student_t(3, 0, 2.5)", "0", ""), ("sigma", "student_t(3, 0, 2.5)", "0", "")],
}


@pytest.mark.parametrize("family", FAMILIES)
def test_default_priors_build_the_model_brms_describes(family):
    formula = f"{RESPONSE[family]} ~ x + (1 | g)"
    pri = ab.default_priors(formula, frame(), family=family)
    m = getattr(ab, family)(formula, frame(), pri)
    rows = [(r["class"], r["prior"], r["lb"], r["ub"]) for r in m.prior_table() if r["coef"] == "" and r["group"] == "" and r["prior"]]
    assert rows == BRMS_CLASS_ROWS[family]
    assert all(r["source"] == "default" for r in m.prior_table())
    for e in get_family(family).extra_parameters:
        assert e.name in m.default_prior_labels and getattr(pri, e.name).source == "default"
    if family == "shifted_lognormal":                         # uniform(0, min_Y): -log(min y) wherever ndt lies
        top = frame().ypos.min()
        pmm = m.pymc_model
        p = m.value_point(intercept=0.1, b=[0.2], sd=0.5, z=np.zeros(4), sigma=0.5, ndt=0.3 * top)
        np.testing.assert_allclose(pmm.compile_logp(vars=[pmm["ndt"]], jacobian=False)(p), -math.log(top), rtol=1e-15)
        assert 'ndt ~ Uniform(lower=0, upper=min_Y)   (brms default uniform(0, min_Y)), uniform on (0, min(y) = ' in m.describe()


def wide_frame():
    """Responses on a wide scale: log(ypos) with a MAD above 2.5, y with one far above it."""
    rng = np.random.default_rng(12)
    n = 30
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "g": [f"l{k}" for k in np.arange(n) % 3],
                         "ypos": np.round(np.exp(rng.normal(1.0, 4.0, n)), 6), "y": np.round(rng.normal(40.0, 25.0, n), 4)})


# brms 2.23.0's default_prior() of y ~ x + (1 | g) on wide_frame() (CSV at 17 digits, R 4.5.2 with LC_ALL=C): the scale is
# round(mad(log y), 1) = 4.1 for the three log-scale families and round(mad(y), 1) = 18.2 for the other two
BRMS_WIDE_ROWS = {
    "weibull": [("Intercept", "student_t(3, 2.2, 4.1)", "", ""), ("sd", "student_t(3, 0, 4.1)", "0", ""),
                ("shape", "gamma(0.01, 0.01)", "0", "")],
    "exponential": [("Intercept", "student_t(3, 2.2, 4.1)", "", ""), ("sd", "student_t(3, 0, 4.1)", "0", "")],
    "skew_normal": [("Intercept", "student_t(3, 45.2, 18.2)", "", ""), ("alpha", "normal(0, 4)", "", ""),
                    ("sd", "student_t(3, 0, 18.2)", "0", ""), ("sigma", "student_t(3, 0, 18.2)", "0", "")],
    "shifted_lognormal": [("Intercept", "student_t(3, 2.2, 4.1)", "", ""), ("ndt", "uniform(0, min_Y)", "0", "min_Y"),
                          ("sd", "student_t(3, 0, 4.1)", "0", ""), ("sigma", "student_t(3, 0, 4.1)", "0", "")],
    "exgaussian": [("Intercept", "student_t(3, 45.2, 18.2)", "", ""), ("beta", "gamma(1, 0.1)", "0", ""),
                   ("sd", "student_t(3, 0, 18.2)", "0", ""), ("sigma", "student_t(3, 0, 18.2)", "0", "")],
}


@pytest.mark.parametrize("family", FAMILIES)
def test_default_priors_read_the_response_on_brms_scale(family):
    formula = f"{RESPONSE[family]} ~ x + (1 | g)"
    m = getattr(ab, family)(formula, wide_frame(), ab.default_priors(formula, wide_frame(), family=family))
    assert [(r["class"], r["prior"], r["lb"], r["ub"]) for r in m.prior_table()
            if r["coef"] == "" and r["group"] == "" and r["prior"]] == BRMS_WIDE_ROWS[family]


# -- describe(), the hash seed, the target identity and the fit record ----------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_describe(family):
    m = build(family)
    text = m.describe()
    spec = get_family(family)
    assert f"family    : {family}, {spec.link} link" in text
    for e in spec.extra_parameters:
        assert f"  {e.coordinate:<24} {e.coordinate_meaning}" in text
        assert text.index(f"  {'sigma_log__':<24} log(sigma)") < text.index(f"  {e.coordinate:<24}")
    assert ab.Model(f"{RESPONSE[family]} ~ x", frame(), m.priors, family=family).describe() == text


def _fresh(seed: int, script: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    return json.loads(lines[-1])


SET_ORDER = 'import json; print(json.dumps({"order": list(set(("mu", "sigma", "alpha")) - {"mu"})}))'   # Bambi's auxiliary_parameters
BUILD = """
import json
from tests.test_continuous_families import build
m = build("skew_normal")
print(json.dumps({"describe": m.describe(), "identity": m.target_identity, "value_vars": [v.name for v in m.pymc_model.value_vars]}))
"""


def test_describe_does_not_depend_on_the_hash_seed():
    """Bambi creates sigma and alpha in the iteration order of the set of their names, so PyMC's order of ``sigma_log__`` and
    ``alpha`` follows ``PYTHONHASHSEED``; ``describe()`` (sigma, then alpha, brms's order) and the target identity do not."""
    seeds = {}
    for seed in range(32):
        seeds.setdefault(tuple(_fresh(seed, SET_ORDER)["order"]), seed)
        if len(seeds) == 2:
            break
    assert len(seeds) == 2
    first, second = (_fresh(seed, BUILD) for seed in seeds.values())
    assert first["value_vars"] != second["value_vars"] and sorted(first["value_vars"]) == sorted(second["value_vars"])
    assert first["describe"] == second["describe"] and first["identity"] == second["identity"]


@pytest.mark.parametrize("family", ["skew_normal", "shifted_lognormal", "exgaussian"])
def test_target_identity_and_fit_record(family, monkeypatch):
    m = build(family)
    (extra,) = get_family(family).extra_parameters
    other = {"alpha": ab.Normal(0, 3), "beta": ab.Gamma(2, 0.1), "ndt": ab.Uniform(0, "min_Y")}[extra.name]
    assert build(family).target_identity == m.target_identity
    assert build(family, **{extra.name: other}).target_identity != m.target_identity
    post = _posterior(m, VALUES[family], n_draw=4)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert extra.name in rows and "sigma" in rows
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == family and rec["model"]["priors"][extra.name] == str(OWN[family][extra.name])
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


@pytest.mark.parametrize("family", FAMILIES)
def test_log_lik_is_the_models_own_pointwise_density(family):
    m = build(family, f"{RESPONSE[family]} ~ x + (1 | g)", sd=ab.HalfNormal(0.5))
    v = VALUES[family]
    post = _posterior(m, v, n_draw=2)
    post.posterior["1|g_sigma"] = (("chain", "draw"), np.full((1, 2), 0.5))
    effects = np.array([0.3, -0.2, 0.1, 0.0])
    coord = m.group_names["g"]["coordinate"]
    post.posterior["1|g"] = (("chain", "draw", coord), np.tile(effects, (1, 2, 1)))
    post.posterior = post.posterior.assign_coords({coord: list(m.design.levels)})
    ll = ab.log_lik(m, post).to_numpy()[0]
    x = frame().x.to_numpy()
    for d, a in enumerate([0.2, 0.6]):
        z = (effects / 0.5)
        p = m.value_point(intercept=a + 0.3 * x.mean(), b=[0.3], sd=0.5, z=z, **v)
        np.testing.assert_allclose(ll[d], rows_logp(m, p), rtol=1e-12)


@pytest.mark.parametrize("case", [f"{f}_varint" for f in FAMILIES])
def test_bayes_R2_and_loo_on_the_frozen_points(case):
    """bayes_R2 at the frozen brms points is brms's definition on brms's own posterior_epred (the fixture's), and loo runs on
    the families' log_lik (C07 takes them as it takes every family; brms's loo of these cases is not frozen)."""
    from tests.oracle import adapter, gates

    b = gates.Bundle(case)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    epred, y = b.fixture.prediction("fitted", "epred"), b.model.design.y
    want = np.var(epred, axis=-1, ddof=1) / (np.var(epred, axis=-1, ddof=1) + np.var(y - epred, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(b.model, idata).to_numpy()[0], want, rtol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # eight one-chain draws: Pareto k is large, as it should be
        result = ab.loo(b.model, idata)
    assert np.isfinite(result.estimates.loc["elpd_loo", "Estimate"])
