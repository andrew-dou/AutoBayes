"""Inverse Gaussian, von Mises, asymmetric Laplace, Frechet and generalized extreme value responses (brms-parity E02,
docs/continuous-families.md): each family's density in brms's parameterisation against SciPy's own distribution (whose moments
confirm the parameterisation) on both paths, log I0 for every kappa, the GEV's derived xi and its density and mean as xi
passes 0, the inverse Gaussian's 1/mu^2 link outside eta > 0, the new parameters kappa, quantile (learned or fixed) and xi
(on tmp_xi), the Frechet nu (learned or fixed), the refusals naming every construct no frozen case verifies, the term model
composing with the new likelihoods, by-hand predictions and response draws, prior-predictive draws, brms's default priors,
describe() under two hash seeds, the target identity and the fit record, and log_lik. No MCMC; the brms comparison is the
oracle cases inverse_gaussian_varint, inverse_gaussian_log, von_mises_varint, asym_laplace_varint, asym_laplace_quantile,
frechet_varint and gen_extreme_value_varint."""
import json
import math
import warnings
from decimal import Decimal, getcontext
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, i0e, i1e

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import FAMILIES, get_family
from autobayes.families.gen_extreme_value import EULER_GAMMA, gev_mean_offset
from tests.test_continuous_families import _fresh, _rows, graph, rows_logp

# key -> (family, constructor keywords): each family, the inverse Gaussian under both links, and the fixed-parameter forms
KEYS = {"inverse_gaussian": ("inverse_gaussian", {}), "inverse_gaussian_log": ("inverse_gaussian", {"link": "log"}),
        "von_mises": ("von_mises", {}), "asym_laplace": ("asym_laplace", {}), "asym_laplace_fixed": ("asym_laplace", {"quantile": 0.3}),
        "frechet": ("frechet", {}), "frechet_fixed": ("frechet", {"nu": 3.5}), "gen_extreme_value": ("gen_extreme_value", {})}
NEW = ("inverse_gaussian", "von_mises", "asym_laplace", "frechet", "gen_extreme_value")
RESPONSE = {"inverse_gaussian": "ypos", "von_mises": "yang", "asym_laplace": "y", "frechet": "ypos", "gen_extreme_value": "y"}
OWN = {"inverse_gaussian": {"shape": ab.Gamma(0.5, 0.2)}, "inverse_gaussian_log": {"shape": ab.Gamma(0.5, 0.2)},
       "von_mises": {"kappa": ab.Gamma(2, 0.01)}, "asym_laplace": {"sigma": ab.HalfNormal(1), "quantile": ab.Beta(2, 2)},
       "asym_laplace_fixed": {"sigma": ab.HalfNormal(1)}, "frechet": {"nu": ab.Gamma(2, 0.1)}, "frechet_fixed": {},
       "gen_extreme_value": {"sigma": ab.HalfNormal(1), "xi": ab.Normal(0, 0.5)}}
VALUES = {"inverse_gaussian": {"shape": 2.5}, "inverse_gaussian_log": {"shape": 2.5}, "von_mises": {"kappa": 2.2},
          "asym_laplace": {"sigma": 0.7, "quantile": 0.3}, "asym_laplace_fixed": {"sigma": 0.7}, "frechet": {"nu": 3.5},
          "frechet_fixed": {}, "gen_extreme_value": {"sigma": 1.1, "tmp_xi": -0.4}}


def frame(n=24, seed=8):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "w": rng.uniform(-1, 1, n), "z": np.array(["p", "q", "r"])[np.arange(n) % 3],
                         "g": [f"l{k}" for k in np.arange(n) % 4], "h": np.array(["u", "v"])[(np.arange(n) // 4) % 2],
                         "y": rng.normal(0.3, 1.2, n), "ypos": 0.4 + rng.gamma(2.0, 0.6, n),
                         "yang": rng.uniform(-3.0, 3.0, n), "off": rng.uniform(1.0, 2.0, n)})


def family_of_key(key):
    return KEYS[key][0]


def formula_of(key, rhs="x"):
    """``<response> ~ rhs``; the inverse Gaussian's 1/mu^2 link takes the offset ``off`` in (1, 2), which keeps eta positive."""
    family = family_of_key(key)
    return f"{RESPONSE[family]} ~ {rhs}" + (" + offset(off)" if key == "inverse_gaussian" else "")


def build(key, formula=None, frame_=None, **extra):
    family, kwargs = KEYS[key]
    priors = ab.Priors(**{"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 1), **OWN[key], **extra})
    return getattr(ab, family)(formula or formula_of(key), frame() if frame_ is None else frame_, priors, **kwargs)


def natural(m, values):
    """``values`` with the model's own parameters: a fixed nu or quantile is no value."""
    return {k: v for k, v in values.items() if k not in m.family_spec.fixed_parameters or getattr(m, k, None) is None}


def mean_direction(eta):
    return 2.0 * np.arctan(eta)


def gev_xi(tmp_xi, y, mu, sigma):
    """brms's derived xi, written from its definition here (not the library's code)."""
    x = (np.asarray(y) - mu) / sigma
    bounds = sorted([-1.0 / x.max(), -1.0 / x.min()])
    return bounds[0] + (bounds[1] - bounds[0]) / (1.0 + math.exp(-tmp_xi))


def scipy_distribution(key, eta, values, y=None):
    """SciPy's distribution of each row from brms's parameterisation at the linear predictor ``eta``, derived without the
    library's formulas (the GEV's xi from its definition at the observed ``y``)."""
    v = {**{"quantile": 0.3, "nu": 3.5}, **values}
    if key == "inverse_gaussian":
        mu = 1.0 / np.sqrt(eta)
        return stats.invgauss(mu / v["shape"], scale=v["shape"])
    if key == "inverse_gaussian_log":
        return stats.invgauss(np.exp(eta) / v["shape"], scale=v["shape"])
    if key == "von_mises":
        return stats.vonmises(v["kappa"], loc=mean_direction(eta))
    if key.startswith("asym_laplace"):
        p, s = v["quantile"], v["sigma"]
        return stats.laplace_asymmetric(math.sqrt(p / (1 - p)), loc=eta, scale=s / math.sqrt(p * (1 - p)))
    if key.startswith("frechet"):
        nu = v["nu"]
        return stats.invweibull(nu, scale=np.exp(eta) / math.gamma(1 - 1 / nu))
    xi = gev_xi(v["tmp_xi"], y, eta, v["sigma"])
    return stats.genextreme(-xi, loc=eta, scale=v["sigma"])


def brms_mean(key, eta, values, y=None):
    """brms's posterior_epred at ``eta``: the vignette's mean of each family."""
    v = {**{"quantile": 0.3}, **values}
    if key == "inverse_gaussian":
        return 1.0 / np.sqrt(eta)
    if key in ("inverse_gaussian_log", "frechet", "frechet_fixed"):
        return np.exp(eta)
    if key == "von_mises":
        return mean_direction(eta)
    if key.startswith("asym_laplace"):
        p = v["quantile"]
        return eta + v["sigma"] * (1 - 2 * p) / (p * (1 - p))
    xi = gev_xi(v["tmp_xi"], y, eta, v["sigma"])
    return eta + v["sigma"] * (math.gamma(1 - xi) - 1) / xi


def point(m, key, intercept=0.4, b=0.3):
    return m.value_point(intercept=intercept, b=[b], **natural(m, VALUES[key]))


# -- the densities: SciPy, the parameterisation by moments, both paths -----------------------------------------------------
@pytest.mark.parametrize("key", KEYS)
def test_density_is_brms_parameterisation_on_both_paths(key):
    m = build(key)
    df, v = frame(), VALUES[key]
    p = point(m, key)
    x = df.x.to_numpy()
    eta = 0.4 + 0.3 * (x - x.mean()) + (df.off.to_numpy() if key == "inverse_gaussian" else 0.0)
    y = df[RESPONSE[family_of_key(key)]].to_numpy()
    dist = scipy_distribution(key, eta, v, y)
    if key != "von_mises":                                       # the parameterisation, by moments: brms's mean is SciPy's
        np.testing.assert_allclose(dist.mean(), brms_mean(key, eta, v, y), rtol=1e-12)
    np.testing.assert_allclose(rows_logp(m, p), dist.logpdf(y), rtol=1e-12)
    (lin,) = graph(m, p, [m.names["linpred"]])                   # the graph's parent is eta, brms's posterior_linpred
    np.testing.assert_allclose(lin.ravel(), eta, rtol=1e-14)
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    pmm = m.pymc_model
    args = [p[w.name] for w in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(p)), rtol=1e-13)
    g_jax = np.concatenate([np.ravel(a) for a in jax.grad(lambda q: jf(q))(args)])
    # PyTensor's digamma (the Frechet's gradient of lgamma(1 - 1 / nu)) is off by about 1e-10 absolute, as for the Weibull (D01)
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float),
                               rtol=1e-8 if key.startswith("frechet") else 1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_log_bessel_i0_is_exact_for_every_kappa(mode):
    """log I0 is i0's log below 500 and the asymptotic series from 500 on: equal to SciPy's log(i0e(k)) + k (Cephes) with the
    gradient I1 / I0 to rounding, at the switch, past 709.8 where i0 overflows and past 713.9 where Stan's von_mises_lpdf
    raises an overflow (brms's density there is an error, probed with rstan's log_prob; divergence D113)."""
    kappa = np.array([0.0, 1e-3, 0.5, 3.0, 100.0, 499.999, 500.0, 500.001, 709.9, 713.95, 1e4, 1e8])
    value, grad = _rows(lambda k: lk.log_bessel_i0(k), mode, kappa)
    want = np.log(i0e(kappa)) + kappa                            # (near 0 both are log(1 + k^2 / 4), within 2e-16 absolute)
    np.testing.assert_allclose(value, want, rtol=4e-16, atol=2e-16)
    # JAX differentiates its i0 through |kappa|, whose derivative it takes as 1 at exactly 0 (the true gradient there is 0);
    # every model's kappa is exp(log kappa) > 0
    inside = slice(1, None) if mode == "jax" else slice(None)
    np.testing.assert_allclose(grad[inside], (i1e(kappa) / i0e(kappa))[inside], rtol=4e-16, atol=2e-16)
    value, grad = _rows(lambda k: lk.log_bessel_i0e(k), mode, kappa)
    np.testing.assert_allclose(value, np.log(i0e(kappa)), rtol=1e-13, atol=2e-16)    # log(i0(k)) - k below 500: 1e-13 absolute


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_von_mises_density_is_finite_beyond_stans_overflow(mode):
    """At kappa = 1000 and 1e6 (Stan's density raises an overflow above 713.9; D113) the density is SciPy's von Mises (whose
    log normaliser is log(2 pi i0e(kappa)) + kappa), with a finite gradient in every argument."""
    y = np.array([-0.05, 0.0, 0.02, 0.1])
    for kappa in (1000.0, 1e6):
        value, *grads = _rows(lk.von_mises_logp, mode, y, np.full(4, 0.01), np.full(4, kappa))
        np.testing.assert_allclose(value, stats.vonmises(kappa, loc=2 * np.arctan(0.01)).logpdf(y), rtol=1e-12)
        assert all(np.isfinite(g).all() for g in grads)


# -- the generalized extreme value family's xi -------------------------------------------------------------------------------
def test_gev_xi_is_brms_derivation_and_keeps_every_row_inside_the_support():
    m = build("gen_extreme_value", "y ~ x + (1 | g)", sd=ab.HalfNormal(1))
    df = frame()
    y, x = df.y.to_numpy(), df.x.to_numpy()
    rng = np.random.default_rng(3)
    for tmp_xi in (-30.0, -2.0, 0.0, 0.7, 30.0):
        z = rng.normal(size=4)
        p = m.value_point(intercept=0.2, b=[0.4], sd=0.6, z=z, sigma=0.9, tmp_xi=tmp_xi)
        mu, xi = graph(m, p, ["mu", "xi"])
        np.testing.assert_allclose(xi, gev_xi(tmp_xi, y, mu.ravel(), 0.9), rtol=1e-14)
        assert (1.0 + xi * (y - mu.ravel()) / 0.9 > 0).all() and np.isfinite(rows_logp(m, p)).all()
    assert m.names["xi"] == "xi" and m.names["tmp_xi"] == "tmp_xi" and "tmp_xi" in {w.name for w in m.pymc_model.value_vars}
    assert "xi" in {d.name for d in m.pymc_model.deterministics}


def _gev_reference(y, mu, sigma, xi):
    """The GEV log density in 50-digit decimal arithmetic: -log sigma - (1 + xi) w - exp(-w), w = ln(1 + xi z) / xi."""
    getcontext().prec = 50
    out = []
    for yy in y:
        s, x = Decimal(repr(float(sigma))), Decimal(repr(float(xi)))
        z = (Decimal(repr(float(yy))) - Decimal(repr(float(mu)))) / s
        w = z if x == 0 else (1 + x * z).ln() / x
        out.append(float(-s.ln() - (1 + x) * w - (-w).exp()))
    return np.array(out)


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_gev_density_is_exact_as_xi_passes_zero(mode):
    """The density's w = log1p(xi z) / xi is z g(xi z) with g's series near 0: equal to a 50-digit reference for xi from -0.2
    to 0.3 through exactly 0 (the Gumbel), with a finite gradient there."""
    y = np.array([-2.5, -0.4, 0.0, 0.3, 1.7, 4.0])
    for xi in (-0.2, -1e-3, -1e-9, 0.0, 1e-12, 1e-6, 2e-3, 0.3):
        value, *grads = _rows(lk.gen_extreme_value_logp, mode, y, np.full(6, 0.2), np.full(6, 1.3), np.full(6, xi))
        np.testing.assert_allclose(value, _gev_reference(y, 0.2, 1.3, xi), rtol=1e-14, atol=1e-15)
        assert all(np.isfinite(g).all() for g in grads)
    value, _, _, _, dxi = _rows(lk.gen_extreme_value_logp, mode, y, np.full(6, 0.2), np.full(6, 1.3), np.zeros(6))
    h = 1e-6                                                     # the xi gradient at 0 by central differences of the reference
    fd = (_gev_reference(y, 0.2, 1.3, h) - _gev_reference(y, 0.2, 1.3, -h)) / (2 * h)
    np.testing.assert_allclose(dxi, fd, rtol=1e-7, atol=1e-9)


def test_gev_mean_offset_against_50_digit_values():
    """(Gamma(1 - xi) - 1) / xi against mpmath 1.3.0 at 50 digits (Euler's constant at 0): exact near 0, where brms's form
    loses 1e-6 relative at |xi| = 1e-10 and is NaN at 0, and infinite for xi >= 1, where the mean is (brms's formula is finite
    there; D114)."""
    ref = {-0.9: 0.04248240899179175621380578, -0.3: 0.3417643456457427050208168, -0.01: 0.5674148808493962864670111,
           -1e-5: 0.5772057744323265067707798, -1e-10: 0.5772156648026272610827896, 0.0: 0.5772156649015328606065121,
           1e-12: 0.577215664902521916601841, 1e-6: 0.5772166539584356686368774, 0.009: 0.5861913968464707102907008,
           0.0101: 0.5872987241944551095427463, 0.3: 0.9935177754918592856039039, 0.9: 9.459452998520813151436097,
           0.999: 999.423195680275741856839}
    np.testing.assert_allclose(gev_mean_offset(np.array(list(ref))), np.array(list(ref.values())), rtol=2e-14)
    assert gev_mean_offset(np.array(0.0)) == pytest.approx(EULER_GAMMA, rel=1e-16)
    assert np.isinf(gev_mean_offset(np.array([1.0, 1.5, 20.0]))).all()


def test_gev_predictions_read_the_posterior_xi():
    m = build("gen_extreme_value")
    post = az.from_dict(posterior={"Intercept": np.array([[0.2, 0.5]]), "x": np.array([[0.3, 0.3]]), "sigma": np.array([[1.1, 0.8]]),
                                   "tmp_xi": np.array([[-0.4, 0.1]]), "xi": np.array([[-0.25, 1.2]])})
    nd = pd.DataFrame({"x": [-1.0, 0.5]})
    mean = m.predict(post, newdata=nd, kind="mean")["mean"].to_numpy()[0]
    eta = np.array([0.2, 0.5])[:, None] + 0.3 * nd.x.to_numpy()[None]
    np.testing.assert_allclose(mean[0], eta[0] + 1.1 * (math.gamma(1.25) - 1) / -0.25, rtol=1e-14)
    assert np.isinf(mean[1]).all()                              # xi = 1.2: the mean is infinite (D114)


# -- the inverse Gaussian's 1/mu^2 link outside eta > 0 -------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_inverse_gaussian_link_needs_a_positive_linear_predictor(mode):
    y = np.array([0.5, 1.0, 2.0, 0.7])
    eta = np.array([-1.0, 0.0, 0.25, 1e-300])
    value, dy, deta, dshape = _rows(lk.inverse_gaussian_inverse_squared_logp, mode, y, eta, np.full(4, 2.0))
    assert np.isneginf(value[:2]).all() and np.isfinite(value[2:]).all()
    assert all(np.isfinite(g).all() for g in (dy, deta, dshape)) and (deta[:2] == 0).all()
    np.testing.assert_allclose(value[2], stats.invgauss(2.0 / 2.0, scale=2.0).logpdf(1.0 * 2.0), rtol=1e-12)   # mu = 2
    m = build("inverse_gaussian")
    post = az.from_dict(posterior={"Intercept": np.array([[0.2]]), "x": np.array([[0.3]]), "alpha": np.array([[2.0]])})
    with pytest.raises(ValueError, match=r"the inverse.gaussian family's 1/mu\^2 link needs a positive linear predictor"):
        m.predict(post, newdata=pd.DataFrame({"x": [-5.0], "off": [0.0]}), kind="mean")
    np.testing.assert_allclose(m.predict(post, newdata=pd.DataFrame({"x": [0.0], "off": [1.0]}), kind="mean")["mu"].to_numpy(),
                               1.0 / math.sqrt(1.2), rtol=1e-15)


def test_links_registry_and_refusals():
    ig = FAMILIES["inverse_gaussian"]
    assert ig.link == "1/mu^2" and set(ig.links) == {"log"} and ig.parent == "inverse_squared_mu"
    log = get_family("inverse_gaussian", "log")
    assert log.link == log.link_scale == "log" and log.parent == "log_mu" and set(log.links) == {"1/mu^2"}
    assert log.with_link("1/mu^2") is ig and get_family("inverse_gaussian", "log") is log and ig.with_link("1/mu^2") is ig
    assert FAMILIES["von_mises"].link == "tan_half" and FAMILIES["von_mises"].parent == "tan_half_mu" and not FAMILIES["von_mises"].links
    df = frame()
    for link in ("inverse", "identity", "softplus"):
        with pytest.raises(ValueError, match=rf"link must be one of \['1/mu\^2', 'log'\] for the inverse_gaussian family, got '{link}'"):
            ab.Model(formula_of("inverse_gaussian"), df, build("inverse_gaussian").priors, family="inverse_gaussian", link=link)
    with pytest.raises(ValueError, match=r"the von_mises family has the tan_half link only; link='identity' is not available"):
        ab.Model("yang ~ x", df, build("von_mises").priors, family="von_mises", link="identity")
    for family, own in (("asym_laplace", "identity"), ("frechet", "log"), ("gen_extreme_value", "identity")):
        with pytest.raises(ValueError, match=rf"the {family} family has the {own} link only; link='inverse' is not available"):
            ab.default_priors(f"{RESPONSE[family]} ~ x", df, family=family, link="inverse")
    m = build("inverse_gaussian_log")
    assert m.link == "log" and m.family_spec is log and m.names["linpred"] == "log_mu" and m.names["mean"] == "mu"
    assert m.target_identity != build("inverse_gaussian", formula_of("inverse_gaussian_log")).target_identity
    assert "family    : inverse_gaussian, log link" in m.describe()


# -- the new parameters: priors, refusals, value_point ------------------------------------------------------------------------
def test_new_parameters_are_required_exactly_for_their_family():
    df = frame()
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"the von_mises family needs a prior on its precision kappa: pass kappa=..., such as "
                                            r"ab.Gamma\(1, 0.1\) \(brms's default is gamma\(2, 0.01\)\)"):
        ab.von_mises("yang ~ x", df, ab.Priors(**base))
    with pytest.raises(ab.PriorError, match=r"needs a prior on its quantile quantile: pass quantile=..., such as ab.Beta\(1, 1\) "
                                            r"\(brms's default is beta\(1, 1\)\)"):
        ab.asym_laplace("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r"needs a prior on its shape xi: pass xi=..., such as ab.Normal\(0, 4\) \(brms's default "
                                            r"is normal\(0, 2.5\)\)"):
        ab.gen_extreme_value("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="the inverse_gaussian family needs a prior on the shape parameter: pass shape=..."):
        ab.inverse_gaussian("ypos ~ x + offset(off)", df, ab.Priors(**base))
    with pytest.raises(ab.PriorError, match=r"the frechet family needs nu: a fixed nu > 1 \(nu=4\), or nu=None with a "
                                            r"positive-support prior Priors\(nu=...\) to learn it"):
        ab.frechet("ypos ~ x", df, ab.Priors(**base))
    with pytest.raises(ab.PriorError, match="the gaussian family has no kappa parameter; do not pass kappa"):
        ab.gaussian("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), kappa=ab.Gamma(2, 0.01)))
    with pytest.raises(ab.PriorError, match="the skew_normal family has no extreme value shape; do not pass xi"):
        ab.skew_normal("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), alpha=ab.Normal(0, 4), xi=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match=r"the von_mises family has no residual SD; do not pass sigma\. Its precision is kappa"):
        ab.von_mises("yang ~ x", df, ab.Priors(**base, kappa=ab.Gamma(2, 0.01), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="the gen_extreme_value family has no quantile; do not pass quantile"):
        ab.gen_extreme_value("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), xi=ab.Normal(0, 1), quantile=ab.Beta(1, 1)))


def test_each_new_parameter_takes_its_own_priors():
    df = frame()
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"prior for 'kappa' must be a positive-support prior \(brms's real<lower=0> kappa\)"):
        ab.von_mises("yang ~ x", df, ab.Priors(**base, kappa=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'quantile' must be a prior on \(0, 1\) \(brms's real<lower=0,upper=1> "
                                            r"quantile\), one of \['Beta'\]; got Gamma"):
        ab.asym_laplace("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), quantile=ab.Gamma(2, 2)))
    with pytest.raises(ab.PriorError, match=r"prior for 'xi' must be a location prior \(brms puts the prior of class xi on its real "
                                            r"tmp_xi\), one of \['Normal', 'StudentT'\]; got HalfNormal"):
        ab.gen_extreme_value("y ~ x", df, ab.Priors(**base, sigma=ab.HalfNormal(1), xi=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'kappa' must be a Prior, got dict \(.*no formula of its own, divergence D112\)"):
        ab.von_mises("yang ~ x", df, ab.Priors(**base, kappa={"intercept": ab.Normal(0, 1)}))
    with pytest.raises(ab.PriorError, match=r"prior for 'nu' must be a positive-support prior"):
        ab.frechet("ypos ~ x", df, ab.Priors(**base, nu=ab.Normal(5, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'shape' must be a positive-support prior"):
        ab.inverse_gaussian("ypos ~ x", df, ab.Priors(**base, shape=ab.Normal(0, 1)), link="log")


def test_value_point_coordinates_and_refusals():
    m = build("von_mises")
    np.testing.assert_allclose(m.value_point(intercept=0.1, b=[0.2], kappa=2.5)["kappa_log__"], math.log(2.5), rtol=1e-15)
    with pytest.raises(ValueError, match=r"kappa \(the precision of the von_mises family\) must be positive"):
        m.value_point(intercept=0.1, b=[0.2], kappa=0.0)
    m = build("asym_laplace")
    np.testing.assert_allclose(m.value_point(intercept=0.1, b=[0.2], sigma=0.5, quantile=0.8)["quantile_logodds__"], math.log(4.0),
                               rtol=1e-14)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"quantile \(the quantile of the asym_laplace family\) must lie strictly between 0 and 1"):
            m.value_point(intercept=0.1, b=[0.2], sigma=0.5, quantile=bad)
    with pytest.raises(ValueError, match=r"quantile \(the quantile\) is required for the asym_laplace family"):
        m.value_point(intercept=0.1, b=[0.2], sigma=0.5)
    fixed = build("asym_laplace_fixed")
    with pytest.raises(ValueError, match="quantile is fixed at 0.3 in this model, not sampled; omit it"):
        fixed.value_point(intercept=0.1, b=[0.2], sigma=0.5, quantile=0.3)
    m = build("gen_extreme_value")
    assert m.value_point(intercept=0.1, b=[0.2], sigma=0.5, tmp_xi=-1.25)["tmp_xi"] == -1.25
    with pytest.raises(ValueError, match=r"the asym_laplace family has no unscaled extreme value shape tmp_xi; omit tmp_xi \(its "
                                         r"quantile is quantile\)"):
        build("asym_laplace").value_point(intercept=0.1, b=[0.2], sigma=0.5, quantile=0.5, tmp_xi=0.0)
    m = build("frechet")
    np.testing.assert_allclose(m.value_point(intercept=0.1, b=[0.2], nu=3.0)["nu_interval__"], math.log(2.0), rtol=1e-15)


# -- a fixed quantile (brms's bf(quantile = p)) and a fixed Frechet nu --------------------------------------------------------
def test_a_fixed_quantile_is_no_parameter():
    df = frame()
    m = build("asym_laplace_fixed")
    assert m.quantile == 0.3 and m.family_spec.fixed_parameters and not m.family_spec.extra_parameters
    assert "quantile" not in m.names and {w.name for w in m.pymc_model.value_vars} == {"Intercept", "x", "sigma_log__"}
    assert get_family("asym_laplace").with_fixed("quantile", 0.3) is m.family_spec and ab.families.family_of(m) is m.family_spec
    assert "  quantile = 0.3 (fixed, not sampled; brms's bf(quantile = 0.3))" in m.describe()
    learned, other = build("asym_laplace"), ab.asym_laplace("y ~ x", df, m.priors, quantile=0.7)
    identities = {json.dumps(model.target_identity, sort_keys=True) for model in (m, learned, other)}
    assert len(identities) == 3 and ab.asym_laplace("y ~ x", df, m.priors, quantile=0.3).target_identity == m.target_identity
    with pytest.raises(ab.PriorError, match=r"the asym_laplace family's quantile is fixed in this model \(quantile=...\), not learned"):
        ab.asym_laplace("y ~ x", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                                               quantile=ab.Beta(2, 2)), quantile=0.3)
    for bad in (0.0, 1.0, -0.2, np.nan, True):
        with pytest.raises(ValueError, match=r"asym_laplace's fixed quantile must be a finite number strictly between 0 and 1"):
            ab.asym_laplace("y ~ x", df, m.priors, quantile=bad)
    with pytest.raises(ValueError, match="quantile= fixes the asym_laplace family's quantile; the gaussian family has none"):
        ab.Model("y ~ x", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)), quantile=0.3)
    pri = ab.default_priors("y ~ x", df, family="asym_laplace", quantile=0.3)
    assert pri.quantile is None and [r["class"] for r in ab.asym_laplace("y ~ x", df, pri, quantile=0.3).prior_table()
                                     if r["prior"]] == ["Intercept", "sigma"]
    with pytest.raises(ValueError, match="quantile= fixes the asym_laplace family's quantile in a univariate linear model"):
        ab.default_priors(["y ~ a * x", "a ~ 1"], df, family="asym_laplace", quantile=0.3, nl=True)


def test_frechet_nu_learned_or_fixed():
    df = frame()
    learned, fixed = build("frechet"), build("frechet_fixed")
    assert learned.nu is None and learned.learned_parameters == {"nu": 1.0} and "nu_interval__" in learned.describe()
    assert fixed.nu == 3.5 and "nu" not in {w.name for w in fixed.pymc_model.value_vars} and "  nu = 3.5 (fixed, not sampled)" in fixed.describe()
    pl, pf = point(learned, "frechet"), point(fixed, "frechet_fixed")
    np.testing.assert_allclose(rows_logp(learned, pl), rows_logp(fixed, pf), rtol=1e-14)   # the same likelihood at nu = 3.5
    pmm = learned.pymc_model                                     # nu's prior: gamma(2, 0.1) truncated to nu > 1, as brms's lprior
    got = float(pmm.compile_logp(vars=[pmm["nu"]], jacobian=False)(pl))
    np.testing.assert_allclose(got, stats.gamma(2, scale=10).logpdf(3.5) - stats.gamma(2, scale=10).logsf(1.0), rtol=1e-12)
    for bad in (1.0, 0.5, np.inf):
        with pytest.raises(ValueError, match=r"frechet requires a fixed finite scalar nu > 1"):
            ab.frechet("ypos ~ x", df, fixed.priors, nu=bad)
    assert ab.default_priors("ypos ~ x", df, family="frechet", nu=3.5).nu is None
    assert ab.default_priors("ypos ~ x", df, family="frechet").nu == ab.Gamma(2, 0.1)


# -- every other construct is refused, naming it (D112) -----------------------------------------------------------------------
def _lhs(family, addition):
    return f"{RESPONSE[family]} | {addition}"


DPAR = {"inverse_gaussian": "shape", "von_mises": "kappa", "asym_laplace": "quantile", "frechet": "nu", "gen_extreme_value": "xi"}
REFUSALS = [
    ("distributional", lambda f: [f"{RESPONSE[f]} ~ x", f"{DPAR[f]} ~ w"], r"brms's {d} ~ \.\.\. is not offered here"),
    ("sigma", lambda f: [f"{RESPONSE[f]} ~ x", "sigma ~ w"], r"(brms's sigma ~ \.\.\. is not offered here|the {f} family has no parameter 'sigma')"),
    ("weights", lambda f: f"{_lhs(f, 'weights(w)')} ~ x", r"'weights\(w\)' is not available for the {f} family"),
    ("cens", lambda f: f"{_lhs(f, 'cens(c)')} ~ x", r"'cens\(c\)' is not available for the {f} family"),
    ("trunc", lambda f: f"{_lhs(f, 'trunc(lb = 0)')} ~ x", r"'trunc\(lb = 0\)' is not available for the {f} family"),
    ("mo", lambda f: f"{RESPONSE[f]} ~ mo(ord)", r"the monotonic term 'mo\(ord\)' is not available for the {f} family"),
    ("me", lambda f: f"{RESPONSE[f]} ~ me(x, sdx)", r"the measurement-error term 'me\(x, sdx\)' is not available for the {f} family"),
    ("mi", lambda f: f"{RESPONSE[f]} ~ mi(xm)", r"the missing-value term 'mi\(xm\)' is not available for the {f} family"),
    ("response_mi", lambda f: f"{_lhs(f, 'mi()')} ~ x", r"'mi\(\)' is not available for the {f} family"),
    ("ar", lambda f: f"{RESPONSE[f]} ~ x + ar(p = 1)", r"the autocorrelation term 'ar\(p = 1\)' is not available for the {f} family"),
    ("mvbind", lambda f: f"mvbind({RESPONSE[f]}, w) ~ x", r"a multivariate {f} model is not offered"),
    ("mm", lambda f: f"{RESPONSE[f]} ~ x + (1 | mm(g, h))", r"the multi-membership term '\(1 \| mm\(g, h\)\)' is not available for the {f} family"),
    ("gr_by", lambda f: f"{RESPONSE[f]} ~ x + (1 | gr(g, by = h))", r"the term with gr\(by = \) .* is not available for the {f} family"),
    ("cs", lambda f: f"{RESPONSE[f]} ~ cs(x)", r"brms's category-specific effect cs\(\) is not supported for the {f} family"),
]


@pytest.mark.parametrize("family", NEW)
@pytest.mark.parametrize("construct", [r[0] for r in REFUSALS])
def test_other_constructs_are_refused_by_name(family, construct):
    _, make, needle = next(r for r in REFUSALS if r[0] == construct)
    df = frame().assign(c=0, ord=np.arange(24) % 4, sdx=0.3, xm=np.where(np.arange(24) % 5 == 0, np.nan, 0.5))
    with pytest.raises((ab.FormulaError, ab.PriorError, ValueError), match=needle.format(f=family, d=DPAR[family])):
        getattr(ab, family)(make(family), df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[family]))


@pytest.mark.parametrize("family", NEW)
def test_mixtures_nonlinear_models_and_shrinkage_priors_are_refused(family):
    df, y = frame(), RESPONSE[family]
    comp = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **OWN[family])
    with pytest.raises(ValueError, match=rf"a {family} mixture component is not offered.*D52"):
        ab.mixture(f"{y} ~ x", df, ab.Priors(components=[comp, comp], theta=ab.Dirichlet(2)), families=[family, family])
    with pytest.raises(ab.FormulaError, match=rf"the {family} family is not offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel([f"{y} ~ a * x", "a ~ 1"], df, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, **OWN[family]), family=family)
    with pytest.raises(ab.PriorError, match=rf"Horseshoe\(\) on b is offered for the Gaussian family only.*the {family} family is refused"):
        getattr(ab, family)(f"{y} ~ x", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Horseshoe(3), **OWN[family]))


def test_responses_outside_the_support_are_refused():
    df = frame()
    pri = {f: ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[f]) for f in NEW}
    for angle in (-3.2, 3.15, np.inf):
        with pytest.raises(ab.FormulaError, match=r"response 'yang' has a (value outside \[-pi, pi\]|non-finite value)"):
            ab.von_mises("yang ~ x", df.assign(yang=np.r_[angle, df.yang.to_numpy()[1:]]), pri["von_mises"])
    assert ab.von_mises("yang ~ x", df.assign(yang=np.r_[math.pi, -math.pi, df.yang.to_numpy()[2:]]), pri["von_mises"]).design.n == 24
    for family in ("inverse_gaussian", "frechet"):
        with pytest.raises(ab.FormulaError, match=rf"response 'ypos' has a non-positive value .*; the {family.replace('inverse_gaussian', 'inverse.gaussian')} family"):
            getattr(ab, family)("ypos ~ x", df.assign(ypos=np.r_[0.0, df.ypos.to_numpy()[1:]]), pri[family])
    for family in ("asym_laplace", "gen_extreme_value"):
        with pytest.raises(ab.FormulaError, match=rf"response 'y' has a non-finite value; the {family} family needs finite responses"):
            getattr(ab, family)("y ~ x", df.assign(y=np.r_[np.inf, df.y.to_numpy()[1:]]), pri[family])


# -- the term model composes with the new likelihoods ---------------------------------------------------------------------
@pytest.mark.parametrize("key", KEYS)
def test_slopes_categorical_interactions_transforms_offset_and_gp_compose(key):
    """Correlated varying slopes, a combined grouping factor g:h, a declared categorical predictor and its interaction, a
    transform, an offset and a Gaussian process all reach the likelihood through the linear predictor: the observed rows'
    density at a point is SciPy's at the graph's linear predictor (the GEV's xi derived from it)."""
    family, kwargs = KEYS[key]
    y = RESPONSE[family]
    extra = dict(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "g:h": ab.HalfNormal(0.5)}, cor={"g": ab.LKJ(2)},
                 sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[key], **extra)
    df = frame().assign(off=frame().off + (4.0 if key == "inverse_gaussian" else 0.0))   # eta > 0 under 1/mu^2
    m = getattr(ab, family)(f"{y} ~ x * z + I(w^2) + offset(off) + gp(w) + (1 + x | g) + (1 | g:h)", df, priors,
                            categorical={"z": "p"}, **kwargs)
    pmm = m.pymc_model
    rng = np.random.default_rng(7)
    p = {k: v + 0.2 * rng.standard_normal(np.shape(v)) for k, v in pmm.initial_point().items()}
    (parent,) = graph(m, p, [m.names["linpred"]])
    values = {"shape": math.exp(p["alpha_log__"])} if "alpha_log__" in p else {}
    values.update({k: math.exp(p[f"{k}_log__"]) for k in ("sigma", "kappa") if f"{k}_log__" in p})
    if "quantile_logodds__" in p:
        values["quantile"] = float(expit(p["quantile_logodds__"]))
    if "nu_interval__" in p:
        values["nu"] = 1.0 + math.exp(p["nu_interval__"])
    if "tmp_xi" in p:
        values["tmp_xi"] = float(p["tmp_xi"])
    ours, want = rows_logp(m, p), scipy_distribution(key, parent.ravel(), values, df[y].to_numpy()).logpdf(df[y].to_numpy())
    finite = np.isfinite(want)                                   # SciPy's Frechet underflows to -inf far in its lower tail
    np.testing.assert_allclose(ours[finite], want[finite], rtol=1e-11)
    assert np.isfinite(ours).all() and (ours[~finite] < -700).all()
    assert m.spec.gp_terms and m.design.offset is not None and "g" in m.correlation_names and {"x:z", "I(w^2)"} <= set(m.spec.terms)


# -- predictions ------------------------------------------------------------------------------------------------------------
def _posterior(m, key, n_draw=3):
    """Synthetic posterior draws: one chain, the same values in every draw but the intercept (the GEV's xi as the graph derives
    it at the first draw's linear predictor, as a fit's posterior holds it)."""
    post = {m.names["intercept"]: np.linspace(0.2, 0.6, n_draw)[None], "x": np.full((1, n_draw), 0.3)}
    v = natural(m, VALUES[key])
    for name, value in v.items():
        post[m.names[name]] = np.full((1, n_draw), value)
    if family_of_key(key) == "gen_extreme_value":
        groups = dict(sd=0.5, z=np.zeros(m.design.J)) if m.spec.groups else {}
        (xi,) = graph(m, m.value_point(intercept=0.2 + 0.3 * frame().x.mean(), b=[0.3], **groups, **v), ["xi"])
        post["xi"] = np.full((1, n_draw), float(xi))
    return az.from_dict(posterior=post)


@pytest.mark.parametrize("key", KEYS)
def test_predictions_by_hand_and_response_draws(key):
    m = build(key)
    v = VALUES[key]
    idata = _posterior(m, key)
    nd = pd.DataFrame({"x": [-1.0, 0.0, 2.0], "off": [1.5, 1.2, 1.8]})
    eta = np.linspace(0.2, 0.6, 3)[:, None] + 0.3 * nd.x.to_numpy()[None] + (nd.off.to_numpy()[None] if key == "inverse_gaussian" else 0)
    lin = m.predict(idata, newdata=nd, kind="linear")["eta"].to_numpy()[0]
    np.testing.assert_allclose(lin, eta, rtol=1e-14)
    mean = m.predict(idata, newdata=nd, kind="mean")[m.names["mean"]].to_numpy()[0]
    if family_of_key(key) == "gen_extreme_value":
        xi = float(idata.posterior["xi"].to_numpy()[0, 0])
        want = eta + v["sigma"] * (math.gamma(1 - xi) - 1) / xi
    else:
        want = brms_mean(key, eta, v)
    np.testing.assert_allclose(mean, want, rtol=1e-12)
    big = pd.DataFrame({"x": np.zeros(20000), "off": np.full(20000, 1.5)})
    draws = m.predict(_posterior(m, key, 1), newdata=big, kind="response", seed=5)["y"].to_numpy().ravel()
    eta0 = 0.2 + (1.5 if key == "inverse_gaussian" else 0.0)
    if family_of_key(key) == "gen_extreme_value":
        xi = float(_posterior(m, key, 1).posterior["xi"].to_numpy()[0, 0])
        ref = stats.genextreme(-xi, loc=eta0, scale=v["sigma"])
    else:
        ref = scipy_distribution(key, eta0, v)
    assert abs(np.mean(draws <= ref.median()) - 0.5) < 4.5 * 0.5 / math.sqrt(draws.size)
    for q in (0.1, 0.9):
        assert abs(np.mean(draws <= ref.ppf(q)) - q) < 4.5 * math.sqrt(q * (1 - q) / draws.size)
    if family_of_key(key) == "von_mises":
        assert (np.abs(draws) <= math.pi).all()
    if family_of_key(key) in ("inverse_gaussian", "frechet"):
        assert (draws > 0).all()


@pytest.mark.parametrize("key", KEYS)
def test_prior_predictive(key):
    family = family_of_key(key)
    tight = {"gen_extreme_value": dict(xi=ab.Normal(0, 0.3), sigma=ab.HalfNormal(0.5)),       # 1/mu^2: eta = a + x b + off > 0
             "inverse_gaussian": dict(shape=ab.Gamma(20, 4), intercept=ab.Normal(0, 0.1), b=ab.Normal(0, 0.1))}
    m = build(key, formula_of(key, "x + (1 | g)"), **{"sd": ab.HalfNormal(0.3 if key != "inverse_gaussian" else 0.1), **tight.get(key, {})})
    out = m.prior_predictive(draws=30, seed=3)
    y = out.prior_predictive[RESPONSE[family]].values
    assert y.shape == (1, 30, 24) and np.isfinite(y).all()
    if family in ("inverse_gaussian", "frechet"):
        assert (y > 0).all()
    if family == "von_mises":
        assert (np.abs(y) <= math.pi).all()


# -- brms's default priors (A07) ----------------------------------------------------------------------------------------
# brms 2.23.0's default_prior() of <response> ~ x + (1 | g) on frame() and wide_frame() (written to CSV at 17 digits and read by
# R 4.5.2), its class rows with a prior, as (class, prior, lb, ub) in brms's order. The inverse Gaussian's 1/mu^2 link reads
# 1 / y^2 (wide_frame: median 20.1, MAD 26.4) and subtracts the offset's mean from the location; the log link and the Frechet
# read log(y); the asymmetric Laplace and GEV read y; the von Mises Intercept is student_t(1, 0, 1) and its sd's scale 2.5
# whatever the data (tan_half); kappa, quantile, nu and xi have constant defaults, and a fixed quantile has no row
BRMS_CLASS_ROWS = {
    "inverse_gaussian": [("Intercept", "student_t(3, -1.10400625855724, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                         ("shape", "gamma(0.01, 0.01)", "0", "")],
    "inverse_gaussian_log": [("Intercept", "student_t(3, 0.4, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                             ("shape", "gamma(0.01, 0.01)", "0", "")],
    "von_mises": [("Intercept", "student_t(1, 0, 1)", "", ""), ("kappa", "gamma(2, 0.01)", "0", ""), ("sd", "student_t(3, 0, 2.5)", "0", "")],
    "asym_laplace": [("Intercept", "student_t(3, 0.3, 2.5)", "", ""), ("quantile", "beta(1, 1)", "0", "1"),
                     ("sd", "student_t(3, 0, 2.5)", "0", ""), ("sigma", "student_t(3, 0, 2.5)", "0", "")],
    "asym_laplace_fixed": [("Intercept", "student_t(3, 0.3, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                           ("sigma", "student_t(3, 0, 2.5)", "0", "")],
    "frechet": [("Intercept", "student_t(3, 0.4, 2.5)", "", ""), ("nu", "gamma(2, 0.1)", "1", ""), ("sd", "student_t(3, 0, 2.5)", "0", "")],
    "gen_extreme_value": [("Intercept", "student_t(3, 0.3, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                          ("sigma", "student_t(3, 0, 2.5)", "0", ""), ("xi", "normal(0, 2.5)", "", "")],
}
BRMS_WIDE_ROWS = {
    "inverse_gaussian": [("Intercept", "student_t(3, 14.5721233333333, 26.4)", "", ""), ("sd", "student_t(3, 0, 26.4)", "0", ""),
                         ("shape", "gamma(0.01, 0.01)", "0", "")],
    "inverse_gaussian_log": [("Intercept", "student_t(3, -1.5, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                             ("shape", "gamma(0.01, 0.01)", "0", "")],
    "von_mises": BRMS_CLASS_ROWS["von_mises"],
    "asym_laplace": [("Intercept", "student_t(3, 10.5, 8.2)", "", ""), ("quantile", "beta(1, 1)", "0", "1"),
                     ("sd", "student_t(3, 0, 8.2)", "0", ""), ("sigma", "student_t(3, 0, 8.2)", "0", "")],
    "asym_laplace_fixed": [("Intercept", "student_t(3, 10.5, 8.2)", "", ""), ("sd", "student_t(3, 0, 8.2)", "0", ""),
                           ("sigma", "student_t(3, 0, 8.2)", "0", "")],
    "frechet": [("Intercept", "student_t(3, -1.5, 2.5)", "", ""), ("nu", "gamma(2, 0.1)", "1", ""), ("sd", "student_t(3, 0, 2.5)", "0", "")],
    "gen_extreme_value": [("Intercept", "student_t(3, 10.5, 8.2)", "", ""), ("sd", "student_t(3, 0, 8.2)", "0", ""),
                          ("sigma", "student_t(3, 0, 8.2)", "0", ""), ("xi", "normal(0, 2.5)", "", "")],
}


def wide_frame(n=30, seed=21):
    """Responses on wide scales: 1 / ypos^2 with a MAD above 2.5, y with a MAD of 8.2; angles near +-pi."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "g": [f"l{k}" for k in np.arange(n) % 3],
                         "y": np.round(rng.normal(12.0, 9.0, n), 4), "ypos": np.round(np.exp(rng.normal(-1.5, 1.3, n)), 5),
                         "yang": np.round(rng.uniform(2.6, 3.1, n) * np.where(np.arange(n) % 2, 1, -1), 4),
                         "off": np.round(rng.uniform(5.0, 6.0, n), 4)})


@pytest.mark.parametrize("key", [k for k in KEYS if k != "frechet_fixed"])
@pytest.mark.parametrize("which", ["frame", "wide"])
def test_default_priors_build_the_model_brms_describes(key, which):
    family, kwargs = KEYS[key]
    df, want = (frame(), BRMS_CLASS_ROWS) if which == "frame" else (wide_frame(), BRMS_WIDE_ROWS)
    formula = f"{RESPONSE[family]} ~ x" + (" + offset(off)" if key == "inverse_gaussian" else "") + " + (1 | g)"
    pri = ab.default_priors(formula, df, family=family, **kwargs)
    m = getattr(ab, family)(formula, df, pri, **kwargs)
    rows = [(r["class"], r["prior"], r["lb"], r["ub"]) for r in m.prior_table() if r["coef"] == "" and r["group"] == "" and r["prior"]]
    assert rows == want[key]
    assert all(r["source"] == "default" for r in m.prior_table())
    for e in m.family_spec.extra_parameters:
        assert e.name in m.default_prior_labels and getattr(pri, e.name).source == "default"


# -- describe(), the hash seed, the target identity and the fit record ----------------------------------------------------
@pytest.mark.parametrize("key", KEYS)
def test_describe(key):
    m = build(key)
    text = m.describe()
    spec = m.family_spec
    assert f"family    : {spec.name}, {spec.link} link" in text and f"AutoBayes {spec.tag} model" in text
    for e in spec.extra_parameters:
        assert f"  {e.coordinate:<24} {e.coordinate_meaning}" in text
        if spec.scale_parameter == "sigma":
            assert text.index(f"  {'sigma_log__':<24} log(sigma)") < text.index(f"  {e.coordinate:<24}")
    family, kwargs = KEYS[key]
    assert ab.Model(formula_of(key), frame(), m.priors, family=family, **kwargs).describe() == text


SET_ORDER = 'import json; print(json.dumps({"order": list(set(("mu", "sigma", "tmp_xi")) - {"mu"})}))'   # Bambi's auxiliary_parameters
BUILD = """
import json
from tests.test_continuous_families_e02 import build
m = build("gen_extreme_value")
print(json.dumps({"describe": m.describe(), "identity": m.target_identity, "value_vars": [v.name for v in m.pymc_model.value_vars]}))
"""


def test_describe_does_not_depend_on_the_hash_seed():
    """The set of the auxiliary parameters' names that Bambi iterates follows ``PYTHONHASHSEED``; the family the model hands
    Bambi lists sigma and tmp_xi in its likelihood's order (F09, ``likelihoods.ordered_family``), so PyMC's order of
    ``sigma_log__`` and ``tmp_xi``, ``describe()`` (sigma, then tmp_xi) and the target identity do not."""
    seeds = {}
    for seed in range(32):
        seeds.setdefault(tuple(_fresh(seed, SET_ORDER)["order"]), seed)
        if len(seeds) == 2:
            break
    assert len(seeds) == 2
    first, second = (_fresh(seed, BUILD) for seed in seeds.values())
    assert first["value_vars"] == second["value_vars"]
    assert first["value_vars"].index("sigma_log__") < first["value_vars"].index("tmp_xi")
    assert first["describe"] == second["describe"] and first["identity"] == second["identity"]


@pytest.mark.parametrize("key", ["von_mises", "asym_laplace", "gen_extreme_value", "asym_laplace_fixed"])
def test_target_identity_and_fit_record(key, monkeypatch):
    m = build(key)
    assert build(key).target_identity == m.target_identity
    if m.family_spec.extra_parameters:
        (extra,) = m.family_spec.extra_parameters
        other = {"kappa": ab.Gamma(3, 0.01), "quantile": ab.Beta(3, 3), "xi": ab.Normal(0, 0.4)}[extra.name]
        assert build(key, **{extra.name: other}).target_identity != m.target_identity
    post = _posterior(m, key, n_draw=4)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert {e.name for e in m.family_spec.extra_parameters} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == family_of_key(key)
    for e in m.family_spec.extra_parameters:
        assert rec["model"]["priors"][e.name] == str(OWN[key][e.name])
    if key == "asym_laplace_fixed":
        assert rec["model"]["family_parameters"] == {"quantile": 0.3} and "quantile" not in rec["model"]["priors"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


@pytest.mark.parametrize("key", KEYS)
def test_log_lik_is_the_models_own_pointwise_density(key):
    m = build(key, formula_of(key, "x + (1 | g)"), sd=ab.HalfNormal(0.5))
    v = natural(m, VALUES[key])
    post = _posterior(m, key, n_draw=2)
    post.posterior["1|g_sigma"] = (("chain", "draw"), np.full((1, 2), 0.5))
    effects = np.array([0.3, -0.2, 0.1, 0.0])
    coord = m.group_names["g"]["coordinate"]
    post.posterior["1|g"] = (("chain", "draw", coord), np.tile(effects, (1, 2, 1)))
    post.posterior = post.posterior.assign_coords({coord: list(m.design.levels)})
    x = frame().x.to_numpy()
    points = [m.value_point(intercept=a + 0.3 * x.mean(), b=[0.3], sd=0.5, z=effects / 0.5, **v) for a in (0.2, 0.6)]
    if "xi" in m.names:                                          # the GEV's xi per draw, as the graph derives it at that draw
        post.posterior["xi"] = (("chain", "draw"), np.array([[float(graph(m, p, ["xi"])[0]) for p in points]]))
    ll = ab.log_lik(m, post).to_numpy()[0]
    for d, p in enumerate(points):
        np.testing.assert_allclose(ll[d], rows_logp(m, p), rtol=1e-12)


@pytest.mark.parametrize("case", ["inverse_gaussian_varint", "inverse_gaussian_log", "von_mises_varint", "asym_laplace_varint",
                                  "asym_laplace_quantile", "frechet_varint", "gen_extreme_value_varint"])
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
