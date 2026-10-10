"""Zero-inflated and hurdle Poisson counts (brms-parity B06): the stable row densities against a 50-digit reference on
both paths (PyTensor's gradient and JAX's autodiff of the jaxified graph, what NumPyro differentiates), the model's
density and gradient by hand, the zi / hu priors and their refusals, ``value_point``, by-hand predictions and the
zero-truncated sampler, ``describe()``, prior-predictive draws, brms's default priors and the fit record. No MCMC; the
brms comparison is the oracle cases ``zero_inflated_poisson_varint`` and ``hurdle_poisson_varint``."""
import json
import math
import warnings
from decimal import Decimal, localcontext
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import logit
from scipy.stats import beta as beta_dist, halfnorm, norm, poisson

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import get_family


def data():
    return pd.DataFrame({"x": [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7], "cnt": [0, 2, 5, 0, 1, 0, 3, 0],
                         "g": ["a", "a", "b", "b", "c", "c", "a", "b"], "e": [1.5, 2.0, 0.5, 1.0, 3.0, 1.2, 0.8, 2.2]})


def priors(group=False, **extra):
    return ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **({"sd": ab.HalfNormal(1)} if group else {}), **extra)


ZIP, HURDLE = ("zero_inflated_poisson", "zi"), ("hurdle_poisson", "hu")


def build(which, formula="cnt ~ x", group=False, prob=ab.Beta(2, 2), frame=None, **kwargs):
    family, par = which
    return getattr(ab, family)(formula, data() if frame is None else frame, priors(group, **{par: prob}), **kwargs)


# -- the row densities: a 50-digit reference ----------------------------------------------------------------------------
def _log_factorial(y):
    return sum((Decimal(k).ln() for k in range(2, y + 1)), Decimal(0))


def _one_minus_exp_neg(mu):
    """1 - exp(-mu) at 50 digits, by its series where the subtraction would cancel."""
    if mu < Decimal("1e-10"):
        return mu * (1 - mu / 2 + mu * mu / 6 - mu ** 3 / 24)
    return 1 - (-mu).exp()


def exact_row(family, y, eta, p):
    """(log density, d/d eta, d/d p) of one row, brms's form at 50 digits (zi or hu is p)."""
    with localcontext() as ctx:
        ctx.prec = 50
        e, p = Decimal(float(eta)), Decimal(float(p))
        mu = e.exp()
        one_minus_q = _one_minus_exp_neg(mu)                      # P_Poisson(y > 0) = 1 - exp(-mu)
        q = 1 - one_minus_q
        if family == "zero_inflated_poisson" and y == 0:
            s = 1 - (1 - p) * one_minus_q                         # zi + (1 - zi) exp(-mu)
            return float(s.ln()), float(-(1 - p) * q * mu / s), float(one_minus_q / s)
        if family == "hurdle_poisson" and y == 0:
            return float(p.ln()), 0.0, float(1 / p)
        value = (1 - p).ln() + y * e - mu - _log_factorial(y)
        d_eta = y - mu
        if family == "hurdle_poisson":                            # minus log P(y > 0), the zero truncation
            value -= one_minus_q.ln()
            d_eta -= q * mu / one_minus_q
        return float(value), float(d_eta), float(-1 / (1 - p))


ETAS = [-1000.0, -745.5, -708.0, -100.0, -40.0, -20.0 - 1e-9, -20.0, -19.999, -3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 3.7, 10.0,
        20.0, 40.0, 41.0, 100.0, 699.999, 700.0, 700.001, 709.7, 710.0, 1000.0]
PROBS = [1e-6, 0.3, 0.97]
LOGP = {"zero_inflated_poisson": lk.zero_inflated_poisson_logp, "hurdle_poisson": lk.hurdle_poisson_logp}


def _rows(family, mode, y, eta, p):
    """Per-row value, d/d eta and d/d p on one path: PyTensor's own gradient graph, or JAX's autodiff of the jaxified
    graph (the path NumPyro samples on)."""
    yv, ev, pv = pt.lvector("y"), pt.dvector("eta"), pt.dvector("p")
    rows = LOGP[family](yv, ev, pv)
    if mode == "pytensor":
        f = pytensor.function([yv, ev, pv], [rows, pt.grad(rows.sum(), ev), pt.grad(rows.sum(), pv)])
        return [np.asarray(a, dtype=float) for a in f(y, eta, p)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[yv, ev, pv], outputs=[rows])
    value = jf(y, eta, p)[0]
    d_eta = jax.grad(lambda e: jf(y, e, p)[0].sum())(eta)
    d_p = jax.grad(lambda q: jf(y, eta, q)[0].sum())(p)
    return [np.asarray(a, dtype=float) for a in (value, d_eta, d_p)]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("family", ["zero_inflated_poisson", "hurdle_poisson"])
def test_row_densities_are_exact_and_never_nan_on_both_paths(family, mode):
    grid = [(y, e, p) for y in (0, 1, 4) for e in ETAS for p in PROBS]
    y = np.array([r[0] for r in grid], dtype=np.int64)
    eta = np.array([r[1] for r in grid])
    p = np.array([r[2] for r in grid])
    value, d_eta, d_p = _rows(family, mode, y, eta, p)
    assert not np.isnan(value).any() and not np.isnan(d_eta).any() and not np.isnan(d_p).any()
    for i, (yi, ei, pi) in enumerate(grid):
        want = exact_row(family, yi, ei, pi)
        if yi > 0 and ei > 709.78:                                # the true log density, about -exp(eta), is below -1.8e308
            assert want[0] == -np.inf and value[i] == -np.inf and d_eta[i] == -np.inf, (yi, ei, pi)
            continue
        for got, ref in zip((value[i], d_eta[i], d_p[i]), want):
            assert np.isfinite(got) and abs(got - ref) <= 1e-12 * max(1.0, abs(ref)), (family, mode, yi, ei, pi, got, ref)


def test_where_brms_forms_break_the_stable_ones_do_not():
    """brms's zero_inflated_poisson y = 0 row has -exp(eta) = -inf beyond eta = 709.78 (its gradient is 0 * inf there);
    its hurdle's log1m_exp(-exp(eta)) is log(0) below eta of about -745, a +inf row. Here both rows are finite, and a
    negative count has no mass."""
    value, d_eta, _ = _rows("zero_inflated_poisson", "pytensor", np.array([0]), np.array([800.0]), np.array([0.25]))
    assert value[0] == math.log(0.25) and d_eta[0] == 0.0
    value, d_eta, _ = _rows("hurdle_poisson", "pytensor", np.array([2]), np.array([-800.0]), np.array([0.25]))
    assert abs(value[0] - (math.log(0.75) - 800.0 - math.log(2))) < 1e-12 and d_eta[0] == 1.0     # (y - 1) eta - log y!
    for family in LOGP:
        value, _, _ = _rows(family, "pytensor", np.array([-1]), np.array([0.3]), np.array([0.2]))
        assert value[0] == -np.inf


# -- the zero-truncated sampler and the likelihood objects ---------------------------------------------------------------
@pytest.mark.parametrize("mu", [1e-3, 0.5, 2.0, 20.0])
def test_zero_truncated_poisson_draws_have_the_truncated_distribution(mu):
    n = 40000
    y = lk.zero_truncated_poisson_draws(np.full(n, mu), np.random.default_rng(5))
    assert y.min() >= 1 and y.dtype.kind == "i"
    mean = mu / -math.expm1(-mu)
    var = mu * (1 + mu) / -math.expm1(-mu) - mean ** 2
    assert abs(y.mean() - mean) < 4.5 * math.sqrt(var / n)
    for k in (1, 2, 3):                                           # the pmf itself, P(k) = Poisson(k | mu) / (1 - exp(-mu))
        pk = poisson.pmf(k, mu) / -math.expm1(-mu)
        assert abs(np.mean(y == k) - pk) < 4.5 * math.sqrt(pk * (1 - pk) / n) + 1e-12
    assert np.all(lk.zero_truncated_poisson_draws(np.zeros(5), np.random.default_rng(1)) == 1)    # mu underflowed: 1


@pytest.mark.parametrize("family", ["zero_inflated_poisson", "hurdle_poisson"])
def test_likelihood_objects_draw_their_distribution(family):
    import pymc as pm

    eta, p = np.array([-1.0, 0.4, 1.5]), 0.3
    if family == "zero_inflated_poisson":
        dist, zero_mass = lk.ZeroInflatedPoissonLog.dist(log_mu=eta, zi=p), p + (1 - p) * np.exp(-np.exp(eta))
    else:
        dist, zero_mass = lk.HurdlePoissonLog.dist(log_mu=eta, hu=p), np.full(3, p)
    draws = pm.draw(dist, draws=20000, random_seed=3)
    assert draws.shape == (20000, 3) and draws.dtype == np.int64 and draws.min() >= 0
    assert np.all(np.abs((draws == 0).mean(0) - zero_mass) < 4.5 * np.sqrt(zero_mass * (1 - zero_mass) / 20000))
    got = pytensor.function([], pm.logp(dist, np.array([0, 2, 5])))()
    np.testing.assert_allclose(got, [exact_row(family, y, e, p)[0] for y, e in zip((0, 2, 5), eta)], rtol=1e-13)


# -- the model: density and gradient by hand ----------------------------------------------------------------------------
def _by_hand(family, y, eta, p):
    if family == "zero_inflated_poisson":
        return np.where(y == 0, np.log(p + (1 - p) * np.exp(-np.exp(eta))), np.log1p(-p) + poisson.logpmf(y, np.exp(eta))).sum()
    return np.where(y == 0, np.log(p), np.log1p(-p) + poisson.logpmf(y, np.exp(eta)) - np.log(-np.expm1(-np.exp(eta)))).sum()


def _log_mu(m, point):
    pmm = m.pymc_model
    return np.ravel(pmm.compile_fn(pmm.replace_rvs_by_values([pmm["log_mu"]]), inputs=pmm.value_vars, on_unused_input="ignore")(point))


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_hierarchical_density_and_gradient_by_hand_both_paths(which):
    family, par = which
    df = data()
    m = build(which, "cnt ~ x + (1 | g)", group=True, prob=ab.Beta(2, 3))
    nat = dict(intercept=0.3, b=[-0.4], sd=0.5, z=np.array([0.2, -0.4, 0.1]))
    p = m.value_point(**nat, **{par: 0.35})
    np.testing.assert_allclose(p[f"{par}_logodds__"], logit(0.35), rtol=1e-15)
    x, y = df.x.to_numpy(), df.cnt.to_numpy()
    eta = 0.3 - 0.4 * (x - x.mean()) + 0.5 * nat["z"][pd.Categorical(df.g).codes]
    lik = _by_hand(family, y, eta, 0.35)
    prior = (norm.logpdf(0.3, 0, 2) + norm.logpdf(-0.4, 0, 1) + halfnorm.logpdf(0.5) + norm.logpdf(nat["z"]).sum()
             + beta_dist.logpdf(0.35, 2, 3))
    jac = np.log(0.5) + np.log(0.35) + np.log1p(-0.35)                 # sd on the log scale, zi / hu on the logit scale
    pmm = m.pymc_model
    np.testing.assert_allclose(pmm.compile_logp()(p), lik + prior + jac, rtol=1e-12)
    np.testing.assert_allclose(pmm.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-12)
    np.testing.assert_allclose(pmm.compile_logp(vars=[pmm[par]], jacobian=False)(p), beta_dist.logpdf(0.35, 2, 3), rtol=1e-14)
    np.testing.assert_allclose(_log_mu(m, p), eta, rtol=1e-14)        # the graph's parent is eta itself
    from pymc.sampling.jax import get_jaxified_logp
    import jax
    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    args = [p[n] for n in names]
    np.testing.assert_allclose(jf(args), lik + prior + jac, rtol=1e-12)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    g_pt = np.asarray(pmm.compile_dlogp()(p), dtype=float)
    np.testing.assert_allclose(g_jax, g_pt, rtol=1e-10, atol=1e-12)
    # d/d logit(p) by hand: the rows' d/dp times p (1 - p), the Beta(2, 3) prior's (1 - p) - 2 p and the Jacobian's 1 - 2 p
    mu, q = np.exp(eta), 0.35
    if family == "zero_inflated_poisson":
        rows = np.where(y == 0, -np.expm1(-mu) / (q + (1 - q) * np.exp(-mu)), -1 / (1 - q))
    else:
        rows = np.where(y == 0, 1 / q, -1 / (1 - q))
    want = rows.sum() * q * (1 - q) + (1 - q) - 2 * q + 1 - 2 * q
    np.testing.assert_allclose(g_pt[names.index(f"{par}_logodds__")], want, rtol=1e-10)


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_other_terms_and_the_offset_compose(which):
    """The term model is shared: categorical predictors, no intercept, independent and correlated slopes and the offset
    all build, and the offset enters log_mu as written."""
    family, par = which
    df = data().assign(z=["p", "q", "p", "q", "p", "q", "p", "q"])
    slopes = {"sd": {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}}
    for formula, extra, kw in (("cnt ~ x + z", {}, {"categorical": {"z": "p"}}), ("cnt ~ 0 + x", {}, {}),
                               ("cnt ~ x + (1 + x || g)", slopes, {}), ("cnt ~ x + (1 + x | g)", {**slopes, "cor": {"g": ab.LKJ(2)}}, {})):
        pri = ab.Priors(intercept=None if formula.startswith("cnt ~ 0") else ab.Normal(0, 2), b=ab.Normal(0, 1), **extra,
                        **{par: ab.Beta(2, 2)})
        m = getattr(ab, family)(formula, df, pri, **kw)
        assert m.family_spec.zero_parameter == par and f"{par}_logodds__" in {v.name for v in m.pymc_model.value_vars}
    m = build(which, "cnt ~ x + offset(log(e))")
    p = m.value_point(intercept=0.2, b=[0.5], **{par: 0.4})
    x = data().x.to_numpy()
    np.testing.assert_allclose(_log_mu(m, p), 0.2 + 0.5 * (x - x.mean()) + np.log(data().e.to_numpy()), rtol=1e-14)
    assert "offset_i" in m.describe() and "The exposure enters through the offset term" in m.describe()


# -- priors and value_point ---------------------------------------------------------------------------------------------
def test_prior_coverage_and_refusals():
    with pytest.raises(ab.PriorError, match=r"the zero_inflated_poisson family needs a prior on the zero-inflation probability zi: "
                                            r"pass zi=..., a prior on \(0, 1\) such as ab.Beta\(1, 1\)"):
        ab.zero_inflated_poisson("cnt ~ x", data(), priors())
    with pytest.raises(ab.PriorError, match="the hurdle_poisson family needs a prior on the hurdle probability hu"):
        ab.hurdle_poisson("cnt ~ x", data(), priors(zi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match=r"the zero_inflated_poisson family has no hurdle probability; do not pass hu. Its "
                                            r"zero-inflation probability is zi: pass zi=... Its intercept and slope priors are on the log scale"):
        ab.zero_inflated_poisson("cnt ~ x", data(), priors(zi=ab.Beta(1, 1), hu=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match=r"the hurdle_poisson family has no residual SD; do not pass sigma. Its hurdle probability is hu"):
        ab.hurdle_poisson("cnt ~ x", data(), priors(hu=ab.Beta(1, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r"the poisson family has no zero-inflation probability; do not pass zi. Its intercept and slope priors are on the log scale"):
        ab.poisson("cnt ~ x", data(), priors(zi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match=r"the gaussian family has no hurdle probability; do not pass hu. Its auxiliary parameter is sigma"):
        ab.gaussian("x ~ cnt", data(), priors(sigma=ab.HalfNormal(1), hu=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'zi' must be a prior on \(0, 1\) \(\['Beta'\]; the zero-inflation probability is "
                                            r"brms's real<lower=0,upper=1> zi\), got HalfNormal"):
        build(ZIP, prob=ab.HalfNormal(1))
    with pytest.raises(ab.PriorError, match="prior for 'hu' must be a Prior, got float"):
        build(HURDLE, prob=0.3)
    for field in ("intercept", "b"):                                   # Beta is for zi and hu only
        with pytest.raises(ab.PriorError, match="must be a location prior"):
            ab.zero_inflated_poisson("cnt ~ x", data(), ab.Priors(**{**dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)),
                                                                     field: ab.Beta(1, 1)}, zi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match="prior for 'sd:g' must be a positive-support prior"):
        ab.zero_inflated_poisson("cnt ~ x + (1 | g)", data(), priors(sd=ab.Beta(1, 1), zi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match="prior for 'nu' must be a positive-support prior"):
        ab.student("x ~ cnt", data(), priors(sigma=ab.HalfNormal(1), nu=ab.Beta(2, 2)))
    with pytest.raises(ab.PriorError, match="alpha must be positive"):
        ab.Beta(0, 1)
    assert ab.Beta(2, 3) == ab.Prior("Beta", {"alpha": 2.0, "beta": 3.0}) and str(ab.Beta(2, 3)) == "Beta(alpha=2, beta=3)"
    from autobayes.priors import brms_prior
    assert brms_prior(ab.Beta(1, 1)) == "beta(1, 1)" and brms_prior(ab.Beta(0.5, 2.25)) == "beta(0.5, 2.25)"


def test_the_beta_prior_reaches_the_graph_in_double_precision():
    """A Python 2.0 would become a float32 constant, and lgamma(2) + lgamma(2) - lgamma(4) would lose 2e-8."""
    m = build(ZIP, prob=ab.Beta(2, 2))
    point = m.value_point(intercept=0.1, b=[0.2], zi=0.734)
    got = float(m.pymc_model.compile_logp(vars=[m.pymc_model["zi"]], jacobian=False)(point))
    assert abs(got - beta_dist.logpdf(0.734, 2, 2)) < 1e-14


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_names_and_value_point(which):
    family, par = which
    other = "hu" if par == "zi" else "zi"
    m = build(which, "cnt ~ x + (1 | g)", group=True)
    assert m.names[par] == par and other not in m.names and "sigma" not in m.names and "shape" not in m.names
    assert m.names["linpred"] == "log_mu" and m.names["mean"] == "mean" and "mu" not in m.names
    assert m.link == "log" and m.family_spec.discrete and m.family_spec.link_scale == "log"
    assert np.array_equal(m.design.y, data().cnt) and m.warnings == []
    p = m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), **{par: 0.25})
    assert set(p) == {v.name for v in m.pymc_model.value_vars}
    with pytest.raises(ValueError, match=rf"{par} \(the .* probability\) is required for the {family} family; {other} does not apply"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3))
    with pytest.raises(ValueError, match="is required"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), **{other: 0.25})
    for bad in (0.0, 1.0, -0.1, np.nan, [0.2, 0.3]):
        with pytest.raises(ValueError, match=f"{par} must be a finite scalar strictly between 0 and 1"):
            m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), **{par: bad})
    with pytest.raises(ValueError, match="sigma, shape and phi do not apply"):
        m.value_point(intercept=0.3, b=[-0.2], sd=0.5, z=np.zeros(3), sigma=1.0, **{par: 0.25})
    pois = ab.poisson("cnt ~ x", data(), priors())
    with pytest.raises(ValueError, match="the poisson family has no zero-inflation or hurdle probability; omit zi and hu"):
        pois.value_point(intercept=0.3, b=[-0.2], **{par: 0.25})
    for column in (par, "log_mu"):                                    # the probability variable and the linear predictor
        with pytest.raises(ab.FormulaError, match="collision"):
            build(which, f"cnt ~ {column}", frame=data().rename(columns={"x": column}))


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_invalid_responses(which):
    for values, message in (([-1, 2], "negative value"), ([0.5, 2], "non-integral"), ([True, False], "boolean"),
                            ([2e6, 1], "exceeds the Poisson input-count limit")):
        with pytest.raises(ab.FormulaError, match=message):
            build(which, "cnt ~ 1", frame=pd.DataFrame({"cnt": values}))
    with pytest.raises(ab.FormulaError, match="binomial"):
        build(which, "cnt | trials(n) ~ x", frame=data().assign(n=10))
    zeros = build(which, "cnt ~ x", frame=data().assign(cnt=0))
    assert len(zeros.warnings) == 1 and "all observed counts are zero" in zeros.warnings[0]


# -- predictions ---------------------------------------------------------------------------------------------------------
def draws(par, a=0.3, p=0.25, group=True):
    post = {"Intercept": np.full((2, 4), a), "x": np.full((2, 4), -0.2), par: np.full((2, 4), p)}
    if group:
        post.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.1, -0.1, 0.3], (2, 4, 3))})
    return az.from_dict(posterior=post, coords={"g__factor_dim": ["a", "b", "c"]}, dims={"1|g": ["g__factor_dim"]})


def _ztp(mu, rng):
    u = rng.uniform(size=mu.shape)
    return 1 + rng.poisson(np.clip(mu + np.log1p(u * np.expm1(-mu)), 0, None))


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_fitted_row_predictions_by_hand(which):
    family, par = which
    m = build(which, "cnt ~ x + (1 | g)", group=True)
    post = draws(par)
    u = np.array([0.1, -0.1, 0.3])[pd.Categorical(data().g).codes]
    eta = np.broadcast_to(0.3 - 0.2 * data().x.to_numpy() + u, (2, 4, 8))
    mu = np.exp(eta)
    lin = m.predict(post, kind="linear")
    np.testing.assert_allclose(lin.eta, eta)
    assert lin.attrs["prediction_scale"] == "linear" and lin.attrs["link"] == "log"
    mean = (1 - 0.25) * mu if family == "zero_inflated_poisson" else (1 - 0.25) * mu / -np.expm1(-mu)
    out = m.predict(post, kind="response", seed=9)
    np.testing.assert_allclose(out["mean"], mean, rtol=1e-14)
    rng = np.random.default_rng(9)
    zero = rng.uniform(size=mu.shape) < 0.25
    counts = rng.poisson(mu) if family == "zero_inflated_poisson" else _ztp(mu, rng)
    np.testing.assert_array_equal(out.y, np.where(zero, 0, counts))
    assert out.y.dtype == np.int64 and set(out.data_vars) == {"mean", "y"}
    nogroup = m.predict(post, kind="mean", include_group_effects=False)
    mu0 = np.exp(np.broadcast_to(0.3 - 0.2 * data().x.to_numpy(), (2, 4, 8)))
    np.testing.assert_allclose(nogroup["mean"], 0.75 * mu0 if family == "zero_inflated_poisson" else 0.75 * mu0 / -np.expm1(-mu0))
    with pytest.raises(ValueError, match="kind"):
        m.predict(post, kind="probability")


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_seeded_new_group_predictions(which, policy):
    family, par = which
    m = build(which, "cnt ~ x + (1 | g)", group=True)
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["new", "new"]})
    rng = np.random.default_rng(8)
    effect = rng.normal(size=(2, 4)) * 0.5 if policy == "population" else np.take([0.1, -0.1, 0.3], rng.integers(0, 3, size=(2, 4)))
    mu = np.exp(0.3 - 0.2 * new.x.to_numpy() + effect[..., None])
    out = m.predict(draws(par), newdata=new, kind="response", new_groups=policy, seed=8)
    zero = rng.uniform(size=mu.shape) < 0.25
    counts = rng.poisson(mu) if family == "zero_inflated_poisson" else _ztp(mu, rng)
    np.testing.assert_array_equal(out.y, np.where(zero, 0, counts))
    assert out.attrs["new_groups"] == policy and list(out.attrs["new_levels"]) == ["new"]
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(draws(par), newdata=new, kind="mean")


def test_prediction_limits():
    zip_m, hur_m = build(ZIP), build(HURDLE)
    with pytest.raises(ValueError, match="overflow"):
        zip_m.predict(draws("zi", a=800.0, group=False), kind="mean")
    with pytest.raises(ValueError, match="overflow"):
        hur_m.predict(draws("hu", a=800.0, group=False), kind="response", seed=1)
    tiny = hur_m.predict(draws("hu", a=-800.0, group=False), kind="response", seed=1)   # mu underflows: E[y | y > 0] -> 1
    np.testing.assert_allclose(tiny["mean"], 0.75)
    assert set(np.unique(tiny.y)) <= {0, 1}
    assert np.isfinite(zip_m.predict(draws("zi", a=40.0, group=False), kind="response", seed=1)["mean"]).all()
    with pytest.raises(ValueError, match="integer random-generator limit"):
        zip_m.predict(draws("zi", a=60.0, group=False), kind="response", seed=1)


# -- describe, identity, prior predictive, defaults, record ---------------------------------------------------------------
@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_describe_identity_and_registry_constructor(which):
    family, par = which
    m = build(which, "cnt ~ x + (1 | g)", group=True)
    text = m.describe()
    tag = "zero-inflated Poisson" if family == "zero_inflated_poisson" else "hurdle Poisson"
    assert text.startswith(f"AutoBayes {tag} model") and f"family    : {family}, log link" in text
    assert f"{par} ~ Beta(alpha=2, beta=2)" in text and f"{par}_logodds__" in text and f"logit({par})" in text
    assert "location coefficients on the log scale" in text and "log_mu = linear predictor" in text
    assert (("P(y_i = 0) = zi + (1 - zi) exp(-mu_i)" in text and "E[y_i] = (1 - zi) mu_i" in text)
            if family == "zero_inflated_poisson" else
            ("P(y_i = 0) = hu" in text and "E[y_i] = (1 - hu) mu_i / (1 - exp(-mu_i))" in text))
    same = ab.Model("cnt ~ x + (1 | g)", data(), priors(True, **{par: ab.Beta(2, 2)}), family=family)
    assert same.describe() == text and same.target_identity == m.target_identity
    assert build(which, "cnt ~ x + (1 | g)", group=True, prob=ab.Beta(1, 1)).target_identity != m.target_identity
    assert ab.poisson("cnt ~ x + (1 | g)", data(), priors(True)).target_identity != m.target_identity
    other = HURDLE if which == ZIP else ZIP
    assert build(other, "cnt ~ x + (1 | g)", group=True).target_identity != m.target_identity


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_prior_predictive(which):
    family, par = which
    m = build(which, "cnt ~ x + (1 | g)", group=True)
    out = m.prior_predictive(draws=40, seed=3)
    assert out.prior[par].shape == (1, 40) and ((out.prior[par].values > 0) & (out.prior[par].values < 1)).all()
    y = out.prior_predictive["cnt"].values
    assert y.shape == (1, 40, 8) and y.dtype.kind == "i" and (y >= 0).all()
    assert out.attrs["autobayes_generation_kind"] == "prior_predictive"


# brms 2.23.0's default_prior() (R 4.5.2, C collation) on these twelve rows, as (class, coef, group, prior, lb, ub); every
# row's source is "default"
DEFAULTS_FRAME = pd.DataFrame({
    "x": [0.12, -1.3, 0.85, 1.7, -0.44, 0.03, -2.1, 0.9, 1.15, -0.6, 0.31, -0.05],
    "e": [1.5, 2.25, 0.8, 3.1, 1.05, 0.6, 2.7, 1.9, 0.45, 1.35, 2.05, 4.2],
    "z": ["p", "q", "r", "p", "q", "r", "p", "q", "r", "p", "q", "r"],
    "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
    "cnt": [0, 3, 0, 1, 7, 2, 0, 0, 12, 4, 1, 0]})
BRMS_DEFAULTS = {
    "zero-inflated, varying intercept": ("cnt ~ x + (1 | g)", "zero_inflated_poisson", {}, [
        ("Intercept", "", "", "student_t(3, 0, 3.1)", "", ""), ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""),
        ("sd", "", "", "student_t(3, 0, 3.1)", "0", ""), ("sd", "", "g", "", "", ""), ("sd", "Intercept", "g", "", "", ""),
        ("zi", "", "", "beta(1, 1)", "0", "1")]),
    "hurdle, offset and categorical": ("cnt ~ x + z + offset(log(e))", "hurdle_poisson", {"categorical": {"z": "p"}}, [
        ("Intercept", "", "", "student_t(3, -0.412687108229976, 3.1)", "", ""), ("b", "", "", "", "", ""),
        ("b", "x", "", "", "", ""), ("b", "zq", "", "", "", ""), ("b", "zr", "", "", "", ""), ("hu", "", "", "beta(1, 1)", "0", "1")]),
    "zero-inflated, no intercept, independent slope": ("cnt ~ 0 + x + (1 + x || g)", "zero_inflated_poisson", {}, [
        ("b", "", "", "", "", ""), ("b", "x", "", "", "", ""), ("sd", "", "", "student_t(3, 0, 3.1)", "0", ""),
        ("sd", "", "g", "", "", ""), ("sd", "Intercept", "g", "", "", ""), ("sd", "x", "g", "", "", ""),
        ("zi", "", "", "beta(1, 1)", "0", "1")]),
}


@pytest.mark.parametrize("label", sorted(BRMS_DEFAULTS))
def test_default_priors_are_brms_default_prior(label):
    formula, family, kwargs, rows = BRMS_DEFAULTS[label]
    pri = ab.default_priors(formula, DEFAULTS_FRAME, family=family, **kwargs)
    m = getattr(ab, family)(formula, DEFAULTS_FRAME, pri, **kwargs)
    table = m.prior_table()
    assert [(r["class"], r["coef"], r["group"], r["prior"], r["lb"], r["ub"]) for r in table] == rows
    assert all(r["source"] == "default" for r in table)
    par = get_family(family).zero_parameter
    assert getattr(pri, par) == ab.Beta(1, 1) and getattr(pri, par).source == "default"
    assert par in m.default_prior_labels
    assert f"{par} ~ Beta(alpha=1, beta=1)   (brms default beta(1, 1), lb 0, ub 1)" in m.describe()


def test_a_zero_parameter_without_a_recorded_default_is_refused(monkeypatch):
    from dataclasses import replace
    from types import MappingProxyType

    import autobayes.families as families

    bare = replace(families.FAMILIES["hurdle_poisson"], default_zero_prior=None)
    monkeypatch.setattr(families, "FAMILIES", MappingProxyType({**families.FAMILIES, "hurdle_poisson": bare}))
    with pytest.raises(ab.PriorError, match="no brms default prior is recorded for the hurdle_poisson family's hu"):
        ab.default_priors("cnt ~ x", DEFAULTS_FRAME, family="hurdle_poisson")


@pytest.mark.parametrize("which", [ZIP, HURDLE], ids=["zi", "hu"])
def test_fit_record_and_summary_rows(which, monkeypatch):
    family, par = which
    m = build(which, "cnt ~ x + (1 | g)", group=True)
    post = draws(par)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "1|g_sigma", par}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == family and rec["model"]["link"] == "log"
    assert rec["model"]["priors"][par] == "Beta(alpha=2, beta=2)" and "sigma" not in rec["model"]["priors"]
    assert "log scale" in rec["model"]["intercept_prior_applies_to"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
