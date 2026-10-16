"""Finite mixtures of Gaussian components (brms-parity C05): the row density against a 60-digit reference on both paths
(PyTensor's gradient and JAX's autodiff of the jaxified graph, what NumPyro differentiates), the draws, the graph and its
names, ``value_point``, the density and gradient of whole models by hand, the ordered intercepts under a prior per component,
every refusal, by-hand predictions with the ``component`` dimension, prior-predictive draws, ``describe()``, the target
identity, brms's default prior tables and the fit record, and that the oracle gates fail on corrupted mixture fixtures. No
MCMC; the brms comparison is the oracle cases ``mixture_gauss_gauss``, ``mixture_gauss3_categorical`` and
``mixture_gauss_default_priors``."""
import copy
import json
import warnings
from decimal import Decimal, localcontext
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import integrate
from scipy.stats import dirichlet, expon, halfnorm, norm
from scipy.stats import t as student_t

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import get_family
from autobayes.monotonic import stick_breaking_constrain, stick_breaking_free

# -- data ---------------------------------------------------------------------------------------------------------------
X = [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7, 0.9, -1.4, 0.2, 1.6]
Y = [0.3, 2.1, -0.4, 1.7, 0.2, 3.0, -1.1, 0.8, 2.6, -0.5, 1.2, 0.9]


def data():
    return pd.DataFrame({"x": X, "y": Y, "z": list("pqrpqrpqrpqr"), "g": list("aabbccaabbcc"),
                         "e": [1.5, 2.0, 0.5, 1.0, 3.0, 1.2, 0.8, 2.2, 1.1, 0.6, 1.9, 2.4], "ord": [1, 2, 3, 4] * 3})


def component(intercept=None, b=None, sigma=None):
    return ab.Priors(intercept=intercept or ab.Normal(0, 2), b=b or ab.Normal(0, 1), sigma=sigma or ab.HalfNormal(1))


def mixture_priors(K=2, theta=None):
    return ab.Priors(components=[component(intercept=ab.Normal(k - 1.0, 2)) for k in range(K)], theta=theta or ab.Dirichlet(2))


def mixture_model(formula="y ~ x", K=2, frame=None, priors=None, **kw):
    return ab.mixture(formula, data() if frame is None else frame, priors or mixture_priors(K), families=["gaussian"] * K, **kw)


# -- the row density: a 60-digit reference ---------------------------------------------------------------------------------
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863")


def exact_mixture_row(y, mu, sigma, theta):
    """(log sum_k theta_k Normal(y | mu_k, sigma_k), d/d mu, d/d sigma, d/d theta) at 60 digits, the sum taken with its
    largest term factored out."""
    with localcontext() as ctx:
        ctx.prec = 60
        y = Decimal(float(y))
        mu, sigma, theta = ([Decimal(float(v)) for v in a] for a in (mu, sigma, theta))
        half_log_2pi = (2 * PI).ln() / 2
        terms = [t.ln() - ((y - m) / s) ** 2 / 2 - s.ln() - half_log_2pi for m, s, t in zip(mu, sigma, theta)]
        top = max(terms)
        total = sum((v - top).exp() for v in terms)
        w = [(v - top).exp() / total for v in terms]               # each component's share of the row
        return (float(top + total.ln()), [float(wk * (y - m) / s ** 2) for wk, m, s in zip(w, mu, sigma)],
                [float(wk * (((y - m) / s) ** 2 - 1) / s) for wk, m, s in zip(w, mu, sigma)], [float(wk / t) for wk, t in zip(w, theta)])


ROWS = [  # (y, mu) with the sigma and theta of their set
    (0.0, (-1.0, 0.5, 3.0)), (40.0, (0.0, 0.0, 0.0)), (-1000.0, (1000.0, -1000.0, 0.0)), (1000.0, (-1000.0, 999.0, 0.0)),
    (2.5, (2.5, 2.5 + 1e-9, -2.5)), (1e5, (-1e5, 0.0, 1e5 - 3.0)), (-7.0, (3.0, 4.0, 5.0)),
]
SETS = [(np.array([1.0, 0.5, 2.0]), np.array([0.2, 0.5, 0.3])),
        (np.array([1e-3, 30.0, 0.7]), np.array([1e-12, 1.0 - 1e-12 - 1e-6, 1e-6]))]


def _rows(mode, y, mu, sigma, theta):
    yv, mv, sv, tv = pt.dvector("y"), pt.dmatrix("mu"), pt.dvector("sigma"), pt.dvector("theta")
    rows = lk.gaussian_mixture_logp(yv, mv, sv, tv)
    if mode == "pytensor":
        f = pytensor.function([yv, mv, sv, tv], [rows, pt.grad(rows.sum(), mv), pt.grad(rows.sum(), sv), pt.grad(rows.sum(), tv)])
        return [np.asarray(a, dtype=float) for a in f(y, mu, sigma, theta)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[yv, mv, sv, tv], outputs=[rows])
    return [np.asarray(a, dtype=float) for a in (jf(y, mu, sigma, theta)[0], jax.grad(lambda m: jf(y, m, sigma, theta)[0].sum())(mu),
                                                 jax.grad(lambda s: jf(y, mu, s, theta)[0].sum())(sigma),
                                                 jax.grad(lambda t: jf(y, mu, sigma, t)[0].sum())(theta))]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_rows_are_exact_and_never_nan_on_both_paths(mode):
    """Every row, one at a time (sigma and theta are shared by the rows): finite with a finite gradient, the value within
    1e-12 relative of the reference and every gradient entry within 1e-10, including rows 1e5 standard deviations from
    every component, tiny and unequal scales and proportions from 1e-12 to nearly 1."""
    for sigma, theta in SETS:
        for y, mu in ROWS:
            value, d_mu, d_sigma, d_theta = _rows(mode, np.array([y]), np.array([mu]), sigma, theta)
            assert np.isfinite(value).all() and np.isfinite(d_mu).all() and np.isfinite(d_sigma).all() and np.isfinite(d_theta).all()
            want = exact_mixture_row(y, mu, sigma, theta)
            assert abs(value[0] - want[0]) <= 1e-12 * max(1.0, abs(want[0])), (mode, y, mu, value[0], want[0])
            for got, ref in ((d_mu[0], want[1]), (d_sigma, want[2]), (d_theta, want[3])):
                ref = np.asarray(ref)
                assert np.all(np.abs(got - ref) <= 1e-10 * np.maximum(1.0, np.abs(ref))), (mode, y, mu, got, ref)


def test_pointwise_rows_are_exact():
    """The NumPy form of the row density that ``ab.log_lik`` evaluates (C07, ``pointwise.gaussian_mixture``) on the same
    rows, all at once: finite and within 1e-12 relative of the reference."""
    from autobayes.families.pointwise import gaussian_mixture

    y, mu = np.array([r[0] for r in ROWS]), np.array([r[1] for r in ROWS])
    for sigma, theta in SETS:
        got = gaussian_mixture(y, mu, sigma, theta)
        want = np.array([exact_mixture_row(r[0], r[1], sigma, theta)[0] for r in ROWS])
        assert got.shape == y.shape and np.isfinite(got).all()
        assert np.all(np.abs(got - want) <= 1e-12 * np.maximum(1.0, np.abs(want))), (got, want)


def test_draws_and_mean_are_the_mixture():
    mu, sigma, theta = np.array([[-2.0, 0.5, 4.0]]), np.array([0.3, 0.2, 0.4]), np.array([0.2, 0.5, 0.3])
    draws = pm.draw(pm.CustomDist.dist(mu, sigma, theta, logp=lk.gaussian_mixture_logp, random=lk.gaussian_mixture_random,
                                       signature="(k),(k),(k)->()"), draws=40000, random_seed=5)
    assert draws.shape == (40000, 1)
    comp = np.argmin(np.abs(draws[:, 0, None] - mu[0]), axis=1)                 # the components do not overlap here
    for k in range(3):
        share = (comp == k).mean()
        assert abs(share - theta[k]) < 4.5 * np.sqrt(theta[k] * (1 - theta[k]) / 40000)
        assert abs(draws[comp == k, 0].std() - sigma[k]) < 0.02
    np.testing.assert_allclose(lk.mixture_mean(np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]), theta), [2.1, 0.7], rtol=1e-15)
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)                # brms's component, then its Normal
    k = lk.draw_categories(np.broadcast_to(theta, (5, 3)), rng1) - 1
    want = rng1.normal(np.take_along_axis(np.broadcast_to(mu, (5, 3)), k[:, None], 1)[:, 0], sigma[k])
    np.testing.assert_array_equal(lk.gaussian_mixture_random(np.broadcast_to(mu, (5, 3)), sigma, theta, rng2), want)


# -- the ordered intercepts with one prior per component ------------------------------------------------------------------
def _intercepts(params, prior="Normal"):
    with pm.Model(coords={"k": ["1", "2"]}) as model:
        lk.ordered_thresholds("a", prior, params, ["1", "2"], "k")
    return model


def test_ordered_intercepts_under_a_prior_per_entry():
    """Each entry's own prior, summed (brms adds normal_lpdf(Intercept_mu<k> | ...) per component), on Stan's ordered
    coordinates; forward draws by rejection follow the product density restricted to the ordered region (sorting the
    independent draws would not, once the priors differ)."""
    model = _intercepts({"mu": np.array([-1.0, 1.0]), "sigma": np.array([2.0, 0.5])})
    value = np.array([-0.3, 0.8])
    point = {"a_ordered__": np.r_[value[0], np.log(np.diff(value))]}
    np.testing.assert_allclose(model.compile_logp(jacobian=False)(point), norm.logpdf(value, [-1, 1], [2, 0.5]).sum(), rtol=1e-14)
    np.testing.assert_allclose(model.compile_logp()(point), norm.logpdf(value, [-1, 1], [2, 0.5]).sum() + np.log(1.1), rtol=1e-14)
    draws = pm.draw(model["a"], draws=40000, random_seed=11)
    assert (np.diff(draws, axis=-1) > 0).all()
    # E[a_1] under N(-1, 2) x N(1, 0.5) restricted to a_1 < a_2: P(a_2 > t) weighs a_1's density
    num = integrate.quad(lambda t: t * norm.pdf(t, -1, 2) * norm.sf(t, 1, 0.5), -30, 30)[0]
    den = integrate.quad(lambda t: norm.pdf(t, -1, 2) * norm.sf(t, 1, 0.5), -30, 30)[0]
    sd = np.sqrt(integrate.quad(lambda t: (t - num / den) ** 2 * norm.pdf(t, -1, 2) * norm.sf(t, 1, 0.5), -30, 30)[0] / den)
    assert abs(draws[:, 0].mean() - num / den) < 4.5 * sd / np.sqrt(40000)
    assert abs(draws[:, 0].mean() - (-1.0)) > 0.2                               # far from the untruncated mean, as it should be
    impossible = _intercepts({"mu": np.array([10.0, -10.0]), "sigma": np.array([0.01, 0.01])})
    with pytest.raises(ValueError, match="after 1000 rounds the entries' priors still give unordered draws"):
        pm.draw(impossible["a"], random_seed=1)
    same = _intercepts({"mu": 0.5, "sigma": 2.0})                               # one prior for every entry: the sorted draws
    assert (np.diff(pm.draw(same["a"], draws=50, random_seed=2), axis=-1) > 0).all()


# -- the model: graph, names and value_point --------------------------------------------------------------------------------
def test_graph_names_and_value_point():
    m = mixture_model("y ~ x + z", K=3, categorical={"z": "p"})
    assert m.family == "mixture" and m.components == ("gaussian",) * 3 and m.family_spec is get_family("mixture")
    assert [v.name for v in m.pymc_model.value_vars] == ["Intercept_ordered__", "x", "z", "sigma1_log__", "sigma2_log__",
                                                         "sigma3_log__", "theta_stickbreaking__"]
    dims = m.pymc_model.named_vars_to_dims
    assert dims["Intercept"] == dims["x"] == dims["theta"] == ("mixture_dim",) and dims["z"] == ("z_dim", "mixture_dim")
    assert dims["mu"] == ("__obs__", "y_dim") and tuple(m.pymc_model.coords["mixture_dim"]) == ("1", "2", "3")
    assert m.names["linpred"] == "mu" and m.names["mean"] == "mean" and m.names["theta"] == "theta"
    assert m.names["component_dim"] == "mixture_dim" and [m.names[f"sigma{k}"] for k in (1, 2, 3)] == ["sigma1", "sigma2", "sigma3"]
    assert "sigma" not in m.names and m.mixture_names["dpars"] == ("mu1", "mu2", "mu3")
    a, s, theta = np.array([-0.5, 0.2, 1.4]), np.array([0.5, 1.0, 2.0]), np.array([0.2, 0.3, 0.5])
    b = np.array([[0.4, -0.2, 0.1], [0.3, 0.0, -0.6], [1.1, 0.5, 0.2]])            # rows x, z[q], z[r]; one column per component
    p = m.value_point(intercept=a, b=b, sigma=s, theta=theta)
    np.testing.assert_allclose(p["Intercept_ordered__"], [-0.5, np.log(0.7), np.log(1.2)], rtol=1e-15)
    np.testing.assert_array_equal(p["x"], b[0])
    np.testing.assert_array_equal(p["z"], b[1:])
    np.testing.assert_allclose([p[f"sigma{k}_log__"] for k in (1, 2, 3)], np.log(s), rtol=1e-15)
    np.testing.assert_allclose(p["theta_stickbreaking__"], stick_breaking_free(theta), rtol=1e-15)
    for bad, message in ((dict(intercept=[0.5, 0.2, 1.4]), "strictly increasing"), (dict(intercept=[0.1, 0.2]), "the 3 components'"),
                         (dict(b=b[:, :2]), r"b must be an array of shape \(3, 3\)"), (dict(sigma=[1.0, -1.0, 1.0]), "finite and positive"),
                         (dict(theta=[0.5, 0.5, 0.5]), "must sum to 1"), (dict(theta=None), "theta .* is required"),
                         (dict(sd=0.5), r"\['sd'\] do not apply to a mixture")):
        with pytest.raises(ValueError, match=message):
            m.value_point(**{**dict(intercept=a, b=b, sigma=s, theta=theta), **bad})
    with pytest.raises(ValueError, match="theta .* is a mixture's"):
        ab.gaussian("y ~ x", data(), component()).value_point(intercept=0.1, b=[0.2], sigma=1.0, theta=[0.5, 0.5])


def _graph(m, point, names):
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n] for n in names]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    return [np.asarray(v, dtype=float) for v in fn(point)]


@pytest.mark.parametrize("K", [2, 3])
def test_density_graph_and_gradient_by_hand(K):
    """A prior per component on every class (different locations and scales, sigma priors of three distributions): the
    graph's mu, the log density with and without the Jacobian, its prior part, and the JAX density and gradient."""
    df = data()
    sig_priors = [ab.HalfNormal(1), ab.HalfStudentT(3, 2), ab.Exponential(1.5)][:K]
    comps = [ab.Priors(intercept=ab.Normal(k - 1.0, 1 + k), b={"x": ab.Normal(0.1 * k, 1), "z": ab.Normal(-0.2 * k, 2 + k)},
                       sigma=sig_priors[k]) for k in range(K)]
    pri = ab.Priors(components=comps, theta=ab.Dirichlet(*range(2, 2 + K)))
    m = ab.mixture("y ~ x + z", df, pri, families=["gaussian"] * K, categorical={"z": "p"})
    a = np.linspace(-0.4, 1.0, K)
    b = np.array([[0.4, -0.2, 0.1], [0.3, 0.0, -0.6], [1.1, 0.5, 0.2]])[:, :K]
    s, theta = np.linspace(0.6, 1.4, K), np.arange(1.0, K + 1) / np.arange(1.0, K + 1).sum()
    p = m.value_point(intercept=a, b=b, sigma=s, theta=theta)
    Xr = np.column_stack([df.x, df.z == "q", df.z == "r"]).astype(float)
    mu = a + (Xr - Xr.mean(0)) @ b                                                # (n, K): the centred design per component
    np.testing.assert_allclose(_graph(m, p, ["mu"])[0], mu, rtol=1e-14)
    y = df.y.to_numpy()
    lik = np.log((theta * norm.pdf(y[:, None], mu, s)).sum(1)).sum()
    sig_logp = [halfnorm.logpdf(s[0]), np.log(2) + student_t.logpdf(s[1], 3, 0, 2) if K > 1 else 0.0,
                expon.logpdf(s[2], scale=1 / 1.5) if K > 2 else 0.0][:K]
    prior = (sum(norm.logpdf(a[k], k - 1.0, 1 + k) + norm.logpdf(b[0, k], 0.1 * k, 1) + norm.logpdf(b[1:, k], -0.2 * k, 2 + k).sum()
                 for k in range(K)) + sum(sig_logp) + dirichlet.logpdf(theta, np.arange(2.0, 2 + K)))
    jac = np.log(np.diff(a)).sum() + np.log(s).sum() + stick_breaking_constrain(stick_breaking_free(theta), K)[1]
    pmm = m.pymc_model
    np.testing.assert_allclose(pmm.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-9)    # PyMC's half-t adds log 2 in float32
    np.testing.assert_allclose(pmm.compile_logp()(p), lik + prior + jac, rtol=1e-9)
    rvs = [pmm[n] for n in ["Intercept", "x", "z", *m.mixture_names["sigma"], "theta"]]
    np.testing.assert_allclose(pmm.compile_logp(vars=rvs, jacobian=False)(p), prior, rtol=1e-9)
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    args = [p[n] for n in names]
    np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(p)), rtol=1e-13)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float), rtol=1e-10, atol=1e-12)


def test_other_terms_compose():
    """Interactions, transforms, no predictors and per-level priors build; the coefficient layout is (levels, components)."""
    df = data()
    for formula, kw in (("y ~ x * z", {"categorical": {"z": "p"}}), ("y ~ x + exp(x)", {}), ("y ~ 1", {})):
        pri = ab.Priors(components=[ab.Priors(intercept=ab.Normal(0, 2), b=None if formula == "y ~ 1" else ab.Normal(0, 1),
                                              sigma=ab.HalfNormal(1))] * 2, theta=ab.Dirichlet(1))
        m = ab.mixture(formula, df, pri, families=["gaussian", "gaussian"], **kw)
        assert {v.name for v in m.pymc_model.value_vars} >= {"Intercept_ordered__", "theta_stickbreaking__"}
    per_level = ab.Priors(components=[component(b={"x": ab.Normal(0, 1), "z": {"q": ab.Normal(0.5, 1), "r": ab.Normal(0, 3)}}),
                                      component(b={"x": ab.Normal(0, 2), "z": ab.Normal(0, 1)})], theta=ab.Dirichlet(2))
    m = ab.mixture("y ~ x + z", df, per_level, families=["gaussian", "gaussian"], categorical={"z": "p"})
    p = m.value_point(intercept=[0.0, 1.0], b=np.zeros((3, 2)), sigma=[1.0, 1.0], theta=[0.5, 0.5])
    np.testing.assert_allclose(m.pymc_model.compile_logp(vars=[m.pymc_model["z"]])(p),
                               norm.logpdf(0, 0.5, 1) + norm.logpdf(0, 0, 3) + 2 * norm.logpdf(0, 0, 1), rtol=1e-14)


# -- refusals ---------------------------------------------------------------------------------------------------------------
def test_refusals():
    df = data()
    two = ["gaussian", "gaussian"]
    for families, message in ((None, "needs its component families"), ("gaussian", "must be a list of family names"),
                              (["gaussian"], "at least two components"), (["gaussian", "tweedie"], "family must be one of"),
                              (["gaussian", "weibull"], "a weibull mixture component is not offered.*D52"),   # a family since D01
                              (["gaussian", "student"], "a student mixture component is not offered.*D52"),
                              (["poisson", "poisson"], "a poisson mixture component is not offered")):
        with pytest.raises(ValueError, match=message):
            ab.mixture("y ~ x", df, mixture_priors(), families=families)
    for order, message in (("none", "order='none' .* is not offered.*D53"), (False, "order=False"), ("sigma", "order must be 'mu'")):
        with pytest.raises(ValueError, match=message):
            ab.mixture("y ~ x", df, mixture_priors(), families=two, order=order)
    for order in ("mu", True, None):
        assert ab.mixture("y ~ x", df, mixture_priors(), families=two, order=order).components == ("gaussian", "gaussian")
    with pytest.raises(ab.FormulaError, match="a mixture takes one formula.*D54"):
        ab.mixture(["y ~ x", "theta1 ~ z"], df, mixture_priors(), families=two)
    for formula, message in (("y ~ x + (1 | g)", r"group-level term \('\(1 \| g\)'\).*D55"), ("y ~ x + offset(e)", "no offset term.*D55"),
                             ("y ~ 0 + x", "the intercept cannot be removed"), ("y ~ x - 1", "the intercept cannot be removed"),
                             ("y ~ x + mo(ord)", "the monotonic term 'mo\\(ord\\)' is not available for the mixture family"),
                             ("y | weights(e) ~ x", "'weights\\(e\\)' is not available for the mixture family")):
        with pytest.raises(ab.FormulaError, match=message):
            ab.mixture(formula, df, mixture_priors(), families=two)
    with pytest.raises(ab.FormulaError, match="center_predictors=False is not available for a mixture"):
        ab.mixture("y ~ x", df, mixture_priors(), families=two, center_predictors=False)
    with pytest.raises(ValueError, match="families= and order= are a mixture's"):
        ab.Model("y ~ x", df, component(), family="gaussian", families=two)
    with pytest.raises(ab.FormulaError, match="Gaussian mixture family needs a numeric response"):
        ab.mixture("z ~ x", df, mixture_priors(), families=two)
    for pri, message in ((mixture_priors(3), "the mixture has 2 components .* gives 3"),
                         (ab.Priors(components=[component()] * 2, theta=ab.Dirichlet(2), sigma=ab.HalfNormal(1)),
                          r"\['sigma'\] at the top level do not apply"),
                         (ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)), r"\['intercept', 'b', 'sigma'\]"),
                         (ab.Priors(theta=ab.Dirichlet(2)), "needs the priors of each component"),
                         (ab.Priors(components=[component()] * 2), "needs a prior on its mixing proportions"),
                         (ab.Priors(components=[component()] * 2, theta=ab.Normal(0, 1)), "must be a prior on a simplex"),
                         (ab.Priors(components=[component()] * 2, theta=ab.Dirichlet(1, 2, 3)), r"theta' \(2 components\).*give 2"),
                         (ab.Priors(components=[component(), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1))], theta=ab.Dirichlet(2)),
                          r"component 2 \(brms's mu2 and sigma2\): the Gaussian family needs a prior on the residual SD"),
                         (ab.Priors(components=[component(), component(intercept=ab.StudentT(3, 0, 2))], theta=ab.Dirichlet(2)),
                          r"intercept priors must share one distribution family.*\['Normal', 'StudentT'\]"),
                         (ab.Priors(components=[component(), component(b=ab.Flat())], theta=ab.Dirichlet(2)),
                          r"priors of 'x' must share one distribution family.*\['Flat', 'Normal'\]"),
                         (ab.Priors(components=[component(sigma=ab.Normal(0, 1))] * 2, theta=ab.Dirichlet(2)), "positive-support prior")):
        with pytest.raises(ab.PriorError, match=message):
            ab.mixture("y ~ x", df, pri, families=two)
    with pytest.raises(ab.PriorError, match="components and theta are the priors of a mixture"):
        ab.gaussian("y ~ x", df, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), theta=ab.Dirichlet(2)))
    for column in ("theta", "sigma2", "mixture"):                                 # the graph's names cannot be columns
        with pytest.raises(ab.FormulaError, match="collision"):
            mixture_model(f"y ~ {column}", frame=data().assign(**{column: data().x if column != "mixture" else data().z}),
                          categorical={"mixture": "p"} if column == "mixture" else None)
    with pytest.raises(ValueError, match="the mixture family has the identity link only"):
        ab.Model("y ~ x", df, mixture_priors(), family="mixture", families=two, link="log")


# -- predictions ------------------------------------------------------------------------------------------------------------
def mixture_draws(K=2, z=False):
    rng = np.random.default_rng(0)
    post = {"Intercept": rng.normal(size=(2, 4, K)), "x": rng.normal(size=(2, 4, K)), "theta": rng.dirichlet(np.ones(K), size=(2, 4))}
    post.update({f"sigma{k}": rng.uniform(0.5, 1.5, size=(2, 4)) for k in range(1, K + 1)})
    dims = {"Intercept": ["mixture_dim"], "x": ["mixture_dim"], "theta": ["mixture_dim"]}
    coords = {"mixture_dim": [str(k) for k in range(1, K + 1)]}
    if z:                                                                       # components before levels, as a posterior may hold it
        post["z"] = rng.normal(size=(2, 4, K, 2))
        dims["z"], coords["z_dim"] = ["mixture_dim", "z_dim"], ["q", "r"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_by_hand():
    df = data()
    m = mixture_model("y ~ x + z", K=3, categorical={"z": "p"})
    post = mixture_draws(3, z=True)
    P = post.posterior
    Xr = np.column_stack([df.x, df.z == "q", df.z == "r"]).astype(float)
    B = np.concatenate([P["x"].values[..., None, :], np.swapaxes(P["z"].values, -1, -2)], axis=-2)   # (chain, draw, 3, K)
    eta = P["Intercept"].values[..., None, :] + np.einsum("cdpk,np->cdnk", B, Xr)          # raw-scale intercepts, raw design
    lin = m.predict(post, kind="linear")
    assert lin.eta.dims == ("chain", "draw", "__obs__", "component") and list(lin.component.values) == ["1", "2", "3"]
    np.testing.assert_allclose(lin.eta, eta, rtol=1e-14)
    assert lin.attrs["link"] == "identity" and lin.attrs["new_groups"] == "not_applicable"
    theta = P["theta"].values
    mean = m.predict(post, kind="mean")
    assert set(mean.data_vars) == {"mean"} and mean["mean"].dims == ("chain", "draw", "__obs__")
    np.testing.assert_allclose(mean["mean"], (theta[..., None, :] * eta).sum(-1), rtol=1e-14)
    out = m.predict(post, kind="response", seed=7)
    sigma = np.stack([P[f"sigma{k}"].values for k in (1, 2, 3)], -1)
    np.testing.assert_array_equal(out.y, lk.gaussian_mixture_random(eta, sigma[..., None, :], theta[..., None, :], np.random.default_rng(7)))
    np.testing.assert_array_equal(out.y, m.predict(post, kind="response", seed=7).y)
    assert not np.array_equal(out.y, m.predict(post, kind="response", seed=8).y)
    new = pd.DataFrame({"x": [0.3, -2.0], "z": ["r", "p"]})
    res = m.predict(post, newdata=new, kind="linear")
    np.testing.assert_allclose(res.eta.values[..., 0, :], P["Intercept"].values + 0.3 * P["x"].values + P["z"].values[..., 1], rtol=1e-14)
    np.testing.assert_allclose(res.eta.values[..., 1, :], P["Intercept"].values - 2.0 * P["x"].values, rtol=1e-14)
    np.testing.assert_allclose(m.predict(post, kind="mean", include_group_effects=False)["mean"], mean["mean"], rtol=1e-15)
    with pytest.raises(ValueError, match="new_groups does not apply"):
        m.predict(post, kind="mean", new_groups="population")
    with pytest.raises(ValueError, match="kind must be 'mean' \\(the mixture's expected response\\)"):
        m.predict(post, kind="probability")


def test_log_lik_bayes_R2_and_loo_by_hand():
    """C07 for a mixture: ``log_lik`` is each row's ``log sum_k theta_k Normal(y_i | mu_k,i, sigma_k)`` per draw (brms's
    log_lik_mixture), ``bayes_R2`` the R-squared of the mixture's mean ``sum_k theta_k mu_k,i`` (brms's posterior_epred) and
    ``loo`` PSIS-LOO of that log_lik; the frozen case mixture_gauss3_post checks the three against brms."""
    from autobayes.post_processing import loo_from_log_lik

    m = mixture_model("y ~ x + z", K=3, categorical={"z": "p"})
    post = mixture_draws(3, z=True)
    P = post.posterior
    eta = m.predict(post, kind="linear").eta.transpose("chain", "draw", "__obs__", "component").values
    theta = P["theta"].values[..., None, :]
    sigma = np.stack([P[f"sigma{k}"].values for k in (1, 2, 3)], -1)[..., None, :]
    y = np.asarray(Y)
    ll = ab.log_lik(m, post)
    assert ll.dims == ("chain", "draw", "__obs__") and ll.shape == (2, 4, 12)
    np.testing.assert_allclose(ll.values, np.log((theta * norm.pdf(y[:, None], eta, sigma)).sum(-1)), rtol=1e-12)
    yhat = (theta * eta).sum(-1)
    r2 = np.var(yhat, axis=-1, ddof=1) / (np.var(yhat, axis=-1, ddof=1) + np.var(y - yhat, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(m, post).values, r2, rtol=1e-12)
    pd.testing.assert_frame_equal(ab.loo(m, post, r_eff=1.0).pointwise, loo_from_log_lik(ll, r_eff=1.0).pointwise)


def test_a_fits_posterior_holds_the_raw_intercepts():
    """What fit() returns: Bambi's result cleaning uncentres the intercept of every component (brms's b_mu<k>_Intercept =
    Intercept_mu<k> - sum_p mean(x_p) b_mu<k>[p]), which is what predict() reads with the raw design."""
    m = mixture_model("y ~ x + z", K=3, categorical={"z": "p"})
    rng = np.random.default_rng(1)
    post = {"Intercept": np.sort(rng.normal(size=(2, 4, 3)), -1), "x": rng.normal(size=(2, 4, 3)), "z": rng.normal(size=(2, 4, 2, 3)),
            "sigma1": np.ones((2, 4)), "sigma2": np.ones((2, 4)), "sigma3": np.ones((2, 4)), "theta": rng.dirichlet(np.ones(3), size=(2, 4))}
    idata = az.from_dict(posterior={k: v.copy() for k, v in post.items()}, coords={"mixture_dim": ["1", "2", "3"], "z_dim": ["q", "r"]},
                         dims={"Intercept": ["mixture_dim"], "x": ["mixture_dim"], "z": ["z_dim", "mixture_dim"], "theta": ["mixture_dim"]})
    out = m.bambi_model.backend._clean_results(idata, omit_offsets=True, include_response_params=False)
    B = np.concatenate([post["x"][..., None, :], post["z"]], axis=-2)              # (chain, draw, columns, K)
    raw = post["Intercept"] - np.einsum("p,cdpk->cdk", m.design.X_raw.mean(0), B)
    np.testing.assert_allclose(out.posterior["Intercept"].transpose("chain", "draw", "mixture_dim").values, raw, rtol=1e-14)
    eta = m.predict(out, kind="linear").eta.values
    np.testing.assert_allclose(eta, post["Intercept"][..., None, :] + np.einsum("cdpk,np->cdnk", B, m.design.X_model), rtol=1e-12,
                               atol=1e-13)                                     # the centred form, as brms's mu_k


def test_prior_predictive():
    m = mixture_model(K=3)
    out = m.prior_predictive(draws=40, seed=3)
    a = out.prior["Intercept"].values[0]
    assert a.shape == (40, 3) and (np.diff(a, axis=-1) > 0).all()              # ordered under a prior per component
    assert out.prior["theta"].shape == (1, 40, 3) and np.allclose(out.prior["theta"].sum("mixture_dim"), 1.0)
    assert out.prior_predictive["y"].shape == (1, 40, 12) and out.attrs["autobayes_generation_kind"] == "prior_predictive"
    flat = ab.mixture("y ~ x", data(), ab.default_priors("y ~ x", data(), family="mixture", families=["gaussian"] * 2),
                      families=["gaussian"] * 2)
    with pytest.raises(ValueError, match=r"the prior of \['mu1:b:x', 'mu2:b:x'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)


# -- describe, identity, defaults, record ------------------------------------------------------------------------------------
def test_describe_and_identity():
    m = mixture_model("y ~ x + z", K=3, categorical={"z": "p"})
    text = m.describe()
    assert text.startswith("AutoBayes mixture model of 3 Gaussian components") and "family    : mixture(gaussian, gaussian, gaussian)" in text
    assert "order = 'mu'" in text and "a_c[1] < a_c[2] < a_c[3] (brms's ordered[3] ordered_Intercept)" in text
    assert "(brms's mu1, mu2, mu3)" in text and "log_sum_exp_k(log theta[k] +\n  normal_lpdf(y_i | mu_k,i, sigma_k))" in text
    assert "component 2 (brms's dpar mu2 and class sigma2):\n    a_c ~ Normal(mu=0, sigma=2)\n    b_x ~ Normal(mu=0, sigma=1)" in text
    assert "    b_z[r] ~ Normal(mu=0, sigma=1)\n    sigma2 ~ HalfNormal(sigma=1)" in text
    assert "theta ~ Dirichlet(a=2)   (brms class theta; concentrations 2, 2, 2)" in text and "theta_stickbreaking__" in text
    assert "z         : categorical, treatment coding, reference 'p'" in text
    same = ab.Model("y ~ x + z", data(), mixture_priors(3), family="mixture", families=["gaussian"] * 3, categorical={"z": "p"})
    assert same.describe() == text and same.target_identity == m.target_identity
    other = ab.Priors(components=list(mixture_priors(3).components), theta=ab.Dirichlet(1, 2, 3))
    assert mixture_model("y ~ x + z", K=3, priors=other, categorical={"z": "p"}).target_identity != m.target_identity
    swapped = ab.Priors(components=[component(intercept=ab.Normal(1, 2)), component(intercept=ab.Normal(-1, 2)),
                                    component(intercept=ab.Normal(0, 2))], theta=ab.Dirichlet(2))
    assert mixture_model("y ~ x + z", K=3, priors=swapped, categorical={"z": "p"}).target_identity != m.target_identity
    assert mixture_model("y ~ x + z", K=2, categorical={"z": "p"}).target_identity != m.target_identity
    assert ab.Priors(components=[component()] * 2) == ab.Priors(components=(component(), component()))   # a list is kept as a tuple


# brms 2.23.0's default_prior() (R 4.5.2, C collation) for mixtures of Gaussian components on data(), as (class, coef, dpar,
# prior, lb); every source "default"
def _brms_rows(K, coefs, location, scale):
    out = [(f"sigma{k}", "", "", f"student_t(3, 0, {scale})", "0") for k in sorted(range(1, K + 1), key=str)]
    out += [("theta", "", "", "dirichlet(1)", "")]
    for k in sorted(range(1, K + 1), key=str):
        out += [("Intercept", "", f"mu{k}", f"student_t(3, {location}, {scale})", "")]
        out += ([("b", "", f"mu{k}", "", "")] + [("b", c, f"mu{k}", "", "") for c in coefs]) if coefs else []
    return out


BRMS_DEFAULTS = {
    "3 components, numeric and categorical": ("y ~ x + z", 3, {"categorical": {"z": "p"}}, data(),
                                              _brms_rows(3, ["x", "zq", "zr"], "0.9", "2.5")),
    "10 components": ("y ~ x", 10, {}, data(), _brms_rows(10, ["x"], "0.9", "2.5")),
    "intercept only": ("y ~ 1", 2, {}, data(), _brms_rows(2, [], "0.9", "2.5")),
    "interaction": ("y ~ x * z", 2, {"categorical": {"z": "p"}}, data(), _brms_rows(2, ["x", "x:zq", "x:zr", "zq", "zr"], "0.9", "2.5")),
    "wide response": ("yw ~ x", 2, {}, data().assign(yw=data().y * 10), _brms_rows(2, ["x"], "8.5", "15.6")),
}


@pytest.mark.parametrize("label", sorted(BRMS_DEFAULTS))
def test_default_priors_are_brms_default_prior(label):
    formula, K, kwargs, frame, rows = BRMS_DEFAULTS[label]
    pri = ab.default_priors(formula, frame, family="mixture", families=["gaussian"] * K, **kwargs)
    assert len(pri.components) == K and pri.theta == ab.Dirichlet(1) and pri.theta.source == "default"
    m = ab.mixture(formula, frame, pri, families=["gaussian"] * K, **kwargs)
    table = m.prior_table()
    assert [(r["class"], r["coef"], r["dpar"], r["prior"], r["lb"]) for r in table] == rows
    assert all(r["source"] == "default" and r["ub"] == "" for r in table)
    assert m.default_prior_labels[:1] == ["mu1:intercept"] and m.default_prior_labels[-1] == "theta"
    assert "Default priors (ab.default_priors()" in m.describe()


def test_default_prior_refusals():
    with pytest.raises(ValueError, match="needs its component families"):
        ab.default_priors("y ~ x", data(), family="mixture")
    with pytest.raises(ValueError, match="families= and order= are a mixture's"):
        ab.default_priors("y ~ x", data(), families=["gaussian", "gaussian"])
    with pytest.raises(ab.FormulaError, match="group-level term"):
        ab.default_priors("y ~ x + (1 | g)", data(), family="mixture", families=["gaussian", "gaussian"])
    with pytest.raises(ab.FormulaError, match="center_predictors=False is not available for a mixture"):
        ab.default_priors("y ~ x", data(), family="mixture", families=["gaussian", "gaussian"], center_predictors=False)


def test_fit_record_and_summary_rows(monkeypatch):
    m = mixture_model()
    post = mixture_draws(2)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = session._summary_rows(m, post)
    assert {"Intercept[1]", "Intercept[2]", "x[1]", "sigma1", "sigma2", "theta[1]", "theta[2]"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "mixture" and rec["model"]["mixture"]["families"] == ["gaussian", "gaussian"]
    assert rec["model"]["priors"] == {"mu1:intercept": "Normal(mu=-1, sigma=2)", "mu1:b:x": "Normal(mu=0, sigma=1)",
                                      "sigma1": "HalfNormal(sigma=1)", "mu2:intercept": "Normal(mu=0, sigma=2)",
                                      "mu2:b:x": "Normal(mu=0, sigma=1)", "sigma2": "HalfNormal(sigma=1)", "theta": "Dirichlet(a=2)"}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
    other_model = mixture_model(priors=mixture_priors(theta=ab.Dirichlet(1)))
    other = ab.fit_record(other_model, SimpleNamespace(**{**vars(fit), "target_identity": other_model.target_identity}),
                          panel={"values": {}, "status": {}, "overall": "not evaluated"})
    with pytest.raises(ValueError, match="target identities differ|records fit different models"):
        ab.compare_records(rec, other)


# -- the oracle gates on corrupted mixture fixtures -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixture_bundle():
    from tests.oracle import fixture as fx
    from tests.oracle import gates

    if not fx.has_fixture("mixture_gauss3_categorical"):
        pytest.skip("mixture fixture not frozen")
    return gates.Bundle("mixture_gauss3_categorical")


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _index(raw, par, transform=None):
    return [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == par or (par is None and e["transform"] == transform)]


def test_corrupted_mixture_fixtures_fail(mixture_bundle):
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    def ordered_u(raw):
        raw["points"][2]["u"]["values"][_index(raw, "ordered_Intercept")[2]] += 1e-6
    with pytest.raises(OracleMismatch, match="point 2: Stan's ordered coordinate 2 of ordered_Intercept"):
        gates.check_gradient(_corrupted(mixture_bundle, ordered_u))

    def theta_u(raw):
        raw["points"][4]["u"]["values"][_index(raw, None, "unreached")[1]] += 1e-6
    with pytest.raises(OracleMismatch, match="point 4: Stan's stick-breaking coordinates of theta"):
        gates.check_gradient(_corrupted(mixture_bundle, theta_u))

    def sigma_gradient(raw):
        raw["points"][0]["grad"]["values"][_index(raw, "sigma3")[0]] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 0: gradient of sigma3_log__\[0\]"):
        gates.check_gradient(_corrupted(mixture_bundle, sigma_gradient))

    def theta_gradient(raw):
        raw["points"][1]["grad"]["values"][_index(raw, None, "unreached")[0]] += 1e-5
    with pytest.raises(OracleMismatch, match=r"point 1: gradient of theta_stickbreaking__\[0\]"):
        gates.check_gradient(_corrupted(mixture_bundle, theta_gradient))

    def zr_of_mu2(raw):                                                         # b_mu2[3] is z[r] of component 2: z's flat entry 1 * 3 + 1
        raw["points"][3]["grad"]["values"][_index(raw, "b_mu2")[2]] += 1e-5
    with pytest.raises(OracleMismatch, match=r"point 3: gradient of z\[4\]"):
        gates.check_gradient(_corrupted(mixture_bundle, zr_of_mu2))

    def con_theta(raw):
        raw["standata"]["con_theta"]["values"][2] = 5
    with pytest.raises(OracleMismatch, match="brms con_theta is"):
        gates.check_design(_corrupted(mixture_bundle, con_theta))

    def raw_intercept(raw):
        raw["points"][5]["pars"]["b_mu3_Intercept"]["values"][0] += 1e-8
    with pytest.raises(OracleMismatch, match="point 5: raw-scale intercept of mu3"):
        gates.check_derived(_corrupted(mixture_bundle, raw_intercept))

    def linpred(raw):
        raw["predictions"]["newdata"]["linpred"]["values"][-1] += 1e-8          # the last row of mu3 at the last point
    with pytest.raises(OracleMismatch, match="newdata: linear predictor differs"):
        gates.check_predictions(_corrupted(mixture_bundle, linpred))

    def epred(raw):
        raw["predictions"]["fitted"]["epred"]["values"][9] += 1e-8
    with pytest.raises(OracleMismatch, match="fitted: 'mean' differs from brms posterior_epred"):
        gates.check_predictions(_corrupted(mixture_bundle, epred))

    def lprior(raw):
        raw["points"][6]["pars"]["lprior"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 6: log density"):
        gates.check_logp(_corrupted(mixture_bundle, lprior))

    def design(raw):
        raw["standata"]["X_mu2"]["values"][-1] += 1e-11
    with pytest.raises(OracleMismatch, match="matches 0 brms columns"):
        _corrupted(mixture_bundle, design)

    def default_row(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "theta"]
        row["prior"] = "dirichlet(2)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(mixture_bundle, default_row))


def test_a_model_of_another_kind_does_not_match(mixture_bundle):
    from tests.oracle import fixture as fx
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    two = ab.mixture("y ~ x + z", mixture_bundle.data, mixture_priors(2), families=["gaussian"] * 2, categorical={"z": "p"})
    with pytest.raises(OracleMismatch, match=r"brms has the designs \['mu1', 'mu2', 'mu3'\]; the AutoBayes mixture's 2 components"):
        gates.Bundle("mixture_gauss3_categorical", model=two)
    gaussian = ab.gaussian("y ~ x + z", mixture_bundle.data, component(), categorical={"z": "p"})
    with pytest.raises(KeyError):
        gates.Bundle("mixture_gauss3_categorical", model=gaussian)
    if fx.has_fixture("gauss_varint"):
        case = fx.load_case("gauss_varint")
        frame = fx.read_frame(case, fx.fixture_dir("gauss_varint") / "data.csv")
        with pytest.raises(OracleMismatch, match=r"brms has the designs \[\]"):
            gates.Bundle("gauss_varint", model=ab.mixture("y ~ x", frame, mixture_priors(2), families=["gaussian"] * 2))


def test_corrupted_mixture_post_fixture_fails():
    """The post gate (C07) passes on the frozen mixture_gauss3_post and fails on one log_lik entry or one R-squared draw
    moved."""
    from tests.oracle import fixture as fx
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    if not fx.has_fixture("mixture_gauss3_post"):
        pytest.skip("mixture post fixture not frozen")
    b = gates.Bundle("mixture_gauss3_post")
    gates.check_post(b)

    def log_lik(raw):
        raw["post"]["log_lik"]["values"][100] += 1e-7                     # brms's log_lik is above -40 here: above 1e-9 relative
    with pytest.raises(OracleMismatch, match="log_lik differs from brms"):
        gates.check_post(_corrupted(b, log_lik))

    def r2(raw):
        raw["post"]["bayes_R2"]["values"][5] += 1e-8
    with pytest.raises(OracleMismatch, match="bayes_R2 differs from brms"):
        gates.check_post(_corrupted(b, r2))
