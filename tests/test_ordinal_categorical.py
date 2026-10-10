"""Ordinal (cumulative logit) and categorical responses (brms-parity B05): the row densities against a 50-digit reference
on both paths (PyTensor's gradient and JAX's autodiff of the jaxified graph, what NumPyro differentiates), the response
coding and its refusals, the formula refusals, the model's density, graph and coordinates by hand, ``value_point``, the
priors, by-hand predictions with the new ``category`` dimension, prior-predictive draws, ``describe()``, the target
identity, brms's default prior tables and the fit record. No MCMC; the brms comparison is the oracle cases
``cumulative_varint`` and ``categorical_poponly``."""
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
from scipy.special import expit, softmax
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import get_family


# -- data ---------------------------------------------------------------------------------------------------------------
X = [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7, 0.9, -1.4, 0.2, 1.6]


def ordinal_data():
    return pd.DataFrame({"x": X, "y": [1, 2, 4, 1, 2, 4, 1, 1, 4, 2, 3, 4], "g": list("aabbccaabbcc"),
                         "z": list("pqrpqrpqrpqr"), "e": [1.5, 2.0, 0.5, 1.0, 3.0, 1.2, 0.8, 2.2, 1.1, 0.6, 1.9, 2.4]})


def categorical_data():
    return pd.DataFrame({"x": X, "c": list("bacabcabcabc"), "z": list("pqrpqrpqrpqr")})


def ordinal_priors(group=True, **kw):
    return ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), **({"sd": ab.HalfNormal(1)} if group else {}), **kw)


def categorical_priors(**kw):
    return ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), **kw)


def cumulative_model(formula="y ~ x + (1 | g)", frame=None, priors=None, **kw):
    return ab.cumulative(formula, ordinal_data() if frame is None else frame, priors or ordinal_priors("|" in formula), **kw)


def categorical_model(formula="c ~ x", frame=None, priors=None, **kw):
    return ab.categorical(formula, categorical_data() if frame is None else frame, priors or categorical_priors(), **kw)


# -- the row densities: a 50-digit reference ----------------------------------------------------------------------------
def _sig(x):
    return 1 / (1 + (-x).exp())


def exact_ordinal_row(y, eta, tau):
    """(log P(y), d/d eta, d/d tau) of brms's cumulative logit row at 50 digits: P(y = k) = s(tau_k - eta) -
    s(tau_{k-1} - eta), the difference formed as e^b (e^(a - b) - 1) / ((1 + e^a)(1 + e^b)) so that nothing cancels."""
    with localcontext() as ctx:
        ctx.prec = 50
        e, t = Decimal(float(eta)), [Decimal(float(v)) for v in tau]
        K = len(t) + 1
        grad_t = [Decimal(0)] * len(t)
        if y == 1:
            a = t[0] - e
            value, d_eta = -(1 + (-a).exp()).ln(), -_sig(-a)
            grad_t[0] = _sig(-a)
        elif y == K:
            b = t[-1] - e
            value, d_eta = -(1 + b.exp()).ln(), _sig(b)
            grad_t[-1] = -_sig(b)
        else:
            a, b = t[y - 1] - e, t[y - 2] - e
            p = b.exp() * ((a - b).exp() - 1) / ((1 + a.exp()) * (1 + b.exp()))
            value = b + ((a - b).exp() - 1).ln() - (1 + a.exp()).ln() - (1 + b.exp()).ln()
            da, db = _sig(a) * _sig(-a), _sig(b) * _sig(-b)
            d_eta = (db - da) / p
            grad_t[y - 1], grad_t[y - 2] = da / p, -db / p
        return float(value), float(d_eta), [float(g) for g in grad_t]


def exact_categorical_row(y, eta):
    with localcontext() as ctx:
        ctx.prec = 50
        full = [Decimal(0)] + [Decimal(float(v)) for v in eta]
        m = max(full)
        z = sum((v - m).exp() for v in full)
        value = full[y - 1] - m - z.ln()
        p = [(v - m).exp() / z for v in full]
        return float(value), [float((1 if j == y - 1 else 0) - p[j]) for j in range(1, len(full))]


ETAS = [-1000.0, -40.0, -3.0, -0.5, 0.0, 0.7, 3.0, 40.0, 1000.0]
TAU = np.array([-1.2, -1.2 + 1e-9, 0.3, 2.0])                    # five categories, one gap of 1e-9


def _ordinal_rows(mode, y, eta, tau):
    yv, ev, tv = pt.lvector("y"), pt.dvector("eta"), pt.dvector("tau")
    rows = lk.ordered_logistic_logp(yv, ev, tv)
    if mode == "pytensor":
        f = pytensor.function([yv, ev, tv], [rows, pt.grad(rows.sum(), ev), pt.grad(rows.sum(), tv)])
        return [np.asarray(a, dtype=float) for a in f(y, eta, tau)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[yv, ev, tv], outputs=[rows])
    return [np.asarray(a, dtype=float) for a in (jf(y, eta, tau)[0], jax.grad(lambda e: jf(y, e, tau)[0].sum())(eta),
                                                 jax.grad(lambda t: jf(y, eta, t)[0].sum())(tau))]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_ordinal_rows_are_exact_and_never_nan_on_both_paths(mode):
    for y in range(1, 6):
        eta = np.array(ETAS)
        value, d_eta, _ = _ordinal_rows(mode, np.full(len(ETAS), y, dtype=np.int64), eta, TAU)
        assert np.isfinite(value).all() and np.isfinite(d_eta).all()
        for i, e in enumerate(ETAS):
            want_v, want_e, want_t = exact_ordinal_row(y, e, TAU)
            assert abs(value[i] - want_v) <= 1e-12 * max(1.0, abs(want_v)), (mode, y, e, value[i], want_v)
            assert abs(d_eta[i] - want_e) <= 1e-12 * max(1.0, abs(want_e)), (mode, y, e, d_eta[i], want_e)
            _, _, d_tau = _ordinal_rows(mode, np.array([y], dtype=np.int64), np.array([e]), TAU)
            np.testing.assert_allclose(d_tau, want_t, rtol=1e-9, atol=1e-12 * max(1.0, max(abs(v) for v in want_t)))


def test_ordinal_rows_outside_the_categories_have_no_mass():
    value, _, _ = _ordinal_rows("pytensor", np.array([0, 6, -1], dtype=np.int64), np.zeros(3), TAU)
    assert (value == -np.inf).all()
    value, _, _ = _ordinal_rows("pytensor", np.arange(1, 6, dtype=np.int64), np.full(5, 0.4), TAU)
    np.testing.assert_allclose(np.exp(value), lk.cumulative_probabilities(0.4, TAU), rtol=1e-12)
    assert abs(np.exp(value).sum() - 1) < 1e-14


def _categorical_rows(mode, y, eta):
    yv, ev = pt.lvector("y"), pt.dmatrix("eta")
    rows = lk.categorical_logit_logp(yv, ev)
    if mode == "pytensor":
        f = pytensor.function([yv, ev], [rows, pt.grad(rows.sum(), ev)])
        return [np.asarray(a, dtype=float) for a in f(y, eta)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[yv, ev], outputs=[rows])
    return [np.asarray(a, dtype=float) for a in (jf(y, eta)[0], jax.grad(lambda e: jf(y, e)[0].sum())(eta))]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_categorical_rows_are_exact_and_never_nan_on_both_paths(mode):
    grid = [(y, (e1, e2)) for y in (1, 2, 3) for e1 in ETAS for e2 in (-1000.0, -2.0, 0.0, 1.5, 40.0, 1000.0)]
    y = np.array([r[0] for r in grid], dtype=np.int64)
    eta = np.array([r[1] for r in grid])
    value, grad = _categorical_rows(mode, y, eta)
    assert np.isfinite(value).all() and np.isfinite(grad).all()
    for i, (yi, ei) in enumerate(grid):
        want_v, want_g = exact_categorical_row(yi, ei)
        assert abs(value[i] - want_v) <= 1e-12 * max(1.0, abs(want_v)), (mode, yi, ei, value[i], want_v)
        np.testing.assert_allclose(grad[i], want_g, rtol=1e-12, atol=1e-15)
    value, _ = _categorical_rows("pytensor", np.array([0, 4], dtype=np.int64), np.zeros((2, 2)))
    assert (value == -np.inf).all()


def test_likelihood_objects_draw_their_distribution():
    import pymc as pm

    eta, tau = np.array([-1.0, 0.4, 1.5]), np.array([-0.5, 0.2, 1.0])
    draws = pm.draw(pm.CustomDist.dist(eta, tau, logp=lk.ordered_logistic_logp, random=lk.ordered_logistic_random,
                                       signature="(),(k)->()", dtype="int64"), draws=20000, random_seed=3)
    p = lk.cumulative_probabilities(eta, tau)
    assert draws.shape == (20000, 3) and draws.dtype == np.int64 and draws.min() >= 1 and draws.max() <= 4
    for k in range(4):
        assert np.all(np.abs((draws == k + 1).mean(0) - p[:, k]) < 4.5 * np.sqrt(p[:, k] * (1 - p[:, k]) / 20000))
    mu = np.array([[0.3, -0.4], [2.0, 1.0]])
    draws = pm.draw(pm.CustomDist.dist(mu, logp=lk.categorical_logit_logp, random=lk.categorical_logit_random,
                                       signature="(k)->()", dtype="int64"), draws=20000, random_seed=4)
    p = lk.categorical_probabilities(mu)
    np.testing.assert_allclose(p, softmax(np.column_stack([np.zeros(2), mu]), axis=1), rtol=1e-14)
    for k in range(3):
        assert np.all(np.abs((draws == k + 1).mean(0) - p[:, k]) < 4.5 * np.sqrt(p[:, k] * (1 - p[:, k]) / 20000))


# -- the response ---------------------------------------------------------------------------------------------------------
def test_ordinal_responses_and_their_refusals():
    m = cumulative_model()
    assert m.categories == ("1", "2", "3", "4") and np.array_equal(m.design.y, ordinal_data().y) and m.warnings == []
    gap = ordinal_data().assign(y=[1, 2, 4, 1, 2, 4, 1, 1, 4, 2, 2, 4])          # nothing in category 3
    m = cumulative_model(frame=gap)
    assert m.categories == ("1", "2", "3", "4") and len(m.warnings) == 1 and "categories ['3']" in m.warnings[0]
    labels = ["lo", "mid", "hi", "top"]
    ordered = ordinal_data().assign(y=pd.Categorical([labels[v - 1] for v in ordinal_data().y], categories=labels, ordered=True))
    m = cumulative_model(frame=ordered)
    assert m.categories == tuple(labels) and np.array_equal(m.design.y, ordinal_data().y)          # declared order, not sorted
    assert m.data["y"].dtype == np.int64
    unused = ordinal_data().assign(y=pd.Categorical([labels[v - 1] for v in ordinal_data().y], categories=labels + ["max"],
                                                    ordered=True))
    m = cumulative_model(frame=unused)
    assert m.categories == (*labels, "max") and "categories ['max']" in m.warnings[0]               # every declared category
    for values, message in (([0, 1, 2], "has the value 0;"), ([1.5, 1, 2], "has the value 1.5;"), ([1, np.inf, 2], "has the value inf;"),
                            (["lo", "hi", "lo"], "Text labels have no order"), ([True, False, True], "is boolean"),
                            ([1, 1, 1], "has one category"), ([1, np.nan, 2], "missing values"),
                            ([1, 2, 2e6], "an ordinal response above 1000000 is refused")):
        with pytest.raises(ab.FormulaError, match=message):
            ab.cumulative("y ~ x", pd.DataFrame({"y": values, "x": [0.1, -0.3, 0.8]}), ordinal_priors(False))
    with pytest.raises(ab.FormulaError, match="unordered pandas Categorical"):
        ab.cumulative("y ~ x", pd.DataFrame({"y": pd.Categorical(["a", "b", "a"]), "x": [0.1, -0.3, 0.8]}), ordinal_priors(False))


def test_categorical_responses_and_their_refusals():
    m = categorical_model()
    assert m.categories == ("a", "b", "c")                                              # sorted, the first the reference
    assert list(m.design.y) == [2, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    codes = categorical_data().assign(c=[10, 2, 3, 2, 10, 3, 2, 10, 3, 2, 10, 3])       # integer codes sort numerically
    m = categorical_model(frame=codes)
    assert m.categories == ("2", "3", "10") and list(m.design.y[:3]) == [3, 1, 2]
    declared = categorical_data().assign(c=pd.Categorical(categorical_data().c, categories=["c", "b", "a", "d"]))
    m = categorical_model(frame=declared)
    assert m.categories == ("a", "b", "c") and "declared categories ['d']" in m.warnings[0]    # declared order not used
    for values, message in ((["a", "a", "a"], "has one category"), ([True, False, True], "boolean"),
                            (["a", 1, "b"], "mixes text labels and numeric codes"), (["a", None, "b"], "missing value")):
        with pytest.raises(ab.FormulaError, match=message):
            ab.categorical("c ~ x", pd.DataFrame({"c": values, "x": [0.1, -0.3, 0.8]}), categorical_priors())
    # the non-reference categories' linear predictors would both be brms's 'mubc' (brms itself renames the second 'mubc1')
    with pytest.raises(ab.FormulaError, match=r"the response categories \['b c', 'b-c'\] would share brms's parameter name 'mubc'"):
        ab.categorical("c ~ x", pd.DataFrame({"c": ["a", "b c", "b-c", "a"], "x": [0.1, -0.3, 0.8, 0.2]}), categorical_priors())
    m = ab.categorical("c ~ x", pd.DataFrame({"c": ["a b", "a-b", "z", "z"], "x": [0.1, -0.3, 0.8, 0.2]}), categorical_priors())
    assert m.categories == ("a b", "a-b", "z")                             # the reference's name plays no part in brms


def test_formula_refusals():
    for formula in ("y ~ 0 + x", "y ~ x - 1"):
        with pytest.raises(ab.FormulaError, match="the cumulative family's thresholds are its intercepts, so the intercept cannot "
                                                  "be removed .*'Cannot remove the intercept in an ordinal model'"):
            ab.cumulative(formula, ordinal_data(), ab.Priors(b=ab.Normal(0, 1)))
    with pytest.raises(ab.FormulaError, match=r"takes population-level terms only in this slice; a group-level term \('\(1 \| g\)'\)"):
        ab.categorical("c ~ x + (1 | g)", categorical_data().assign(g=list("aabbccaabbcc")), categorical_priors(sd=ab.HalfNormal(1)))
    with pytest.raises(ab.FormulaError, match="the categorical family takes no offset term in this slice"):
        ab.categorical("c ~ x + offset(e)", categorical_data().assign(e=1.0), categorical_priors())
    with pytest.raises(ab.FormulaError, match="Cannot remove the intercept"):
        ab.default_priors("y ~ 0 + x", ordinal_data(), family="cumulative")
    with pytest.raises(ab.FormulaError, match="group-level term"):
        ab.default_priors("c ~ x + (1 | g)", categorical_data().assign(g=list("aabbccaabbcc")), family="categorical")


# -- the model: graph, density and gradient by hand ------------------------------------------------------------------------
def _graph(m, point, names):
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n] for n in names]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    return [np.asarray(v, dtype=float) for v in fn(point)]


@pytest.mark.parametrize("center", [True, False])
def test_cumulative_density_graph_and_coordinates_by_hand(center):
    df = ordinal_data()
    m = cumulative_model(center_predictors=center)
    tau_c, b, sd, z = np.array([-0.8, 0.1, 1.3]), 0.6, 0.7, np.array([0.3, -0.5, 0.2])
    p = m.value_point(intercept=tau_c, b=[b], sd=sd, z=z)
    np.testing.assert_allclose(p["Intercept_ordered__"], [-0.8, math.log(0.9), math.log(1.2)], rtol=1e-15)
    x, y = df.x.to_numpy(), df.y.to_numpy()
    eta = b * x + sd * z[pd.Categorical(df.g).codes]                     # mu: the raw design, no intercept
    tau = tau_c + (x.mean() * b if center else 0.0)                       # brms's b_Intercept
    mu, b_int = _graph(m, p, ["mu", "b_Intercept"])
    np.testing.assert_allclose(mu, eta, rtol=1e-14)
    np.testing.assert_allclose(b_int, tau, rtol=1e-14)
    lik = np.log(lk.cumulative_probabilities(eta, tau)[np.arange(len(y)), y - 1]).sum()
    prior = norm.logpdf(tau_c, 0, 3).sum() + norm.logpdf(b, 0, 1.5) + halfnorm.logpdf(sd) + norm.logpdf(z).sum()
    jac = np.log(np.diff(tau_c)).sum() + np.log(sd)                        # the ordered transform's log gaps, log(sd)
    pmm = m.pymc_model
    np.testing.assert_allclose(pmm.compile_logp()(p), lik + prior + jac, rtol=1e-12)
    np.testing.assert_allclose(pmm.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-12)
    np.testing.assert_allclose(pmm.compile_logp(vars=[pmm["Intercept"]], jacobian=False)(p), norm.logpdf(tau_c, 0, 3).sum(), rtol=1e-14)
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    args = [p[n] for n in names]
    np.testing.assert_allclose(jf(args), lik + prior + jac, rtol=1e-12)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float), rtol=1e-10, atol=1e-12)


def test_categorical_density_graph_and_coordinates_by_hand():
    df = categorical_data()
    m = ab.categorical("c ~ x + z", df, categorical_priors(), categorical={"z": "p"})
    a = np.array([0.2, -0.4])
    b = np.array([[0.5, -0.3], [0.1, 0.7], [-0.6, 0.2]])                 # rows x, z[q], z[r]; columns b, c
    p = m.value_point(intercept=a, b=b)
    assert p["x"].shape == (2,) and p["z"].shape == (2, 2)
    np.testing.assert_array_equal(p["z"], b[1:])
    Xr = np.column_stack([df.x, df.z == "q", df.z == "r"]).astype(float)
    eta = a + (Xr - Xr.mean(0)) @ b                                       # (n, 2): the centred design, one column per category
    (mu,) = _graph(m, p, ["mu"])
    np.testing.assert_allclose(mu, eta, rtol=1e-14)
    y = m.design.y.astype(int)
    full = np.column_stack([np.zeros(len(y)), eta])
    lik = (full[np.arange(len(y)), y - 1] - np.log(np.exp(full).sum(1))).sum()
    prior = norm.logpdf(a, 0, 3).sum() + norm.logpdf(b, 0, 1.5).sum()
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), lik + prior, rtol=1e-12)
    with pytest.raises(ValueError, match=r"b must be an array of shape \(3, 2\): one row per design column"):
        m.value_point(intercept=a, b=b.ravel())
    with pytest.raises(ValueError, match=r"intercept must have 2 finite entries, one per non-reference category \['b', 'c'\]"):
        m.value_point(intercept=[0.1], b=b)


def test_jax_density_and_gradient_of_larger_models():
    """300 rows: the one-hot selections and the log-sum-exp lower to JAX at any size (PyTensor's take_along_axis would
    not), and the JAX density and gradient equal PyTensor's."""
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.normal(size=300), "y": rng.integers(1, 6, 300), "c": rng.choice(list("pqrs"), 300)})
    for m, point in ((ab.cumulative("y ~ x", df, ordinal_priors(False)), dict(intercept=[-1.5, -0.2, 0.4, 1.9], b=[0.8])),
                     (ab.categorical("c ~ x", df, categorical_priors()), dict(intercept=[0.3, -0.2, 0.1], b=[[0.5, -0.4, 0.9]]))):
        p = m.value_point(**point)
        pmm = m.pymc_model
        names = [v.name for v in pmm.value_vars]
        jf = get_jaxified_logp(pmm)
        args = [p[n] for n in names]
        np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(p)), rtol=1e-12)
        g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
        np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float), rtol=1e-10, atol=1e-12)


def test_names_value_point_and_refusals():
    m = cumulative_model()
    assert m.names["linpred"] == "mu" and m.names["mean"] == "p" and m.names["thresholds"] == "b_Intercept"
    assert m.names["threshold_dim"] == "Intercept_dim" and m.link == "logit" and m.family_spec.discrete
    assert [v.name for v in m.pymc_model.value_vars] == ["Intercept_ordered__", "x", "1|g_sigma_log__", "1|g_offset"]
    assert tuple(m.pymc_model.coords["Intercept_dim"]) == ("1", "2", "3")
    for bad in ([0.1, 0.0, 0.5], [0.1, 0.2], [0.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match="intercept must be the 3 thresholds, finite and strictly increasing"):
            m.value_point(intercept=bad, b=[0.1], sd=0.5, z=np.zeros(3))
    with pytest.raises(ValueError, match="sigma, shape and phi do not apply to the cumulative family"):
        m.value_point(intercept=[0, 1, 2], b=[0.1], sd=0.5, z=np.zeros(3), sigma=1.0)
    c = categorical_model()
    assert c.names["category_dim"] == "c_reduced_dim" and "thresholds" not in c.names
    assert tuple(c.pymc_model.coords["c_reduced_dim"]) == ("b", "c")
    with pytest.raises(ab.PriorError, match="the categorical family has no residual SD; do not pass sigma"):
        categorical_model(priors=categorical_priors(sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="an intercept prior is required"):
        cumulative_model(priors=ab.Priors(b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match="must be a location prior"):
        cumulative_model(priors=ab.Priors(intercept=ab.HalfNormal(1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    for column in ("mu", "b_Intercept"):                                   # the graph's names cannot be columns
        with pytest.raises(ab.FormulaError, match="collision"):
            cumulative_model(f"y ~ {column}", frame=ordinal_data().rename(columns={"x": column}), priors=ordinal_priors(False))


def test_other_terms_compose():
    """The term model is shared: categorical predictors, interactions, transforms, the offset (which enters mu), independent
    and correlated slopes and several factors build for the cumulative family; categorical predictors, interactions and
    no intercept for the categorical family."""
    df = ordinal_data().assign(h=list("uvuvuvuvuvuv"), xpos=np.exp(X))
    slopes = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
    for formula, extra, kw in (("y ~ x + z + x:z", {}, {"categorical": {"z": "p"}}), ("y ~ x + log(xpos)", {}, {}),
                               ("y ~ x + (1 + x || g)", {"sd": slopes}, {}), ("y ~ x + (1 + x | g)", {"sd": slopes, "cor": ab.LKJ(2)}, {}),
                               ("y ~ x + (1 | g) + (1 | h)", {"sd": {"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}}, {}),
                               ("y ~ 1 + (1 | g)", {"sd": ab.HalfNormal(1)}, {})):
        pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1) if "~ 1" not in formula else None, **extra)
        m = ab.cumulative(formula, df, pri, **kw)
        assert m.family_spec.thresholds and "Intercept_ordered__" in {v.name for v in m.pymc_model.value_vars}
    m = cumulative_model("y ~ x + offset(e)", priors=ordinal_priors(False))
    p = m.value_point(intercept=[-1, 0, 1], b=[0.4])
    x = ordinal_data().x.to_numpy()
    np.testing.assert_allclose(_graph(m, p, ["mu"])[0], 0.4 * x + ordinal_data().e.to_numpy(), rtol=1e-14)
    np.testing.assert_allclose(_graph(m, p, ["b_Intercept"])[0], np.array([-1, 0, 1]) + 0.4 * x.mean(), rtol=1e-14)
    assert "offset_i" in m.describe()
    for formula, kw in (("c ~ x + z + x:z", {"categorical": {"z": "p"}}), ("c ~ 0 + x + z", {"categorical": {"z": "p"}})):
        pri = ab.Priors(intercept=None if "~ 0" in formula else ab.Normal(0, 3), b=ab.Normal(0, 1))
        m = ab.categorical(formula, categorical_data(), pri, **kw)
        assert m.pymc_model.named_vars_to_dims["z"] == ("z_dim", "c_reduced_dim")
    m = ab.categorical("c ~ 0 + x + z", categorical_data(), ab.Priors(b=ab.Normal(0, 1)), categorical={"z": "p"})
    assert m.spec.columns == ("x", "z[p]", "z[q]", "z[r]") and "Intercept" not in m.pymc_model.named_vars


# -- predictions ---------------------------------------------------------------------------------------------------------
def cumulative_draws(tau=(-0.5, 0.4, 1.2), b=0.6, group=True):
    post = {"Intercept": np.broadcast_to(np.array(tau) - 1.0, (2, 4, 3)), "b_Intercept": np.broadcast_to(tau, (2, 4, 3)),
            "x": np.full((2, 4), b)}
    dims = {"Intercept": ["Intercept_dim"], "b_Intercept": ["Intercept_dim"]}
    if group:
        post.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.2, -0.3, 0.1], (2, 4, 3))})
        dims["1|g"] = ["g__factor_dim"]
    return az.from_dict(posterior=post, coords={"g__factor_dim": ["a", "b", "c"], "Intercept_dim": ["1", "2", "3"]}, dims=dims)


def test_cumulative_predictions_by_hand():
    m = cumulative_model()
    df = ordinal_data()
    u = np.array([0.2, -0.3, 0.1])[pd.Categorical(df.g).codes]
    eta = np.broadcast_to(0.6 * df.x.to_numpy() + u, (2, 4, 12))
    lin = m.predict(cumulative_draws(), kind="linear")
    np.testing.assert_allclose(lin.eta, eta)                              # no thresholds in mu
    assert lin.eta.dims == ("chain", "draw", "__obs__") and lin.attrs["link"] == "logit"
    out = m.predict(cumulative_draws(), kind="response", seed=7)
    probs = lk.cumulative_probabilities(eta, np.array([-0.5, 0.4, 1.2]))
    assert out.p.dims == ("chain", "draw", "__obs__", "category") and list(out.category.values) == ["1", "2", "3", "4"]
    np.testing.assert_allclose(out.p, probs, rtol=1e-14)
    np.testing.assert_allclose(out.p.sum("category"), 1.0, rtol=1e-14)
    np.testing.assert_array_equal(out.y, lk.draw_categories(probs, np.random.default_rng(7)))
    assert out.y.dtype == np.int64 and set(np.unique(out.y)) <= {1, 2, 3, 4}
    assert json.loads(out.attrs["categories_json"]) == ["1", "2", "3", "4"] and set(out.data_vars) == {"p", "y"}
    nogroup = m.predict(cumulative_draws(), kind="mean", include_group_effects=False)
    np.testing.assert_allclose(nogroup.p, lk.cumulative_probabilities(np.broadcast_to(0.6 * df.x.to_numpy(), (2, 4, 12)),
                                                                       np.array([-0.5, 0.4, 1.2])), rtol=1e-14)
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["new", "a"]})
    for policy in ("population", "resample_observed"):
        rng = np.random.default_rng(8)
        effect = rng.normal(size=(2, 4)) * 0.5 if policy == "population" else np.take([0.2, -0.3, 0.1], rng.integers(0, 3, size=(2, 4)))
        res = m.predict(cumulative_draws(), newdata=new, kind="linear", new_groups=policy, seed=8)
        np.testing.assert_allclose(res.eta[..., 0], 0.6 * 2.0 + effect)
        np.testing.assert_allclose(res.eta[..., 1], 0.6 * -1.0 + 0.2)
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(cumulative_draws(), newdata=new, kind="mean")
    with pytest.raises(ValueError, match="kind must be 'mean' .* or 'linear' \\(mu\\)"):
        m.predict(cumulative_draws(), kind="probability")


def test_categorical_predictions_by_hand():
    df = categorical_data()
    m = ab.categorical("c ~ x + z", df, categorical_priors(), categorical={"z": "p"})
    a, bx, bz = np.array([0.2, -0.4]), np.array([0.5, -0.3]), np.array([[0.1, 0.7], [-0.6, 0.2]])
    # the posterior's coefficient of z with its axes as Bambi may leave them after sampling (categories before levels)
    post = az.from_dict(posterior={"Intercept": np.broadcast_to(a, (2, 4, 2)), "x": np.broadcast_to(bx, (2, 4, 2)),
                                   "z": np.broadcast_to(bz.T, (2, 4, 2, 2))},
                        coords={"c_reduced_dim": ["b", "c"], "z_dim": ["q", "r"]},
                        dims={"Intercept": ["c_reduced_dim"], "x": ["c_reduced_dim"], "z": ["c_reduced_dim", "z_dim"]})
    Xr = np.column_stack([df.x, df.z == "q", df.z == "r"]).astype(float)
    eta = a + Xr @ np.vstack([bx, bz])                                     # raw design, raw-scale intercepts
    lin = m.predict(post, kind="linear")
    assert lin.eta.dims == ("chain", "draw", "__obs__", "category") and list(lin.category.values) == ["b", "c"]
    np.testing.assert_allclose(lin.eta.values[0, 0], eta, rtol=1e-14)
    out = m.predict(post, kind="response", seed=3)
    probs = softmax(np.column_stack([np.zeros(12), eta]), axis=1)
    assert list(out.category.values) == ["a", "b", "c"]
    np.testing.assert_allclose(out.p.values[1, 2], probs, rtol=1e-14)
    np.testing.assert_array_equal(out.y, lk.draw_categories(np.broadcast_to(probs, (2, 4, 12, 3)), np.random.default_rng(3)))
    new = pd.DataFrame({"x": [0.3], "z": ["r"]})
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="linear").eta.values[0, 0, 0], a + 0.3 * bx + bz[1], rtol=1e-14)
    with pytest.raises(ValueError, match="new_groups does not apply"):
        m.predict(post, kind="mean", new_groups="population")


def test_prior_predictive():
    m = cumulative_model()
    out = m.prior_predictive(draws=40, seed=3)
    tau = out.prior["Intercept"].values[0]
    assert tau.shape == (40, 3) and (np.diff(tau, axis=-1) > 0).all()           # ordered draws: the sorted prior draws
    y = out.prior_predictive["y"].values
    assert y.shape == (1, 40, 12) and y.min() >= 1 and y.max() <= 4 and out.attrs["autobayes_generation_kind"] == "prior_predictive"
    c = categorical_model().prior_predictive(draws=40, seed=3)
    assert c.prior["Intercept"].shape == (1, 40, 2) and set(np.unique(c.prior_predictive["c"].values)) <= {1, 2, 3}
    flat = cumulative_model(priors=ab.Priors(intercept=ab.Flat(), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"the prior of \['intercept'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)


# -- describe, identity, defaults, record ----------------------------------------------------------------------------------
def test_describe_and_identity():
    m = cumulative_model()
    text = m.describe()
    assert text.startswith("AutoBayes cumulative (ordinal) model") and "family    : cumulative, logit link" in text
    assert "mu_i = x_x * b_x + u[g_i]" in text and "P(y_i <= k) = expit(b_Intercept[k] - mu_i), k = 1..3" in text
    assert "Intercept[k] ~ Normal(mu=0, sigma=3), for each of the 3 thresholds" in text and "Intercept_ordered__" in text
    assert "response categories 1=1, 2=2, 3=3, 4=4 (ordered)" in text
    same = ab.Model("y ~ x + (1 | g)", ordinal_data(), ordinal_priors(), family="cumulative")
    assert same.describe() == text and same.target_identity == m.target_identity
    assert cumulative_model(center_predictors=False).target_identity != m.target_identity
    labels = ["lo", "mid", "hi", "top"]
    ordered = ordinal_data().assign(y=pd.Categorical([labels[v - 1] for v in ordinal_data().y], categories=labels, ordered=True))
    assert cumulative_model(frame=ordered).target_identity != m.target_identity       # the same codes, other categories
    c = categorical_model()
    text = c.describe()
    assert "response categories 1=a, 2=b, 3=c (the reference is 'a')" in text and "brms's mub, muc" in text
    assert "a_c[k] ~ Normal(mu=0, sigma=3)" in text and "b_x[k] ~ Normal(mu=0, sigma=1.5)" in text
    relabelled = categorical_data().assign(c=categorical_data().c.map({"a": "a", "b": "b", "c": "d"}))
    assert categorical_model(frame=relabelled).target_identity != c.target_identity


# brms 2.23.0's default_prior() (R 4.5.2, C collation) for these models, as (class, coef, dpar, prior); every source "default"
BRMS_DEFAULTS = {
    "cumulative, 12 categories": ("y ~ x", "cumulative", {}, pd.DataFrame({"y": list(range(1, 13)) * 2, "x": np.linspace(-1, 1, 24)}),
                                  [("Intercept", "", "", "student_t(3, 0, 2.5)")]
                                  + [("Intercept", k, "", "") for k in ("1", "10", "11", "2", "3", "4", "5", "6", "7", "8", "9")]
                                  + [("b", "", "", ""), ("b", "x", "", "")]),
    "cumulative, not centred": ("y ~ x", "cumulative", {"center_predictors": False}, ordinal_data(),
                                [("Intercept", "", "", "student_t(3, 0, 2.5)"), ("Intercept", "1", "", ""), ("Intercept", "2", "", ""),
                                 ("Intercept", "3", "", ""), ("b", "", "", ""), ("b", "x", "", "")]),
    "cumulative, ordered factor": ("y ~ x", "cumulative", {},
                                   pd.DataFrame({"y": pd.Categorical(["lo", "mid", "hi", "lo"], categories=["lo", "mid", "hi"], ordered=True),
                                                 "x": [0.1, -0.2, 0.7, 1.1]}),
                                   [("Intercept", "", "", "student_t(3, 0, 2.5)"), ("Intercept", "1", "", ""), ("Intercept", "2", "", ""),
                                    ("b", "", "", ""), ("b", "x", "", "")]),
    "categorical, not centred": ("c ~ x", "categorical", {"center_predictors": False}, categorical_data(),
                                 [("b", "", "mub", ""), ("b", "Intercept", "mub", ""), ("b", "x", "mub", ""),
                                  ("b", "", "muc", ""), ("b", "Intercept", "muc", ""), ("b", "x", "muc", "")]),
    "categorical, labels to names": ("c ~ x", "categorical", {},
                                     pd.DataFrame({"c": ["a b", "x-y", "Z", "1.5", "_u", "Z"], "x": [0.1, -0.2, 0.7, 1.1, -0.4, 0.3]}),
                                     [row for dp in ("muZ", "muab", "muu", "muxy")
                                      for row in (("Intercept", "", dp, "student_t(3, 0, 2.5)"), ("b", "", dp, ""), ("b", "x", dp, ""))]),
    "categorical, integer codes": ("c ~ x", "categorical", {}, pd.DataFrame({"c": [3, 1, 2, 1], "x": [0.1, -0.2, 0.7, 1.1]}),
                                   [row for dp in ("mu2", "mu3")
                                    for row in (("Intercept", "", dp, "student_t(3, 0, 2.5)"), ("b", "", dp, ""), ("b", "x", dp, ""))]),
}


@pytest.mark.parametrize("label", sorted(BRMS_DEFAULTS))
def test_default_priors_are_brms_default_prior(label):
    formula, family, kwargs, frame, rows = BRMS_DEFAULTS[label]
    pri = ab.default_priors(formula, frame, family=family, **kwargs)
    m = getattr(ab, family)(formula, frame, pri, **kwargs)
    table = m.prior_table()
    assert [(r["class"], r["coef"], r["dpar"], r["prior"]) for r in table] == rows
    assert all(r["source"] == "default" for r in table) and all(r["lb"] == r["ub"] == "" for r in table)
    assert m.default_prior_labels[:2] == ["intercept", "b:x"]
    with pytest.raises(ab.PriorError, match=f"the {family} family's prior table needs the response categories"):
        pri.table(m.spec, m.family_spec)


def test_fit_record_and_summary_rows(monkeypatch):
    m = cumulative_model()
    post = cumulative_draws()
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # constant synthetic draws: R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert {"Intercept[1]", "Intercept[3]", "x", "1|g_sigma"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "cumulative" and rec["model"]["link"] == "logit"
    assert rec["model"]["priors"]["intercept"] == "Normal(mu=0, sigma=3)"
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
    assert get_family("cumulative").thresholds and get_family("categorical").per_category
