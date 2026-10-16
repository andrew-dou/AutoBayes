"""More ordinal models (brms-parity D04, docs/ordinal-categorical.md): brms's ``sratio``, ``cratio`` and ``acat`` families under
the logit link, category-specific effects ``cs(x)``, and the cumulative family's probit link, built but withheld (D88).

The row densities and their gradients against a 50-digit reference written from the published definitions, on both paths
(PyTensor's gradient and JAX's autodiff of the jaxified graph, what NumPyro differentiates); the models' graph, density and
gradient by hand (unordered thresholds, ``mu_cs``), ``value_point``, the formula and data refusals, by-hand predictions with the
``threshold`` dimension, ``log_lik`` and ``bayes_R2``, prior-predictive draws, ``describe()``, the target identity, brms's
default prior tables, and the probit link's park. No MCMC; the brms comparison is the oracle cases ``sratio_varint``,
``cratio_varint``, ``acat_varint``, ``acat_cs``, ``sratio_cs``, ``cratio_cs`` (and ``cumulative_probit_varint``, gated)."""
import json
import re
from decimal import Context, Decimal, localcontext
from types import MappingProxyType

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import expit, ndtr
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import FAMILIES, get_family

FAMS = ("sratio", "cratio", "acat")
KIND_FAMILY = {"sratio_logit": "sratio", "cratio_logit": "cratio", "acat_logit": "acat", "cumulative_probit": "probit"}

# -- data -------------------------------------------------------------------------------------------------------------
X = [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7, 0.9, -1.4, 0.2, 1.6]


def data():
    return pd.DataFrame({"x": X, "y": [1, 2, 4, 1, 2, 4, 1, 1, 4, 2, 3, 4], "g": list("aabbccaabbcc"), "z": list("pqrpqrpqrpqr"),
                         "w": np.round(np.cos(np.arange(12.0)), 4), "v": np.round(np.sin(np.arange(12.0)), 4),
                         "e": np.round(np.linspace(0.5, 2.0, 12), 4)})


def priors(group=True, **kw):
    return ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), **({"sd": ab.HalfNormal(1)} if group else {}), **kw)


def model(family, formula="y ~ x + (1 | g)", frame=None, pri=None, **kw):
    return getattr(ab, family)(formula, data() if frame is None else frame, pri or priors("|" in formula), **kw)


# -- references written from the definitions ------------------------------------------------------------------------------
def reference_log_probabilities(family, eta, tau):
    """``log P(y = k)``, ``(..., K)``, from the definitions (NumPy): ``eta`` ``(..., K - 1)`` (one predictor per threshold),
    ``tau`` ``(K - 1,)``. sratio: P(y = k | y >= k) = expit(tau_k - eta_k); cratio: P(y > k | y >= k) = expit(eta_k - tau_k);
    acat: log P(y = k + 1) / P(y = k) = eta_k - tau_k; cumulative probit: P(y <= k) = Phi(tau_k - eta_k)."""
    eta = np.asarray(eta, dtype=float)
    K1 = eta.shape[-1]
    if family in ("sratio", "cratio"):
        stop = expit(tau - eta) if family == "sratio" else 1.0 - expit(eta - tau)
        p, surv = [], np.ones(eta.shape[:-1])
        for k in range(K1):
            p.append(surv * stop[..., k])
            surv = surv * (1.0 - stop[..., k])
        return np.log(np.stack(p + [surv], axis=-1))
    if family == "acat":
        s = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), np.cumsum(eta - tau, axis=-1)], axis=-1)
        return s - np.log(np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True)) - s.max(-1, keepdims=True)
    c = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), ndtr(tau - eta), np.ones(eta.shape[:-1] + (1,))], axis=-1)
    return np.log(np.diff(c, axis=-1))


_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798")
_CTX = Context(prec=90, Emin=-9999999, Emax=9999999)


def _erfc(x: Decimal) -> Decimal:
    """erfc(x), x >= 0, to far more than 50 digits (at 130 digits it agrees with mpmath to 1e-81 on |x| <= 707): the series of
    positive terms of erf below 5, the Laplace continued fraction (400 terms) from 5 on."""
    with localcontext(_CTX):
        if x < 5:
            x2, term, s, n = x * x, x, x, 0
            while term > Decimal(10) ** -88 * s:
                n += 1
                term = term * 2 * x2 / (2 * n + 1)
                s += term
            return 1 - 2 / _PI.sqrt() * (-x2).exp() * s
        f = x
        for k in range(400, 0, -1):
            f = x + Decimal(k) / 2 / f
        return (-x * x).exp() / _PI.sqrt() / f


def _phi_mass(a, b) -> Decimal:
    """Phi(b) - Phi(a) for a < b (either may be None: -inf, inf), from the tail where it lies."""
    with localcontext(_CTX):
        r2 = Decimal(2).sqrt()
        Q = lambda t: _erfc(t / r2) / 2                                  # noqa: E731
        P = lambda t: _erfc(-t / r2) / 2                                 # noqa: E731, t < 0
        if a is not None and a > 0:
            return Q(a) - (Q(b) if b is not None else 0)
        if b is not None and b < 0:
            return P(b) - (P(a) if a is not None else 0)
        return 1 - (Q(b) if b is not None else 0) - (P(a) if a is not None else 0)


def exact_row(kind, y, eta, tau) -> Decimal:
    """log P(y | eta, tau) at 50+ digits from the definitions, ``eta`` and ``tau`` Decimals (``eta`` one per threshold)."""
    with localcontext(_CTX):
        d = [t - e for t, e in zip(tau, eta)]
        K1 = len(d)
        if kind in ("sratio_logit", "cratio_logit"):
            # sratio: log expit(d_y) + sum log(1 - expit(d_j)); cratio: log(1 - expit(-d_y)) + sum log expit(-d_j), the same
            # numbers once 1 - expit(-d) is written expit(d) (an identity, so that nothing cancels at 50 digits)
            out = -(1 + (-d[y - 1]).exp()).ln() if y <= K1 else Decimal(0)
            return out - sum(((1 + dj.exp()).ln() for dj in d[:y - 1]), Decimal(0))
        if kind == "acat_logit":
            s = [Decimal(0)]
            for dj in d:
                s.append(s[-1] - dj)
            m = max(s)
            return s[y - 1] - m - sum((v - m).exp() for v in s).ln()
        a = d[y - 2] if y >= 2 else None
        b = d[y - 1] if y <= K1 else None
        return _phi_mass(a, b).ln()


_EXACT: dict = {}


def exact_row_and_gradient(kind, y, eta, tau):
    """(value, d/d eta, d/d tau) at 50+ digits: central differences of ``exact_row`` at 90 digits with step 1e-30 (truncation
    error below 1e-55, rounding below 1e-55 relative to the derivative's scale);
    cached, as both paths compare with the same values."""
    key = (kind, y, float(eta), tuple(float(v) for v in tau))
    if key not in _EXACT:
        _EXACT[key] = _exact_row_and_gradient(kind, y, eta, tau)
    return _EXACT[key]


def _exact_row_and_gradient(kind, y, eta, tau):
    e, t = Decimal(float(eta)), [Decimal(float(v)) for v in tau]
    h = Decimal("1e-30")
    value = exact_row(kind, y, [e] * len(t), t)
    with localcontext(_CTX):
        d_eta = (exact_row(kind, y, [e + h] * len(t), t) - exact_row(kind, y, [e - h] * len(t), t)) / (2 * h)
        d_tau = [(exact_row(kind, y, [e] * len(t), t[:j] + [t[j] + h] + t[j + 1:])
                  - exact_row(kind, y, [e] * len(t), t[:j] + [t[j] - h] + t[j + 1:])) / (2 * h) for j in range(len(t))]
    return float(value), float(d_eta), [float(v) for v in d_tau]


ETAS = [-1000.0, -40.0, -3.0, -0.5, 0.0, 0.7, 3.0, 40.0, 1000.0]
TAU_UNORDERED = np.array([0.4, -1.2, -1.2 + 1e-9, 2.0])           # five categories; unordered, one gap of 1e-9
TAU_PROBIT = np.array([-1.2, 0.3, 2.0])                            # four categories, ordered


def _rows(kind, mode, y, eta, tau):
    """Each row's log density, its derivative in its own eta and its gradient in the thresholds, from one compiled graph: each
    row reads its own copy of the thresholds (a row of ``tv``), so the gradient of the sum is every row's own."""
    yv, ev, tv = pt.lvector("y"), pt.dvector("eta"), pt.dmatrix("tau")
    rows = lk.ordinal_logp_rows(kind, yv, lk.ordinal_gaps(ev, tv, False))
    taus = np.tile(np.asarray(tau, dtype=float), (len(eta), 1))
    if mode == "pytensor":
        f = pytensor.function([yv, ev, tv], [rows, pt.grad(rows.sum(), ev), pt.grad(rows.sum(), tv)])
        return [np.asarray(a, dtype=float) for a in f(y, eta, taus)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[yv, ev, tv], outputs=[rows])
    return [np.asarray(a, dtype=float) for a in (jf(y, eta, taus)[0], jax.grad(lambda e: jf(y, e, taus)[0].sum())(eta),
                                                 jax.grad(lambda t: jf(y, eta, t)[0].sum())(taus))]


def _grid(K):
    """Every category at every eta of ETAS."""
    return np.repeat(np.arange(1, K + 1, dtype=np.int64), len(ETAS)), np.tile(np.array(ETAS), K)


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
@pytest.mark.parametrize("kind", ["sratio_logit", "cratio_logit", "acat_logit"])
def test_rows_are_exact_and_never_nan_on_both_paths(kind, mode):
    y, eta = _grid(len(TAU_UNORDERED) + 1)
    value, d_eta, d_tau = _rows(kind, mode, y, eta, TAU_UNORDERED)
    assert np.isfinite(value).all() and np.isfinite(d_eta).all() and np.isfinite(d_tau).all()
    for i, (yi, e) in enumerate(zip(y, eta)):
        want_v, want_e, want_t = exact_row_and_gradient(kind, int(yi), e, TAU_UNORDERED)
        assert abs(value[i] - want_v) <= 1e-12 * max(1.0, abs(want_v)), (kind, mode, yi, e, value[i], want_v)
        assert abs(d_eta[i] - want_e) <= 1e-12 * max(1.0, abs(want_e)), (kind, mode, yi, e, d_eta[i], want_e)
        np.testing.assert_allclose(d_tau[i], want_t, rtol=1e-9, atol=1e-12 * max(1.0, max(abs(v) for v in want_t)))


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_probit_rows_are_exact_and_never_nan_on_both_paths(mode):
    """The withheld cumulative probit link's rows (D88): exact to 1e-12 for thresholds 1.5 apart and more. Two thresholds
    closer than about 1e-6 in a far tail keep about ``|tau - eta| eps / gap`` relative accuracy, the log Normal mass between
    them being a difference of two log tail probabilities (likelihoods.normal_log_mass); Stan's is no better there."""
    y, eta = _grid(len(TAU_PROBIT) + 1)
    value, d_eta, d_tau = _rows("cumulative_probit", mode, y, eta, TAU_PROBIT)
    assert np.isfinite(value).all() and np.isfinite(d_eta).all() and np.isfinite(d_tau).all()
    for i, (yi, e) in enumerate(zip(y, eta)):
        want_v, want_e, want_t = exact_row_and_gradient("cumulative_probit", int(yi), e, TAU_PROBIT)
        assert abs(value[i] - want_v) <= 1e-12 * max(1.0, abs(want_v)), (mode, yi, e, value[i], want_v)
        assert abs(d_eta[i] - want_e) <= 1e-11 * max(1.0, abs(want_e)), (mode, yi, e, d_eta[i], want_e)
        np.testing.assert_allclose(d_tau[i], want_t, rtol=1e-9, atol=1e-11 * max(1.0, max(abs(v) for v in want_t)))
    close = np.array([-1.2, -1.2 + 1e-9, 2.0])                        # the middle category of width 1e-9 at |tau - eta| <= 3
    value, _, _ = _rows("cumulative_probit", mode, np.full(5, 2, dtype=np.int64), np.array([-3.0, -0.5, 0.0, 0.7, 3.0]), close)
    for v, e in zip(value, [-3.0, -0.5, 0.0, 0.7, 3.0]):
        want = exact_row_and_gradient("cumulative_probit", 2, e, close)[0]
        assert abs(v - want) <= 5 * abs(-1.2 - e) * 2.3e-16 / 1e-9 + 1e-12 * abs(want), (mode, e, v, want)


def test_rows_outside_the_categories_have_no_mass_and_match_the_probabilities():
    for kind in lk.ORDINAL_KINDS:
        tau = TAU_PROBIT if kind == "cumulative_probit" else TAU_UNORDERED
        K = len(tau) + 1
        value, _, _ = _rows(kind, "pytensor", np.array([0, K + 1, -1], dtype=np.int64), np.zeros(3), tau)
        assert (value == -np.inf).all()
        value, _, _ = _rows(kind, "pytensor", np.arange(1, K + 1, dtype=np.int64), np.full(K, 0.4), tau)
        np.testing.assert_allclose(value, lk.ordinal_log_probabilities(kind, tau - 0.4), rtol=1e-12)
        assert abs(np.exp(value).sum() - 1) < 1e-14
        eta = np.array([[0.4], [-1.3]]) + 0 * tau                          # (2, K - 1)
        np.testing.assert_allclose(lk.ordinal_log_probabilities(kind, tau - eta), reference_log_probabilities(KIND_FAMILY[kind], eta, tau),
                                   rtol=1e-12)
    with pytest.raises(ValueError, match="kind must be one of"):
        lk.ordinal_log_probabilities("cumulative_cloglog", np.zeros(3))


def test_likelihood_objects_draw_their_distribution():
    import pymc as pm

    eta, tau = np.array([-1.0, 0.4, 1.5]), np.array([0.5, -0.2, 1.0])
    for kind in ("sratio_logit", "acat_logit"):
        logp, random = lk._ordinal_dist(kind, False)
        draws = pm.draw(pm.CustomDist.dist(eta, tau, logp=logp, random=random, signature="(),(k)->()", dtype="int64"),
                        draws=20000, random_seed=3)
        p = np.exp(lk.ordinal_log_probabilities(kind, tau - eta[:, None]))
        assert draws.shape == (20000, 3) and draws.dtype == np.int64 and draws.min() >= 1 and draws.max() <= 4
        for k in range(4):
            assert np.all(np.abs((draws == k + 1).mean(0) - p[:, k]) < 4.5 * np.sqrt(p[:, k] * (1 - p[:, k]) / 20000))


# -- the models: graph, density and gradient by hand -----------------------------------------------------------------------
def _graph(m, point, names):
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n] for n in names]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    return [np.asarray(v, dtype=float) for v in fn(point)]


def _jax_equals_pytensor(m, p):
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    pmm = m.pymc_model
    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    args = [p[n] for n in names]
    np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(p)), rtol=1e-12)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("family", FAMS)
def test_density_graph_and_coordinates_by_hand(family, center):
    df = data()
    m = model(family, center_predictors=center)
    tau_c, b, sd, z = np.array([0.6, -0.8, 1.3]), 0.6, 0.7, np.array([0.3, -0.5, 0.2])     # unordered thresholds are fine
    p = m.value_point(intercept=tau_c, b=[b], sd=sd, z=z)
    np.testing.assert_array_equal(p["Intercept"], tau_c)                  # sampled as they are, no transform
    x, y = df.x.to_numpy(), df.y.to_numpy()
    eta = b * x + sd * z[pd.Categorical(df.g).codes]                     # mu: the raw design, no intercept
    tau = tau_c + (x.mean() * b if center else 0.0)                       # brms's b_Intercept
    mu, b_int = _graph(m, p, ["mu", "b_Intercept"])
    np.testing.assert_allclose(mu, eta, rtol=1e-14)
    np.testing.assert_allclose(b_int, tau, rtol=1e-14)
    lp = reference_log_probabilities(family, eta[:, None] + 0 * tau, tau)
    lik = lp[np.arange(len(y)), y - 1].sum()
    prior = norm.logpdf(tau_c, 0, 3).sum() + norm.logpdf(b, 0, 1.5) + halfnorm.logpdf(sd) + norm.logpdf(z).sum()
    pmm = m.pymc_model
    np.testing.assert_allclose(pmm.compile_logp()(p), lik + prior + np.log(sd), rtol=1e-12)       # the Jacobian: log(sd) only
    np.testing.assert_allclose(pmm.compile_logp(jacobian=False)(p), lik + prior, rtol=1e-12)
    np.testing.assert_allclose(pmm.compile_logp(vars=[pmm["Intercept"]], jacobian=False)(p), norm.logpdf(tau_c, 0, 3).sum(), rtol=1e-14)
    _jax_equals_pytensor(m, p)


@pytest.mark.parametrize("family", FAMS)
def test_category_specific_effects_by_hand(family):
    """cs(w) and cs(v): one coefficient per threshold each (brms's bcs rows), never centred, in every threshold's predictor
    mu_cs = mu + w bcs_w + v bcs_v, under the b prior; the thresholds are centred on the population-level design alone."""
    df = data()
    m = model(family, "y ~ x + cs(w) + cs(v) + (1 | g)")
    tau_c, b, sd, z = np.array([-0.4, 0.2, 1.1]), 0.5, 0.6, np.array([0.1, -0.4, 0.3])
    bw, bv = np.array([0.3, -0.7, 0.2]), np.array([-0.5, 0.1, 0.9])
    p = m.value_point(intercept=tau_c, b=[b], sd=sd, z=z, bcs={"w": bw, "v": bv})
    assert p["cs(w)"].shape == (3,) and m.names["linpred"] == "mu_cs" and m.names["b:cs(w)"] == "cs(w)"
    x, y, w, v = df.x.to_numpy(), df.y.to_numpy(), df.w.to_numpy(), df.v.to_numpy()
    mu = b * x + sd * z[pd.Categorical(df.g).codes]
    eta = mu[:, None] + w[:, None] * bw + v[:, None] * bv                  # (n, K - 1)
    tau = tau_c + x.mean() * b
    g_mu, g_cs, g_tau = _graph(m, p, ["mu", "mu_cs", "b_Intercept"])
    np.testing.assert_allclose(g_mu, mu, rtol=1e-14)
    np.testing.assert_allclose(g_cs, eta, rtol=1e-14)
    np.testing.assert_allclose(g_tau, tau, rtol=1e-14)
    assert m.pymc_model.named_vars_to_dims["mu_cs"] == ("__obs__", "Intercept_dim")
    assert m.pymc_model.named_vars_to_dims["cs(w)"] == ("Intercept_dim",)
    lik = reference_log_probabilities(family, eta, tau)[np.arange(len(y)), y - 1].sum()
    prior = (norm.logpdf(tau_c, 0, 3).sum() + norm.logpdf(b, 0, 1.5) + norm.logpdf(bw, 0, 1.5).sum() + norm.logpdf(bv, 0, 1.5).sum()
             + halfnorm.logpdf(sd) + norm.logpdf(z).sum())
    np.testing.assert_allclose(m.pymc_model.compile_logp()(p), lik + prior + np.log(sd), rtol=1e-12)
    np.testing.assert_array_equal(m.design.category_specific["w"], w)
    assert list(m.design.category_specific) == ["w", "v"] and m.spec.columns == ("x",)
    _jax_equals_pytensor(m, p)
    one = model(family, "y ~ x + cs(w) + (1 | g)")
    q = one.value_point(intercept=tau_c, b=[b], sd=sd, z=z, bcs=bw)       # one cs() term also takes the bare vector
    np.testing.assert_array_equal(q["cs(w)"], bw)
    for bad, message in ((None, r"bcs is required"), ({"w": bw}, r"bcs must cover exactly the category-specific columns \['w', 'v'\]"),
                         (bw, "bcs requires a mapping by column"), ({"w": bw[:2], "v": bv}, "bcs for 'w' must be 3 finite numbers")):
        with pytest.raises(ValueError, match=message):
            m.value_point(intercept=tau_c, b=[b], sd=sd, z=z, bcs=bad)
    with pytest.raises(ValueError, match="this model has none"):
        model(family).value_point(intercept=tau_c, b=[b], sd=sd, z=z, bcs=bw)


def test_larger_models_lower_to_jax():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.normal(size=300), "w": rng.normal(size=300), "y": rng.integers(1, 6, 300)})
    for family in FAMS:
        m = getattr(ab, family)("y ~ x + cs(w)", df, priors(False))
        _jax_equals_pytensor(m, m.value_point(intercept=[0.3, -1.5, 0.4, 1.9], b=[0.8], bcs=[0.2, -0.3, 0.5, 0.1]))


def test_names_value_point_and_refusals():
    m = model("acat")
    assert m.names["linpred"] == "mu" and m.names["thresholds"] == "b_Intercept" and m.names["threshold_dim"] == "Intercept_dim"
    assert [v.name for v in m.pymc_model.value_vars] == ["Intercept", "x", "1|g_sigma_log__", "1|g_offset"]
    for bad in ([0.1, 0.2], [0.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match="intercept must be the 3 thresholds, finite .*unordered Intercept of the acat family"):
            m.value_point(intercept=bad, b=[0.1], sd=0.5, z=np.zeros(3))
    with pytest.raises(ValueError, match="sigma, shape and phi do not apply to the acat family"):
        m.value_point(intercept=[0, 1, 2], b=[0.1], sd=0.5, z=np.zeros(3), sigma=1.0)
    with pytest.raises(ab.PriorError, match="must be a location prior"):
        model("sratio", pri=ab.Priors(intercept=ab.HalfNormal(1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    for column in ("mu", "mu_cs", "b_Intercept"):                         # the graph's names cannot be columns
        with pytest.raises(ab.FormulaError, match="collision"):
            model("acat", f"y ~ {column} + cs(w)", frame=data().rename(columns={"x": column}), pri=priors(False))
    cs = model("sratio", "y ~ x + cs(w)", pri=ab.Priors(intercept=ab.Normal(0, 3), b={"x": ab.Normal(0, 1), "cs(w)": ab.Normal(0, 2)}))
    assert ab.Normal(0, 2) == cs.priors.coefficient_priors(cs.spec)["cs(w)"]
    assert "bcs_w[k] ~ Normal(mu=0, sigma=2)" in cs.describe()


REFUSALS = [
    ("gaussian", "y ~ x + cs(w)", {}, "brms's category-specific effect cs() is not supported for the gaussian family"),
    ("cumulative", "y ~ x + cs(w)", {}, "brms's category-specific effect cs() is not supported for the cumulative family: brms takes it "
                                         "in its ordinal families, and this library in the sratio, cratio, acat families"),
    ("acat", "y ~ cs(x + w)", {}, "cs() takes one column name here, not 'x + w'"),
    ("acat", "y ~ x + cs(1)", {}, "cs() takes one column name here, not '1'"),
    ("acat", "y ~ x + cs(x)", {}, "'x' is read by 'cs(x)' and by another term of the formula"),
    ("acat", "y ~ x + z + cs(x):z", {"categorical": {"z": "p"}}, "an interaction with a category-specific term ('cs(x)')"),
    ("acat", "y ~ x + cs(z)", {"categorical": {"z": "p"}}, "'z' is declared categorical; cs() takes a numeric column"),
    ("acat", "y ~ x + cs(y)", {}, "the response 'y' cannot be a category-specific predictor"),
    ("acat", "y ~ x + cs(g) + (1 | g)", {}, "the grouping factor cannot also be a predictor"),
    ("acat", "y ~ x + cs(w) + cs(w)", {}, "duplicated predictor"),
    ("acat", "y ~ x + (cs(1) | g)", {}, "function calls and transformations inside a group term are not supported"),
    ("sratio", "y ~ 0 + x", {}, "the sratio family's thresholds are its intercepts, so the intercept cannot be removed"),
    ("cratio", "y ~ x - 1", {}, "the cratio family's thresholds are its intercepts"),
    ("acat", "y ~ x + mo(e)", {}, "the monotonic term 'mo(e)' is not available for the acat family"),
    ("sratio", "y | weights(e) ~ x", {}, "'weights(e)' is not available for the sratio family"),
    ("acat", "y ~ x + (1 | mm(g, z))", {}, "the multi-membership term '(1 | mm(g, z))' is not available for the acat family"),
]


@pytest.mark.parametrize("family,formula,kwargs,message", REFUSALS, ids=[f"{r[0]}: {r[1]}" for r in REFUSALS])
def test_formula_refusals(family, formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, data(), family, **kwargs)


def test_data_and_link_refusals():
    for frame, message in ((data().assign(w=list("abababababab")), "column 'w' of 'cs(w)' has dtype"),
                           (data().assign(w=1.0), "predictor 'w' of 'cs(w)' is constant"),
                           (data().assign(w=[np.inf] + [0.5] * 11), "non-finite")):
        with pytest.raises(ab.FormulaError, match=re.escape(message)):
            ab.acat("y ~ x + cs(w)", frame, priors(False))
    for family in FAMS:
        for link in ("probit", "cloglog", "cauchit", "probit_approx"):
            with pytest.raises(ValueError, match=f"the {link} link of the {family} family is withheld: brms's {link} link of the ordinal "
                                                 "families is not verified against a frozen brms case here"):
                get_family(family, link)
    with pytest.raises(ValueError, match="the cloglog link of the cumulative family is withheld"):
        ab.cumulative("y ~ x", data(), priors(False), link="cloglog")


def test_other_terms_compose():
    """The term model is shared: categorical predictors, interactions, transforms, the offset (which enters mu), slopes and
    several factors build for these families, with or without cs(); a Gaussian process is added to mu and checked at build
    (C03's graph check)."""
    df = data().assign(h=list("uvuvuvuvuvuv"), xpos=np.exp(X))
    slopes = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
    for family in FAMS:
        for formula, extra, kw in (("y ~ x + z + x:z + cs(w)", {}, {"categorical": {"z": "p"}}), ("y ~ x + log(xpos)", {}, {}),
                                   ("y ~ x + cs(w) + (1 + x | g)", {"sd": slopes, "cor": ab.LKJ(2)}, {}),
                                   ("y ~ x + (1 | g) + (1 | h)", {"sd": {"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}}, {}),
                                   ("y ~ 1 + cs(w) + (1 | g)", {"sd": ab.HalfNormal(1)}, {}),
                                   ("y ~ x + gp(w)", {"sdgp": ab.HalfNormal(1), "lscale": ab.InverseGamma(2, 1)}, {})):
            m = getattr(ab, family)(formula, df, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1), **extra), **kw)
            assert m.family_spec.thresholds and "Intercept" in {v.name for v in m.pymc_model.value_vars}
        m = model(family, "y ~ x + cs(w) + offset(e)", pri=priors(False))
        p = m.value_point(intercept=[-1, 0, 1], b=[0.4], bcs=[0.1, 0.2, 0.3])
        x, w, e = data().x.to_numpy(), data().w.to_numpy(), data().e.to_numpy()
        np.testing.assert_allclose(_graph(m, p, ["mu_cs"])[0], (0.4 * x + e)[:, None] + w[:, None] * [0.1, 0.2, 0.3], rtol=1e-14)
        np.testing.assert_allclose(_graph(m, p, ["b_Intercept"])[0], np.array([-1, 0, 1]) + 0.4 * x.mean(), rtol=1e-14)


# -- predictions, log_lik, bayes_R2 -----------------------------------------------------------------------------------------
def draws(tau=(0.5, -0.4, 1.2), b=0.6, cs=None, group=True):
    post = {"Intercept": np.broadcast_to(np.array(tau) - 1.0, (2, 4, 3)), "b_Intercept": np.broadcast_to(tau, (2, 4, 3)),
            "x": np.full((2, 4), b)}
    dims = {"Intercept": ["Intercept_dim"], "b_Intercept": ["Intercept_dim"]}
    for name, value in (cs or {}).items():
        post[name], dims[name] = np.broadcast_to(value, (2, 4, 3)), ["Intercept_dim"]
    if group:
        post.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.2, -0.3, 0.1], (2, 4, 3))})
        dims["1|g"] = ["g__factor_dim"]
    return az.from_dict(posterior=post, coords={"g__factor_dim": ["a", "b", "c"], "Intercept_dim": ["1", "2", "3"]}, dims=dims)


@pytest.mark.parametrize("family", FAMS)
def test_predictions_by_hand(family):
    df = data()
    tau = np.array([0.5, -0.4, 1.2])
    u = np.array([0.2, -0.3, 0.1])[pd.Categorical(df.g).codes]
    m = model(family)
    eta = 0.6 * df.x.to_numpy() + u
    lin = m.predict(draws(), kind="linear")
    assert lin.eta.dims == ("chain", "draw", "__obs__") and lin.attrs["link"] == "logit"
    np.testing.assert_allclose(lin.eta, np.broadcast_to(eta, (2, 4, 12)))
    out = m.predict(draws(), kind="response", seed=7)
    probs = np.exp(reference_log_probabilities(family, eta[:, None] + 0 * tau, tau))
    assert out.p.dims == ("chain", "draw", "__obs__", "category") and list(out.category.values) == ["1", "2", "3", "4"]
    np.testing.assert_allclose(out.p.values[1, 3], probs, rtol=1e-13)
    np.testing.assert_allclose(out.p.sum("category"), 1.0, rtol=1e-14)
    np.testing.assert_array_equal(out.y, lk.draw_categories(np.broadcast_to(probs, (2, 4, 12, 4)), np.random.default_rng(7)))
    nogroup = m.predict(draws(), kind="mean", include_group_effects=False)
    eta0 = 0.6 * df.x.to_numpy()
    np.testing.assert_allclose(nogroup.p.values[0, 0], np.exp(reference_log_probabilities(family, eta0[:, None] + 0 * tau, tau)), rtol=1e-13)
    c = model(family, "y ~ x + cs(w) + (1 | g)")
    bw = np.array([0.3, -0.7, 0.2])
    lin = c.predict(draws(cs={"cs(w)": bw}), kind="linear")
    assert lin.eta.dims == ("chain", "draw", "__obs__", "threshold") and list(lin.threshold.values) == ["1", "2", "3"]
    eta_cs = eta[:, None] + df.w.to_numpy()[:, None] * bw
    np.testing.assert_allclose(lin.eta.values[1, 2], eta_cs, rtol=1e-14)
    mean = c.predict(draws(cs={"cs(w)": bw}), kind="mean")
    np.testing.assert_allclose(mean.p.values[0, 1], np.exp(reference_log_probabilities(family, eta_cs, tau)), rtol=1e-13)
    new = pd.DataFrame({"x": [2.0, -1.0], "w": [0.5, -2.0], "g": ["new", "a"]})
    for policy in ("population", "resample_observed"):
        rng = np.random.default_rng(8)
        effect = rng.normal(size=(2, 4)) * 0.5 if policy == "population" else np.take([0.2, -0.3, 0.1], rng.integers(0, 3, size=(2, 4)))
        res = c.predict(draws(cs={"cs(w)": bw}), newdata=new, kind="linear", new_groups=policy, seed=8)
        np.testing.assert_allclose(res.eta[..., 0, :], (0.6 * 2.0 + effect)[..., None] + 0.5 * bw)
        np.testing.assert_allclose(res.eta[..., 1, :], np.broadcast_to(0.6 * -1.0 + 0.2 - 2.0 * bw, (2, 4, 3)))
    with pytest.raises(ab.FormulaError, match=r"newdata lacks columns \['w'\]"):
        c.predict(draws(cs={"cs(w)": bw}), newdata=new.drop(columns="w"), kind="mean", new_groups="population")


@pytest.mark.parametrize("family", FAMS)
def test_log_lik_is_the_model_density_and_bayes_r2_the_expected_category(family):
    m = model(family, "y ~ x + cs(w) + (1 | g)")
    bw = np.array([0.3, -0.7, 0.2])
    post = draws(cs={"cs(w)": bw})
    ll = ab.log_lik(m, post).transpose("chain", "draw", "__obs__").to_numpy()
    p = m.value_point(intercept=np.array([0.5, -0.4, 1.2]) - 1.0, b=[0.6], sd=0.5, z=np.array([0.4, -0.6, 0.2]), bcs=bw)
    rows = m.pymc_model.compile_fn(m.pymc_model.logp(vars=[m.pymc_model["y"]], sum=False)[0], inputs=m.pymc_model.value_vars,
                                   on_unused_input="ignore")(p)
    # the posterior's b_Intercept is tau = Intercept + 1 (draws()), the graph's Intercept + mean(x) b: compare like with like
    tau = np.array([0.5, -0.4, 1.2])
    df = data()
    eta = (0.6 * df.x.to_numpy() + np.array([0.2, -0.3, 0.1])[pd.Categorical(df.g).codes])[:, None] + df.w.to_numpy()[:, None] * bw
    want = reference_log_probabilities(family, eta, tau)[np.arange(12), df.y.to_numpy() - 1]
    np.testing.assert_allclose(ll[1, 3], want, rtol=1e-13)
    graph_tau = p["Intercept"] + 0.6 * df.x.mean()
    graph_eta = (0.6 * df.x.to_numpy() + 0.5 * np.array([0.4, -0.6, 0.2])[pd.Categorical(df.g).codes])[:, None] + df.w.to_numpy()[:, None] * bw
    np.testing.assert_allclose(rows, reference_log_probabilities(family, graph_eta, graph_tau)[np.arange(12), df.y.to_numpy() - 1],
                               rtol=1e-12)
    r2 = ab.bayes_R2(m, post).to_numpy()
    probs = np.exp(reference_log_probabilities(family, eta, tau))
    yhat = (probs * np.arange(1, 5)).sum(-1)
    want_r2 = np.var(yhat, ddof=1) / (np.var(yhat, ddof=1) + np.var(df.y.to_numpy() - yhat, ddof=1))
    np.testing.assert_allclose(r2, want_r2, rtol=1e-12)


def test_prior_predictive():
    m = model("acat", "y ~ x + cs(w) + (1 | g)")
    out = m.prior_predictive(draws=60, seed=3)
    tau = out.prior["Intercept"].values[0]
    assert tau.shape == (60, 3) and not (np.diff(tau, axis=-1) > 0).all()     # unordered: the prior's independent draws
    assert out.prior["cs(w)"].shape == (1, 60, 3) and out.prior["mu_cs"].shape == (1, 60, 12, 3)
    y = out.prior_predictive["y"].values
    assert y.shape == (1, 60, 12) and y.min() >= 1 and y.max() <= 4
    flat = model("sratio", "y ~ x + cs(w)", pri=ab.Priors(intercept=ab.Normal(0, 1), b=ab.Flat()))
    with pytest.raises(ValueError, match=r"the prior of \['b:x', 'b:cs\(w\)'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)


# -- describe, identity, defaults, the registry ----------------------------------------------------------------------------
def test_describe_and_identity():
    texts = {}
    for family in FAMS:
        m = model(family)
        text = m.describe()
        texts[family] = text
        assert text.startswith(f"AutoBayes {family} (ordinal) model") and f"family    : {family}, logit link" in text
        assert "mu_i = x_x * b_x + u[g_i]" in text and "Intercept[k] ~ Normal(mu=0, sigma=3), for each of the 3 thresholds" in text
        assert "an unordered vector" in text and "(vector; dims Intercept_dim, unordered, as brms's)" in text
        same = ab.Model("y ~ x + (1 | g)", data(), priors(), family=family)
        assert same.describe() == text and same.target_identity == m.target_identity
        assert model(family, center_predictors=False).target_identity != m.target_identity
    assert "P(y_i = k | y_i >= k) = expit(b_Intercept[k] - mu_i), k = 1..3 (the stopping ratio)" in texts["sratio"]
    assert "P(y_i > k | y_i >= k) = expit(mu_i - b_Intercept[k]), k = 1..3 (the continuation ratio)" in texts["cratio"]
    assert "log(P(y_i = k + 1) / P(y_i = k)) = mu_i - b_Intercept[k], k = 1..3 (adjacent categories)" in texts["acat"]
    identities = {model(f).target_identity["sha256"] for f in (*FAMS, "cumulative")}
    assert len(identities) == 4                                            # sratio and cratio: the same density, other families
    cs = model("acat", "y ~ x + cs(w) + (1 | g)")
    text = cs.describe()
    assert "with category-specific effect(s) cs()" in text and "mu_i[k] = mu_i + x_w,i * bcs_w[k]" in text
    assert "bcs_w[k] ~ Normal(mu=0, sigma=1.5), for each of the 3 thresholds k (brms's bcs, class b, coefficient w)" in text
    assert "mu_cs = mu_i[k], one per row and threshold" in text
    assert cs.target_identity != model("acat", "y ~ x + cs(v) + (1 | g)", frame=data().assign(v=data().w)).target_identity
    assert cs.target_identity != model("acat", "y ~ x + cs(w) + (1 | g)", frame=data().assign(w=data().w + 1)).target_identity


# brms 2.23.0's default_prior() (R 4.5.2, C collation) for these models (any data with four categories), as (class, coef,
# prior); every source "default"
BRMS_DEFAULTS = {
    "acat, cs": ("y ~ x + cs(w)", "acat", [("Intercept", "", "student_t(3, 0, 2.5)"), ("Intercept", "1", ""), ("Intercept", "2", ""),
                                           ("Intercept", "3", ""), ("b", "", ""), ("b", "w", ""), ("b", "x", "")]),
    "acat, cs first": ("y ~ cs(w) + x", "acat", [("Intercept", "", "student_t(3, 0, 2.5)"), ("Intercept", "1", ""), ("Intercept", "2", ""),
                                                 ("Intercept", "3", ""), ("b", "", ""), ("b", "w", ""), ("b", "x", "")]),
    "sratio": ("y ~ x", "sratio", [("Intercept", "", "student_t(3, 0, 2.5)"), ("Intercept", "1", ""), ("Intercept", "2", ""),
                                   ("Intercept", "3", ""), ("b", "", ""), ("b", "x", "")]),
}


@pytest.mark.parametrize("label", sorted(BRMS_DEFAULTS))
def test_default_priors_are_brms_default_prior(label):
    formula, family, rows = BRMS_DEFAULTS[label]
    pri = ab.default_priors(formula, data(), family=family)
    m = getattr(ab, family)(formula, data(), pri)
    table = m.prior_table()
    assert [(r["class"], r["coef"], r["prior"]) for r in table] == rows
    assert all(r["source"] == "default" for r in table)


def test_registry():
    for family in FAMS:
        spec = get_family(family)
        assert spec is FAMILIES[family] and spec.thresholds and not spec.ordered_thresholds and spec.category_specific
        assert spec.link == "logit" and not spec.links and getattr(ab, family).__name__ == family
    assert get_family("cumulative").ordered_thresholds and not get_family("cumulative").category_specific
    assert not any(FAMILIES[f].category_specific for f in FAMILIES if f not in FAMS)


# -- the withheld cumulative probit link (D88) -------------------------------------------------------------------------------
def test_the_probit_link_is_withheld_with_its_reason():
    for call in (lambda: ab.cumulative("y ~ x", data(), priors(False), link="probit"),
                 lambda: ab.default_priors("y ~ x", data(), family="cumulative", link="probit")):
        with pytest.raises(ValueError, match="the probit link of the cumulative family is withheld: brms's gradient of the frozen case "
                                             "cumulative_probit_varint .* divergence D88"):
            call()


def test_the_withheld_probit_link_equals_brms_but_for_brms_s_gradient(monkeypatch):
    """With the park lifted (the registry patched for this test), the model of the frozen case cumulative_probit_varint passes
    G1, G2's log densities and G3-G6 against brms; its gradient equals the exact one (by hand, the derivative in b of the
    probit likelihood and b's prior) where brms's is off by 6.5e-5 relative (D88)."""
    from dataclasses import replace

    import autobayes.families as families
    from autobayes.families.cumulative import CUMULATIVE, _probit
    from tests.oracle import gates
    from tests.oracle.adapter import natural_point, value_layout

    unparked = replace(CUMULATIVE, links=MappingProxyType({"probit": _probit}))
    monkeypatch.setattr(families, "FAMILIES", MappingProxyType({**FAMILIES, "cumulative": unparked}))
    b = gates.Bundle("cumulative_probit_varint")
    assert b.model.link == "probit" and b.model.family_spec is _probit() and b.model.family_spec.ordered_thresholds
    for check in (gates.check_design, gates.check_logp, gates.check_derived, gates.check_predictions, gates.check_jax,
                  gates.check_default_prior):
        check(b)
    _, p, vp = next(b.points())
    nat = natural_point(b.model, p, b.cmap, b.gmap)
    d = b.model.design
    eta = d.X_raw[:, 0] * nat["b"][0] + nat["sd"]["g"] * nat["z"]["g"][d.group_index]
    tau = nat["intercept"] + d.x_means[0] * nat["b"][0]
    y = d.y.astype(int)
    hi = np.r_[tau, np.inf][y - 1] - eta
    lo = np.r_[-np.inf, tau][y - 1] - eta
    mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))   # moderate gaps here: no digits lost
    dens = norm.pdf(hi) - norm.pdf(lo)                                    # pdf(+-inf) = 0
    exact = ((d.x_means[0] - d.X_raw[:, 0]) * dens / mass).sum() - nat["b"][0] / 1.5 ** 2   # d(tau - eta)/db = mean(x) - x
    ours = float(np.asarray(b.dlogp(vp))[value_layout(b.model, vp)["x"].start])
    assert abs(ours - exact) <= 1e-12 * abs(exact)
    assert abs(p.grad[0] - exact) > 1e-5 * abs(exact)                     # brms's: 6.5e-5 relative off at this point


def test_fit_record_and_summary_rows(monkeypatch):
    import warnings
    from types import SimpleNamespace

    import autobayes.session as session

    m = model("cratio", "y ~ x + cs(w) + (1 | g)")
    post = draws(cs={"cs(w)": np.array([0.3, -0.7, 0.2])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # constant synthetic draws: R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert {"Intercept[1]", "Intercept[3]", "x", "cs(w)[1]", "cs(w)[3]", "1|g_sigma"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "cratio" and rec["model"]["link"] == "logit" and rec["model"]["terms"] == ["x", "cs(w)"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
