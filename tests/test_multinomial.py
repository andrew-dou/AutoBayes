"""Count-matrix responses, brms's multinomial() and dirichlet_multinomial() (brms-parity F01a, docs/multinomial.md): the row
densities against a 50-digit reference on both paths (PyTensor's gradient and JAX's autodiff of the jaxified graph), their
NumPy forms and draws, the formula and its refusals, the counts, the priors, the model's names, design, density and gradient
by hand on both paths, ``value_point``, ``describe()``, predictions by hand with every group policy, ``log_lik()`` / ``loo()``
and the refused post-processing, brms's default prior tables (printed by brms 2.23 for these formulas), the prior table, the
target identity, the fit plumbing with a stubbed sampler, the fit record and prior-predictive draws. No MCMC; the brms
comparison is the oracle cases multinomial_varint, multinomial_dpar_formula, multinomial_varint_post,
dirichlet_multinomial_varint, multinomial_terms and dirichlet_multinomial_defaults."""
import json
import math
from decimal import Decimal, localcontext

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy.special import gammaln, softmax
from scipy.stats import gamma as gamma_dist
from scipy.stats import halfnorm, multinomial as multinomial_dist, norm

import autobayes as ab
from autobayes.families import FAMILIES, get_family
from autobayes.families import multinomial as fm
from autobayes.formula import FormulaError
from autobayes.multinomial import (MultinomialModel, MultinomialSpec, category_priors, check_multinomial_newdata, multinomial_improper,
                                   parse_multinomial, prior_labels, response_counts)
from autobayes.priors import PriorError

# -- 50-digit references -------------------------------------------------------------------------------------------------


def _d(x):
    """A double as the Decimal it denotes (a Decimal passes through, so that the references' perturbations survive)."""
    return x if isinstance(x, Decimal) else Decimal(repr(float(x)))


_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
_BERNOULLI = ((1, 6), (-1, 30), (1, 42), (-1, 30), (5, 66), (-691, 2730), (7, 6), (-3617, 510), (43867, 798), (-174611, 330))


def decimal_log_factorial(n: int) -> Decimal:
    """log(n!) at the context's precision: exact below 300, Stirling's series with ten Bernoulli terms above (the first
    omitted term is below 1e-50 there)."""
    if n < 300:
        return Decimal(math.factorial(n)).ln()
    x = Decimal(n + 1)
    out = (x - Decimal("0.5")) * x.ln() - x + (2 * _PI).ln() / 2
    for k, (num, den) in enumerate(_BERNOULLI, start=1):
        out += Decimal(num) / Decimal(den) / (2 * k * (2 * k - 1) * x ** (2 * k - 1))
    return out


def exact_log_coefficient(y):
    """log(N! / prod y_k!) at 50 digits."""
    with localcontext() as ctx:
        ctx.prec = 50
        return decimal_log_factorial(int(sum(y))) - sum(decimal_log_factorial(int(v)) for v in y)


def exact_log_softmax(eta):
    full = [Decimal(0)] + [_d(v) for v in eta]
    m = max(full)
    z = sum((v - m).exp() for v in full)
    return [v - m - z.ln() for v in full]


def exact_multinomial_row(y, eta):
    """(log Multinomial(y | sum y, softmax(0, eta)), its gradient in eta) at 50 digits: the gradient is y_k - N p_k."""
    with localcontext() as ctx:
        ctx.prec = 50
        lp = exact_log_softmax(eta)
        value = exact_log_coefficient(y) + sum(int(v) * l for v, l in zip(y, lp))
        n = sum(int(v) for v in y)
        grad = [int(y[k]) - n * lp[k].exp() for k in range(1, len(lp))]
        return float(value), [float(g) for g in grad]


def exact_log_rising(log_alpha: Decimal, y: int) -> Decimal:
    """lgamma(y + a) - lgamma(a) for a whole y: sum_{j < y} log(a + j), exact at 50 digits."""
    a = log_alpha.exp()
    return sum(((a + j).ln() for j in range(int(y))), Decimal(0))


def exact_dm_value(y, eta, phi) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 50
        lp = exact_log_softmax(eta)
        log_phi = _d(phi).ln()
        n = sum(int(v) for v in y)
        return (exact_log_coefficient(y) + sum(exact_log_rising(log_phi + l, v) for l, v in zip(lp, y))
                - exact_log_rising(log_phi, n))


def exact_dm_row(y, eta, phi, step=Decimal("1e-12")):
    """(value, gradient in eta, derivative in phi) of the Dirichlet-multinomial row at 50 digits, the derivatives as fourth-order
    central differences of the 50-digit value (error ~ step^4, far below double precision)."""
    with localcontext() as ctx:
        ctx.prec = 60

        def f(e, p):
            return exact_dm_value(y, e, p)

        def diff(g):
            return (-g(2 * step) + 8 * g(step) - 8 * g(-step) + g(-2 * step)) / (12 * step)

        value = f(eta, phi)
        grad = []
        for k in range(len(eta)):
            grad.append(diff(lambda h, k=k: f([_d(v) + (h if j == k else 0) for j, v in enumerate(eta)], phi)))
        dphi = diff(lambda h: f(eta, _d(phi) + h))
        return float(value), [float(g) for g in grad], float(dphi)


# -- the row densities on both paths ------------------------------------------------------------------------------------------
def _rows(mode, kind, y, eta, phi=None):
    yv, ev = pt.lmatrix("y"), pt.dmatrix("eta")
    full = pt.concatenate([pt.zeros_like(ev[:, :1]), ev], axis=1)
    n = pt.sum(yv, axis=1)
    if kind == "multinomial":
        inputs, rows = [yv, ev], fm.multinomial_logit_logp(yv, full, n)
        args = (y, eta)
    else:
        pv = pt.dscalar("phi")
        inputs, rows = [yv, ev, pv], fm.dirichlet_multinomial_logit_logp(yv, full, pv, n)
        args = (y, eta, np.float64(phi))
    if mode == "pytensor":
        outs = [rows, pt.grad(rows.sum(), ev)] + ([pt.grad(rows.sum(), inputs[2])] if kind != "multinomial" else [])
        return [np.asarray(a, dtype=float) for a in pytensor.function(inputs, outs)(*args)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=inputs, outputs=[rows])
    out = [np.asarray(jf(*args)[0], dtype=float), np.asarray(jax.grad(lambda e: jf(args[0], e, *args[2:])[0].sum())(eta))]
    if kind != "multinomial":
        out.append(np.asarray(jax.grad(lambda p: jf(y, eta, p)[0].sum())(args[2])))
    return out


COUNTS = [(3, 0, 2), (0, 0, 0), (0, 5, 0), (1, 1, 1), (12, 7, 30), (400000, 1, 599999)]
ETAS = [(0.0, 0.0), (-1000.0, 1000.0), (40.0, -40.0), (-3.0, 0.7), (2.5, 2.5), (1000.0, 999.0)]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_multinomial_rows_are_exact_and_finite_on_both_paths(mode):
    grid = [(y, e) for y in COUNTS for e in ETAS]
    y = np.array([g[0] for g in grid], dtype=np.int64)
    eta = np.array([g[1] for g in grid], dtype=float)
    value, grad = _rows(mode, "multinomial", y, eta)
    assert np.isfinite(value).all() and np.isfinite(grad).all()
    for i, (yi, ei) in enumerate(grid):
        want, want_g = exact_multinomial_row(yi, ei)
        assert abs(value[i] - want) <= 1e-12 * max(1.0, abs(want)), (mode, yi, ei, value[i], want)
        np.testing.assert_allclose(grad[i], want_g, rtol=1e-11, atol=1e-9)
    # SciPy's multinomial at ordinary counts, and the coefficient of a zero row is exactly 0
    moderate = [i for i, e in enumerate(ETAS) if max(map(abs, e)) < 50]
    np.testing.assert_allclose(value[moderate], [multinomial_dist.logpmf([3, 0, 2], 5, softmax([0.0, *ETAS[i]])) for i in moderate],
                               rtol=1e-12)
    assert np.all(value[len(ETAS):2 * len(ETAS)] == 0.0)


DM_PHIS = [1e-3, 0.4, 3.0, 1e4, 1e8]
DM_ETAS = [(0.0, 0.0), (-3.0, 0.7), (25.0, -30.0), (-800.0, 2.0), (-38.0, 1.0)]
DM_COUNTS = [(3, 0, 2), (0, 0, 0), (1, 4, 1), (20, 7, 3)]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_dirichlet_multinomial_rows_are_exact_and_finite_on_both_paths(mode):
    for phi in DM_PHIS:
        grid = [(y, e) for y in DM_COUNTS for e in DM_ETAS]
        y = np.array([g[0] for g in grid], dtype=np.int64)
        eta = np.array([g[1] for g in grid], dtype=float)
        value, grad, dphi = _rows(mode, "dirichlet_multinomial", y, eta, phi)
        assert np.isfinite(value).all() and np.isfinite(grad).all() and np.isfinite(dphi)
        want_dphi = 0.0
        for i, (yi, ei) in enumerate(grid):
            want, want_g, want_p = exact_dm_row(yi, ei, phi)
            want_dphi += want_p
            assert abs(value[i] - want) <= 2e-12 * max(1.0, abs(want)), (mode, phi, yi, ei, value[i], want)
            np.testing.assert_allclose(grad[i], want_g, rtol=1e-8, atol=1e-9 * max(1.0, max(abs(g) for g in want_g)))
        assert abs(float(dphi) - want_dphi) <= 1e-7 * max(1.0, abs(want_dphi)), (mode, phi, float(dphi), want_dphi)


def test_rows_outside_the_support_have_no_mass():
    y = np.array([[-1, 3, 2], [1, 2, 2], [2, 2, 1]], dtype=np.int64)
    yv, fv, nv = pt.lmatrix("y"), pt.dmatrix("full"), pt.lvector("n")
    full = np.zeros((3, 3))
    rows = pytensor.function([yv, fv, nv], fm.multinomial_logit_logp(yv, fv, nv))(y, full, np.array([4, 4, 5]))
    assert rows[0] == -np.inf and rows[1] == -np.inf and np.isfinite(rows[2])        # a negative count; a wrong total
    rows = pytensor.function([yv, fv, nv], fm.dirichlet_multinomial_logit_logp(yv, fv, np.float64(2.0), nv))(y, full, np.array([4, 4, 5]))
    assert rows[0] == -np.inf and rows[1] == -np.inf and np.isfinite(rows[2])
    yd = pt.dmatrix("yd")                                                            # a count that is not whole
    rows = pytensor.function([yd, fv, nv], fm.multinomial_logit_logp(yd, fv, nv))(np.array([[0.5, 1.5, 2.0]]), np.zeros((1, 3)), np.array([4]))
    assert rows[0] == -np.inf


def test_numpy_rows_equal_the_graph_rows_and_their_helpers():
    rng = np.random.default_rng(5)
    y = np.array([[3, 0, 2], [0, 0, 0], [1, 4, 1], [20, 7, 3]], dtype=np.int64)
    eta = rng.normal(size=(2, 3, 4, 2)) * np.array([1.0, 30.0])                       # (chain, draw, row, K - 1)
    phi = np.exp(rng.normal(size=(2, 3)))
    rows = fm.multinomial_logit_rows(y, eta)
    rows_dm = fm.dirichlet_multinomial_logit_rows(y, eta, phi[..., None])
    for c in range(2):
        for d in range(3):
            g, _ = _rows("pytensor", "multinomial", y, eta[c, d])
            np.testing.assert_allclose(rows[c, d], g, rtol=1e-13, atol=1e-13)
            g, _, _ = _rows("pytensor", "dirichlet_multinomial", y, eta[c, d], phi[c, d])
            np.testing.assert_allclose(rows_dm[c, d], g, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fm.log_multinomial_coefficient_np(y), [float(exact_log_coefficient(v)) for v in y], rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(fm.log_multinomial_coefficient_np([[400000, 1, 599999]]),
                               float(exact_log_coefficient((400000, 1, 599999))), rtol=1e-12)
    # log_rising_np: zero counts, the tiny-alpha branch and the betaln branch against the exact sums
    la = np.array([-800.0, -45.0, -35.0, -2.0, 0.5, 25.0])
    for v in (0, 1, 3, 40):
        with localcontext() as ctx:
            ctx.prec = 50
            want = [float(exact_log_rising(_d(a), v)) for a in la]
        np.testing.assert_allclose(fm.log_rising_np(la, v), want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fm.probabilities(eta[0, 0]), softmax(np.column_stack([np.zeros(4), eta[0, 0]]), axis=1), rtol=1e-12)
    np.testing.assert_allclose(fm._log_softmax_np(np.array([1000.0, -1000.0])), [-1000.0, 0.0, -2000.0])


def test_count_draws_and_their_distributions():
    rng = np.random.default_rng(7)
    p = np.array([[0.2, 0.5, 0.3], [0.9, 0.05, 0.05]])
    draws = fm.draw_multinomial(np.broadcast_to(p, (20000, 2, 3)), np.array([10, 4]), rng)
    assert draws.shape == (20000, 2, 3) and draws.dtype == np.int64 and np.array_equal(draws.sum(-1), np.broadcast_to([10, 4], (20000, 2)))
    np.testing.assert_allclose(draws.mean(0), p * np.array([[10], [4]]), atol=0.05)
    alpha = np.array([[2.0, 3.0, 5.0], [1e-300, 1e-300, 1.0], [1e-300, 1e-300, 1e-300]])
    q = fm.draw_dirichlet_log(np.log(np.broadcast_to(alpha, (20000, 3, 3))), rng)
    assert np.isfinite(q).all() and np.allclose(q.sum(-1), 1.0)                   # never 0 / 0, even at concentrations of 1e-300
    assert (q[:, 2].max(-1) == 1.0).all() and np.allclose(q[:, 2].mean(0), 1 / 3, atol=0.02)   # one-hot, each category alike
    np.testing.assert_allclose(q[:, 0].mean(0), alpha[0] / alpha[0].sum(), atol=0.01)
    np.testing.assert_allclose(q[:, 0].var(0), [0.2 * 0.8 / 11, 0.3 * 0.7 / 11, 0.5 * 0.5 / 11], rtol=0.06)
    full = np.array([[0.0, 0.3, -0.4], [0.0, 2.0, 1.0]])
    one = fm.multinomial_logit_random(full, np.array([6, 9]), rng=rng)
    assert one.shape == (2, 3) and np.array_equal(one.sum(-1), [6, 9])
    many = fm.multinomial_logit_random(full, np.array([6, 9]), rng=rng, size=(5, 2))
    assert many.shape == (5, 2, 3) and np.array_equal(many.sum(-1), np.broadcast_to([6, 9], (5, 2)))
    dm = fm.dirichlet_multinomial_logit_random(full, 2.0, np.array([6, 9]), rng=rng)
    assert dm.shape == (2, 3) and np.array_equal(dm.sum(-1), [6, 9])
    dm = fm.dirichlet_multinomial_logit_random(full, 2.0, np.array([6, 9]), rng=rng, size=(4, 2))
    assert dm.shape == (4, 2, 3) and np.array_equal(dm.sum(-1), np.broadcast_to([6, 9], (4, 2)))
    # the Dirichlet-multinomial's variance exceeds the multinomial's by (n + phi) / (1 + phi)
    big = fm.dirichlet_multinomial_logit_random(np.zeros((1, 3)), 2.0, np.array([20]), rng=rng, size=(40000, 1))
    np.testing.assert_allclose(big[:, 0].var(0), 20 * (1 / 3) * (2 / 3) * (20 + 2) / 3, rtol=0.05)


def test_family_specs_and_hooks():
    m, dm = get_family("multinomial"), get_family("dirichlet_multinomial")
    assert m is FAMILIES["multinomial"] and dm is FAMILIES["dirichlet_multinomial"]
    assert m.matrix_response and dm.matrix_response and m.per_category and m.has_trials and m.discrete
    assert (m.scale_parameter, dm.scale_parameter, dm.bambi_scale_parameter) == (None, "phi", "phi")
    assert str(dm.default_scale_prior(2.5)) == "Gamma(alpha=0.01, beta=0.01)" and m.default_scale_prior is None
    with pytest.raises(RuntimeError, match="built by autobayes.multinomial, not by Bambi"):
        m.bambi_family()
    assert m.validate_response(None, None) is None and m.response_warnings(None, None) == []
    with pytest.raises(ValueError, match="the multinomial family has the logit link only"):
        m.with_link("probit")


# -- data, priors and models -------------------------------------------------------------------------------------------------
def frame(n: int = 12, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 5, size=(n, 3))
    counts[:, 0] += 1                                                               # every category has counts
    return pd.DataFrame({"x": rng.normal(size=n).round(3), "w": rng.normal(size=n).round(3), "z": list("pqr") * (n // 3),
                         "g": list("aabbcc") * (n // 6), "h": list("uv") * (n // 2), "xpos": np.exp(rng.normal(size=n)).round(3),
                         "m1": counts[:, 0], "m2": counts[:, 1], "m3": counts[:, 2], "size": counts.sum(1)})


F = "cbind(m1, m2, m3) | trials(size) ~ x + (1 | g)"
RICH = ["cbind(m1, m2, m3) | trials(size) ~ x + z + (1 | g)", "mum3 ~ w + (1 | h)"]
OWN = {"intercept": ab.StudentT(3, 0, 2), "b": ab.Normal(0, 0.5), "sd": ab.Exponential(2)}


def pri(**kw):
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
    base.update(kw)
    return ab.Priors(**base)


def rich(family="multinomial", **kw):
    extra = {"phi": ab.Gamma(2, 0.5)} if family == "dirichlet_multinomial" else {}
    return ab.Model(RICH, frame(), pri(dpars={"mum3": OWN}, **extra), family=family, categorical={"z": "p"}, **kw)


def rich_point(m, seed=3):
    rng = np.random.default_rng(seed)
    out = {"dpars": {"mum2": {"intercept": 0.4, "b": rng.normal(size=3), "sd": 0.7, "z": rng.normal(size=3)},
                     "mum3": {"intercept": -0.3, "b": rng.normal(size=1), "sd": 1.3, "z": rng.normal(size=2)}}}
    if "phi" in m.names:
        out["phi"] = 2.5
    return out


# -- the formula ---------------------------------------------------------------------------------------------------------------
def test_the_formula_and_its_views():
    s = parse_multinomial(RICH, frame(), "multinomial", {"z": "p"})
    assert isinstance(s, MultinomialSpec) and s.is_multinomial and not s.is_nonlinear
    assert (s.categories, s.reference, s.trials, s.response, s.mu_names, s.own) == (("m1", "m2", "m3"), "m1", "size", "cbind(m1, m2, m3)",
                                                                                  ("mum2", "mum3"), ("mum3",))
    assert s.formulas == tuple(RICH) and s.formula == RICH[0] and s.rhs == "x + z + (1 | g)"
    assert s.mu("mum2").formula == "mum2 ~ x + z + (1 | g)" and s.mu("mum3").formula == "mum3 ~ w + (1 | h)"
    with pytest.raises(KeyError):
        s.mu("mum1")
    assert s.factors == ("g", "h") and [c.name for c in s.all_categoricals] == ["z"]
    assert s.model_columns == ("x", "z", "w") and s.group_columns == ("g", "h")
    assert s.data_columns == ("m1", "m2", "m3", "size", "x", "z", "w", "g", "h")
    assert s.canonical_formula == "cbind(m1, m2, m3) | trials(size) ~ x + z + (1 | g)" and s.addition("weights") is None
    assert [(name, t.factor) for name, t in s.mu_group_terms] == [("mum2", "g"), ("mum3", "h")]
    for view in ("pop_terms", "group_terms", "terms", "columns", "dpars", "dpar_names", "additions", "groups", "categoricals"):
        assert getattr(s, view) == ()
    assert s.intercept is False and s.offset is None and s.autocor is None
    every = parse_multinomial(["cbind(m1, m2, m3) | trials(size) ~ 1", "mum2 ~ x", "mum3 ~ w"])   # the response formula unused
    assert every.own == ("mum2", "mum3") and every.canonical_formula == "cbind(m1, m2, m3) | trials(size) ~ 1"
    nested = parse_multinomial("cbind(m2, m1, m3) | trials(size) ~ x + (1 | g/h)", frame())
    assert nested.categories == ("m2", "m1", "m3") and nested.mu_names == ("mum1", "mum3")
    assert [t.factor for t in nested.mu("mum1").group_terms] == ["g", "g:h"]


REFUSALS = [
    ("y ~ x", "the multinomial families' response is a count matrix written over its count columns with cbind\\(\\)"),
    ("cbind(m1, 2) | trials(size) ~ x", "cbind\\(\\) takes the count columns by name"),
    ("cbind(m1) | trials(size) ~ x", "At least 2 response categories are required"),
    ("cbind(m1, m1, m2) | trials(size) ~ x", "cbind\\(\\) names a column twice"),
    ("cbind(m1, m2, m3) ~ x", "Specifying 'trials' is required for this model"),
    ("cbind(m1, m2, m3) | trials(size) + weights(w) ~ x", "the addition term 'weights\\(w\\)'.*divergence D136"),
    ("cbind(m1, m2, m3) | trials(size) + trials(size) ~ x", "exactly one trials\\(n\\)"),
    ("cbind(m1, m2, size) | trials(size) ~ x", "the trials column 'size' cannot also be a category's counts"),
    ("cbind(m1, m2, m3) | trials(size) ~ 0 + x", "without an intercept"),
    ("cbind(m1, m2, m3) | trials(size) ~ x - 1", "without an intercept"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + offset(w)", "the offset term 'offset\\(w\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ mo(size2)", "the monotonic term 'mo\\(size2\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ me(x, w)", "the measurement-error term"),
    ("cbind(m1, m2, m3) | trials(size) ~ mi(x)", "the missing-value term"),
    ("cbind(m1, m2, m3) | trials(size) ~ cs(x)", "the category-specific term 'cs\\(x\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ gp(x)", "the Gaussian process 'gp\\(x\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + ar(time = w, gr = g)", "the autocorrelation term"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + sar(M, type = \"lag\")", "the SAR term 'sar\\(M, type = \"lag\"\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + (1 | mm(g, h))", "mm\\(\\) and gr\\(by = \\)"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + (1 |p| g)", "the correlation ID of '\\(1 \\|p\\| g\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + (1 + x | g)", "the varying slope\\(s\\) of '\\(1 \\+ x \\| g\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + (1 + x || g)", "the varying slope\\(s\\) of '\\(1 \\+ x \\|\\| g\\)'"),
    ("cbind(m1, m2, m3) | trials(size) ~ m1 + x", "'m1' is a category of the response; it cannot also be a predictor"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + (1 | size)", "'size' is the trials column"),
    (["cbind(m1, m2, m3) | trials(size) ~ x", "mum1 ~ w"], "The parameter 'mum1' is not a valid distributional or non-linear "
                                                            "parameter \\(brms\\) \\('mum1' is the reference category's"),
    (["cbind(m1, m2, m3) | trials(size) ~ x", "sigma ~ w"], "The parameter 'sigma' is not a valid"),
    (["cbind(m1, m2, m3) | trials(size) ~ x", "mum3 ~ w", "mum3 ~ x"], "'mum3' has two formulas"),
    (["cbind(m1, m2, m3) | trials(size) ~ x", "mum3 | weights(w) ~ x"], "a category's formula is '<mu<category>> ~ <terms>'"),
    (["cbind(m1, m2, m3) | trials(size) ~ x", "mum2 ~ w", "mum3 ~ w"], "apply to none of them \\(brms ignores them silently\\)"),
    (["cbind(m1, m2, m3) | trials(size) ~ x", 3], "the multinomial families take"),
    ("cbind(m1, m2, m3) | trials(size) ~ x + (1 | g)", None),
]


@pytest.mark.parametrize("formula, match", REFUSALS)
def test_formula_refusals(formula, match):
    if match is None:
        assert parse_multinomial(formula).mu_names == ("mum2", "mum3")
        return
    with pytest.raises(FormulaError, match=match):
        parse_multinomial(formula)


def test_phi_formula_categorical_declarations_and_other_families():
    with pytest.raises(FormulaError, match="a formula of phi .*divergence D136"):
        parse_multinomial(["cbind(m1, m2, m3) | trials(size) ~ x", "phi ~ w"], family="dirichlet_multinomial")
    with pytest.raises(FormulaError, match=r"\['h'\] declared categorical, but no category's formula names it"):
        parse_multinomial(RICH, categorical={"h": "u"})
    with pytest.raises(FormulaError, match="categorical must be a mapping"):
        parse_multinomial(RICH, categorical=["z"])
    s = parse_multinomial(["cbind(m1, m2, m3) | trials(size) ~ x", "mum3 ~ z"], frame(), categorical={"z": "q"})
    assert s.mu("mum2").categoricals == () and s.mu("mum3").categoricals[0].reference == "q"
    with pytest.raises(FormulaError, match="the multinomial family's matrix response is parsed by autobayes.multinomial"):
        ab.parse_formula(F, family="multinomial")


def test_the_counts_and_their_refusals():
    s = parse_multinomial(F)
    Y, n = response_counts(s, frame())
    assert Y.shape == (12, 3) and Y.dtype == np.int64 and np.array_equal(Y.sum(1), n) and np.array_equal(n, frame()["size"])
    with pytest.raises(FormulaError, match=r"columns not in data: \['m3'\]"):
        response_counts(s, frame().drop(columns="m3"))
    with pytest.raises(FormulaError, match=r"missing values in columns \['m2'\]"):
        response_counts(s, frame().assign(m2=np.where(np.arange(12) == 3, np.nan, frame()["m2"])))
    with pytest.raises(FormulaError, match="count column 'm1' has a non-integral value"):
        response_counts(s, frame().assign(m1=frame()["m1"] + 0.5, size=frame()["size"] + 0.5))
    with pytest.raises(FormulaError, match="count column 'm2' has a negative value"):
        response_counts(s, frame().assign(m2=-1, size=frame()["m1"] + frame()["m3"] - 1))
    bad = frame().assign(size=frame()["size"] + np.where(np.arange(12) == 4, 1, 0))
    with pytest.raises(FormulaError, match=r"Number of trials does not match the number of events \(brms\): row 4 has counts"):
        response_counts(s, bad)
    with pytest.raises(FormulaError, match="Number of trials does not match"):
        parse_multinomial(F, bad)
    zero = frame().assign(m1=lambda d: np.where(np.arange(12) == 0, 0, d["m1"]), m2=lambda d: np.where(np.arange(12) == 0, 0, d["m2"]),
                          m3=lambda d: np.where(np.arange(12) == 0, 0, d["m3"]))
    zero["size"] = zero[["m1", "m2", "m3"]].sum(1)
    m = ab.multinomial(F, zero, pri())                                             # a row of zero trials is kept, as brms keeps it
    assert m.design.n == 12 and m.design.trials[0] == 0


# -- the priors ----------------------------------------------------------------------------------------------------------------
def test_category_priors_labels_and_improper():
    s = parse_multinomial(RICH, frame(), categorical={"z": "p"})
    p = pri(dpars={"mum3": OWN})
    two, three = category_priors(p, s, "mum2"), category_priors(p, s, "mum3")
    assert (two.intercept, two.b, two.sd) == (ab.Normal(0, 2), ab.Normal(0, 1), ab.HalfNormal(1))
    assert (three.intercept, three.b, three.sd) == (OWN["intercept"], OWN["b"], OWN["sd"])
    labels = [label for label, _ in prior_labels(p, s, get_family("multinomial"))]
    assert labels == ["mum2:intercept", "mum2:b:x", "mum2:b:z[q]", "mum2:b:z[r]", "mum2:sd:g", "mum3:intercept", "mum3:b:w", "mum3:sd:h"]
    dm = pri(dpars={"mum3": OWN}, phi=ab.Gamma(2, 0.5))
    assert prior_labels(dm, s, get_family("dirichlet_multinomial"))[-1] == ("phi", ab.Gamma(2, 0.5))
    flat = ab.default_priors(RICH, frame(), family="multinomial", categorical={"z": "p"})
    assert multinomial_improper(flat, s) == ["mum2:b:x", "mum2:b:z[q]", "mum2:b:z[r]", "mum3:b:w"]
    assert flat.improper(s) == multinomial_improper(flat, s) and multinomial_improper(p, s) == []
    every = ab.Priors(dpars={"mum2": {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), "sd": ab.HalfNormal(1)}, "mum3": OWN})
    every.validate(s, get_family("multinomial"))                                   # every category in dpars: no top-level classes


PRIOR_REFUSALS = [
    (dict(nlpars={"a": {}}), "nlpars is the priors of a non-linear model"),
    (dict(cor=ab.LKJ(2)), "cor is an LKJ prior of a correlated term"),
    (dict(theta=ab.Dirichlet(2)), "theta is the priors of a mixture"),
    (dict(nu=ab.Gamma(2, 0.1)), "nu is the Student-t degrees of freedom"),
    (dict(sigma=ab.HalfNormal(1)), "the multinomial family has no residual SD; do not pass sigma"),
    (dict(phi=ab.Gamma(2, 2)), "the multinomial family has no precision parameter; do not pass phi"),
    (dict(dpars=[1]), "dpars must be a mapping"),
    (dict(dpars={"mum1": OWN}), r"dpars names \['mum1'\]; the non-reference categories' linear predictors are \['mum2', 'mum3'\]"),
    (dict(dpars={"mum3": {"intercept": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)}}), r"dpars\['mum3'\] is a mapping with the keys"),
    (dict(dpars={"mum3": ab.Normal(0, 1)}), r"dpars\['mum3'\] is a mapping with the keys"),
    (dict(dpars=None), r"the categories \['mum3'\] have formulas of their own .* need their own priors"),
    (dict(dpars={"mum3": {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}}),
     r"category 'mum3' \('mum3 ~ w \+ \(1 \| h\)', its priors from dpars\['mum3'\]\)"),
    (dict(b=None), r"category 'mum2' .*its priors from the top-level intercept, b and sd"),
]


@pytest.mark.parametrize("change, match", PRIOR_REFUSALS)
def test_prior_refusals(change, match):
    s = parse_multinomial(RICH, frame(), categorical={"z": "p"})
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), dpars={"mum3": OWN})
    base.update(change)
    with pytest.raises(PriorError, match=match):
        ab.Priors(**base).validate(s, get_family("multinomial"))


def test_phi_dpars_and_stray_priors():
    s = parse_multinomial(RICH, frame(), categorical={"z": "p"})
    dm = get_family("dirichlet_multinomial")
    with pytest.raises(PriorError, match="the dirichlet_multinomial family needs a prior on the precision parameter phi"):
        pri(dpars={"mum3": OWN}).validate(s, dm)
    for bad in (ab.Normal(0, 1), {"intercept": ab.Normal(0, 1)}):
        with pytest.raises(PriorError, match="prior for 'phi' must be one positive-support prior .*divergence D136"):
            pri(dpars={"mum3": OWN}, phi=bad).validate(s, dm)
    every = {"mum2": {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), "sd": ab.HalfNormal(1)}, "mum3": OWN}
    with pytest.raises(PriorError, match=r"every category's priors are in dpars, so \['b'\] at the top level apply to none"):
        ab.Priors(b=ab.Normal(0, 1), dpars=every).validate(s, get_family("multinomial"))
    gauss = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), dpars={"mum2": {}})
    with pytest.raises(PriorError, match="dpars gives the priors of a multinomial model's category formulas"):
        ab.gaussian("x ~ w", frame(), gauss)
    with pytest.raises(PriorError, match="a multivariate model takes none"):
        ab.Priors(resp={"x": {}}, dpars={"mum2": {}}).validate_multivariate(SimpleSpec(), "gaussian")


class SimpleSpec:
    responses = ("x", "w")
    rescor = False


# -- the model -----------------------------------------------------------------------------------------------------------------
def test_names_design_and_graph():
    m = rich()
    assert isinstance(m, MultinomialModel) and m.bambi_model is None and m.categories == ("m1", "m2", "m3")
    assert m.names == {"linpred": "mu", "mean": "p", "y": "cbind(m1, m2, m3)", "category_dim": "category", "mu_dim": "mu_dim",
                       "mu:mum2": "mum2", "mu:mum3": "mum3"}
    assert m.mu_names["mum2"] == {"value": "mum2", "intercept": "mum2_Intercept", "raw_intercept": "mum2_b_Intercept",
                                  "b:x": "mum2_x", "b:z": "mum2_z"}
    assert m.mu_coefficients == {"mum2": (("x", "mum2_x", None), ("z[q]", "mum2_z", 0), ("z[r]", "mum2_z", 1)),
                                 "mum3": (("w", "mum3_w", None),)}
    assert m.mu_group_names[("mum3", "h", "Intercept")] == {"sd": "mum3_1|h_sigma", "offset": "mum3_1|h_offset",
                                                            "group_effect": "mum3_1|h", "coordinate": "h__factor_dim"}
    assert [v.name for v in m.pymc_model.value_vars] == ["mum2_Intercept", "mum2_x", "mum2_z", "mum2_1|g_sigma_log__", "mum2_1|g_offset",
                                                         "mum3_Intercept", "mum3_w", "mum3_1|h_sigma_log__", "mum3_1|h_offset"]
    coords = m.pymc_model.coords
    assert list(coords["category"]) == ["m1", "m2", "m3"] and list(coords["mu_dim"]) == ["m2", "m3"]
    assert list(coords["mum2_z_dim"]) == ["q", "r"] and list(coords["g__factor_dim"]) == ["a", "b", "c"]
    d, df = m.design, frame()
    assert np.array_equal(d.y, df[["m1", "m2", "m3"]].to_numpy()) and np.array_equal(d.trials, df["size"]) and d.categories == m.categories
    md = d.mus["mum2"]
    X = np.column_stack([df["x"], df["z"] == "q", df["z"] == "r"]).astype(float)
    np.testing.assert_allclose(md.X_raw, X) and np.testing.assert_allclose(md.x_means, X.mean(0))
    np.testing.assert_allclose(md.X_model, X - X.mean(0))
    assert md.columns == ("x", "z[q]", "z[r]") and md.Z.shape == (12, 3) and d.mus["mum3"].Z.shape == (12, 2)
    assert np.array_equal(md.group_blocks[("g", "Intercept")].group_index, [0, 0, 1, 1, 2, 2] * 2)
    assert m.coefficients == () and m.group_names == {} and m.learned_parameters == {} and m.link == "logit"
    dm = rich("dirichlet_multinomial")
    assert dm.names["phi"] == "phi" and "phi_log__" in {v.name for v in dm.pymc_model.value_vars}


def _by_hand(m, point, jacobian=True):
    """The model's log density at ``point`` written out with SciPy: each category's linear predictor, the likelihood, every
    prior, the standardised effects and the log-scale Jacobians."""
    df = frame()
    dp = point["dpars"]
    X2 = np.column_stack([df["x"], df["z"] == "q", df["z"] == "r"]).astype(float)
    X3 = df[["w"]].to_numpy(dtype=float)
    g, h = pd.factorize(df["g"], sort=True)[0], pd.factorize(df["h"], sort=True)[0]
    eta2 = dp["mum2"]["intercept"] + (X2 - X2.mean(0)) @ dp["mum2"]["b"] + dp["mum2"]["sd"] * dp["mum2"]["z"][g]
    eta3 = dp["mum3"]["intercept"] + (X3 - X3.mean(0)) @ dp["mum3"]["b"] + dp["mum3"]["sd"] * dp["mum3"]["z"][h]
    full = np.column_stack([np.zeros(12), eta2, eta3])
    p = softmax(full, axis=1)
    y = df[["m1", "m2", "m3"]].to_numpy()
    n = y.sum(1)
    if "phi" in point:
        phi = point["phi"]
        a = phi * p
        lik = (gammaln(n + 1) - gammaln(y + 1).sum(1) + gammaln(phi) - gammaln(n + phi) + (gammaln(y + a) - gammaln(a)).sum(1)).sum()
        lik += gamma_dist.logpdf(phi, 2, scale=1 / 0.5) + (np.log(phi) if jacobian else 0.0)
    else:
        lik = sum(multinomial_dist.logpmf(y[i], n[i], p[i]) for i in range(12))
    from scipy.stats import expon, t as student

    prior = (norm.logpdf(dp["mum2"]["intercept"], 0, 2) + norm.logpdf(dp["mum2"]["b"], 0, 1).sum() + halfnorm.logpdf(dp["mum2"]["sd"])
             + student.logpdf(dp["mum3"]["intercept"], 3, 0, 2) + norm.logpdf(dp["mum3"]["b"], 0, 0.5).sum()
             + expon.logpdf(dp["mum3"]["sd"], scale=0.5) + norm.logpdf(dp["mum2"]["z"]).sum() + norm.logpdf(dp["mum3"]["z"]).sum())
    jac = np.log(dp["mum2"]["sd"]) + np.log(dp["mum3"]["sd"]) if jacobian else 0.0
    return lik + prior + jac


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_density_by_hand_and_on_the_jax_path(family):
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    m = rich(family)
    nat = rich_point(m)
    vp = m.value_point(**nat)
    pmm = m.pymc_model
    np.testing.assert_allclose(float(pmm.compile_logp()(vp)), _by_hand(m, nat), rtol=1e-12)
    np.testing.assert_allclose(float(pmm.compile_logp(jacobian=False)(vp)), _by_hand(m, nat, jacobian=False), rtol=1e-12)
    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    args = [vp[n] for n in names]
    np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(vp)), rtol=1e-12)
    g_jax = np.concatenate([np.ravel(v) for v in jax.grad(lambda q: jf(q))(args)])
    # PyTensor differentiates lgamma through its digamma, about 1e-10 off relative (likelihoods; JAX's is exact to rounding):
    # the Dirichlet-multinomial's phi coordinate differs by 6e-10 relative here, inside the oracle's 1e-7
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(vp), dtype=float), rtol=1e-8, atol=1e-12)
    # the graph's deterministics: each category's values, mu and the raw-scale intercept
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"], pmm["mum2_b_Intercept"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    mu, raw = fn(vp)
    X2 = m.design.mus["mum2"].X_raw
    assert mu.shape == (12, 2)
    np.testing.assert_allclose(raw, nat["dpars"]["mum2"]["intercept"] - X2.mean(0) @ nat["dpars"]["mum2"]["b"], rtol=1e-14)


def test_model_refusals_warnings_and_dispatch():
    df = frame()
    with pytest.raises(ValueError, match=r"'probit' is not a supported link for family 'multinomial'. Supported links are: 'logit'"):
        ab.Model(F, df, pri(), family="multinomial", link="probit")
    assert ab.Model(F, df, pri(), family="multinomial", link="logit").link == "logit"
    for kw, match in ((dict(nu=4), "nu applies only to the student or frechet family"),
                      (dict(quantile=0.3), "quantile= fixes the asym_laplace family's quantile"),
                      (dict(families=["gaussian"]), "families= and order= are a mixture's")):
        with pytest.raises(ValueError, match=match):
            ab.Model(F, df, pri(), family="multinomial", **kw)
    with pytest.raises(FormulaError, match="in a multivariate formula is not supported"):
        ab.Model(F, df, pri(), family="multinomial", imputed=("x",))
    with pytest.raises(FormulaError, match="data2 holds the matrices of terms such as sar\\(M\\)"):
        ab.Model(F, df, pri(), family="multinomial", data2={"M": np.eye(12)})
    with pytest.raises(FormulaError, match="center_predictors=False with the multinomial family is not supported"):
        ab.multinomial(F, df, pri(), center_predictors=False)
    with pytest.raises(PriorError, match="priors must be an ab.Priors"):
        ab.multinomial(F, df, {"b": ab.Normal(0, 1)})
    with pytest.raises(FormulaError, match=r"columns not in data: \['g'\]"):
        ab.multinomial(F, df.drop(columns="g"), pri())
    with pytest.raises(ValueError, match="MultinomialModel builds the multinomial and dirichlet_multinomial families; got 'gaussian'"):
        MultinomialModel(F, df, pri(), family="gaussian")
    empty = df.assign(m3=0, size=df["m1"] + df["m2"])
    m = ab.multinomial(F, empty, pri())
    assert any("the categories ['m3'] have no counts" in w for w in m.warnings)
    cells = ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ z * h", df.assign(h=np.where(df["z"] == "q", "u", df["h"])), pri(sd=None),
                           categorical={"z": "p", "h": "u"})
    assert any("have no fitted rows" in w and "formula of mum2" in w for w in cells.warnings)
    unused = ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ z", df.assign(z=pd.Categorical(df["z"], categories=["p", "q", "r", "s"])),
                            pri(sd=None), categorical={"z": "p"})
    assert any("declared categories ['s'] of 'z'" in w for w in unused.warnings)
    with pytest.raises(FormulaError, match=r"generated variable/coordinate name collision: \['mum2_b_Intercept', 'mum3_b_Intercept'\]"):
        ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ b_Intercept", df.assign(b_Intercept=df["x"]), pri(sd=None))
    with pytest.raises(FormulaError, match="the suffix '_offset' is reserved"):
        ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ x_offset", df.assign(x_offset=df["x"]), pri(sd=None))
    same = ab.Model(F, df, pri(), family="multinomial")
    other = ab.multinomial(F, df, pri())
    assert type(same) is MultinomialModel and same.target_identity == other.target_identity and same.describe() == other.describe()
    assert type(ab.Model("x ~ w", df, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))) is ab.Model


def test_value_point_and_its_refusals():
    m, dm = rich(), rich("dirichlet_multinomial")
    vp = m.value_point(**rich_point(m))
    assert vp["mum2_1|g_sigma_log__"] == pytest.approx(math.log(0.7)) and vp["mum2_z"].shape == (2,) and vp["mum3_w"].shape == ()
    assert dm.value_point(**rich_point(dm))["phi_log__"] == pytest.approx(math.log(2.5))
    good = rich_point(m)

    def with_(name, **entry):
        out = json.loads(json.dumps({k: {kk: (np.asarray(vv).tolist()) for kk, vv in v.items()} for k, v in good["dpars"].items()}))
        out[name].update(entry)
        return {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in out.items()}

    cases = [(dict(dpars=good["dpars"], sigma=1.0), "point takes dpars=\\{...\\}; unexpected \\['sigma'\\]"),
             (dict(dpars={"mum2": good["dpars"]["mum2"]}), "dpars must be a mapping over exactly"),
             (dict(dpars=with_("mum2", L=1.0)), "dpars\\['mum2'\\] is a mapping with the keys"),
             (dict(dpars=with_("mum2", intercept=float("nan"))), "the intercept of 'mum2' must be a finite scalar"),
             (dict(dpars=with_("mum2", b=[0.1, 0.2])), "dpars\\['mum2'\\]\\['b'\\] must have 3 finite entries"),
             (dict(dpars=with_("mum3", z=[0.1, 0.2, 0.3])), "z of 'h' in 'mum3' must have 2 finite entries"),
             (dict(dpars=with_("mum3", sd=-1.0)), "sd of 'h' in 'mum3' must be a finite positive scalar"),
             (dict(dpars=with_("mum3", sd=None)), "sd is required for the group-level terms of 'mum3'"),
             (dict(dpars=with_("mum3", sd={"g": 1.0})), "sd of 'mum3' must be a mapping over exactly its grouping factors"),
             (dict(dpars=good["dpars"], phi=1.0), "sigma, shape and phi do not apply to the multinomial family")]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            m.value_point(**kwargs)
    with pytest.raises(ValueError, match="phi \\(the precision\\) is required for the dirichlet_multinomial family"):
        dm.value_point(dpars=good["dpars"])
    other = rich()
    other.pymc_model = dm.pymc_model                                                # a graph built from other names
    with pytest.raises(RuntimeError, match=r"value names .* do not match PyMC's .*'phi_log__'"):
        other.value_point(dpars=good["dpars"])
    flat = ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ x", frame(), pri(sd=None))
    assert set(flat.value_point(dpars={"mum2": {"intercept": 0.0, "b": [0.1]}, "mum3": {"intercept": 0.0, "b": [0.1]}})) == {
        "mum2_Intercept", "mum2_x", "mum3_Intercept", "mum3_x"}
    with pytest.raises(ValueError, match="has no group-level term; sd and z do not apply to 'mum2'"):
        flat.value_point(dpars={"mum2": {"intercept": 0.0, "b": [0.1], "sd": 1.0}, "mum3": {"intercept": 0.0, "b": [0.1]}})
    two = ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ x + (1 | g) + (1 | h)", frame(), pri(sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}))
    entry = {"intercept": 0.0, "b": [0.1], "sd": {"g": 1.0, "h": 0.5}, "z": {"g": np.zeros(3), "h": np.zeros(2)}}
    assert two.value_point(dpars={"mum2": entry, "mum3": entry})["mum3_1|h_sigma_log__"] == pytest.approx(math.log(0.5))
    with pytest.raises(ValueError, match="sd of 'mum2' must be a mapping over exactly its grouping factors \\['g', 'h'\\]"):
        two.value_point(dpars={"mum2": {**entry, "sd": 1.0}, "mum3": entry})


# -- describe() ----------------------------------------------------------------------------------------------------------------
def test_describe():
    text = rich().describe()
    for line in ("AutoBayes multinomial model: 3 categories (m1, m2, m3; reference 'm1'), 2 linear predictor(s) (mum2, mum3)",
                 "  mum3      : mum3 ~ w + (1 | h)   (the category's own formula)",
                 "  levels g  : 0=a, 1=b, 2=c", "  z         : categorical in the formula(s) of mum2, reference 'p'",
                 "Linear predictor mum2 (category 'm2'): mum2 ~ x + z + (1 | g)",
                 "  mum2_i = Intercept_mum2 + (x_i - mean) * b_mum2[x] + (1[z_i = q] - mean) * b_mum2[zq] + (1[z_i = r] - mean) * b_mum2[zr]"
                 " + u_mum2_g[g_i]",
                 "  Intercept_mum3 ~ StudentT(nu=3, mu=0, sigma=2)   (from dpars['mum3'])", "  sd_mum3_h ~ Exponential(lam=2)",
                 "  y_i ~ Multinomial(size_i, p_i): log p(y_i) = log C(y_i) + sum_k y_k,i log p_k,i (brms's",
                 "  mum2_z                   b_mum2[zq, zr] (vector; dims mum2_z_dim, coordinates q, r)",
                 "  mum3_1|h_sigma_log__     log(sd_mum3_h)"):
        assert line in text, line
    assert "Deterministic: mum2_b_Intercept = b_mum2_Intercept, mum3_b_Intercept" in text and "WARNING" not in text
    dm = ab.dirichlet_multinomial(" cbind(m1, m2, m3)|trials(size) ~ x+(1|g)", frame(),
                                  ab.default_priors(F, frame(), family="dirichlet_multinomial"))
    text = dm.describe()
    assert "  expanded  : cbind(m1, m2, m3) | trials(size) ~ x + (1 | g)" in text
    assert ("  phi ~ Gamma(alpha=0.01, beta=0.01)   (brms default gamma(0.01, 0.01), lb 0)   (the precision; sampled as log(phi))"
            in text)
    assert "  y_i ~ DirichletMultinomial(size_i, phi p_i)" in text and "  phi_log__                log(phi)" in text
    assert "Default priors (ab.default_priors; docs/multinomial.md)" not in text and "gamma(0.01, 0.01) on phi." in text
    empty = ab.multinomial(F, frame().assign(m3=0, size=lambda d: d["m1"] + d["m2"]), pri())
    assert "WARNING: the categories ['m3'] have no counts" in empty.describe()


# -- predictions ---------------------------------------------------------------------------------------------------------------
def posterior(m, seed=0, draws=4, chains=1):
    rng = np.random.default_rng(seed)
    shape = (chains, draws)
    data, coords = {}, {"chain": np.arange(chains), "draw": np.arange(draws)}
    for d in m.spec.mus:
        roles = m.mu_names[d.response]
        data[roles["intercept"]] = (("chain", "draw"), rng.normal(size=shape))
        data[roles["raw_intercept"]] = data[roles["intercept"]]          # a fit's posterior holds it; predictions never read it
        for t in d.pop_terms:
            var = roles[f"b:{t.name}"]
            if t.is_vector:
                data[var] = (("chain", "draw", f"{var}_dim"), rng.normal(size=shape + (len(t.levels),)))
                coords[f"{var}_dim"] = list(t.levels)
            else:
                data[var] = (("chain", "draw"), rng.normal(size=shape))
        for t in d.group_terms:
            r = m.mu_group_names[(d.response, t.factor, "Intercept")]
            levels = m.design.mus[d.response].group_blocks[(t.factor, "Intercept")].levels
            data[r["sd"]] = (("chain", "draw"), np.exp(rng.normal(size=shape)))
            data[r["group_effect"]] = (("chain", "draw", r["coordinate"]), rng.normal(size=shape + (len(levels),)))
            coords[r["coordinate"]] = list(levels)
    if "phi" in m.names:
        data["phi"] = (("chain", "draw"), np.exp(rng.normal(size=shape)))
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def eta_by_hand(m, post, df, groups=True):
    """(chain, draw, row, K - 1): each category's centred intercept, its design centred on the fitted means and its group
    effects at the rows' (fitted) levels."""
    out = []
    for d in m.spec.mus:
        name, md = d.response, m.design.mus[d.response]
        cols = {"x": df["x"].to_numpy(float), "z[q]": (df["z"] == "q").to_numpy(float), "z[r]": (df["z"] == "r").to_numpy(float),
                "w": df["w"].to_numpy(float)}
        X = np.column_stack([cols[c] for c in md.columns]) - md.x_means
        B = np.stack([post[v].values if j is None else post[v].values[..., j] for _, v, j in m.mu_coefficients[name]], axis=-1)
        eta = post[m.mu_names[name]["intercept"]].values[..., None] + B @ X.T
        for t in d.group_terms if groups else ():
            r = m.mu_group_names[(name, t.factor, "Intercept")]
            levels = list(md.group_blocks[(t.factor, "Intercept")].levels)
            eta = eta + post[r["group_effect"]].values[..., [levels.index(v) for v in df[t.factor]]]
        out.append(eta)
    return np.stack(out, axis=-1)


NEW = pd.DataFrame({"x": [0.3, -1.2, 2.0], "w": [1.0, 0.0, -0.5], "z": ["q", "p", "r"], "g": ["b", "c", "a"], "h": ["v", "u", "v"],
                    "size": [4, 0, 9]})


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_predictions_by_hand(family):
    m = rich(family)
    idata = posterior(m, draws=5)
    post = idata.posterior
    lin = m.predict(idata, kind="linear")
    assert lin["eta"].dims == ("chain", "draw", "__obs__", "category") and list(lin["category"].values) == ["m2", "m3"]
    np.testing.assert_allclose(lin["eta"].values, eta_by_hand(m, post, frame()), rtol=1e-13)
    assert (lin.attrs["prediction_scale"], lin.attrs["link"], lin.attrs["new_groups"]) == ("linear", "logit", "in_sample")
    assert json.loads(lin.attrs["new_groups_by_factor_json"]) == {"g": "error", "h": "error"} and lin.attrs["fit_binding_status"] == "unverified"
    mean = m.predict(idata, kind="mean")
    p = softmax(np.concatenate([np.zeros(lin["eta"].shape[:-1] + (1,)), lin["eta"].values], axis=-1), axis=-1)
    np.testing.assert_allclose(mean["p"].values, p, rtol=1e-13)
    np.testing.assert_allclose(mean["expected"].values, frame()["size"].to_numpy()[:, None] * p, rtol=1e-13)
    assert list(mean["category"].values) == ["m1", "m2", "m3"] and json.loads(mean.attrs["categories_json"]) == ["m1", "m2", "m3"]
    checked = check_multinomial_newdata(m, NEW.assign(extra=1.0), need_trials=True)
    assert list(checked.columns) == ["x", "z", "w", "g", "h", "size"] and checked["size"].dtype == np.int64
    new = m.predict(idata, newdata=NEW, kind="linear")
    np.testing.assert_allclose(new["eta"].values, eta_by_hand(m, post, NEW), rtol=1e-13)
    assert (new.attrs["new_groups"], new.attrs["include_group_effects"]) == ("per_factor", 1)
    bare = m.predict(idata, newdata=NEW.drop(columns=["g", "h"]), kind="mean", include_group_effects=False)
    np.testing.assert_allclose(bare["expected"].values[..., 1, :], 0.0) and np.testing.assert_allclose(bare["expected"].values.sum(-1)[..., 2], 9.0)
    nog = m.predict(idata, kind="linear", include_group_effects=False)
    np.testing.assert_allclose(nog["eta"].values, eta_by_hand(m, post, frame(), groups=False), rtol=1e-13)
    assert (nog.attrs["new_groups"], nog.attrs["new_levels"]) == ("not_applicable", [])
    y = m.predict(idata, newdata=NEW, kind="response", seed=4)
    assert y["y"].dtype == np.int64 and np.array_equal(y["y"].values.sum(-1), np.broadcast_to([4, 0, 9], (1, 5, 3)))
    assert np.array_equal(m.predict(idata, newdata=NEW, kind="response", seed=4)["y"].values, y["y"].values)


def test_unseen_levels_and_prediction_refusals():
    m = rich()
    idata = posterior(m, draws=6)
    post = idata.posterior
    unseen = NEW.assign(g=["zz", "c", "zz"])
    with pytest.raises(FormulaError, match=r"newdata has levels of 'g' not seen in training: \['zz'\]"):
        m.predict(idata, newdata=unseen, kind="linear")
    known = eta_by_hand(m, post, unseen.assign(g=["a", "c", "a"]))
    u_a = post["mum2_1|g"].values[..., 0]
    got = m.predict(idata, newdata=unseen, kind="linear", new_groups={"g": "population", "h": "error"}, seed=8)
    rng = np.random.default_rng(8)
    effect = post["mum2_1|g_sigma"].values * rng.normal(0.0, 1.0, size=(1, 6))
    want = known.copy()
    want[..., [0, 2], 0] += (effect - u_a)[..., None]
    np.testing.assert_allclose(got["eta"].values, want, rtol=1e-12)
    assert json.loads(got.attrs["new_levels_by_factor_json"]) == {"g": ["zz"], "h": []}
    got = m.predict(idata, newdata=unseen, kind="linear", new_groups="resample_observed", seed=9)
    chosen = np.random.default_rng(9).integers(0, 3, size=(1, 6))
    want = known.copy()
    want[..., [0, 2], 0] += (np.take_along_axis(post["mum2_1|g"].values, chosen[..., None], axis=-1)[..., 0] - u_a)[..., None]
    np.testing.assert_allclose(got["eta"].values, want, rtol=1e-12)
    for kwargs, error, match in (
            (dict(kind="probability"), ValueError, "kind must be 'mean'"),
            (dict(kind="linear", dpar="mum2"), ValueError, "dpar and nlpar do not apply \\(got 'mum2'\\)"),
            (dict(kind="linear", nlpar="a"), ValueError, "dpar and nlpar do not apply \\(got 'a'\\)"),
            (dict(kind="linear", include_group_effects="no"), ValueError, "include_group_effects must be boolean"),
            (dict(kind="linear", include_group_effects=False, new_groups="population"), ValueError, "new_groups must stay at its default"),
            (dict(kind="linear", new_groups={"g": "population"}), ValueError, "new_groups must cover exactly the grouping factors"),
            (dict(kind="mean", newdata=NEW.drop(columns="size")), FormulaError, r"newdata lacks columns \['size'\] \(the trials 'size'"),
            (dict(kind="linear", newdata=NEW.drop(columns="w")), FormulaError, r"newdata lacks columns \['w'\]$"),
            (dict(kind="linear", newdata=NEW.assign(x=[0.1, np.nan, 0.2])), FormulaError, "newdata contains missing values"),
            (dict(kind="linear", newdata=NEW.assign(x=["a", "b", "c"])), FormulaError, "newdata column 'x' must be numeric"),
            (dict(kind="linear", newdata=NEW.assign(z=["q", "s", "r"])), FormulaError, "levels of the categorical predictor 'z'"),
            (dict(kind="mean", newdata=NEW.assign(size=[1.5, 0, 9])), FormulaError, "newdata trials 'size' has a non-integral value")):
        with pytest.raises(error, match=match):
            m.predict(idata, **kwargs)
    bad = idata.copy()
    bad.posterior["mum2_Intercept"] = bad.posterior["mum2_Intercept"] * np.inf
    with pytest.raises(ValueError, match="the categories' linear predictors contain nonfinite draws"):
        m.predict(bad, kind="linear")
    flat = ab.multinomial("cbind(m1, m2, m3) | trials(size) ~ log(xpos)", frame(), pri(sd=None))
    one = posterior(flat)
    assert flat.predict(one, kind="mean").attrs["new_groups"] == "not_applicable"
    with pytest.raises(ValueError, match="this model has no group-level term; new_groups does not apply"):
        flat.predict(one, kind="mean", new_groups="population")
    with pytest.raises(FormulaError, match="log"):
        flat.predict(one, newdata=NEW.assign(xpos=[-1.0, 1.0, 2.0]), kind="linear")
    single = ab.multinomial(F, frame(), pri())
    s1 = single.predict(posterior(single), newdata=NEW.assign(g=["a", "zz", "b"]), kind="linear", new_groups="population", seed=1)
    assert (s1.attrs["new_groups"], s1.attrs["new_levels"]) == ("population", ["zz"])


# -- post-processing -----------------------------------------------------------------------------------------------------------
def test_log_lik_loo_and_the_refused_summaries():
    for family in ("multinomial", "dirichlet_multinomial"):
        m = rich(family)
        idata = posterior(m, draws=6)
        ll = ab.log_lik(m, idata)
        p = ab.predict(m, idata, kind="mean")["p"].values
        y = frame()[["m1", "m2", "m3"]].to_numpy()
        n = y.sum(1)
        if family == "multinomial":
            want = np.array([[[multinomial_dist.logpmf(y[i], n[i], p[c, d, i]) for i in range(12)] for d in range(6)] for c in range(1)])
        else:
            phi = idata.posterior["phi"].values[..., None, None]
            a = phi * p
            want = gammaln(n + 1) - gammaln(y + 1).sum(1) + gammaln(phi[..., 0]) - gammaln(n + phi[..., 0]) + (gammaln(y + a) - gammaln(a)).sum(-1)
        np.testing.assert_allclose(ll.values, want, rtol=1e-11)
        assert ll.dims == ("chain", "draw", "__obs__")
        res = ab.loo(m, idata)
        assert res.pointwise.shape == (12, 5) and ab.waic(m, idata).pointwise.shape == (12, 3)
        with pytest.raises(ValueError, match="bayes_R2\\(\\) is not defined for the .*multinomial family's unordered categories"):
            ab.bayes_R2(m, idata)
        with pytest.raises(ValueError, match="hypothesis\\(\\) of a model with the .*multinomial family's categories is not offered"):
            ab.hypothesis(m, idata, "mum2_x > 0")
        with pytest.raises(ValueError, match="conditional_effects\\(\\) of a model with the .*multinomial family's categories"):
            ab.conditional_effects(m, idata, "x")
        with pytest.raises(ValueError, match="prior_draws\\(\\) of a model with the .*multinomial family's categories"):
            ab.prior_draws(m)


# -- brms's default priors (brms 2.23's default_prior(), printed for these formulas) ---------------------------------------------
ST = "student_t(3, 0, 2.5)"


def _rows_of(dpar, coefs=(), groups=()):
    rows = [(ST, "Intercept", "", "", dpar, "")]
    rows += [("", "b", "", "", dpar, "")] + [("", "b", c, "", dpar, "") for c in coefs] if coefs else []
    if groups:
        rows += [(ST, "sd", "", "", dpar, "0")]
        for g in groups:
            rows += [("", "sd", "", g, dpar, ""), ("", "sd", "Intercept", g, dpar, "")]
    return rows


BRMS_TABLES = {   # (formula, family, keyword arguments) -> brms's rows (prior, class, coef, group, dpar, lb), every source "default"
    "own formulas": ((["cbind(m1, m2, m3) | trials(size) ~ 1", "mum2 ~ z", "mum3 ~ w"], "multinomial", {"categorical": {"z": "p"}}),
                     _rows_of("mum2", ("zq", "zr")) + _rows_of("mum3", ("w",))),
    "no group": (("cbind(m1, m2, m3) | trials(size) ~ x", "multinomial", {}), _rows_of("mum2", ("x",)) + _rows_of("mum3", ("x",))),
    "intercept only": (("cbind(m1, m2, m3) | trials(size) ~ 1", "dirichlet_multinomial", {}),
                       [("gamma(0.01, 0.01)", "phi", "", "", "", "0")] + _rows_of("mum2") + _rows_of("mum3")),
    "two factors, an interaction": (("cbind(m1, m2, m3) | trials(size) ~ x * z + (1 | g) + (1 | h)", "multinomial",
                                     {"categorical": {"z": "p"}}),
                                    _rows_of("mum2", ("x", "x:zq", "x:zr", "zq", "zr"), ("g", "h"))
                                    + _rows_of("mum3", ("x", "x:zq", "x:zr", "zq", "zr"), ("g", "h"))),
    "reordered, a group of its own": ((["cbind(m2, m1, m3) | trials(size) ~ x + (1 | g)", "mum3 ~ log(xpos) + (1 | h)"],
                                       "dirichlet_multinomial", {}),
                                      [("gamma(0.01, 0.01)", "phi", "", "", "", "0")] + _rows_of("mum1", ("x",), ("g",))
                                      + _rows_of("mum3", ("logxpos",), ("h",))),
}


@pytest.mark.parametrize("label", sorted(BRMS_TABLES))
def test_default_priors_are_brms_tables(label):
    (formula, family, kwargs), want = BRMS_TABLES[label]
    priors = ab.default_priors(formula, frame(), family=family, **kwargs)
    m = ab.Model(formula, frame(), priors, family=family, **kwargs)
    rows = m.prior_table()
    got = [(r["prior"], r["class"], r["coef"], r["group"], r["dpar"], r["lb"]) for r in rows]
    assert got == want, label
    assert {r["source"] for r in rows} == {"default"} and {r["resp"] + r["nlpar"] + r["ub"] + r["tag"] for r in rows} == {""}
    assert m.default_prior_labels == [lab for lab, _ in prior_labels(priors, m.spec, m.family_spec)]


def test_default_prior_refusals_and_user_prior_tables():
    with pytest.raises(FormulaError, match="center_predictors=False with the multinomial family is not supported"):
        ab.default_priors(F, frame(), family="multinomial", center_predictors=False)
    with pytest.raises(FormulaError, match="in a multivariate formula is not supported"):
        ab.default_priors(F, frame(), family="multinomial", imputed=("x",))
    for kw in (dict(nu=4), dict(data2={"M": np.eye(12)})):
        with pytest.raises(ValueError, match="nu, quantile, families, order and data2 do not apply to the multinomial family"):
            ab.default_priors(F, frame(), family="multinomial", **kw)
    p = pri(b={"x": ab.Normal(0, 1), "z": ab.Normal(0, 2)}, sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(2)},
            dpars={"mum3": {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}})
    m = ab.multinomial(["cbind(m1, m2, m3) | trials(size) ~ x + z + (1 | g) + (1 | h)", "mum3 ~ w"], frame(), p, categorical={"z": "p"})
    got = [(r["prior"], r["class"], r["coef"], r["group"], r["dpar"], r["lb"], r["source"]) for r in m.prior_table()]
    assert got == [("normal(0, 2)", "Intercept", "", "", "mum2", "", "user"),
                   ("", "b", "", "", "mum2", "", "default"), ("normal(0, 1)", "b", "x", "", "mum2", "", "user"),
                   ("normal(0, 2)", "b", "zq", "", "mum2", "", "user"), ("normal(0, 2)", "b", "zr", "", "mum2", "", "user"),
                   ("", "sd", "", "", "mum2", "0", "default"), ("normal(0, 1)", "sd", "", "g", "mum2", "", "user"),
                   ("", "sd", "Intercept", "g", "mum2", "", "user"), ("normal(0, 2)", "sd", "", "h", "mum2", "", "user"),
                   ("", "sd", "Intercept", "h", "mum2", "", "user"),
                   ("normal(0, 1)", "Intercept", "", "", "mum3", "", "user"), ("normal(0, 0.5)", "b", "", "", "mum3", "", "user"),
                   ("", "b", "w", "", "mum3", "", "user")]
    assert m.prior_table() == p.table(m.spec, m.family_spec)


# -- the identity, the fit plumbing and records, the prior predictive -------------------------------------------------------
def test_target_identity():
    a, b = rich(), rich()
    assert a.target_identity == b.target_identity and set(a.target_identity) == {"schema", "sha256"}
    other = ab.Model(RICH, frame(), pri(dpars={"mum3": {**OWN, "b": ab.Normal(0, 0.6)}}), family="multinomial", categorical={"z": "p"})
    data = ab.Model(RICH, frame(seed=2), pri(dpars={"mum3": OWN}), family="multinomial", categorical={"z": "p"})
    assert len({a.target_identity["sha256"], other.target_identity["sha256"], data.target_identity["sha256"],
                rich("dirichlet_multinomial").target_identity["sha256"]}) == 4


@pytest.mark.nosampling                                     # pm.sample is stubbed: nothing samples
def test_fit_samples_the_graph_and_records_it(monkeypatch):
    """``ab.fit`` hands the model to its own ``sample_posterior`` (PyMC's ``sample`` on the graph, stubbed here: nothing
    samples); the fit is bound to the model, predicts, and makes a fit record that ``compare_records`` accepts."""
    import copy
    import warnings

    import pymc as pm

    import autobayes.session as session

    m = rich("dirichlet_multinomial")
    fake = posterior(m, draws=3, chains=2)
    calls = {}

    def sample(**kwargs):
        calls.update(kwargs)
        out = copy.deepcopy(fake)
        out.posterior["mum2_1|g_offset"] = out.posterior["mum2_1|g"]              # dropped from the posterior, as Bambi drops offsets
        return out

    monkeypatch.setattr(pm, "sample", sample)
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = ab.fit(m, chains=2, draws=3, tune=0, seed=5)
    assert calls["model"] is m.pymc_model and calls["nuts_sampler"] == "pymc" and calls["random_seed"] == 5
    assert "mu" not in calls["var_names"] and "mum2" not in calls["var_names"] and "mum2_1|g" in calls["var_names"]
    assert "mum2_1|g_offset" not in fit.idata.posterior and fit.target_identity == m.target_identity
    assert m.predict(fit, kind="mean").attrs["fit_binding_status"] == "matched"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # synthetic draws: ArviZ's diagnostics divide by zero
        rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["formula"] == RICH and rec["model"]["fit_binding_status"] == "matched"
    assert rec["model"]["multinomial"]["categories"] == ["m1", "m2", "m3"] and rec["model"]["priors"]["mum3:sd:h"] == "Exponential(lam=2)"
    assert rec["model"]["multinomial"]["mus"]["mum3"]["own_formula"] and rec["model"]["priors"]["phi"] == "Gamma(alpha=2, beta=0.5)"
    assert set(rec["posterior"]) == {"mum2_Intercept", "mum2_b_Intercept", "mum3_Intercept", "mum3_b_Intercept", "mum2_x", "mum2_z[q]",
                                     "mum2_z[r]", "mum3_w", "mum2_1|g_sigma", "mum3_1|h_sigma", "phi"}
    same = ab.compare_records(rec, json.loads(json.dumps(rec)))
    assert same["identity_status"] == "matched" and set(same["rows"]) == set(rec["posterior"])
    flat = ab.multinomial(F, frame(), ab.default_priors(F, frame(), family="multinomial"))
    monkeypatch.setattr(pm, "sample", lambda **kw: copy.deepcopy(posterior(flat, draws=3, chains=2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rec = ab.fit_record(flat, ab.fit(flat, chains=2, draws=3, tune=0), panel={"values": {}, "status": {}, "overall": "x"})
    assert rec["model"]["formula"] == F and rec["model"]["default_priors"]["parameters"][0] == "mum2:intercept"
    from types import SimpleNamespace

    unbound = copy.deepcopy(fit.idata)
    unbound.attrs.clear()
    legacy = SimpleNamespace(**{k: getattr(fit, k) for k in ("backend", "device_requested", "device_actual", "platform", "dtype",
                                                              "chain_method", "chains", "draws", "tune", "target_accept", "seed",
                                                              "wall_seconds", "notes")}, idata=unbound)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rec = ab.fit_record(m, legacy, panel={"values": {}, "status": {}, "overall": "x"}, allow_unverified=True)
    assert rec["model"]["fit_binding_status"] == "unverified" and rec["model"]["requested_target_identity"] == m.target_identity


def test_prior_predictive_draws_and_flat_priors():
    for family in ("multinomial", "dirichlet_multinomial"):
        m = rich(family)
        pp = m.prior_predictive(draws=15, seed=2)
        y = pp.prior_predictive["cbind(m1, m2, m3)"].values
        assert y.shape == (1, 15, 12, 3) and np.array_equal(y.sum(-1), np.broadcast_to(frame()["size"].to_numpy(), (1, 15, 12)))
        assert pp.attrs["autobayes_generation_kind"] == "prior_predictive" and pp.prior["mu"].shape == (1, 15, 12, 2)
        assert np.array_equal(ab.prior_predictive(m, draws=15, seed=2).prior_predictive["cbind(m1, m2, m3)"].values, y)
    flat = ab.multinomial(F, frame(), ab.default_priors(F, frame(), family="multinomial"))
    with pytest.raises(ValueError, match=r"the prior of \['mum2:b:x', 'mum3:b:x'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)
    for kwargs, match in ((dict(draws=0), "draws must be a positive integer"), (dict(draws=True), "draws must be a positive integer"),
                          (dict(draws=3, seed=-1), "seed must be a nonnegative integer or None")):
        with pytest.raises(ValueError, match=match):
            rich().prior_predictive(**kwargs)
    import pymc as pm

    real = pm.sample_prior_predictive
    m = rich()

    def reshaped(**kw):
        out = real(**kw)
        out.prior_predictive = out.prior_predictive.isel(category=[0, 1])
        return out

    def shifted(**kw):
        out = real(**kw)
        out.prior_predictive["cbind(m1, m2, m3)"] = out.prior_predictive["cbind(m1, m2, m3)"] + 1
        return out

    for fake, match in ((reshaped, "prior predictive response dimensions do not match"), (shifted, "counts violate the response's support")):
        pm.sample_prior_predictive = fake
        try:
            with pytest.raises(ValueError, match=match):
                m.prior_predictive(draws=4, seed=1)
        finally:
            pm.sample_prior_predictive = real
    with pytest.raises(ValueError, match="prior predictive generation failed"):
        ab.multinomial(F, frame(), pri(intercept=ab.Normal(0, 1e308))).prior_predictive(draws=40, seed=1)
    far = frame().assign(x=lambda d: d["x"] + 1000.0)                        # b x stays finite, its mean times b does not
    with pytest.raises(ValueError, match="prior predictive output contains nonfinite draws"):
        ab.multinomial(F, far, pri(b=ab.Normal(0, 1e306))).prior_predictive(draws=20, seed=1)


# -- the oracle harness sees a corrupted fixture ------------------------------------------------------------------------------
def _corrupted(bundle, mutate):
    import copy

    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_a_corrupted_multinomial_fixture():
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    b = gates.Bundle("multinomial_terms")
    for gate in gates.ALL:
        gate(b)

    def counts(raw):
        raw["standata"]["Y"]["values"][4] += 1
    with pytest.raises(OracleMismatch, match=r"brms's Y \(36, 3\) differs from the AutoBayes count matrix"):
        gates.check_design(_corrupted(b, counts))

    def ncat(raw):
        raw["standata"]["ncat"]["values"][0] = 4
    with pytest.raises(OracleMismatch, match="brms has ncat = 4"):
        gates.check_design(_corrupted(b, ncat))

    def trials(raw):
        raw["standata"]["trials"]["values"][0] += 1
    with pytest.raises(OracleMismatch, match="brms's trials differ"):
        gates.check_design(_corrupted(b, trials))

    def kc(raw):
        raw["standata"]["K_mum3"]["values"][0] = 3
    with pytest.raises(OracleMismatch, match="brms has K_mum3 = 3"):
        gates.check_design(_corrupted(b, kc))

    def linear(raw):
        raw["standata"]["X"] = raw["standata"]["X_mum3"]
    with pytest.raises(OracleMismatch, match="brms has the linear design X"):
        _corrupted(b, linear)

    def design(raw):
        raw["standata"]["X_mum1"]["values"][40] += 1e-11
    with pytest.raises(OracleMismatch, match=r"mum1's design column 'x' matches 0 brms X_mum1 columns"):
        _corrupted(b, design)

    def intercept(raw):
        raw["standata"]["X_mum3"]["values"][2] = 0.0
    with pytest.raises(OracleMismatch, match=r"the one AutoBayes's design leaves, is not the intercept's ones"):
        _corrupted(b, intercept)

    def renamed(raw):
        raw["standata"]["X_mumx"] = raw["standata"].pop("X_mum3")
    with pytest.raises(OracleMismatch, match=r"brms has the category designs \['X_mum1', 'X_mumx'\]"):
        _corrupted(b, renamed)

    def group(raw):
        raw["standata"]["Z_3_mum3_1"]["values"][5] = 2.0
    with pytest.raises(OracleMismatch, match=r"the group-level term of 'g' in 'mum3': 0 brms groups of mum3"):
        _corrupted(b, group)

    def levels(raw):
        raw["groups"][1]["levels"] = ["h2", "h1", "h3", "h4"]
    with pytest.raises(OracleMismatch, match=r"'h' in 'mum1': brms group 2"):
        _corrupted(b, levels)

    def extra(raw):
        raw["groups"].append({"id": 4, "group": "w", "coef": "Intercept", "cn": 1, "cor": True, "levels": ["a"]})
        raw["standata"]["J_4"] = raw["standata"]["J_1"]
    with pytest.raises(OracleMismatch, match=r"brms group-level terms without an AutoBayes counterpart: \[\(4, 'w'\)\]"):
        _corrupted(b, extra)

    def coefficient(raw):
        raw["points"][3]["pars"]["b_mum3"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 3: log density"):
        gates.check_logp(_corrupted(b, coefficient))

    def gradient(raw):
        (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "Intercept_mum3"]
        raw["points"][1]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 1: gradient of mum3_Intercept\[0\]"):
        gates.check_gradient(_corrupted(b, gradient))

    def transform(raw):
        (e,) = [e for e in raw["upars"] if e["par"] == "sd_2"]
        e["transform"] = "identity"
    with pytest.raises(OracleMismatch, match=r"brms transform of sd_2\[1\] is 'identity'; AutoBayes uses 'log'"):
        gates.check_gradient(_corrupted(b, transform))

    def unknown(raw):
        raw["upars"][0]["par"] = "bsp"
    with pytest.raises(OracleMismatch, match="no adapter for brms parameter 'bsp'"):
        gates.check_gradient(_corrupted(b, unknown))

    def unreached(raw):
        raw["upars"][0]["par"] = None
    with pytest.raises(OracleMismatch, match="was not reached by the probe"):
        gates.check_gradient(_corrupted(b, unreached))

    def linpred(raw):
        raw["predictions"]["fitted"]["linpred"]["values"][7] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 7: the graph's 'mu' differs from brms's posterior_linpred"):
        gates.check_derived(_corrupted(b, linpred))
    with pytest.raises(OracleMismatch, match=r"fitted: linear predictor differs from brms posterior_linpred"):
        gates.check_predictions(_corrupted(b, linpred))

    def epred(raw):
        raw["predictions"]["newdata"]["epred_nogroup"]["values"][3] += 1e-6
    with pytest.raises(OracleMismatch, match=r"newdata_nogroup: 'expected' differs from brms posterior_epred"):
        gates.check_predictions(_corrupted(b, epred))

    def effects(raw):
        raw["points"][2]["pars"]["r_3_mum3_1"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 2: the graph's values of 'mum3' differ from brms's linear predictor of mum3"):
        gates.check_derived(_corrupted(b, effects))

    def raw_intercept(raw):
        raw["points"][4]["pars"]["b_mum1_Intercept"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 4: the raw-scale intercept of 'mum1' differs from brms's b_mum1_Intercept"):
        gates.check_derived(_corrupted(b, raw_intercept))

    def group_effect(raw):
        raw["points"][5]["pars"]["r_2_mum1_1"]["values"][1] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 5: the graph's values of 'mum1' differ from brms's linear predictor of mum1"):
        gates.check_derived(_corrupted(b, group_effect))

    def default(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "sd" and r["dpar"] == "mum3" and r["group"] == ""]
        row["prior"] = "student_t(3, 0, 2.6)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(b, default))

    def no_rows(raw):
        raw["default_prior"] = [r for r in raw["default_prior"] if not (r["dpar"] == "mum3" and r["class"] == "sd")]
    with pytest.raises(OracleMismatch, match=r"brms default_prior lacks the categories' rows \[\('sd', 'mum3'\)\]"):
        gates.check_default_prior(_corrupted(b, no_rows))

    other = ab.multinomial("cbind(m2, m1, m3) | trials(size) ~ x + (1 | g) + (1 | h)", b.data,
                           ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}))
    with pytest.raises(OracleMismatch, match=r"mum1's design column|brms X_mum1 has 6 columns"):
        gates.Bundle("multinomial_terms", model=other)
    uncentred = copy_without_centring(b)
    with pytest.raises(OracleMismatch, match="brms does not centre the formula of mum1 as AutoBayes does"):
        _corrupted(b, uncentred)


def copy_without_centring(b):
    def mutate(raw):
        raw["standata"]["Kc_mum1"]["values"][0] = 6
    return mutate


def test_the_gates_see_a_corrupted_dirichlet_multinomial_fixture():
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    b = gates.Bundle("dirichlet_multinomial_varint")

    def gradient(raw):
        (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "phi"]
        raw["points"][2]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 2: gradient of phi_log__\[0\]"):
        gates.check_gradient(_corrupted(b, gradient))

    def phi(raw):
        raw["points"][0]["pars"]["phi"]["values"][0] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match="point 0: log density"):
        gates.check_logp(_corrupted(b, phi))
    multinomial = ab.multinomial(b.case["autobayes"]["formula"], b.data, pri())
    with pytest.raises(OracleMismatch, match="no adapter for brms parameter 'phi'"):
        gates.check_gradient(gates.Bundle("dirichlet_multinomial_varint", model=multinomial))
