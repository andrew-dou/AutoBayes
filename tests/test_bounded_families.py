"""Bounded and hurdle families (brms-parity E01, docs/bounded-families.md): ``zero_one_inflated_beta``,
``zero_inflated_beta_binomial``, ``xbeta`` and ``hurdle_cumulative``.

- the regularized incomplete Beta function ``likelihoods.shape_betainc`` and its gradient in all three arguments on both paths
  (PyTensor's, and JAX's custom derivative, the path NumPyro samples on) against a 50-digit reference (the hypergeometric series
  of ``B_x(a, b)``, Stirling's series for ``lgamma``, fourth-order central differences);
- every family's rows, value and gradient on both paths, against the same 50-digit arithmetic; the branches a switch does not
  take stay finite with a finite gradient at extreme linear predictors and precisions;
- the likelihoods' forward draws, each model's density and gradient against its rows and priors, priors and their refusals,
  ``value_point``, responses (brms's coding of the hurdle ordinal response, probed), predictions by hand (xbeta's censored mean
  against quadrature), ``weights()``, every other construct refused naming it or built, ``hypothesis()``,
  ``conditional_effects()``, ``describe()``, prior-predictive draws, default priors and the fit record. Nothing samples; the
  brms comparison is the oracle's four frozen cases (tests/oracle)."""
import json
import math
import warnings
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import integrate, stats
from scipy.special import betainc, expit, logit

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import get_family
from autobayes.families.xbeta import censored_mean
from tests.oracle import adapter, datasets, gates

NEW = ["zero_one_inflated_beta", "zero_inflated_beta_binomial", "xbeta", "hurdle_cumulative"]
SPECS = {                                   # family -> (left-hand side on DATA, its auxiliary priors)
    "zero_one_inflated_beta": ("zoprop", {"phi": ab.Gamma(2, 0.5), "zoi": ab.Beta(2, 2), "coi": ab.Beta(2, 2)}),
    "zero_inflated_beta_binomial": ("zk | trials(size)", {"phi": ab.Gamma(2, 0.5), "zi": ab.Beta(2, 2)}),
    "xbeta": ("zoprop", {"phi": ab.Gamma(2, 0.5), "kappa": ab.Gamma(2, 2)}),
    "hurdle_cumulative": ("ordh", {"hu": ab.Beta(2, 2)}),
}
DATA = datasets.tier_e(7, 40)               # zoprop in [0, 1] with exact 0s and 1s, ordh in 0..4, zk of size, positive weights w
CASES = {"zero_one_inflated_beta": "zero_one_inflated_beta_varint", "zero_inflated_beta_binomial": "zero_inflated_beta_binomial_varint",
         "xbeta": "xbeta_varint", "hurdle_cumulative": "hurdle_cumulative_varint"}


def priors(family, group=True, **override):
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **({"sd": ab.HalfNormal(1)} if group else {}))
    return ab.Priors(**{k: v for k, v in {**base, **SPECS[family][1], **override}.items() if v is not None})


def build(family, rhs="x + (1 | g)", frame=None, lhs=None, **override):
    return getattr(ab, family)(f"{lhs or SPECS[family][0]} ~ {rhs}", DATA if frame is None else frame,
                               priors(family, "|" in rhs, **override))


# -- 0. 50-digit arithmetic: lgamma, the incomplete Beta function, central differences ---------------------------------------
_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798")


def _bernoulli_even(count):
    """B_2, B_4, ..., B_{2 count} as fractions (Akiyama-Tanigawa)."""
    n = 2 * count
    a, out = [Fraction(0)] * (n + 1), []
    for m in range(n + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        out.append(a[0])
    return out[2::2]


_B2K = _bernoulli_even(30)


def _lgamma(z):
    """log Gamma(z) for Decimal z > 0: shifted to z >= 40, then Stirling's series through B_60 (error below 1e-60), or through
    B_40 at a precision of 45 digits or less (error below 1e-46)."""
    shift = Decimal(0)
    while z < 40:
        shift += z.ln()
        z += 1
    total = (z - Decimal("0.5")) * z.ln() - z + (2 * _PI).ln() / 2
    power, z2 = z, z * z
    for k, b in enumerate(_B2K if getcontext().prec > 45 else _B2K[:20], start=1):
        total += Decimal(b.numerator) / Decimal(b.denominator) / (2 * k * (2 * k - 1) * power)
        power *= z2
    return total - shift


def _log_beta(a, b):
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


def _log_betainc(a, b, x):
    """log I_x(a, b) (Decimal): log(x^a (1 - x)^b / (a B(a, b)) 2F1(a + b, 1; a + 1; x)) for x <= 1/2, a series of positive
    terms whose ratio tends to x; through 1 - I_{1-x}(b, a) above."""
    if x > Decimal("0.5"):
        return (1 - _log_betainc(b, a, 1 - x).exp()).ln()
    tiny = Decimal(10) ** (-(getcontext().prec + 5))
    term = total = Decimal(1)
    n = 0
    while True:
        ratio = (a + b + n) * x / (a + 1 + n)
        term *= ratio
        total += term
        n += 1
        if ratio < 1 and term < tiny * total:
            break
    return a * x.ln() + b * (1 - x).ln() - a.ln() + total.ln() - _log_beta(a, b)


def _derivatives(f, args, h):
    """f(*args) and its fourth-order central difference in each argument (Decimal; error about h^4 plus the rounding error over
    h)."""
    out = [f(*args)]
    for k in range(len(args)):
        def g(v, k=k):
            moved = list(args)
            moved[k] = v
            return f(*moved)
        x = args[k]
        out.append((g(x - 2 * h) - 8 * g(x - h) + 8 * g(x + h) - g(x + 2 * h)) / (12 * h))
    return out


def exact(f, *args):
    """``f`` (a function of Decimals) and its derivatives at the float ``args``, at 40 digits with steps of 1e-10 (truncation
    1e-40, rounding 1e-30, far below the tolerances of 1e-12), as floats."""
    with localcontext() as ctx:
        ctx.prec = 40
        return [float(v) for v in _derivatives(f, [Decimal(float(a)) for a in args], Decimal("1e-10"))]


_REFERENCE = {}


def reference(name, grid, row):
    """``exact(row(...), ...)`` at every grid point, computed once per grid for both paths."""
    if name not in _REFERENCE:
        _REFERENCE[name] = [row(*point) for point in grid]
    return _REFERENCE[name]


def _expit(e):
    return 1 / (1 + (-e).exp())


def _paths(fn, mode, ints, floats):
    """``fn(*ints, *floats)`` per row and its gradient in each float input, on one path: PyTensor's gradient graph, or JAX's
    autodiff of the jaxified graph (the path NumPyro samples on). ``ints`` are data (integer or real arrays), no gradient."""
    iv = [pt.vector(f"i{k}", dtype=np.asarray(a).dtype) for k, a in enumerate(ints)]
    fv = [pt.dvector(f"f{k}") for k in range(len(floats))]
    rows = fn(*iv, *fv)
    if mode == "pytensor":
        f = pytensor.function([*iv, *fv], [rows, *(pt.grad(rows.sum(), v) for v in fv)])
        return [np.asarray(a, dtype=float) for a in f(*ints, *floats)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=[*iv, *fv], outputs=[rows])
    out = [np.asarray(jf(*ints, *floats)[0], dtype=float)]
    for k in range(len(floats)):
        def total(v, k=k):
            args = list(floats)
            args[k] = v
            return jf(*ints, *args)[0].sum()
        out.append(np.asarray(jax.grad(total)(floats[k]), dtype=float))
    return out


def _close(got, ref, tol):
    return np.isfinite(got) and abs(got - ref) <= tol * max(1.0, abs(ref))


def test_the_reference_is_the_incomplete_beta_function():
    """The 50-digit reference agrees with SciPy's betainc and its own symmetry, and lgamma with known values."""
    with localcontext() as ctx:
        ctx.prec = 60
        assert abs(float(_lgamma(Decimal("0.5"))) - 0.5 * math.log(math.pi)) < 1e-15
        assert abs(float(_lgamma(Decimal(7))) - math.log(720.0)) < 1e-15
        for a, b, x in ((0.3, 2.0, 0.2), (12.0, 3.5, 0.45), (2.7, 80.0, 0.01), (5.0, 0.7, 0.8)):
            ref = float(_log_betainc(Decimal(a), Decimal(b), Decimal(x)))
            assert abs(ref - math.log(betainc(a, b, x))) <= 1e-13 * max(1.0, abs(ref))
            flipped = float((1 - _log_betainc(Decimal(b), Decimal(a), 1 - Decimal(x)).exp()).ln())
            assert abs(ref - flipped) <= 1e-40 * max(1.0, abs(ref))


# -- 1. the incomplete Beta function and its gradient on both paths -----------------------------------------------------------
BETAINC_GRID = [(a, b, x) for a in (0.3, 1.0, 2.7, 12.0, 80.0) for b in (0.4, 1.0, 3.3, 25.0) for x in (0.01, 0.2, 0.45, 0.8)]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_shape_betainc_and_its_gradient_are_exact_on_both_paths(mode):
    """log I_x(a, b) and its gradient in a, b and x. PyTensor's gradient in the shapes (its Boik and Robison-Cox continued
    fraction, at most 200 steps to 1e-12) is within 1e-8 relative of the reference here; JAX's custom derivative (the same
    continued fraction, rescaled, to 1e-14, at most 1000 steps; B(a, b) from gammaln, since jax.scipy.special.betaln is off by
    7e-8 relative at B(10, 3.3)) within 1e-11."""
    a, b, x = (np.array([r[k] for r in BETAINC_GRID]) for k in range(3))
    got = _paths(lambda av, bv, xv: pt.log(lk.shape_betainc(av, bv, xv)), mode, [], [a, b, x])
    table = reference("betainc", BETAINC_GRID, lambda *point: exact(_log_betainc, *point))
    for i, row in enumerate(BETAINC_GRID):
        want = table[i]
        tolerances = (1e-12, *([2e-8] * 3 if mode == "pytensor" else [1e-11] * 3))
        for k, (ref, tol) in enumerate(zip(want, tolerances)):
            assert _close(got[k][i], ref, tol), (mode, row, k, got[k][i], ref)


def test_shape_betainc_is_its_own_op_and_matches_pytensors():
    av, bv, xv = pt.dscalar("a"), pt.dscalar("b"), pt.dscalar("x")
    ours, theirs = lk.shape_betainc(av, bv, xv), pt.betainc(av, bv, xv)
    assert type(ours.owner.op.scalar_op) is not type(theirs.owner.op.scalar_op)
    f = pytensor.function([av, bv, xv], [ours, theirs, *pt.grad(ours, [av, bv, xv]), *pt.grad(theirs, [av, bv, xv])])
    values = f(2.5, 4.0, 0.3)
    assert all(np.array_equal(u, v) for u, v in zip(values[0:1] + values[2:5], values[1:2] + values[5:8]))


def test_the_jax_derivative_is_nan_where_it_does_not_converge_and_vmaps():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    grad = jax.grad(lambda a, b, x: jnp.log(lk.jax_shape_betainc(a, b, x)).sum(), argnums=(0, 1, 2))
    batched = jax.vmap(lambda a: jax.grad(lambda v: jnp.log(lk.jax_shape_betainc(v, 2.0, 0.3)))(a))(jnp.array([0.5, 1.0, 3.0]))
    for k, a in enumerate((0.5, 1.0, 3.0)):
        assert _close(float(batched[k]), exact(_log_betainc, a, 2.0, 0.3)[1], 1e-11)
    da, db, dx = grad(jnp.array([3.0, 2000.0]), jnp.array([2.0, 1.0]), jnp.array([0.3, 1e-6]))
    assert np.isfinite(np.asarray(da)[0]) and np.isnan(np.asarray(da)[1])      # I underflows to 0: log I and its gradient are lost


# -- 2. the rows of each family on both paths -----------------------------------------------------------------------------------
def _zoib_row(y, e, phi, zoi, coi):
    if y == 0:
        return zoi.ln() + (1 - coi).ln()
    if y == 1:
        return zoi.ln() + coi.ln()
    m = _expit(e)
    a, b = m * phi, (1 - m) * phi
    return (1 - zoi).ln() - _log_beta(a, b) + (a - 1) * y.ln() + (b - 1) * (1 - y).ln()


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_zero_one_inflated_beta_rows_on_both_paths(mode):
    """Value and gradient in eta, phi, zoi and coi; the rows at 0 and 1 have a zero gradient in eta and phi at every eta."""
    grid = [(y, e, s, z, c) for y in (0.0, 1.0, 0.03, 0.5, 0.98) for e in (-30.0, -3.0, 0.0, 0.7, 4.0, 30.0) for s in (0.4, 3.2, 150.0)
            for z, c in ((0.2, 0.7), (0.9, 0.05))]
    y, eta, phi, zoi, coi = (np.array([r[k] for r in grid]) for k in range(5))
    got = _paths(lambda yv, ev, sv, zv, cv: lk.zero_one_inflated_beta_logp(yv, ev, sv, zv, cv), mode, [y], [eta, phi, zoi, coi])
    table = reference("zoib", grid, lambda y, *rest: exact(lambda e, s, z, c: _zoib_row(Decimal(y), e, s, z, c), *rest))
    for i, row in enumerate(grid):
        want = table[i]
        for k, ref in enumerate(want):
            assert _close(got[k][i], ref, 1e-12 if mode == "jax" or k not in (1, 2) else 2e-9), (mode, row, k, got[k][i], ref)
    edge = np.array([-1000.0, 1000.0])
    v, de, ds, _, _ = _paths(lambda yv, ev, sv, zv, cv: lk.zero_one_inflated_beta_logp(yv, ev, sv, zv, cv), mode, [np.array([0.0, 1.0])],
                             [edge, np.full(2, 1e6), np.full(2, 0.3), np.full(2, 0.6)])
    np.testing.assert_allclose(v, [math.log(0.3) + math.log(0.4), math.log(0.3) + math.log(0.6)], rtol=1e-15)
    assert np.array_equal(de, [0.0, 0.0]) and np.array_equal(ds, [0.0, 0.0])


def _zibb_row(y, n, e, phi, zi):
    m = _expit(e)
    a, b = m * phi, (1 - m) * phi
    coef = _lgamma(Decimal(n + 1)) - _lgamma(Decimal(y + 1)) - _lgamma(Decimal(n - y + 1))
    log_bb = coef + _log_beta(y + a, n - y + b) - _log_beta(a, b)
    if y == 0:
        return (zi + (1 - zi) * log_bb.exp()).ln()
    return (1 - zi).ln() + log_bb


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_zero_inflated_beta_binomial_rows_on_both_paths(mode):
    """Value and gradient in eta, phi and zi; PyMC's BetaBinomial, as the beta-binomial family's (D02; on the PyTensor path its
    gradient goes through PyTensor's digamma, 1.5e-10 off relative between 1 and 12)."""
    grid = [(y, n, e, s, z) for n in (1, 7, 40) for y in sorted({0, 1, n // 2, n}) for e in (-6.0, 0.0, 3.0)
            for s in (0.3, 4.0, 90.0) for z in (1e-4, 0.97)]
    y, n, eta, phi, zi = (np.array([r[k] for r in grid], dtype=np.int64 if k < 2 else float) for k in range(5))
    got = _paths(lambda yv, nv, ev, sv, zv: lk.zero_inflated_beta_binomial_logp(yv, ev, sv, zv, nv), mode, [y, n], [eta, phi, zi])
    table = reference("zibb", grid, lambda y, n, *rest: exact(lambda e, s, z: _zibb_row(y, n, e, s, z), *rest))
    for i, row in enumerate(grid):
        want = table[i]
        for k, ref in enumerate(want):
            assert _close(got[k][i], ref, 1e-12 if mode == "jax" else 3e-9), (mode, row, k, got[k][i], ref)


def _xbeta_row(y, e, phi, kappa):
    m = _expit(e)
    a, b = m * phi, (1 - m) * phi
    c = kappa / (1 + 2 * kappa)
    if y == 0:
        return _log_betainc(a, b, c)
    if y == 1:
        return _log_betainc(b, a, c)
    u = (y + kappa) / (1 + 2 * kappa)
    return -_log_beta(a, b) + (a - 1) * u.ln() + (b - 1) * (1 - u).ln() - (1 + 2 * kappa).ln()


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_xbeta_rows_on_both_paths(mode):
    """Value and gradient in eta, phi and kappa against the 50-digit row (log I_c(a, b) at 0, log I_c(b, a) at 1, the stretched
    Beta density inside); the masses' shape derivatives are PyTensor's (to 2e-8 relative) and the custom JAX ones (to 1e-11)."""
    grid = [(y, e, s, k) for y in (0.0, 1.0, 0.3, 0.97) for e in (-3.0, 0.0, 0.7, 3.0) for s in (0.5, 3.0, 40.0)
            for k in (0.05, 0.6, 3.0)]
    y, eta, phi, kappa = (np.array([r[k] for r in grid]) for k in range(4))
    got = _paths(lambda yv, ev, sv, kv: lk.xbeta_logp(yv, ev, sv, kv), mode, [y], [eta, phi, kappa])
    table = reference("xbeta", grid, lambda y, *rest: exact(lambda e, s, k: _xbeta_row(Decimal(y), e, s, k), *rest))
    for i, row in enumerate(grid):
        want = table[i]
        for k, ref in enumerate(want):
            tol = 1e-12 if k == 0 else (1e-11 if mode == "jax" else 3e-8)
            assert _close(got[k][i], ref, tol), (mode, row, k, got[k][i], ref)


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_untaken_branches_stay_finite(mode):
    """A row inside (0, 1) at a precision and mean where I_c(a, b) underflows to 0 (its zero branch would be -inf with an
    infinite gradient) and a zero row of the zero-one-inflated Beta at |eta| = 1000 keep a finite value and gradient: every
    branch a switch does not take is evaluated at fixed interior arguments."""
    y = np.array([0.5, 0.5, 0.2, 0.9])
    eta, phi, kappa = np.array([8.0, -8.0, 0.0, 30.0]), np.array([5e3, 5e3, 3e4, 40.0]), np.array([0.3, 0.3, 0.3, 2.0])
    for values in (_paths(lambda yv, ev, sv, kv: lk.xbeta_logp(yv, ev, sv, kv), mode, [y], [eta, phi, kappa]),
                   _paths(lambda yv, ev, sv, zv, cv: lk.zero_one_inflated_beta_logp(yv, ev, sv, zv, cv), mode, [np.array([0.0, 1.0, 0.5])],
                          [np.array([-1000.0, 1000.0, 2.0]), np.full(3, 2.0), np.full(3, 0.4), np.full(3, 0.5)])):
        assert all(np.isfinite(v).all() for v in values), values


def _hurdle_row(y, e, t1, t2, t3, hu):
    if y == 0:
        return hu.ln()
    tau = [None, t1, t2, t3, None]
    upper = 1 if y == 4 else _expit(tau[y] - e)
    lower = 0 if y == 1 else _expit(tau[y - 1] - e)
    return (1 - hu).ln() + (upper - lower).ln()


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_hurdle_cumulative_rows_on_both_paths(mode):
    """Value and gradient in eta, the three thresholds and hu."""
    grid = [(y, e, *tau, h) for y in (0, 1, 2, 3, 4) for e in (-30.0, -2.0, 0.0, 0.8, 4.0, 30.0)
            for tau in ((-1.0, 0.3, 2.0), (-0.2, -0.19, 5.0)) for h in (1e-3, 0.4)]
    y = np.array([r[0] for r in grid], dtype=np.int64)
    eta, t1, t2, t3, hu = (np.array([r[k] for r in grid]) for k in range(1, 6))

    def rows(yv, ev, a, b, c, hv):
        return lk.hurdle_ordered_logistic_logp(yv, ev, pt.stack([a, b, c], axis=-1), hv)

    got = _paths(rows, mode, [y], [eta, t1, t2, t3, hu])
    table = reference("hurdle", grid, lambda y, *rest: exact(lambda *args: _hurdle_row(y, *args), *rest))
    for i, row in enumerate(grid):
        want = table[i]
        for k, ref in enumerate(want):
            assert _close(got[k][i], ref, 1e-10), (mode, row, k, got[k][i], ref)
    v = pytensor.function([], lk.hurdle_ordered_logistic_logp(pt.as_tensor(np.array([-1, 5])), pt.as_tensor(np.zeros(2)),
                                                              pt.as_tensor(np.array([-1.0, 0.0, 1.0])), np.float64(0.3)))()
    assert np.array_equal(v[:1], [-np.inf]) and np.isneginf(v[1])               # outside 0..K: no mass


def test_the_rows_outside_the_support_have_no_mass():
    zoib = pytensor.function([], lk.zero_one_inflated_beta_logp(pt.as_tensor(np.array([-0.1, 1.2])), np.float64(0.0), np.float64(2.0),
                                                                np.float64(0.3), np.float64(0.5)))()
    xb = pytensor.function([], lk.xbeta_logp(pt.as_tensor(np.array([-0.1, 1.2])), np.float64(0.0), np.float64(2.0), np.float64(0.5)))()
    zibb = pytensor.function([], lk.zero_inflated_beta_binomial_logp(pt.as_tensor(np.array([4, -1])), np.float64(0.0), np.float64(2.0),
                                                                     np.float64(0.3), pt.as_tensor(np.array([3, 3]))))()
    assert np.isneginf(zoib).all() and np.isneginf(xb).all() and np.isneginf(zibb).all()


# -- 3. the likelihoods' forward draws ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", NEW)
def test_likelihood_objects_draw_their_distribution(family):
    import pymc as pm

    eta = np.array([-1.0, 0.4, 1.5])
    phi, kappa, n, p, q = 3.0, 0.4, np.array([3, 5, 8]), 0.3, 0.6
    a, b = expit(eta) * phi, expit(-eta) * phi
    S = 40000
    if family == "hurdle_cumulative":
        tau = np.array([-0.5, 0.4, 1.6])
        draws = pm.draw(pm.CustomDist.dist(eta, tau, np.float64(p), logp=lk.hurdle_ordered_logistic_logp,
                                           random=lk.hurdle_ordered_logistic_random, signature="(),(k),()->()", dtype="int64"),
                        draws=S, random_seed=3)
        probs = lk.hurdle_cumulative_probabilities(eta, tau, p)
        assert draws.shape == (S, 3) and set(np.unique(draws)) == {0, 1, 2, 3, 4}
        for k in range(5):
            freq = (draws == k).mean(0)
            assert np.all(np.abs(freq - probs[:, k]) < 4.5 * np.sqrt(probs[:, k] * (1 - probs[:, k]) / S)), (k, freq, probs[:, k])
        return
    dist = {"zero_one_inflated_beta": lambda: lk.zero_one_inflated_beta_family().likelihood.dist.dist(logit_mu=eta, phi=phi, zoi=p, coi=q),
            "zero_inflated_beta_binomial": lambda: lk.zero_inflated_beta_binomial_family().likelihood.dist.dist(logit_mu=eta, phi=phi,
                                                                                                                zi=p, n=n),
            "xbeta": lambda: lk.xbeta_family().likelihood.dist.dist(logit_mu=eta, phi=phi, kappa=kappa)}[family]()
    draws = pm.draw(dist, draws=S, random_seed=3)
    assert draws.shape == (S, 3) and draws.min() >= 0
    if family == "zero_one_inflated_beta":
        masses = {0.0: np.full(3, p * (1 - q)), 1.0: np.full(3, p * q)}
        mean = p * q + (1 - p) * expit(eta)
    elif family == "zero_inflated_beta_binomial":
        masses = {0: p + (1 - p) * np.exp(stats.betabinom.logpmf(0, n, a, b))}
        mean = (1 - p) * n * expit(eta)
    else:
        c = kappa / (1 + 2 * kappa)
        masses = {0.0: betainc(a, b, c), 1.0: betainc(b, a, c)}
        mean = censored_mean(eta, phi, kappa)
    for value, mass in masses.items():
        assert np.all(np.abs((draws == value).mean(0) - mass) < 4.5 * np.sqrt(mass * (1 - mass) / S)), (value, (draws == value).mean(0), mass)
    assert np.all(np.abs(draws.mean(0) - mean) < 5 * draws.std(0) / math.sqrt(S)), (draws.mean(0), mean)


# -- 4. each model's density and gradient: its rows and its priors -----------------------------------------------------------
FRAME = pd.DataFrame({"x": [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7], "g": ["a", "a", "b", "b", "c", "c", "a", "b"],
                      "zoprop": [0.0, 0.3, 1.0, 0.0, 0.55, 1.0, 0.2, 0.9], "zk": [0, 2, 3, 0, 1, 0, 4, 2],
                      "size": [3, 4, 5, 2, 3, 6, 4, 2], "ordh": [0, 2, 4, 0, 1, 3, 2, 1],
                      "w": [1.5, 0.6, 2.0, 1.0, 0.8, 1.2, 0.7, 1.9]})
NAT = {"zero_one_inflated_beta": {"phi": 3.2, "zoi": 0.35, "coi": 0.6}, "zero_inflated_beta_binomial": {"phi": 3.2, "zi": 0.35},
       "xbeta": {"phi": 3.2, "kappa": 0.45}, "hurdle_cumulative": {"hu": 0.35}}


def _rowwise(model):
    pmm = model.pymc_model
    return pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False)[0], inputs=pmm.value_vars, on_unused_input="ignore")


def _point(model, family):
    nat = dict(b=[-0.4], sd=0.5, z=np.array([0.2, -0.4, 0.1]), **NAT[family])
    nat["intercept"] = [-0.6, 0.4, 1.5] if family == "hurdle_cumulative" else 0.3
    return model.value_point(**nat)


def _by_hand_rows(family, eta):
    y, n = FRAME[SPECS[family][0].split()[0]].to_numpy(), FRAME["size"].to_numpy()
    aux = NAT[family]
    if family == "zero_one_inflated_beta":
        inside = np.where((y > 0) & (y < 1), y, 0.5)
        beta = np.log1p(-aux["zoi"]) + stats.beta.logpdf(inside, expit(eta) * aux["phi"], expit(-eta) * aux["phi"])
        return np.where(y == 0, math.log(0.35 * 0.4), np.where(y == 1, math.log(0.35 * 0.6), beta))
    if family == "zero_inflated_beta_binomial":
        bb = stats.betabinom.logpmf(y, n, expit(eta) * aux["phi"], expit(-eta) * aux["phi"])
        return np.where(y == 0, np.log(0.35 + 0.65 * np.exp(bb)), np.log(0.65) + bb)
    if family == "xbeta":
        a, b, k = expit(eta) * aux["phi"], expit(-eta) * aux["phi"], aux["kappa"]
        c = k / (1 + 2 * k)
        inside = stats.beta.logpdf((np.where((y > 0) & (y < 1), y, 0.5) + k) / (1 + 2 * k), a, b) - np.log1p(2 * k)
        return np.where(y == 0, np.log(betainc(a, b, c)), np.where(y == 1, np.log(betainc(b, a, c)), inside))
    tau = np.array([-0.6, 0.4, 1.5]) - 0.4 * FRAME.x.mean()           # b_Intercept: the centred thresholds plus mean(x) b
    cdf = np.column_stack([np.zeros(8), expit(tau[None, :] - eta[:, None]), np.ones(8)])
    p = np.clip(y, 1, 4)
    return np.where(y == 0, math.log(0.35), math.log(0.65) + np.log(cdf[np.arange(8), p] - cdf[np.arange(8), p - 1]))


@pytest.mark.parametrize("family", NEW)
def test_the_model_density_is_its_rows_and_its_priors(family):
    """The PyMC model's row densities equal the family's by hand (SciPy) at a value point, and its density without the Jacobian
    is their sum plus the priors' and the standardised effects' log densities."""
    m = build(family, frame=FRAME)
    point = _point(m, family)
    u = np.array([0.2, -0.4, 0.1])[pd.Categorical(FRAME.g).codes] * 0.5
    x = FRAME.x.to_numpy()
    eta = (-0.4 * x if family == "hurdle_cumulative" else 0.3 - 0.4 * (x - x.mean())) + u
    rows = np.asarray(_rowwise(m)(point), dtype=float)
    np.testing.assert_allclose(rows, _by_hand_rows(family, eta), rtol=1e-11)
    lp = float(m.pymc_model.compile_logp(jacobian=False)(point))
    pri = stats.halfnorm.logpdf(0.5) + stats.norm.logpdf(-0.4) + stats.norm.logpdf([0.2, -0.4, 0.1]).sum()
    pri += stats.norm.logpdf([-0.6, 0.4, 1.5], 0, 2).sum() if family == "hurdle_cumulative" else stats.norm.logpdf(0.3, 0, 2)
    for name, value in NAT[family].items():
        prior = SPECS[family][1][name]
        pri += (stats.beta.logpdf(value, 2, 2) if prior.name == "Beta" else
                stats.gamma.logpdf(value, prior.params["alpha"], scale=1 / prior.params["beta"]))
    np.testing.assert_allclose(lp, rows.sum() + pri, rtol=1e-12)


# -- 5. priors, value_point, responses -------------------------------------------------------------------------------------
def test_priors_are_required_and_refused_by_name():
    with pytest.raises(ab.PriorError, match="the zero_one_inflated_beta family needs a prior on its zero-one-inflation probability zoi: "
                                            r"pass zoi=\.\.\., such as ab.Beta\(1, 1\) \(brms's default is beta\(1, 1\)\)"):
        build("zero_one_inflated_beta", "x", zoi=None)
    with pytest.raises(ab.PriorError, match="needs a prior on its conditional one-inflation probability coi"):
        build("zero_one_inflated_beta", "x", coi=None)
    with pytest.raises(ab.PriorError, match=r"the xbeta family needs a prior on its exceedance .* kappa: pass kappa=\.\.\., such as "
                                            r"ab.Gamma\(1, 0.1\) \(brms's default is gamma\(0.01, 0.01\)\)"):
        build("xbeta", "x", kappa=None)
    with pytest.raises(ab.PriorError, match="the zero_inflated_beta_binomial family needs a prior on the precision parameter phi"):
        build("zero_inflated_beta_binomial", "x", phi=None)
    with pytest.raises(ab.PriorError, match="the hurdle_cumulative family needs a prior on the hurdle probability hu"):
        build("hurdle_cumulative", "x", hu=None)
    with pytest.raises(ab.PriorError, match=r"the xbeta family has no zero-one-inflation probability; do not pass zoi. Its auxiliary "
                                            r"parameter is phi: pass phi=\.\.\. Its exceedance .* is kappa"):
        build("xbeta", "x", zoi=ab.Beta(1, 1))
    with pytest.raises(ab.PriorError, match="the zero_one_inflated_beta family has no kappa parameter; do not pass kappa"):
        build("zero_one_inflated_beta", "x", kappa=ab.Gamma(1, 1))
    with pytest.raises(ab.PriorError, match="the gaussian family has no conditional one-inflation probability; do not pass coi"):
        ab.gaussian("y ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), coi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'zoi' must be a prior on \(0, 1\) \(brms's real<lower=0,upper=1> zoi\), one of "
                                            r"\['Beta'\]; got HalfNormal"):
        build("zero_one_inflated_beta", "x", zoi=ab.HalfNormal(1))
    with pytest.raises(ab.PriorError, match=r"prior for 'kappa' must be a positive-support prior"):
        build("xbeta", "x", kappa=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match="the hurdle_cumulative family has no zero-inflation probability; do not pass zi"):
        build("hurdle_cumulative", "x", zi=ab.Beta(1, 1))
    with pytest.raises(ab.PriorError, match="the hurdle_cumulative family has no precision parameter; do not pass phi"):
        build("hurdle_cumulative", "x", phi=ab.Gamma(1, 1))
    with pytest.raises(ab.PriorError, match="a mixture's priors are per component.*'zoi'"):
        ab.mixture("y ~ x", DATA, ab.Priors(components=[ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))] * 2,
                                            theta=ab.Dirichlet(2), zoi=ab.Beta(1, 1)), families=["gaussian", "gaussian"])


@pytest.mark.parametrize("family", NEW)
def test_names_and_value_point(family):
    fam = get_family(family)
    m = build(family)
    names = {v.name for v in m.pymc_model.value_vars}
    for name in NAT[family]:
        assert m.names[name] == name
    assert m.names["linpred"] == fam.parent and m.names["mean"] == fam.mean_name
    point = _point(build(family, frame=FRAME), family)
    for name, value in NAT[family].items():
        coordinate = f"{name}_log__" if name in ("phi", "kappa") else f"{name}_logodds__"
        assert coordinate in names and coordinate in point
        np.testing.assert_allclose(point[coordinate], math.log(value) if name in ("phi", "kappa") else logit(value), rtol=1e-15)
    small = build(family, frame=FRAME)
    base = dict(b=[-0.4], sd=0.5, z=np.zeros(3), intercept=[-0.6, 0.4, 1.5] if family == "hurdle_cumulative" else 0.3)
    for name in NAT[family]:
        rest = {k: v for k, v in NAT[family].items() if k != name}
        with pytest.raises(ValueError, match=f"{name}"):
            small.value_point(**base, **rest)
    for name in ("zoi", "coi"):
        if name in NAT[family]:
            for bad in (0.0, 1.0, np.nan):
                with pytest.raises(ValueError, match=f"{name} .* must (lie strictly between 0 and 1|be a finite scalar)"):
                    small.value_point(**base, **{**NAT[family], name: bad})
        else:
            with pytest.raises(ValueError, match=f"the {family} family has no"):
                small.value_point(**base, **NAT[family], **{name: 0.5})
    if family == "xbeta":
        with pytest.raises(ValueError, match="kappa .* must be positive"):
            small.value_point(**base, **{**NAT[family], "kappa": 0.0})


@pytest.mark.parametrize("family, values, message", [
    ("zero_one_inflated_beta", [0.0, 1.2], r"value outside \[0, 1\] \(1.2\); the zero-one-inflated Beta family needs proportions in \[0, 1\]"),
    ("zero_one_inflated_beta", [-0.1, 0.5], r"outside \[0, 1\]"), ("xbeta", [np.inf, 0.5], "non-finite"),
    ("xbeta", [1.0001, 0.2], r"the xbeta family needs proportions in \[0, 1\] \(exact 0s and 1s included\)"),
    ("xbeta", [True, False], "numeric response"),
    ("hurdle_cumulative", [-1, 2, 3], "has the value -1; the hurdle_cumulative family needs whole numbers 0..K"),
    ("hurdle_cumulative", [0, 1.5, 3], "has the value 1.5"), ("hurdle_cumulative", ["a", "b", "c"], "Text labels have no order"),
    ("hurdle_cumulative", [0, 1, 1], r"has 1 category above the hurdle category '0'.*Could not extract the number of thresholds"),
    ("hurdle_cumulative", [True, False, True], "is boolean"),
])
def test_invalid_responses(family, values, message):
    with pytest.raises(ab.FormulaError, match=message):
        getattr(ab, family)(f"{SPECS[family][0]} ~ 1", pd.DataFrame({SPECS[family][0]: values}), priors(family, False, b=None))


def test_zero_inflated_beta_now_points_to_the_families_that_take_exact_ones():
    with pytest.raises(ab.FormulaError, match=r"an exact 1 is refused \(brms's zero_one_inflated_beta and xbeta model exact 1s: "
                                              r"ab.zero_one_inflated_beta\(\), ab.xbeta\(\)\)"):
        ab.zero_inflated_beta("zp ~ 1", pd.DataFrame({"zp": [0.0, 0.5, 1.0]}), ab.Priors(intercept=ab.Normal(0, 1), phi=ab.Gamma(1, 1),
                                                                                         zi=ab.Beta(1, 1)))


def test_the_hurdle_ordinal_response_is_coded_as_brms_codes_it():
    """brms 2.23's standata() (probed): whole numbers keep their values, 0 the hurdle, and the ordinal part's categories are
    1..max(y), observed or not; an ordered factor's first level is the hurdle, coded 0, its others 1..K-1 in order; an unordered
    factor and negative or fractional values are refused, and so is a response without two categories above the hurdle."""
    pri = priors("hurdle_cumulative", False, b=None)
    m = ab.hurdle_cumulative("y ~ 1", pd.DataFrame({"y": [0, 2, 2, 3, 4, 0, 3]}), pri)
    assert m.categories == ("1", "2", "3", "4") and m.hurdle_category == "0"
    assert np.array_equal(m.design.y, [0, 2, 2, 3, 4, 0, 3]) and list(m.pymc_model.coords[m.names["threshold_dim"]]) == ["1", "2", "3"]
    assert any("categories ['1']" in w for w in m.warnings)
    ordered = pd.Categorical(["none", "low", "mid", "high", "low", "none"], categories=["none", "low", "mid", "high"], ordered=True)
    m = ab.hurdle_cumulative("y ~ 1", pd.DataFrame({"y": ordered}), pri)
    assert m.categories == ("low", "mid", "high") and m.hurdle_category == "none"
    assert np.array_equal(m.design.y, [0, 1, 2, 3, 1, 0])
    assert "0=none (the hurdle), 1=low, 2=mid, 3=high (ordered); fitted as brms's Y" in m.describe()
    with pytest.raises(ab.FormulaError, match="unordered pandas Categorical"):
        ab.hurdle_cumulative("y ~ 1", pd.DataFrame({"y": pd.Categorical(["a", "b", "c"])}), pri)
    no_hurdle = ab.hurdle_cumulative("y ~ 1", pd.DataFrame({"y": [1, 2, 3, 1]}), pri)
    assert no_hurdle.categories == ("1", "2", "3") and not (no_hurdle.design.y == 0).any()
    only = ab.hurdle_cumulative("y ~ 1", pd.DataFrame({"y": [0, 0, 0, 0, 2]}), pri)
    assert only.categories == ("1", "2") and any("categories ['1']" in w for w in only.warnings)
    allzero = ab.hurdle_cumulative("y ~ 1", pd.DataFrame({"y": pd.Categorical([0, 0, 0], categories=[0, 1, 2], ordered=True)}), pri)
    assert any("the ordinal part has no data" in w for w in allzero.warnings)


def test_zero_inflated_beta_binomial_trials_and_warnings():
    pri = priors("zero_inflated_beta_binomial", False, b=None)
    with pytest.raises(ab.FormulaError, match="exceed"):
        ab.zero_inflated_beta_binomial("zk | trials(size) ~ 1", pd.DataFrame({"zk": [3, 1], "size": [2, 2]}), pri)
    m = ab.zero_inflated_beta_binomial("zk | trials(size) ~ 1", pd.DataFrame({"zk": [0, 1, 0], "size": [0, 2, 3]}), pri)
    assert m.design.n == 2 and m.dropped_rows == (0,)                # the zero-trial row is dropped (D4), as for the binomial
    m = ab.zero_inflated_beta_binomial("zk | trials(size) ~ 1", pd.DataFrame({"zk": [0, 0, 0], "size": [1, 2, 3]}), pri)
    assert len(m.warnings) == 1 and "zero successes" in m.warnings[0]
    m = ab.zero_one_inflated_beta("zoprop ~ 1", pd.DataFrame({"zoprop": [0.0, 1.0, 1.0]}), priors("zero_one_inflated_beta", False, b=None))
    assert len(m.warnings) == 1 and "the Beta part has no data" in m.warnings[0]


# -- 6. predictions ------------------------------------------------------------------------------------------------------------
def draws(family, a=0.3, group=True):
    fam = get_family(family)
    post = {"x": np.full((2, 4), -0.2)}
    for name, value in NAT[family].items():
        post[name] = np.full((2, 4), value)
    if family == "hurdle_cumulative":
        post["b_Intercept"] = np.broadcast_to([-0.5, 0.4, 1.6], (2, 4, 3))
        post["Intercept"] = np.broadcast_to([-0.5, 0.4, 1.6], (2, 4, 3))
    else:
        post["Intercept"] = np.full((2, 4), a)
    if group:
        post.update({"1|g_sigma": np.full((2, 4), 0.5), "1|g": np.broadcast_to([0.1, -0.1, 0.3], (2, 4, 3))})
    dims = {"1|g": ["g__factor_dim"]}
    coords = {"g__factor_dim": ["a", "b", "c"]}
    if fam.thresholds:
        dims.update({"b_Intercept": ["Intercept_dim"], "Intercept": ["Intercept_dim"]})
        coords["Intercept_dim"] = ["1", "2", "3"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def _expected(family, eta, trials):
    aux = NAT[family]
    if family == "zero_one_inflated_beta":
        return aux["zoi"] * aux["coi"] + (1 - aux["zoi"]) * expit(eta)
    if family == "zero_inflated_beta_binomial":
        return (1 - aux["zi"]) * trials * expit(eta)
    if family == "xbeta":
        out = np.empty(eta.shape)
        for idx in np.ndindex(eta.shape):                            # the censored mean by quadrature: atoms plus the inside
            a, b, k = expit(eta[idx]) * aux["phi"], expit(-eta[idx]) * aux["phi"], aux["kappa"]
            c = k / (1 + 2 * k)
            inside = integrate.quad(lambda y: y * stats.beta.pdf((y + k) / (1 + 2 * k), a, b) / (1 + 2 * k), 0, 1, epsabs=1e-14,
                                    epsrel=1e-13)[0]
            out[idx] = betainc(b, a, c) + inside
        return out
    tau = np.array([-0.5, 0.4, 1.6])
    cdf = np.concatenate([np.zeros(eta.shape + (1,)), expit(tau - eta[..., None]), np.ones(eta.shape + (1,))], axis=-1)
    return np.concatenate([np.full(eta.shape + (1,), aux["hu"]), (1 - aux["hu"]) * np.diff(cdf, axis=-1)], axis=-1)


@pytest.mark.parametrize("family", NEW)
def test_fitted_and_new_row_predictions_by_hand(family):
    fam = get_family(family)
    m = build(family, frame=FRAME)
    post = draws(family)
    u = np.array([0.1, -0.1, 0.3])[pd.Categorical(FRAME.g).codes]
    base = 0.0 if fam.thresholds else 0.3
    eta = np.broadcast_to(base - 0.2 * FRAME.x.to_numpy() + u, (2, 4, 8))
    trials = FRAME["size"].to_numpy()
    lin = m.predict(post, kind="linear")
    np.testing.assert_allclose(lin.eta, eta, rtol=1e-14)
    assert lin.attrs["link"] == "logit"
    out = m.predict(post, kind="response", seed=9)
    mean_var = fam.mean_name
    np.testing.assert_allclose(out[mean_var], _expected(family, eta, trials), rtol=1e-10 if family == "xbeta" else 1e-13)
    y = out.y.to_numpy()
    assert y.dtype == (np.int64 if fam.discrete else np.float64)
    if family == "hurdle_cumulative":
        assert list(out["category"].values) == ["0", "1", "2", "3", "4"] and set(np.unique(y)) <= {0, 1, 2, 3, 4}
        assert out.attrs["hurdle_category"] == "0" and json.loads(out.attrs["categories_json"]) == ["1", "2", "3", "4"]
    elif fam.has_trials:
        assert (y >= 0).all() and (y <= trials).all()
    else:
        assert ((y >= 0) & (y <= 1)).all()
    new = pd.DataFrame({"x": [2.0, -1.0], "g": ["b", "c"], "size": [5, 9]})
    eta_new = np.broadcast_to(base - 0.2 * new.x.to_numpy() + np.array([-0.1, 0.3]), (2, 4, 2))
    mean = m.predict(post, newdata=new, kind="mean")[mean_var]
    np.testing.assert_allclose(mean, _expected(family, eta_new, new["size"].to_numpy()), rtol=1e-10 if family == "xbeta" else 1e-13)
    nogroup = m.predict(post, newdata=new, kind="mean", include_group_effects=False)[mean_var]
    np.testing.assert_allclose(nogroup, _expected(family, np.broadcast_to(base - 0.2 * new.x.to_numpy(), (2, 4, 2)), new["size"].to_numpy()),
                               rtol=1e-10 if family == "xbeta" else 1e-13)
    if fam.has_trials:
        with pytest.raises(Exception, match="size"):
            m.predict(post, newdata=new.drop(columns="size"), kind="mean")
    with pytest.raises(ValueError, match="kind"):
        m.predict(post, kind="probability")


def test_xbeta_censored_mean_is_the_quadrature_mean_and_not_mu():
    eta = np.array([-4.0, -0.3, 0.0, 1.2, 5.0])
    for phi, kappa in ((0.5, 0.05), (3.0, 1.0), (60.0, 2.5)):
        got = censored_mean(eta, phi, kappa)
        a, b = expit(eta) * phi, expit(-eta) * phi
        for i in range(eta.size):
            k = kappa
            inside = integrate.quad(lambda y: y * stats.beta.pdf((y + k) / (1 + 2 * k), a[i], b[i]) / (1 + 2 * k), 0, 1,
                                    epsabs=1e-14, epsrel=1e-13, limit=200)[0]
            assert abs(got[i] - (betainc(b[i], a[i], k / (1 + 2 * k)) + inside)) < 1e-10
    assert abs(censored_mean(np.array([0.0]), 3.0, 1.0)[0] - 0.5) < 1e-15             # symmetric: the mean is 1/2 = mu
    assert abs(censored_mean(np.array([1.2]), 3.0, 1.0)[0] - expit(1.2)) > 0.05        # otherwise the censoring moves it


def test_draws_outside_the_shapes_range_are_refused():
    for family in ("zero_one_inflated_beta", "xbeta", "zero_inflated_beta_binomial"):
        with pytest.raises(ValueError, match="underflowed to zero"):
            build(family, frame=FRAME).predict(draws(family, a=800.0), kind="response", seed=1)


# -- 7. weights() --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["zero_one_inflated_beta", "zero_inflated_beta_binomial", "xbeta"])
def test_weighted_log_density_is_the_rows_times_their_weights(family):
    """brms adds weights[n] * lpdf(Y[n] | ...) row by row: the weighted rows are the unweighted ones times the weights, on both
    paths; log_lik() is weighted the same way and the predictions ignore the weights."""
    lhs = SPECS[family][0]
    weighted = f"{lhs} + weights(w)" if "|" in lhs else f"{lhs} | weights(w)"
    m0, mw = build(family, frame=FRAME), build(family, frame=FRAME, lhs=weighted)
    point = _point(m0, family)
    rows = np.asarray(_rowwise(m0)(point), dtype=float)
    w = mw.design.weights
    assert np.array_equal(w, FRAME.w.to_numpy()) and np.all(np.isfinite(rows))
    np.testing.assert_allclose(np.asarray(_rowwise(mw)(point), dtype=float), w * rows, rtol=1e-12, atol=1e-12)
    from pymc.sampling.jax import get_jaxified_logp

    names = [v.name for v in mw.pymc_model.value_vars]
    lw = float(mw.pymc_model.compile_logp()(point))
    assert abs(float(get_jaxified_logp(mw.pymc_model)([point[n] for n in names])) - lw) <= 1e-12 * max(1.0, abs(lw))
    post = draws(family)
    np.testing.assert_allclose(ab.log_lik(mw, post).to_numpy(), w * ab.log_lik(m0, post).to_numpy(), rtol=1e-13)
    var = get_family(family).mean_name
    np.testing.assert_allclose(mw.predict(post, kind="mean")[var], m0.predict(post, kind="mean")[var], rtol=1e-15)
    assert mw.target_identity["sha256"] != m0.target_identity["sha256"]


# -- 8. every other construct: refused naming it, or built -------------------------------------------------------------------
def _add(lhs, term):
    return f"{lhs} + {term}" if "|" in lhs else f"{lhs} | {term}"


_OWN = {"zero_one_inflated_beta": "zoi", "zero_inflated_beta_binomial": "zi", "xbeta": "kappa", "hurdle_cumulative": "hu"}
CONTEXTS = [
    ("dpar_formula", lambda f, lhs: [f"{lhs} ~ x", f"{_OWN[f]} ~ w"], r"brms's (zi|hu|zoi|kappa) ~ \.\.\. is not offered here"),
    ("cens", lambda f, lhs: f"{_add(lhs, 'cens(c)')} ~ x", r"'cens\(c\)' is not available for the \w+ family"),
    ("trunc", lambda f, lhs: f"{_add(lhs, 'trunc(lb = 0)')} ~ x", r"'trunc\(lb = 0\)' is not available"),
    ("mo", lambda f, lhs: f"{lhs} ~ x + mo(ord)", r"the monotonic term 'mo\(ord\)' is not available for the \w+ family"),
    ("ar", lambda f, lhs: f"{lhs} ~ x + ar(time = t, gr = g)", r"the autocorrelation term .* is not available.*D70"),
    ("me", lambda f, lhs: f"{lhs} ~ me(xobs, sdx)", r"the measurement-error term 'me\(xobs, sdx\)' is not available"),
    ("mm", lambda f, lhs: f"{lhs} ~ x + (1 | mm(g, g2))", r"the multi-membership term .* is not available for the \w+ family"),
    ("gr_by", lambda f, lhs: f"{lhs} ~ x + (1 | gr(g, by = gby))", r"the term with gr\(by = \) .* is not available"),
    ("mvbind", lambda f, lhs: f"mvbind({lhs.split()[0]}, cnt) ~ x", r"a multivariate \w+ model is not offered.*D58"),
    ("cs", lambda f, lhs: f"{lhs} ~ x + cs(w)", r"brms's category-specific effect cs\(\) is not supported for the \w+ family"),
]


@pytest.mark.parametrize("family", NEW)
@pytest.mark.parametrize("label", [c[0] for c in CONTEXTS])
def test_other_constructs_are_refused_naming_them(family, label):
    _, make, message = next(c for c in CONTEXTS if c[0] == label)
    formula = make(family, SPECS[family][0])
    frame = DATA.assign(c=DATA.cens_ind)
    extra = {"simo": ab.Dirichlet(1)} if label == "mo" else {"ar": ab.Normal(0, 0.5)} if label == "ar" else {}
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1) if "(1 |" in str(formula) else None,
                    **SPECS[family][1], **extra)
    with pytest.raises(ab.FormulaError, match=message):
        getattr(ab, family)(formula, frame, pri)


@pytest.mark.parametrize("family", NEW)
def test_the_scale_formula_weights_links_mixtures_shrinkage_and_nonlinear_formulas_are_refused(family):
    lhs, extra = SPECS[family]
    fam = get_family(family)
    if fam.scale_parameter:
        with pytest.raises(ab.FormulaError, match=r"brms's phi ~ \.\.\. is not offered here"):
            getattr(ab, family)([f"{lhs} ~ x", "phi ~ w"], DATA, priors(family, False))
    if family == "hurdle_cumulative":
        with pytest.raises(ab.FormulaError, match=r"'weights\(w\)' is not available for the hurdle_cumulative family"):
            ab.hurdle_cumulative("ordh | weights(w) ~ x", DATA, priors(family, False))
        with pytest.raises(ValueError, match=r"the probit link of the hurdle_cumulative family is withheld: .*D109"):
            ab.Model(f"{lhs} ~ x", DATA, priors(family, False), family=family, link="probit")
        with pytest.raises(ab.FormulaError, match="cannot be removed"):
            ab.hurdle_cumulative("ordh ~ 0 + x", DATA, ab.Priors(b=ab.Normal(0, 1), **extra))
    else:
        with pytest.raises(ValueError, match=rf"the {family} family has the logit link only; link='probit' is not available"):
            ab.Model(f"{lhs} ~ x", DATA, priors(family, False), family=family, link="probit")
    assert ab.Model(f"{lhs} ~ x", DATA, priors(family, False), family=family, link="logit").link == "logit"
    with pytest.raises(ValueError, match=rf"a {family} mixture component is not offered.*D52"):
        ab.mixture("y ~ x", DATA, ab.Priors(components=[ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)),
                                                       ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **extra)],
                                           theta=ab.Dirichlet(2)), families=["gaussian", family])
    with pytest.raises(ab.FormulaError, match=rf"the {family} family is not offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel([f"{lhs.split()[0]} ~ a * x", "a ~ 1"], DATA, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, **extra),
                          family=family)
    with pytest.raises(ab.PriorError, match=rf"the {family} family is refused \(docs/shrinkage-priors.md, divergence D102\)"):
        getattr(ab, family)(f"{lhs} ~ x + w", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Horseshoe(), **extra))


@pytest.mark.parametrize("family", NEW)
def test_the_term_model_and_gaussian_processes_compose(family):
    """The term model is shared (categorical predictors, interactions, independent and correlated slopes, several factors, the
    offset, which enters the linear predictor) and a Gaussian process enters it before the link, as for every family but the
    categorical one (C03)."""
    lhs, extra = SPECS[family]
    slopes = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}
    for formula, pri, kw in (("x * z", dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)), {"categorical": {"z": "p"}}),
                             ("x + (1 + x || g)", dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=slopes), {}),
                             ("x + (1 + x | g) + (1 | h)", dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                               sd={**slopes, "h": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)}), {})):
        m = getattr(ab, family)(f"{lhs} ~ {formula}", DATA, ab.Priors(**pri, **extra), **kw)
        assert set(NAT[family]) <= set(m.names)
    if family != "hurdle_cumulative":
        m = getattr(ab, family)(f"{lhs} ~ 0 + x", DATA, ab.Priors(b=ab.Normal(0, 1), **extra))
        assert not m.spec.intercept
    m = getattr(ab, family)(f"{lhs} ~ x + offset(log(exposure))", DATA, priors(family, False))
    nat = {**NAT[family], "b": [0.5], "intercept": [-0.6, 0.4, 1.5] if family == "hurdle_cumulative" else 0.2}
    point = m.value_point(**nat)
    pmm = m.pymc_model
    lin = np.ravel(pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars,
                                  on_unused_input="ignore")(point))
    x = DATA.x.to_numpy()
    want = (0.5 * x if family == "hurdle_cumulative" else 0.2 + 0.5 * (x - x.mean())) + np.log(DATA.exposure.to_numpy())
    np.testing.assert_allclose(lin, want, rtol=1e-13)
    gp = getattr(ab, family)(f"{lhs} ~ gp(x)", DATA, ab.Priors(intercept=ab.Normal(0, 1), sdgp=ab.HalfNormal(1),
                                                               lscale=ab.InverseGamma(3, 1), **extra))
    assert len(gp.design.gp) == 1 and ab.default_priors(f"{lhs} ~ gp(x)", DATA, family=family).lscale is not None


@pytest.mark.parametrize("family", NEW)
def test_post_processing_hypothesis_and_conditional_effects(family):
    """log_lik(), loo() and waic() take every family (log_lik is checked against each model's own pointwise density in
    tests/test_post_processing.py); bayes_R2() the three that are not ordinal. hypothesis() reads the coefficients of the three,
    and conditional_effects() summarises predict(kind="mean") without group effects on its grid (G4 checks it against brms's
    posterior_epred) for the two without trials; the hurdle ordinal family is refused as the cumulative one is (D99)."""
    b = gates.Bundle(CASES[family])
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    ll = ab.log_lik(b.model, idata)
    assert ll.shape == (1, 8, b.model.design.n) and np.isfinite(ll).all()
    fixed = gates.fixed_parameter_draws(b)
    assert np.isfinite(ab.loo(b.model, fixed).estimates.loc["elpd_loo", "Estimate"])
    assert np.isfinite(ab.waic(b.model, fixed).estimates.loc["elpd_waic", "Estimate"])
    if family == "hurdle_cumulative":
        with pytest.raises(ValueError, match=r"bayes_R2\(\) of the hurdle_cumulative family is not offered.*D109"):
            ab.bayes_R2(b.model, idata)
        for fn, args in ((ab.hypothesis, (["x > 0"],)), (ab.conditional_effects, (["x"],))):
            with pytest.raises(ValueError, match="hurdle_cumulative family's thresholds is not offered"):
                fn(b.model, idata, *args)
        return
    epred, y = b.fixture.prediction("fitted", "epred"), b.model.design.y            # brms's definition on brms's own epred
    r2 = np.var(epred, axis=-1, ddof=1) / (np.var(epred, axis=-1, ddof=1) + np.var(y - epred, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(b.model, idata).to_numpy().ravel(), r2, rtol=1e-9)
    h = ab.hypothesis(b.model, idata, ["x > 0"])
    np.testing.assert_allclose(h.table["Estimate"].iloc[0], idata.posterior["x"].mean().item(), rtol=1e-12)
    if b.model.family_spec.has_trials:
        with pytest.raises(ValueError, match="needs the trials of each grid row"):
            ab.conditional_effects(b.model, idata, ["x"])
        return
    table = ab.conditional_effects(b.model, idata, ["x"], resolution=7)["x"]
    grid = b.model.predict(idata, newdata=table[["x"]].reset_index(drop=True), kind="mean", include_group_effects=False)
    np.testing.assert_allclose(table["estimate__"].to_numpy(), np.median(grid["mean"].to_numpy().reshape(-1, len(table)), axis=0),
                               rtol=1e-12)


# -- 9. describe(), identity, prior predictive, defaults, the fit record ------------------------------------------------------
LIKELIHOOD_LINES = {
    "zero_one_inflated_beta": "the density is zoi (1 - coi) at 0, zoi coi at 1",
    "zero_inflated_beta_binomial": "P(zk_i = 0) = zi + (1 - zi) BetaBinomial(0 | size_i, mu_i phi,",
    "xbeta": "P(y_i = 0) = I_c(mu_i phi, (1 - mu_i) phi), P(y_i = 1) = I_c((1 - mu_i) phi, mu_i phi)",
    "hurdle_cumulative": "log P(y_i = 0) = log(hu),",
}


@pytest.mark.parametrize("family", NEW)
def test_describe_identity_and_registry_constructor(family):
    fam = get_family(family)
    m = build(family)
    text = m.describe()
    assert text.startswith(f"AutoBayes {fam.tag} model") and f"family    : {family}, logit link" in text
    assert LIKELIHOOD_LINES[family] in text
    for name in NAT[family]:
        prior = SPECS[family][1][name]
        assert f"{name} ~ {prior}" in text
        assert (f"{name}_log__" if name in ("phi", "kappa") else f"{name}_logodds__") in text
    same = ab.Model(f"{SPECS[family][0]} ~ x + (1 | g)", DATA, priors(family), family=family)
    assert same.describe() == text and same.target_identity == m.target_identity
    first = next(iter(NAT[family]))
    other = ab.Beta(1, 1) if first in ("zoi", "coi", "zi", "hu") else ab.Gamma(1, 1)
    assert build(family, **{first: other}).target_identity != m.target_identity
    last = list(NAT[family])[-1]
    other = ab.Beta(1, 3) if last in ("zoi", "coi", "zi", "hu") else ab.Gamma(1, 3)
    assert build(family, **{last: other}).target_identity != m.target_identity


def test_target_identity_tells_the_families_apart():
    """zero_one_inflated_beta and xbeta on the same response, and hurdle_cumulative against the cumulative family on the same
    codes 1..4 (the hurdle category and hu make a different model)."""
    ids = {build(f).target_identity["sha256"] for f in ("zero_one_inflated_beta", "xbeta")}
    frame = DATA.assign(o=DATA.ordh.clip(lower=1))
    ids.add(ab.hurdle_cumulative("o ~ x", frame, priors("hurdle_cumulative", False)).target_identity["sha256"])
    ids.add(ab.cumulative("o ~ x", frame, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1))).target_identity["sha256"])
    assert len(ids) == 4


@pytest.mark.parametrize("family", NEW)
def test_prior_predictive(family):
    m = build(family)
    out = m.prior_predictive(draws=40, seed=3)
    for name in NAT[family]:
        v = out.prior[name].values
        assert v.shape == (1, 40) and (v > 0).all() and (name in ("phi", "kappa") or (v < 1).all())
    y = out.prior_predictive[m.names["y"]].values
    assert y.shape == (1, 40, m.design.n) and (y >= 0).all() and (y == 0).any()
    if family == "zero_inflated_beta_binomial":
        assert y.dtype.kind == "i" and (y <= DATA["size"].to_numpy()).all()
    elif family == "hurdle_cumulative":
        assert y.dtype.kind == "i" and (y <= 4).all() and (y > 0).any()
    else:
        assert (y <= 1).all() and (y == 1).any() and ((y > 0) & (y < 1)).any()


@pytest.mark.parametrize("family, defaults", [
    ("zero_one_inflated_beta", {"phi": ab.Gamma(0.01, 0.01), "zoi": ab.Beta(1, 1), "coi": ab.Beta(1, 1)}),
    ("zero_inflated_beta_binomial", {"phi": ab.Gamma(0.01, 0.01), "zi": ab.Beta(1, 1)}),
    ("xbeta", {"phi": ab.Gamma(0.01, 0.01), "kappa": ab.Gamma(0.01, 0.01)}),
    ("hurdle_cumulative", {"hu": ab.Beta(1, 1), "intercept": ab.StudentT(3, 0, 2.5)}),
])
def test_default_priors_build_the_model(family, defaults):
    """brms's defaults (probed with default_prior(); G6 checks every row of brms's table on each frozen case)."""
    lhs = SPECS[family][0]
    pri = ab.default_priors(f"{lhs} ~ x + (1 | g)", DATA, family=family)
    for name, prior in defaults.items():
        assert getattr(pri, name) == prior and getattr(pri, name).source == "default"
    m = getattr(ab, family)(f"{lhs} ~ x + (1 | g)", DATA, pri)
    assert set(NAT[family]) <= set(m.default_prior_labels)
    rows = {(r["class"], r["coef"]): r for r in m.prior_table()}
    for name in NAT[family]:
        bounds = ("0", "") if name in ("phi", "kappa") else ("0", "1")
        assert (rows[(name, "")]["lb"], rows[(name, "")]["ub"]) == bounds


def test_fit_record_and_summary_rows(monkeypatch):
    import autobayes.session as session

    m = build("zero_one_inflated_beta", frame=FRAME)
    post = draws("zero_one_inflated_beta")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert set(rows) == {"Intercept", "x", "1|g_sigma", "phi", "zoi", "coi"}
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "zero_one_inflated_beta" and rec["model"]["link"] == "logit"
    assert rec["model"]["priors"]["zoi"] == "Beta(alpha=2, beta=2)" and rec["model"]["priors"]["phi"] == "Gamma(alpha=2, beta=0.5)"
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
