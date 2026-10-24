"""The Wiener diffusion model (brms-parity F04, docs/wiener.md): the first-passage time density against a 60-digit evaluation of
both Navarro-Fuss series (``decimal``), its gradient against central differences of that evaluation, on PyTensor's and JAX's
paths; the edges (t <= 0, extreme u and w, the series switch); the mean decision time against its closed form at 60 digits
and against numerical integration of the density, which also integrates to 1; brms's posterior_epred formula and the frozen
points (the linear predictor is brms's, brms's expected values are its formula on the same draws, and G4's expected values fail
for that reason only); log_lik, loo and bayes_R2; dec() and its data; the refusals naming every construct no frozen case
verifies; priors and value points; the term model and bs's formula composing with the likelihood; by-hand predictions and the
refused draws; brms's default priors on two frames; describe() under two hash seeds, the target identity and the fit record;
G1's dec check. No MCMC; the brms comparison is the oracle cases wiener_varint and wiener_distributional_bs."""
import copy
import json
import math
import os
import subprocess
import sys
import warnings
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import integrate

import autobayes as ab
from autobayes import wiener_diffusion as wn
from autobayes.families import FAMILIES, get_family
from autobayes.families.wiener import BIAS, BS, NDT, WIENER, _draw_response
from autobayes.formula import FormulaError, addition_data, parse_formula
from tests.oracle import adapter, gates
from tests.oracle import fixture as fx
from tests.oracle.tolerances import PRED_REL, rel_err
from tests.test_continuous_families import graph, rows_logp

REPO = Path(__file__).resolve().parent.parent
PREC = 60


# -- a 60-digit reference, written from the paper (Navarro and Fuss 2009), not from the library ------------------------------
def _pi() -> Decimal:
    """pi at the context's precision (the decimal module's documented recipe)."""
    with localcontext() as ctx:
        ctx.prec += 2
        lasts, t, s, n, na, d, da = 0, Decimal(3), Decimal(3), 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    return +s


def _sin(x: Decimal, pi: Decimal) -> Decimal:
    """sin(x): reduced modulo 2 pi, then its Taylor series."""
    x = x - 2 * pi * (x / (2 * pi)).to_integral_value()
    with localcontext() as ctx:
        ctx.prec += 4
        i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= x * x
            sign = -sign
            s += num / fact * sign
    return +s


def ref_small(u: Decimal, w: Decimal) -> Decimal:
    """log of the small-time series, k from -40 to 40."""
    pi = _pi()
    s = sum((w + 2 * k) * (-(w + 2 * k) ** 2 / (2 * u)).exp() for k in range(-40, 41))
    return s.ln() - (2 * pi * u ** 3).ln() / 2


def ref_large(u: Decimal, w: Decimal) -> Decimal:
    """log of the large-time series, k from 1 to 40."""
    pi = _pi()
    s = sum(k * (-(k * k) * pi * pi * u / 2).exp() * _sin(k * pi * w, pi) for k in range(1, 41))
    return (pi * s).ln()


def ref_standard(u, w) -> Decimal:
    """log f(u | 0, 1, w), each series where it converges fast (both agree there: test_reference_series_agree)."""
    with localcontext() as ctx:
        ctx.prec = PREC
        u, w = Decimal(u), Decimal(w)
        return ref_small(u, w) if u <= 1 else ref_large(u, w)


def ref_wiener(rt, dec, mu, bs, ndt, bias) -> Decimal:
    """brms's wiener log density of one row at 60 digits: the upper boundary's (w = 1 - bias, drift -mu) for dec = 1."""
    with localcontext() as ctx:
        ctx.prec = PREC
        rt, mu, bs, ndt, bias = (Decimal(x) for x in (rt, mu, bs, ndt, bias))
        w, v = (1 - bias, -mu) if dec else (bias, mu)
        t = rt - ndt
        u = t / (bs * bs)
        return -v * bs * w - v * v * t / 2 - 2 * bs.ln() + (ref_small(u, w) if u <= 1 else ref_large(u, w))


def ref_gradient(row) -> list[float]:
    """d/d(mu, bs, ndt, bias) of ``ref_wiener`` by central differences at 60 digits (step 1e-25, error below 1e-35)."""
    rt, dec, *pars = row
    out = []
    with localcontext() as ctx:
        ctx.prec = PREC
        h = Decimal("1e-25")
        for j in range(4):
            hi = [Decimal(p) + (h if k == j else 0) for k, p in enumerate(pars)]
            lo = [Decimal(p) - (h if k == j else 0) for k, p in enumerate(pars)]
            out.append(float((ref_wiener(rt, dec, *hi) - ref_wiener(rt, dec, *lo)) / (2 * h)))
    return out


def test_reference_series_agree():
    """The reference itself: both series at 60 digits agree where both converge fast."""
    with localcontext() as ctx:
        ctx.prec = PREC
        for u in ("0.3", "0.7", "1.5", "3"):
            for w in ("0.1", "0.5", "0.95"):
                a, b = ref_small(Decimal(u), Decimal(w)), ref_large(Decimal(u), Decimal(w))
                assert abs(a - b) < Decimal("1e-40"), (u, w)


# -- the density ---------------------------------------------------------------------------------------------------------------
US = (1e-4, 1e-3, 0.01, 0.05, 0.2, 0.4, 0.5, 0.5000001, 0.6, 0.9, 1.0, 1.7, 3.0, 10.0, 60.0, 200.0)
WS = (1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1 - 1e-6, 1 - 1e-12)


def test_standard_density_is_exact_on_both_sides_of_the_switch():
    """log f(u | 0, 1, w) within 4e-15 of the 60-digit value (absolute below 1, relative above) from u = 1e-4 to 200 and w from
    1e-12 to 1 - 1e-12: both groupings of the small-time terms (w <= 1/2 and above), the reflected sine of the large-time ones,
    each series on its side of u = 1/2."""
    worst = 0.0
    for w in WS:
        e = 1.0 - w                                           # exact for w >= 1/2 (Sterbenz); unused by the w <= 1/2 forms
        ours = wn.log_standard_density(np.array(US), np.full(len(US), w), np.full(len(US), e))
        for u, got in zip(US, ours):
            want = float(ref_standard(u, w))
            worst = max(worst, abs(got - want) / max(1.0, abs(want)))
    assert worst < 4e-15, worst


ROWS = [  # rt, dec, mu, bs, ndt, bias: both boundaries, both series, drifts of both signs, extreme starting points
    (0.9, 1, 0.8, 1.2, 0.3, 0.45), (0.9, 0, 0.8, 1.2, 0.3, 0.45), (0.41, 1, -1.3, 0.9, 0.4, 0.7), (0.41, 0, -1.3, 0.9, 0.4, 0.7),
    (2.5, 1, 2.1, 0.6, 0.2, 0.2), (2.5, 0, 2.1, 0.6, 0.2, 0.2), (1.1, 1, 0.0, 1.0, 0.3, 0.5), (0.55, 0, 0.4, 1.6, 0.5, 1e-6),
    (0.55, 1, 0.4, 1.6, 0.5, 1e-6), (3.0, 0, -0.2, 2.4, 0.1, 0.999), (0.7, 1, 1.5, 1.0, 0.2, 0.5),
]


def _rows(mode, rows):
    """Per-row log density and its gradient in (mu, bs, ndt, bias) on PyTensor's or JAX's path."""
    arr = np.array(rows, dtype=float)
    rt, upper, pars = arr[:, 0], arr[:, 1] == 1, [arr[:, k] for k in range(2, 6)]
    vs = [pt.dvector(name) for name in ("mu", "bs", "ndt", "bias")]
    out = wn.wiener_logp(rt, *vs, upper)
    if mode == "pytensor":
        f = pytensor.function(vs, [out, *pt.grad(out.sum(), vs)])
        return [np.asarray(a, dtype=float) for a in f(*pars)]
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    jf = get_jaxified_graph(inputs=vs, outputs=[out])
    grads = jax.grad(lambda *a: jf(*a)[0].sum(), argnums=(0, 1, 2, 3))(*pars)
    return [np.asarray(jf(*pars)[0], dtype=float), *(np.asarray(g, dtype=float) for g in grads)]


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_density_and_gradient_against_the_60_digit_reference(mode):
    value, *grads = _rows(mode, ROWS)
    want = np.array([float(ref_wiener(*r)) for r in ROWS])
    assert rel_err(value, want) < 2e-14
    np.testing.assert_allclose(value, wn.wiener_log_density(np.array([r[0] for r in ROWS]), np.array([r[1] for r in ROWS]) == 1,
                                                           *(np.array([r[k] for r in ROWS]) for k in range(2, 6))), rtol=2e-15)
    ref = np.array([ref_gradient(r) for r in ROWS])
    for k in range(4):
        assert rel_err(grads[k], ref[:, k]) < 1e-11, k


@pytest.mark.parametrize("mode", ["pytensor", "jax"])
def test_density_edges_stay_finite_with_a_finite_gradient(mode):
    """rt <= ndt is -inf with a zero gradient; u far below and above the switch, at it and w at 1/2 are finite, with finite
    gradients (the branch not taken is evaluated on its own side)."""
    rows = [(0.3, 1, 0.5, 1.0, 0.3, 0.5), (0.2, 0, 0.5, 1.0, 0.3, 0.5),            # t = 0 and t < 0
            (0.300001, 1, 0.5, 1.0, 0.3, 0.4), (30.0, 0, 0.5, 0.05, 0.3, 0.4),     # u = 1e-6 and u = 1.2e4
            (0.8, 1, 0.5, 1.0, 0.3, 0.6), (0.8, 0, 0.5, 1.0, 0.3, 0.5)]            # u at the switch (1/2); w = 1/2 exactly
    value, *grads = _rows(mode, rows)
    assert np.isneginf(value[:2]).all() and np.isfinite(value[2:]).all()
    assert all(np.isfinite(g).all() for g in grads) and all((g[:2] == 0).all() for g in grads)
    for r, v in zip(rows[2:], value[2:]):
        assert abs(v - float(ref_wiener(*r))) <= 1e-14 * max(1.0, abs(v)), r


def _density(t, v, a, w):
    """The density of the decision time at each boundary (NumPy path): (upper, lower)."""
    with np.errstate(all="ignore"):
        up = np.exp(wn.log_lower_density(np.asarray(t, float), -v, a, 1.0 - w, w))
        low = np.exp(wn.log_lower_density(np.asarray(t, float), v, a, w, 1.0 - w))
    return up, low


@pytest.mark.parametrize("v, a, w", [(0.8, 1.7, 0.4), (-1.3, 0.6, 0.7), (0.0, 1.2, 0.3), (2.5, 2.0, 0.5), (0.05, 0.9, 0.85)])
def test_density_integrates_to_one_and_its_mean_is_the_mean_decision_time(v, a, w):
    """By quadrature of the library's density over both boundaries: total mass 1, the upper boundary's mass P(upper) =
    (1 - exp(-2 v a w)) / (1 - exp(-2 v a)) (w at v = 0), and the mean decision time ``mean_decision_time``."""
    def quad(f):
        return sum(integrate.quad(f, lo, hi, limit=200, epsabs=1e-14, epsrel=1e-13)[0]
                   for lo, hi in ((0, 0.05 * a * a), (0.05 * a * a, a * a), (a * a, 8 * a * a), (8 * a * a, np.inf)))
    total = quad(lambda t: sum(_density(t, v, a, w)))
    upper = quad(lambda t: _density(t, v, a, w)[0])
    mean = quad(lambda t: t * sum(_density(t, v, a, w)))
    p_up = w if v == 0 else math.expm1(-2 * v * a * w) / math.expm1(-2 * v * a)
    assert abs(total - 1) < 1e-10 and abs(upper - p_up) < 1e-10
    assert abs(mean - float(wn.mean_decision_time(v, a, w))) < 1e-9 * mean


# -- the mean decision time -------------------------------------------------------------------------------------------------
def ref_mean(v, a, w) -> float:
    """(a P(upper) - a w) / v at 60 digits, a^2 w (1 - w) at v = 0."""
    with localcontext() as ctx:
        ctx.prec = PREC
        v, a, w = Decimal(v), Decimal(a), Decimal(w)
        if v == 0:
            return float(a * a * w * (1 - w))
        p_up = (1 - (-2 * v * a * w).exp()) / (1 - (-2 * v * a).exp())
        return float((a * p_up - a * w) / v)


def test_mean_decision_time_against_its_closed_form():
    """Within 3e-15 of the 60-digit closed form for drifts from -400 to 400 (1e-9 and 0 included: no cancellation as v -> 0, no
    overflow at v a = 1200) and starts from 1e-9 to 1 - 1e-9, on every side of the forms' switches (v a = 1 and 2, v z = 1,
    w = 1/2), and the symmetric start's a / (2 v) tanh(v a / 2)."""
    worst = 0.0
    for v in (-400.0, -50.0, -5.0, -2.0, -1.6, -0.5, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 0.5, 0.9, 1.0, 1.4, 1.6, 2.0, 2.5, 5.0, 50.0,
              400.0):
        for a in (0.5, 1.3, 3.0):
            for w in (1e-9, 0.01, 0.3, 0.5, 0.5000001, 0.8, 0.99, 1 - 1e-9):
                got, want = float(wn.mean_decision_time(v, a, w)), ref_mean(v, a, w)
                worst = max(worst, abs(got - want) / want)
    assert worst < 3e-15, worst
    v, a = np.array([0.3, -2.0, 7.0]), np.array([1.1, 0.7, 2.0])
    np.testing.assert_allclose(wn.mean_decision_time(v, a, 0.5), a / (2 * v) * np.tanh(v * a / 2), rtol=1e-14)
    assert wn.mean_decision_time(np.array([[0.2], [0.0]]), 1.0, np.array([0.3, 0.6])).shape == (2, 2)   # broadcast


def ref_log_sinhc(x) -> float:
    with localcontext() as ctx:
        ctx.prec = PREC
        x = Decimal(x)
        return 0.0 if x == 0 else float((((x.exp() - (-x).exp()) / 2) / x).ln())


def test_log_sinhc_on_both_branches():
    xs = (0.0, 1e-12, 1e-4, 0.3, 0.999, 1.0, 1.001, 5.0, 50.0, 800.0)
    for x, got in zip(xs, wn.log_sinhc(np.array(xs))):
        want = ref_log_sinhc(x)
        assert abs(got - want) <= 4e-16 * abs(want), (x, got, want)


def test_expected_response_time_and_brms_formula():
    """E[rt] = ndt + E[T]; brms's posterior_epred formula equals it at bs = 1 only (relative bias where bs * bias belongs)."""
    mu, ndt, bias = np.array([0.7, -1.2, 2.0]), 0.25, 0.4
    np.testing.assert_allclose(wn.expected_response_time(mu, 1.3, ndt, bias), ndt + wn.mean_decision_time(mu, 1.3, bias), rtol=0)
    np.testing.assert_allclose(wn.brms_epred(mu, 1.0, ndt, bias), wn.expected_response_time(mu, 1.0, ndt, bias), rtol=1e-13)
    assert np.all(np.abs(wn.brms_epred(mu, 1.6, ndt, bias) - wn.expected_response_time(mu, 1.6, ndt, bias)) > 1e-3)


# -- the frozen brms points -------------------------------------------------------------------------------------------------
CASES = ("wiener_varint", "wiener_distributional_bs")


@pytest.fixture(scope="module", params=CASES)
def bundle(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gates.Bundle(request.param)


def _draws_with_dpars(b, idata, nd, include):
    from autobayes.predict import _with_dpars

    post = idata.posterior
    return _with_dpars(b.model, post, b.model.data if nd is None else nd, include_group_effects=include) if b.model.spec.dpars else post


def test_frozen_points_linear_predictor_and_brms_epred(bundle):
    """At brms's frozen points: the linear predictor is brms's posterior_linpred in every block (G4's first part, which the
    case's expected failure of G4 would not show), brms's posterior_epred is brms's formula on the same draws (so G4's expected
    values fail for the formula alone), and predict(kind="mean") is the mean response time of those draws, off brms's by more
    than the tolerance."""
    from autobayes.families.spec import extra_draws

    b = bundle
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    for block, nd in (("fitted", None), ("newdata", b.newdata)):
        for include, suffix in ((True, ""), (False, "_nogroup")):
            lin = b.model.predict(idata, newdata=nd, kind="linear", include_group_effects=include)["eta"].to_numpy()[0]
            assert rel_err(lin, b.fixture.prediction(block, "linpred" + suffix)) <= PRED_REL
            post = _draws_with_dpars(b, idata, nd, include)
            bs, ndt, bias = (extra_draws(b.model, post, name)[0] for name in ("bs", "ndt", "bias"))
            brms = b.fixture.prediction(block, "epred" + suffix)
            assert rel_err(wn.brms_epred(lin, bs, ndt, bias), brms) <= PRED_REL
            mean = b.model.predict(idata, newdata=nd, kind="mean", include_group_effects=include)["mean"].to_numpy()[0]
            np.testing.assert_allclose(mean, wn.expected_response_time(lin, bs, ndt, bias), rtol=1e-15)
            assert rel_err(mean, brms) > 1e-3
    with pytest.raises(gates.OracleMismatch, match=r"fitted: 'mean' differs from brms posterior_epred"):
        gates.check_predictions(b)


def test_frozen_points_log_lik_loo_and_bayes_R2(bundle):
    """log_lik row by row is the model's own density at the frozen points; its sum is brms's likelihood (its log density less
    lprior and the standard normal offsets) within 2e-10 of the log density, the inexactness of Stan's wiener_lpdf (D150);
    loo runs; bayes_R2 is its definition on predict(kind="mean")."""
    b = bundle
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    ll = ab.log_lik(b.model, idata).to_numpy()[0]
    for k, p, vp in b.points():
        assert rel_err(ll[k], rows_logp(b.model, vp)) <= 1e-12
        z = np.asarray(p.pars["z_1"], dtype=float).ravel()
        brms = p.lp_nojac - float(p.pars["lprior"]) - float(np.sum(-0.5 * math.log(2 * math.pi) - z * z / 2))
        assert abs(ll[k].sum() - brms) < 2e-10 * abs(p.lp_nojac)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # eight one-chain draws: Pareto k is large, as it should be
        assert np.isfinite(ab.loo(b.model, idata).estimates.loc["elpd_loo", "Estimate"])
    yhat, y = b.model.predict(idata, kind="mean")["mean"].to_numpy()[0], b.model.design.y
    want = np.var(yhat, axis=-1, ddof=1) / (np.var(yhat, axis=-1, ddof=1) + np.var(y - yhat, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(b.model, idata).to_numpy()[0], want, rtol=1e-12)


def _with_standata(b, standata):
    """The bundle with other Stan data (the fixture's cached ``standata`` replaced in a copy)."""
    fixture = copy.copy(b.fixture)
    fixture.__dict__["standata"] = standata
    return SimpleNamespace(fixture=fixture, model=b.model, cmap=b.cmap, gmap=b.gmap)


def test_g1_checks_dec(bundle):
    """G1 (the gate's F04 branch): brms's dec equals the model's decisions; a changed one, or none in brms's data, is a
    mismatch."""
    b = bundle
    np.testing.assert_array_equal(b.model.design.dec, b.fixture.standata["dec"])
    gates.check_design(b)
    with pytest.raises(gates.OracleMismatch, match="dec differs"):
        gates.check_design(_with_standata(b, {**b.fixture.standata, "dec": 1 - b.fixture.standata["dec"]}))
    with pytest.raises(gates.OracleMismatch, match="AutoBayes has dec but brms standata has none"):
        gates.check_design(_with_standata(b, {k: v for k, v in b.fixture.standata.items() if k != "dec"}))


def test_g1_refuses_dec_on_a_model_without_it():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = gates.Bundle("shifted_lognormal_varint")
    with pytest.raises(gates.OracleMismatch, match="brms has dec but AutoBayes does not"):
        gates.check_design(_with_standata(b, {**b.fixture.standata, "dec": np.ones(b.model.design.n)}))


# -- dec() and the formula ----------------------------------------------------------------------------------------------------
def frame(n=24, seed=8):
    """Response times above 0.5 with decisions, a numeric and a categorical predictor, two grouping factors and an offset."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "w": np.round(rng.uniform(-1, 1, n), 4),
                         "z": np.array(["p", "q", "r"])[np.arange(n) % 3], "g": [f"l{k}" for k in np.arange(n) % 4],
                         "h": np.array(["u", "v"])[(np.arange(n) // 4) % 2], "rt": np.round(0.35 + rng.gamma(2.0, 0.3, n), 4),
                         "dec": (np.arange(n) * 7 % 3 == 0).astype(int), "off": np.round(rng.uniform(-0.5, 0.5, n), 4)})


OWN = dict(bs=ab.Gamma(1, 1), ndt=ab.Normal(0, 0.3), bias=ab.Beta(2, 2))


def build(formula="rt | dec(dec) ~ x", frame_=None, **extra):
    priors = ab.Priors(**{"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1), **OWN, **extra})
    return ab.wiener(formula, frame() if frame_ is None else frame_, priors)


def test_dec_values_are_ones_and_zeros():
    spec = parse_formula("rt | dec(dec) ~ x", frame(), "wiener")
    assert [a.term for a in spec.additions] == ["dec(dec)"] and spec.addition("dec").columns == ("dec",)
    for given in (frame().dec, frame().dec.astype(float), frame().dec == 1):
        out = addition_data(spec, frame().assign(dec=given))["dec"]
        assert out.dtype == np.int64 and np.array_equal(out, frame().dec.to_numpy())
    bad = [(np.where(frame().dec == 1, "upper", "lower"), r"dtype .*; the decisions of 'dec\(dec\)' are 1 \(upper boundary\) or 0"),
           (frame().dec * 2, r"found 2\.0 \(brms codes every nonzero value as the upper boundary; divergence D151\)"),
           (frame().dec - 1, r"found -1\.0"), (frame().dec * 0.5, r"found 0\.5"),
           (np.where(np.arange(24) == 3, np.nan, frame().dec), r"has missing or non-finite values"),
           (frame().dec.astype(complex), r"is complex-valued")]
    for values, needle in bad:
        with pytest.raises(FormulaError, match=needle):
            addition_data(spec, frame().assign(dec=values))
    m = build()
    np.testing.assert_array_equal(m.design.dec, frame().dec.to_numpy())
    assert build("rt | dec(dec) ~ x", frame().assign(dec=frame().dec == 1)).target_identity == m.target_identity


L = "rt | dec(dec)"
REFUSALS = [   # (construct, formula, extra priors, frame changes, error, message)
    ("no dec()", "rt ~ x", {}, {}, FormulaError, r"the wiener family needs each response time's decision, 'rt \| dec\(dec\) ~ \.\.\.'.*"
     r"Addition argument 'dec' is required"),
    ("dec() twice", "rt | dec(dec) + dec(dec) ~ x", {}, {}, FormulaError, r"dec\(\) appears more than once"),
    ("dec() of two columns", "rt | dec(dec, x) ~ x", {}, {}, FormulaError, r"dec\(\) takes one column name of decisions, 1 or 0"),
    ("another addition term", "rt | foo(x) ~ x", {}, {}, FormulaError, r"not a supported response addition term.*dec\(col\) for the wiener family"),
    ("weights()", f"{L} + weights(w) ~ x", {}, {"w": 1.5}, FormulaError, r"'weights\(w\)' is not available for the wiener family"),
    ("cens()", f"{L} + cens(c) ~ x", {}, {"c": 0}, FormulaError, r"'cens\(c\)' is not available for the wiener family"),
    ("trunc()", f"{L} + trunc(lb = 0) ~ x", {}, {}, FormulaError, r"'trunc\(lb = 0\)' is not available for the wiener family"),
    ("mi() on the response", f"{L} + mi() ~ x", {}, {}, FormulaError, r"'mi\(\)' is not available for the wiener family"),
    ("ndt's formula", [f"{L} ~ x", "ndt ~ w"], {}, {}, FormulaError, r"'ndt ~ w': brms's ndt ~ \.\.\. is not offered here"),
    ("bias's formula", [f"{L} ~ x", "bias ~ w"], {}, {}, FormulaError, r"'bias ~ w': brms's bias ~ \.\.\. is not offered here"),
    ("sigma's formula", [f"{L} ~ x", "sigma ~ w"], {}, {}, FormulaError, r"the wiener family has no parameter 'sigma' \(its parameters "
     r"besides the mean are \['bs', 'ndt', 'bias'\]\)"),
    ("group terms in bs's formula", [f"{L} ~ x + (1 | g)", "bs ~ w + (1 | g)"], {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"group-level terms in a distributional formula"),
    ("a shared ID in bs's formula", [f"{L} ~ x + (1 |p| g)", "bs ~ w + (1 |p| g)"], {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"group-level terms in the formula of bs of the wiener family .* not supported in this slice"),
    ("mo()", f"{L} ~ mo(ord)", {}, {"ord": np.arange(24) % 4}, FormulaError, r"the monotonic term 'mo\(ord\)' is not available for the wiener"),
    ("me()", f"{L} ~ me(x, sdx)", {}, {"sdx": 0.3}, FormulaError, r"the measurement-error term 'me\(x, sdx\)' is not available for the wiener"),
    ("mi()", f"{L} ~ mi(xm)", {}, {"xm": np.where(np.arange(24) % 5 == 0, np.nan, 0.5)}, FormulaError,
     r"the missing-value term 'mi\(xm\)' is not available for the wiener family"),
    ("ar()", f"{L} ~ x + ar(p = 1)", {}, {}, FormulaError, r"the autocorrelation term 'ar\(p = 1\)' is not available for the wiener family"),
    ("mvbind()", "mvbind(rt, x) ~ w", {}, {}, FormulaError, r"a multivariate wiener model is not offered"),
    ("mm()", f"{L} ~ x + (1 | mm(g, h))", {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"the multi-membership term '\(1 \| mm\(g, h\)\)' next to the addition term 'dec\(dec\)' is not supported"),
    ("gr(by = )", f"{L} ~ x + (1 | gr(g, by = h))", {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"the term with gr\(by = \) .* next to the addition term 'dec\(dec\)' is not supported"),
    ("cs()", f"{L} ~ cs(x)", {}, {}, FormulaError, r"brms's category-specific effect cs\(\) is not supported for the wiener family"),
    ("horseshoe()", f"{L} ~ x", {"b": ab.Horseshoe(3)}, {}, ab.PriorError, r"Horseshoe\(\) on b is offered for the Gaussian family only.*wiener"),
    ("a response time of 0", f"{L} ~ x", {}, {"rt": np.r_[0.0, frame().rt.to_numpy()[1:]]}, FormulaError,
     r"response 'rt' has a non-positive value .*; the wiener family needs strictly positive responses"),
    ("a missing decision", f"{L} ~ x", {}, {"dec": np.where(np.arange(24) == 3, np.nan, frame().dec)}, FormulaError,
     r"missing values in columns \['dec'\]"),
]


@pytest.mark.parametrize("construct", [r[0] for r in REFUSALS])
def test_other_constructs_are_refused_by_name(construct):
    _, formula, extra, changes, error, needle = next(r for r in REFUSALS if r[0] == construct)
    with pytest.raises(error, match=needle):
        build(formula, frame().assign(**changes), **extra)


def test_links_mixtures_nonlinear_models_and_other_families_are_refused():
    df = frame().assign(y=frame().x)
    with pytest.raises(ValueError, match=r"the wiener family has the identity link only; link='log' is not available"):
        ab.Model(f"{L} ~ x", df, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **OWN), family="wiener", link="log")
    comp = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **OWN)
    with pytest.raises(ValueError, match=r"a wiener mixture component is not offered.*D52"):
        ab.mixture("rt ~ x", df, ab.Priors(components=[comp, comp], theta=ab.Dirichlet(2)), families=["wiener", "wiener"])
    with pytest.raises(FormulaError, match=r"the wiener family is not offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel(["rt ~ a * x", "a ~ 1"], df, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, **OWN), family="wiener")
    with pytest.raises(FormulaError, match=r"'dec\(dec\)' is not available for the gaussian family; brms's dec\(\) is verified here "
                                           r"for the wiener family"):
        ab.gaussian("y | dec(dec) ~ x", df, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a model with the addition term\(s\) dec\(dec\) is not offered"):
        ab.conditional_effects(build(), az.from_dict(posterior={"Intercept": np.zeros((1, 2))}), "x")
    with pytest.raises(ValueError, match=r"the wiener family's bs, ndt, bias"):
        ab.prior_draws(build())


# -- priors and value points ------------------------------------------------------------------------------------------------
def test_each_parameter_takes_its_own_priors():
    base = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1))
    for missing, example in (("bs", r"such as ab\.Gamma\(1, 0\.1\) \(brms's default is gamma\(1, 1\)\)"),
                             ("ndt", r"such as ab\.Uniform\(0, \"min_Y\"\) \(brms's default is uniform\(0, min_Y\)\)"),
                             ("bias", r"such as ab\.Beta\(1, 1\) \(brms's default is beta\(1, 1\)\)")):
        with pytest.raises(ab.PriorError, match=rf"the wiener family needs a prior on its .* {missing}: pass {missing}=\.\.\., {example}"):
            ab.wiener(f"{L} ~ x", frame(), ab.Priors(**base, **{k: v for k, v in OWN.items() if k != missing}))
    wrong = [("bs", ab.Normal(0, 1), r"prior for 'bs' must be a positive-support prior \(brms's real<lower=0> bs\)"),
             ("bias", ab.Gamma(1, 1), r"prior for 'bias' must be a prior on \(0, 1\) \(brms's real<lower=0,upper=1> bias\)"),
             ("ndt", ab.HalfStudentT(3, 1), r"prior for 'ndt' must be a prior truncated to \(0, min\(y\)\)"),
             ("bs", {"intercept": ab.Normal(0, 1)}, r"a mapping is the form of the priors of bs's own formula.*divergence D151\)"),
             ("sigma", ab.HalfNormal(1), r"the wiener family has no residual SD; do not pass sigma\. Its boundary separation is bs")]
    for name, prior, needle in wrong:
        with pytest.raises(ab.PriorError, match=needle):
            ab.wiener(f"{L} ~ x", frame(), ab.Priors(**base, **{**OWN, name: prior}))
    with pytest.raises(ab.PriorError, match=r"bs has its own formula 'bs ~ w': its priors are a mapping"):
        ab.wiener([f"{L} ~ x", "bs ~ w"], frame(), ab.Priors(**base, **OWN))
    for name in ("bs", "bias"):
        with pytest.raises(ab.PriorError, match=rf"the gaussian family has no (boundary separation|relative starting point); do not pass {name}"):
            ab.gaussian("x ~ w", frame(), ab.Priors(**base, sigma=ab.HalfNormal(1), **{name: OWN[name]}))
    assert build(ndt=ab.Uniform(0, "min_Y")).priors.ndt == ab.Uniform(0, "min_Y")
    assert (BS.support, NDT.support, BIAS.support) == ("positive", "below_min_y", "unit")
    assert {e.divergence for e in WIENER.extra_parameters} == {"D151"}


def test_value_point_coordinates_and_refusals():
    m = build()
    top = frame().rt.min()
    p = m.value_point(intercept=0.1, b=[0.2], bs=1.3, ndt=0.25 * top, bias=0.3)
    np.testing.assert_allclose([p["bs_log__"], p["ndt_interval__"], p["bias_logodds__"]],
                               [math.log(1.3), math.log(0.25 / 0.75), math.log(0.3 / 0.7)], rtol=1e-14)
    for kwargs, needle in (({"bs": 0.0}, r"bs \(the boundary separation of the wiener family\) must be positive"),
                           ({"bias": 1.0}, r"bias \(the relative starting point of the wiener family\) must lie strictly between 0 and 1"),
                           ({"ndt": top}, r"ndt \(the non-decision time of the wiener family\) must lie strictly between 0 and min\(y\)"),
                           ({"bias": None}, r"bias \(the relative starting point\) is required for the wiener family"),
                           ({"kappa": 2.0}, r"the wiener family has no kappa parameter; omit kappa \(its boundary separation is bs\)"),
                           ({"sigma": 1.0}, r"sigma, shape and phi do not apply to the wiener family")):
        with pytest.raises(ValueError, match=needle):
            m.value_point(**{"intercept": 0.1, "b": [0.2], "bs": 1.3, "ndt": 0.2, "bias": 0.3, **kwargs})
    comp = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
    mix = ab.mixture("x ~ w", frame(), ab.Priors(components=[comp, comp], theta=ab.Dirichlet(2)), families=["gaussian", "gaussian"])
    with pytest.raises(ValueError, match=r"alpha, beta, ndt, zoi, coi, kappa, quantile, tmp_xi, bs and bias are the .* and wiener "
                                         r"families' parameters; a mixture has none"):
        mix.value_point(intercept=[0.0, 1.0], b=np.zeros((1, 2)), sigma=[1.0, 1.0], theta=[0.5, 0.5], bs=1.0)


# -- the term model and bs's formula compose with the likelihood -------------------------------------------------------------
def _natural(m, p):
    """bs (or None under its own formula), ndt and bias at a value point."""
    top = m.design.y.min()
    bs = math.exp(p["bs_log__"]) if "bs_log__" in p else None
    return bs, top / (1.0 + math.exp(-p["ndt_interval__"])), 1.0 / (1.0 + math.exp(-p["bias_logodds__"]))


def test_slopes_categorical_interactions_transforms_offset_and_gp_compose():
    """Correlated varying slopes, a combined grouping factor g:h, a declared categorical predictor and its interaction, a
    transform, an offset and a Gaussian process all reach the likelihood through the drift: the observed rows' density at a
    point is the Wiener density at the graph's linear predictor."""
    extra = dict(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "g:h": ab.HalfNormal(0.5)}, cor={"g": ab.LKJ(2)},
                 sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    m = ab.wiener(f"{L} ~ x * z + I(w^2) + offset(off) + gp(w) + (1 + x | g) + (1 | g:h)", frame(),
                  ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **OWN, **extra), categorical={"z": "p"})
    rng = np.random.default_rng(7)
    p = {k: v + 0.2 * rng.standard_normal(np.shape(v)) for k, v in m.pymc_model.initial_point().items()}
    (mu,) = graph(m, p, ["mu"])
    bs, ndt, bias = _natural(m, p)
    df = frame()
    want = wn.wiener_log_density(df.rt.to_numpy(), df.dec.to_numpy() == 1, mu.ravel(), bs, ndt, bias)
    np.testing.assert_allclose(rows_logp(m, p), want, rtol=1e-13)
    assert m.spec.gp_terms and m.design.offset is not None and "g" in m.correlation_names and {"x:z", "I(w^2)"} <= set(m.spec.terms)


def test_bs_formula_gives_each_row_its_boundary_separation():
    m = ab.wiener([f"{L} ~ x + (1 | g)", "bs ~ z + w"], frame(),
                  ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), ndt=OWN["ndt"], bias=OWN["bias"],
                            bs={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}), categorical={"z": "p"})
    assert m.design.dpars["bs"].link == "log" and m.dpar_names["bs"] == {"value": "bs", "intercept": "bs_Intercept", "b:z": "bs_z",
                                                                          "b:w": "bs_w"}
    p = m.value_point(intercept=0.2, b=[0.4], sd=0.5, z=np.linspace(-1, 1, 4), ndt=0.3, bias=0.6,
                      bs={"intercept": 0.1, "b": [0.2, -0.3, 0.4]})
    mu, bs = graph(m, p, ["mu", "bs"])
    X = m.design.dpars["bs"].X_model
    np.testing.assert_allclose(bs.ravel(), np.exp(0.1 + X @ np.array([0.2, -0.3, 0.4])), rtol=1e-14)
    df = frame()
    want = wn.wiener_log_density(df.rt.to_numpy(), df.dec.to_numpy() == 1, mu.ravel(), bs.ravel(), 0.3, 0.6)
    np.testing.assert_allclose(rows_logp(m, p), want, rtol=1e-13)


# -- predictions, draws, log_lik -----------------------------------------------------------------------------------------
def _posterior(m, n_draw=3, **pars):
    """Synthetic draws: one chain, the intercept varying, the rest constant."""
    post = {"Intercept": np.linspace(0.2, 0.6, n_draw)[None], "x": np.full((1, n_draw), 0.3)}
    for name, value in {"bs": 1.3, "ndt": 0.2, "bias": 0.4, **pars}.items():
        post[name] = np.full((1, n_draw), value)
    return az.from_dict(posterior=post)


def test_predictions_by_hand_and_refused_draws():
    m = build()
    idata = _posterior(m)
    nd = pd.DataFrame({"x": [-1.0, 0.0, 2.0]})
    eta = np.linspace(0.2, 0.6, 3)[:, None] + 0.3 * nd.x.to_numpy()[None]
    np.testing.assert_allclose(m.predict(idata, newdata=nd, kind="linear")["eta"].to_numpy()[0], eta, rtol=1e-14)
    mean = m.predict(idata, newdata=nd, kind="mean")["mean"].to_numpy()[0]
    np.testing.assert_allclose(mean, 0.2 + np.vectorize(ref_mean)(eta, 1.3, 0.4), rtol=1e-14)
    with pytest.raises(ValueError, match=r"kind must be 'mean' or 'linear' for the wiener family; draws of the wiener family's "
                                         r"response .* RWiener.*divergence D149"):
        m.predict(idata, newdata=nd, kind="response")
    with pytest.raises(ValueError, match=r"prior predictive draws are not offered for the wiener family: .*divergence D149"):
        m.prior_predictive(draws=5, seed=1)
    with pytest.raises(ValueError, match=r"draws of the wiener family's response"):
        pm.draw(m.pymc_model["rt"])
    with pytest.raises(ValueError, match=r"draws of the wiener family's response"):
        _draw_response(m, idata.posterior, eta, mean, nd, np.random.default_rng(0))
    nan = _posterior(m, ndt=np.nan)
    with pytest.raises(ValueError, match=r"wiener expected response time is nonfinite"):
        m.predict(nan, newdata=nd, kind="mean")


def test_predictions_with_bs_formula_read_each_rows_bs():
    m = ab.wiener([f"{L} ~ x", "bs ~ w"], frame(), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), ndt=OWN["ndt"],
                                                           bias=OWN["bias"], bs={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}))
    post = _posterior(m, n_draw=2)
    del post.posterior["bs"]
    post.posterior["bs_Intercept"] = (("chain", "draw"), np.full((1, 2), 0.1))
    post.posterior["bs_w"] = (("chain", "draw"), np.full((1, 2), 0.5))
    nd = pd.DataFrame({"x": [0.5, -0.5], "w": [-1.0, 1.0]})
    mean = m.predict(post, newdata=nd, kind="mean")["mean"].to_numpy()[0]
    eta = np.array([0.2, 0.6])[:, None] + 0.3 * nd.x.to_numpy()[None]
    bs = np.exp(0.1 + 0.5 * nd.w.to_numpy())                    # raw-scale intercept draws, as a fit's posterior holds them
    np.testing.assert_allclose(mean, 0.2 + np.vectorize(ref_mean)(eta, bs[None], 0.4), rtol=1e-14)


@pytest.mark.parametrize("formula", ["rt | dec(dec) ~ x + (1 | g)", ["rt | dec(dec) ~ x + (1 | g)", "bs ~ w"]])
def test_log_lik_is_the_models_own_pointwise_density(formula):
    distributional = isinstance(formula, list)
    bs_prior = {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)} if distributional else OWN["bs"]
    m = ab.wiener(formula, frame(), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(0.5), ndt=OWN["ndt"],
                                              bias=OWN["bias"], bs=bs_prior))
    post = _posterior(m, n_draw=2)
    effects = np.array([0.3, -0.2, 0.1, 0.0])
    coord = m.group_names["g"]["coordinate"]
    post.posterior["1|g_sigma"] = (("chain", "draw"), np.full((1, 2), 0.5))
    post.posterior["1|g"] = (("chain", "draw", coord), np.tile(effects, (1, 2, 1)))
    post.posterior = post.posterior.assign_coords({coord: list(m.design.levels)})
    bs_point = 1.3
    if distributional:
        del post.posterior["bs"]
        post.posterior["bs_Intercept"] = (("chain", "draw"), np.full((1, 2), 0.1))
        post.posterior["bs_w"] = (("chain", "draw"), np.full((1, 2), 0.5))
        w = frame().w.to_numpy()
        bs_point = {"intercept": 0.1 + 0.5 * w.mean(), "b": [0.5]}       # the formula's centred intercept, as sampled
    x = frame().x.to_numpy()
    points = [m.value_point(intercept=a + 0.3 * x.mean(), b=[0.3], sd=0.5, z=effects / 0.5, bs=bs_point, ndt=0.2, bias=0.4)
              for a in (0.2, 0.6)]
    ll = ab.log_lik(m, post).to_numpy()[0]
    for d, p in enumerate(points):
        np.testing.assert_allclose(ll[d], rows_logp(m, p), rtol=1e-12)
    assert WIENER.log_likelihood is not None and WIENER.bambi_family() is None


# -- brms's default priors (A07) ----------------------------------------------------------------------------------------------
def wide_frame(n=30, seed=21):
    """Response times on a wide scale: median 8.4, MAD 4.5."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "z": np.array(["p", "q", "r"])[np.arange(n) % 3],
                         "g": [f"l{k}" for k in np.arange(n) % 3], "rt": np.round(0.4 + rng.gamma(1.5, 6.0, n), 4),
                         "dec": (np.arange(n) % 2 == 0).astype(int)})


# brms 2.23.0's default_prior() of rt | dec(dec) ~ x + (1 | g), and of bf(..., bs ~ z), on frame() and wide_frame() (written to
# CSV and read by R 4.5.2): its rows with a prior, as (class, prior, lb, ub, dpar). brms reads rt as given (the Intercept's
# location is its median, the scale max(2.5, MAD)); bs, ndt, bias and the Intercept of bs's formula have constant defaults
BRMS_ROWS = {
    ("frame", False): [("bias", "beta(1, 1)", "0", "1", ""), ("bs", "gamma(1, 1)", "0", "", ""),
                       ("Intercept", "student_t(3, 1, 2.5)", "", "", ""), ("ndt", "uniform(0, min_Y)", "0", "min_Y", ""),
                       ("sd", "student_t(3, 0, 2.5)", "0", "", "")],
    ("frame", True): [("bias", "beta(1, 1)", "0", "1", ""), ("Intercept", "student_t(3, 1, 2.5)", "", "", ""),
                      ("ndt", "uniform(0, min_Y)", "0", "min_Y", ""), ("sd", "student_t(3, 0, 2.5)", "0", "", ""),
                      ("Intercept", "normal(-0.6, 1.3)", "", "", "bs")],
    ("wide", False): [("bias", "beta(1, 1)", "0", "1", ""), ("bs", "gamma(1, 1)", "0", "", ""),
                      ("Intercept", "student_t(3, 8.4, 4.5)", "", "", ""), ("ndt", "uniform(0, min_Y)", "0", "min_Y", ""),
                      ("sd", "student_t(3, 0, 4.5)", "0", "", "")],
    ("wide", True): [("bias", "beta(1, 1)", "0", "1", ""), ("Intercept", "student_t(3, 8.4, 4.5)", "", "", ""),
                     ("ndt", "uniform(0, min_Y)", "0", "min_Y", ""), ("sd", "student_t(3, 0, 4.5)", "0", "", ""),
                     ("Intercept", "normal(-0.6, 1.3)", "", "", "bs")],
}


@pytest.mark.parametrize("which, distributional", sorted(BRMS_ROWS))
def test_default_priors_are_brms_and_build_the_model(which, distributional):
    df = frame() if which == "frame" else wide_frame()
    formula = [f"{L} ~ x + (1 | g)", "bs ~ z"] if distributional else f"{L} ~ x + (1 | g)"
    kwargs = {"categorical": {"z": "p"}} if distributional else {}
    priors = ab.default_priors(formula, df, family="wiener", **kwargs)
    m = ab.wiener(formula, df, priors, **kwargs)
    rows = [(r["class"], gates.normalise_prior(r["prior"]), r["lb"], r["ub"], r["dpar"]) for r in m.prior_table() if r["prior"]]
    want = [(c, gates.normalise_prior(p), lb, ub, d) for c, p, lb, ub, d in BRMS_ROWS[(which, distributional)]]
    assert sorted(rows) == sorted(want)
    assert set(m.default_prior_labels) >= {"ndt", "bias", "intercept"} and ("bs:intercept" if distributional else "bs") in m.default_prior_labels
    with pytest.raises(FormulaError, match=r"Addition argument 'dec' is required"):
        ab.default_priors("rt ~ x", df, family="wiener")


# -- describe(), the target identity, the fit record ----------------------------------------------------------------------
def test_describe():
    m = build("rt | dec(dec) ~ x + (1 | g)", sd=ab.HalfNormal(1))
    text = m.describe()
    assert "AutoBayes Wiener diffusion model with response addition term(s) dec(dec)" in text
    assert "family    : wiener, identity link" in text
    assert "  decisions : dec(dec): 8 row(s) at the upper boundary (1), 16 at the lower one (0)" in text
    for e in WIENER.extra_parameters:
        assert f"  {e.coordinate:<24} {e.coordinate_meaning}" in text
    assert text.index("bs_log__") < text.index("ndt_interval__") < text.index("bias_logodds__")
    assert "  bs ~ Gamma(alpha=1, beta=1)   (boundary separation, > 0)" in text and "bs is sampled as log(bs)." in text
    assert "  bias ~ Beta(alpha=2, beta=2)   (relative starting point, in (0, 1))" in text
    assert "not brms's posterior_epred (divergence D148)" in text and "trunc()" not in text
    assert ab.Model("rt | dec(dec) ~ x + (1 | g)", frame(), m.priors, family="wiener").describe() == text
    f = ab.wiener([f"{L} ~ x", "bs ~ w"], frame(), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), ndt=OWN["ndt"],
                                                         bias=OWN["bias"], bs={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}))
    text = f.describe()
    assert "Distributional parameter bs (its own formula 'bs ~ w', log link" in text and "bs is its formula's exp(eta_bs,i)" in text
    assert "E[T_i] = (bs_i P_i(upper) - bs_i bias) / mu_i" in text and "bs_log__" not in text


SET_ORDER = 'import json; print(json.dumps({"order": list(set(("mu", "bs", "ndt", "bias")) - {"mu"})}))'   # Bambi's auxiliary_parameters
BUILD = """
import json
from tests.test_wiener import build
m = build()
print(json.dumps({"describe": m.describe(), "identity": m.target_identity, "value_vars": [v.name for v in m.pymc_model.value_vars]}))
"""


def _fresh(seed: int, script: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    return json.loads(lines[-1])


def test_describe_does_not_depend_on_the_hash_seed():
    """Bambi creates bs, ndt and bias in the iteration order of the set of their names, so PyMC's order of their coordinates
    follows ``PYTHONHASHSEED``; ``describe()`` (bs, ndt, bias: brms's order) and the target identity do not."""
    seeds = {}
    for seed in range(32):
        seeds.setdefault(tuple(_fresh(seed, SET_ORDER)["order"]), seed)
        if len(seeds) == 2:
            break
    assert len(seeds) == 2
    first, second = (_fresh(seed, BUILD) for seed in seeds.values())
    assert first["value_vars"] != second["value_vars"] and sorted(first["value_vars"]) == sorted(second["value_vars"])
    assert first["describe"] == second["describe"] and first["identity"] == second["identity"]


def test_target_identity_and_fit_record(monkeypatch):
    m = build()
    assert build().target_identity == m.target_identity
    flipped = frame().assign(dec=np.r_[1 - frame().dec[0], frame().dec.to_numpy()[1:]])
    assert build(frame_=flipped).target_identity != m.target_identity             # the decisions are part of the target
    for name, other in (("bs", ab.Gamma(2, 1)), ("bias", ab.Beta(3, 3)), ("ndt", ab.Normal(0, 0.2))):
        assert build(**{name: other}).target_identity != m.target_identity
    post = _posterior(m, n_draw=4)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert {"bs", "ndt", "bias"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    run = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a sampler run"])
    rec = ab.fit_record(m, run, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "wiener" and rec["model"]["addition_terms"] == ["dec(dec)"]
    assert {k: rec["model"]["priors"][k] for k in ("bs", "ndt", "bias")} == {k: str(v) for k, v in OWN.items()}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_bambi_family_reads_the_decisions():
    fam = wn.wiener_family(np.array([1, 0, 1]) == 1)
    assert fam.name == "wiener" and fam.likelihood.params == ["mu", "bs", "ndt", "bias"] and fam.likelihood.parent == "mu"
    assert {k: v.name for k, v in fam.link.items()} == {"mu": "identity", "bs": "log", "ndt": "log", "bias": "logit"}
    assert get_family("wiener") is WIENER is FAMILIES["wiener"] and WIENER.addition_terms == frozenset({"dec"})
