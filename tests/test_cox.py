"""brms's cox() proportional hazards model (brms-parity F05, docs/cox.md): the B-, M- and I-spline bases against SciPy's
B-splines and their antiderivatives (an independent construction) and against the frozen fixtures; brms's knot rule; bhaz()
and its canonical spelling; the likelihood against the hazard written from the mathematics (H0 by numerical integration of
h0, and the density over the boundary knots integrating to 1 - S(U)); the refusals naming every construct no frozen case
verifies; priors and value points; the term model composing with the likelihood; predictions, log_lik and loo; brms's default
priors on two frames (probed with brms 2.23's default_prior()); describe(), the target identity and the fit record; G1's
baseline hazard check. No MCMC; the brms comparison is the oracle cases cox_varint, cox_uncensored_default_priors, cox_bhaz_df,
cox_bhaz_knots and cox_varint_post."""
import copy
import json
import math
import warnings
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy import integrate
from scipy.interpolate import BSpline

import autobayes as ab
from autobayes import baseline_hazard as cox
from autobayes.families import FAMILIES, get_family
from autobayes.families.cox import COX, _draw_response, _inverse_link
from autobayes.formula import AdditionTerm, FormulaError, addition_data, parse_formula
from tests.oracle import adapter, gates
from tests.oracle import fixture as fx
from tests.test_continuous_families import graph, rows_logp

CASES = ["cox_varint", "cox_uncensored_default_priors", "cox_bhaz_df", "cox_bhaz_knots", "cox_varint_post"]


# -- the bases, from an independent construction -------------------------------------------------------------------
def scipy_bases(x, interior, boundary, degree=3):
    """The M-splines ``k B_i / (t_{i+k} - t_i)`` of SciPy's B-splines on the clamped knots and their I-splines, the
    antiderivatives from the lower boundary knot."""
    k = degree + 1
    t = np.r_[[boundary[0]] * k, interior, [boundary[1]] * k]
    n = len(t) - k
    M, I = np.empty((len(x), n)), np.empty((len(x), n))
    for i in range(n):
        spline = BSpline(t, np.eye(n)[i] * k / (t[i + k] - t[i]), degree)
        integral = spline.antiderivative()
        M[:, i], I[:, i] = spline(x), integral(x) - integral(boundary[0])
    return M, I


SETTINGS = [((), (0.0, 1.0)), ((0.4,), (0.0, 1.0)), ((0.3, 1.1, 1.7), (-0.5, 2.5)), ((2.0, 3.0, 5.0, 8.0), (1.0, 10.0))]


@pytest.mark.parametrize("interior, boundary", SETTINGS)
def test_bsplines_are_scipys_and_a_partition_of_unity(interior, boundary):
    for order in (1, 2, 4, 5):
        t = np.r_[[boundary[0]] * order, interior, [boundary[1]] * order]
        x = np.r_[np.linspace(boundary[0], boundary[1], 41), interior]
        B = cox.bsplines(x, t, order)
        n = len(t) - order
        want = np.column_stack([BSpline(t, np.eye(n)[i], order - 1)(x) for i in range(n)])
        np.testing.assert_allclose(B, want, atol=1e-15, rtol=0)
        np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-15, rtol=0)           # the upper boundary knot included
    assert np.array_equal(cox.bsplines([boundary[0] - 1.0, boundary[1] + 1.0], t, 4), np.zeros((2, len(t) - 4)))


@pytest.mark.parametrize("interior, boundary", SETTINGS)
def test_m_and_i_splines_from_their_definitions(interior, boundary):
    """M_i = k B_i / (t_{i+k} - t_i) integrates to 1 and I_i is its integral from the lower boundary knot: 0 there, 1 at the
    upper one, nondecreasing; both equal the independent construction, for cubic and quadratic bases."""
    x = np.linspace(boundary[0], boundary[1], 57)
    for degree in (3, 2):
        M, I = cox.spline_bases(x, interior, boundary, degree=degree)
        sM, sI = scipy_bases(x, np.asarray(interior), boundary, degree)
        assert M.shape == I.shape == (len(x), len(interior) + degree + 1)
        np.testing.assert_allclose(M, sM, atol=1e-13, rtol=0)
        np.testing.assert_allclose(I, sI, atol=1e-14, rtol=0)
        np.testing.assert_allclose(I[0], 0.0, atol=1e-16)
        np.testing.assert_allclose(I[-1], 1.0, atol=1e-15)
        assert (np.diff(I, axis=0) >= -1e-15).all() and (M >= 0).all()
        for i in range(M.shape[1]):
            area = integrate.quad(lambda u: cox.spline_bases([u], interior, boundary, degree=degree)[0][0, i], *boundary,
                                  points=interior or None, limit=200)[0]
            assert abs(area - 1.0) < 1e-10


def _knots(case):
    """The interior and boundary knots each case's note records."""
    return {"cox_varint": ([2.12045], (0.481216, 7.131184)), "cox_bhaz_df": ([1.43395, 2.2069, 3.144575], (0.589992, 11.047608)),
            "cox_bhaz_knots": ([1.5, 3.0], (0.0, 6.0)), "cox_uncensored_default_priors": ([2.2557], (0.136392, 10.989208)),
            "cox_varint_post": ([2.42055], (0.406036, 12.442164))}[case]


@pytest.mark.parametrize("case_id", CASES)
def test_bases_and_knots_are_brms_on_the_frozen_cases(case_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = gates.Bundle(case_id)
    block, sd = b.model.design.bhaz, b.fixture.standata
    interior, boundary = _knots(case_id)
    np.testing.assert_allclose(block.knots, interior, atol=5e-7)
    np.testing.assert_allclose(block.boundary, boundary, atol=5e-7)
    assert np.max(np.abs(block.Zbhaz - sd["Zbhaz"])) <= 1e-15 and np.max(np.abs(block.Zcbhaz - sd["Zcbhaz"])) <= 1e-15
    assert block.K == int(sd["Kbhaz"]) and block.labels == [str(k) for k in range(1, block.K + 1)] == b.fixture.colnames("Zbhaz")
    np.testing.assert_array_equal(block.event, sd["cens"] == 0 if "cens" in sd else np.ones(block.Zbhaz.shape[0], dtype=bool))
    gates.check_baseline_hazard(b)


# -- bhaz() and the knots --------------------------------------------------------------------------------------------
def test_bhaz_terms_parse_to_their_canonical_spelling():
    cases = {"bhaz()": ("bhaz()", None, None, None, 5), "bhaz(df = 7)": ("bhaz(df = 7)", 7, None, None, 7),
             "bhaz(df = 4.0)": ("bhaz(df = 4)", 4, None, None, 4),
             "bhaz(knots = c(1.5, 3), Boundary.knots = c(0, 6))": ("bhaz(knots = c(1.5, 3), Boundary.knots = c(0, 6))", None,
                                                                    (1.5, 3.0), (0.0, 6.0), 6),
             "bhaz(Boundary.knots = c(-1, 1e1), df = 6)": ("bhaz(df = 6, Boundary.knots = c(-1, 10))", 6, None, (-1.0, 10.0), 6),
             "bhaz(knots = 2.5)": ("bhaz(knots = c(2.5))", None, (2.5,), None, 5)}
    for written, (term, df, knots, boundary, K) in cases.items():
        spec = parse_formula(f"t | cens(c) + {written} ~ x", family="cox")
        a = spec.addition("bhaz")
        assert (a.term, a.df, a.knots, a.boundary, a.columns) == (term, df, knots, boundary, ())
        assert cox.baseline_term(spec) is a and cox.n_bases(spec) == K
        assert spec.canonical_formula.startswith(f"t | cens(c) + {term} ~")
    spec = parse_formula("t ~ x", family="cox")
    assert cox.baseline_term(spec) is cox.DEFAULT_BHAZ and cox.n_bases(spec) == 5 and cox.DEFAULT_BHAZ.term == "bhaz()"
    assert addition_data(parse_formula("t | bhaz(df = 6) ~ x", family="cox"), frame()) == {}   # bhaz() reads no column


BHAZ_REFUSALS = [
    ("bhaz(5)", r"bhaz\(\) takes named arguments, df = , knots = and Boundary.knots = ; got '5'"),
    ("bhaz(gr = h)", r"a stratified baseline hazard, bhaz\(gr = \).*divergence D155"),
    ("bhaz(degree = 2)", r"bhaz\(degree = \) is not offered: the baseline hazard is brms's cubic M-spline basis.*D155"),
    ("bhaz(intercept = FALSE)", r"bhaz\(intercept = \) is not offered.*D155"),
    ("bhaz(df = 5, df = 6)", r"bhaz\(\) names df twice"),
    ("bhaz(df = 5.5)", r"bhaz\(df = \) is a whole number of bases, got '5.5'"),
    ("bhaz(df = 3)", r"bhaz\(df = 3\) gives too few bases.*brms: 'The specified df was too small.'"),
    ("bhaz(knots = c(3, 1.5))", r"the interior knots must be strictly increasing, got \[3.0, 1.5\]"),
    ("bhaz(knots = c(a, 2))", r"knots must be a number or c\(\) of numbers, got 'c\(a, 2\)'"),
    ("bhaz(knots = 1e999)", r"knots must be finite numbers"),
    ("bhaz(df = 6, knots = c(1, 2))", r"bhaz\(\) takes df or knots, not both: brms ignores df.*6 bases; docs/cox.md, divergence D152"),
    ("bhaz(Boundary.knots = c(1, 0))", r"Boundary.knots are two numbers, the lower one first, got \[1.0, 0.0\]"),
    ("bhaz(Boundary.knots = 3)", r"Boundary.knots are two numbers"),
    ("bhaz(knots = c(1, 7), Boundary.knots = c(0, 6))", r"the interior knots \[1.0, 7.0\] must lie strictly inside the boundary"),
]


@pytest.mark.parametrize("written, needle", BHAZ_REFUSALS)
def test_bhaz_refusals(written, needle):
    with pytest.raises(FormulaError, match=needle):
        parse_formula(f"t | {written} ~ x", family="cox")


def test_knots_follow_brms_rule():
    y = np.array([0.7, 1.9, 0.3, 4.2, 2.5, 1.1, 3.3, 0.9])
    lo, hi = 0.3, 4.2
    interior, boundary = cox.baseline_knots(y, cox.DEFAULT_BHAZ)
    np.testing.assert_allclose(boundary, (lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)), rtol=1e-15)
    np.testing.assert_allclose(interior, [np.median(y)], rtol=1e-15)                     # df = 5: the median
    interior, _ = cox.baseline_knots(y, AdditionTerm("bhaz", df=7))
    s = np.sort(y)                                                                         # R's type 7 by hand: x[h] + (h - [h]) dx
    want = [s[int(h)] + (h - int(h)) * (s[int(h) + 1] - s[int(h)]) for h in (7 * 0.25, 7 * 0.5, 7 * 0.75)]
    np.testing.assert_allclose(interior, want, rtol=1e-15)
    assert cox.baseline_knots(y, AdditionTerm("bhaz", df=4))[0].size == 0                 # no interior knot
    interior, boundary = cox.baseline_knots(y, AdditionTerm("bhaz", boundary=(0.0, 5.0)))
    assert boundary == (0.0, 5.0) and interior.tolist() == [float(np.median(y))]           # Boundary.knots keeps the median
    interior, boundary = cox.baseline_knots(y, AdditionTerm("bhaz", knots=(1.0, 2.0), boundary=(0.0, 5.0)))
    assert interior.tolist() == [1.0, 2.0] and boundary == (0.0, 5.0)
    with pytest.raises(FormulaError, match=r"every time is 2.0: brms's boundary knots.*need two different times"):
        cox.baseline_knots(np.full(5, 2.0), cox.DEFAULT_BHAZ)
    with pytest.raises(FormulaError, match=r"interior knots \[1.0, 1.0, 1.0\] of the baseline hazard are not strictly increasing.*D152"):
        cox.baseline_knots(np.r_[np.ones(12), 0.5, 3.0], AdditionTerm("bhaz", df=7))       # tied times, equal quartiles
    with pytest.raises(FormulaError, match=r"not strictly increasing inside the boundary knots \[2.0, 5.0\]"):
        cox.baseline_knots(y, AdditionTerm("bhaz", boundary=(2.0, 5.0)))                   # the median below the lower knot


# -- frames and models ------------------------------------------------------------------------------------------------
def frame(n=24, seed=5):
    """Survival times with right-censoring (every fifth row from the third), a numeric and a categorical predictor, two
    grouping factors and an offset (the frame of the brms probe below)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "w": np.round(rng.uniform(-1, 1, n), 4),
                         "z": np.array(["p", "q", "r"])[np.arange(n) % 3], "g": [f"l{k}" for k in np.arange(n) % 4],
                         "h": np.array(["u", "v"])[(np.arange(n) // 4) % 2],
                         "t": np.round(0.2 + rng.weibull(1.4, n) * 2.0, 4), "c": (np.arange(n) % 5 == 2).astype(int),
                         "off": np.round(rng.uniform(-0.5, 0.5, n), 4)})


OWN = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sbhaz=ab.Dirichlet(1))


def build(formula="t | cens(c) ~ x", frame_=None, **extra):
    return ab.cox(formula, frame() if frame_ is None else frame_, ab.Priors(**{**OWN, **extra}))


def simplex(K, seed=1):
    v = np.random.default_rng(seed).uniform(0.5, 2.0, K)
    return v / v.sum()


def hazard_rows(t, cens, eta, interior, boundary, s):
    """Each row's log likelihood from the mathematics: h0 from SciPy's M-splines, H0 by numerical integration of h0."""
    M, _ = scipy_bases(np.asarray(t, dtype=float), np.asarray(interior), boundary)
    h0 = M @ s
    H0 = np.array([integrate.quad(lambda u: (scipy_bases(np.array([u]), np.asarray(interior), boundary)[0] @ s)[0], boundary[0], ti,
                                  points=[k for k in interior if k < ti] or None, limit=200, epsabs=1e-14, epsrel=1e-13)[0] for ti in t])
    return np.where(cens == 0, np.log(h0) + eta - H0 * np.exp(eta), -H0 * np.exp(eta))


@pytest.mark.parametrize("formula", ["t | cens(c) ~ x", "t ~ x", "t | cens(c) + bhaz(df = 7) ~ x",
                                     "t | bhaz(knots = c(1, 2.5), Boundary.knots = c(0, 9)) ~ x"])
def test_likelihood_rows_are_the_hazard_mathematics(formula):
    m = build(formula)
    df, block = frame(), m.design.bhaz
    s = simplex(block.K)
    p = m.value_point(intercept=0.3, b=[0.6], sbhaz=s)
    x = df.x.to_numpy()
    eta = 0.3 + 0.6 * (x - x.mean())
    cens = df.c.to_numpy() if "cens" in formula else np.zeros(len(df))
    want = hazard_rows(df.t.to_numpy(), cens, eta, block.knots, block.boundary, s)
    np.testing.assert_allclose(rows_logp(m, p), want, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(cox.cox_log_lik(eta, block.Zbhaz, block.Zcbhaz, s, block.event), want, rtol=1e-11, atol=1e-12)
    from pymc.sampling.jax import get_jaxified_logp
    jf = get_jaxified_logp(m.pymc_model)
    assert abs(float(jf([p[v.name] for v in m.pymc_model.value_vars])) - float(m.pymc_model.compile_logp()(p))) < 1e-9


def test_density_over_the_boundary_knots_integrates_to_one_minus_the_survival_there():
    """f(t) = h0(t) e^eta exp(-H0(t) e^eta) on (L, U) has mass 1 - S(U) = 1 - exp(-e^eta), H0(U) = 1 (sbhaz a simplex)."""
    interior, boundary = [0.8, 1.6], (0.1, 3.0)
    s = simplex(6, seed=4)
    for eta in (-1.2, 0.0, 0.9):
        def f(u):
            M, I = cox.spline_bases([u], interior, boundary)
            return math.exp(cox.cox_log_lik(np.array([eta]), M, I, s, np.array([True]))[0])
        mass = integrate.quad(f, *boundary, points=interior, limit=200, epsabs=1e-13)[0]
        assert abs(mass - (1.0 - math.exp(-math.exp(eta)))) < 1e-10
    draws = cox.cox_log_lik(np.zeros((2, 3, 4)), *cox.spline_bases(np.linspace(0.2, 2.9, 4), interior, boundary),
                            np.tile(s, (2, 3, 1)), np.array([True, False, True, False]))
    assert draws.shape == (2, 3, 4)


def test_slopes_categorical_interactions_transforms_offset_gp_and_uncentred_designs_compose():
    """Correlated varying slopes, a combined grouping factor g:h, a declared categorical predictor and its interaction, a
    transform, an offset and a Gaussian process reach the likelihood through the linear predictor, centred or not: the observed
    rows' density at a point is the cox density at the graph's linear predictor."""
    extra = dict(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "g:h": ab.HalfNormal(0.5)}, cor={"g": ab.LKJ(2)},
                 sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    rng = np.random.default_rng(7)
    for formula, kwargs in (("t | cens(c) ~ x * z + I(w^2) + offset(off) + gp(w) + (1 + x | g) + (1 | g:h)", {"categorical": {"z": "p"}}),
                            ("t | cens(c) ~ x + (1 + x | g) + (1 | g:h)", {"center_predictors": False})):
        priors = ab.Priors(**OWN, **{k: v for k, v in extra.items() if k not in ("sdgp", "lscale") or "gp(" in formula})
        m = ab.cox(formula, frame(), priors, **kwargs)
        p = {k: v + 0.2 * rng.standard_normal(np.shape(v)) for k, v in m.pymc_model.initial_point().items()}
        (eta, s) = graph(m, p, ["log_mu", "sbhaz"])
        block = m.design.bhaz
        np.testing.assert_allclose(rows_logp(m, p), cox.cox_log_lik(eta.ravel(), block.Zbhaz, block.Zcbhaz, s, block.event), rtol=1e-12)
    assert not m.design.centered and m.design.bhaz.event.sum() == 19


# -- refusals -------------------------------------------------------------------------------------------------------------
L = "t | cens(c)"
REFUSALS = [   # (construct, formula, extra priors, frame changes, error, message)
    ("interval censoring", "t | cens(c, t2) ~ x", {}, {"t2": 9.0}, FormulaError,
     r"'cens\(c, t2\)': interval censoring of a cox model is not offered: brms 2.23 evaluates both ends.*-inf.*D153"),
    ("trunc()", "t | trunc(lb = 0.1) ~ x", {}, {}, FormulaError,
     r"'trunc\(lb = 0.1\)': the truncation of a cox model is not offered: brms 2.23 normalises.*17.13382.*0.3908019.*D153"),
    ("weights()", f"{L} + weights(w2) ~ x", {}, {"w2": 1.5}, FormulaError, r"'weights\(w2\)': weights\(\) of a cox model.*D155"),
    ("left censoring", f"{L} ~ x", {}, {"c": [-1] + [0] * 23}, FormulaError,
     r"row 0 of 'cens\(c\)' is left-censored \(-1\); the cox family takes right censoring only.*D155"),
    ("a time outside the boundary knots", "t | bhaz(Boundary.knots = c(0.5, 9)) ~ x", {}, {}, FormulaError,
     r"the time 0.\d+ of row \d+ lies outside the boundary knots \[0.5, 9.0\] of 'bhaz\(Boundary.knots = c\(0.5, 9\)\)'.*D152"),
    ("a time of 0", f"{L} ~ x", {}, {"t": [0.0] + [1.0] * 23}, FormulaError,
     r"response 't' has a non-positive value .*; the cox family needs strictly positive responses"),
    ("mi() on the response", "t | mi() ~ x", {}, {}, FormulaError, r"'mi\(\)' is not available for the cox family"),
    ("dec()", "t | dec(c) ~ x", {}, {}, FormulaError, r"'dec\(c\)' is not available for the cox family"),
    ("se()", "t | se(w2) ~ x", {}, {"w2": 0.3}, FormulaError, r"'se\(w2\)' is not available for the cox family"),
    ("rate()", "t | rate(w2) ~ x", {}, {"w2": 2.0}, FormulaError, r"'rate\(w2\)' is not available for the cox family"),
    ("thres()", "t | thres(gr = h) ~ x", {}, {}, FormulaError, r"thres\(\) in the cox family is not verified here"),
    ("a formula on sigma", [f"{L} ~ x", "sigma ~ w"], {}, {}, FormulaError, r"the cox family has no parameter 'sigma'"),
    ("mo()", f"{L} ~ mo(o)", {"simo": ab.Dirichlet(1)}, {"o": np.arange(24) % 4}, FormulaError,
     r"the monotonic term 'mo\(o\)' is not available for the cox family"),
    ("me()", f"{L} ~ me(x, sdx)", {}, {"sdx": 0.3}, FormulaError, r"the measurement-error term 'me\(x, sdx\)' is not available for the cox"),
    ("ar()", f"{L} ~ x + ar(p = 1)", {}, {}, FormulaError, r"the autocorrelation term 'ar\(p = 1\)' is not available for the cox family"),
    ("mvbind()", "mvbind(t, x) ~ w", {}, {}, FormulaError, r"a multivariate cox model is not offered"),
    ("mm()", f"{L} ~ x + (1 | mm(g, h))", {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"the multi-membership term '\(1 \| mm\(g, h\)\)' next to the addition term 'cens\(c\)' is not supported"),
    ("mm() without cens()", "t ~ x + (1 | mm(g, h))", {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"the multi-membership term '\(1 \| mm\(g, h\)\)' is not available for the cox family"),
    ("gr(by = )", "t ~ x + (1 | gr(g, by = h))", {"sd": ab.HalfNormal(1)}, {}, FormulaError,
     r"the term with gr\(by = \) .* is not available for the cox family"),
    ("cs()", f"{L} ~ cs(x)", {}, {}, FormulaError, r"brms's category-specific effect cs\(\) is not supported for the cox family"),
    ("horseshoe()", f"{L} ~ x", {"b": ab.Horseshoe(3)}, {}, ab.PriorError, r"Horseshoe\(\) on b is offered for the Gaussian family only.*cox"),
    ("a column named sbhaz", "t ~ sbhaz", {}, {"sbhaz": 1.0}, FormulaError, r"generated variable/coordinate name collision: \['sbhaz'\]"),
]


@pytest.mark.parametrize("construct", [r[0] for r in REFUSALS])
def test_other_constructs_are_refused_by_name(construct):
    _, formula, extra, changes, error, needle = next(r for r in REFUSALS if r[0] == construct)
    df = frame().assign(**changes)
    if construct == "a column named sbhaz":
        df["sbhaz"] = np.linspace(0, 1, 24)
    with pytest.raises(error, match=needle):
        build(formula, df, **extra)


def test_links_mixtures_nonlinear_models_other_families_and_post_processing_are_refused():
    df = frame()
    with pytest.raises(ValueError, match=r"the cox family has the log link only; link='identity' is not available"):
        ab.Model(f"{L} ~ x", df, ab.Priors(**OWN), family="cox", link="identity")
    with pytest.raises(ValueError, match=r"a cox mixture component is not offered.*D52"):
        ab.mixture("t ~ x", df, ab.Priors(components=[ab.Priors(**OWN)] * 2, theta=ab.Dirichlet(2)), families=["cox", "cox"])
    with pytest.raises(FormulaError, match=r"the cox family is not offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel(["t ~ a * x", "a ~ 1"], df, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, sbhaz=ab.Dirichlet(1)), family="cox")
    gauss = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
    with pytest.raises(FormulaError, match=r"'bhaz\(df = 5\)' is not available for the gaussian family; brms's bhaz\(\) is verified here "
                                           r"for the cox family"):
        ab.gaussian("t | bhaz(df = 5) ~ x", df, ab.Priors(**gauss))
    m = build()
    idata = _posterior(m)
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a cox model is not offered: it needs the expected response; kind must "
                                         r"be 'linear' or 'relative_hazard' for the cox family.*D154"):
        ab.conditional_effects(m, idata, "x")
    with pytest.raises(ValueError, match=r"bayes_R2\(\) is not offered for the cox family: it needs the expected response.*D154"):
        ab.bayes_R2(m, idata)
    with pytest.raises(ValueError, match=r"prior_draws\(\) of a model with the cox family's baseline hazard simplex sbhaz is not offered"):
        ab.prior_draws(m)


# -- priors and value points ------------------------------------------------------------------------------------------------
def test_sbhaz_takes_a_dirichlet_prior_for_the_cox_family_only():
    base = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"the cox family needs a prior on its baseline hazard simplex sbhaz: pass sbhaz=ab.Dirichlet\(1\)"):
        ab.cox(f"{L} ~ x", frame(), ab.Priors(**base))
    with pytest.raises(ab.PriorError, match=r"prior for 'sbhaz' must be a Dirichlet prior on the simplex of the baseline hazard's weights"):
        ab.cox(f"{L} ~ x", frame(), ab.Priors(**base, sbhaz=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match=r"gives 3 concentrations for a simplex of 5 steps.*Expected input of length 5"):
        ab.cox(f"{L} ~ x", frame(), ab.Priors(**base, sbhaz=ab.Dirichlet(1, 2, 3)))
    with pytest.raises(ab.PriorError, match=r"sbhaz is the prior of the cox family's baseline hazard simplex \(brms class sbhaz\); the "
                                            r"gaussian family has none"):
        ab.gaussian("t ~ x", frame(), ab.Priors(**base, sigma=ab.HalfNormal(1), sbhaz=ab.Dirichlet(1)))
    m = build("t | cens(c) + bhaz(df = 6) ~ x", sbhaz=ab.Dirichlet(1, 2, 3, 4, 5, 6))
    p = m.value_point(intercept=0.0, b=[0.0], sbhaz=np.full(6, 1 / 6))
    from autobayes.priors import dirichlet_concentration
    want = float(np.sum((np.arange(1, 7) - 1) * np.log(1 / 6)) + math.lgamma(21) - sum(math.lgamma(k) for k in range(1, 7)))
    assert abs(float(m.pymc_model.compile_logp(vars=[m.pymc_model["sbhaz"]], jacobian=False)(p)) - want) < 1e-10
    assert dirichlet_concentration(m.priors.sbhaz, 6).tolist() == [1, 2, 3, 4, 5, 6]


def test_value_point_coordinates_and_refusals():
    m = build()
    s = simplex(5)
    p = m.value_point(intercept=0.1, b=[0.2], sbhaz=s)
    rest = np.cumsum(s[::-1])[::-1]
    np.testing.assert_allclose(p["sbhaz_stickbreaking__"], np.log(s[:-1]) - np.log(rest[1:]) + np.log(np.arange(4, 0, -1)), rtol=1e-14)
    for kwargs, needle in (({"sbhaz": None}, r"sbhaz \(the baseline hazard's 5 weights, a simplex\) is required for the cox family"),
                           ({"sbhaz": s[:4] / s[:4].sum()}, r"sbhaz must be a simplex of 5 finite entries"),
                           ({"sbhaz": np.r_[1.2, -0.2, 0.0, 0.0, 0.0]}, r"sbhaz must have strictly positive entries"),
                           ({"sigma": 1.0}, r"sigma, shape and phi do not apply to the cox family")):
        with pytest.raises(ValueError, match=needle):
            m.value_point(**{"intercept": 0.1, "b": [0.2], "sbhaz": s, **kwargs})
    g = ab.gaussian("t ~ x", frame(), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"sbhaz is the cox family's baseline hazard simplex; the gaussian family has none"):
        g.value_point(intercept=0.0, b=[0.0], sigma=1.0, sbhaz=s)


class _Graph:
    """A PyMC model with some attributes replaced, for check_graph's refusals."""

    def __init__(self, pmm, **override):
        self._pmm, self._override = pmm, override

    def __getattr__(self, name):
        return self._override.get(name, getattr(self._pmm, name))

    def __getitem__(self, key):
        return self._pmm[key]


def test_check_graph_refuses_a_wrong_simplex():
    m = build()
    pmm = m.pymc_model
    cox.check_graph(m)
    with pytest.raises(RuntimeError, match=r"'sbhaz' is not a free simplex of 6 entries in the graph"):
        cox.check_graph(SimpleNamespace(pymc_model=pmm, design=SimpleNamespace(bhaz=SimpleNamespace(K=6))))
    with pytest.raises(RuntimeError, match=r"'sbhaz' is not sampled on Stan's 4 stick-breaking coordinates"):
        cox.check_graph(SimpleNamespace(pymc_model=_Graph(pmm, rvs_to_transforms={pmm["sbhaz"]: None}), design=m.design))
    with pytest.raises(RuntimeError, match=r"PyMC dims of 'sbhaz' are not \('sbhaz_dim',\) with coordinates 1..5"):
        cox.check_graph(SimpleNamespace(pymc_model=_Graph(pmm, named_vars_to_dims={}), design=m.design))


# -- predictions, draws, log_lik ----------------------------------------------------------------------------------------------
def _posterior(m, n_draw=3, sbhaz=None):
    """Synthetic draws: one chain, the intercept varying, the slope and the simplex constant."""
    K = m.design.bhaz.K
    s = simplex(K) if sbhaz is None else sbhaz
    post = {"Intercept": np.linspace(0.2, 0.6, n_draw)[None], "x": np.full((1, n_draw), 0.3), "sbhaz": np.tile(s, (1, n_draw, 1))}
    return az.from_dict(posterior=post, dims={"sbhaz": ["sbhaz_dim"]}, coords={"sbhaz_dim": m.design.bhaz.labels})


def test_predictions_by_hand_and_refused_draws():
    m = build()
    idata = _posterior(m)
    nd = pd.DataFrame({"x": [-1.0, 0.0, 2.0]})                           # no time column: the linear predictor needs none
    eta = np.linspace(0.2, 0.6, 3)[:, None] + 0.3 * nd.x.to_numpy()[None]
    np.testing.assert_allclose(m.predict(idata, newdata=nd, kind="linear")["eta"].to_numpy()[0], eta, rtol=1e-14)
    np.testing.assert_allclose(m.predict(idata, newdata=nd, kind="relative_hazard")["mu"].to_numpy()[0], np.exp(eta), rtol=1e-14)
    for kind in ("mean", "response"):
        with pytest.raises(ValueError, match=r"kind must be 'linear' or 'relative_hazard' for the cox family: brms refuses posterior_epred "
                                             r"and posterior_predict.*D154"):
            m.predict(idata, newdata=nd, kind=kind)
    with pytest.raises(ValueError, match=r"prior predictive draws are not offered for the cox family"):
        m.prior_predictive(draws=5, seed=1)
    with pytest.raises(ValueError, match=r"draws of a cox model's response are not offered.*D154"):
        pm.draw(m.pymc_model["t"])
    with pytest.raises(ValueError, match=r"draws of a cox model's response are not offered"):
        _draw_response(m, idata.posterior, eta, None, nd, np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"draws of a cox model's response are not offered"):
        cox.cox_family(m.design.bhaz, np.ones(5)).likelihood.dist.dist(log_mu=0.0)
    with pytest.raises(ValueError, match=r"cox relative hazard overflow/nonfinite prediction"):
        _inverse_link(np.array([800.0]))


@pytest.mark.parametrize("formula", ["t | cens(c) ~ x + (1 | g)", "t ~ x + (1 | g)"])
def test_log_lik_is_the_models_own_pointwise_density_and_loo_runs(formula):
    m = build(formula, sd=ab.HalfNormal(0.5))
    post = _posterior(m, n_draw=2)
    effects = np.array([0.3, -0.2, 0.1, 0.0])
    coord = m.group_names["g"]["coordinate"]
    post.posterior["1|g_sigma"] = (("chain", "draw"), np.full((1, 2), 0.5))
    post.posterior["1|g"] = (("chain", "draw", coord), np.tile(effects, (1, 2, 1)))
    post.posterior = post.posterior.assign_coords({coord: list(m.design.levels)})
    x = frame().x.to_numpy()
    s = np.asarray(post.posterior["sbhaz"])[0, 0]
    points = [m.value_point(intercept=a + 0.3 * x.mean(), b=[0.3], sd=0.5, z=effects / 0.5, sbhaz=s) for a in (0.2, 0.6)]
    ll = ab.log_lik(m, post).to_numpy()[0]
    for d, p in enumerate(points):
        np.testing.assert_allclose(ll[d], rows_logp(m, p), rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # two draws: Pareto k is large, as it should be
        assert np.isfinite(ab.loo(m, post).estimates.loc["elpd_loo", "Estimate"])


# -- G1, the frozen points ------------------------------------------------------------------------------------------------------
def _with_standata(b, standata, model=None):
    fixture = copy.copy(b.fixture)
    fixture.__dict__["standata"] = standata
    return SimpleNamespace(fixture=fixture, model=model or b.model, cmap=b.cmap, gmap=b.gmap)


def test_g1_checks_the_baseline_hazard():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = gates.Bundle("cox_varint")
    sd = b.fixture.standata
    gates.check_design(b)
    bad = [({**sd, "Zbhaz": sd["Zbhaz"] * (1 + 1e-9)}, r"Zbhaz differs from the AutoBayes bases \(\(36, 5\) vs \(36, 5\), by"),
           ({**sd, "Zcbhaz": sd["Zcbhaz"][:, :4]}, r"Zcbhaz differs from the AutoBayes bases \(\(36, 4\) vs \(36, 5\)\)"),
           ({**sd, "Kbhaz": np.array(6)}, r"brms has Kbhaz = 6; the AutoBayes baseline hazard has 5 bases"),
           ({**sd, "con_sbhaz": np.full(5, 2.0)}, r"brms con_sbhaz is .*; the prior of sbhaz gives"),
           ({k: v for k, v in sd.items() if k != "Kbhaz"}, r"the AutoBayes model has a baseline hazard; brms's Stan data lack \['Kbhaz'\]")]
    for standata, needle in bad:
        with pytest.raises(gates.OracleMismatch, match=needle):
            gates.check_baseline_hazard(_with_standata(b, standata))
    g = ab.gaussian("ypos2 ~ x + (1 | g)", b.data, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sd=ab.HalfNormal(1),
                                                             sigma=ab.HalfNormal(1)))
    with pytest.raises(gates.OracleMismatch, match=r"brms has the baseline hazard data \['Zbhaz', 'Zcbhaz', 'Kbhaz', 'con_sbhaz'\]; the "
                                                    r"AutoBayes model has no baseline hazard"):
        gates.check_baseline_hazard(_with_standata(b, sd, model=g))
    assert "sbhaz" in gates.prior_names(b.model) and "sbhaz" not in gates.prior_names(g)


# -- brms's default priors (A07) ----------------------------------------------------------------------------------------------
# brms 2.23.0's default_prior() of each formula with family = cox() on frame() and on it in other units (the times times 30),
# written to CSV and read by R 4.5.2: its rows with a prior, as (class, prior, lb). brms reads log(t) for the Intercept's
# location (0.7, and 4.1 = 0.7 + log 30 rounded); the scale stays 2.5 here; sbhaz's default is dirichlet(1)
BRMS_ROWS = {
    ("frame", True): [("Intercept", "student_t(3, 0.7, 2.5)", ""), ("sbhaz", "dirichlet(1)", ""), ("sd", "student_t(3, 0, 2.5)", "0")],
    ("frame", False): [("Intercept", "student_t(3, 0.7, 2.5)", ""), ("sbhaz", "dirichlet(1)", "")],
    ("days", True): [("Intercept", "student_t(3, 4.1, 2.5)", ""), ("sbhaz", "dirichlet(1)", ""), ("sd", "student_t(3, 0, 2.5)", "0")],
    ("days", False): [("Intercept", "student_t(3, 4.1, 2.5)", ""), ("sbhaz", "dirichlet(1)", "")],
}


@pytest.mark.parametrize("which, grouped", sorted(BRMS_ROWS))
def test_default_priors_are_brms_and_build_the_model(which, grouped):
    df = frame() if which == "frame" else frame().assign(t=frame().t * 30)
    formula = "t | cens(c) ~ x + (1 | g)" if grouped else "t ~ x"
    priors = ab.default_priors(formula, df, family="cox")
    m = ab.cox(formula, df, priors)
    rows = [(r["class"], gates.normalise_prior(r["prior"]), r["lb"]) for r in m.prior_table() if r["prior"]]
    assert sorted(rows) == sorted((c, gates.normalise_prior(p), lb) for c, p, lb in BRMS_ROWS[(which, grouped)])
    assert {"intercept", "sbhaz"} <= set(m.default_prior_labels) and priors.sbhaz == ab.Dirichlet(1)


# -- describe(), the target identity, the fit record ----------------------------------------------------------------------------
def test_describe():
    m = build("t | cens(c) ~ x + (1 | g)", sd=ab.HalfNormal(1))
    text = m.describe()
    assert "AutoBayes Cox proportional hazards model with response addition term(s) cens(c)" in text
    assert "family    : cox, log link" in text and "  censoring : c: 0 left-censored row(s)" in text
    assert "  sbhaz ~ Dirichlet(a=1)   (brms class sbhaz; concentrations 1, 1, 1, 1, 1: the weights of the 5 baseline hazard bases" in text
    assert "brms's defaults, df = 5" in text and "19 event(s)" in text and "5 right-censored row(s)" in text
    assert "  sbhaz_stickbreaking__    sbhaz[1..5] on Stan's stick-breaking simplex coordinates (4 entries; brms's sbhaz); dims sbhaz_dim" in text
    assert "kind='relative_hazard' mu" in text and "cens(): a left-censored row" not in text
    assert ab.Model("t | cens(c) ~ x + (1 | g)", frame(), m.priors, family="cox").describe() == text
    text = build("t | bhaz(knots = c(1, 2.5), Boundary.knots = c(0, 9)) ~ x").describe()
    assert "of bhaz(knots = c(1, 2.5), Boundary.knots = c(0, 9)): boundary knots 0 and 9," in text and "interior knot(s) 1, 2.5" in text


def test_target_identity_and_fit_record(monkeypatch):
    m = build()
    assert build().target_identity == m.target_identity
    for other in (build("t | cens(c) + bhaz(df = 6) ~ x"), build(sbhaz=ab.Dirichlet(2)), build(frame_=frame().assign(c=0))):
        assert other.target_identity != m.target_identity
    post = _posterior(m, n_draw=4)
    import autobayes.session as session
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    run = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a sampler run"])
    rec = ab.fit_record(m, run, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == "cox" and rec["model"]["addition_terms"] == ["cens(c)"]
    assert rec["model"]["priors"]["sbhaz"] == str(ab.Dirichlet(1))
    hz = rec["model"]["baseline_hazard"]
    assert hz["term"] == "bhaz()" and hz["K"] == 5 and hz["events"] == 19 and len(hz["knots"]) == 1 and len(hz["boundary"]) == 2
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_bambi_family_and_registry():
    m = build()
    fam = cox.cox_family(m.design.bhaz, np.ones(5))
    assert fam.name == "cox" and fam.likelihood.params == ["log_mu"] and fam.likelihood.parent == "log_mu"
    assert {k: v.name for k, v in fam.link.items()} == {"log_mu": "identity"}
    assert get_family("cox") is COX is FAMILIES["cox"] and COX.bambi_family() is None and COX.log_likelihood is not None
    assert COX.addition_terms == COX.own_additions == frozenset({"cens", "bhaz"}) and set(COX.refused_additions) == {"trunc", "weights"}
    assert m.names["sbhaz"] == "sbhaz" and m.names["linpred"] == "log_mu" and m.names["mean"] == "mu"
