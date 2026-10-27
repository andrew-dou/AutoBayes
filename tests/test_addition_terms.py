"""brms's response addition terms (brms-parity feature B02; docs/addition-terms.md): ``weights()``, ``cens()`` and
``trunc()`` (the last two parked until P6 accepted them against two references).

The grammar and every refusal by message; the term model (``AdditionTerm`` in ``ModelSpec.additions``, the canonical
formula, Bambi's formula without them); data validation; the weighted likelihood equal to the unweighted one plus
``sum((w - 1) * log p(y_i))`` for every family (with correlated slopes, an offset and the probit link too); predictions
that ignore the weights; the target identity, the fit record, ``describe()`` and prior-predictive draws; the oracle's G1
and G2 failing when the weights are dropped or misread. ``cens()`` and ``trunc()``: their numerics against a 200-digit
Decimal reference on both paths, models that build, describe and draw, and the frozen fixtures ``gauss_censored`` and
``gauss_truncated``, which fail the brms-only gates exactly where brms's own numbers are measurably off the exact ones and
pass every gate with their two-reference opt-in. Nothing samples; the numbers against brms are the oracle's (tests/oracle:
``gauss_weighted`` and one ``<family>_weighted`` case per family, ``gauss_censored``, ``gauss_truncated``)."""
import copy
import json
import re
from dataclasses import asdict
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

import autobayes as ab
import autobayes.formula as F
import autobayes.likelihoods as lk
from autobayes.formula import AdditionTerm
from tests.oracle import adapter, datasets, gates, two_reference
from tests.oracle.adapter import OracleMismatch
from tests.test_population_only import PANEL_STUB

DATA = datasets.tier_b(5, 40)               # every response column, positive weights w, censoring codes cens_ind
GAUSS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
NOPOOL = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
FAMILIES = {                                # left-hand side, priors and keyword arguments per constructor
    "gaussian": ("y", GAUSS, {}),
    "student": ("y", GAUSS, {"nu": 4}),
    "poisson": ("cnt", ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)), {}),
    "negbinomial": ("ynb", ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), shape=ab.Gamma(0.01, 0.01)), {}),
    "gamma": ("ypos", ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), shape=ab.Gamma(0.01, 0.01)), {}),
    "lognormal": ("ypos", GAUSS, {}),
    "beta": ("prop", ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), phi=ab.Gamma(0.01, 0.01)), {}),
    "bernoulli": ("yb", ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1)), {}),
    "binomial": ("k | trials(size)", ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1)), {}),
}


def weighted_lhs(lhs: str) -> str:
    return f"{lhs} + weights(w)" if "|" in lhs else f"{lhs} | weights(w)"


@pytest.fixture(scope="module")
def weighted():
    return ab.gaussian("y | weights(w) ~ x + (1 | g)", DATA, GAUSS)


# -- 1. the grammar and the term model ------------------------------------------------------------------------
def test_grammar_term_model_and_canonical_forms():
    s = ab.parse_formula("y | weights(w) ~ x + (1 | g)")
    assert s.additions == (AdditionTerm("weights", column="w"),) and s.trials is None and s.response == "y"
    assert s.addition("weights").term == "weights(w)" and s.addition("weights").columns == ("w",) and s.addition("cens") is None
    assert s.canonical_formula == "y | weights(w) ~ x + (1 | g)" and s.bambi_formula == "y ~ x + (1 | g)"
    s = ab.parse_formula("k|weights( w )+trials(n) ~ x", family="binomial")       # trials() first in the canonical form
    assert s.trials == "n" and s.additions == (AdditionTerm("weights", column="w"),)
    assert s.canonical_formula == "k | trials(n) + weights(w) ~ x" and s.bambi_formula == "p(k, n) ~ x"
    assert ab.parse_formula("k | trials(n) ~ x", family="binomial").additions == ()          # unchanged without addition terms
    for text, lb, ub, term in [("trunc(lb = 0)", 0.0, None, "trunc(lb = 0)"), ("trunc(ub=5)", None, 5.0, "trunc(ub = 5)"),
                               ("trunc(lb = -1, ub = 2.5)", -1.0, 2.5, "trunc(lb = -1, ub = 2.5)"),
                               ("trunc(0, 10)", 0.0, 10.0, "trunc(lb = 0, ub = 10)"),
                               ("trunc(lb = 1e-3, ub = Inf)", 0.001, None, "trunc(lb = 0.001)"),
                               ("trunc(ub = 2, lb = -Inf)", None, 2.0, "trunc(ub = 2)")]:
        a = ab.parse_formula(f"y | {text} ~ x").addition("trunc")
        assert (a.kind, a.column, a.lb, a.ub, a.term, a.columns) == ("trunc", None, lb, ub, term, ()), text
    s = ab.parse_formula("y | cens(c) + weights(w) ~ x + (1 | g)")
    assert [a.term for a in s.additions] == ["cens(c)", "weights(w)"] and s.canonical_formula == "y | cens(c) + weights(w) ~ x + (1 | g)"
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):
        assert replica == s and hash(replica) == hash(s) and replica.additions == s.additions


REFUSALS = [
    # F06a made se(s) a supported term (tests/test_se_rate.py); brms's vreal() of custom families stays unsupported
    ("y | vreal(s) ~ x", "gaussian", r"'vreal\(s\)' is not a supported response addition term.*exactly 'successes \| trials\(n\)'"),
    ("y | w ~ x", "gaussian", r"'w' is not a supported response addition term"),
    ("y | weights(w) + weights(v) ~ x", "gaussian", r"weights\(\) appears more than once on the left-hand side"),
    ("k | trials(n) + trials(m) ~ x", "binomial", r"trials\(\) appears more than once"),
    ("y | weights(w, scale = TRUE) ~ x", "gaussian", r"weights\(\) takes one column name; brms's weights\(col, scale = TRUE\)"),
    ("y | weights(2 * w) ~ x", "gaussian", r"'weights\(2 \* w\)': weights\(\) takes one column name"),
    ("y | weights() ~ x", "gaussian", r"weights\(\) takes one column name"),
    ("k | weights(n) ~ x", "binomial", r"the binomial family needs exactly 'successes \| trials\(n\)' on the left-hand side"),
    ("y | trials(n) + weights(w) ~ x", "gaussian", r"'successes \| trials\(n\)' is the binomial family's left-hand side"),
    ("log(y) | weights(w) ~ x", "gaussian", r"the left-hand side must be 'response \| term \+ term'"),
]


@pytest.mark.parametrize("formula,family,needle", REFUSALS)
def test_refusals(formula, family, needle):
    with pytest.raises(ab.FormulaError, match=needle):
        ab.parse_formula(formula, family=family)


CENS_TRUNC_REFUSALS = [
    # E04 offers interval censoring, cens(c, y2) (tests/test_cens_trunc_families.py); a third argument is refused
    ("y | cens(c, y2, z) ~ x", "gaussian", r"cens\(\) takes one column name or, for interval censoring, cens\(col, y2\)"),
    ("y | trunc(lb = x) ~ x", "gaussian", r"a truncation bound must be a number; per-row bounds read from a column \('x'\)"),
    ("y | trunc(lb = 2, ub = 1) ~ x", "gaussian", r"the truncation bounds must satisfy lb < ub"),
    ("y | trunc(lb = 1, ub = 1) ~ x", "gaussian", r"lb < ub"),
    ("y | trunc(lb = 1, lb = 2) ~ x", "gaussian", r"trunc\(\) takes the bounds lb and ub, each at most once"),
    ("y | trunc(0, 1, 2) ~ x", "gaussian", r"each at most once"),
    ("y | trunc(df = 1) ~ x", "gaussian", r"each at most once"),
    ("y | trunc() ~ x", "gaussian", r"'trunc\(\)' has no finite bound and truncates nothing"),
    ("y | trunc(lb = -Inf) ~ x", "gaussian", r"has no finite bound"),
    ("y | cens(c) + trunc(lb = 0) ~ x", "gaussian", r"cens\(\) together with trunc\(\) is not supported"),
    ("cnt | cens(c) ~ x", "poisson", r"'cens\(c\)' is not available for the poisson family; brms's cens\(\) is verified here for the "
                                     r"gaussian, gamma, lognormal, weibull, cox families"),   # E04: the Weibull and lognormal too; the Gamma since P6's
                                                                                               # reference (D120); F05: cox's right censoring
    # E04: trunc() is offered for the Poisson family, its lower bound too since D121's acceptance (test_no_park_...), and a
    # family without it refuses it
    ("ypos | trunc(lb = 0) ~ x", "gamma", r"'trunc\(lb = 0\)' is not available for the gamma family; brms's trunc\(\) is verified here for "
                                         r"the gaussian, poisson, negbinomial families"),
]


@pytest.mark.parametrize("formula,family,needle", CENS_TRUNC_REFUSALS)
def test_refusals_of_cens_and_trunc(formula, family, needle):
    with pytest.raises(ab.FormulaError, match=needle):
        ab.parse_formula(formula, family=family)


def test_no_park_and_the_family_registry():
    assert not F.PARKED_ADDITIONS                           # P6 lifted the parks of cens() and trunc() (D24, D25)
    assert ab.parse_formula("cnt | trunc(lb = 0) ~ x", family="poisson").addition("trunc").lb == 0.0   # and of counts' (D121)
    for formula in ("y | cens(c) ~ x", "y | trunc(lb = 0) ~ x", "y | cens(c) + weights(w) ~ x + (1 | g)"):
        assert ab.parse_formula(formula, family="gaussian").additions
    from autobayes.families import FAMILIES, get_family
    verified = {"gaussian", "student", "poisson", "negbinomial", "gamma", "lognormal", "beta", "bernoulli", "binomial",
                # D03: the frozen weighted cases of the zero-inflated and hurdle Poisson, and the six families whose unweighted
                # densities their own cases verify (the weighting is the same wrapper; tests/test_zero_inflated_hurdle_more.py)
                "zero_inflated_poisson", "hurdle_poisson", "zero_inflated_negbinomial", "hurdle_negbinomial",
                "zero_inflated_binomial", "zero_inflated_beta", "hurdle_gamma", "hurdle_lognormal",
                # E01: likewise for three of the four bounded and hurdle families (tests/test_bounded_families.py)
                "zero_one_inflated_beta", "zero_inflated_beta_binomial", "xbeta",
                # E03: the frozen weighted cases weibull_weighted and beta_binomial_weighted
                "weibull", "beta_binomial"}
    for name, spec in FAMILIES.items():                     # weights() only where a frozen weighted brms case verifies it
        assert ("weights" in spec.addition_terms) == (name in verified), name
        # E04: cens() for the Weibull, lognormal and (with P6's reference, D120) Gamma families, trunc() for the Poisson and
        # negative binomial families; every family with either has tails
        offered = spec.addition_terms - set(spec.parked_additions)
        assert ("cens" in offered) == (name in {"gaussian", "weibull", "lognormal", "gamma", "cox"}), name   # F05: cox's own likelihood
        assert ("cens" in spec.own_additions) == (name == "cox"), name                                      # reads it, without tails
        assert not spec.parked_additions, name
        assert ("trunc" in offered) == (name in {"gaussian", "poisson", "negbinomial"}), name
        assert (spec.tails is not None) == (name in {"gaussian", "weibull", "lognormal", "gamma", "poisson", "negbinomial"}), name
    assert get_family("bernoulli", "probit").addition_terms == get_family("binomial", "probit").addition_terms == frozenset({"weights"})
    assert get_family("gaussian").tails() is lk.NORMAL_TAILS


# -- 2. data validation -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("weights,needle", [
    (["a"] * 40, r"column 'w' has dtype \w+; the weights of 'weights\(w\)' must be numeric"),       # object or str (pandas 3)
    ([True, False] * 20, r"column 'w' has dtype bool; the weights of 'weights\(w\)' must be numeric"),
    ([1 + 0j] * 40, r"column 'w' is complex-valued"),
    ([np.nan] + [1.0] * 39, r"missing values in columns \['w'\]"),
    ([np.inf] + [1.0] * 39, r"column 'w' of 'weights\(w\)' has missing or non-finite values"),
    ([1.0, 0.0] + [1.0] * 38, r"column 'w': weights must be positive; row 1 has 0.0 \(brms accepts a zero weight.*divergence D26\)"),
    ([1.0] * 39 + [-0.5], r"weights must be positive; row 39 has -0.5"),
])
def test_weights_are_positive_finite_reals(weights, needle):
    with pytest.raises(ab.FormulaError, match=needle):
        ab.gaussian("y | weights(w) ~ x + (1 | g)", DATA.assign(w=weights), GAUSS)


def test_weights_column_must_exist_and_may_be_any_numeric_column():
    with pytest.raises(ab.FormulaError, match=r"columns not in data: \['v'\]"):
        ab.gaussian("y | weights(v) ~ x + (1 | g)", DATA, GAUSS)
    m = ab.gaussian("y | weights(size) ~ x + (1 | g)", DATA, GAUSS)          # integer weights are weights too
    assert m.design.weights.dtype == np.float64 and np.array_equal(m.design.weights, DATA["size"].to_numpy(dtype=float))
    assert m.design.cens is None and m.design.lb is None and m.design.ub is None


def test_censoring_codes_and_truncation_bounds_are_validated():
    for codes, needle in [([2] + [0] * 39, r"code 2 \(interval censoring\) needs brms's cens\(c, y2\)"),
                          ([0.5] + [0] * 39, r"censoring codes must be -1 \(left\), 0 \(none\) or 1 \(right\); found 0.5"),
                          (["Left"] * 40, r"the censoring labels of 'cens\(c\)' are brms's 'left', 'none', 'right' and 'interval'"),
                          ([1.5, "none"] * 20, r"the censoring labels .*; found 1.5")]:   # F08 reads brms's labels and TRUE / FALSE
        with pytest.raises(ab.FormulaError, match=needle):
            ab.gaussian("y | cens(c) ~ x + (1 | g)", DATA.assign(c=codes), GAUSS)
    with pytest.raises(ab.FormulaError, match=r"response 'y' at row \d+ is .*, outside the bounds of 'trunc\(lb = 0\)' \(\d+ row\(s\) outside\)"):
        ab.gaussian("y | trunc(lb = 0) ~ x + (1 | g)", DATA, GAUSS)
    m = ab.gaussian("y | trunc(lb = -10, ub = 10) ~ x", DATA, NOPOOL)
    assert np.array_equal(m.design.lb, np.full(40, -10.0)) and np.array_equal(m.design.ub, np.full(40, 10.0))
    m = ab.gaussian("ypos | trunc(lb = 0) ~ x", DATA, NOPOOL)                  # brms's standata: ub = Inf on the open side
    assert np.array_equal(m.design.lb, np.zeros(40)) and np.array_equal(m.design.ub, np.full(40, np.inf))
    m = ab.gaussian("y | cens(cens_ind) ~ x", DATA, NOPOOL)
    assert m.design.cens.dtype == np.int64 and np.array_equal(m.design.cens, DATA["cens_ind"].to_numpy())
    assert m.design.lb is None and m.design.ub is None and m.design.weights is None


# -- 3. the weighted likelihood ---------------------------------------------------------------------------------
def _point(model, seed=3):
    """A random point in value space around PyMC's initial point (the same names in a weighted and an unweighted model)."""
    rng = np.random.default_rng(seed)
    return {k: np.asarray(v + rng.normal(0.0, 0.3, np.shape(v))) for k, v in model.pymc_model.initial_point().items()}


def _rowwise(model):
    """The model's log likelihood per row, as a compiled function of the value point."""
    pm_model = model.pymc_model
    rv = pm_model[model.names["y"]]
    return pm_model.compile_fn(pm_model.logp(vars=[rv], sum=False)[0], inputs=pm_model.value_vars, on_unused_input="ignore")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_weighted_log_density_is_the_rows_times_their_weights(family):
    lhs, priors, kwargs = FAMILIES[family]
    rhs = " ~ x + (1 | g)"
    m0 = getattr(ab, family)(lhs + rhs, DATA, priors, **kwargs)
    mw = getattr(ab, family)(weighted_lhs(lhs) + rhs, DATA, priors, **kwargs)
    ones = getattr(ab, family)(weighted_lhs(lhs) + rhs, DATA.assign(w=1.0), priors, **kwargs)
    point = _point(m0)
    l0, lw, l1 = (float(m.pymc_model.compile_logp()(point)) for m in (m0, mw, ones))
    rows = np.asarray(_rowwise(m0)(point), dtype=float)
    w = mw.design.weights
    assert np.array_equal(w, DATA["w"].to_numpy()) and np.all(np.isfinite(rows))
    assert abs((lw - l0) - np.sum((w - 1.0) * rows)) <= 1e-10 * max(1.0, abs(lw)), family
    np.testing.assert_allclose(np.asarray(_rowwise(mw)(point), dtype=float), w * rows, rtol=1e-12, atol=1e-12)   # pointwise too
    assert abs(l1 - l0) <= 1e-12 * max(1.0, abs(l0)), family                    # unit weights: the unweighted density
    assert len({mw.target_identity["sha256"], m0.target_identity["sha256"], ones.target_identity["sha256"]}) == 3
    assert {v.name for v in mw.pymc_model.value_vars} == {v.name for v in m0.pymc_model.value_vars}


def test_weighted_density_on_the_jax_path(weighted):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(weighted.pymc_model)
    names = [v.name for v in weighted.pymc_model.value_vars]
    for seed in range(3):
        point = _point(weighted, seed)
        a, b = float(jf([point[n] for n in names])), float(weighted.pymc_model.compile_logp()(point))
        assert abs(a - b) <= 1e-12 * max(1.0, abs(b))


@pytest.mark.parametrize("family,formula,priors,kwargs", [
    ("gaussian", "y | weights(w) ~ x + (1 + x | g)",                       # the correlated slopes' graph rewrite (A02)
     ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
               cor={"g": ab.LKJ(2)}, sigma=ab.HalfNormal(1)), {}),
    ("poisson", "cnt | weights(w) ~ x + offset(log(exposure)) + (1 | g:h)",   # an offset (A06) and a derived factor (A03)
     ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)), {}),
    ("bernoulli", "yb | weights(w) ~ x + (1 | g)", ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1)),
     {"link": "probit"}),
    ("binomial", "k | trials(size) + weights(w) ~ x", ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5)), {"link": "probit"}),
])
def test_weights_compose_with_other_constructs(family, formula, priors, kwargs):
    plain = re.sub(r" \+ weights\(w\)| \| weights\(w\)", "", formula)
    mw, m0 = (getattr(ab, family)(f, DATA, priors, **kwargs) for f in (formula, plain))
    assert "weights(w)" in mw.spec.canonical_formula and "weights" not in m0.spec.canonical_formula
    point = _point(m0)
    lw, l0 = (float(m.pymc_model.compile_logp()(point)) for m in (mw, m0))
    rows = np.asarray(_rowwise(m0)(point), dtype=float)
    assert abs((lw - l0) - np.sum((mw.design.weights - 1.0) * rows)) <= 1e-10 * max(1.0, abs(lw)), formula


# -- 4. predictions, records, describe() ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture_draws():
    """The oracle case gauss_weighted with its eight seeded points as posterior draws."""
    b = gates.Bundle("gauss_weighted")
    return b, adapter.draws(b.fixture, b.model, b.cmap, b.gmap)


def test_predictions_ignore_the_weights(fixture_draws):
    b, idata = fixture_draws
    plain = ab.gaussian("y ~ x + (1 | g)", b.data, b.model.priors)
    nd = b.newdata.drop(columns=["w"])                                        # new rows need no weights column
    for newdata, include in ((None, True), (nd, True), (nd, False)):
        for kind in ("linear", "mean"):
            got = b.model.predict(idata, kind=kind, newdata=newdata, include_group_effects=include)
            want = plain.predict(idata, kind=kind, newdata=newdata, include_group_effects=include)
            assert set(got.data_vars) == set(want.data_vars)
            for var in got.data_vars:
                np.testing.assert_allclose(got[var].to_numpy(), want[var].to_numpy(), rtol=1e-12, atol=1e-12)
        resp = b.model.predict(idata, kind="response", newdata=newdata, include_group_effects=include, seed=7)
        sigma = idata.posterior["sigma"].to_numpy()[..., None]
        np.testing.assert_array_equal(resp["y"].to_numpy(), np.random.default_rng(7).normal(resp["mu"].to_numpy(), sigma))
    mean = b.model.predict(idata, kind="mean")["mu"].to_numpy()[0]
    np.testing.assert_allclose(mean, b.fixture.prediction("fitted", "epred"), rtol=1e-12, atol=1e-12)   # brms's epred ignores weights


def test_identity_record_describe_and_prior_predictive(fixture_draws, weighted):
    b, idata = fixture_draws
    m = b.model

    def record(model):
        fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 8, 0, 0.9, 0, seed=0,
                           _target_identity_json=json.dumps(model.target_identity))
        return ab.fit_record(model, fit, panel=PANEL_STUB)
    rec = record(m)
    json.dumps(rec)
    assert rec["model"]["addition_terms"] == ["weights(w)"] and rec["model"]["canonical_formula"] == "y | weights(w) ~ x + (1 | g)"
    plain = ab.gaussian("y ~ x + (1 | g)", b.data, m.priors)
    assert "addition_terms" not in record(plain)["model"]
    same = ab.gaussian("y | weights(w) ~ x + (1 | g)", b.data.copy(), m.priors)
    other = ab.gaussian("y | weights(w) ~ x + (1 | g)", b.data.assign(w=b.data.w * 2), m.priors)
    assert same.target_identity == m.target_identity
    assert len({m.target_identity["sha256"], other.target_identity["sha256"], plain.target_identity["sha256"]}) == 3
    text = m.describe()
    w = m.design.weights
    for s in ["AutoBayes Gaussian model with response addition term(s) weights(w)",
              f"  weights   : w, each fitted row's log likelihood multiplied by its weight (brms weights()); weights {w.min():.4f} to "
              f"{w.max():.4f}, sum {w.sum():.4f} over 36 rows; the priors are not weighted, and predict() ignores the weights",
              "  weights(): the log likelihood is sum_i w_i log p(y_i | ...), w_i the w of row i (brms's weights[n] *)"]:
        assert s in text, s
    assert "weights" not in plain.describe() and "addition term" not in plain.describe()
    pp = weighted.prior_predictive(draws=20, seed=2)
    assert pp.prior_predictive["y"].shape == (1, 20, weighted.design.n)
    np.testing.assert_array_equal(pp.observed_data["y"].to_numpy(), weighted.design.y)


def test_weights_gate_discriminates(fixture_draws):
    """G1 and G2 fail when the AutoBayes model drops the weights or reads the wrong column: they pass for the right reason."""
    b, _ = fixture_draws
    case = copy.deepcopy(b.case)
    case["autobayes"]["formula"] = "y ~ x + (1 | g)"
    plain = gates.Bundle("gauss_weighted", model=adapter.build_model(case, b.data))
    with pytest.raises(OracleMismatch, match="brms has weights but AutoBayes does not"):
        gates.check_design(plain)
    with pytest.raises(OracleMismatch, match="log density"):
        gates.check_logp(plain)
    case["autobayes"]["formula"] = "y | weights(size) ~ x + (1 | g)"
    wrong = gates.Bundle("gauss_weighted", model=adapter.build_model(case, b.data))
    with pytest.raises(OracleMismatch, match="weights differ by"):
        gates.check_design(wrong)


# -- 5. cens() and trunc(): numerics on both paths against 200 digits --------------------------------------------------------
# A reference for |z| <= 15: erf's Taylor series summed at 200 digits (its largest term there is about 1e49, so the sum
# keeps 150 digits, and Phi(-15) = 3.7e-51 keeps 99). Every function below runs inside a local 200-digit context: other
# modules set the global Decimal context at import (tests/test_links.py sets 50 digits).
PREC = 200


@lru_cache(maxsize=None)
def _pi() -> Decimal:
    with localcontext() as ctx:
        ctx.prec = PREC + 10

        def atan_inv(n):
            term, s, k = Decimal(1) / n, Decimal(0), 0
            while abs(term) > Decimal(10) ** -(PREC + 5):
                s += term / (2 * k + 1) * (1 if k % 2 == 0 else -1)
                term /= n * n
                k += 1
            return s
        return +(16 * atan_inv(5) - 4 * atan_inv(239))


def _erf(x: Decimal) -> Decimal:
    x2, term, s, n = x * x, x, Decimal(0), 0
    while True:
        t = term / (2 * n + 1)
        s += t
        if abs(t) < Decimal(10) ** -(PREC - 10):
            return 2 * s / _pi().sqrt()
        n += 1
        term = -term * x2 / n


def _cdf(z: Decimal) -> Decimal:
    return (1 + _erf(z / Decimal(2).sqrt())) / 2


def _pdf(z: Decimal) -> Decimal:
    return (-(z * z) / 2).exp() / (2 * _pi()).sqrt()


def _mass(a: Decimal, b: Decimal) -> Decimal:
    """Phi(b) - Phi(a), the difference of erfs at 200 digits."""
    r2 = Decimal(2).sqrt()
    return (_erf(b / r2) - _erf(a / r2)) / 2


GRID = [-15.0, -12.0, -8.25, -6.17, -3.0, -1.0, -1e-3, 0.0, 0.4, 1.0, 2.9, 4.1, 6.17, 7.34, 8.25, 12.0, 15.0]
INTERVALS = [(-15.0, -12.0), (-8.0, -7.999), (-6.0, -1.0), (-2.0, 0.0), (-1.0, 1.0), (-1e-3, 2e-3), (0.0, 3.0),
             (2.0, 2.001), (2.0, 9.0), (6.17, 15.0), (12.0, 15.0)]


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_normal_tails_are_exact_on_both_paths(mode):
    if mode == "JAX":
        pytest.importorskip("jax")
    v, mu, s, lo, hi = (pt.dvector(n) for n in ("v", "mu", "s", "lo", "hi"))

    def with_grads(expr):                                         # value, d/dmu and d/dsigma per row
        return [expr, pt.grad(expr.sum(), mu), pt.grad(expr.sum(), s)]
    tails = pytensor.function([v, mu, s], with_grads(lk.normal_log_cdf(v, mu, s)) + with_grads(lk.normal_log_ccdf(v, mu, s)), mode=mode)
    mass = pytensor.function([lo, hi, mu, s], with_grads(lk.normal_log_mass(lo, hi, mu, s)), mode=mode)
    sigma = 1.7
    z = np.array(GRID)
    mu_z = 0.3 - sigma * z                                        # the value 0.3 sits z SD from the location
    got = [np.asarray(r, dtype=float) for r in tails(np.full(len(z), 0.3), mu_z, np.full(len(z), sigma))]
    a, b = np.array([p[0] for p in INTERVALS]), np.array([p[1] for p in INTERVALS])
    lo_v, hi_v = 0.25 + sigma * a, 0.25 + sigma * b
    got_m = [np.asarray(r, dtype=float) for r in mass(lo_v, hi_v, np.full(len(a), 0.25), np.full(len(a), sigma))]
    with localcontext() as ctx:
        ctx.prec = PREC
        S = Decimal(sigma)
        for i in range(len(z)):
            Z = (Decimal(0.3) - Decimal(float(mu_z[i]))) / S
            P, Q, d = _cdf(Z), 1 - _cdf(Z), _pdf(Z)
            want = [P.ln(), -d / (S * P), -Z * d / (S * P), Q.ln(), d / (S * Q), Z * d / (S * Q)]
            for k, w in enumerate(want):
                assert np.isfinite(got[k][i]) and abs(got[k][i] - float(w)) <= 1e-11 * max(1.0, abs(float(w))), (mode, GRID[i], k, got[k][i], float(w))
        for i, (ai, bi) in enumerate(INTERVALS):
            A, B = (Decimal(float(lo_v[i])) - Decimal(0.25)) / S, (Decimal(float(hi_v[i])) - Decimal(0.25)) / S
            M = _mass(A, B)
            want = [M.ln(), -(_pdf(B) - _pdf(A)) / (S * M), -(B * _pdf(B) - A * _pdf(A)) / (S * M)]
            for k, w in enumerate(want):
                assert np.isfinite(got_m[k][i]) and abs(got_m[k][i] - float(w)) <= 1e-11 * max(1.0, abs(float(w))), (mode, ai, bi, k, got_m[k][i], float(w))


def test_truncated_mean_is_exact_where_brms_formula_is_not():
    from scipy.special import ndtr
    from scipy.stats import norm

    a = np.array([-15.0, -8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 6.17, 7.34, 8.0, 12.0, 15.0, -2.0, 1.0, 4.0, -9.0])
    b = np.array([np.inf] * 12 + [-1.0, 1.5, 4.5, 9.0])
    got = lk.standard_truncated_mean(a, b)
    with localcontext() as ctx:
        ctx.prec = PREC
        for i in range(len(a)):
            A = Decimal(float(a[i]))
            want = _pdf(A) / (1 - _cdf(A)) if np.isinf(b[i]) else (_pdf(A) - _pdf(Decimal(float(b[i])))) / _mass(A, Decimal(float(b[i])))
            assert abs(got[i] - float(want)) <= 1e-12 * max(1.0, abs(float(want))), (a[i], b[i], got[i], float(want))
        exact = float(_pdf(Decimal(7.34)) / (1 - _cdf(Decimal(7.34))))
    # brms's posterior_epred forms 1 - pnorm(z) by subtraction: the ratio is 6e-5 relative off here, and the mean
    # mu + sigma * ratio, which is small next to its two terms, 1e-3 (the fixture gauss_truncated at point 4)
    brms = norm.pdf(7.34) / (1.0 - ndtr(7.34))
    assert abs(brms - exact) / exact > 1e-5 and abs(lk.standard_truncated_mean(7.34, np.inf) - exact) / exact < 1e-14
    assert np.all(np.isfinite(lk.standard_truncated_mean(np.array([-40.0, 40.0]), np.array([np.inf, np.inf]))))


# -- 6. cens() and trunc() in models: they build, describe, draw, and meet brms where brms is exact, the reference elsewhere --
def test_cens_and_trunc_build_describe_and_draw():
    for formula in ["y | cens(cens_ind) + weights(w) ~ x + (1 | g)", "ypos | trunc(lb = 0) ~ x + (1 | g)",
                    "y | trunc(lb = -5, ub = 5) ~ x", "y | trunc(ub = 6) + weights(w) ~ x"]:
        m = ab.gaussian(formula, DATA, GAUSS if "(1 | g)" in formula else NOPOOL)
        point = _point(m)
        lp = float(m.pymc_model.compile_logp()(point))
        grad = np.asarray(m.pymc_model.compile_dlogp()(point), dtype=float)
        assert np.isfinite(lp) and np.all(np.isfinite(grad)), formula
        y = m.prior_predictive(draws=30, seed=1).prior_predictive[m.names["y"]].to_numpy()
        trunc = m.spec.addition("trunc")
        if trunc is not None:                                     # brms's posterior_predict draws from the truncated distribution
            assert (trunc.lb is None or y.min() >= trunc.lb) and (trunc.ub is None or y.max() <= trunc.ub), formula
        assert f"response addition term(s) {', '.join(a.term for a in m.spec.additions)}" in m.describe()
    m = ab.gaussian("y | cens(cens_ind) ~ x + (1 | g)", DATA, GAUSS)
    codes = DATA["cens_ind"].to_numpy()
    assert (f"  censoring : cens_ind: {(codes == -1).sum()} left-censored row(s) (-1: y_i is an upper bound, log P(Y_i <= y_i)), "
            f"{(codes == 1).sum()} right-censored (1: a lower bound, log P(Y_i > y_i)), {(codes == 0).sum()} observed (0)") in m.describe()


@pytest.fixture(scope="module")
def frozen_bundles():
    return {cid: gates.Bundle(cid) for cid in ("gauss_censored", "gauss_truncated")}


def _brms_only(b):
    """``b`` without its two-reference opt-in (P6): its gates compare with brms alone, as before."""
    other = copy.copy(b)
    other.case = {k: v for k, v in b.case.items() if k != "gate"}
    other.__dict__.pop("reference", None)
    return other


def _gradient(b, k):
    """AutoBayes's gradient without the Jacobian at seeded point k, on brms's unconstrained coordinates in brms's order."""
    vp = adapter.value_point(b.model, b.fixture.points[k], b.cmap, b.gmap)
    layout = adapter.value_layout(b.model, vp)
    ga = np.asarray(b.dlogp_nojac(vp), dtype=float)
    return np.array([ga[layout[var].start + idx] for _, var, idx in sorted(adapter.coordinate_map(b.fixture, b.model, b.cmap, b.gmap))])


def _rel(u, v):
    return float(np.max(np.abs(u - v) / np.maximum(1.0, np.maximum(np.abs(u), np.abs(v)))))


def test_fixtures_fail_against_brms_alone_only_where_brms_is_inexact(frozen_bundles):
    """Both frozen cases pass G1, the G2 density, G3, G5 and G6, the censored one G4 too, against brms alone; against brms
    alone the gradient gate fails exactly at the seeded points where brms's gradient is off the exact one, and G4 where
    brms's truncated mean is (the 50-digit comparison is the next test's). With their two-reference opt-in (P6) the
    committed reference stands in there, and both gates pass."""
    for cid, b in frozen_bundles.items():
        for check in (gates.check_design, gates.check_logp, gates.check_derived, gates.check_jax, gates.check_default_prior):
            check(b)
        with pytest.raises(OracleMismatch, match="gradient"):
            gates.check_gradient(_brms_only(b))
        over = [k for k, p in enumerate(b.fixture.points) if _rel(_gradient(b, k), p.grad_nojac) > 1e-7]
        assert over == {"gauss_censored": [0, 4, 5, 6, 7], "gauss_truncated": [4]}[cid], (cid, over)
        with pytest.warns(two_reference.ReferenceFallback, match=rf"{cid} G2 gradient: "):
            gates.check_gradient(b)
    gates.check_predictions(frozen_bundles["gauss_censored"])                  # epred = linpred: censoring changes no prediction
    with pytest.raises(OracleMismatch, match=r"fitted: 'mu' differs from brms posterior_epred by 4\.1"):
        gates.check_predictions(_brms_only(frozen_bundles["gauss_truncated"]))
    with pytest.warns(two_reference.ReferenceFallback, match="gauss_truncated G4 epred: 6 of 656 entries"):
        gates.check_predictions(frozen_bundles["gauss_truncated"])


def _exact_gradient(b, k):
    """The exact gradient (200-digit arithmetic, rounded to double) of brms's log density without the Jacobian at seeded
    point k of the censored or lower-truncated Gaussian case ``y ~ x + (1 | g)`` (priors normal(0, 2) on b, normal(0, 5)
    on the Intercept, half-normal(0, 1) on sigma and sd), on brms's unconstrained coordinates in brms's order."""
    sd, p = b.fixture.standata, b.fixture.points[k]
    with localcontext() as ctx:
        ctx.prec = PREC
        D = lambda v: Decimal(float(v))                                       # noqa: E731
        Y, Z = [D(v) for v in sd["Y"]], [D(v) for v in sd["Z_1_1"]]
        Xc = [D(v) for v in (sd["X"][:, 1] - sd["X"][:, 1].mean())]          # brms centres in double precision
        J = (sd["J_1"].astype(int) - 1).tolist()
        cens = sd["cens"].astype(int).tolist() if "cens" in sd else [0] * len(Y)
        lb = D(sd["lb"][0]) if "lb" in sd else None
        bx, a, s = D(np.atleast_1d(p.pars["b"])[0]), D(p.pars["Intercept"]), D(p.pars["sigma"])
        sd1, z = D(np.atleast_1d(p.pars["sd_1"])[0]), [D(v) for v in np.atleast_2d(p.pars["z_1"])[0]]
        dmu, dsig = [], []
        for i, y in enumerate(Y):
            mu = a + bx * Xc[i] + sd1 * z[J[i]] * Z[i]
            t = (y - mu) / s
            if cens[i] == 0:
                dm, ds = t / s, (t * t - 1) / s
            else:                                                             # d/dmu log Q(t) = h / s, log Phi(t): -h / s
                h = _pdf(t) / (1 - _cdf(t)) if cens[i] == 1 else -_pdf(t) / _cdf(t)
                dm, ds = h / s, h * t / s
            if lb is not None:                                                # minus log Q((lb - mu) / s)
                tl = (lb - mu) / s
                h = _pdf(tl) / (1 - _cdf(tl))
                dm, ds = dm - h / s, ds - h * tl / s
            dmu.append(dm)
            dsig.append(ds)
        g = [sum(dm * x for dm, x in zip(dmu, Xc)) - bx / 4, sum(dmu) - a / 25, s * (sum(dsig) - s),
             sd1 * (sum(dmu[i] * Z[i] * z[J[i]] for i in range(len(Y))) - sd1)]
        g += [sum(dmu[i] * Z[i] * sd1 for i in range(len(Y)) if J[i] == j) - z[j] for j in range(len(z))]
        return np.array([float(v) for v in g])


@pytest.mark.parametrize("cid,k,brms_off", [("gauss_censored", 4, 2.4e-6), ("gauss_censored", 0, 2.2e-7), ("gauss_truncated", 4, 4.1e-7)])
def test_autobayes_gradient_is_exact_where_brms_is_not(frozen_bundles, cid, k, brms_off):
    b = frozen_bundles[cid]
    assert [e["par"] for e in b.fixture.upars] == ["b", "Intercept", "sigma", "sd_1"] + ["z_1"] * 6
    exact = _exact_gradient(b, k)
    ours, theirs = _gradient(b, k), b.fixture.points[k].grad_nojac
    assert _rel(ours, exact) < 1e-13 and _rel(theirs, exact) > brms_off, (_rel(ours, exact), _rel(theirs, exact))
