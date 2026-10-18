"""D06 (docs/post-processing.md): ``hypothesis`` and ``conditional_effects`` beyond the oracle gate, which compares them with
brms on the frozen cases ``gauss_hypothesis_effects`` and ``bernoulli_hypothesis_effects``.

- ``hypothesis``: brms 2.23's ``hypothesis()`` on recorded draws (``tests/hypothesis_probes``): 45 hypotheses on 40 draws with
  ``alpha`` 0.05, 0.1 and 0.2 and ``robust`` on and off, and eight edge sets (draws all on one side, ties at zero, one draw,
  posterior probabilities at and just above ``1 - alpha``), every table entry, label, star and draw; what brms refuses is
  refused, and so is what brms evaluates beyond the parser's grammar; R's precedence; brms's coefficient names on every
  frozen case, whose draws must be brms's ``b`` at the fixture's points; the models and names it refuses.
- ``conditional_effects``: on every frozen case it takes, the grid, the values held and the summaries of
  ``predict(kind="mean", include_group_effects=False)`` at the grid; its options and the models and effects it refuses.
- The post-processing integrations: ``log_lik``, ``bayes_R2``, ``loo`` and ``waic`` refuse a multivariate model naming it,
  and ``waic`` of a non-linear model and of a mixture is that of their ``log_lik`` (tested in test_nonlinear and test_mixture).
No sampler runs here: every posterior is a set of frozen or recorded draws.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from autobayes.conditional_effects import _refuse_unsupported_effects
from autobayes.hypothesis import HYPOTHESIS_COLUMNS, hypothesis_from_draws, parse_hypothesis, refuse_unsupported
from autobayes.post_processing import waic_from_log_lik
from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import coefficients
from tests.oracle.tolerances import rel_err

RECORD = json.loads((Path(__file__).parent / "hypothesis_probes" / "brms-2.23.0-hypothesis.json").read_text())


def _decode(obj) -> np.ndarray:
    """A recorded array; R's NA and NaN as NaN."""
    values = np.array([math.nan if v in ("NA", "NaN") else float(v) for v in obj["values"]], dtype=float)
    return values.reshape([int(d) for d in obj["dims"]], order="F")


def _same(ours, theirs, tol=1e-12):
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert ours.shape == theirs.shape
    assert np.array_equal(np.isnan(ours), np.isnan(theirs))
    assert np.array_equal(ours[np.isinf(ours)], theirs[np.isinf(theirs)])
    finite = np.isfinite(theirs)
    assert np.array_equal(finite, np.isfinite(ours))
    assert rel_err(ours[finite], theirs[finite]) <= tol


def _probe_sets():
    return [("main", RECORD["main"])] + [(f"edge {k}", v) for k, v in RECORD["edge"].items()]


@pytest.mark.parametrize("name, probe", _probe_sets(), ids=[k for k, _ in _probe_sets()])
def test_hypothesis_equals_brms_on_recorded_draws(name, probe):
    draws = {k: _decode(v) for k, v in probe["draws"].items()}
    for run in probe["runs"]:
        h = hypothesis_from_draws(draws, run["hypotheses"], alpha=run["alpha"], robust=run["robust"])
        assert list(h.table.columns) == run["columns"] == list(HYPOTHESIS_COLUMNS)
        assert list(h.table["Hypothesis"]) == run["labels"]
        assert list(h.table["Star"]) == run["star"]
        for column, values in run["table"].items():
            _same(h.table[column].to_numpy(), _decode(values))
        if "samples" in run:                                           # recorded for the first setting of each set
            assert list(h.samples.columns) == run["sample_names"]
            _same(h.samples.to_numpy(), _decode(run["samples"]))
    assert "samples" in probe["runs"][0]


def test_the_probes_cover_both_stars_ties_and_infinite_ratios():
    stars = [s for _, p in _probe_sets() for run in p["runs"] for s in run["star"]]
    ratios = np.concatenate([_decode(run["table"]["Evid.Ratio"]) for _, p in _probe_sets() for run in p["runs"]])
    assert "*" in stars and "" in stars
    assert np.isinf(ratios).any() and (ratios == 0).any() and np.isnan(ratios).any()
    tiny = RECORD["edge"]["tiny_negative_20"]["runs"][0]                # Post.Prob 0.95 = 1 - alpha: no star, a CI above 0
    assert tiny["star"][0] == "" and _decode(tiny["table"]["CI.Lower"])[0] > 0


def test_what_brms_refuses_is_refused():
    draws = {k: _decode(v) for k, v in RECORD["main"]["draws"].items()}
    assert len(RECORD["errors"]) == 19
    for record in RECORD["errors"]:
        assert record["error"]
        with pytest.raises(ValueError, match="hypothesis"):
            hypothesis_from_draws(draws, [record["hypothesis"]])


@pytest.mark.parametrize("text, match", [
    ("x %% 2 > 0", "unexpected '%'"),                                   # brms evaluates these as R would
    ("max(x, w) > 0", "the function max\\(\\) is not offered"),
    ("`x` > 0", "unexpected '`'"),
    ("x = 0.0e0", "write numbers in decimal notation"),
    ("x & w > 0", "unexpected '&'"),
    ("x$w > 0", "unexpected '\\$'"),
    ("x > 0;", "unexpected ';'"),
    ("__import__('os') > 0", "unexpected '_'"),
    ("exp(x, w) > 0", "unexpected ','"),
    ("log(x > 0", "every hypothesis must be of the form|ends too early"),
    ("(x > 0", "ends too early"),
    ("x) > 0", "unexpected '\\)'"),
    ("x + > 0", "ends too early"),
])
def test_what_the_grammar_does_not_take_is_refused(text, match):
    draws = {"x": np.array([1.0, 2.0]), "w": np.array([0.5, -1.0])}
    with pytest.raises(ValueError, match=match):
        hypothesis_from_draws(draws, [text])


@pytest.mark.parametrize("text, expected", [
    ("-x^2 = 0", -9.0), ("2^3^2 * x = 0", 512.0 * 3), ("2^-1 * x = 0", 1.5), ("x--w = 0", 5.0), ("-x*w = 0", -6.0),
    ("x/w/2 = 0", 0.75), ("x - w - 1 = 0", 0.0), ("(x - w) * (x + w) = 0", 5.0), ("sqrt(abs(-x)) = 0", math.sqrt(3.0)),
    ("exp(log(x)) = 0", 3.0), ("+x = 0", 3.0), ("x^2^0.5 = 0", 3.0 ** (2.0 ** 0.5)), ("x = w", 1.0), ("w < x", -1.0),
    ("2 * x > w + 1", 3.0), ("- - x = 0", 3.0),
])
def test_r_precedence(text, expected):
    """``^`` binds tighter than a sign on its left, takes a signed power on its right and groups to the right; the signs bind
    tighter than ``*`` and ``/``; everything else groups to the left."""
    h = hypothesis_from_draws({"x": np.array([3.0]), "w": np.array([2.0])}, [text])
    assert h.samples["H1"].to_numpy()[0] == pytest.approx(expected, rel=1e-15)


def test_labels_are_brms_text():
    assert parse_hypothesis(" zr <  zq ").label == "(zr)-(zq) < 0"
    assert parse_hypothesis("x + w = 0").label == "(x+w) = 0"
    assert parse_hypothesis("x > 00").label == "(x)-(00) > 0"
    assert parse_hypothesis("x:zq>0").names == ("x:zq",)
    assert parse_hypothesis("Intercept + x * 2 < w").names == ("Intercept", "x", "w")


def _ungated_bundles():
    for cid in fx.case_ids():
        case = fx.load_case(cid)
        if fx.has_fixture(cid) and "autobayes" in case and (case.get("gate") or {}).get("expect") != "fail":
            yield gates.Bundle(cid)


def test_names_are_brms_coefficient_names_on_every_case():
    """On every frozen case whose model ``hypothesis`` takes, each of brms's design columns, by brms's name for it (the
    fixture's ``X`` column names), and ``Intercept`` (brms's ``b_Intercept``) are hypotheses whose draws are exactly brms's
    coefficient at the fixture's points; the cases it refuses are the multivariate, non-linear, mixture, cumulative and
    categorical ones."""
    taken, refused = [], []
    for b in _ungated_bundles():
        m = b.model
        try:
            refuse_unsupported(m, "hypothesis()")
        except ValueError as e:
            refused.append((b.case["id"], str(e)))
            continue
        names = b.fixture.colnames("X") or []
        columns = [(names[b.cmap.x_column[c]], b.cmap.b_index(b.cmap.x_column[c])) for c in range(len(m.spec.columns))]
        hyps = [f"{n} = 0" for n, _ in columns] + (["Intercept = 0"] if m.spec.intercept else [])
        if not hyps:
            continue
        h = ab.hypothesis(m, adapter.draws(b.fixture, m, b.cmap, b.gmap), hyps)
        for k, (name, index) in enumerate(columns):
            assert np.array_equal(h.samples[f"H{k + 1}"].to_numpy(), [coefficients(p.pars)[index] for p in b.fixture.points]), name
        if m.spec.intercept:
            raw = [float(p.pars["b_Intercept"]) if b.cmap.centred_intercept else coefficients(p.pars)[b.cmap.b_index(b.cmap.intercept_column)]
                   for p in b.fixture.points]
            assert np.array_equal(h.samples[f"H{len(hyps)}"].to_numpy(), raw)
        taken.append(b.case["id"])
    assert len(taken) >= 80 and {"gauss_transforms", "gauss_interaction_categorical_categorical", "gauss_no_intercept",
                                 "bernoulli_interaction_two_groups", "gauss_interaction_numeric_numeric"} <= set(taken)
    kinds = {why.split(" is not offered")[0] for _, why in refused}
    assert kinds == {"hypothesis() of a multivariate model", "hypothesis() of a non-linear model (nl=True)",
                     "hypothesis() of a model with a mixture", "hypothesis() of a model with the cumulative family's thresholds",
                     "hypothesis() of a model with the categorical family's categories",
                     # D04: the sratio, cratio and acat families' unordered thresholds, refused as the cumulative family's
                     "hypothesis() of a model with the sratio family's thresholds", "hypothesis() of a model with the cratio family's "
                     "thresholds", "hypothesis() of a model with the acat family's thresholds",
                     # E01: the hurdle_cumulative family's thresholds, refused as the cumulative family's
                     "hypothesis() of a model with the hurdle_cumulative family's thresholds"}


def _bundle(case_id):
    return gates.Bundle(case_id)


def test_hypothesis_refuses_names_it_does_not_take():
    b = _bundle("gauss_monotonic")
    draws = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    with pytest.raises(ValueError, match="'moord' is the coefficient of mo\\(ord\\), brms's class bsp, not b"):
        ab.hypothesis(b.model, draws, "moord > 0")
    assert ab.hypothesis(b.model, draws, "Intercept > 0").table.shape == (1, 8)
    b = _bundle("gauss_distributional_sigma")
    draws = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    for name in ("sigma_Intercept", "sigma_zq"):
        with pytest.raises(ValueError, match=f"'{name}' is a coefficient of the distributional formula for sigma"):
            ab.hypothesis(b.model, draws, f"{name} > 0")
    with pytest.raises(ValueError, match="no population-level coefficient \\(class b\\) 'b_x'; the population-level coefficient "
                                         "\\(class b\\)s are \\['Intercept', 'x'\\]"):
        ab.hypothesis(b.model, draws, "b_x > 0")
    b = _bundle("gauss_me")
    draws = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    with pytest.raises(ValueError, match="'mexobssdx' is the coefficient of me\\(xobs, sdx\\), brms's class bsp"):
        ab.hypothesis(b.model, draws, "mexobssdx = 0")


def test_hypothesis_options_and_result():
    b = _bundle("gauss_hypothesis_effects")
    draws = gates.fixed_parameter_draws(b)
    for bad in ({"alpha": 0}, {"alpha": 1.0}, {"alpha": True}, {"alpha": "0.05"}, {"robust": 1}):
        with pytest.raises(ValueError, match="alpha must be|robust must be"):
            ab.hypothesis(b.model, draws, "x > 0", **bad)
    for bad in ([], (), 3, None):
        with pytest.raises(ValueError, match="hypotheses must be"):
            ab.hypothesis(b.model, draws, bad)
    h = ab.hypothesis(b.model, draws, "x > 0")                           # one string is one hypothesis
    assert isinstance(h, ab.Hypothesis) and h.hypotheses == ("x > 0",) and h.samples.shape == (8, 1)
    assert h.fit_binding_status == "unverified" and h.alpha == 0.05 and h.robust is False
    text = str(h)
    assert "(x) > 0" in text and "Evid.Ratio" in text and "90% interval" in text and "D96" in text
    robust = ab.hypothesis(b.model, draws, ["x > 0", "x = 0"], alpha=0.1, robust=True)
    v = robust.samples["H1"].to_numpy()
    assert robust.table.loc[0, "Estimate"] == np.median(v)
    assert robust.table.loc[0, "Est.Error"] == pytest.approx(1.4826 * np.median(np.abs(v - np.median(v))), rel=1e-15)
    assert robust.table.loc[1, "CI.Lower"] == pytest.approx(np.quantile(v, 0.05), rel=1e-15)
    assert np.isnan(robust.table.loc[1, "Evid.Ratio"]) and np.isnan(robust.table.loc[1, "Post.Prob"])
    with pytest.raises(ValueError, match="is NaN at 1 of 2 draws"):
        hypothesis_from_draws({"x": np.array([-1.0, 2.0])}, ["log(x) > 0"])
    with pytest.raises(ValueError, match="the same positive number of draws"):
        hypothesis_from_draws({"x": np.array([1.0, 2.0]), "w": np.array([1.0])}, ["x > 0"])


# -- conditional_effects -------------------------------------------------------------------------------------------
def test_conditional_effects_on_every_case_it_takes():
    """On every frozen case whose model ``conditional_effects`` takes, each population-level predictor's table: the grid (100
    evenly spaced values over the fitted rows' range, or the levels reference first), the other predictors at their means or
    reference levels, the response at its mean, the grouping factors None, and the summaries (median, MAD, 2.5 and 97.5 %
    quantiles) of ``predict(kind="mean", include_group_effects=False)``'s draws at the grid's rows, which gate G4 checks
    against brms's ``posterior_epred(re_formula = NA)``."""
    taken = []
    for b in _ungated_bundles():
        m = b.model
        try:
            _refuse_unsupported_effects(m)
        except ValueError:
            continue
        s = m.spec
        if not s.pop_columns:
            continue
        draws = adapter.draws(b.fixture, m, b.cmap, b.gmap)
        res = ab.conditional_effects(m, draws, list(s.pop_columns))
        assert list(res) == list(s.pop_columns)
        cats = {c.name: c for c in s.categoricals}
        groups = [c for t in s.group_terms for c in t.operands if c not in s.pop_columns]
        for effect, table in res.items():
            others = [c for c in dict.fromkeys([s.response, *s.pop_columns, *groups]) if c != effect]
            assert list(table.columns) == [effect, *others, "cond__", "effect1__", "estimate__", "se__", "lower__", "upper__"]
            if effect in cats:
                assert list(table[effect]) == list(cats[effect].categories)
            else:
                x = m.data[effect].to_numpy(dtype=float)
                assert np.array_equal(table[effect].to_numpy(), np.linspace(x.min(), x.max(), 100))
            assert list(table["effect1__"]) == list(table[effect]) and set(table["cond__"]) == {"1"}
            for c in others:
                if c in groups:
                    assert table[c].isna().all()
                elif c in cats:
                    assert set(table[c]) == {cats[c].reference}
                else:
                    values = m.design.y if c == s.response else m.data[c].to_numpy(dtype=float)
                    assert np.allclose(table[c].to_numpy(dtype=float), np.mean(values), rtol=1e-14, atol=0)
            grid = pd.DataFrame({c: table[c].tolist() for c in s.pop_columns})
            e = ab.predict(m, draws, newdata=grid, kind="mean", include_group_effects=False)[m.names["mean"]]
            e = e.transpose("chain", "draw", "__obs__").to_numpy().reshape(-1, len(table))
            med = np.median(e, axis=0)
            assert np.array_equal(table["estimate__"].to_numpy(), med)
            assert np.array_equal(table["se__"].to_numpy(), 1.4826 * np.median(np.abs(e - med), axis=0))
            assert np.allclose(table["lower__"].to_numpy(), np.quantile(e, 0.025, axis=0), rtol=1e-14, atol=0)
            assert np.allclose(table["upper__"].to_numpy(), np.quantile(e, 0.975, axis=0), rtol=1e-14, atol=0)
        taken.append((b.case["id"], m.family))
    families = {f for _, f in taken}
    assert len(taken) >= 40 and {"gaussian", "student", "bernoulli", "poisson", "negbinomial", "gamma", "lognormal", "beta",
                                 "zero_inflated_poisson", "hurdle_poisson"} <= families


def test_conditional_effects_options():
    b = _bundle("gauss_hypothesis_effects")
    draws = gates.fixed_parameter_draws(b)
    res = ab.conditional_effects(b.model, draws, "x", resolution=7, prob=0.5)
    table = res["x"]
    assert list(res) == ["x"] and len(table) == 7 and table["g"].isna().all()
    e = ab.predict(b.model, draws, newdata=table[["x", "w", "z"]], kind="mean", include_group_effects=False)["mu"]
    e = e.transpose("chain", "draw", "__obs__").to_numpy().reshape(-1, 7)
    assert np.allclose(table["lower__"], np.quantile(e, 0.25, axis=0), rtol=1e-14) and np.allclose(table["upper__"], np.quantile(e, 0.75, axis=0), rtol=1e-14)
    assert table.attrs["fit_binding_status"] == "unverified"
    for bad in ({"resolution": 1}, {"resolution": 10.0}, {"resolution": True}, {"prob": 0}, {"prob": 1}, {"prob": "0.9"}):
        with pytest.raises(ValueError, match="resolution must be|prob must be"):
            ab.conditional_effects(b.model, draws, "x", **bad)
    for effects, match in ((["x:z:w"], "the effect 'x:z:w' interacts 3 predictors"),   # E05 lifted two-way interactions
                           (["g"], "'g' is a grouping factor"),
                           (["y"], "'y' is not a predictor of the population-level formula; its predictors are \\['x', 'w', 'z'\\]"),
                           (["nope"], "'nope' is not a predictor"),
                           (["x", "x"], "name a predictor twice"),
                           ([], "effects must be"), (None, "effects must be"), ([1], "effects must be")):
        with pytest.raises(ValueError, match=match):
            ab.conditional_effects(b.model, draws, effects)


@pytest.mark.parametrize("case_id, match", [
    ("binomial_varint", "conditional_effects\\(\\) of a binomial model is not offered: its expected response needs the trials"),
    ("poisson_offset", "the offset 'offset\\(log\\(exposure\\)\\)'"),
    ("gauss_monotonic", "the monotonic term 'mo\\(ord\\)'"),
    ("gauss_gp_exact", "the Gaussian process 'gp\\(ord\\)'"),
    ("gauss_me", "the measurement-error term 'me\\(xobs, sdx\\)'"),
    ("gauss_ar1", "the autocorrelation term 'ar\\(time = t, gr = g, p = 1\\)'"),
    ("gauss_distributional_sigma", "a distributional formula for sigma"),
    ("gauss_weighted", "the addition term\\(s\\) weights\\(w\\)"),
    ("gauss_censored", "the addition term\\(s\\) cens\\(cens_ind\\)"),
    ("mixture_gauss_gauss", "of a model with a mixture"),
    ("cumulative_varint", "the cumulative family's thresholds"),
    ("categorical_poponly", "the categorical family's categories"),
    ("gauss_mv_rescor", "of a multivariate model"),
    ("gauss_mi", "of a multivariate model"),
    ("gauss_nonlinear_exp", "of a non-linear model \\(nl=True\\)"),
])
def test_conditional_effects_refuses_what_no_case_verifies(case_id, match):
    case = fx.load_case(case_id)
    model = adapter.build_model(case, fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv"))
    with pytest.raises(ValueError, match=match) as refused:
        ab.conditional_effects(model, None, ["x"])
    assert str(refused.value).startswith("conditional_effects() of a") and "divergence D99" in str(refused.value)
    if case_id in ("mixture_gauss_gauss", "cumulative_varint", "categorical_poponly", "gauss_mv_rescor", "gauss_nonlinear_exp"):
        with pytest.raises(ValueError, match="hypothesis\\(\\) of .*divergence D99"):
            ab.hypothesis(model, None, ["x > 0"])


# -- the integrations of post-processing ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_id", ["gauss_mv_rescor", "gauss_mv_norescor", "gauss_mv_slopes"])
def test_post_processing_refuses_a_multivariate_model(case_id):
    """``log_lik``, ``bayes_R2`` and ``loo`` (through ``log_lik``) raised an AttributeError on a multivariate model; they now
    refuse it naming the construct, as ``waic`` does (divergence D95), and so do ``hypothesis`` and ``conditional_effects``."""
    b = _bundle(case_id)
    draws = gates.fixed_parameter_draws(b)
    for fn, what in ((ab.log_lik, "log_lik()"), (ab.bayes_R2, "bayes_R2()"), (ab.loo, "log_lik()"), (ab.waic, "waic()")):
        with pytest.raises(ValueError, match=re.escape(f"{what} of a multivariate model is not offered")):
            fn(b.model, draws)
    with pytest.raises(ValueError, match="hypothesis\\(\\) of a multivariate model"):
        ab.hypothesis(b.model, draws, "x > 0")
    with pytest.raises(ValueError, match="conditional_effects\\(\\) of a multivariate model"):
        ab.conditional_effects(b.model, draws, "x")


@pytest.mark.parametrize("case_id", ["gauss_nonlinear_exp", "gauss_nonlinear_terms", "mixture_gauss3_post", "mixture_gauss_gauss"])
def test_waic_of_non_linear_and_mixture_models_is_their_log_liks(case_id):
    """``log_lik`` of these models is checked against the graph's own density (test_nonlinear, test_mixture) and, for
    mixture_gauss3_post, against brms by the post gate; ``waic`` is WAIC of that log-likelihood."""
    b = _bundle(case_id)
    draws = gates.fixed_parameter_draws(b)
    res, ref = ab.waic(b.model, draws), waic_from_log_lik(ab.log_lik(b.model, draws))
    pd.testing.assert_frame_equal(res.pointwise, ref.pointwise)
    pd.testing.assert_frame_equal(res.estimates, ref.estimates)
