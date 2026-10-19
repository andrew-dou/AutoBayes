"""E05 (docs/post-processing.md): prior draws, Savage-Dickey evidence ratios and interaction effects beyond the oracle gate,
which compares them with brms on the frozen cases ``gauss_hypothesis_prior_effects`` and ``bernoulli_hypothesis_prior_effects``.

- ``autobayes.kde``: brms 2.23's ``density_ratio()`` (and R 4.5's ``bw.nrd0()``) on recorded draws
  (``tests/hypothesis_probes/brms-2.23.0-density-ratio.json``: 14 sets of 2 to 4000 draws, ties, constant draws, skewed and
  heavy-tailed draws, scales from 1e-5 to 1e6; 276 points inside the draws, at their ends, mid-cell next to the ends, just and
  far outside), every density to ``1e-13`` relative plus ``1e-15`` of the estimate's peak, every ratio where both densities
  are above round-off; what brms refuses is refused.
- ``ab.prior_draws``: brms 2.23's ``prior_draws()`` names and order on five fixed-parameter fits
  (``brms-2.23.0-prior-draws.json``: a class-wide and coefficient-specific ``b``, group-wide and coefficient-specific SDs,
  correlations of two and three coefficients, two grouping factors, the Student-t ``nu``, ``shape``, ``hu``, ``phi``, ``zi``);
  their distributions: brms's 1000 recorded draws of every column and 20000 of ``ab.prior_draws``'s follow the prior's
  distribution (one-sample Kolmogorov-Smirnov, p > 1e-3); options and refusals.
- ``ab.hypothesis(..., prior=)``: brms's evidence ratios of those fits recomputed from the recorded posterior and prior
  draws, every one-parameter point hypothesis to 1e-12; ``Intercept`` keeps NaN (brms's NA, with brms's reason); a
  hypothesis on two parameters pairs their prior draws through a seeded permutation (brms's is R's); options and refusals.
- ``ab.conditional_effects`` of interaction effects: the grids brms builds (``brms-2.23.0-effects-grid.json``: numeric and
  categorical effects and moderators in both orders, the moderator at six scales) and R's labels of 3313 values; on every
  frozen case it takes, the grid and summaries of every pair of predictors; refusals.
No sampler runs here: every posterior is a set of frozen or recorded draws.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import autobayes as ab
from autobayes.conditional_effects import _refuse_unsupported_effects, r_number_label
from autobayes.hypothesis import INTERCEPT_NO_PRIOR, coefficient_prior_draws, hypothesis_from_draws, parse_hypothesis
from autobayes.kde import bw_nrd0, density_at, density_ratio
from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import gates

PROBES = Path(__file__).parent / "hypothesis_probes"
DENSITY = json.loads((PROBES / "brms-2.23.0-density-ratio.json").read_text())
PRIOR = json.loads((PROBES / "brms-2.23.0-prior-draws.json").read_text())
GRID = json.loads((PROBES / "brms-2.23.0-effects-grid.json").read_text())


def _numbers(values) -> np.ndarray:
    return np.array([math.nan if v in ("NA", "NaN") else float(v) for v in values], dtype=float)


# -- the kernel density at a point (autobayes.kde) ------------------------------------------------------------------
@pytest.mark.parametrize("name", list(DENSITY["sets"]))
def test_density_and_ratio_equal_brms_on_recorded_draws(name):
    """Each density to 1e-13 relative plus 1e-15 of its estimate's peak (the transform's round-off, brms's and ours); a ratio
    to the sum of its two densities' tolerances where both are above 1e-12 of their peaks; below that, far in a tail, both
    are round-off and only ours is checked to be round-off too."""
    s = DENSITY["sets"][name]
    x, y, points = _numbers(s["x"]), _numbers(s["y"]), _numbers(s["points"])
    assert bw_nrd0(x) == pytest.approx(float(s["bw_x"]), rel=2e-15, abs=0)   # R's SD in another order
    if s.get("bw_y") is not None:
        assert bw_nrd0(y) == pytest.approx(float(s["bw_y"]), rel=2e-15, abs=0)
    rel = {}
    for draws, key in ((x, "density_x"), (y, "density_y")):
        theirs = _numbers(s[key])
        for k, (point, ref) in enumerate(zip(points, theirs)):
            ours, peak = density_at(draws, point)
            assert abs(ours - ref) <= 1e-13 * abs(ref) + 1e-15 * peak, (key, point, ours, ref)
            rel[key, k] = (ref, peak)
    for k, (point, ref) in enumerate(zip(points, _numbers(s["ratio"]))):
        (dx, px), (dy, py) = rel["density_x", k], rel["density_y", k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = density_ratio(x, y, point)
        if dx > 1e-12 * px and dy > 1e-12 * py:
            tol = 2e-13 + 1e-15 * (px / dx + py / dy)
            assert abs(ours - ref) <= tol * abs(ref), (point, ours, ref)
        elif dx > 1e-12 * px:
            assert density_at(y, point)[0] <= 1e-12 * py                 # the prior's density is round-off, in both


def test_density_ratio_settles_the_grid_rules():
    """The probes' points settle brms's rules: the range is the draws' own, extended to ``point -/+ sd / 4`` outside them
    (the range ``[point, max]`` misses by 1e-6), and the spline is R's FMM one (not-a-knot misses next to the ends)."""
    from autobayes.kde import _binned_density, _fmm_spline_at
    from scipy.interpolate import CubicSpline

    s = DENSITY["sets"]["gate_like_8"]
    y, points, theirs = _numbers(s["y"]), _numbers(s["points"]), _numbers(s["density_y"])
    k = [i for i, p in enumerate(points) if y.min() - 1 < p < y.min()][0]  # a point below the draws
    grid, dens = _binned_density(y, bw_nrd0(y), points[k], y.max(), 4096)   # [point, max]
    assert abs(dens[0] - theirs[k]) > 1e-7 * theirs[k]
    x, theirs_x = _numbers(s["x"]), _numbers(s["density_x"])
    grid, dens = _binned_density(x, bw_nrd0(x), x.min(), x.max(), 4096)
    near_end = [i for i, p in enumerate(points) if x.min() < p < x.min() + 3 * (x.max() - x.min()) / 4095]
    assert len(near_end) >= 3
    for i in near_end:
        assert abs(_fmm_spline_at(grid, dens, points[i]) - theirs_x[i]) <= 1e-13 * theirs_x[i]
    assert max(abs(float(CubicSpline(grid, dens)(points[i])) - theirs_x[i]) / theirs_x[i] for i in near_end) > 1e-10


def test_density_ratio_refusals_and_tail_warning():
    for name, rec in DENSITY["refused"].items():
        assert rec["error"]                                                 # brms stops on each
        with pytest.raises(ValueError, match="at least two draws|must be finite"):
            density_ratio(_numbers(rec["x"]), point=0.0)
    with pytest.raises(ValueError, match="the point must be finite"):
        density_at([1.0, 2.0], math.inf)
    with pytest.warns(UserWarning, match="far in the draws' tail"):
        density_ratio([0.0, 0.1, 0.3], [0.2, 0.5, 0.9], 50.0)


# -- ab.prior_draws -------------------------------------------------------------------------------------------------
def _probe_data() -> pd.DataFrame:
    d = PRIOR["data"]
    frame = pd.DataFrame({k: (v if k in ("z", "g", "h") else _numbers(v)) for k, v in d.items()})
    frame["count"] = frame["count"].astype(int)
    return frame


def _probe_models() -> dict:
    """The AutoBayes models of the probe's five brms fits: the same data and priors (a coefficient-specific prior in brms is a
    mapping by term here; a group-wide SD prior is a mapping whose priors are equal)."""
    data, N, HN = _probe_data(), ab.Normal, ab.HalfNormal
    return {
        "gaussian_slopes": ab.gaussian("y ~ x * z + w + (1 + x | g)", data, ab.Priors(
            intercept=N(0, 5), b={"x": N(0, 2), "z": N(0, 2), "w": N(1, 0.5), "x:z": {"q": N(0, 1), "r": N(0, 2)}},
            sd={"g": {"Intercept": HN(1), "x": HN(0.5)}}, cor={"g": ab.LKJ(2)}, sigma=HN(1)), categorical={"z": "p"}),
        "student": ab.student("y ~ x + (1 | g)", data, ab.Priors(intercept=N(0, 5), b=N(0, 2), sd=HN(1), sigma=HN(1),
                                                                 nu=ab.Gamma(2, 0.1)), nu=None),
        "hurdle_negbinomial": ab.hurdle_negbinomial("count ~ x + (1 | g)", data, ab.Priors(
            intercept=N(0, 2), b=N(0, 1), sd=HN(1), shape=ab.Gamma(2, 1), hu=ab.Beta(2, 2))),
        "zero_inflated_beta": ab.zero_inflated_beta("prop ~ x + w", data, ab.Priors(
            intercept=N(0, 2), b=N(0, 1), phi=ab.Gamma(2, 0.5), zi=ab.Beta(1, 3))),
        "gaussian_two_groups": ab.gaussian("y ~ x + w + (1 + x + w | g) + (1 | h)", data, ab.Priors(
            intercept=N(0, 5), b=N(0, 2), sd={"g": {"Intercept": HN(1), "x": HN(1), "w": HN(1)}, "h": ab.Exponential(2)},
            cor={"g": ab.LKJ(2)}, sigma=HN(1))),
    }


@pytest.fixture(scope="module")
def probe_models():
    return _probe_models()


def _prior_cdfs() -> dict:
    """The distribution of each recorded column, from the probe's brms priors (a bounded class truncated as brms bounds it)."""
    norm, half = stats.norm, stats.halfnorm
    lkj = lambda eta, d: (lambda v: stats.beta.cdf((v + 1) / 2, eta - 1 + d / 2, eta - 1 + d / 2))
    nu = stats.gamma(2, scale=10)
    return {
        "gaussian_slopes": {"Intercept": norm(0, 5).cdf, "b_x": norm(0, 2).cdf, "b_zq": norm(0, 2).cdf, "b_zr": norm(0, 2).cdf,
                            "b_w": norm(1, 0.5).cdf, "b_x:zq": norm(0, 1).cdf, "b_x:zr": norm(0, 2).cdf, "sigma": half(0, 1).cdf,
                            "sd_g__Intercept": half(0, 1).cdf, "sd_g__x": half(0, 0.5).cdf, "cor_g": lkj(2, 2)},
        "student": {"Intercept": norm(0, 5).cdf, "b": norm(0, 2).cdf, "sigma": half(0, 1).cdf, "sd_g": half(0, 1).cdf,
                    "nu": lambda v: (nu.cdf(v) - nu.cdf(1)) / nu.sf(1)},
        "hurdle_negbinomial": {"Intercept": norm(0, 2).cdf, "b": norm(0, 1).cdf, "shape": stats.gamma(2, scale=1).cdf,
                               "hu": stats.beta(2, 2).cdf, "sd_g": half(0, 1).cdf},
        "zero_inflated_beta": {"Intercept": norm(0, 2).cdf, "b": norm(0, 1).cdf, "phi": stats.gamma(2, scale=2).cdf,
                               "zi": stats.beta(1, 3).cdf},
        "gaussian_two_groups": {"Intercept": norm(0, 5).cdf, "b": norm(0, 2).cdf, "sigma": half(0, 1).cdf, "sd_g": half(0, 1).cdf,
                                "cor_g": lkj(2, 3), "sd_h": stats.expon(scale=0.5).cdf},
    }


@pytest.mark.parametrize("name", list(PRIOR["models"]))
def test_prior_draws_names_and_distributions_equal_brms(name, probe_models):
    """``ab.prior_draws`` has brms's columns in brms's order; brms's 1000 recorded draws of each and 20000 of ours follow the
    prior's distribution (so each column is the prior brms samples, and the draws are right: one-sample Kolmogorov-Smirnov,
    p above 1e-3, fixed seeds)."""
    rec = PRIOR["models"][name]
    ours = ab.prior_draws(probe_models[name], draws=20000, seed=5)
    assert list(ours.columns) == rec["prior_draws"]["columns"] == rec["long_prior_draws"]["columns"]
    cdfs = _prior_cdfs()[name]
    assert set(cdfs) == set(ours.columns)
    for column, cdf in cdfs.items():
        theirs = _numbers(rec["long_prior_draws"]["values"][column])
        assert theirs.size == 1000
        assert stats.kstest(theirs, cdf).pvalue > 1e-3, (column, "brms")
        assert stats.kstest(ours[column].to_numpy(), cdf).pvalue > 1e-3, (column, "ours")


def test_prior_draws_options_and_result(probe_models):
    m = probe_models["student"]
    a, b = ab.prior_draws(m, draws=50, seed=3), ab.prior_draws(m, draws=50, seed=3)
    assert a.equals(b) and a.shape == (50, 5) and not a.equals(ab.prior_draws(m, draws=50, seed=4))
    assert a.attrs["autobayes_generation_kind"] == "prior_draws" and a.attrs["autobayes_requested_draws"] == 50
    assert json.loads(a.attrs["autobayes_built_target_identity_json"]) == json.loads(json.dumps(m.target_identity, sort_keys=True))
    assert (a["nu"] > 1).all() and (a["sigma"] > 0).all()
    for bad in (1, 0, 2.5, True, "10"):
        with pytest.raises(ValueError, match="draws must be a whole number of at least 2"):
            ab.prior_draws(m, draws=bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
            ab.prior_draws(m, draws=10, seed=bad)


def test_prior_draws_leave_out_a_flat_prior():
    """brms does not sample an improper prior: a flat ``b`` has no column (and a point hypothesis on it no ratio)."""
    data = _probe_data()
    m = ab.gaussian("y ~ x + w", data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Flat(), sigma=ab.HalfNormal(1)))
    assert list(ab.prior_draws(m, draws=10, seed=1).columns) == ["Intercept", "sigma"]
    m = ab.gaussian("y ~ x + w", data, ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Flat(), "w": ab.Normal(0, 1)},
                                                 sigma=ab.HalfNormal(1)))
    assert list(ab.prior_draws(m, draws=10, seed=1).columns) == ["Intercept", "b_w", "sigma"]


@pytest.mark.parametrize("case_id, match", [
    ("gauss_mv_rescor", "several responses"),
    ("gauss_nonlinear_exp", "a non-linear formula"),
    ("mixture_gauss_gauss", "a mixture"),
    ("cumulative_varint", "thresholds"),
    ("categorical_poponly", "categories"),
    ("gauss_distributional_sigma", "a distributional formula for sigma"),
    ("gauss_monotonic", "the monotonic term 'mo\\(ord\\)'"),
    ("gauss_gp_exact", "the Gaussian process"),
    ("gauss_me", "several responses|the measurement-error term"),
    ("gauss_ar1", "the autocorrelation term"),
    ("gauss_varint_uncentered", "center_predictors=False"),
    ("skew_normal_varint", "the skew_normal family's alpha"),
    ("gauss_horseshoe", "a Horseshoe\\(\\) prior on b"),
    ("gauss_mm", "mm\\(\\), gr\\(by = \\) or a shared ID"),
])
def test_prior_draws_refuse_what_no_probe_verifies(case_id, match):
    if not fx.has_fixture(case_id):
        pytest.skip(f"{case_id} has no frozen fixture")
    case = fx.load_case(case_id)
    model = adapter.build_model(case, fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv"))
    with pytest.raises(ValueError, match=match) as refused:
        ab.prior_draws(model, draws=10)
    assert str(refused.value).startswith("prior_draws() of a model with") and "divergence D124" in str(refused.value)


# -- Savage-Dickey evidence ratios ----------------------------------------------------------------------------------
HYPOTHESES = {                                  # what probe_prior_draws.R asked brms, in its order
    "gaussian_slopes": ["Intercept = 0", "x = 0", "w = 1", "x:zq = 0", "zr = 0.2", "x - w = 0", "x > 0", "Intercept + x = 1"],
    "student": ["x = 0", "Intercept = 0.5"],
    "zero_inflated_beta": ["x = 0", "w = 0", "Intercept = 0"],
}


def _probe_hypothesis_inputs(name, probe_models):
    rec = PRIOR["models"][name]
    draws = {c[2:] if c.startswith("b_") else c: _numbers(v) for c, v in rec["draws"]["values"].items() if c.startswith("b_")}
    prior = pd.DataFrame({c: _numbers(v) for c, v in rec["prior_draws"]["values"].items()})
    return rec, draws, prior


@pytest.mark.parametrize("name", [k for k, v in PRIOR["models"].items() if "hypothesis" in v])
def test_evidence_ratios_equal_brms_on_recorded_draws(name, probe_models):
    """brms's hypothesis() of each fit under two seeds, recomputed from its recorded posterior and prior draws: every column
    of every hypothesis to 1e-12, except that a point hypothesis on two parameters has brms's ratio only in distribution
    (brms pairs the prior draws through R's RNG; its two seeds differ, ours is finite and reproducible); ``Intercept`` keeps
    brms's NA."""
    rec, draws, prior = _probe_hypothesis_inputs(name, probe_models)
    coef_prior, no_prior = coefficient_prior_draws(probe_models[name], prior, draws)
    assert no_prior["Intercept"] == INTERCEPT_NO_PRIOR
    texts = HYPOTHESES[name]
    parsed = [parse_hypothesis(t) for t in texts]
    intercept = [i for i, h in enumerate(parsed) if not h.one_sided and "Intercept" in h.names]
    several = [i for i, h in enumerate(parsed) if not h.one_sided and len(h.names) > 1 and i not in intercept]
    assert intercept
    for run in rec["hypothesis"]:
        table = run["table"]["values"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = hypothesis_from_draws(draws, texts, prior=coef_prior, no_prior=no_prior, seed=run["seed"])
        assert len([w for w in caught if "raw-scale intercept" in str(w.message)]) == len(intercept)
        assert list(res.table["Hypothesis"]) == table["Hypothesis"]
        np.testing.assert_allclose(res.samples.to_numpy(), np.column_stack([_numbers(v) for v in run["samples"]["values"].values()]),
                                   rtol=1e-15, atol=1e-15)
        for column in ("Estimate", "Est.Error", "CI.Lower", "CI.Upper", "Evid.Ratio", "Post.Prob"):
            ours, theirs = res.table[column].to_numpy(), _numbers(table[column])
            for i in range(len(texts)):
                if column in ("Evid.Ratio", "Post.Prob") and i in several:
                    assert np.isfinite(ours[i]) and np.isfinite(theirs[i])
                elif math.isnan(theirs[i]):
                    assert math.isnan(ours[i]), (texts[i], column)
                else:
                    assert ours[i] == pytest.approx(theirs[i], rel=1e-12, abs=0), (texts[i], column)
        assert list(res.table["Star"]) == table["Star"]
    if name == "gaussian_slopes":                                     # brms's two-parameter ratio moves with its seed
        er = [_numbers(r["table"]["values"]["Evid.Ratio"]) for r in rec["hypothesis"]]
        i = rec["hypothesis"][0]["table"]["values"]["Hypothesis"].index("(x-w) = 0")
        assert er[0][i] != er[1][i] and abs(er[0][1] - er[1][1]) <= 1e-15 * er[0][1]


def test_hypothesis_with_prior_draws_options(probe_models):
    m = probe_models["gaussian_slopes"]
    b = gates.Bundle("gauss_hypothesis_prior_effects")
    idata = gates.fixed_parameter_draws(b)
    prior = gates.recorded_prior_draws(b)
    one = ab.hypothesis(b.model, idata, ["x = 0", "x > 0"], prior=prior, seed=1)
    assert one.prior and np.isfinite(one.table["Evid.Ratio"]).all() and "Savage-Dickey" in str(one)
    assert one.table.loc[0, "Post.Prob"] == pytest.approx(one.table.loc[0, "Evid.Ratio"] / (1 + one.table.loc[0, "Evid.Ratio"]))
    assert ab.hypothesis(b.model, idata, ["x = 0"], prior=prior, seed=99).table.loc[0, "Evid.Ratio"] == pytest.approx(
        one.table.loc[0, "Evid.Ratio"], rel=1e-14)                      # one parameter: the seed moves nothing
    two = [ab.hypothesis(b.model, idata, ["x - w = 0"], prior=prior, seed=s).table.loc[0, "Evid.Ratio"] for s in (1, 1, 2)]
    assert two[0] == two[1] != two[2]                                     # several: the seed fixes the pairing
    with pytest.warns(UserWarning, match="raw-scale intercept"):
        h = ab.hypothesis(b.model, idata, ["Intercept = 0", "Intercept < 0"], prior=prior)
    assert np.isnan(h.table.loc[0, "Evid.Ratio"]) and np.isfinite(h.table.loc[1, "Evid.Ratio"])
    no_b = prior.drop(columns="b")
    with pytest.warns(UserWarning, match="no column b_x or b"):
        assert np.isnan(ab.hypothesis(b.model, idata, ["x = 0"], prior=no_b).table.loc[0, "Evid.Ratio"])
    for bad, match in (([1, 2], "prior must be the prior draws by name"), (prior.iloc[:1], "at least two"),
                       (prior.assign(b=np.nan), "are not all finite")):
        with pytest.raises(ValueError, match=match):
            ab.hypothesis(b.model, idata, ["x = 0"], prior=bad)
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        ab.hypothesis(b.model, idata, ["x = 0"], prior=prior, seed=-1)
    with pytest.raises(ValueError, match="not finite at 8 of 8 prior draws"):
        ab.hypothesis(b.model, idata, ["log(x + 100) = 0"], prior=prior.assign(b=-200.0))
    case = fx.load_case("gauss_varint_uncentered")
    uncentred = adapter.build_model(case, fx.read_frame(case, fx.fixture_dir("gauss_varint_uncentered") / "data.csv"))
    with pytest.raises(ValueError, match="center_predictors=False is not offered"):
        ab.hypothesis(uncentred, None, ["x = 0"], prior=prior)
    assert ab.prior_draws(m, draws=10).shape == (10, 11)


# -- interaction effects ---------------------------------------------------------------------------------------------
def _grid_data(name) -> pd.DataFrame:
    d = GRID["sets"][name]["data"]
    return pd.DataFrame({k: (v["values"] if v["type"] == "text" else _numbers(v["values"])) for k, v in d.items()})


def _forward_draws(model, draws=4):
    """A posterior of ``draws`` forward draws from the priors (only the grid is compared, not the summaries)."""
    import arviz as az

    return az.InferenceData(posterior=model.prior_predictive(draws=draws, seed=3).prior)


@pytest.mark.parametrize("name", list(GRID["sets"]))
def test_interaction_grids_equal_brms(name):
    """Every column brms's conditional_effects() writes but the summaries, for every effect of every call: numeric and
    categorical effects and moderators in both orders, and a numeric moderator at six scales, its labels R's."""
    data = _grid_data(name)
    m = ab.gaussian("y ~ x * w + z + h", data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)),
                    categorical={"z": "p", "h": "s"})
    draws = _forward_draws(m)
    for run in GRID["sets"][name]["runs"]:
        effects = [e for e in run["effects"]]
        res = ab.conditional_effects(m, draws, effects)
        assert list(res) == effects
        for effect, rec in run["tables"].items():
            table = res[effect]
            assert list(table.columns) == rec["columns"], effect
            for column, theirs in rec["grid"].items():
                ours = table[column].tolist()
                if theirs["type"] == "text":
                    assert [None if v is None else str(v) for v in ours] == theirs["values"], (effect, column)
                else:
                    # R's mean differs from the correctly rounded one by a few ulps (R's long double is a double here)
                    np.testing.assert_allclose(np.asarray(ours, dtype=float), _numbers(theirs["values"]), rtol=4e-15, atol=0)


def test_r_number_labels_equal_r():
    labels = GRID["labels"]
    assert len(labels["values"]) == 3313
    assert [r_number_label(float(v)) for v in labels["values"]] == labels["text"]


def _ungated_bundles():
    for cid in fx.case_ids():
        case = fx.load_case(cid)
        if not fx.has_fixture(cid) or "autobayes" not in case or (case.get("gate") or {}).get("expect") == "fail":
            continue
        yield gates.Bundle(cid)


def test_interaction_effects_on_every_case_they_take():
    """On every frozen case whose model ``conditional_effects`` takes, with two predictors or more: the first two predictors as
    ``a:b`` and ``b:a``, the grid (a's values, b's levels or mean -/+ SD, b fastest), the values held, ``effect2__`` and the
    summaries of ``predict(kind="mean", include_group_effects=False)`` at the grid's rows."""
    taken = []
    for b in _ungated_bundles():
        m = b.model
        try:
            _refuse_unsupported_effects(m)
        except ValueError:
            continue
        s = m.spec
        if len(s.pop_columns) < 2:
            continue
        first, second = s.pop_columns[:2]
        draws = adapter.draws(b.fixture, m, b.cmap, b.gmap)
        cats = {c.name: c for c in s.categoricals}
        for e1, e2 in ((first, second), (second, first)):
            table = ab.conditional_effects(m, draws, f"{e1}:{e2}")[f"{e1}:{e2}"]
            if e1 in cats:
                g1 = list(cats[e1].categories)
            else:
                x = m.data[e1].to_numpy(dtype=float)
                g1 = list(np.linspace(x.min(), x.max(), 100))
            if e2 in cats:
                g2, labels = list(cats[e2].categories), list(cats[e2].categories)
            else:
                v = m.data[e2].to_numpy(dtype=float)
                g2 = [v.mean() - v.std(ddof=1), v.mean(), v.mean() + v.std(ddof=1)]
                labels = [r_number_label(u) for u in g2]
            assert len(table) == len(g1) * len(g2)
            assert table.columns[:2].tolist() == [e1, e2] and "effect2__" in table.columns
            if e1 in cats:
                assert table[e1].tolist() == [u for u in g1 for _ in g2]
            else:
                np.testing.assert_array_equal(table[e1].to_numpy(dtype=float), np.repeat(g1, len(g2)))
            if e2 in cats:
                assert table[e2].tolist() == g2 * len(g1)
            else:
                np.testing.assert_allclose(table[e2].to_numpy(dtype=float), np.tile(g2, len(g1)), rtol=1e-14, atol=0)
            assert table["effect2__"].tolist() == labels * len(g1)
            grid = pd.DataFrame({c: table[c].tolist() for c in s.pop_columns})
            e = ab.predict(m, draws, newdata=grid, kind="mean", include_group_effects=False)[m.names["mean"]]
            e = e.transpose("chain", "draw", "__obs__").to_numpy().reshape(-1, len(table))
            assert np.array_equal(table["estimate__"].to_numpy(), np.median(e, axis=0))
        taken.append(m.family)
    assert len(taken) >= 20 and {"gaussian", "bernoulli", "poisson"} <= set(taken)


def test_interaction_effects_refusals():
    b = gates.Bundle("gauss_hypothesis_effects")
    draws = gates.fixed_parameter_draws(b)
    for effects, match in ((["x:z:w"], "interacts 3 predictors"), (["x:x"], "names 'x' twice"),
                           (["x:g"], "'g' is a grouping factor"), (["x:nope"], "'nope' is not a predictor"),
                           (["x:z", "z:x"], "a pair of predictors twice"), (["x", "x:z", "x"], "name a predictor")):
        with pytest.raises(ValueError, match=match):
            ab.conditional_effects(b.model, draws, effects)
    table = ab.conditional_effects(b.model, draws, "z:w", prob=0.5)["z:w"]
    assert len(table) == 9 and table["g"].isna().all() and table.columns[:3].tolist() == ["z", "w", "y"]
