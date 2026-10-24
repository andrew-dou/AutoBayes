"""F07 (docs/summaries.md): brms's summaries beyond the oracle gate, which compares them with brms on the frozen cases
``gauss_varint_summaries``, ``bernoulli_slope_correlated_summaries`` and ``gauss_distributional_sigma_summaries``.

- ``autobayes.convergence``: ``rhat``, ``ess_bulk`` and ``ess_tail`` equal posterior 1.6.1's on the recorded draws of
  ``tests/summary_probes/posterior-1.6.1.json`` (chains of 1 to 101 draws, ties, constant and near-constant draws, NA and Inf),
  except chains of two or three draws (one draw per split chain), where posterior returns numbers and these NaN (D164); the
  rank-normalised R-hat equals ArviZ's; the pieces by hand.
- On the frozen cases: ``brms_draws`` equals brms's own draws array (every variable, in brms's order, ``lprior`` and ``lp__``
  included), and every summary equals brms's record of it.
- On the model structures of ``tests/summary_probes/brms-2.23.0-names.json`` (two grouping factors, independent slopes, three
  and four correlated coefficients, categorical predictors and interactions, a formula of sigma with several terms or none,
  no group-level terms): every label and order equals brms's, brms's refusals are refused; ``log_densities`` maps each draw
  back to the sampled coordinates exactly.
- The refusals, each naming the construct, and the options.
No sampler runs here: every posterior is a set of frozen draws or of points pushed through the model's own graph.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr
from scipy.special import ndtri

import autobayes as ab
from autobayes import convergence as cv
from autobayes import summaries as sm
from autobayes.inference import FitResult
from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.tolerances import rel_err

PROBES = Path(__file__).parent / "summary_probes"
POSTERIOR = json.loads((PROBES / "posterior-1.6.1.json").read_text())
NAMES = json.loads((PROBES / "brms-2.23.0-names.json").read_text())
CASES = ["gauss_varint_summaries", "bernoulli_slope_correlated_summaries", "gauss_distributional_sigma_summaries"]
SUMMARY_NAMES = ("fixef", "ranef", "coef", "VarCorr", "posterior_summary", "summary", "residuals")


def _number(text):
    if text in ("NA", "NaN"):
        return math.nan
    return float(text) if "Inf" in text else float.fromhex(text) if "0x" in text else float(text)


def _draws(probe) -> np.ndarray:
    """A probe's draws as ``(chains, draws)``: the record is R's ``(iterations, chains)``, column-major."""
    d = probe["draws"]
    return np.array([_number(v) for v in d["values"]]).reshape([int(k) for k in d["dims"]], order="F").T


def _same_number(ours, theirs, tol=1e-12) -> bool:
    if theirs is None or theirs == "NA" or (isinstance(theirs, float) and math.isnan(theirs)):
        return math.isnan(ours)
    theirs = float(theirs)
    return ours == theirs if math.isinf(theirs) else abs(ours - theirs) <= tol * max(1.0, abs(theirs))


# -- the convergence diagnostics -------------------------------------------------------------------------------------
def _one_draw_per_split_chain(probe) -> bool:
    return int(probe["draws"]["dims"][0]) in (2, 3)


@pytest.mark.parametrize("probe", POSTERIOR["probes"], ids=[p["label"] for p in POSTERIOR["probes"]])
def test_diagnostics_equal_posterior_1_6_1(probe):
    x = _draws(probe)
    ours = {"rhat": cv.rhat(x), "ess_bulk": cv.ess_bulk(x), "ess_tail": cv.ess_tail(x)}
    if _one_draw_per_split_chain(probe):                  # D164: posterior's numbers from one draw per split chain; NaN here
        assert all(math.isnan(v) for v in ours.values()) and probe["rhat"] not in (None, "NA")
        return
    for name, value in ours.items():
        assert _same_number(value, probe[name]), (name, value, probe[name])


def test_the_probes_reach_every_rule():
    """The record has chains of every length class, NA where posterior gives none, ties, Inf and the near-constant draws that
    set ess_tail's tolerance; and posterior 1.6.1, the fixtures' version."""
    assert POSTERIOR["posterior"] == fx.Fixture.load(CASES[0]).meta["posterior"] == "1.6.1"
    lengths = {int(p["draws"]["dims"][0]) for p in POSTERIOR["probes"]}
    assert {1, 2, 3, 4, 5, 6, 7, 9, 10, 50, 51, 52, 100, 101} <= lengths
    labels = {p["label"]: p for p in POSTERIOR["probes"]}
    assert labels["one draw 0.5 eps above 0"]["ess_tail"] is None or labels["one draw 0.5 eps above 0"]["ess_tail"] == "NA"
    assert float(labels["one draw 1 eps above 0"]["ess_tail"]) > 0 and labels["constant chains, different levels"]["rhat"] == "Inf"


@pytest.mark.parametrize("probe", [p for p in POSTERIOR["probes"] if int(p["draws"]["dims"][0]) % 2 == 0
                                   and int(p["draws"]["dims"][0]) >= 4 and int(p["draws"]["dims"][1]) >= 2
                                   and p["rhat"] not in (None, "NA", "Inf")], ids=lambda p: p["label"])
def test_rhat_equals_arviz_rank_normalised_rhat_on_even_chains(probe):
    """An independent implementation (ArviZ 0.23's ``rhat(method="rank")``, which folds at the median of the split draws: the
    same draws when no chain has a middle one; ArviZ takes two chains or more)."""
    x = _draws(probe)
    assert rel_err(cv.rhat(x), float(az.rhat(x, method="rank"))) <= 1e-12


def test_split_chains_and_rank_normalisation_by_hand():
    np.testing.assert_array_equal(cv.split_chains([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), [[1, 2], [6, 7], [4, 5], [9, 10]])
    np.testing.assert_allclose(cv.z_scale([[1.0, 2.0], [2.0, 3.0]]),
                               ndtri(np.array([[1 - 0.375, 2.5 - 0.375], [2.5 - 0.375, 4 - 0.375]]) / 4.25), rtol=1e-15)
    with pytest.raises(ValueError, match=r"\(chains, draws\)"):
        cv.rhat(np.zeros(10))


def test_autocovariance_is_the_direct_sum():
    x = np.random.default_rng(1).normal(size=(3, 17))
    c = x - x.mean(axis=1, keepdims=True)
    direct = np.array([[np.dot(c[m, :17 - t], c[m, t:]) / 17 for t in range(17)] for m in range(3)])
    np.testing.assert_allclose(cv._autocovariance(x), direct, atol=1e-14)


def test_ess_rules_by_hand():
    """Antithetic chains (+1, -1, ...): W = 1.2, var+ = 1, rho_1 = -5/6 - 0.2, so the first pair is negative, no pair is summed,
    tau = -1 + rho_0 = 0 and the ESS is capped at S log10 S; fewer than six draws per chain: S / 2; fewer than three or constant
    draws: none."""
    alt = np.array([[1.0, -1.0] * 3, [-1.0, 1.0] * 3])
    assert cv.ess_split(alt) == pytest.approx(12 * math.log10(12), rel=1e-15)
    assert cv.ess_split(np.arange(10.0).reshape(2, 5)) == 5.0
    assert math.isnan(cv.ess_split(np.arange(4.0).reshape(2, 2))) and math.isnan(cv.ess_split(np.ones((2, 8))))
    assert all(math.isnan(f(np.ones((2, 8)))) for f in (cv.rhat, cv.ess_bulk, cv.ess_tail))
    nan = np.arange(16.0).reshape(2, 8)
    nan[0, 3] = math.nan
    assert all(math.isnan(f(nan)) for f in (cv.rhat, cv.ess_bulk, cv.ess_tail))


def test_summarise_by_hand():
    """Mean, SD with divisor S - 1, and R's type 7 quantiles: 1 + 3 * 0.025 and 1 + 3 * 0.975 of 1, 2, 3, 4."""
    np.testing.assert_allclose(sm.summarise(np.array([[4.0], [1.0], [3.0], [2.0]])),
                               [[2.5, math.sqrt(5 / 3), 1.075, 3.925]], rtol=1e-15)


# -- the frozen cases --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=CASES)
def frozen(request):
    """A frozen case's bundle and its summary draws as a fit's posterior holds them (the gate's ``summary_draws_fit``)."""
    b = gates.Bundle(request.param)
    return b, gates.summary_draws_fit(b)


def test_brms_draws_equal_brms_draws_array(frozen):
    """Every variable of brms's draws array, in brms's order, at every draw: the coefficients, SDs, correlations and effects to
    1e-12, the centred intercepts, lprior and lp__ (the model's own log densities at the draw) to 1e-10 relative."""
    b, idata = frozen
    block = b.fixture.summary_draws
    ours = sm.brms_draws(b.model, idata)
    assert list(ours) == block["variables"]
    for k, name in enumerate(block["variables"]):
        theirs = block["draws"][:, :, k].T                                      # (chains, iterations)
        assert rel_err(ours[name], theirs) <= (1e-10 if name in ("lprior", "lp__") else 1e-12), name


def _same_table(frame, ref, tol=1e-12):
    values, (rows, columns) = fx.decode_labelled(ref)
    assert (list(frame.index) == rows or not rows) and list(frame.columns) == columns
    assert frame.shape == values.shape and rel_err(frame.to_numpy(), values) <= tol


def _same_array(frame, ref, tol=1e-12):
    values, (rows, stats, coefs) = fx.decode_labelled(ref)
    assert list(frame.index) == rows and list(frame.columns) == [(c, s) for c in coefs for s in stats]
    assert rel_err(frame.to_numpy().reshape(len(rows), len(coefs), len(stats)).transpose(0, 2, 1), values) <= tol


def test_summaries_equal_brms_on_the_frozen_cases(frozen):
    b, idata = frozen
    post = b.fixture.raw["post"]
    _same_table(ab.fixef(b.model, idata), post["fixef"])
    _same_table(ab.posterior_summary(b.model, idata), post["posterior_summary"], tol=1e-10)
    _same_table(ab.residuals(b.model, idata), post["residuals"])
    for name in ("ranef", "coef"):
        res = getattr(ab, name)(b.model, idata)
        assert list(res) == list(post[name])
        for g, array in post[name].items():
            _same_array(res[g], array)
    vc = ab.VarCorr(b.model, idata)
    assert list(vc) == list(post["VarCorr"])
    for g, parts in post["VarCorr"].items():
        assert list(vc[g]) == list(parts)
        for part, ref in parts.items():
            (_same_table if part == "sd" else _same_array)(vc[g][part], ref)
    s, ref = ab.summary(b.model, idata), post["summary"]
    for table in ("fixed", "spec_pars", "cor_pars"):
        _same_table(getattr(s, table), ref[table])
    assert list(s.random) == list(ref["random"])
    for g, table in ref["random"].items():
        _same_table(s.random[g], table)
    assert (s.nobs, list(s.ngrps.items()), s.chains, s.iter, s.warmup, s.thin, s.total_ndraws) == \
        (ref["nobs"], list(ref["ngrps"].items()), ref["chains"], ref["iter"], ref["warmup"], ref["thin"], ref["total_ndraws"])


def test_summary_prints_brms_s_layout(frozen):
    b, idata = frozen
    text = str(ab.summary(b.model, idata))
    assert text.startswith(f" Family: {b.model.family}\n") and "total post-warmup draws = 200" in text
    assert "Regression Coefficients:" in text and "Multilevel Hyperparameters:\n~g (Number of levels: 6)" in text
    assert ("Further Distributional Parameters:" in text) == ("sigma" in b.model.names and not b.model.spec.dpars)


def test_draw_counts_read_the_warmup(frozen):
    """A FitResult's tuning steps are its warmup, and so are an InferenceData's ``tuning_steps`` (PyMC's); none otherwise."""
    b, idata = frozen
    fit = FitResult(idata=idata, backend="pymc", device_requested="cpu", device_actual="cpu", platform="cpu", dtype="float64",
                    chain_method=None, chains=4, draws=50, tune=300, target_accept=0.9, wall_seconds=0.0)
    s = ab.summary(b.model, fit)
    assert (s.chains, s.iter, s.warmup, s.total_ndraws) == (4, 350, 300, 200)
    tuned = idata.copy()
    tuned.posterior.attrs["tuning_steps"] = 100
    assert (ab.summary(b.model, tuned).iter, ab.summary(b.model, tuned).warmup) == (150, 100)
    assert ab.fixef(b.model, fit).equals(ab.fixef(b.model, idata))


def test_residuals_take_brms_s_ordinary_expected_value_only(frozen):
    b, idata = frozen
    for kwargs in ({"type": "pearson"}, {"method": "posterior_predict"}, {"method": "posterior_linpred"}):
        with pytest.raises(ValueError, match="only type='ordinary' with method='posterior_epred'.*D163"):
            ab.residuals(b.model, idata, **kwargs)


# -- brms's names of further structures (the names probe) ---------------------------------------------------------------
NAMES_DATA = Path(__file__).parent.parent / NAMES["data"]


def _probe_model(probe):
    data = pd.read_csv(NAMES_DATA, dtype={"z": str, "g": str, "h": str})
    formula = probe["formula"][0] if len(probe["formula"]) == 1 else list(probe["formula"])
    categorical = {"z": "p"} if any("z" in f.split("~")[1] for f in probe["formula"]) else None
    kwargs = {"categorical": categorical} if categorical else {}
    priors = ab.default_priors(formula, data, family=probe["family"], **kwargs)
    return getattr(ab, probe["family"])(formula, data, priors, **kwargs)


def _synthetic_fit(model, chains=2, draws=3, seed=7):
    """Draws as a fit's posterior holds them, from value-space points (the initial point plus noise) pushed through the
    model's own graph: every variable the summaries read, the intercepts on the raw scale (``a_c - mean(X) b``), as Bambi
    reports them; and the value points."""
    pmm = model.pymc_model
    names = [model.names["intercept"], *dict.fromkeys(v for _, v, _ in model.coefficients)]
    names += [r[k] for r in model.group_coefficient_names.values() for k in ("sd", "group_effect")]
    names += [r[k] for r in model.correlation_names.values() for k in ("L", "cor")]
    names += [v for d in model.spec.dpars for k, v in model.dpar_names[d.response].items() if k != "value"]
    names += ["sigma"] if "sigma" in model.names and not model.spec.dpars else []
    outputs = pmm.replace_rvs_by_values([pmm[n] for n in names])        # each in terms of the value variables
    fn = pytensor.function(pmm.value_vars, outputs, on_unused_input="ignore")
    rng = np.random.default_rng(seed)
    start = pmm.initial_point()
    points, values = [], []
    for _ in range(chains * draws):
        u = {k: np.asarray(v + rng.normal(0.0, 0.4, np.shape(v))) for k, v in start.items()}
        points.append(u)
        values.append(dict(zip(names, fn(*(u[v.name] for v in pmm.value_vars)))))
    b = [np.atleast_1d(v[var])[j] if j is not None else float(v[var]) for v in values for _, var, j in model.coefficients]
    b = np.array(b).reshape(len(values), -1)
    for v, bv in zip(values, b):
        v[model.names["intercept"]] = float(v[model.names["intercept"]]) - (bv @ model.design.x_means if bv.size else 0.0)
        for d in model.spec.dpars:
            roles = model.dpar_names[d.response]
            bd = np.array([np.atleast_1d(v[roles[f"b:{t}"]])[j] if j is not None else float(v[roles[f"b:{t}"]])
                           for _, t, j in d.column_terms])
            v[roles["intercept"]] = float(v[roles["intercept"]]) - (bd @ model.design.dpars[d.response].x_means if bd.size else 0.0)
    data = {n: (("chain", "draw", *pmm.named_vars_to_dims.get(n, ())),
                np.stack([np.asarray(v[n], dtype=float) for v in values]).reshape(chains, draws, *np.shape(values[0][n])))
            for n in names}
    dims = {d for n in names for d in pmm.named_vars_to_dims.get(n, ())}
    coords = {d: list(pmm.coords[d]) for d in dims}
    posterior = xr.Dataset(data, coords={**coords, "chain": np.arange(chains), "draw": np.arange(draws)})
    return az.InferenceData(posterior=posterior), points


@pytest.fixture(scope="module", params=NAMES["probes"], ids=[p["id"] for p in NAMES["probes"]])
def probed(request):
    model = _probe_model(request.param)
    idata, points = _synthetic_fit(model)
    return request.param, model, idata, points


def _labels(frame) -> list[list[str]]:
    return [list(frame.index), list(frame.columns)]


def _array_labels(frame) -> list[list[str]]:
    coefs = list(dict.fromkeys(frame.columns.get_level_values(0)))
    return [list(frame.index), list(dict.fromkeys(frame.columns.get_level_values(1))), coefs]


def test_names_equal_brms_on_the_probed_structures(probed):
    probe, model, idata, _ = probed
    assert list(sm.brms_draws(model, idata)) == probe["variables"]
    assert _labels(ab.posterior_summary(model, idata)) == probe["posterior_summary"]
    assert _labels(ab.fixef(model, idata)) == probe["fixef"]
    for name in ("ranef", "coef"):
        if "error" in probe[name]:                             # brms refuses it for a model without group-level terms
            with pytest.raises(ValueError, match=re.escape(probe[name]["error"])):
                getattr(ab, name)(model, idata)
            continue
        res = getattr(ab, name)(model, idata)
        assert list(res) == list(probe[name]) == probe["ranef_groups"]
        assert {g: _array_labels(f) for g, f in res.items()} == probe[name]
    if "error" in probe["VarCorr"]:
        with pytest.raises(ValueError, match=re.escape(probe["VarCorr"]["error"])):
            ab.VarCorr(model, idata)
    else:
        vc = ab.VarCorr(model, idata)
        assert list(vc) == probe["VarCorr_groups"] and {g: list(p) for g, p in vc.items()} == probe["VarCorr_parts"]
        labels = {g: {k: (_labels(f) if k == "sd" else _array_labels(f)) for k, f in p.items()} for g, p in vc.items()}
        assert labels == probe["VarCorr"]
    s, ref = ab.summary(model, idata), probe["summary"]
    assert {t: _labels(getattr(s, t)) for t in ref["tables"]} == {t: ref[t] for t in ref["tables"]}
    assert list(s.random) == ref["random_groups"] and {g: _labels(f) for g, f in s.random.items()} == (ref["random"] or {})
    assert list(s.ngrps) == ref["ngrps_groups"] and list(s.ngrps.values()) == list((ref["ngrps"] or {}).values())
    assert s.nobs == ref["nobs"] and _labels(ab.residuals(model, idata))[1] == list(sm.STATS)


def test_log_densities_invert_the_sampled_coordinates(probed):
    """``log_densities`` maps each draw back to the point it came from: its lp__ is the graph's log density with the Jacobian
    there, and its lprior the graph's log density of the priors (every free variable but the standardised group-level effects)
    without it, to 1e-10."""
    _, model, idata, points = probed
    pmm = model.pymc_model
    offsets = {r["offset"] for r in model.group_coefficient_names.values()}
    lp = pmm.compile_logp()
    lprior = pmm.compile_logp(vars=[rv for rv in pmm.free_RVs if rv.name not in offsets], jacobian=False)
    ours_prior, ours_lp = sm.log_densities(model, idata.posterior)
    assert rel_err(ours_lp.reshape(-1), [float(lp(u)) for u in points]) <= 1e-10
    assert rel_err(ours_prior.reshape(-1), [float(lprior(u)) for u in points]) <= 1e-10


def test_the_names_probe_covers_its_structures():
    ids = {p["id"]: p for p in NAMES["probes"]}
    assert NAMES["brms"] == "2.23.0" and set(ids) == {"two_factors", "independent_slopes", "distributional_terms",
                                                     "distributional_intercept", "three_correlated", "four_correlated",
                                                     "population_only", "population_only_gaussian"}
    assert "cor_g__w__cnt" in ids["four_correlated"]["variables"] and "error" in ids["population_only"]["VarCorr"]


# -- refusals ----------------------------------------------------------------------------------------------------------
def _case_model(case_id):
    """A frozen case's model, its data2 matrices handed to the constructor as the oracle hands them (sar(M))."""
    case, d = fx.load_case(case_id), fx.fixture_dir(case_id)
    data2 = fx.read_data2(case, d)
    if data2:
        case = {**case, "autobayes": {**case["autobayes"], "kwargs": {**case["autobayes"].get("kwargs", {}), "data2": data2}}}
    return adapter.build_model(case, fx.read_frame(case, d / "data.csv"))


@pytest.mark.parametrize("case_id, construct", [
    ("gauss_mv_norescor", "a multivariate model"), ("gauss_nonlinear_exp", "a non-linear model"),
    ("poisson_varint", "the poisson family"), ("student_varint", "the student family"),
    ("gauss_no_intercept", "a formula without an intercept"), ("gauss_varint_uncentered", "a model with center_predictors=False"),
    ("gauss_horseshoe", "a shrinkage prior on b"), ("gauss_monotonic", "a formula with monotonics"),
    ("gauss_gp_exact", "a formula with gp_terms"), ("gauss_ar1", "a formula with autocor"), ("gauss_sar_lag", "a formula with sar"),
    ("gauss_weighted", "a formula with additions"), ("gauss_me", "the measurement_error term"),
    ("gauss_fcor", "a formula with fcor"), ("gauss_car_icar", "a formula with car"), ("gauss_unstr", "a formula with timecor"),
    ("gauss_cosy", "a formula with timecor"), ("multinomial_varint", "the multinomial family"),
    ("dirichlet_varint", "the dirichlet family"), ("cumulative_thres_gr", "the cumulative family"),
    ("gauss_se_meta", "a formula with additions"),
    ("gauss_mm", r"the group-level term \(1 \| mm\(.*\(members, weights\)|the group-level term \(1 \| mm\(.*\(members\)"),
    ("gauss_gr_by", r"the group-level term \(1 \| gr\(g, by = .*\(by\)"),
    ("gauss_shared_id", "the distributional formula 'sigma ~ "),
    ("gauss_nested_colon", r"the group-level term \(1 \| g:h\) \(a grouping factor of two columns\)"),
])
def test_unverified_structures_are_refused_by_name(case_id, construct):
    model = _case_model(case_id)
    for fn in (ab.fixef, ab.ranef, ab.coef, ab.VarCorr, ab.posterior_summary, ab.summary, ab.residuals, sm.brms_draws):
        with pytest.raises(ValueError, match=f"of {construct}.*divergence D162"):
            fn(model, None)


def test_the_families_taken_are_those_the_frozen_cases_verify():
    from autobayes.families import FAMILIES

    assert [name for name, fam in FAMILIES.items() if fam.summaries] == ["gaussian", "bernoulli"]
    assert all(ab.bernoulli("yb ~ x", pd.read_csv(NAMES_DATA), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1)),
                            link=link).family_spec.summaries for link in ("probit", "cloglog"))


def test_an_offset_and_a_sigma_formula_without_intercept_are_refused():
    data = pd.read_csv(NAMES_DATA, dtype={"z": str, "g": str, "h": str})
    priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    offset = ab.gaussian("y ~ x + offset(w) + (1 | g)", data, priors)
    with pytest.raises(ValueError, match="of a formula with offset .*D162"):
        ab.fixef(offset, None)
    no_intercept = ab.gaussian(["y ~ x", "sigma ~ 0 + x"], data,
                               ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma={"b": ab.Normal(0, 1)}))
    with pytest.raises(ValueError, match="of the distributional formula 'sigma ~ 0 \\+ x'.*D162"):
        ab.summary(no_intercept, None)
