"""The harness itself: frozen fixtures are what their cases say, the brms side agrees with the hand-written
reference densities of tests/reference_logp.py and tests/reference_binomial.py (so the oracle is calibrated
independently of autobayes), and a corrupted fixture fails every gate it touches. Two-reference acceptance (P6): the
committed 50-digit references regenerate byte for byte from the fixtures they name, and the three rules hold."""
import copy
import json
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

from tests.oracle import datasets
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle import adapter, two_reference
from tests.oracle.adapter import OracleMismatch
from tests.oracle.tolerances import GRAD_REL, LOGP_REL, rel_err
from tests.reference_binomial import reference_grad_binomial, reference_logp_binomial
from tests.reference_logp import reference_logp

FROZEN = [cid for cid in fx.case_ids() if fx.has_fixture(cid)]


@pytest.mark.parametrize("case_id", FROZEN)
def test_frozen_fixture_matches_its_case(case_id):
    case = fx.load_case(case_id)
    d = fx.fixture_dir(case_id)
    assert (d / "case_brms.json").read_text(encoding="utf-8") == fx.canonical_json(fx.frozen_section(case))
    meta = fx.Fixture.load(case_id).meta
    assert meta["id"] == case_id
    assert meta["case_brms_sha256"] == fx.sha256(d / "case_brms.json")
    assert meta["data_sha256"] == fx.sha256(d / "data.csv")
    if (d / "newdata.csv").exists():
        assert meta["newdata_sha256"] == fx.sha256(d / "newdata.csv")
    else:
        assert meta.get("newdata_sha256") is None
    assert meta["n_points"] == len(fx.Fixture.load(case_id).points) == fx.frozen_section(case)["n_points"]


@pytest.mark.parametrize("case_id", FROZEN)
def test_frozen_data_regenerates_byte_for_byte(case_id, tmp_path):
    case = fx.load_case(case_id)
    data, nd = datasets.make(case["data"])
    datasets.write_csv(data, tmp_path / "data.csv")
    assert (tmp_path / "data.csv").read_bytes() == (fx.fixture_dir(case_id) / "data.csv").read_bytes()
    if nd is not None:
        datasets.write_csv(nd, tmp_path / "newdata.csv")
        assert (tmp_path / "newdata.csv").read_bytes() == (fx.fixture_dir(case_id) / "newdata.csv").read_bytes()
    missing = list(datasets.NA_COLUMNS.get(case["data"]["generator"], ()))   # missing by design only (tier_c's xm, for mi())
    assert not data.drop(columns=missing).isna().any().any()
    assert nd is None or not nd.isna().any().any()
    na_columns = case["brms"].get("na_columns", [])                  # a case that reads them as missing (NA in R)
    assert set(na_columns) <= set(missing)
    for col in na_columns:
        assert fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")[col].isna().sum() == data[col].isna().sum() > 0
    assert (data["size"] >= 1).all()


LOO_COMPARE = [cid for cid in FROZEN if "loo_compare" in fx.load_case(cid)["brms"]]


@pytest.mark.parametrize("case_id", LOO_COMPARE)
def test_loo_compare_is_with_a_frozen_case_on_the_same_data(case_id):
    """D05: a case whose fixture compares its model with another (``brms.loo_compare``, the other case's frozen brms section)
    names a frozen case on the same data, and the comparison's other model is that case's fixture: the same points (its
    log density there) and the same pointwise elpd_loo and elpd_waic as its own ``post`` block; this case's column is its own."""
    case = fx.load_case(case_id)
    other = fx.load_case(case["brms"]["loo_compare"]["id"])
    assert fx.has_fixture(other["id"]) and case["autobayes"].get("compare_with") == other["id"]
    assert case["brms"]["loo_compare"] == fx.frozen_section(other)
    assert case["data"] == other["data"]
    assert (fx.fixture_dir(case_id) / "data.csv").read_bytes() == (fx.fixture_dir(other["id"]) / "data.csv").read_bytes()
    ours, theirs = fx.Fixture.load(case_id), fx.Fixture.load(other["id"])
    cmp = ours.raw["post"]["loo_compare"]
    assert list(cmp["models"]) == list(cmp["elpd_loo"]["colnames"]) == [case_id, other["id"]]
    np.testing.assert_allclose(fx.decode(cmp["other_lp"]), [p.lp for p in theirs.points], rtol=1e-12, atol=0)
    for criterion in ("loo", "waic"):
        elpd = fx.decode(cmp[f"elpd_{criterion}"])
        for column, fixture in enumerate((ours, theirs)):
            block = fixture.raw["post"][criterion]["pointwise"]
            pointwise = fx.decode(block)[:, list(block["colnames"]).index(f"elpd_{criterion}")]
            np.testing.assert_allclose(elpd[:, column], pointwise, rtol=1e-12, atol=0)


@pytest.mark.parametrize("case_id", [cid for cid in FROZEN if {"hypothesis", "conditional_effects"} & set(fx.load_case(cid)["brms"])])
def test_hypothesis_and_conditional_effects_are_recorded_as_asked(case_id):
    """D06: one row and one column of draws per hypothesis, one table per effect (its grid, estimate__, se__, lower__, upper__)."""
    case = fx.load_case(case_id)
    brms, post = case["brms"], fx.Fixture.load(case_id).raw["post"]
    if "hypothesis" in brms:
        assert len(post["hypothesis"]["table"]["values"]["Hypothesis"]) == len(brms["hypothesis"])
        assert fx.decode(post["hypothesis"]["samples"]).shape == (fx.frozen_section(case)["n_points"], len(brms["hypothesis"]))
    if "conditional_effects" in brms:
        assert list(post["conditional_effects"]) == brms["conditional_effects"]
        for effect, table in post["conditional_effects"].items():   # an interaction "x:z" (Tier E): each variable and effect2__
            variables = effect.split(":")
            assert {*variables, "estimate__", "se__", "lower__", "upper__"} <= set(table["columns"])
            assert ("effect2__" in table["columns"]) == (len(variables) == 2)


PRIOR_DRAWS = [cid for cid in FROZEN if fx.load_case(cid)["brms"].get("prior_draws")]


@pytest.mark.parametrize("case_id", PRIOR_DRAWS)
def test_prior_draws_are_recorded_with_the_fit_that_drew_them(case_id):
    """Tier E: a case with ``prior_draws`` is fitted with sample_prior = "yes"; its fixture has brms's prior_draws(), one finite row
    per point under brms's names, its seeded points leave out the prior_* generated quantities (constrain_pars() draws them anew on
    every call), and each point hypothesis has a Savage-Dickey evidence ratio, a number where a fit without prior draws has NA."""
    case = fx.load_case(case_id)
    assert case["brms"]["brm_args"]["sample_prior"] == "yes"
    fixture = fx.Fixture.load(case_id)
    block = fixture.raw["post"]["prior_draws"]
    draws = fx.decode(block)
    assert draws.shape == (len(fixture.points), len(block["colnames"])) and np.isfinite(draws).all()
    assert not [name for p in fixture.points for name in p.pars if name.startswith("prior_")]
    if "hypothesis" in case["brms"]:
        table = fixture.raw["post"]["hypothesis"]["table"]["values"]
        point = [" = " in h for h in table["Hypothesis"]]
        assert any(point) and np.isfinite(fx.decode(table["Evid.Ratio"])[point]).all()


STACKING = [cid for cid in LOO_COMPARE if "loo_model_weights" in fx.load_case(cid)["brms"].get("extras", [])]


@pytest.mark.parametrize("case_id", STACKING)
def test_stacking_weights_are_recorded_for_the_two_models(case_id):
    """Tier E: a comparison case with ``extras`` "loo_model_weights" records brms's stacking weights of its two models, in the
    comparison's order, non-negative and summing to one."""
    post = fx.Fixture.load(case_id).raw["post"]
    block = post["loo_model_weights"]
    assert list(block["models"]) == list(post["loo_compare"]["models"]) and block["method"] == "stacking"
    w = fx.decode(block["weights"])
    assert w.shape == (2,) and (w >= 0).all() and abs(w.sum() - 1.0) < 1e-12


def _formula_text(case) -> str:
    """The brms formula of a case, its parts joined ("" for a multivariate case, whose formulas are under mvbf)."""
    formula = case["brms"].get("formula", "")
    return " ".join(formula) if isinstance(formula, list) else formula


@pytest.mark.parametrize("case_id", FROZEN)
def test_data2_matrices_are_the_frozen_files(case_id, tmp_path):
    """Tier F: a case's data2 matrices (its brms section's names, its data section's kinds) are the committed data2_<name>.csv files,
    byte for byte as tests/oracle/datasets.py writes them, with their sha256 in the fixture's meta; a case without data2 has
    neither. brms saw what Python reads: sar()'s Msar and fcor()'s Mfcor equal the file exactly, and car(W, gr = site)'s edges
    (edges1, edges2), numbers of neighbours (Nneigh) and row locations (Jloc) are W's, its labels the levels of the grouping
    factor in order. Tier G: gr(g, cov = A)'s A is labelled by the levels of g in order, and brms's Lcov_k is its lower Cholesky
    factor (tests/oracle/test_group_distributional_harness.py checks it on the levels and the effects built from it)."""
    case = fx.load_case(case_id)
    d = fx.fixture_dir(case_id)
    names = case["brms"].get("data2", [])
    assert names == list(case["data"].get("data2", {}))
    assert sorted(p.name for p in d.glob("data2_*.csv")) == sorted(fx.data2_path(d, n).name for n in names)
    fixture = fx.Fixture.load(case_id)
    if not names:
        assert "data2_sha256" not in fixture.meta
        return
    assert fixture.meta["data2_sha256"] == {n: fx.sha256(fx.data2_path(d, n)) for n in names}
    data, _ = datasets.make(case["data"])
    for name, matrix in datasets.data2(case["data"], data).items():
        datasets.write_matrix(matrix, tmp_path / f"data2_{name}.csv")
        assert (tmp_path / f"data2_{name}.csv").read_bytes() == fx.data2_path(d, name).read_bytes()
    terms = {m.group(2): m.group(1) for m in re.finditer(r"\b(car|sar|fcor)\((\w+)", _formula_text(case))}
    covs = {m.group(2): m.group(1) for m in re.finditer(r"\bgr\((\w+),[^()]*\bcov\s*=\s*(\w+)", _formula_text(case))}   # Tier G
    terms.update({name: "gr" for name in covs})
    assert sorted(terms) == sorted(names)
    sd, frame = fixture.standata, fx.read_frame(case, d / "data.csv")
    for name, matrix in fx.read_data2(case, d).items():
        M = matrix.to_numpy()
        if terms[name] == "gr":                                    # gr(g, cov = A): A on the levels of g, Lcov its Cholesky factor
            assert list(matrix.index) == list(matrix.columns) == case["brms"]["factors"][covs[name]]
            assert any(np.allclose(sd[k], np.linalg.cholesky(M), rtol=0, atol=1e-14) for k in sd if k.startswith("Lcov_"))
            continue
        if terms[name] != "car":
            assert np.array_equal(sd[{"sar": "Msar", "fcor": "Mfcor"}[terms[name]]], M)
            continue
        (gr,) = re.findall(rf"\bcar\({name},\s*gr\s*=\s*(\w+)", _formula_text(case))
        levels = case["brms"]["factors"][gr]
        assert list(matrix.index) == list(matrix.columns) == levels and int(sd["Nloc"]) == len(levels)
        edges = sorted(zip(sd["edges1"].astype(int).tolist(), sd["edges2"].astype(int).tolist()))
        assert edges == sorted(zip(*(np.nonzero(np.tril(M))[k] + 1 for k in range(2)))) and len(edges) == int(sd["Nedges"])
        if "Nneigh" in sd:
            assert np.array_equal(sd["Nneigh"], M.sum(1))
        assert np.array_equal(sd["Jloc"], [levels.index(s) + 1 for s in frame[gr]])


CBIND = [cid for cid in FROZEN if re.match(r"\s*cbind\(", _formula_text(fx.load_case(cid)))]


@pytest.mark.parametrize("case_id", CBIND)
def test_matrix_response_is_the_bound_columns(case_id):
    """Tier F: brms's Y of a ``cbind(m1, m2, m3)`` response is those columns of data.csv, exactly and in order (ncat of them), and
    its trials() are their row sums (brms refuses any other)."""
    case = fx.load_case(case_id)
    lhs = _formula_text(case).split("~")[0]
    columns = [c.strip() for c in re.match(r"\s*cbind\(([^)]*)\)", lhs).group(1).split(",")]
    frame = fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")
    sd = fx.Fixture.load(case_id).standata
    assert int(sd["ncat"]) == len(columns) and np.array_equal(sd["Y"], frame[columns].to_numpy(dtype=float))
    if "trials" in sd:
        assert np.array_equal(sd["trials"], frame[columns].sum(axis=1).to_numpy())


REFUSED = [cid for cid in FROZEN if fx.load_case(cid)["brms"].get("prediction_refused")]


@pytest.mark.parametrize("case_id", REFUSED)
def test_refused_predictions_are_recorded_as_brms_s_messages(case_id):
    """Tier F: a prediction kind that a case lists in ``prediction_refused`` (brms fails to compute it: posterior_epred of
    logistic_normal and of sar(type = "error")) is recorded in every block, with and without group effects, as brms's error
    message in place of the array; every other kind is an array."""
    refused = fx.load_case(case_id)["brms"]["prediction_refused"]
    fixture = fx.Fixture.load(case_id)
    for block, entries in fixture.raw["predictions"].items():
        for kind in ("linpred", "epred"):
            for name in (kind, f"{kind}_nogroup") if fixture.groups else (kind,):
                if kind in refused:
                    assert name not in entries and fixture.prediction_refusal(block, name)
                else:
                    assert fixture.prediction(block, name) is not None and fixture.prediction_refusal(block, name) is None


# -- Tier F part 2: F06 addition terms (se(), rate(), thres(gr = ), subset()) and F07 summaries ------------------------------
def _formulas(case) -> list[str]:
    """Every brms formula of a case: its formula (or its parts), or each of a multivariate case's ``mvbf`` formulas."""
    brms = case["brms"]
    parts = brms.get("mvbf") or brms.get("formula") or []
    return [parts] if isinstance(parts, str) else list(parts)


ADDITION = [cid for cid in FROZEN if any(re.search(r"\|.*\b(se|rate|thres|subset)\(", f) for f in _formulas(fx.load_case(cid)))]


def _group_effects(sd, p, resp=""):
    """The varying intercept of ``(1 | g)`` per row (r_1_1 or r_1_<resp>_1 at J_1, times Z_1_1), 0 without one."""
    suffix = f"_{resp}" if resp else ""
    if f"J_1{suffix}" not in sd:
        return 0.0
    return np.atleast_1d(p.pars[f"r_1{suffix}_1"])[sd[f"J_1{suffix}"].astype(int) - 1] * sd[f"Z_1{suffix}_1"]


def _centred_eta(sd, p, resp=""):
    """Intercept + (X without its intercept column, centred) b + group effects, brms's linear predictor of a centred design."""
    suffix = f"_{resp}" if resp else ""
    X = sd[f"X{suffix}"][:, 1:]
    eta = float(p.pars[f"Intercept{suffix}"]) + _group_effects(sd, p, resp)
    return eta + ((X - X.mean(0)) @ np.atleast_1d(p.pars[f"b{suffix}"]) if X.shape[1] else 0.0)


def _addition_log_likelihood(case, sd, frame, p) -> float:
    """F06: the log likelihood of a frozen point, written from the mathematics, with brms's data checked on the way: se(col) is a
    known standard deviation per row, the scale itself without sigma (brms's data sigma = 0) or sqrt(sigma^2 + se^2) with
    sigma = TRUE; rate(col) multiplies the mean exp(eta) by the row's denominator, and a negative binomial's shape too;
    thres(gr = z) gives each level of z its own ordered thresholds, as many as the level's largest response category less one
    (or the K of thres(K, gr = z)), on an uncentred design (brms does not centre with grouped thresholds); subset(col) puts the
    rows whose indicator is 1 in that response's likelihood, in data order."""
    stats = pytest.importorskip("scipy.stats")
    from scipy.special import expit
    family = case["brms"]["family"]["family"]
    (formula, *_) = _formulas(case)
    if "mvbf" in case["brms"]:                                  # subset(): one Gaussian likelihood per response, on its rows
        total = 0.0
        for f in _formulas(case):
            resp, indicator = re.match(r"\s*(\w+)\s*\|\s*subset\((\w+)\)", f).groups()
            rows = frame[indicator].to_numpy() == 1
            assert int(sd[f"N_{resp}"]) == rows.sum() and np.array_equal(sd[f"Y_{resp}"], frame[resp].to_numpy()[rows])
            total += stats.norm.logpdf(sd[f"Y_{resp}"], _centred_eta(sd, p, resp), float(p.pars[f"sigma_{resp}"])).sum()
        return total
    y = sd["Y"]
    se = re.search(r"\bse\((\w+)(,\s*sigma\s*=\s*TRUE)?\)", formula)
    if se:
        assert np.array_equal(sd["se"], frame[se.group(1)].to_numpy())
        if se.group(2):
            assert "sigma" not in sd
            scale = np.sqrt(float(p.pars["sigma"]) ** 2 + sd["se"] ** 2)
        else:
            assert float(sd["sigma"]) == 0.0
            scale = sd["se"]
        if family == "student":
            return stats.t.logpdf(y, float(p.pars["nu"]), _centred_eta(sd, p), scale).sum()
        return stats.norm.logpdf(y, _centred_eta(sd, p), scale).sum()
    rate = re.search(r"\brate\((\w+)\)", formula)
    if rate:
        assert np.array_equal(sd["denom"], frame[rate.group(1)].to_numpy())
        mean = np.exp(_centred_eta(sd, p)) * sd["denom"]
        if family == "poisson":
            return stats.poisson.logpmf(y, mean).sum()
        shape = float(p.pars["shape"]) * sd["denom"]
        return stats.nbinom.logpmf(y, shape, shape / (shape + mean)).sum()
    thres = re.search(r"\bthres\((?:(\d+),\s*)?gr\s*=\s*(\w+)\)", formula)
    assert thres and family == "cumulative"
    levels = case["brms"]["factors"][thres.group(2)]
    codes = frame[thres.group(2)].map(levels.index).to_numpy()
    response = frame[formula.split("|")[0].strip()].to_numpy()
    K = [int(thres.group(1))] * len(levels) if thres.group(1) else [int(response[codes == j].max()) - 1 for j in range(len(levels))]
    assert np.atleast_1d(sd["nthres"]).astype(int).tolist() == K and np.array_equal(y, response)
    starts = np.cumsum([0, *K])[:-1] + 1
    assert np.array_equal(sd["Jthres"], np.column_stack([starts[codes], starts[codes] + np.array(K)[codes] - 1]))
    mu = sd["X"] @ np.atleast_1d(p.pars["b"]) + _group_effects(sd, p)
    total = 0.0
    for i, (j, k) in enumerate(zip(codes, y.astype(int))):
        th = np.atleast_1d(p.pars[f"Intercept_{j + 1}"])
        assert np.array_equal(th, np.atleast_1d(p.pars[f"b_Intercept_{j + 1}"]))   # no centring: the thresholds are brms's b_Intercept
        cdf = np.concatenate([[0.0], expit(th - mu[i]), [1.0]])
        total += np.log(cdf[k] - cdf[k - 1])
    return total


@pytest.mark.parametrize("case_id", ADDITION)
def test_addition_term_fixture_matches_a_hand_written_likelihood(case_id):
    """F06: at every frozen point brms's log density without the Jacobian is its prior (lprior), the standard normal of the
    group-level z, and the likelihood ``_addition_log_likelihood`` writes from the mathematics."""
    case = fx.load_case(case_id)
    f = fx.Fixture.load(case_id)
    frame = fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")
    sd = f.standata
    for p in f.points:
        z = sum(float(-0.5 * np.sum(np.ravel(v) ** 2) - 0.5 * np.size(v) * np.log(2 * np.pi)) for k, v in p.pars.items() if k.startswith("z_"))
        expected = p.lp_nojac - float(p.pars["lprior"]) - z
        assert rel_err(_addition_log_likelihood(case, sd, frame, p), expected) <= 1e-12


PER_RESPONSE = [cid for cid in FROZEN if fx.load_case(cid)["brms"].get("prediction_resp")]


@pytest.mark.parametrize("case_id", PER_RESPONSE)
def test_per_response_predictions_are_on_each_response_s_subset(case_id):
    """F06: a multivariate case with ``prediction_resp`` (subset(); brms refuses its joint predictions, which the case lists in
    ``prediction_refused``) records each response's posterior_linpred and posterior_epred, with and without group effects, on
    the rows of its subset: the fitted block on the data rows whose indicator is 1 (brms's linear predictor of that response at
    every point, in data order), the newdata block on the new rows whose indicator is 1; a Gaussian's epred is its linpred."""
    case = fx.load_case(case_id)
    f = fx.Fixture.load(case_id)
    d = fx.fixture_dir(case_id)
    frames = {"fitted": fx.read_frame(case, d / "data.csv"), "newdata": fx.read_frame(case, d / "newdata.csv")}
    assert set(case["brms"]["prediction_refused"]) == {"linpred", "epred"}
    indicators = dict(re.match(r"\s*(\w+)\s*\|\s*subset\((\w+)\)", form).groups() for form in _formulas(case))
    assert sorted(indicators) == sorted(case["brms"]["prediction_resp"])
    for block, frame in frames.items():
        for resp, indicator in indicators.items():
            rows = int((frame[indicator] == 1).sum())
            for kind in ("linpred", "epred", "linpred_nogroup", "epred_nogroup"):
                arr = f.response_prediction(block, resp, kind)
                assert arr.shape == (len(f.points), rows), (block, resp, kind)
            np.testing.assert_array_equal(f.response_prediction(block, resp, "epred"), f.response_prediction(block, resp, "linpred"))
            if block == "fitted":
                eta = np.stack([_centred_eta(f.standata, p, resp) * np.ones(rows) for p in f.points])
                np.testing.assert_allclose(f.response_prediction(block, resp, "linpred"), eta, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case_id", PER_RESPONSE)
def test_per_response_predictions_are_not_left_unchecked(case_id, monkeypatch):
    """F06: until gates.py compares each response's predictions (``check_response_predictions``, with subset()), a bundle of a
    case that records them fails with the oracle's mismatch, whatever model it is built with."""
    if hasattr(gates, "check_response_predictions"):
        pytest.skip("the per-response comparison has landed")
    monkeypatch.setattr(adapter, "in_data_order", lambda case, fixture, model, data: fixture)
    with pytest.raises(OracleMismatch, match="check_response_predictions"):
        gates.Bundle(case_id, model=object())


SUMMARIES = [cid for cid in FROZEN if fx.load_case(cid)["brms"].get("summaries")]


@pytest.mark.parametrize("case_id", SUMMARIES)
def test_summary_draws_are_the_seeded_chains(case_id):
    """F07: a case with ``summary_draws`` records its second fixed-parameter fit as chains x iterations seeded points, chain by
    chain, and brms's draws array (iterations, chains, variables) holds exactly those draws: lp__ is each point's log density
    with the Jacobian (rstan's log_prob, as a sampler records it), lprior and b_Intercept the point's own; none of the
    parameters brm() leaves out (z_, L_) is a variable; the draws move within and between chains; the meta records the posterior
    package's version."""
    case = fx.load_case(case_id)
    f = fx.Fixture.load(case_id)
    spec, block = case["brms"]["summary_draws"], f.summary_draws
    assert (block["chains"], block["iter"]) == (spec["chains"], spec["iter"]) and len(block["points"]) == spec["chains"] * spec["iter"]
    assert f.meta["posterior"] and block["draws"].shape == (spec["iter"], spec["chains"], len(block["variables"]))
    assert not [v for v in block["variables"] if v.startswith(("z_", "L_"))]
    column = {v: block["draws"][:, :, k] for k, v in enumerate(block["variables"])}
    for c in range(spec["chains"]):
        for t in range(spec["iter"]):
            p = block["points"][c * spec["iter"] + t]
            assert column["lp__"][t, c] == p.lp and column["lprior"][t, c] == float(p.pars["lprior"])
            assert column["b_Intercept"][t, c] == float(np.ravel(p.pars["b_Intercept"])[0])
            assert np.isfinite(p.lp_nojac) and p.grad is None
    u = np.stack([p.u for p in block["points"]]).reshape(spec["chains"], spec["iter"], -1)
    assert (np.diff(u, axis=1) != 0).all() and (u.mean(axis=1).std(axis=0) > 0).all()


def _summary(draws: np.ndarray) -> np.ndarray:
    """brms's posterior_summary of draws on the first axis: mean, sd (n - 1), and the 2.5 % and 97.5 % quantiles (R's type 7)."""
    return np.stack([draws.mean(0), draws.std(0, ddof=1), *np.quantile(draws, [0.025, 0.975], axis=0)], axis=-1)


def _close(ours, theirs):
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("case_id", SUMMARIES)
def test_summaries_are_moments_and_quantiles_of_the_draws(case_id):
    """F07: on the summary draws (all chains together), posterior_summary() is ``_summary`` of every variable, fixef() of the
    b_ variables, ranef() of r_<g>[level, coef], coef() of b_<coef> + r_<g>[level, coef] (b_<coef> alone for a coefficient
    without group effects), VarCorr() of sd_<g>__<coef> (residual__: sigma), cor_<g>__<a>__<b> (1 on the diagonal) and
    sd_a sd_b cor_ab (sd^2 on the diagonal); summary()'s first four columns are the same numbers for its rows (fixed b_<name>,
    spec_pars <name>, random sd(<coef>) and cor(<a>,<b>)), its Rhat and ESS columns finite and positive; and residuals()
    (ordinary, method posterior_epred) is ``_summary`` of Y - posterior_epred per draw, the inverse link of
    b_Intercept + sum of b_<col> col + r_<g>[level, Intercept] (+ r_<g>[level, x] x), row by row. Tier G part 2: a distributional
    parameter's group-level effect (sigma ~ ... + (1 | g)) is r_<g>__<dpar>[level, coef], labelled <dpar>_<coef> in ranef(), coef()
    and VarCorr() (and left out of the mean); coef() also has every coefficient with group effects and no population-level term
    (``(0 + x | g)`` without ``x``: its group effect alone); and VarCorr() has cor and cov only for a group with correlations (two
    terms of one factor, or of mu and sigma, without a shared ID have neither)."""
    from scipy.special import expit
    case = fx.load_case(case_id)
    f = fx.Fixture.load(case_id)
    post, block = f.raw["post"], f.summary_draws
    D = block["draws"].reshape(-1, len(block["variables"]), order="F")       # (draws, variables), chain after chain
    var = {v: D[:, k] for k, v in enumerate(block["variables"])}

    def r_name(g, lv, c):                                       # Tier G: a distributional parameter's effects, r_<g>__<dpar>[...]
        dpar, _, rest = c.partition("_")
        return f"r_{g}[{lv},{c}]" if f"r_{g}[{lv},{c}]" in var or not rest else f"r_{g}__{dpar}[{lv},{rest}]"

    table, (rows, cols) = fx.decode_labelled(post["posterior_summary"])
    assert rows == block["variables"] and cols == ["Estimate", "Est.Error", "Q2.5", "Q97.5"]
    _close(table, _summary(D))
    fixef, (names, _) = fx.decode_labelled(post["fixef"])
    _close(fixef, _summary(np.stack([var[f"b_{n}"] for n in names], 1)))
    for g, entry in post["ranef"].items():
        arr, (levels, _, coefs) = fx.decode_labelled(entry)
        for j, c in enumerate(coefs):
            _close(arr[:, :, j], _summary(np.stack([var[r_name(g, lv, c)] for lv in levels], 1)))
    for g, entry in post["coef"].items():
        arr, (levels, _, coefs) = fx.decode_labelled(entry)
        grouped = fx.decode_labelled(post["ranef"][g])[1][2]
        assert set(coefs) == set(names) | set(grouped) and [c for c in coefs if c in names] == names   # fixef's order kept
        for j, c in enumerate(coefs):
            r = [var.get(r_name(g, lv, c), 0.0) for lv in levels]
            _close(arr[:, :, j], _summary(np.stack([var.get(f"b_{c}", 0.0) + rv for rv in r], 1)))
    for g, entry in post["VarCorr"].items():
        sd_arr, (coefs, _) = fx.decode_labelled(entry["sd"])
        sds = [var["sigma"]] if g == "residual__" else [var[f"sd_{g}__{c}"] for c in coefs]
        _close(sd_arr, _summary(np.stack(sds, 1)))
        correlated = any(v.startswith(f"cor_{g}__") for v in var)
        assert set(entry) == ({"sd", "cor", "cov"} if correlated else {"sd"})
        if correlated:
            cor = lambda a, b: np.ones(len(D)) if a == b else var.get(f"cor_{g}__{coefs[a]}__{coefs[b]}", var.get(f"cor_{g}__{coefs[b]}__{coefs[a]}"))
            for key, value in (("cor", cor), ("cov", lambda a, b: sds[a] * sds[b] * cor(a, b))):
                arr, _ = fx.decode_labelled(entry[key])
                for a in range(len(coefs)):
                    _close(arr[:, :, a], _summary(np.stack([value(b, a) for b in range(len(coefs))], 1)))
    assert ("residual__" in post["VarCorr"]) == ("sigma" in var)
    summary = post["summary"]
    assert (summary["chains"], summary["iter"], summary["warmup"], summary["total_ndraws"]) == (block["chains"], block["iter"], 0, len(D))
    lookups = [("fixed", lambda n: f"b_{n}"), ("spec_pars", lambda n: n)]
    lookups += [(g, lambda n, g=g: f"sd_{g}__{n[3:-1]}" if n.startswith("sd(") else f"cor_{g}__" + "__".join(n[4:-1].split(","))) for g in summary["random"]]
    for name, variable in lookups:
        arr, (rows, cols) = fx.decode_labelled(summary[name] if name in ("fixed", "spec_pars") else summary["random"][name])
        assert cols == ["Estimate", "Est.Error", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS", "Tail_ESS"]
        if rows:
            _close(arr[:, :4], _summary(np.stack([var[variable(n)] for n in rows], 1)))
            assert np.isfinite(arr[:, 4:]).all() and (arr[:, 4:] > 0).all()
    frame = fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")
    formula = _formulas(case)[0]
    response = formula.split("~")[0].strip()
    eta = var["b_Intercept"][:, None] + sum(var[f"b_{n}"][:, None] * frame[n].to_numpy()[None, :] for n in names
                                             if n != "Intercept" and not n.startswith("sigma_"))
    for g in post["ranef"]:
        _, (levels, _, coefs) = fx.decode_labelled(post["ranef"][g])
        for c in coefs:
            if "__" in r_name(g, levels[0], c):                   # a distributional parameter's, not the mean's
                continue
            r = np.stack([var[r_name(g, lv, c)] for lv in frame[g]], 1)
            eta = eta + r * (1.0 if c == "Intercept" else frame[c].to_numpy()[None, :])
    inverse_link = {"gaussian": lambda e: e, "bernoulli": expit}[case["brms"]["family"]["family"]]
    resid, (rows, cols) = fx.decode_labelled(post["residuals"])
    assert rows == [] and cols == ["Estimate", "Est.Error", "Q2.5", "Q97.5"]
    _close(resid, _summary(frame[response].to_numpy()[None, :] - inverse_link(eta)))


# -- Tier F part 3: F05 cox(), the M-spline baseline hazard ----------------------------------------------------------------------
COX = [cid for cid in FROZEN if fx.load_case(cid)["brms"]["family"].get("family") == "cox"]


def _bhaz_arguments(case) -> dict:
    """The arguments of a cox case's bhaz() addition term ({} without one): df an integer, knots and Boundary.knots lists."""
    m = re.search(r"\bbhaz\(((?:[^()]|\([^()]*\))*)\)", _formula_text(case))
    out = {}
    for name, value in re.findall(r"([\w.]+)\s*=\s*(c\([^)]*\)|[^,]+)", m.group(1) if m else ""):
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d*)?(?:e-?\d+)?", value)]
        out[name] = int(numbers[0]) if name == "df" else numbers
    return out


def _cox_knots(case, y) -> tuple[np.ndarray, np.ndarray]:
    """The interior and boundary knots of brms 2.23's baseline hazard (probed: splines2's knots and Boundary.knots attributes of
    standata()'s Zbhaz): the boundary knots are the range of every row's time, censored rows too, widened at each end by 2% of
    its length; the interior knots are the times' quantiles (R's type 7, numpy's linear) at the df - 3 equally spaced
    probabilities inside (0, 1), df - degree - 1 knots of a cubic basis with its intercept (bhaz()'s df, 5 by default: the median
    alone); bhaz(knots = ) and bhaz(Boundary.knots = ) give either."""
    args = _bhaz_arguments(case)
    lo, hi = float(y.min()), float(y.max())
    boundary = np.array(args.get("Boundary.knots", [lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)]))
    if "knots" in args:
        return np.array(args["knots"]), boundary
    inner = args.get("df", 5) - 4
    return np.quantile(y, np.linspace(0.0, 1.0, inner + 2)[1:-1]), boundary


def _m_and_i_splines(x, knots, boundary, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The M-spline basis of ``degree`` on the knots, every basis function kept (an intercept), and its I-spline basis, the
    integrals from the lower boundary knot (Ramsay 1988): M_i = k B_i / (t_{i+k} - t_i) for the B-splines B_i of order k = degree
    + 1 on the knot sequence t, each boundary knot repeated k times, so that each M_i integrates to 1 and each I_i rises from 0
    to 1."""
    from scipy.interpolate import BSpline
    k = degree + 1
    t = np.concatenate([[boundary[0]] * k, knots, [boundary[1]] * k])
    n = len(t) - k
    M, I = np.empty((len(x), n)), np.empty((len(x), n))
    for i in range(n):
        spline = BSpline(t, np.eye(n)[i] * k / (t[i + k] - t[i]), degree)
        integral = spline.antiderivative()
        M[:, i], I[:, i] = spline(x), integral(x) - integral(boundary[0])
    return M, I


def _prior_of(fixture, cls: str, *, coef: str = "", group: str = "") -> str:
    """The prior brms applies to one parameter: the most specific non-empty row of its resolved prior table (prior_summary) for
    the class, the coefficient's or the group's before the class's ("" for a flat prior)."""
    rows = [r for r in fixture.prior_summary if r["class"] == cls]
    for c, g in ((coef, group), ("", group), ("", "")):
        for r in rows:
            if r["coef"] == c and r["group"] == g and r["prior"]:
                return r["prior"]
    return ""


def _log_prior_density(prior: str, x, *, half: bool = False) -> float:
    """Stan's log density of a brms prior string (normal, student_t; "" flat), of a half density (lower bound 0) when ``half``."""
    stats = pytest.importorskip("scipy.stats")
    if not prior:
        return 0.0
    name, args = re.fullmatch(r"(\w+)\((.*)\)", prior).groups()
    a = [float(v) for v in args.split(",")]
    dist = {"normal": lambda: stats.norm(a[0], a[1]), "student_t": lambda: stats.t(a[0], a[1], a[2])}[name]()
    assert not half or dist.median() == 0.0, "a half density of a prior centred at 0 only"
    return float(np.sum(dist.logpdf(np.ravel(x)))) + (np.size(x) * np.log(2.0) if half else 0.0)


def _cox_rows(sd, p) -> np.ndarray:
    """F05: each row's log likelihood at a frozen point, from the mathematics. With eta brms's linear predictor, the baseline hazard
    h0 = Zbhaz sbhaz and the cumulative baseline hazard H0 = Zcbhaz sbhaz (sbhaz a simplex, so H0 rises from 0 at the lower
    boundary knot to 1 at the upper), the hazard is h0 exp(eta) and the survival function exp(-H0 exp(eta)): an event (cens 0, or
    every row without cens()) contributes log h0 + eta - H0 exp(eta), a right-censored row (cens 1) -H0 exp(eta)."""
    s = np.atleast_1d(p.pars["sbhaz"])
    eta = _centred_eta(sd, p)
    h0, H0 = sd["Zbhaz"] @ s, sd["Zcbhaz"] @ s
    cens = sd["cens"] if "cens" in sd else np.zeros(len(eta))
    assert set(np.unique(cens).tolist()) <= {0.0, 1.0}
    return np.where(cens == 0, np.log(h0) + eta - H0 * np.exp(eta), -H0 * np.exp(eta))


@pytest.mark.parametrize("case_id", COX)
def test_cox_baseline_hazard_bases_are_m_and_i_splines(case_id):
    """F05: brms's Y is the time column of the formula, Zbhaz the cubic M-spline basis of Y with its intercept (splines2's mSpline)
    on the knots `_cox_knots` sets, Zcbhaz its I-spline basis (in [0, 1]), Kbhaz the number of basis functions (df, or the number
    of knots plus 4) and con_sbhaz the concentration of the Dirichlet prior on the simplex sbhaz (the case's dirichlet(), 1 by
    default), one per basis function."""
    case = fx.load_case(case_id)
    f = fx.Fixture.load(case_id)
    sd = f.standata
    time = re.match(r"\s*(\w+)", _formula_text(case)).group(1)
    y = fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")[time].to_numpy(dtype=float)
    assert np.array_equal(sd["Y"], y)
    knots, boundary = _cox_knots(case, y)
    M, I = _m_and_i_splines(y, knots, boundary)
    assert int(sd["Kbhaz"]) == M.shape[1] == len(knots) + 4 == _bhaz_arguments(case).get("df", len(knots) + 4)
    assert f.colnames("Zbhaz") == f.colnames("Zcbhaz") == [str(k + 1) for k in range(M.shape[1])]
    for name, basis in (("Zbhaz", M), ("Zcbhaz", I)):
        assert sd[name].shape == basis.shape and np.max(np.abs(sd[name] - basis)) <= 1e-13 * max(1.0, np.max(np.abs(basis)))
    assert (sd["Zcbhaz"] >= 0).all() and (sd["Zcbhaz"] <= 1).all()
    concentration = re.fullmatch(r"dirichlet\((.*)\)", _prior_of(f, "sbhaz")).group(1)
    assert np.array_equal(sd["con_sbhaz"], np.full(M.shape[1], float(concentration)))


@pytest.mark.parametrize("case_id", COX)
def test_cox_fixture_matches_a_hand_written_likelihood(case_id):
    """F05: at every frozen point brms's prior (lprior) is the normal or Student-t densities of its prior table on Intercept, b
    and the group-level sd (a half density) plus the Dirichlet density of sbhaz with concentration con_sbhaz; its log density
    without the Jacobian is lprior, the standard normal of the group-level z and the likelihood `_cox_rows` writes from the
    mathematics; brms's fitted linear predictor is eta, the logarithm of the factor exp(eta) on the baseline hazard (its
    posterior_epred fails); a post case's log_lik is `_cox_rows` per point, (draws, rows)."""
    stats = pytest.importorskip("scipy.stats")
    case = fx.load_case(case_id)
    f = fx.Fixture.load(case_id)
    sd = f.standata
    linpred = f.prediction("fitted", "linpred")
    log_lik = f.raw.get("post", {}).get("log_lik")
    assert (log_lik is not None) == ("log_lik" in case["brms"].get("extras", []))
    assert f.prediction_refusal("fitted", "epred") == ("Cannot compute expected values of the posterior predictive distribution "
                                                       "for family 'cox'.")
    (group,) = f.groups
    for k, p in enumerate(f.points):
        s = np.atleast_1d(p.pars["sbhaz"])
        assert abs(s.sum() - 1.0) <= 1e-15 and (s > 0).all()
        lprior = (_log_prior_density(_prior_of(f, "Intercept"), p.pars["Intercept"])
                  + _log_prior_density(_prior_of(f, "b", coef="x"), p.pars["b"])
                  + _log_prior_density(_prior_of(f, "sd", coef=group["coef"], group=group["group"]), p.pars["sd_1"], half=True)
                  + float(stats.dirichlet.logpdf(s, sd["con_sbhaz"])))
        assert rel_err(lprior, float(p.pars["lprior"])) <= 1e-12
        z = sum(float(-0.5 * np.sum(np.ravel(v) ** 2) - 0.5 * np.size(v) * np.log(2 * np.pi)) for n, v in p.pars.items()
                if n.startswith("z_"))
        rows = _cox_rows(sd, p)
        assert rel_err(rows.sum(), p.lp_nojac - float(p.pars["lprior"]) - z) <= 1e-12
        eta = _centred_eta(sd, p)
        assert np.max(np.abs(linpred[k] - eta)) <= 1e-13 * max(1.0, np.max(np.abs(eta)))
        if log_lik is not None:
            assert np.max(np.abs(fx.decode(log_lik)[k] - rows)) <= 1e-12 * max(1.0, np.max(np.abs(rows)))


def _sorted_levels(fixture, group):
    (grp,) = [g for g in fixture.groups if g["group"] == group]
    assert list(grp["levels"]) == sorted(grp["levels"]), "the direct indexing below needs sorted brms levels"
    return grp


@pytest.mark.skipif(not fx.has_fixture("gauss_varint"), reason="fixture not frozen")
def test_gaussian_fixture_matches_the_hand_written_reference():
    """brms's log_prob at every point equals tests/reference_logp.py, written from the mathematics."""
    f = fx.Fixture.load("gauss_varint")
    case = fx.load_case("gauss_varint")
    data = fx.read_frame(case, fx.fixture_dir("gauss_varint") / "data.csv")
    _sorted_levels(f, "g")
    assert f.colnames("X") == ["Intercept", "x"]
    priors = {"intercept": ("Normal", {"mu": 0.0, "sigma": 5.0}), "b": {"x": ("Normal", {"mu": 0.0, "sigma": 2.0})},
              "sd": ("HalfNormal", {"sigma": 1.0}), "sigma": ("HalfNormal", {"sigma": 1.0})}
    for p in f.points:
        point = {"a_c": float(p.pars["Intercept"]), "b": p.pars["b"], "log_sigma": np.log(float(p.pars["sigma"])),
                 "log_sd": np.log(float(p.pars["sd_1"][0])), "z": np.atleast_2d(p.pars["z_1"])[0]}
        for jac, lp in ((True, p.lp), (False, p.lp_nojac)):
            ref = reference_logp(point, data, response="y", predictors=["x"], group="g", priors=priors, center=True, jacobian=jac)
            assert rel_err(ref, lp) <= LOGP_REL, (jac, ref, lp)
        assert abs(p.lp - p.lp_nojac - (point["log_sigma"] + point["log_sd"])) < 1e-10


@pytest.mark.skipif(not fx.has_fixture("binomial_varint"), reason="fixture not frozen")
def test_binomial_fixture_matches_the_hand_written_reference():
    """brms's log_prob and gradient equal tests/reference_binomial.py (60-digit decimal row terms)."""
    f = fx.Fixture.load("binomial_varint")
    case = fx.load_case("binomial_varint")
    data = fx.read_frame(case, fx.fixture_dir("binomial_varint") / "data.csv")
    _sorted_levels(f, "g")
    assert f.colnames("X") == ["Intercept", "x"]
    priors = {"intercept": ("Normal", {"mu": 0.0, "sigma": 2.5}), "b": {"x": ("Normal", {"mu": 0.0, "sigma": 1.5})},
              "sd": ("HalfNormal", {"sigma": 1.0})}
    by_coordinate = {int(e["index"]) - 1: (e["par"], int(e["elem"]) - 1) for e in f.upars}
    for p in f.points:
        point = {"a_c": float(p.pars["Intercept"]), "b": p.pars["b"], "log_sd": np.log(float(p.pars["sd_1"][0])),
                 "z": np.atleast_2d(p.pars["z_1"])[0]}
        for jac, lp, grad in ((True, p.lp, p.grad), (False, p.lp_nojac, p.grad_nojac)):
            ref = reference_logp_binomial(point, data, response="k", trials="size", predictors=["x"], group="g",
                                          priors=priors, center=True, jacobian=jac)
            assert rel_err(ref, lp) <= LOGP_REL, (jac, ref, lp)
            g = reference_grad_binomial(point, data, response="k", trials="size", predictors=["x"], group="g",
                                        prior_intercept_sigma=2.5, prior_b_sigma=1.5, jacobian=jac, center=True)
            for i, (par, elem) in by_coordinate.items():
                expected = {"Intercept": lambda: g["Intercept"], "b": lambda: g["x"], "sd_1": lambda: g["1|g_sigma_log__"],
                            "z_1": lambda: g["1|g_offset"][elem]}[par]()
                assert rel_err(expected, grad[i]) <= GRAD_REL, (jac, par, elem, expected, grad[i])


@pytest.fixture(scope="module")
def gauss_bundle():
    if not fx.has_fixture("gauss_varint"):
        pytest.skip("fixture not frozen")
    return gates.Bundle("gauss_varint")


def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_gates_pass_on_the_frozen_gaussian_fixture(gauss_bundle):
    for gate in gates.ALL:
        gate(gauss_bundle)


def test_corrupted_log_density_fails(gauss_bundle):
    def mutate(raw):
        raw["points"][3]["lp"] += 1e-5                                # |lp| is about 120: 1e-5 is 8e-8 relative
    with pytest.raises(OracleMismatch, match="point 3: log density"):
        gates.check_logp(_corrupted(gauss_bundle, mutate))


def test_tolerance_floor_is_documented(gauss_bundle):
    """1e-8 relative at |lp| about 120 means absolute differences below about 1e-6 pass; the harness is not
    claimed to be finer than that."""
    lp = gauss_bundle.fixture.points[3].lp
    assert 50 < abs(lp) < 500
    assert rel_err(lp, lp + 1e-6) < LOGP_REL < rel_err(lp, lp + 1e-5)


def test_corrupted_gradient_fails(gauss_bundle):
    def mutate(raw):
        raw["points"][0]["grad"]["values"][2] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match="gradient"):
        gates.check_gradient(_corrupted(gauss_bundle, mutate))


def test_corrupted_design_fails(gauss_bundle):
    def mutate(raw):
        raw["standata"]["X"]["values"][-1] += 1e-11                  # the last entry of the last column
    with pytest.raises(OracleMismatch, match="matches 0 brms columns"):
        _corrupted(gauss_bundle, mutate)


def test_corrupted_group_coding_fails(gauss_bundle):
    def mutate(raw):
        J = raw["standata"]["J_1"]["values"]
        J[0], J[1] = J[1], J[0]
    with pytest.raises(OracleMismatch, match="partition"):
        _corrupted(gauss_bundle, mutate)


def test_corrupted_prediction_fails(gauss_bundle):
    def mutate(raw):
        raw["predictions"]["fitted"]["linpred"]["values"][5] += 1e-8
    with pytest.raises(OracleMismatch, match="linear predictor differs"):
        gates.check_predictions(_corrupted(gauss_bundle, mutate))


def test_swapped_levels_fail(gauss_bundle):
    def mutate(raw):
        z = raw["points"][1]["pars"]["z_1"]["values"]
        z[0], z[1] = z[1], z[0]
    with pytest.raises(OracleMismatch):
        gates.check_logp(_corrupted(gauss_bundle, mutate))


def test_corrupted_default_prior_table_fails(gauss_bundle):
    """G6 compares every row of brms's default_prior() with ab.default_priors() written as brms's table (A07): a changed
    location, a missing row or a changed bound fails; the same table spelled with other whitespace and number
    formatting passes."""
    def location(raw):
        raw["default_prior"][0]["prior"] = "student_t(3, 0.8, 2.5)"            # brms: student_t(3, 0.9, 2.5)
    def row(raw):
        del raw["default_prior"][2]                                            # the b row of coefficient x
    def bound(raw):
        (sigma,) = [r for r in raw["default_prior"] if r["class"] == "sigma"]
        sigma["lb"] = ""
    for mutate in (location, row, bound):
        with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
            gates.check_default_prior(_corrupted(gauss_bundle, mutate))
    def respelled(raw):
        raw["default_prior"][0]["prior"] = "student_t( 3,0.90, 2.5e0 )"
        (sigma,) = [r for r in raw["default_prior"] if r["class"] == "sigma"]
        sigma["lb"] = "0.0"
    gates.check_default_prior(_corrupted(gauss_bundle, respelled))


def test_default_prior_numbers_compare_to_the_last_digit_brms_prints():
    """brms writes the offset-shifted location of poisson_offset with 15 significant digits; one unit in the last of
    them is a different prior."""
    if not fx.has_fixture("poisson_offset"):
        pytest.skip("fixture not frozen")
    bundle = gates.Bundle("poisson_offset")
    assert bundle.fixture.default_prior[0]["prior"] == "student_t(3, -0.978004218445523, 2.5)"
    def last_digit(raw):
        raw["default_prior"][0]["prior"] = "student_t(3, -0.978004218445524, 2.5)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(bundle, last_digit))


def test_fixture_json_has_no_timestamps(gauss_bundle):
    text = json.dumps(gauss_bundle.fixture.raw)
    assert "generated" not in text and "date" not in gauss_bundle.fixture.meta


# -- the correlated case (A02): L_1 is matched on Stan's cholesky_corr coordinate, not passed through -----------
@pytest.fixture(scope="module")
def correlated_bundle():
    if not fx.has_fixture("gauss_slope_correlated") or fx.load_case("gauss_slope_correlated").get("gate"):
        pytest.skip("correlated case not frozen or still gated")
    return gates.Bundle("gauss_slope_correlated")


def _L_coordinate(raw) -> int:
    (idx,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "L_1"]
    return idx


def test_corrupted_correlation_coordinate_fails(correlated_bundle):
    """brms's unconstrained L_1 value must be Stan's cholesky_corr coordinate of L_1 at every point, so a fixture
    whose u disagrees (as if brms sampled L_1 through another transform) stops the gradient gate."""
    def mutate(raw):
        raw["points"][2]["u"]["values"][_L_coordinate(raw)] += 1e-6
    with pytest.raises(OracleMismatch, match="point 2: Stan's cholesky_corr coordinate 0 of L_1"):
        gates.check_gradient(_corrupted(correlated_bundle, mutate))


def test_corrupted_correlation_gradient_fails(correlated_bundle):
    def mutate(raw):
        raw["points"][0]["grad"]["values"][_L_coordinate(raw)] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 0: gradient of L\|g_cholesky_corr__\[0\]"):
        gates.check_gradient(_corrupted(correlated_bundle, mutate))


def test_corrupted_correlation_matrix_fails(correlated_bundle):
    def mutate(raw):
        raw["points"][1]["pars"]["Cor_1"]["values"][1] += 1e-8          # Cor_1[2, 1], column-major
    with pytest.raises(OracleMismatch, match="point 1: the correlation matrix of 'g' differs from brms Cor_1"):
        gates.check_derived(_corrupted(correlated_bundle, mutate))


def test_independent_model_does_not_match_a_correlated_fixture(correlated_bundle):
    """The adapter refuses to put '(1 + x || g)' against brms's '(1 + x | g)' (and the reverse) instead of
    comparing the parameters the two share."""
    import dataclasses

    import autobayes as ab

    pr = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                   sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}})
    independent = ab.gaussian("y ~ x + (1 + x || g)", correlated_bundle.data, pr)
    with pytest.raises(OracleMismatch, match="AutoBayes's term '\\(1 \\+ x \\|\\| g\\)' is independent"):
        gates.Bundle("gauss_slope_correlated", model=independent)
    if fx.has_fixture("gauss_slope_independent"):
        case = fx.load_case("gauss_slope_independent")
        data = fx.read_frame(case, fx.fixture_dir("gauss_slope_independent") / "data.csv")
        correlated = ab.gaussian("y ~ x + (1 + x | g)", data, dataclasses.replace(pr, cor=ab.LKJ(2)))
        with pytest.raises(OracleMismatch, match="AutoBayes's term '\\(1 \\+ x \\| g\\)' is correlated"):
            gates.Bundle("gauss_slope_independent", model=correlated)


# -- the ordinal and categorical cases (B05): the ordered thresholds and the per-category parameters ---------------------
@pytest.fixture(scope="module")
def cumulative_bundle():
    if not fx.has_fixture("cumulative_varint") or fx.load_case("cumulative_varint").get("gate"):
        pytest.skip("cumulative case not frozen or still gated")
    return gates.Bundle("cumulative_varint")


@pytest.fixture(scope="module")
def categorical_bundle():
    if not fx.has_fixture("categorical_poponly") or fx.load_case("categorical_poponly").get("gate"):
        pytest.skip("categorical case not frozen or still gated")
    return gates.Bundle("categorical_poponly")


def _coordinates(raw, par) -> list[int]:
    return [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == par]


def test_corrupted_threshold_coordinate_fails(cumulative_bundle):
    """brms's unconstrained Intercept values must be Stan's ordered coordinates of Intercept at every point, so a fixture
    whose u disagrees (as if brms sampled the thresholds through another transform) stops the gradient gate."""
    def mutate(raw):
        raw["points"][2]["u"]["values"][_coordinates(raw, "Intercept")[1]] += 1e-6
    with pytest.raises(OracleMismatch, match="point 2: Stan's ordered coordinate 1 of Intercept"):
        gates.check_gradient(_corrupted(cumulative_bundle, mutate))


def test_corrupted_threshold_gradient_fails(cumulative_bundle):
    def mutate(raw):
        raw["points"][0]["grad"]["values"][_coordinates(raw, "Intercept")[2]] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 0: gradient of Intercept_ordered__\[2\]"):
        gates.check_gradient(_corrupted(cumulative_bundle, mutate))


def test_corrupted_raw_thresholds_fail(cumulative_bundle):
    def mutate(raw):
        raw["points"][1]["pars"]["b_Intercept"]["values"][2] += 1e-8
    with pytest.raises(OracleMismatch, match="point 1: raw-scale thresholds .* vs brms b_Intercept"):
        gates.check_derived(_corrupted(cumulative_bundle, mutate))


def test_corrupted_category_probability_fails(cumulative_bundle, categorical_bundle):
    for bundle in (cumulative_bundle, categorical_bundle):
        def mutate(raw):
            raw["predictions"]["newdata"]["epred"]["values"][7] += 1e-8
        with pytest.raises(OracleMismatch, match="newdata: 'p' differs from brms posterior_epred"):
            gates.check_predictions(_corrupted(bundle, mutate))


def test_corrupted_category_parameters_fail(categorical_bundle):
    def gradient(raw):
        raw["points"][3]["grad"]["values"][_coordinates(raw, "b_muc3")[0]] += 1e-6     # it is 0.067: 1e-7 is the floor
    with pytest.raises(OracleMismatch, match=r"point 3: gradient of x\[1\]"):                 # muc3 is the second category
        gates.check_gradient(_corrupted(categorical_bundle, gradient))
    def linpred(raw):
        raw["predictions"]["fitted"]["linpred"]["values"][40] += 1e-8                        # row 4 of muc3 (column-major)
    with pytest.raises(OracleMismatch, match="fitted: linear predictor differs"):
        gates.check_predictions(_corrupted(categorical_bundle, linpred))
    def raw_intercept(raw):
        raw["points"][5]["pars"]["b_muc2_Intercept"]["values"][0] += 1e-8
    with pytest.raises(OracleMismatch, match="point 5: raw-scale intercept of muc2"):
        gates.check_derived(_corrupted(categorical_bundle, raw_intercept))
    def design(raw):
        raw["standata"]["X_muc3"]["values"][-1] += 1e-11                                      # the last x of the muc3 design
    with pytest.raises(OracleMismatch, match="matches 0 brms columns"):
        _corrupted(categorical_bundle, design)


def test_a_model_of_the_other_kind_does_not_match(cumulative_bundle, categorical_bundle):
    """The adapter refuses to put a model without thresholds against brms's ordered thresholds, or a model without
    category designs against brms's X_mu<k>."""
    import autobayes as ab

    data = cumulative_bundle.data
    pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
    with pytest.raises(OracleMismatch, match="expected one brms intercept column"):
        gates.Bundle("cumulative_varint", model=ab.poisson("ord ~ x + (1 | g)", data, pri))
    with pytest.raises(KeyError):
        gates.Bundle("categorical_poponly", model=ab.cumulative("ord ~ x", categorical_bundle.data,
                                                              ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5))))


# -- D04: unordered thresholds and category-specific effects -------------------------------------------------------------
@pytest.fixture(scope="module")
def acat_cs_bundle():
    if not fx.has_fixture("acat_cs") or fx.load_case("acat_cs").get("gate"):
        pytest.skip("acat_cs not frozen or still gated")
    return gates.Bundle("acat_cs")


def test_corrupted_unordered_threshold_and_bcs_gradients_fail(acat_cs_bundle):
    def threshold(raw):
        raw["points"][0]["grad"]["values"][_coordinates(raw, "Intercept")[1]] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 0: gradient of Intercept\[1\]"):
        gates.check_gradient(_corrupted(acat_cs_bundle, threshold))
    def bcs(raw):
        raw["points"][4]["grad"]["values"][_coordinates(raw, "bcs")[2]] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 4: gradient of cs\(x\)\[2\]"):
        gates.check_gradient(_corrupted(acat_cs_bundle, bcs))


def test_corrupted_threshold_predictor_and_design_fail(acat_cs_bundle):
    def linpred(raw):
        raw["predictions"]["fitted"]["linpred"]["values"][40] += 1e-8        # point 0, row 5, threshold 1 (column-major)
    with pytest.raises(OracleMismatch, match="point 0: the graph's 'mu_cs' differs from brms"):
        gates.check_derived(_corrupted(acat_cs_bundle, linpred))
    with pytest.raises(OracleMismatch, match="fitted: linear predictor differs"):
        gates.check_predictions(_corrupted(acat_cs_bundle, linpred))
    def design(raw):
        raw["standata"]["Xcs"]["values"][-1] += 1e-11
    with pytest.raises(OracleMismatch, match="brms's Xcs column 1 matches 0 category-specific columns"):
        _corrupted(acat_cs_bundle, design)


def test_a_model_without_the_fixture_s_thresholds_or_cs_terms_does_not_match(acat_cs_bundle):
    import autobayes as ab

    data = acat_cs_bundle.data
    pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
    with pytest.raises(OracleMismatch, match="AutoBayes's cumulative model has ordered thresholds but brms declares 'Intercept' as "
                                             "'vector\\[nthres\\] Intercept;'"):
        gates.Bundle("acat_varint", model=ab.cumulative("ord ~ x + (1 | g)", fx.read_frame(fx.load_case("acat_varint"),
                                                                                           fx.fixture_dir("acat_varint") / "data.csv"), pri))
    with pytest.raises(OracleMismatch, match="brms has Kcs = 1 category-specific columns"):
        gates.Bundle("acat_cs", model=ab.acat("ord ~ w + (1 | g)", data, pri))


# -- two-reference acceptance (P6): brms where it is accurate, the committed 50-digit reference where it is not -----------
REFERENCE_FILES = sorted(two_reference.REFERENCES.glob("*.json"))
OPTED_IN = sorted(cid for cid in fx.case_ids() if "reference" in (fx.load_case(cid).get("gate") or {}))


def test_every_opted_in_case_has_its_reference_and_every_reference_a_case():
    assert OPTED_IN == ["bernoulli_cloglog", "cumulative_probit_varint", "gamma_censored_interval", "gauss_censored", "gauss_truncated",
                        "wiener_distributional_bs", "wiener_varint"]
    assert [p.stem for p in REFERENCE_FILES] == OPTED_IN
    for cid in OPTED_IN:
        assert fx.load_case(cid)["gate"]["reference"] == f"{cid}.json"


@pytest.mark.parametrize("path", REFERENCE_FILES, ids=lambda p: p.stem)
def test_reference_was_computed_from_the_frozen_fixture(path):
    """The sha256 of every file a reference was computed from is that of the committed file: the reference certifies
    exactly the frozen fixture."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    d = fx.fixture_dir(path.stem)
    assert raw["case"] == path.stem and raw["generator"] == "oracle/reference.py"
    read = [d / name for name in ("case_brms.json", "data.csv", "newdata.csv", "fixture.json") if (d / name).exists()]
    assert list(raw["sources"]) == [str(f.relative_to(two_reference.REPO)) for f in read]
    for name, digest in raw["sources"].items():
        assert fx.sha256(two_reference.REPO / name) == digest, name


def test_references_regenerate_byte_for_byte():
    """oracle/reference.py, an independent implementation (the standard library only, nothing from autobayes), recomputes
    every committed reference from its fixture and reproduces the file exactly."""
    proc = subprocess.run([sys.executable, "oracle/reference.py", "--check", "--all"], cwd=two_reference.REPO,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines() == [f"{p.stem}: identical" for p in REFERENCE_FILES]


@pytest.mark.parametrize("path", REFERENCE_FILES, ids=lambda p: p.stem)
def test_reference_log_densities_agree_with_brms(path):
    """brms's log densities of these cases are exact to LOGP_REL at every frozen point (log densities never fall back), so
    the reference's equal them: its 50-digit model is brms's, normalising constants included. Every value is written with
    30 significant digits."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    f = fx.Fixture.load(path.stem)
    assert raw["upars"] == [f"{e['par']}[{e['elem']}]" for e in f.upars] and len(raw["points"]) == len(f.points)
    for k, (r, p) in enumerate(zip(raw["points"], f.points)):
        assert rel_err(float(r["lp"]), p.lp) <= LOGP_REL and rel_err(float(r["lp_nojac"]), p.lp_nojac) <= LOGP_REL, k
        for value in [r["lp"], r["lp_nojac"], *r["grad"], *r["grad_nojac"]]:
            assert len(value.split("E")[0].lstrip("-").replace(".", "")) == 30, value


@pytest.fixture(scope="module")
def opted_in_bundles():
    return {cid: gates.Bundle(cid) for cid in OPTED_IN}


def _with(bundle, case=None, **cached):
    """A shallow copy of ``bundle``, with another case (its reference then read afresh) or other cached values."""
    other = copy.copy(bundle)
    if case is not None:
        other.case = case
        other.__dict__.pop("reference", None)
    other.__dict__.update(cached)
    return other


def _gate(bundle, **gate):
    return _with(bundle, case={**bundle.case, "gate": {**bundle.case["gate"], **gate}})


def _no_gate(bundle):
    return _with(bundle, case={k: v for k, v in bundle.case.items() if k != "gate"})


def test_fallback_is_reported_with_its_count_and_place(opted_in_bundles):
    """Each opted-in case passes G2's gradient (and gauss_truncated G4) with exactly these entries on the reference: the
    points where brms's own arithmetic is off the exact value (divergences D18, D24, D25); the wiener cases pass G4 on the
    reference at every expected value (D148: brms's posterior_epred is not the mean). The report is a
    ReferenceFallback warning, which pytest's summary shows, and the gate's return value."""
    censored = ["point 0 z_1[1]"] + [f"point 4 {c}" for c in ("b[1]", "Intercept[1]", "sigma[1]", "sd_1[1]", "z_1[1]", "z_1[2]", "z_1[4]")] \
        + [f"point 5 {c}" for c in ("b[1]", "Intercept[1]", "sigma[1]", "sd_1[1]", "z_1[1]", "z_1[2]")] + ["point 6 z_1[4]", "point 7 Intercept[1]"]
    expected = {("bernoulli_cloglog", "G2 gradient"): (144, ["point 5 b[1]"]),
                ("gauss_censored", "G2 gradient"): (160, censored),
                ("gauss_truncated", "G2 gradient"): (160, ["point 4 b[1]"])}
    for (cid, gate), (entries, coordinates) in expected.items():
        with pytest.warns(two_reference.ReferenceFallback, match=rf"{cid} {gate}: {2 * len(coordinates)} of {entries} entries passed on the reference"):
            report = gates.check_gradient(opted_in_bundles[cid])
        assert (report.case, report.gate, report.entries) == (cid, gate, entries)
        assert list(report.where) == [f"{c} ({j} the Jacobian)" for c in coordinates for j in ("with", "without")]
    epred = ["fitted epred point 4 row 28"] + [f"fitted epred_nogroup point 4 row {r}" for r in (4, 13, 28, 32)] + ["newdata epred_nogroup point 3 row 3"]
    with pytest.warns(two_reference.ReferenceFallback, match=r"gauss_truncated G4 epred: 6 of 656 entries passed on the reference, where "
                                                              r"brms is off it by more than the tolerance: fitted epred point 4 row 28; "):
        report = gates.check_predictions(opted_in_bundles["gauss_truncated"])
    assert report.entries == 656 and list(report.where) == epred
    assert "point 4 b[1] (with and without the Jacobian)" in str(gates.check_gradient(opted_in_bundles["gauss_truncated"]))
    for cid in ("wiener_distributional_bs", "wiener_varint"):        # D148: brms's posterior_epred is off the mean everywhere
        with pytest.warns(two_reference.ReferenceFallback, match=rf"{cid} G4 epred: 656 of 656 entries passed on the reference"):
            report = gates.check_predictions(opted_in_bundles[cid])
        assert report.entries == report.count == 656


def _corrupt_autobayes_gradient(b, k, i, factor):
    """``b`` with AutoBayes's gradient at seeded point k, with and without the Jacobian, scaled by ``factor`` in brms's
    unconstrained coordinate i."""
    ((var, idx),) = [(v, j) for c, v, j in adapter.coordinate_map(b.fixture, b.model, b.cmap, b.gmap) if c == i]
    target = adapter.value_point(b.model, b.fixture.points[k], b.cmap, b.gmap)
    pos = adapter.value_layout(b.model, target)[var].start + idx

    def corrupt(f):
        def g(vp):
            out = np.array(f(vp), dtype=float)
            if all(np.array_equal(np.asarray(vp[n]), np.asarray(target[n])) for n in target):
                out[pos] *= factor
            return out
        return g
    return _with(b, dlogp=corrupt(b.dlogp), dlogp_nojac=corrupt(b.dlogp_nojac))


def test_corrupted_autobayes_gradient_off_both_brms_and_the_reference_fails(opted_in_bundles):
    """At point 5 of bernoulli_cloglog brms's gradient in b[1] is 3.8e-7 off the reference and AutoBayes passes on the
    reference; AutoBayes moved 1e-6 off is off both and fails (rule 2). Moved off where brms is accurate, it fails too:
    the reference agrees with brms there, so nothing falls back."""
    b = opted_in_bundles["bernoulli_cloglog"]
    with pytest.raises(OracleMismatch, match=r"point 5: gradient of x\[0\] \(brms b\[1\]\) with the Jacobian: AutoBayes .* is off both "
                                             r"brms -1\.00559172179031\d* \(1\.\d+e-06\) and the reference -1\.0055921064960\d* \(1\.00e-06\)"):
        gates.check_gradient(_corrupt_autobayes_gradient(b, 5, 0, 1 + 1e-6))
    with pytest.raises(OracleMismatch, match=r"point 0: gradient of x\[0\] \(brms b\[1\]\) with the Jacobian: AutoBayes .* off brms .*, "
                                             r"which agrees with the reference .* so nothing falls back"):
        gates.check_gradient(_corrupt_autobayes_gradient(b, 0, 0, 1 + 1e-6))


def _corrupt_reference(b, mutate):
    ref = b.reference
    raw = copy.deepcopy(ref.raw)
    mutate(raw)
    return _with(b, reference=two_reference.Reference(ref.case_id, ref.file, ref.divergence, ref.fallback, raw))


def test_corrupted_reference_where_autobayes_and_brms_agree_fails(opted_in_bundles):
    """Rule 3: wherever AutoBayes and brms agree, the reference must agree with brms; one entry 1e-6 off fails the gate,
    whatever AutoBayes does (here AutoBayes and brms agree at point 1 of gauss_censored and at every expected value of
    point 0 of gauss_truncated)."""
    def gradient(raw):
        g = raw["points"][1]["grad_nojac"]
        g[3] = repr(float(g[3]) * (1 + 1e-6))                                    # sd_1, without the Jacobian
    with pytest.raises(OracleMismatch, match=r"point 1: gradient of 1\|g_sigma_log__\[0\] \(brms sd_1\[1\]\) without the Jacobian: "
                                             r"the reference is wrong: AutoBayes .* and brms .* agree"):
        gates.check_gradient(_corrupt_reference(opted_in_bundles["gauss_censored"], gradient))

    def epred(raw):
        row = raw["predictions"]["fitted"]["epred"][0]
        row[0] = repr(float(row[0]) * (1 + 1e-6))
    with pytest.raises(OracleMismatch, match=r"fitted: point 0, row 0: 'mu' vs brms posterior_epred: the reference is wrong"):
        gates.check_predictions(_corrupt_reference(opted_in_bundles["gauss_truncated"], epred))


@pytest.mark.parametrize("kind", two_reference.LOG_DENSITIES)
def test_fallback_on_a_log_density_is_refused(opted_in_bundles, kind):
    """A gate field that lets a log density fall back is refused; and an opted-in case's log densities compare with brms
    only (a fixture whose lp is moved fails G2 although the reference holds the exact value)."""
    b = opted_in_bundles["gauss_censored"]
    with pytest.raises(OracleMismatch, match=rf"gauss_censored: log densities never fall back on a reference \(\['{kind}'\] in the gate's fallback\)"):
        gates.check_gradient(_gate(b, fallback=["gradient", kind]))
    with pytest.raises(OracleMismatch, match="log densities never fall back"):
        two_reference.load({**b.case, "gate": {**b.case["gate"], "fallback": [kind]}})

    def moved(raw):
        raw["points"][2]["lp"] += 1e-5
    with pytest.raises(OracleMismatch, match="point 2: log density"):
        gates.check_logp(_corrupted(b, moved))


@pytest.mark.parametrize("gate,needle", [
    ({"fallback": ["gradient", "linpred"]}, r"\['linpred'\] is not a quantity kind that may fall back"),
    ({"fallback": []}, "must be a non-empty list"),
    ({"fallback": "gradient"}, "must be a non-empty list"),
    ({"expect": "fail"}, "a gate field that names a reference expects 'pass'"),
    ({"divergence": None}, "the gate names no divergence"),
    ({"reference": "gauss_truncated.json"}, "gauss_truncated.json is the reference of 'gauss_truncated'"),
    ({"reference": "../fixtures/gauss_censored/fixture.json"}, "no reference file"),
])
def test_malformed_opt_in_is_refused(opted_in_bundles, gate, needle):
    with pytest.raises(OracleMismatch, match=needle):
        gates.check_gradient(_gate(opted_in_bundles["gauss_censored"], **gate))


def test_a_case_without_the_opt_in_behaves_as_before(gauss_bundle, opted_in_bundles):
    """Without a gate field nothing is read and nothing falls back: the gates return None, warn nothing and fail as
    before. An opted-in case with its opt-in removed fails the brms-only gates exactly where brms is inexact (its reference
    file is not used), and a kind not listed in the fallback compares with brms only."""
    assert gauss_bundle.reference is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", two_reference.ReferenceFallback)
        assert gates.check_gradient(gauss_bundle) is None and gates.check_predictions(gauss_bundle) is None

    def mutate(raw):
        raw["points"][0]["grad"]["values"][2] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"^point 0: gradient of sigma_log__\[0\] AutoBayes .* vs brms "):
        gates.check_gradient(_corrupted(gauss_bundle, mutate))
    for cid, gate, needle in [
            ("bernoulli_cloglog", gates.check_gradient, r"^point 5: gradient of x\[0\] AutoBayes np\.float64\(-1\.0055921064960\d*\) vs brms "
                                                        r"np\.float64\(-1\.00559172179031"),
            ("gauss_censored", gates.check_gradient, r"^point 0: gradient of 1\|g_offset\[0\] AutoBayes"),
            ("gauss_truncated", gates.check_gradient, r"^point 4: gradient of x\[0\] AutoBayes np\.float64\(-1\.519346934017\d*\) vs brms "
                                                      r"np\.float64\(-1\.5193463027823"),
            ("gauss_truncated", gates.check_predictions, r"^fitted: 'mu' differs from brms posterior_epred by 4\.1")]:
        b = _no_gate(opted_in_bundles[cid])
        assert b.reference is None and (two_reference.REFERENCES / f"{cid}.json").exists()
        with warnings.catch_warnings():
            warnings.simplefilter("error", two_reference.ReferenceFallback)
            with pytest.raises(OracleMismatch, match=needle):
                gate(b)
    truncated = opted_in_bundles["gauss_truncated"]
    with pytest.raises(OracleMismatch, match=r"^fitted: 'mu' differs from brms posterior_epred by 4\.1"):
        gates.check_predictions(_gate(truncated, fallback=["gradient"]))
    with pytest.raises(OracleMismatch, match=r"^point 4: gradient of x\[0\] AutoBayes"):
        gates.check_gradient(_gate(truncated, fallback=["epred"]))


# -- C07: post-processing: log_lik, bayes_R2 and loo against the fixture's post block ----------------------------------
@pytest.fixture(scope="module")
def post_bundle():
    if not fx.has_fixture("gauss_varint_post") or fx.load_case("gauss_varint_post").get("gate"):
        pytest.skip("post case not frozen or still gated")
    return gates.Bundle("gauss_varint_post")


@pytest.fixture(scope="module")
def psis_bundle():
    if not fx.has_fixture("gauss_varint_post_psis") or fx.load_case("gauss_varint_post_psis").get("gate"):
        pytest.skip("post case not frozen or still gated")
    return gates.Bundle("gauss_varint_post_psis")


def _bump(values, i, delta):
    """Add ``delta`` to entry ``i`` of a fixture array, whose values are numbers, or strings when the array has an Inf."""
    values[i] = repr(float(values[i]) + delta) if isinstance(values[i], str) else values[i] + delta


def test_post_gate_passes_on_the_frozen_post_fixtures(post_bundle, psis_bundle):
    for bundle in (post_bundle, psis_bundle):
        idata = gates.fixed_parameter_draws(bundle)                 # brms's fixed-parameter fit: one chain per point
        assert dict(idata.posterior.sizes)["chain"] == len(bundle.fixture.points)
        assert dict(idata.posterior.sizes)["draw"] == 1
        gates.check_post(bundle)


def test_corrupted_log_lik_fails(post_bundle):
    def mutate(raw):
        _bump(raw["post"]["log_lik"]["values"], 17, 1e-8)           # a log density of a few units: 1e-8 is above 1e-9
    with pytest.raises(OracleMismatch, match="log_lik differs from brms"):
        gates.check_post(_corrupted(post_bundle, mutate))


def test_corrupted_bayes_R2_draw_fails(post_bundle):
    def mutate(raw):
        _bump(raw["post"]["bayes_R2"]["values"], 3, 1e-8)
    with pytest.raises(OracleMismatch, match="bayes_R2 differs from brms"):
        gates.check_post(_corrupted(post_bundle, mutate))


def test_corrupted_pointwise_elpd_fails(post_bundle, psis_bundle):
    for bundle in (post_bundle, psis_bundle):                       # raw importance weights (k Inf), and PSIS
        def mutate(raw):
            _bump(raw["post"]["loo"]["pointwise"]["values"], 4, 1e-7)   # row 5's elpd_loo (column-major: the first column)
        with pytest.raises(OracleMismatch, match="loo pointwise elpd_loo differs from brms"):
            gates.check_post(_corrupted(bundle, mutate))


def test_corrupted_loo_diagnostics_fail(psis_bundle):
    def k_hat(raw):
        _bump(raw["post"]["loo"]["pareto_k"]["values"], 9, 1e-8)
    with pytest.raises(OracleMismatch, match="loo pareto_k differs from brms"):
        gates.check_post(_corrupted(psis_bundle, k_hat))
    def n_eff(raw):
        _bump(raw["post"]["loo"]["n_eff"]["values"], 0, 1e-6)       # n_eff is about 62
    with pytest.raises(OracleMismatch, match="loo n_eff differs from brms"):
        gates.check_post(_corrupted(psis_bundle, n_eff))
    def se(raw):
        _bump(raw["post"]["loo"]["estimates"]["values"], 3, 1e-7)   # SE of elpd_loo (column-major: rows, then SE)
    with pytest.raises(OracleMismatch, match="loo estimates differs from brms"):
        gates.check_post(_corrupted(psis_bundle, se))


def test_post_non_finite_entries_must_match_exactly(post_bundle):
    """brms's k-hat is Inf on every row of the 8-draw fixture (a Pareto tail too short to fit): a finite brms value there
    fails, as does an Inf where AutoBayes is finite; equal infinities and NaNs pass, a different one does not."""
    def finite_k(raw):
        raw["post"]["loo"]["pareto_k"]["values"][2] = "0.5"
    with pytest.raises(OracleMismatch, match="loo pareto_k: non-finite entries differ"):
        gates.check_post(_corrupted(post_bundle, finite_k))
    def finite_influence(raw):
        raw["post"]["loo"]["pointwise"]["values"][4 * 36 + 7] = "1.5"     # influence_pareto_k of row 8 (fifth column)
    with pytest.raises(OracleMismatch, match="loo pointwise influence_pareto_k: non-finite entries differ"):
        gates.check_post(_corrupted(post_bundle, finite_influence))
    def infinite_n_eff(raw):
        raw["post"]["loo"]["n_eff"]["values"][0] = "Inf"
    with pytest.raises(OracleMismatch, match="loo n_eff: non-finite entries differ"):
        gates.check_post(_corrupted(post_bundle, infinite_n_eff))
    gates.compare_post("same", [np.inf, -np.inf, np.nan, 1.0], [np.inf, -np.inf, np.nan, 1.0 + 1e-12])
    for ours, theirs in (([np.inf], [-np.inf]), ([np.nan], [np.inf]), ([np.inf], [1e300]), ([np.nan], [0.0]), ([0.0], [np.nan])):
        with pytest.raises(OracleMismatch, match="non-finite entries differ"):
            gates.compare_post("differs", ours, theirs)


def test_post_block_and_columns_are_required(post_bundle):
    def no_post(raw):
        del raw["post"]
    with pytest.raises(OracleMismatch, match="has no post block"):
        gates.check_post(_corrupted(post_bundle, no_post))
    def renamed(raw):
        raw["post"]["loo"]["pointwise"]["colnames"][1] = "mcse"
    with pytest.raises(OracleMismatch, match="loo pointwise columns"):
        gates.check_post(_corrupted(post_bundle, renamed))


# -- grouped thresholds (F06b): brms's Intercept_k per level of thres(gr = z) -----------------------------------------------
@pytest.fixture(scope="module")
def grouped_bundle():
    if not fx.has_fixture("cumulative_thres_gr") or fx.load_case("cumulative_thres_gr").get("gate"):
        pytest.skip("grouped-threshold case not frozen or still gated")
    return gates.Bundle("cumulative_thres_gr")


def test_corrupted_threshold_groups_fail_g1(grouped_bundle):
    def jthres(raw):
        raw["standata"]["Jthres"]["values"][0] = 4                     # row 0 (level p) pointed to level q's block
    def nthres(raw):
        raw["standata"]["nthres"]["values"][2] = 3
    with pytest.raises(OracleMismatch, match="is no level of the AutoBayes model's 'thres\\(gr = z\\)'|is not its block"):
        gates.check_design(_corrupted(grouped_bundle, jthres))
    with pytest.raises(OracleMismatch, match="brms has ngrthres = 3 levels with nthres \\[3, 3, 3\\]"):
        gates.check_design(_corrupted(grouped_bundle, nthres))


def test_corrupted_grouped_threshold_coordinate_and_gradient_fail(grouped_bundle):
    def coordinate(raw):
        raw["points"][1]["u"]["values"][_coordinates(raw, "Intercept_2")[1]] += 1e-6
    def gradient(raw):
        raw["points"][0]["grad"]["values"][_coordinates(raw, "Intercept_3")[1]] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match="point 1: Stan's ordered coordinate 1 of Intercept_2"):
        gates.check_gradient(_corrupted(grouped_bundle, coordinate))
    with pytest.raises(OracleMismatch, match=r"point 0: gradient of Intercept_ordered__\[7\]"):   # level r's second, the merged vector's last
        gates.check_gradient(_corrupted(grouped_bundle, gradient))


def test_corrupted_grouped_thresholds_fail_g3(grouped_bundle):
    def raw_threshold(raw):
        raw["points"][2]["pars"]["b_Intercept_2"]["values"][1] += 1e-8
    def merged(raw):
        raw["points"][0]["pars"]["merged_Intercept"]["values"][4] += 1e-8
    with pytest.raises(OracleMismatch, match="(?s)point 2: raw-scale thresholds .* vs brms b_Intercept_k"):
        gates.check_derived(_corrupted(grouped_bundle, raw_threshold))
    with pytest.raises(OracleMismatch, match="point 0: brms's merged_Intercept is not its Intercept_k in turn"):
        gates.check_derived(_corrupted(grouped_bundle, merged))


def test_corrupted_zero_probability_fails_g4(grouped_bundle):
    """brms's expected value of category 4 on the rows of level r is exactly 0; a fixture with any other value there fails."""
    r = int(np.flatnonzero(grouped_bundle.model.design.thresholds.index == 2)[0])
    def mutate(raw):
        values = np.asarray(raw["predictions"]["fitted"]["epred"]["values"], dtype=float).reshape(raw["predictions"]["fitted"]["epred"]["dims"], order="F")
        values[0, r, 3] = 1e-6
        raw["predictions"]["fitted"]["epred"]["values"] = values.ravel(order="F").tolist()
    with pytest.raises(OracleMismatch, match="fitted: 'p' differs from brms posterior_epred"):
        gates.check_predictions(_corrupted(grouped_bundle, mutate))


def test_grouped_and_ungrouped_thresholds_do_not_match_each_other(grouped_bundle):
    import autobayes as ab

    pri = ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
    with pytest.raises(OracleMismatch, match="AutoBayes's cumulative model has ordered thresholds but brms declares 'Intercept' as ''"):
        gates.Bundle("cumulative_thres_gr", model=ab.cumulative("ord ~ x + (1 | g)", grouped_bundle.data, pri))
    plain = fx.read_frame(fx.load_case("cumulative_varint"), fx.fixture_dir("cumulative_varint") / "data.csv")
    with pytest.raises(OracleMismatch, match="brms's standata has no Jthres / ngrthres"):
        gates.Bundle("cumulative_varint", model=ab.cumulative("ord | thres(gr = z) ~ x + (1 | g)", plain, pri))


# -- a gate field that lists the gates expected to fail (E06: cumulative_cloglog) ---------------------------------------------
def test_listed_gates_fail_strictly_and_the_others_must_pass():
    """A gate field with ``"gates"`` makes only those gates strict expected failures: one that fails with an
    ``OracleMismatch`` is an expected failure, one that passes is an error, and an unlisted gate runs as usual."""
    from tests.oracle.test_baseline import GATE_NAMES, _gate

    b = gates.Bundle("cumulative_cloglog")
    assert set(b.case["gate"]["gates"]) == {"g2_logp", "g2_gradient"} and set(b.case["gate"]["gates"]) <= set(GATE_NAMES)
    with pytest.raises(pytest.xfail.Exception):                       # brms's own log density is inexact here (D132)
        _gate(b, "g2_logp", gates.check_logp)
    _gate(b, "g1", gates.check_design)                                # not listed: must pass, and does
    listed_but_passing = _with(b, case={**b.case, "gate": {**b.case["gate"], "gates": ["g1"]}})
    with pytest.raises(pytest.fail.Exception, match="gate g1 was expected to fail"):
        _gate(listed_but_passing, "g1", gates.check_design)
