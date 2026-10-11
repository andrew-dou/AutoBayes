"""The harness itself: frozen fixtures are what their cases say, the brms side agrees with the hand-written
reference densities of tests/reference_logp.py and tests/reference_binomial.py (so the oracle is calibrated
independently of autobayes), and a corrupted fixture fails every gate it touches. Two-reference acceptance (P6): the
committed 50-digit references regenerate byte for byte from the fixtures they name, and the three rules hold."""
import copy
import json
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


# -- two-reference acceptance (P6): brms where it is accurate, the committed 50-digit reference where it is not -----------
REFERENCE_FILES = sorted(two_reference.REFERENCES.glob("*.json"))
OPTED_IN = sorted(cid for cid in fx.case_ids() if "reference" in (fx.load_case(cid).get("gate") or {}))


def test_every_opted_in_case_has_its_reference_and_every_reference_a_case():
    assert OPTED_IN == ["bernoulli_cloglog", "gauss_censored", "gauss_truncated"]
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
    points where brms's own arithmetic is off the exact value (divergences D18, D24, D25). The report is a
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
