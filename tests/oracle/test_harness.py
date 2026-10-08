"""The harness itself: frozen fixtures are what their cases say, the brms side agrees with the hand-written
reference densities of tests/reference_logp.py and tests/reference_binomial.py (so the oracle is calibrated
independently of autobayes), and a corrupted fixture fails every gate it touches."""
import copy
import json

import numpy as np
import pytest

from tests.oracle import datasets
from tests.oracle import fixture as fx
from tests.oracle import gates
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
    for frame in (data, nd):
        if frame is not None:
            assert not frame.isna().any().any()
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
