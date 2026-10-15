"""The oracle's multivariate paths (C01) bite: a fixture corrupted in what a multivariate model adds (the residual correlation
``Lrescor`` and ``Rescor``, a response's slice of the predictions, its view's parameters and data, brms's responses and
group scopes, the shared ``cor`` and ``rescor`` rows of the default-prior table) fails the gate that reads it."""
import copy

import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch


def _bundle(case_id):
    if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
        pytest.skip(f"{case_id} not frozen or still gated")
    return gates.Bundle(case_id)


@pytest.fixture(scope="module")
def rescor():
    return _bundle("gauss_mv_rescor")


@pytest.fixture(scope="module")
def slopes():
    return _bundle("gauss_mv_slopes")


def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _index(raw, par, elem=1) -> int:
    (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == par and int(e["elem"]) == elem]
    return i


def _add(path, delta):
    def mutate(raw):
        values, k = path(raw)
        values[k] += delta
    return mutate


def _scale(path, factor):
    def mutate(raw):
        values, k = path(raw)
        values[k] *= factor
    return mutate


N = 36          # the cases' rows; predictions are (points, rows, responses), column-major


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_gradient, _add(lambda raw: (raw["points"][2]["u"]["values"], _index(raw, "Lrescor", 2)), 1e-6),
     "point 2: Stan's cholesky_corr coordinate 0 of Lrescor"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][0]["grad"]["values"], _index(raw, "Lrescor", 2)), 1 + 1e-6),
     r"point 0: gradient of Lrescor_cholesky_corr__\[0\]"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][1]["grad"]["values"], _index(raw, "sigma_y2")), 1 + 1e-6),
     r"point 1: gradient of y2_sigma_log__\[0\]"),
    (gates.check_logp, _add(lambda raw: (raw["points"][3]["pars"]["lprior"]["values"], 0), 1e-5), "point 3: log density"),
    (gates.check_derived, _add(lambda raw: (raw["points"][1]["pars"]["Rescor"]["values"], 1), 1e-8),
     "point 1: the residual correlation matrix differs from brms Rescor"),
    (gates.check_derived, _add(lambda raw: (raw["points"][1]["pars"]["r_2_y2_1"]["values"], 4), 1e-8),
     "point 1: group effects of 'Intercept' of 'g' for 'y2' differ from brms r_2_y2_1"),
    (gates.check_derived, _add(lambda raw: (raw["points"][5]["pars"]["b_y2_Intercept"]["values"], 0), 1e-8),
     "point 5: raw-scale intercept of 'y2'"),
    (gates.check_derived, _add(lambda raw: (raw["predictions"]["fitted"]["linpred"]["values"], 8 * N + 5), 1e-8),
     "point 5: the graph's linear predictor of 'y2' differs from brms"),
    (gates.check_predictions, _add(lambda raw: (raw["predictions"]["newdata"]["epred"]["values"], 8 * 5 + 7), 1e-8),
     "newdata: 'mu' differs from brms posterior_epred"),
    (gates.check_default_prior, lambda raw: [r.update(prior="lkj(2)") for r in raw["default_prior"] if r["class"] == "rescor"],
     "differs from brms's default_prior"),
    (gates.check_design, lambda raw: raw.update(responses=["y2", "y"]), r"brms's responses are \['y2', 'y'\]"),
    (gates.check_design, _add(lambda raw: (raw["standata"]["nrescor"]["values"], 0), 1), "nrescor = 2.0"),
])
def test_a_corrupted_multivariate_fixture_fails(rescor, gate, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        gate(_corrupted(rescor, mutate))


def test_swapped_group_scopes_fail(rescor):
    def mutate(raw):
        for g in raw["groups"]:
            g["resp"] = {"y": "y2", "y2": "y"}[g["resp"]]
    with pytest.raises(OracleMismatch, match="brms data without an AutoBayes counterpart in the multivariate model"):
        _corrupted(rescor, mutate)


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_gradient, _add(lambda raw: (raw["points"][2]["u"]["values"], _index(raw, "L_3", 2)), 1e-6),
     "point 2: Stan's cholesky_corr coordinate 0 of L_3"),
    (gates.check_derived, _add(lambda raw: (raw["points"][1]["pars"]["Cor_3"]["values"], 1), 1e-8),
     "point 1: the correlation matrix of 'g' for 'y2' differs from brms Cor_3"),
    (gates.check_derived, _add(lambda raw: (raw["points"][1]["pars"]["r_3_y2_2"]["values"], 2), 1e-8),
     "point 1: group effects of 'x' of 'g' for 'y2' differ from brms r_3_y2_2"),
    (gates.check_design, _add(lambda raw: (raw["standata"]["offsets_y2"]["values"], 7), 1e-11), "response 'y2': offsets differ"),
    (gates.check_default_prior, lambda raw: [r.update(resp="y") for r in raw["default_prior"] if r["class"] == "cor"],
     r"lack \[\('', 'cor'\)\]"),
])
def test_a_corrupted_multivariate_slopes_fixture_fails(slopes, gate, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        gate(_corrupted(slopes, mutate))
