"""The oracle's paths for G08's multivariate extensions bite: a fixture corrupted in what they add (the correlation ID across
responses' group: its L_m and Cor_m, its coefficients' responses and order, each response's slice of sd_m and z_m; the weights of
each response and of the joint density) fails the gate that reads it, and a model that drops what brms has (the ID, the
weights) is not put in correspondence with the fixture."""
import copy

import pytest

import autobayes as ab
from tests.oracle import adapter, gates
from tests.oracle import fixture as fx
from tests.oracle.adapter import OracleMismatch


def _bundle(case_id, model=None):
    if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
        pytest.skip(f"{case_id} not frozen or still gated")
    return gates.Bundle(case_id, model=model)


@pytest.fixture(scope="module")
def id_across():
    return _bundle("gauss_mv_id_across_rescor")


@pytest.fixture(scope="module")
def weights():
    return _bundle("gauss_mv_weights_rescor")


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


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_gradient, _add(lambda raw: (raw["points"][2]["u"]["values"], _index(raw, "L_1", 2)), 1e-6),
     "point 2: Stan's cholesky_corr coordinate 0 of L_1"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][0]["grad"]["values"], _index(raw, "L_1", 2)), 1 + 1e-6),
     r"point 0: gradient of L\|g_cholesky_corr__\[0\]"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][1]["grad"]["values"], _index(raw, "sd_1", 2)), 1 + 1e-6),
     r"point 1: gradient of y2_1\|g_sigma_log__\[0\]"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][1]["grad"]["values"], _index(raw, "z_1", 4)), 1 + 1e-6),
     r"point 1: gradient of y2_1\|g_offset\[1\]"),
    (gates.check_logp, _add(lambda raw: (raw["points"][3]["pars"]["lprior"]["values"], 0), 1e-5), "point 3: log density"),
    (gates.check_derived, _add(lambda raw: (raw["points"][1]["pars"]["Cor_1"]["values"], 1), 1e-8),
     "point 1: the correlation matrix of 'g' across the responses differs from brms Cor_1"),
    (gates.check_derived, _add(lambda raw: (raw["points"][1]["pars"]["r_1_y2_2"]["values"], 3), 1e-8),
     "point 1: group effects of 'Intercept' of 'g' for 'y2' differ from brms r_1_y2_2"),
    (gates.check_design, _add(lambda raw: (raw["standata"]["M_1"]["values"], 0), 1), "brms group 1 of 'g': M_1 = 3"),
    (gates.check_design, lambda raw: [g.update(cor=False) for g in raw["groups"]], r"cor \[False, False\]"),
    (gates.check_design, lambda raw: [g.update(cn={1: 2, 2: 1}[g["cn"]]) for g in raw["groups"]],
     r"brms data without an AutoBayes counterpart in the multivariate model: \['Z_1_y2_2', 'Z_1_y_1'\]"),
])
def test_a_corrupted_id_across_fixture_fails(id_across, gate, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        gate(_corrupted(id_across, mutate))


def test_a_model_without_the_id_is_not_brms_s(id_across):
    case = id_across.case
    spec = case["autobayes"]
    priors = adapter.priors_from_spec({**spec["priors"], "cor": None})
    plain = ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", id_across.data, priors, **spec["kwargs"])
    with pytest.raises(OracleMismatch, match=r"brms group 1 correlates \[\('y', 'Intercept'\), \('y2', 'Intercept'\)\] of 'g' across"):
        gates.Bundle(case["id"], model=plain)


def test_a_model_with_an_id_brms_lacks_is_not_brms_s():
    other = _bundle("gauss_mv_rescor")
    spec = other.case["autobayes"]
    priors = adapter.priors_from_spec({**spec["priors"], "cor": {"g": ["LKJ", 2]}})
    with_id = ab.gaussian("mvbind(y, y2) ~ x + (1 |p| g)", other.data, priors, **spec["kwargs"])
    with pytest.raises(OracleMismatch, match=r"the AutoBayes model correlates the responses' varying intercepts of \['g'\]"):
        gates.Bundle("gauss_mv_rescor", model=with_id)


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_design, _scale(lambda raw: (raw["standata"]["weights_y"]["values"], 4), 1.01),
     "brms weights each row's joint density by weights_y"),
    (gates.check_design, _scale(lambda raw: (raw["standata"]["weights_y2"]["values"], 4), 1.01), "response 'y2': weights differ"),
    (gates.check_logp, _add(lambda raw: (raw["points"][0]["pars"]["lprior"]["values"], 0), 1e-5), "point 0: log density"),
])
def test_a_corrupted_weights_fixture_fails(weights, gate, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        gate(_corrupted(weights, mutate))


def test_a_model_without_the_weights_is_not_brms_s(weights):
    spec = weights.case["autobayes"]
    plain = ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", weights.data, adapter.priors_from_spec(spec["priors"]), **spec["kwargs"])
    with pytest.raises(OracleMismatch, match="brms weights each row's joint density by weights_y"):
        gates.check_design(gates.Bundle(weights.case["id"], model=plain))
