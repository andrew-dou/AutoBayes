"""The oracle's C06 paths bite: a fixture corrupted in what ``me()`` and ``mi()`` add (brms's coding of the missing rows,
``Y`` with Inf, ``Jmi`` and ``Nmi``; the missing values ``Ymi``; the special coefficient ``bsp`` and its ``Ksp``; the noisy values
``Xn``, their noise and the latent values ``meanme``, ``sdme``, ``zme`` and ``Xme``; the ``mi``/``me`` rows of the default-prior
table) fails the gate that reads it, or the adapter that puts the fixture and the model in correspondence."""
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
def mi():
    return _bundle("gauss_mi")


@pytest.fixture(scope="module")
def me():
    return _bundle("gauss_me")


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


def _set(path, value):
    def mutate(raw):
        values, k = path(raw)
        values[k] = value
    return mutate


def _scale(path, factor):
    def mutate(raw):
        values, k = path(raw)
        values[k] *= factor
    return mutate


N = 36          # the cases' rows; the mi case's predictions are (points, rows, responses), column-major


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_design, _set(lambda raw: (raw["standata"]["Y_xm"]["values"], 1), "Inf"),
     "brms's responses are infinite (missing) at other rows than the AutoBayes model's missing ones"),
    (gates.check_design, _add(lambda raw: (raw["standata"]["Jmi_xm"]["values"], 2), 1), "response 'xm': brms's Jmi"),
    (gates.check_design, _add(lambda raw: (raw["standata"]["Nmi_xm"]["values"], 0), 1), "response 'xm': brms's Jmi"),
    (gates.check_design, _set(lambda raw: (raw["standata"]["ubmi_xm"]["values"], 0), 5.0), "brms bounds the missing values"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][0]["grad"]["values"], _index(raw, "Ymi_xm", 3)), 1 + 1e-6),
     r"point 0: gradient of Ymi_xm\[2\]"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][3]["grad_nojac"]["values"], _index(raw, "bsp_y")), 1 + 1e-6),
     r"point 3: gradient of y_mi\(xm\)\[0\]"),
    (gates.check_logp, _add(lambda raw: (raw["points"][1]["pars"]["Ymi_xm"]["values"], 4), 1e-4), "point 1: log density"),
    (gates.check_derived, _add(lambda raw: (raw["predictions"]["fitted"]["linpred"]["values"], 2), 1e-8),
     "point 2: the graph's linear predictor of 'y' differs from brms"),
    (gates.check_predictions, _add(lambda raw: (raw["predictions"]["fitted"]["epred"]["values"], 8 * 7 + 6), 1e-8),
     "fitted: 'mu' differs from brms posterior_epred"),
    (gates.check_predictions, _add(lambda raw: (raw["predictions"]["newdata"]["linpred"]["values"], 8 * 3 + 1), 1e-8),
     "newdata: linear predictor differs from brms posterior_linpred"),
    (gates.check_default_prior, lambda raw: [r.update(prior="normal(0, 1)") for r in raw["default_prior"] if r["coef"] == "mixm"],
     "differs from brms's default_prior"),
])
def test_a_corrupted_mi_fixture_fails(mi, gate, mutate, message):
    import re

    with pytest.raises(OracleMismatch, match=message if "\\" in message else re.escape(message)):
        gate(_corrupted(mi, mutate))


def test_mi_special_terms_are_counted(mi):
    with pytest.raises(OracleMismatch, match=r"brms has special terms \(Ksp, Xmo_\[\]\); the AutoBayes model has no monotonic term and 1 "
                                             r"me\(\) / mi\(\) term\(s\)"):
        _corrupted(mi, _set(lambda raw: (raw["standata"]["Ksp_y"]["values"], 0), 2))


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_gradient, _scale(lambda raw: (raw["points"][1]["grad"]["values"], _index(raw, "zme_1", 5)), 1 + 1e-6),
     r"point 1: gradient of zme_xobs\[4\]"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][2]["grad"]["values"], _index(raw, "sdme_1")), 1 + 1e-6),
     r"point 2: gradient of sdme_xobs_log__\[0\]"),
    (gates.check_gradient, _scale(lambda raw: (raw["points"][4]["grad"]["values"], _index(raw, "meanme_1")), 1 + 1e-6),
     r"point 4: gradient of meanme_xobs\[0\]"),
    (gates.check_logp, _add(lambda raw: (raw["points"][3]["pars"]["lprior"]["values"], 0), 1e-5), "point 3: log density"),
    (gates.check_derived, _add(lambda raw: (raw["points"][5]["pars"]["Xme_1"]["values"], 9), 1e-8),
     "point 5: the graph's latent values of 'xobs' differ from brms Xme_1"),
    (gates.check_predictions, _add(lambda raw: (raw["predictions"]["fitted"]["linpred"]["values"], 8 * 11 + 3), 1e-8),
     "fitted: linear predictor differs from brms posterior_linpred"),
    (gates.check_default_prior, lambda raw: [r.update(prior="normal(0, 1)") for r in raw["default_prior"]
                                             if r["class"] == "meanme" and not r["coef"]], "differs from brms's default_prior"),
    (gates.check_default_prior, lambda raw: [r.update(lb="") for r in raw["default_prior"] if r["class"] == "sdme" and not r["coef"]],
     "differs from brms's default_prior"),
])
def test_a_corrupted_me_fixture_fails(me, gate, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        gate(_corrupted(me, mutate))


@pytest.mark.parametrize("mutate, message", [
    (_add(lambda raw: (raw["standata"]["Xn_1"]["values"], 6), 1e-11), r"brms's Xn_1 is not the noisy values of 'me\(xobs, sdx\)'"),
    (_scale(lambda raw: (raw["standata"]["noise_1"]["values"], 0), 1.5), r"brms's noise_1 is not the known noise of 'me\(xobs, sdx\)'"),
    (_set(lambda raw: (raw["standata"]["Mme_1"]["values"], 0), 2), r"are not the one latent variable of 'me\(xobs, sdx\)'"),
])
def test_me_data_are_matched_by_content(me, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        _corrupted(me, mutate)
