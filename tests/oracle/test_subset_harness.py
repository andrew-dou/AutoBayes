"""The oracle's subset() paths (F06c) bite: a fixture of gauss_mv_subset corrupted in what subset() adds (each response's rows,
``N_<resp>`` and ``Y_<resp>``, each response's own predictions on its rows, the linear predictor G3 reads from them) fails the
gate that reads it, and the joint predictions brms refuses must be refused by the model."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch

CASE = "gauss_mv_subset"


@pytest.fixture(scope="module")
def bundle():
    if not fx.has_fixture(CASE) or fx.load_case(CASE).get("gate"):
        pytest.skip(f"{CASE} not frozen or still gated")
    return gates.Bundle(CASE)


def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _add(path, delta):
    def mutate(raw):
        values, k = path(raw)
        values[k] += delta
    return mutate


def _resp(raw, block, resp, kind):
    return raw["predictions"][block]["resp"][resp][kind]


@pytest.mark.parametrize("gate, mutate, message", [
    (gates.check_design, _add(lambda raw: (raw["standata"]["N_y2"]["values"], 0), 1), r"response 'y2': brms N_y2 = 19 rows"),
    (gates.check_design, _add(lambda raw: (raw["standata"]["Y_y"]["values"], 4), 1e-9), "response 'y': Y_y differs from the response on its rows"),
    (gates.check_derived, _add(lambda raw: (_resp(raw, "fitted", "y2", "linpred")["values"], 8 * 3 + 5), 1e-8),
     "point 5: the graph's linear predictor of 'y2' differs from brms"),
    (gates.check_predictions, _add(lambda raw: (_resp(raw, "newdata", "y", "epred_nogroup")["values"], 8 + 2), 1e-8),
     r"newdata_nogroup: the response 'y''s 'mu' differs from brms posterior_epred\(resp = 'y'\)"),
    (gates.check_predictions, _add(lambda raw: (_resp(raw, "fitted", "y", "linpred")["values"], 3), 1e-8),
     r"fitted: the response 'y''s 'eta' differs from brms posterior_linpred\(resp = 'y'\)"),
])
def test_a_corrupted_subset_fixture_fails(bundle, gate, mutate, message):
    with pytest.raises(OracleMismatch, match=message):
        gate(_corrupted(bundle, mutate))


def test_a_response_on_other_rows_fails(bundle):
    def drop_a_row(raw):
        entry = _resp(raw, "fitted", "y2", "linpred")
        values = np.asarray(entry["values"]).reshape(entry["dims"], order="F")[:, 1:]
        entry.update(dims=list(values.shape), values=values.ravel(order="F").tolist())
    with pytest.raises(OracleMismatch, match=r"fitted: the response 'y2' is predicted on 18 rows; brms's posterior_linpred\(resp = 'y2'\) on 17"):
        gates.check_predictions(_corrupted(bundle, drop_a_row))
    with pytest.raises(OracleMismatch, match="fixture lacks the newdata linpred_nogroup of the response 'y'"):
        gates.check_predictions(_corrupted(bundle, lambda raw: raw["predictions"]["newdata"]["resp"]["y"].pop("linpred_nogroup")))
    with pytest.raises(OracleMismatch, match="the fixture has neither the joint linear predictor nor each response's"):
        gates.check_derived(_corrupted(bundle, lambda raw: raw["predictions"]["fitted"]["resp"]["y"].pop("linpred")))


def test_joint_predictions_must_be_refused(bundle):
    """A model that returned the joint predictions brms refuses, or refused them for another reason, fails G4."""
    m = bundle.model
    lenient = SimpleNamespace(**{k: getattr(m, k) for k in ("responses", "submodels", "names", "family_spec", "spec", "pymc_model",
                                                             "design", "rescor", "variable", "value_name")})
    lenient.predict = lambda idata, **kwargs: m.predict(idata, **({"resp": "y"} | kwargs))
    b = SimpleNamespace(**vars(bundle))
    b.model = lenient
    with pytest.raises(OracleMismatch, match=r"fitted: brms refuses the joint posterior_linpred .*; the AutoBayes model returns it"):
        gates.check_response_predictions(b)

    def other(idata, **kwargs):
        raise ValueError("something else")
    lenient.predict = other
    with pytest.raises(OracleMismatch, match="fitted: the joint posterior_linpred is refused for another reason than brms's: something else"):
        gates.check_response_predictions(b)
