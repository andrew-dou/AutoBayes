"""Every frozen case with an `autobayes` section passes the oracle gates G1-G6 (deterministic; no sampling)."""
import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates

def _cases():
    """Every frozen case with an AutoBayes call. A case whose feature has not landed carries
    ``"gate": {"expect": "fail", "reason": ...}`` and runs as a strict expected failure: the fixture is frozen
    first, the tests go red for the right reason, and the implementing branch removes the field."""
    out = []
    for cid in fx.case_ids():
        case = fx.load_case(cid)
        if not fx.has_fixture(cid) or "autobayes" not in case:
            continue
        gate = case.get("gate") or {}
        marks = [pytest.mark.xfail(strict=True, reason=gate["reason"])] if gate.get("expect") == "fail" else []
        out.append(pytest.param(cid, id=cid, marks=marks))
    return out


@pytest.fixture(scope="module", params=_cases())
def bundle(request):
    return gates.Bundle(request.param)


def test_g1_design_equals_standata(bundle):
    gates.check_design(bundle)


def test_g2_log_density_equals_log_prob(bundle):
    gates.check_logp(bundle)


def test_g2_gradient_equals_log_prob_gradient(bundle):
    gates.check_gradient(bundle)


def test_g3_derived_quantities(bundle):
    gates.check_derived(bundle)


def test_g4_predictions_equal_linpred_and_epred(bundle):
    gates.check_predictions(bundle)


def test_g5_jax_density_equals_pytensor(bundle):
    pytest.importorskip("jax")
    gates.check_jax(bundle)


def test_g6_default_prior_table(bundle):
    gates.check_default_prior(bundle)
