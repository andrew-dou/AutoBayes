"""Every frozen case with an `autobayes` section passes the oracle gates G1-G6 (deterministic; no sampling)."""
import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates

# the names a case's gate field may list under "gates" (below): one per test of this module
GATE_NAMES = ("g1", "g2_logp", "g2_gradient", "g3", "g4", "g5", "g6", "post")


def _cases():
    """Every frozen case with an AutoBayes call. A case whose feature has not landed carries
    ``"gate": {"expect": "fail", "reason": ...}`` and runs as a strict expected failure: the fixture is frozen
    first, the tests go red for the right reason, and the implementing branch removes the field. A case that opts in to
    two-reference acceptance (P6) carries ``"gate": {"expect": "pass", "reference": ..., "fallback": [...]}`` and runs
    as every other case; the kinds it lists (G2's gradient, G4's expected values) follow ``two_reference``'s rule.

    A gate field with ``"expect": "fail"`` may also list the gates that fail, ``"gates": ["g2_logp", "g2_gradient"]``
    (names in ``GATE_NAMES``): then only those run as strict expected failures, each passing only by raising the oracle's
    ``OracleMismatch``, and every other gate of the case must pass. It is for a case whose feature has landed but on which
    brms's own arithmetic fails a gate no reference may stand in for (a log density), so that the case stays frozen and
    checked everywhere else (E06: ``cumulative_cloglog``, whose link ``cumulative_cloglog_narrow`` accepts)."""
    out = []
    for cid in fx.case_ids():
        case = fx.load_case(cid)
        if not fx.has_fixture(cid) or "autobayes" not in case:
            continue
        gate = case.get("gate") or {}
        if gate.get("expect") == "fail" and "gates" in gate:
            unknown = sorted(set(gate["gates"]) - set(GATE_NAMES))
            if unknown or not gate["gates"]:
                raise ValueError(f"{cid}: the gate field lists {gate['gates']}; the gates are {list(GATE_NAMES)}")
            marks = []
        else:
            marks = [pytest.mark.xfail(strict=True, reason=gate["reason"])] if gate.get("expect") == "fail" else []
        out.append(pytest.param(cid, id=cid, marks=marks))
    return out


@pytest.fixture(scope="module", params=_cases())
def bundle(request):
    return gates.Bundle(request.param)


def _gate(bundle, name, check):
    """Run ``check`` on the case; when its gate field lists ``name`` under ``"gates"``, as a strict expected failure: an
    ``OracleMismatch`` is the expected failure, a pass is an error, and any other exception propagates."""
    gate = bundle.case.get("gate") or {}
    if gate.get("expect") == "fail" and name in gate.get("gates", ()):
        try:
            check(bundle)
        except gates.OracleMismatch as e:
            pytest.xfail(f"{gate['reason']} ({e})")
        pytest.fail(f"{bundle.case['id']}: gate {name} was expected to fail ({gate['reason']}) but passed: remove it from the "
                    "gate field's 'gates'")
    check(bundle)


def test_g1_design_equals_standata(bundle):
    _gate(bundle, "g1", gates.check_design)


def test_g2_log_density_equals_log_prob(bundle):
    _gate(bundle, "g2_logp", gates.check_logp)


def test_g2_gradient_equals_log_prob_gradient(bundle):
    _gate(bundle, "g2_gradient", gates.check_gradient)


def test_g3_derived_quantities(bundle):
    _gate(bundle, "g3", gates.check_derived)


def test_g4_predictions_equal_linpred_and_epred(bundle):
    _gate(bundle, "g4", gates.check_predictions)


def test_g5_jax_density_equals_pytensor(bundle):
    pytest.importorskip("jax")
    _gate(bundle, "g5", gates.check_jax)


def test_g6_default_prior_table(bundle):
    _gate(bundle, "g6", gates.check_default_prior)


def test_c07_post_processing_equals_log_lik_bayes_R2_and_loo(bundle):
    """C07: a case that lists post-processing functions (``"post"`` in its autobayes section) has them equal brms's
    log_lik(), bayes_R2() and loo() in the fixture's ``post`` block; the other cases have nothing to compare."""
    if not bundle.post:
        pytest.skip("the case lists no post-processing function")
    _gate(bundle, "post", gates.check_post)
