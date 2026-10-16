"""D05: the post gate's waic, loo_compare and model_weights checks (``gates.check_waic``, ``gates.check_loo_compare``) pass on
the frozen cases gauss_loo_compare_x and gauss_loo_compare_xw, and a corrupted entry of each fails: a pointwise or
estimated WAIC, a comparison table's entry, row order or column name, either model's pointwise elpd, a weight (the tiny one
in its exponent), the other model's log density, and a case that names another model than the fixture compared."""
import copy

import numpy as np
import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch


def _bundle(case_id):
    if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
        pytest.skip(f"{case_id} is not frozen or still gated")
    return gates.Bundle(case_id)


@pytest.fixture(scope="module")
def x_bundle():
    return _bundle("gauss_loo_compare_x")


@pytest.fixture(scope="module")
def xw_bundle():
    return _bundle("gauss_loo_compare_xw")


def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _bump(values, i, rel=1e-7):
    """Move entry ``i`` of a fixture array by ``rel`` on the gates' scale ``max(1, |v|)``: a hundred times their tolerance."""
    v = float(values[i])
    values[i] = v + rel * max(1.0, abs(v))


def test_post_gate_passes_with_waic_and_loo_compare(x_bundle, xw_bundle):
    assert set(x_bundle.post) == {"log_lik", "loo", "waic", "loo_compare"} and set(xw_bundle.post) == {"log_lik", "loo", "waic"}
    for bundle in (x_bundle, xw_bundle):
        gates.check_post(bundle)


@pytest.mark.parametrize("case_id", ["gauss_loo_compare_x", "gauss_loo_compare_xw"])
def test_corrupted_waic_fails(case_id):
    bundle = _bundle(case_id)
    for i, column in ((3, "elpd_waic"), (36 + 5, "p_waic"), (2 * 36 + 7, "waic")):     # column-major (36, 3)
        def pointwise(raw, i=i):
            _bump(raw["post"]["waic"]["pointwise"]["values"], i)
        with pytest.raises(OracleMismatch, match=f"waic pointwise {column} differs from brms"):
            gates.check_post(_corrupted(bundle, pointwise))
    def se(raw):
        _bump(raw["post"]["waic"]["estimates"]["values"], 3)                   # SE of elpd_waic (rows, then SE)
    with pytest.raises(OracleMismatch, match="waic estimates differs from brms"):
        gates.check_post(_corrupted(bundle, se))
    def renamed(raw):
        raw["post"]["waic"]["pointwise"]["colnames"][1] = "p"
    with pytest.raises(OracleMismatch, match="waic pointwise columns"):
        gates.check_post(_corrupted(bundle, renamed))


def test_corrupted_comparison_table_fails(x_bundle):
    for crit in ("loo", "waic"):
        def elpd_diff(raw, crit=crit):
            _bump(raw["post"]["loo_compare"][crit]["table"]["values"], 1)         # the second model's elpd_diff
        with pytest.raises(OracleMismatch, match=f"loo_compare by {crit} differs from brms"):
            gates.check_post(_corrupted(x_bundle, elpd_diff))
        def se_diff(raw, crit=crit):
            _bump(raw["post"]["loo_compare"][crit]["table"]["values"], 3)         # the second model's se_diff
        with pytest.raises(OracleMismatch, match=f"loo_compare by {crit} differs from brms"):
            gates.check_post(_corrupted(x_bundle, se_diff))
        def rows(raw, crit=crit):
            raw["post"]["loo_compare"][crit]["rows"].reverse()
        with pytest.raises(OracleMismatch, match=f"loo_compare by {crit}: rows"):
            gates.check_post(_corrupted(x_bundle, rows))
        def columns(raw, crit=crit):
            raw["post"]["loo_compare"][crit]["table"]["colnames"][6] = "ic"
        with pytest.raises(OracleMismatch, match=f"loo_compare by {crit}: columns"):
            gates.check_post(_corrupted(x_bundle, columns))


def test_corrupted_pointwise_elpd_of_either_model_fails(x_bundle):
    for crit in ("loo", "waic"):
        for i in (4, 36 + 4):                                                             # this model's row 5, the other's
            def mutate(raw, crit=crit, i=i):
                _bump(raw["post"]["loo_compare"][f"elpd_{crit}"]["values"], i)
            with pytest.raises(OracleMismatch, match=f"loo_compare pointwise elpd_{crit} differs from brms"):
                gates.check_post(_corrupted(x_bundle, mutate))


def test_corrupted_weights_fail_the_tiny_one_in_its_exponent(x_bundle):
    """A weight of 3.7e-115 made ten times larger passes the ``max(1, |a|, |b|)`` comparison but not that of the logarithms."""
    for kind in ("loo", "pseudobma"):
        def tiny(raw, kind=kind):
            raw["post"]["loo_compare"][f"weights_{kind}"]["values"][1] *= 10.0
        with pytest.raises(OracleMismatch, match=f"model_weights {kind} \\(log\\) differs from brms"):
            gates.check_post(_corrupted(x_bundle, tiny))
        def large(raw, kind=kind):
            raw["post"]["loo_compare"][f"weights_{kind}"]["values"][0] -= 1e-8
        with pytest.raises(OracleMismatch, match=f"model_weights {kind} differs from brms"):
            gates.check_post(_corrupted(x_bundle, large))
    def models(raw):
        raw["post"]["loo_compare"]["weights_models"].reverse()
    with pytest.raises(OracleMismatch, match="model_weights loo: models"):
        gates.check_post(_corrupted(x_bundle, models))


def test_the_other_model_must_be_the_one_brms_compared(x_bundle):
    def other_lp(raw):
        _bump(raw["post"]["loo_compare"]["other_lp"]["values"], 2)
    with pytest.raises(OracleMismatch, match="gauss_loo_compare_xw's log density at its points differs"):
        gates.check_post(_corrupted(x_bundle, other_lp))
    def swapped(raw):
        raw["post"]["loo_compare"]["models"].reverse()
    with pytest.raises(OracleMismatch, match="loo_compare: the fixture compares"):
        gates.check_post(_corrupted(x_bundle, swapped))
    elsewhere = x_bundle.with_fixture(x_bundle.fixture)
    elsewhere.case = copy.deepcopy(x_bundle.case)
    elsewhere.case["autobayes"]["compare_with"] = "gauss_varint_post"
    with pytest.raises(OracleMismatch, match="the case compares \\['gauss_loo_compare_x', 'gauss_varint_post'\\]"):
        gates.check_post(elsewhere)


def test_the_fixture_must_hold_what_the_case_lists(x_bundle):
    def no_waic(raw):
        del raw["post"]["waic"]
    with pytest.raises(OracleMismatch, match="the case lists waic but the fixture's post block has"):
        gates.check_post(_corrupted(x_bundle, no_waic))
    def no_compare(raw):
        del raw["post"]["loo_compare"]
    with pytest.raises(OracleMismatch, match="the case lists loo_compare but the fixture's post block has"):
        gates.check_post(_corrupted(x_bundle, no_compare))
    assert np.isfinite(fx.decode(x_bundle.fixture.raw["post"]["loo_compare"]["weights_loo"])).all()
