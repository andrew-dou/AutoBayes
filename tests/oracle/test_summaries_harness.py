"""F07: the post gate's comparison of brms's summaries (``gates.check_summary``, on ``gates.summary_draws_fit``) passes on the
frozen cases gauss_varint_summaries, bernoulli_slope_correlated_summaries and gauss_distributional_sigma_summaries, and a
corrupted record fails: an entry of each summary (a fixef estimate, posterior_summary's lp__, a ranef or coef quantile, a
VarCorr covariance, a residual, an Rhat and a bulk ESS of summary()), a label (a row name, a coefficient, a part), a count,
the case's arguments, and a fixture without its summary draws."""
import copy

import numpy as np
import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch

CASES = ["gauss_varint_summaries", "bernoulli_slope_correlated_summaries", "gauss_distributional_sigma_summaries"]


def _bundle(case_id):
    if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
        pytest.skip(f"{case_id} is not frozen or still gated")
    return gates.Bundle(case_id)


@pytest.mark.parametrize("case_id", CASES)
def test_post_gate_passes_with_the_summaries(case_id):
    bundle = _bundle(case_id)
    assert list(bundle.post) == list(gates.SUMMARIES)
    gates.check_post(bundle)


@pytest.fixture(scope="module")
def correlated():
    bundle = _bundle("bernoulli_slope_correlated_summaries")
    return bundle, gates.summary_draws_fit(bundle)


def _check(bundle, idata, name, mutate=None, args=None):
    ref = copy.deepcopy(bundle.fixture.raw["post"][name])
    if mutate is not None:
        mutate(ref)
    if args is not None:
        bundle = copy.copy(bundle)
        bundle.case = {**bundle.case, "autobayes": {**bundle.case["autobayes"], "post_args": args}}
    gates.check_summary(bundle, name, bundle.post[name], idata, ref)


def _bump(values, i, rel=1e-7):
    """Move entry ``i`` of a fixture array by ``rel`` on the gates' scale ``max(1, |v|)``: a hundred times their tolerance."""
    v = float(values[i])
    values[i] = v + rel * max(1.0, abs(v))


def _bump_at(table, row, column):
    """``_bump`` of entry (row, column) of a labelled table (column-major)."""
    _bump(table["values"], column * int(table["dims"][0]) + row)


@pytest.mark.parametrize("name, mutate, match", [
    ("fixef", lambda r: _bump_at(r, 1, 0), "fixef differs from brms"),
    ("fixef", lambda r: r["dimnames"][0].reverse(), "fixef rows"),
    ("posterior_summary", lambda r: _bump_at(r, int(r["dims"][0]) - 1, 0), "posterior_summary differs from brms"),
    ("posterior_summary", lambda r: r["dimnames"][0].__setitem__(4, "cor_g__x__Intercept"), "posterior_summary rows"),
    ("ranef", lambda r: _bump(r["g"]["values"], 6 * 4 + 6 * 2 + 3), "ranef g differs from brms"),       # [level 3, Q2.5, x]
    ("ranef", lambda r: r["g"]["dimnames"][2].reverse(), "ranef g columns"),
    ("coef", lambda r: _bump(r["g"]["values"], 6 * 3 + 5), "coef g differs from brms"),                  # [level 5, Q97.5, Intercept]
    ("VarCorr", lambda r: _bump(r["g"]["cov"]["values"], 2 * 4 + 1), "VarCorr g cov differs from brms"),  # [x, Estimate, x]
    ("VarCorr", lambda r: r["g"].pop("cov"), "VarCorr g parts"),
    ("residuals", lambda r: _bump_at(r, 17, 1), "residuals differs from brms"),
    ("summary", lambda r: _bump_at(r["random"]["g"], 2, 4), "summary random g differs from brms"),     # cor's Rhat
    ("summary", lambda r: _bump_at(r["fixed"], 0, 5), "summary fixed differs from brms"),             # Intercept's Bulk_ESS
    ("summary", lambda r: r["random"]["g"]["dimnames"][0].__setitem__(2, "cor(x,Intercept)"), "summary random g rows"),
    ("summary", lambda r: r.__setitem__("total_ndraws", 199), "summary total_ndraws"),
    ("summary", lambda r: r.__setitem__("ngrps", {"g": 5}), "summary ngrps"),
])
def test_a_corrupted_summary_fails(correlated, name, mutate, match):
    bundle, idata = correlated
    _check(bundle, idata, name)                                       # the record as frozen passes
    with pytest.raises(OracleMismatch, match=match):
        _check(bundle, idata, name, mutate)


def test_other_arguments_than_the_fixture_s_fail(correlated):
    bundle, idata = correlated
    with pytest.raises(OracleMismatch, match="residuals: the case's post_args"):
        _check(bundle, idata, "residuals", args={"residuals": {"type": "ordinary"}})
    with pytest.raises(OracleMismatch, match="fixef: the case's post_args"):
        _check(bundle, idata, "fixef", args={"fixef": {"robust": True}, "residuals": gates.SUMMARY_ARGS["residuals"]})


def test_a_fixture_without_summary_draws_fails(correlated):
    bundle, _ = correlated
    raw = copy.deepcopy(bundle.fixture.raw)
    raw["post"].pop("summary_draws")
    with pytest.raises(OracleMismatch, match="no summary draws"):
        gates.summary_draws_fit(bundle.with_fixture(fx.Fixture(raw)))


def test_the_summary_draws_are_the_seeded_chains_in_order(correlated):
    """The posterior the gate hands the summaries is the fixture's draws chain by chain: its raw-scale intercept is brms's
    b_Intercept at every (chain, iteration)."""
    bundle, idata = correlated
    block = bundle.fixture.summary_draws
    k = block["variables"].index("b_Intercept")
    np.testing.assert_array_equal(idata.posterior["Intercept"].to_numpy(), block["draws"][:, :, k].T)
