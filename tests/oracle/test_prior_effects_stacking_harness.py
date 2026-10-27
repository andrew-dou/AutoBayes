"""E05: the post gate's prior-draw, Savage-Dickey, interaction-effect and stacking checks (``gates.check_prior_draws``,
``gates.check_hypothesis`` with the fixture's prior draws, ``gates.check_conditional_effects`` of interaction effects,
``gates.check_loo_model_weights``) pass on the frozen cases gauss_hypothesis_prior_effects and
bernoulli_hypothesis_prior_effects, and a corrupted entry of each fails: a prior draw, a prior-draw column's name or count,
an evidence ratio or posterior probability (a number made NaN and NaN made a number), a moderator's value, an effect2__
label, the effect2__ column; and the stacking case gauss_loo_compare_x_stacking passes every gate, its stacking weights on
its 50-digit reference (P6, D126: brms's are 1.044e-8 off the optimum), fixture weights moved to the exact optimum pass with
nothing on the reference, and a corrupted model order, method or other model fails (tests/oracle/test_harness.py checks the
three rules on the weights)."""
import copy

import numpy as np
import pytest

import autobayes as ab
from autobayes import post_processing
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch
from tests.oracle.two_reference import ReferenceFallback

CASES = ("gauss_hypothesis_prior_effects", "bernoulli_hypothesis_prior_effects")
STACKING = "gauss_loo_compare_x_stacking"


@pytest.fixture(scope="module", params=CASES)
def bundle(request):
    if fx.load_case(request.param).get("gate"):
        pytest.skip(f"{request.param} is still gated")
    return gates.Bundle(request.param)


def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _bump(values, i, rel=1e-7):
    """Move entry ``i`` of a fixture array by ``rel`` on the gates' scale ``max(1, |v|)``: a hundred times their tolerance."""
    v = float(values[i])
    values[i] = repr(v + rel * max(1.0, abs(v)))


def _fails(bundle, mutate, match):
    with pytest.raises(OracleMismatch, match=match):
        gates.check_post(_corrupted(bundle, mutate))


def _point(bundle):
    """The rows of brms's table that are point hypotheses (their evidence ratios come from the prior draws)."""
    return [i for i, h in enumerate(bundle.case["brms"]["hypothesis"]) if "=" in h]


def test_post_gate_passes_with_prior_draws_and_interaction_effects(bundle):
    assert set(bundle.post) == {"hypothesis", "conditional_effects", "prior_draws"}
    assert list(bundle.fixture.raw["post"]["conditional_effects"]) == ["x:z", "x:w"]
    gates.check_post(bundle)


def test_corrupted_prior_draws_fail(bundle):
    colnames = bundle.fixture.raw["post"]["prior_draws"]["colnames"]
    b = colnames.index("b")
    for k in (0, 3, 7):                                                   # (8 draws, columns), column-major: the b column
        def mutate(raw, k=k):
            _bump(raw["post"]["prior_draws"]["values"], 8 * b + k, rel=1e-3)
        _fails(bundle, mutate, "hypothesis Evid.Ratio differs from brms")
    def renamed(raw):
        raw["post"]["prior_draws"]["colnames"][b] = "b_x"
    _fails(bundle, renamed, "hypothesis Evid.Ratio: non-finite entries differ")   # w and x:zq (or zq, x:w) have none
    wrong = _corrupted(bundle, renamed)
    with pytest.raises(OracleMismatch, match="prior_draws columns"):
        gates.check_prior_draws(wrong, wrong.fixture.raw["post"]["prior_draws"])
    def other_column(raw):                                                # the Intercept's prior draws play no part ...
        _bump(raw["post"]["prior_draws"]["values"], 0, rel=1e-3)
    gates.check_post(_corrupted(bundle, other_column))                     # ... as brms's hypothesis() does not use them
    def fewer(raw):
        pd_ = raw["post"]["prior_draws"]
        n = len(pd_["colnames"])
        pd_["values"] = [v for i, v in enumerate(pd_["values"]) if i % 8 != 7]
        pd_["dims"] = [7, n]
    with pytest.raises(OracleMismatch, match="hypothesis Evid.Ratio differs from brms"):
        gates.check_post(_corrupted(bundle, fewer))


def test_corrupted_evidence_ratios_fail(bundle):
    for i in _point(bundle):
        for column in ("Evid.Ratio", "Post.Prob"):
            def bumped(raw, i=i, column=column):
                _bump(raw["post"]["hypothesis"]["table"]["values"][column]["values"], i)
            _fails(bundle, bumped, f"hypothesis {column} differs from brms")
            def emptied(raw, i=i, column=column):
                raw["post"]["hypothesis"]["table"]["values"][column]["values"][i] = "NaN"
            _fails(bundle, emptied, f"hypothesis {column}: non-finite entries differ")


def test_without_the_prior_draws_the_point_hypotheses_have_no_ratio(bundle):
    """The same table without the case's prior draws is D06's: NaN where brms's has the Savage-Dickey ratio."""
    res = ab.hypothesis(bundle.model, gates.fixed_parameter_draws(bundle), bundle.case["brms"]["hypothesis"])
    assert np.isnan(res.table["Evid.Ratio"].to_numpy()[_point(bundle)]).all()


def test_corrupted_interaction_effects_fail(bundle):
    for effect, column, i in (("x:w", "w", 0), ("x:w", "w", 4), ("x:z", "estimate__", 151), ("x:w", "x", 299)):
        def mutate(raw, effect=effect, column=column, i=i):
            _bump(raw["post"]["conditional_effects"][effect]["values"][column]["values"], i)
        _fails(bundle, mutate, f"conditional_effects {effect} {column} differs from brms")
    for effect, column, i, value in (("x:w", "effect2__", 2, "1.65"), ("x:z", "effect2__", 1, "r"), ("x:z", "z", 5, "p")):
        def relabel(raw, effect=effect, column=column, i=i, value=value):
            raw["post"]["conditional_effects"][effect]["values"][column][i] = value
        _fails(bundle, relabel, f"conditional_effects {effect} {column} differs from brms at rows \\[{i}\\]")
    def dropped(raw):
        raw["post"]["conditional_effects"]["x:z"]["columns"].remove("effect2__")
    _fails(bundle, dropped, "conditional_effects x:z: columns")
    def swapped(raw):                                                     # the moderator varying slowest
        v = raw["post"]["conditional_effects"]["x:z"]["values"]["z"]
        v[0], v[1] = v[1], v[0]
    _fails(bundle, swapped, "conditional_effects x:z z differs from brms at rows \\[0, 1\\]")


# -- stacking ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stacking():
    return gates.Bundle(STACKING)


def test_stacking_case_passes_every_gate_with_its_weights_on_the_reference(stacking):
    """The case opts in to two-reference acceptance of its stacking weights (P6, D126): every gate passes, the post gate's
    weights on the 50-digit optimum where brms's are 1.044e-8 off it (tests/oracle/test_harness.py checks the three rules)."""
    assert stacking.case["gate"] == {"expect": "pass", "reference": "gauss_loo_compare_x_stacking.json", "divergence": "D126",
                                     "fallback": ["weights"]}
    assert ab.loo_model_weights is post_processing.loo_model_weights
    for check in gates.ALL:
        if check is gates.check_jax:
            pytest.importorskip("jax")
        check(stacking)
    with pytest.warns(ReferenceFallback, match="post weights: 8 of 8 entries passed on the reference"):
        gates.check_post(stacking)


def test_stacking_gate_compares_the_weights(stacking):
    """With the fixture's weights moved to the exact optimum nothing falls back; one of brms's weights moved 1e-8 off it is
    off the reference by more than the tolerance where AutoBayes agrees with the reference, so that weight and its logarithm
    pass on the reference in both functions (4 of 8); the models' order, the method and the other model fail."""
    exact = None
    def to_optimum(raw):
        nonlocal exact
        lpd = fx.decode(raw["post"]["loo_compare"]["elpd_loo"])
        exact = post_processing.stacking_weights(lpd)
        raw["post"]["loo_model_weights"]["weights"]["values"] = [repr(float(w)) for w in exact]
    at_optimum = _corrupted(stacking, to_optimum)
    with pytest.warns(ReferenceFallback, match="post weights: 0 of 8 entries"):
        gates.check_post(at_optimum)
    assert exact is not None and abs(exact[0] - 0.5619733092607880) < 1e-15
    def moved(raw):
        _bump(raw["post"]["loo_model_weights"]["weights"]["values"], 1, rel=1e-8)
    with pytest.warns(ReferenceFallback, match="post weights: 4 of 8 entries"):   # the second model's weight and its log, twice
        gates.check_post(_corrupted(at_optimum, moved))
    def reordered(raw):
        raw["post"]["loo_model_weights"]["models"].reverse()
    with pytest.raises(OracleMismatch, match="loo_model_weights: the fixture weighs"):
        gates.check_post(_corrupted(at_optimum, reordered))
    def method(raw):
        raw["post"]["loo_model_weights"]["method"] = "pseudobma"
    with pytest.raises(OracleMismatch, match="by 'pseudobma'"):
        gates.check_post(_corrupted(at_optimum, method))
    def other(raw):
        _bump(raw["post"]["loo_compare"]["other_lp"]["values"], 0, rel=1e-6)
    with pytest.raises(OracleMismatch, match="other_lp"):
        gates.check_post(_corrupted(at_optimum, other))
