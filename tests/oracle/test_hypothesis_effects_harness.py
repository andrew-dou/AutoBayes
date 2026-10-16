"""D06: the post gate's hypothesis and conditional_effects checks (``gates.check_hypothesis``, ``gates.check_conditional_effects``)
pass on the frozen cases gauss_hypothesis_effects and bernoulli_hypothesis_effects, and a corrupted entry of each fails: any
number of brms's hypothesis table (brms's NA among them), a label, a star, a column name, a draw of a hypothesis; any number of
a conditional_effects table (the summaries, the grid, a value held, the response), a level, a grouping factor's NA, cond__, the
column order and the order of the effects; and a case whose arguments are not the ones brms was given."""
import copy

import pytest

from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch

CASES = ("gauss_hypothesis_effects", "bernoulli_hypothesis_effects")


def _bundle(case_id):
    if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
        pytest.skip(f"{case_id} is not frozen or still gated")
    return gates.Bundle(case_id)


@pytest.fixture(scope="module", params=CASES)
def bundle(request):
    return _bundle(request.param)


def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def _bump(values, i, rel=1e-7):
    """Move entry ``i`` of a fixture array by ``rel`` on the gates' scale ``max(1, |v|)``: a hundred times their tolerance."""
    v = float(values[i])
    values[i] = repr(v + rel * max(1.0, abs(v))) if isinstance(values[i], str) else v + rel * max(1.0, abs(v))


def _fails(bundle, mutate, match):
    with pytest.raises(OracleMismatch, match=match):
        gates.check_post(_corrupted(bundle, mutate))


def test_post_gate_passes_with_hypothesis_and_conditional_effects(bundle):
    assert set(bundle.post) == {"hypothesis", "conditional_effects"}
    gates.check_post(bundle)


@pytest.mark.parametrize("column", ["Estimate", "Est.Error", "CI.Lower", "CI.Upper"])
def test_corrupted_hypothesis_numbers_fail(bundle, column):
    for i in range(3):
        def mutate(raw, i=i):
            _bump(raw["post"]["hypothesis"]["table"]["values"][column]["values"], i)
        _fails(bundle, mutate, f"hypothesis {column} differs from brms")


def test_corrupted_evidence_ratio_and_posterior_probability_fail(bundle):
    """A one-sided hypothesis's Evid.Ratio and Post.Prob are numbers; a point hypothesis's are brms's NA (stored NaN), which must
    stay NaN, and a number where brms has one must not become NaN."""
    values = bundle.fixture.raw["post"]["hypothesis"]["table"]["values"]
    point = [i for i, v in enumerate(values["Evid.Ratio"]["values"]) if v == "NaN"]
    one_sided = [i for i, v in enumerate(values["Evid.Ratio"]["values"]) if v != "NaN"]
    assert point and one_sided
    for column in ("Evid.Ratio", "Post.Prob"):
        def bumped(raw, column=column):
            _bump(raw["post"]["hypothesis"]["table"]["values"][column]["values"], one_sided[0])
        _fails(bundle, bumped, f"hypothesis {column} differs from brms")
        def filled(raw, column=column):
            raw["post"]["hypothesis"]["table"]["values"][column]["values"][point[0]] = "1"
        _fails(bundle, filled, f"hypothesis {column}: non-finite entries differ")
        def emptied(raw, column=column):
            raw["post"]["hypothesis"]["table"]["values"][column]["values"][one_sided[0]] = "NaN"
        _fails(bundle, emptied, f"hypothesis {column}: non-finite entries differ")


def test_corrupted_hypothesis_text_fails(bundle):
    def label(raw):
        raw["post"]["hypothesis"]["table"]["values"]["Hypothesis"][0] = "(x) < 0"
    _fails(bundle, label, "hypothesis Hypothesis differs from brms at rows \\[0\\]")
    def star(raw):
        raw["post"]["hypothesis"]["table"]["values"]["Star"][2] = "*"
    _fails(bundle, star, "hypothesis Star differs from brms at rows \\[2\\]")
    def renamed(raw):
        raw["post"]["hypothesis"]["table"]["columns"][5] = "Evid"
    _fails(bundle, renamed, "hypothesis columns")


def test_corrupted_hypothesis_draws_fail(bundle):
    for i in (0, 8 + 3, 2 * 8 + 7):                                   # (8 draws, 3 hypotheses), column-major
        def mutate(raw, i=i):
            _bump(raw["post"]["hypothesis"]["samples"]["values"], i)
        _fails(bundle, mutate, "hypothesis samples differs from brms")
    def renamed(raw):
        raw["post"]["hypothesis"]["samples"]["colnames"][1] = "H9"
    _fails(bundle, renamed, "hypothesis samples columns")


@pytest.mark.parametrize("column", ["estimate__", "se__", "lower__", "upper__"])
def test_corrupted_conditional_effects_summaries_fail(bundle, column):
    for effect, i in (("x", 0), ("x", 57), ("x", 99), ("z", 1)):
        def mutate(raw, effect=effect, i=i):
            _bump(raw["post"]["conditional_effects"][effect]["values"][column]["values"], i)
        _fails(bundle, mutate, f"conditional_effects {effect} {column} differs from brms")


def test_corrupted_conditional_effects_grid_fails(bundle):
    response = bundle.model.spec.response
    for effect, column, i in (("x", "x", 31), ("x", "effect1__", 64), ("x", response, 5), ("z", "x", 0), ("z", response, 2)):
        def mutate(raw, effect=effect, column=column, i=i):
            _bump(raw["post"]["conditional_effects"][effect]["values"][column]["values"], i)
        _fails(bundle, mutate, f"conditional_effects {effect} {column} differs from brms")
    for effect, column, i, value in (("z", "z", 1, "r"), ("z", "effect1__", 2, "p"), ("x", "z", 10, "q"), ("x", "cond__", 0, "2")):
        def relabel(raw, effect=effect, column=column, i=i, value=value):
            raw["post"]["conditional_effects"][effect]["values"][column][i] = value
        _fails(bundle, relabel, f"conditional_effects {effect} {column} differs from brms at rows \\[{i}\\]")


def test_corrupted_grouping_factor_and_columns_fail():
    """The Gaussian case's grouping factor g is brms's NA on every grid row (the group-level effects are left out)."""
    bundle = _bundle("gauss_hypothesis_effects")
    def grouped(raw):
        raw["post"]["conditional_effects"]["x"]["values"]["g"][3] = "a"
    _fails(bundle, grouped, "conditional_effects x g differs from brms at rows \\[3\\]")
    def reordered(raw):
        cols = raw["post"]["conditional_effects"]["z"]["columns"]
        cols[2], cols[3] = cols[3], cols[2]
    _fails(bundle, reordered, "conditional_effects z: columns")
    def effects(raw):
        ce = raw["post"]["conditional_effects"]
        raw["post"]["conditional_effects"] = {"z": ce["z"], "x": ce["x"]}
    _fails(bundle, effects, "conditional_effects: the case asks for \\['x', 'z'\\]")
    def shorter(raw):
        for column, v in raw["post"]["conditional_effects"]["x"]["values"].items():
            if isinstance(v, dict):
                v["dims"], v["values"] = [99], v["values"][:99]
            else:
                del v[99]
    _fails(bundle, shorter, "conditional_effects x x: AutoBayes has shape \\(100,\\), brms \\(99,\\)")


def test_the_case_must_ask_what_brms_was_given(bundle):
    other = bundle.with_fixture(bundle.fixture)
    other.case = copy.deepcopy(bundle.case)
    other.case["autobayes"]["post_args"]["hypothesis"] = other.case["autobayes"]["post_args"]["hypothesis"][:2]
    with pytest.raises(OracleMismatch, match="hypothesis: the case's post_args"):
        gates.check_post(other)
    other.case = copy.deepcopy(bundle.case)
    other.case["autobayes"]["post_args"]["conditional_effects"] = ["x"]
    with pytest.raises(OracleMismatch, match="conditional_effects: the case's post_args \\['x'\\]"):
        gates.check_post(other)
    def no_hypothesis(raw):
        del raw["post"]["hypothesis"]
    _fails(bundle, no_hypothesis, "the case lists hypothesis but the fixture's post block has")
