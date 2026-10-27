"""The Tier G part 1 harness keys (G02-G04): the data generator `tier_g1` extends `tier_c` without changing it; a logical column
(``brms.logical``) and a pandas Categorical (``brms.pandas_categorical``) reach the AutoBayes side as bool and as a Categorical with
brms's level order, and brms coded exactly those values; rows with a missing value that brms drops (a ``na_columns`` column outside
``mi()``) are the rows missing from brms's data, in data order; a case whose new-row predictions brms refuses
(``newdata_prediction_refused``) records brms's message for those rows only, and fails in ``gates.Bundle`` until gates.py has the
comparison of the model's predictions on them (``check_newdata_refusal``)."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import autobayes as ab
from tests.oracle import datasets
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch

FROZEN = [cid for cid in fx.case_ids() if fx.has_fixture(cid)]


def _with(key):
    return [cid for cid in FROZEN if fx.load_case(cid)["brms"].get(key)]


def test_tier_g1_extends_tier_c_column_for_column():
    spec = {"seed": 2211, "n": 36, "newdata": {"seed": 2235, "n": 5}}
    data, nd = datasets.make({"generator": "tier_g1", **spec})
    base, base_nd = datasets.make({"generator": "tier_c", **spec})
    assert list(data.columns) == [*base.columns, "v", "lg", "off"] and list(nd.columns) == [*base_nd.columns, "v", "lg", "off"]
    pd.testing.assert_frame_equal(data[base.columns], base)
    pd.testing.assert_frame_equal(nd[base_nd.columns], base_nd)
    for frame in (data, nd):
        assert frame["lg"].dtype == bool and set(frame["lg"]) == {False, True}
        assert frame["off"].between(-0.5, 0.5).all() and not frame[["v", "off"]].isna().any().any()


@pytest.mark.parametrize("case_id", _with("logical"))
def test_logical_columns_are_bools_that_brms_coded_as_TRUE(case_id):
    case = fx.load_case(case_id)
    d, fixture = fx.fixture_dir(case_id), fx.Fixture.load(case_id)
    frame = fx.read_frame(case, d / "data.csv")
    X, names = fixture.standata["X"], fixture.colnames("X")
    for col in case["brms"]["logical"]:
        assert frame[col].dtype == bool and set(frame[col]) == {False, True}
        assert np.array_equal(X[:, names.index(f"{col}TRUE")], frame[col].to_numpy(dtype=float))
        if (d / "newdata.csv").exists():
            assert fx.read_frame(case, d / "newdata.csv")[col].dtype == bool


@pytest.mark.parametrize("case_id", _with("pandas_categorical"))
def test_pandas_categoricals_have_brms_level_order_and_brms_coded_them_against_the_first(case_id):
    case = fx.load_case(case_id)
    d, fixture = fx.fixture_dir(case_id), fx.Fixture.load(case_id)
    X, names = fixture.standata["X"], fixture.colnames("X")
    for col in case["brms"]["pandas_categorical"]:
        levels = case["brms"]["factors"][col]
        assert levels != sorted(levels)                          # the reference is not the first sorted label
        for path in (d / "data.csv", d / "newdata.csv"):
            values = fx.read_frame(case, path)[col]
            assert isinstance(values.dtype, pd.CategoricalDtype) and not values.dtype.ordered
            assert list(values.dtype.categories) == levels
        values = fx.read_frame(case, d / "data.csv")[col]
        assert f"{col}{levels[0]}" not in names
        for level in levels[1:]:
            assert np.array_equal(X[:, names.index(f"{col}{level}")], (values == level).to_numpy(dtype=float))


def test_typed_columns_refuse_values_brms_would_not_read():
    case = {"brms": {"factors": {"z": ["r", "p", "q"]}, "logical": ["lg"], "pandas_categorical": ["z"]}}
    fine = fx._typed_columns(case, pd.DataFrame({"lg": ["True", "False"], "z": ["q", "r"]}))
    assert fine["lg"].tolist() == [True, False] and list(fine["z"].cat.categories) == ["r", "p", "q"]
    with pytest.raises(ValueError, match="logical column 'lg'"):
        fx._typed_columns(case, pd.DataFrame({"lg": ["True", "1"], "z": ["q", "r"]}))
    with pytest.raises(ValueError, match="pandas_categorical column 'z'"):
        fx._typed_columns(case, pd.DataFrame({"lg": ["True", "False"], "z": ["q", "s"]}))


DROPPED = [cid for cid in _with("na_columns") if "mi(" not in str(fx.load_case(cid)["brms"].get("formula", ""))
           and "mvbf" not in fx.load_case(cid)["brms"]]


@pytest.mark.parametrize("case_id", DROPPED)
def test_brms_drops_exactly_the_rows_with_a_missing_value(case_id):
    """A case reading a column with missing values outside mi() (G02's dropped rows): brms's data are the complete rows, in data
    order, its N their number."""
    case = fx.load_case(case_id)
    frame = fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")
    sd = fx.Fixture.load(case_id).standata
    kept = frame.dropna(subset=case["brms"]["na_columns"])
    assert 0 < int(sd["N"]) == len(kept) < len(frame)
    assert np.array_equal(sd["Y"], kept["y"].to_numpy())
    X, names = sd["X"], fx.Fixture.load(case_id).colnames("X")
    for col in case["brms"]["na_columns"]:
        assert np.array_equal(X[:, names.index(col)], kept[col].to_numpy())


@pytest.mark.parametrize("case_id", _with("newdata_prediction_refused"))
def test_newdata_refusals_are_recorded_for_the_new_rows_only(case_id):
    case = fx.load_case(case_id)
    fixture = fx.Fixture.load(case_id)
    kinds = case["brms"]["newdata_prediction_refused"]
    suffixes = ("", "_nogroup") if fixture.groups else ("",)
    for kind in ("linpred", "epred"):
        for suffix in suffixes:
            assert fixture.prediction("fitted", kind + suffix) is not None
            assert fixture.prediction_refusal("fitted", kind + suffix) is None
            refused = fixture.prediction_refusal("newdata", kind + suffix)
            assert (refused is not None) == (kind in kinds) == (fixture.prediction("newdata", kind + suffix) is None)
            if refused is not None:       # brms's message; the error under it is "non-conformable arguments" (U51)
                assert "transformed numeric variables to factors" in refused


@pytest.mark.parametrize("case_id", _with("newdata_prediction_refused"))
def test_a_refused_newdata_prediction_fails_until_gates_compare_it(case_id):
    if "check_newdata_refusal" in vars(gates):
        pytest.skip("gates.py compares the model's predictions on the refused new rows")
    case = fx.load_case(case_id)
    data = fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv")
    stand_in = ab.gaussian("y ~ x + (1 | g)", data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                                                              sigma=ab.HalfNormal(1)))
    with pytest.raises(OracleMismatch, match="check_newdata_refusal"):
        gates.Bundle(case_id, model=stand_in)


# -- G04: gates.check_newdata_refusal, the new rows brms refuses to predict ------------------------------------------------------
@pytest.fixture(scope="module")
def newlevels():
    if not fx.has_fixture("gauss_factor_call_newlevels"):
        pytest.skip("fixture not frozen")
    return gates.Bundle("gauss_factor_call_newlevels")


def _with_raw(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_newdata_refusal_compares_the_new_rows_through_the_fitted_levels(newlevels):
    b = newlevels
    assert sorted(set(b.newdata["ord"])) == [1, 2, 3] and sorted(set(b.data["ord"])) == [1, 2, 3, 4]    # the new rows lack level 4
    Xn = gates.code_new_rows(b)
    names = b.fixture.colnames("X")
    assert names == ["Intercept", "x", "factorord2", "factorord3", "factorord4"] and Xn.shape == (5, 5) and not Xn[:, 4].any()
    assert np.array_equal(Xn[:, 2:], np.column_stack([(b.newdata["ord"] == k).to_numpy(dtype=float) for k in (2, 3, 4)]))
    gates.check_newdata_refusal(b)
    gates.check_predictions(b)                                   # the fitted rows against brms, the new rows through the gate


def test_newdata_refusal_fails_on_a_wrong_prediction_or_record(newlevels, monkeypatch):
    b = newlevels

    def unrefuse(raw):                                           # the fixture says brms predicted the new rows after all
        raw["predictions"]["newdata"]["linpred"] = raw["predictions"]["fitted"]["linpred"]
        del raw["predictions"]["newdata"]["linpred_refused"]
    with pytest.raises(OracleMismatch, match="the fixture's newdata linpred is recorded; the case lists"):
        gates.check_newdata_refusal(_with_raw(b, unrefuse))
    predict = b.model.predict
    for kind, message in (("linear", "the linear predictor on the new rows brms refuses differs"),
                          ("mean", "the expected values on the new rows brms refuses differ")):
        def wrong(idata, newdata=None, kind_=kind, **kw):        # a model whose predictions on the new rows are off by 1e-6
            out = predict(idata, newdata=newdata, **kw)
            return out + 1e-6 if kw.get("kind") == kind_ and newdata is not None else out
        monkeypatch.setattr(b.model, "predict", wrong)
        with pytest.raises(OracleMismatch, match=message):
            gates.check_newdata_refusal(b)
    monkeypatch.setattr(b.model, "predict", predict)
    gates.check_newdata_refusal(b)


def test_code_new_rows_codes_only_what_it_knows(newlevels):
    b = newlevels

    def gate_view(mutate):                                       # the coding reads the fixture, the data and the new rows only
        raw = copy.deepcopy(b.fixture.raw)
        mutate(raw)
        return SimpleNamespace(fixture=fx.Fixture(raw), data=b.data, newdata=b.newdata)

    def rename(raw):
        raw["standata"]["X"]["colnames"][2] = "polyord1"
    with pytest.raises(OracleMismatch, match="brms's design column 'polyord1' is not one the gate codes"):
        gates.code_new_rows(gate_view(rename))

    def recode(raw):
        X = np.asarray(raw["standata"]["X"]["values"], dtype=float)
        X[-1] = 1.0 - X[-1]
        raw["standata"]["X"]["values"] = X.tolist()
    with pytest.raises(OracleMismatch, match="the gate's coding of the fitted rows does not reproduce brms's X"):
        gates.code_new_rows(gate_view(recode))
