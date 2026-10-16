"""D05 (docs/post-processing.md): ``waic``, ``loo_compare`` and ``model_weights`` beyond the oracle gate, which compares them
with brms on the frozen cases ``gauss_loo_compare_x`` and ``gauss_loo_compare_xw``.

- WAIC, ``loo_compare`` (by loo and by waic) and pseudo-BMA weights against loo 2.9.0's outputs on the deterministic inputs of
  ``tests/loo_probes/compare_inputs.py`` (``loo-compare-2.9.0.json``): three to five models, models given unnamed, exact ties,
  raw and Pareto-smoothed weights, log-likelihoods near -800, nearly equal models and a weight that underflows to zero;
- brms 2.23's ``loo_compare`` and ``model_weights`` by ``"loo"``, ``"waic"`` and ``"pseudobma"`` (``BB = FALSE``) of three
  fixed-parameter fits with close elpd, from the log-likelihoods brms computed (``brms-2.23.0-weights.json``);
- the API: names, the order of tied models, the response check, the result objects and every refusal.
No sampler runs here: every posterior is a set of frozen or synthetic draws.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import autobayes as ab
from autobayes.post_processing import WAIC_COLUMNS, Loo, Waic, loo_from_log_lik, response_fingerprint, waic_from_log_lik
from tests.loo_probes.compare_inputs import compare_inputs, sha256
from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.tolerances import PRED_REL, rel_err

PROBES = Path(__file__).parent / "loo_probes"
RECORD = json.loads((PROBES / "loo-compare-2.9.0.json").read_text())
BRMS = json.loads((PROBES / "brms-2.23.0-weights.json").read_text())
INPUTS = compare_inputs()


def _same(ours, theirs, tol=PRED_REL):
    """Equal to ``tol`` where finite, and the same non-finite values elsewhere."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert ours.shape == theirs.shape
    assert np.array_equal(np.isfinite(ours), np.isfinite(theirs))
    assert np.array_equal(ours[~np.isfinite(ours)], theirs[~np.isfinite(theirs)], equal_nan=True)
    finite = np.isfinite(ours)
    assert rel_err(ours[finite], theirs[finite]) <= tol


def _same_weights(ours, theirs):
    """Weights equal to ``PRED_REL``, and so are their logarithms: a weight far below one is checked in its exponent."""
    _same(ours, theirs)
    with np.errstate(divide="ignore"):
        _same(np.log(np.asarray(ours, dtype=float)), np.log(np.asarray(theirs, dtype=float)))


def _as_log_lik(ll):
    """A (draws, rows) matrix as a one-chain log-likelihood."""
    return xr.DataArray(np.asarray(ll)[None], dims=("chain", "draw", "__obs__"))


def _results(probe, kind):
    """The probe's models as loo was given them: a mapping by name, or a list (loo names them model1, model2, ...)."""
    named, models = INPUTS[probe["label"]]
    assert named == probe["named"] and list(models) == probe["models"]
    assert [sha256(ll) for ll in models.values()] == probe["sha256"], "the probe inputs changed: re-record them"
    make = (lambda ll: loo_from_log_lik(_as_log_lik(ll), r_eff=1.0)) if kind == "loo" else (lambda ll: waic_from_log_lik(_as_log_lik(ll)))
    out = {name: make(ll) for name, ll in models.items()}
    return out if named else list(out.values())


def _check_table(table, ref):
    assert list(table.index) == ref["rows"] and list(table.columns) == ref["columns"]
    assert table.index.name == "model"
    _same(table.to_numpy(), fx.decode(ref["table"]))


@pytest.mark.parametrize("probe", RECORD["probes"], ids=lambda p: p["label"])
def test_waic_equals_loo_2_9_0_on_the_probe_inputs(probe):
    """Every pointwise column and the estimates (with their SEs) of each model's WAIC equal loo 2.9.0's; so do the loo
    estimates the comparison is built on (C07's PSIS-LOO, with r_eff = 1)."""
    waics, loos = (_results(probe, kind) for kind in ("waic", "loo"))
    if probe["named"]:
        waics, loos = list(waics.values()), list(loos.values())
    for k, (w, l) in enumerate(zip(waics, loos)):
        ref = probe["waic"][k]
        assert list(w.pointwise.columns) == ref["columns"] == list(WAIC_COLUMNS)
        assert list(w.estimates.index) == ref["rows"] and list(w.estimates.columns) == ["Estimate", "SE"]
        _same(w.pointwise.to_numpy(), fx.decode(ref["pointwise"]))
        _same(w.estimates.to_numpy(), fx.decode(ref["estimates"]))
        _same(l.estimates.to_numpy(), fx.decode(probe["loo_estimates"][k]))


@pytest.mark.parametrize("probe", RECORD["probes"], ids=lambda p: p["label"])
def test_loo_compare_equals_loo_2_9_0_on_the_probe_inputs(probe):
    """loo_compare by loo and by waic: the rows (best first, ties in the order given, loo's model1, model2, ... for unnamed
    models), loo's eight columns and every entry."""
    _check_table(ab.loo_compare(_results(probe, "loo")), probe["compare_loo"])
    _check_table(ab.loo_compare(_results(probe, "waic")), probe["compare_waic"])


@pytest.mark.parametrize("probe", RECORD["probes"], ids=lambda p: p["label"])
def test_pseudobma_weights_equal_loo_model_weights(probe):
    """loo_model_weights(method = "pseudobma", BB = FALSE), in the order given: from equal thirds to a weight of 2.5e-13 and
    one that underflows to zero."""
    w = ab.model_weights(_results(probe, "loo"), "pseudobma", BB=False)
    assert list(w.index) == probe["pseudobma"]["models"] and w.name == "pseudobma"
    _same_weights(w.to_numpy(), fx.decode(probe["pseudobma"]["weights"]))


def test_the_probes_reach_ties_unnamed_models_and_underflow():
    by_label = {p["label"]: p for p in RECORD["probes"]}
    assert by_label["ties S=100"]["compare_loo"]["rows"] == ["b", "c", "a", "d"]          # exact ties keep the order given
    assert by_label["three S=100"]["compare_loo"]["rows"][0] == "model3" and not by_label["three S=100"]["named"]
    assert fx.decode(by_label["far apart S=100"]["pseudobma"]["weights"]).tolist() == [1.0, 0.0]
    five = fx.decode(by_label["five S=1000"]["pseudobma"]["weights"])
    assert 1e-14 < five.min() < 1e-12 and 0.9 < five.max() < 1.0


def test_brms_loo_compare_and_model_weights_of_three_fits():
    """brms's loo_compare (rows best first, given worst first) and model_weights by "loo" and "waic" (Akaike weights of looic
    and WAIC) and "pseudobma" (BB = FALSE), recomputed from the log-likelihoods brms computed for three fixed-parameter fits."""
    data = fx.fixture_dir("gauss_loo_compare_x") / "data.csv"
    assert hashlib.md5(data.read_bytes()).hexdigest() == BRMS["data_md5"]
    models = BRMS["models"]
    lls = [fx.decode(e) for e in BRMS["log_lik"]]
    loos = {m: loo_from_log_lik(_as_log_lik(ll), r_eff=1.0) for m, ll in zip(models, lls)}
    waics = {m: waic_from_log_lik(_as_log_lik(ll)) for m, ll in zip(models, lls)}
    for k, m in enumerate(models):
        _same(loos[m].estimates.to_numpy(), fx.decode(BRMS["loo_estimates"][k]))
        _same(waics[m].estimates.to_numpy(), fx.decode(BRMS["waic_estimates"][k]))
    _check_table(ab.loo_compare(loos), BRMS["compare_loo"])
    _check_table(ab.loo_compare(waics), BRMS["compare_waic"])
    assert BRMS["compare_loo"]["rows"] == ["m2", "m3", "m1"]
    for kind, results, kwargs in (("loo", loos, {}), ("waic", waics, {}), ("pseudobma", loos, {"BB": False})):
        w = ab.model_weights(results, kind, **kwargs)
        assert list(w.index) == BRMS[f"weights_{kind}"]["models"] == models
        _same_weights(w.to_numpy(), fx.decode(BRMS[f"weights_{kind}"]["weights"]))
    assert 1e-11 < fx.decode(BRMS["weights_waic"]["weights"]).min() < 1e-10


# -- the API ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def frozen():
    """The frozen case's model and its points held as brms's fixed-parameter fit holds them."""
    if not fx.has_fixture("gauss_loo_compare_x") or fx.load_case("gauss_loo_compare_x").get("gate"):
        pytest.skip("gauss_loo_compare_x is not frozen or still gated")
    b = gates.Bundle("gauss_loo_compare_x")
    return b, gates.fixed_parameter_draws(b)


def test_waic_result(frozen):
    b, idata = frozen
    res = ab.waic(b.model, idata)
    assert isinstance(res, ab.Waic) and res.n_draws == 8 and len(res.pointwise) == b.model.design.n
    ll = ab.log_lik(b.model, idata).transpose("chain", "draw", "__obs__").to_numpy().reshape(8, -1)
    lpd = np.log(np.mean(np.exp(ll), axis=0))
    np.testing.assert_allclose(res.pointwise["p_waic"], np.var(ll, axis=0, ddof=1), rtol=1e-14)
    np.testing.assert_allclose(res.pointwise["elpd_waic"], lpd - np.var(ll, axis=0, ddof=1), rtol=1e-12)
    assert res.elpd_waic == pytest.approx(res.pointwise["elpd_waic"].sum()) and res.waic == pytest.approx(-2 * res.elpd_waic)
    assert res.p_waic == pytest.approx(res.pointwise["p_waic"].sum())
    assert res.se_elpd_waic == pytest.approx(math.sqrt(36 * np.var(res.pointwise["elpd_waic"], ddof=1)))
    assert res.response == response_fingerprint(b.model) == ab.loo(b.model, idata).response is not None
    text = str(res)
    assert "WAIC of 36 observations from 8 posterior draws" in text and "p_waic above 0.4" in text


def test_waic_refusals(frozen):
    with pytest.raises(ValueError, match="at least two posterior draws"):
        waic_from_log_lik(_as_log_lik(np.array([[-1.0, -2.0]])))
    with pytest.raises(ValueError, match="finite log-likelihood"):
        waic_from_log_lik(_as_log_lik(np.array([[0.0, -np.inf], [-1.0, -2.0]])))
    single = waic_from_log_lik(_as_log_lik(np.array([[-1.0], [-2.0], [-1.5]])))      # one observation: loo's SE is NA
    assert np.isnan(single.estimates["SE"]).all()
    for case_id, needle in (("gauss_mv_rescor", "multivariate model"), ("gauss_me", "model with me\\(\\) or mi\\(\\)")):
        if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
            continue
        b = gates.Bundle(case_id)
        with pytest.raises(ValueError, match=f"waic\\(\\) of a {needle}"):
            ab.waic(b.model, adapter.draws(b.fixture, b.model, b.cmap, b.gmap))


def _loo(ll, **kw):
    return loo_from_log_lik(_as_log_lik(ll), r_eff=1.0, **kw)


def test_loo_compare_names_order_and_columns():
    rng = np.random.default_rng(7)
    a, b = rng.normal(-1.0, 0.3, (40, 10)), rng.normal(-1.3, 0.3, (40, 10))
    unnamed = ab.loo_compare([_loo(b), _loo(a), _loo(a)])                 # loo's names; the tie keeps the order given
    assert list(unnamed.index) == ["model2", "model3", "model1"]
    assert list(unnamed.columns) == ["elpd_diff", "se_diff", "elpd_loo", "se_elpd_loo", "p_loo", "se_p_loo", "looic", "se_looic"]
    assert (unnamed.loc[["model2", "model3"], ["elpd_diff", "se_diff"]].to_numpy() == 0.0).all()
    by_waic = ab.loo_compare({"first": waic_from_log_lik(_as_log_lik(b)), "second": waic_from_log_lik(_as_log_lik(a))})
    assert list(by_waic.index) == ["second", "first"]
    assert list(by_waic.columns) == ["elpd_diff", "se_diff", "elpd_waic", "se_elpd_waic", "p_waic", "se_p_waic", "waic", "se_waic"]
    diff = waic_from_log_lik(_as_log_lik(b)).pointwise["elpd_waic"] - waic_from_log_lik(_as_log_lik(a)).pointwise["elpd_waic"]
    assert by_waic.loc["first", "elpd_diff"] == pytest.approx(diff.sum(), rel=1e-14)
    assert by_waic.loc["first", "se_diff"] == pytest.approx(math.sqrt(10 * np.var(diff, ddof=1)), rel=1e-14)


def test_loo_compare_and_model_weights_refusals():
    rng = np.random.default_rng(8)
    a, b = _loo(rng.normal(-1.0, 0.3, (40, 10))), _loo(rng.normal(-1.0, 0.3, (40, 10)))
    w = waic_from_log_lik(_as_log_lik(rng.normal(-1.0, 0.3, (40, 10))))
    for fn in (ab.loo_compare, lambda r: ab.model_weights(r, "loo")):
        with pytest.raises(ValueError, match="at least two models"):
            fn(a)
        with pytest.raises(ValueError, match="at least two models"):
            fn({"a": a})
        with pytest.raises(ValueError, match="PSIS-LOO and WAIC results together"):
            fn({"a": a, "w": w})
        with pytest.raises(ValueError, match="same number of observations"):
            fn([a, _loo(rng.normal(-1.0, 0.3, (40, 9)))])
        with pytest.raises(ValueError, match="at least two observations"):
            fn([_loo(rng.normal(-1.0, 0.3, (40, 1))), _loo(rng.normal(-1.0, 0.3, (40, 1)))])
        with pytest.raises(ValueError, match="results of ab.loo\\(\\) or ab.waic\\(\\)"):
            fn({"a": a, "b": pd.DataFrame()})
    for kind in ("stacking", "bma", "kfold"):
        with pytest.raises(ValueError, match="divergence D92"):
            ab.model_weights([a, b], kind)
    with pytest.raises(ValueError, match="must be one of"):
        ab.model_weights([a, b], "aic")
    with pytest.raises(TypeError):
        ab.model_weights([a, b])                                   # no default: brms's default, stacking, is not offered
    for BB in (None, True):
        with pytest.raises(ValueError, match="needs BB=False"):
            ab.model_weights([a, b], "pseudobma", BB=BB)
    with pytest.raises(ValueError, match="BB applies to weights='pseudobma' only"):
        ab.model_weights([a, b], "loo", BB=False)
    with pytest.raises(ValueError, match="takes the results of ab.waic"):
        ab.model_weights([a, b], "waic")
    w2 = waic_from_log_lik(_as_log_lik(rng.normal(-1.0, 0.3, (40, 10))))
    for kind in ("loo", "pseudobma"):
        with pytest.raises(ValueError, match="takes the results of ab.loo"):
            ab.model_weights([w, w2], kind, **({"BB": False} if kind == "pseudobma" else {}))
    np.testing.assert_allclose(ab.model_weights([a, b], "loo").to_numpy(), ab.model_weights([a, b], "pseudobma", BB=False).to_numpy(),
                               rtol=0, atol=0)


def test_different_responses_are_compared_with_a_warning(frozen):
    """loo warns when brms's yhash attributes differ; so does loo_compare (and model_weights) when the models' response
    fingerprints differ. Results of a bare log-likelihood carry none and are not checked."""
    b, idata = frozen
    shifted = b.data.assign(y=b.data["y"] + 0.1)
    other = adapter.build_model(fx.load_case("gauss_loo_compare_x"), shifted)
    ours, theirs = ab.loo(b.model, idata), ab.loo(other, idata)
    assert ours.response != theirs.response
    with pytest.warns(UserWarning, match="not all models have the same response"):
        ab.loo_compare({"y": ours, "y + 0.1": theirs})
    with pytest.warns(UserWarning, match="not all models have the same response"):
        ab.model_weights({"y": ours, "y + 0.1": theirs}, "loo")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ab.loo_compare({"y": ours, "again": ab.loo(b.model, idata)})
        ab.loo_compare({"y": ours, "bare": loo_from_log_lik(ab.log_lik(other, idata))})
    assert not [w for w in caught if "same response" in str(w.message)]
    assert isinstance(ours, Loo) and isinstance(ab.waic(b.model, idata), Waic)
