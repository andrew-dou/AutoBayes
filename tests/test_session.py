"""Session helpers: environment, fit record, agreement between records."""
import copy
import json

import pytest

import autobayes as ab


def test_import_enables_float64():
    jax = pytest.importorskip("jax")
    assert jax.config.jax_enable_x64 is True
    assert str(jax.numpy.ones(1).dtype) == "float64"


def test_environment_reports_stack():
    env = ab.environment()
    assert env["versions"]["pymc"] and env["versions"]["bambi"] == "0.17.2"
    assert env["jax_devices"] is None or all(":" in d for d in env["jax_devices"])
    text = ab.format_environment(env)
    assert "pymc" in text and "jax" in text


def test_fit_record_is_complete_and_serialisable(model, small_fit):
    rec = ab.fit_record(model, small_fit)
    json.dumps(rec)  # no numpy scalars or arrays left inside
    assert rec["model"]["formula"] == "y ~ x + (1 | g)"
    assert rec["model"]["priors"] == {"intercept": "Normal(mu=0, sigma=5)", "b:x": "Normal(mu=0, sigma=2)",
                                      "sd": "HalfNormal(sigma=1)", "sigma": "HalfNormal(sigma=1)"}
    assert rec["model"]["center_predictors"] is True and rec["model"]["J"] == model.design.J
    ex = rec["execution"]
    assert ex["backend"] == "pymc" and ex["device_actual"] == "cpu" and ex["seed"] == 1
    assert ex["chains"] == 2 and ex["draws"] == 150 and ex["tune"] == 150
    assert set(rec["posterior"]) == {"Intercept", "x", "1|g_sigma", "sigma"}
    assert {"mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean", "ess_bulk", "ess_tail", "r_hat"} <= set(rec["posterior"]["x"])
    assert rec["diagnostics"]["overall"] in {"pass", "investigate", "fail"}
    assert "seed             : 1" in small_fit.report()


def test_compare_records(model, small_fit):
    a = ab.fit_record(model, small_fit)
    same = ab.compare_records(a, copy.deepcopy(a))
    assert same["agree"] and all(r["diff"] == 0.0 for r in same["rows"].values())
    shifted = copy.deepcopy(a)
    shifted["posterior"]["x"]["mean"] += 100 * shifted["posterior"]["x"]["mcse_mean"] + 1.0
    c = ab.compare_records(a, shifted)
    assert not c["agree"] and not c["rows"]["x"]["agree"] and c["rows"]["sigma"]["agree"]
    assert "DISAGREE" in ab.format_comparison(c)
    other = copy.deepcopy(a)
    other["model"]["priors"]["sigma"] = "HalfNormal(sigma=2)"
    with pytest.raises(ValueError, match="different models"):
        ab.compare_records(a, other)
