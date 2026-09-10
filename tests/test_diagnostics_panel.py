"""The diagnostic panel's statuses, driven with synthetic InferenceData: pass, investigate,
fail and n/a, and the formatted output."""
import arviz as az
import numpy as np
import xarray as xr

from autobayes.diagnostics import THRESHOLDS, format_panel, panel


def make_idata(*, chains=4, draws=1000, shift=0.0, n_divergent=0, treedepth_frac=0.0, with_energy=True,
               with_treedepth=True, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(chains, draws)) + shift * np.arange(chains)[:, None]
    post = xr.Dataset({"theta": (("chain", "draw"), theta)})
    ss = {}
    div = np.zeros((chains, draws), dtype=bool)
    div.ravel()[:n_divergent] = True
    ss["diverging"] = (("chain", "draw"), div)
    if with_energy:
        ss["energy"] = (("chain", "draw"), rng.normal(size=(chains, draws)) + theta ** 2 / 2)
    if with_treedepth:
        td = np.full((chains, draws), 5)
        k = int(round(treedepth_frac * chains * draws))
        td.ravel()[:k] = 10
        ss["tree_depth"] = (("chain", "draw"), td)
    return az.InferenceData(posterior=post, sample_stats=xr.Dataset(ss))


def test_clean_run_passes():
    p = panel(make_idata())
    assert p["overall"] == "pass" and all(s == "pass" for s in p["status"].values())
    text = format_panel(p)
    assert "overall: pass" in text and "divergences            0 of 4000" in text


def test_divergence_thresholds():
    assert panel(make_idata(n_divergent=1))["status"]["divergence_fraction"] == "investigate"   # 1/4000 < 0.1 %
    p = panel(make_idata(n_divergent=8))
    assert p["status"]["divergence_fraction"] == "fail" and p["overall"] == "fail"                # 0.2 %


def test_rhat_and_ess_thresholds():
    p = panel(make_idata(shift=1.0))                       # chains centred 1 SD apart
    assert p["values"]["max_rhat"] > 1.05 and p["status"]["max_rhat"] == "fail"
    p = panel(make_idata(chains=2, draws=120))
    assert p["values"]["min_ess_bulk"] < 400 and p["status"]["min_ess_bulk"] in {"investigate", "fail"}


def test_treedepth_saturation_thresholds():
    assert panel(make_idata(treedepth_frac=0.05))["status"]["treedepth_saturation"] == "investigate"
    assert panel(make_idata(treedepth_frac=0.5))["status"]["treedepth_saturation"] == "fail"


def test_missing_stats_are_not_applicable_not_pass():
    p = panel(make_idata(with_energy=False, with_treedepth=False))
    assert np.isnan(p["values"]["min_bfmi"]) and np.isnan(p["values"]["treedepth_saturation"])
    assert p["status"]["min_bfmi"] == "n/a" and p["status"]["treedepth_saturation"] == "n/a"
    assert p["overall"] == "pass"                           # n/a does not fail, and it is not reported as pass either
    assert "n/a" in format_panel(p)


def test_threshold_table_is_the_documented_one():
    ok, inv = THRESHOLDS["max_rhat"]
    assert ok(1.01) and not ok(1.011) and inv(1.05) and not inv(1.051)
    ok, inv = THRESHOLDS["min_ess_bulk"]
    assert ok(401) and not ok(400) and inv(100) and not inv(99)
