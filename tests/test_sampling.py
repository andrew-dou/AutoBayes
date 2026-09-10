"""B1-B2: sampling checks on the CPU. Marked slow; run with ``pytest -m slow``."""
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import autobayes as ab
from autobayes.diagnostics import format_panel, panel
from tests.conftest import PRIORS, simulate
from tests.reference_logp import conditional_gaussian_posterior

# Candidate evidence goes to the run-specific directory (conftest.evidence_dir), never to the
# committed docs/m*-evidence/ history. Acceptance is the statistical criteria asserted below,
# not agreement with any earlier JSON.


@pytest.mark.slow
def test_conditional_gaussian_posterior(model, data, evidence_dir):
    """B1: with scales fixed, NUTS must reproduce the exact Gaussian posterior."""
    pmm = model.pymc_model
    sigma, sd = 0.3, 0.5
    fixed = pm.do(pmm, {pmm["sigma"]: sigma, pmm["1|g_sigma"]: sd})
    with fixed:
        idata = pm.sample(draws=2000, tune=1000, chains=4, cores=1, random_seed=11,
                          progressbar=False, compute_convergence_checks=False)
    mean, cov, _ = conditional_gaussian_posterior(
        data, response="y", predictors=["x"], group="g", center=True, sigma=sigma, sd=sd,
        prior_intercept={"mu": 0.0, "sigma": 5.0}, prior_b={"x": {"mu": 0.0, "sigma": 2.0}})
    post = idata.posterior
    draws = np.concatenate([post["Intercept"].to_numpy().reshape(-1, 1), post["x"].to_numpy().reshape(-1, 1),
                            post["1|g_offset"].to_numpy().reshape(-1, model.design.J)], axis=1)
    ess = np.concatenate([np.atleast_1d(az.ess(post[v]).to_array().to_numpy().ravel()) for v in ["Intercept", "x", "1|g_offset"]])
    exact_sd = np.sqrt(np.diag(cov))
    mcse = exact_sd / np.sqrt(ess)
    zscores = (draws.mean(0) - mean) / mcse
    rows = {"exact_mean": mean.tolist(), "mcmc_mean": draws.mean(0).tolist(), "exact_sd": exact_sd.tolist(),
            "mcmc_sd": draws.std(0, ddof=1).tolist(), "ess": ess.tolist(), "z": zscores.tolist()}
    (evidence_dir / "B1_conditional_gaussian.json").write_text(json.dumps(rows, indent=1))
    assert np.all(np.abs(zscores) < 4), zscores
    assert np.allclose(draws.std(0, ddof=1), exact_sd, rtol=0.10)


@pytest.mark.slow
def test_simulated_recovery_both_backends(evidence_dir):
    """B2: simulated data, n=200, J=20; PyMC and NumPyro CPU paths agree and recover truth.

    4 x 1000 draws gave R-hat 1.02 and bulk ESS 350 on the intercept on both backends
    (the intercept trades off against the mean of the 20 offsets); 2500 draws per chain
    is the run that meets the panel thresholds. Recorded in docs/m0-evidence.
    """
    truth = dict(a=1.0, b=0.5, sd=0.5, sigma=0.3)
    df = simulate(n=200, J=20, seed=20, **truth)
    m = ab.gaussian("y ~ x + (1 | g)", df, PRIORS)
    results, summaries = {}, {}
    for backend in ["pymc", "numpyro"]:
        r = m.fit(backend=backend, device="cpu", chains=4, draws=2500, tune=1000, seed=7, cores=4)
        p = panel(r.idata, max_treedepth=10)
        s = az.summary(r.idata, var_names=["Intercept", "x", "1|g_sigma", "sigma"], hdi_prob=0.94, round_to="none")
        results[backend] = (r, p, s)
        summaries[backend] = {"report": r.report(), "panel": p["values"], "status": p["status"],
                              "summary": s.reset_index().to_dict(orient="records")}
        assert p["values"]["divergences"] == 0, format_panel(p)
        assert p["values"]["max_rhat"] <= 1.01, format_panel(p)
        assert p["values"]["min_ess_bulk"] > 400 and p["values"]["min_ess_tail"] > 400, format_panel(p)
        for name, val in [("Intercept", truth["a"]), ("x", truth["b"]), ("1|g_sigma", truth["sd"]), ("sigma", truth["sigma"])]:
            lo, hi = s.loc[name, "hdi_3%"], s.loc[name, "hdi_97%"]
            assert lo <= val <= hi, (backend, name, val, lo, hi)
    # the two backends target the same posterior: means within 4 combined MCSE
    sp, sn = results["pymc"][2], results["numpyro"][2]
    for name in ["Intercept", "x", "1|g_sigma", "sigma"]:
        diff = abs(sp.loc[name, "mean"] - sn.loc[name, "mean"])
        tol = 4 * np.hypot(sp.loc[name, "mcse_mean"], sn.loc[name, "mcse_mean"])
        assert diff < tol, (name, diff, tol)
        summaries.setdefault("agreement", {})[name] = {"diff": float(diff), "tol": float(tol)}
    summaries["truth"] = truth
    (evidence_dir / "B2_recovery.json").write_text(json.dumps(summaries, indent=1, default=str))


@pytest.mark.slow
def test_bernoulli_recovery_both_backends(evidence_dir):
    """M1 B1: simulated binary data, n=600, J=30; PyMC and NumPyro CPU paths agree and recover truth.

    Thresholds set in docs/m1-decision.md before the run. One recovery is not calibration.
    """
    from scipy.special import expit

    truth = dict(a=-0.5, b=0.8, sd=0.7)
    rng = np.random.default_rng(21)
    n, J = 600, 30
    g = rng.integers(0, J, n); x = rng.normal(size=n); u = rng.normal(0, truth["sd"], J)
    df = pd.DataFrame({"x": x, "g": [f"lvl{k:02d}" for k in g]})
    df["y"] = rng.binomial(1, expit(truth["a"] + truth["b"] * x + u[g]))
    priors = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
    m = ab.bernoulli("y ~ x + (1 | g)", df, priors)
    results, summaries = {}, {"truth": truth, "n": n, "J": J}
    for backend in ["pymc", "numpyro"]:
        r = m.fit(backend=backend, device="cpu", chains=4, draws=2000, tune=1000, seed=7, cores=4)
        p = panel(r.idata, max_treedepth=10)
        rec = ab.fit_record(m, r, panel=p)
        results[backend] = rec
        summaries[backend] = rec
        assert p["values"]["divergences"] == 0, format_panel(p)
        assert p["values"]["max_rhat"] <= 1.01, format_panel(p)
        assert p["values"]["min_ess_bulk"] > 400 and p["values"]["min_ess_tail"] > 400, format_panel(p)
        for name, val in [("Intercept", truth["a"]), ("x", truth["b"]), ("1|g_sigma", truth["sd"])]:
            row = rec["posterior"][name]
            assert row["hdi_3%"] <= val <= row["hdi_97%"], (backend, name, val, row)
    c = ab.compare_records(results["pymc"], results["numpyro"])
    summaries["agreement"] = c
    (evidence_dir / "M1_B1_recovery.json").write_text(json.dumps(summaries, indent=1, default=str))
    assert c["agree"], ab.format_comparison(c)
