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
from tests.evidence import write_evidence
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
    write_evidence(evidence_dir, "B1_conditional_gaussian.json", json.dumps(rows, indent=1))
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
    write_evidence(evidence_dir, "B2_recovery.json", json.dumps(summaries, indent=1, default=str))


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
    write_evidence(evidence_dir, "M1_B1_recovery.json", json.dumps(summaries, indent=1, default=str))
    assert c["agree"], ab.format_comparison(c)


@pytest.mark.slow
def test_history_bytes_unchanged_after_sampling(evidence_dir):
    """After the sampling checks wrote their records, every committed evidence file still has
    its committed bytes (a byte comparison against HEAD, not an existence check)."""
    from tests.conftest import REPO
    from tests.evidence import committed_bytes, working_tree_bytes

    head = committed_bytes(REPO, ["docs/m0-evidence", "docs/m1-evidence"])
    assert len(head) >= 10
    assert working_tree_bytes(REPO, list(head)) == head
    assert not any(p.resolve().is_relative_to((REPO / "docs").resolve()) for p in evidence_dir.iterdir())


@pytest.mark.slow
def test_binomial_recovery_both_backends(evidence_dir):
    """M2 B1: 300 rows, 30 groups all present, trials 5-50; PyMC and NumPyro CPU paths agree
    and recover the truth. Thresholds set in docs/m2-decision.md before the run. One recovery
    on one simulated data set; not calibration."""
    from scipy.special import expit

    truth = dict(a=-0.5, b=0.8, sd=0.7)
    rng = np.random.default_rng(7)
    rows, J = 300, 30
    g = np.array([i % J for i in range(rows)]); rng.shuffle(g)
    x = rng.normal(size=rows); u = rng.normal(0, truth["sd"], J)
    n = rng.integers(5, 51, rows)
    df = pd.DataFrame({"x": x, "g": [f"lvl{j:02d}" for j in g], "n": n})
    df["k"] = rng.binomial(n, expit(truth["a"] + truth["b"] * x + u[g]))
    assert df.g.nunique() == J
    m = ab.binomial("k | trials(n) ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1)))
    results, summaries = {}, {"truth": truth, "rows": rows, "J": J}
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
    write_evidence(evidence_dir, "M2_B1_recovery.json", json.dumps(summaries, indent=1, default=str))
    assert c["agree"], ab.format_comparison(c)


@pytest.mark.slow
def test_categorical_recovery_both_backends(evidence_dir):
    """C10 (docs/m3a1-decision.md): Gaussian y ~ x + z + (1 | g), one categorical predictor with unequal
    level proportions, prespecified design, seed and gates; both CPU samplers. One recovery, not calibration."""
    from tests.test_categorical import BETA, LEVELS, PROPS, TRUTH
    rng = np.random.default_rng(11)
    n, J = 300, 10
    z = rng.choice(LEVELS, size=n, p=PROPS)
    g = np.array([f"g{i % J:02d}" for i in range(n)]); rng.shuffle(g)
    x = rng.normal(size=n)
    u = rng.normal(0, TRUTH["sd"], J)
    y = TRUTH["a"] + TRUTH["bx"] * x + np.array([BETA[v] for v in z]) + u[[int(s[1:]) for s in g]] + rng.normal(0, TRUTH["sigma"], n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "g": g})
    coverage = pd.crosstab(df.g, df.z)
    assert coverage.shape == (J, len(LEVELS)) and (coverage > 0).all().all(), coverage     # every level in every group
    priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    m = ab.gaussian("y ~ x + z + (1 | g)", df, priors, categorical={"z": "a"})
    truth = {"Intercept": TRUTH["a"], "x": TRUTH["bx"], "z[b]": BETA["b"], "z[c]": BETA["c"], "1|g_sigma": TRUTH["sd"], "sigma": TRUTH["sigma"]}
    results, records = {}, {}
    for backend in ["pymc", "numpyro"]:
        fit = m.fit(backend=backend, device="cpu", chains=4, draws=1000, tune=1000, seed=11, target_accept=0.9)
        p = panel(fit.idata)
        s = az.summary(fit.idata, var_names=["Intercept", "x", "z", "1|g_sigma", "sigma"], hdi_prob=0.99, round_to="none")
        rec = ab.fit_record(m, fit, panel=p)
        records[backend] = rec
        rows = {}
        for name, r in s.iterrows():
            rows[name] = {"mean": float(r["mean"]), "sd": float(r["sd"]), "hdi_0.5%": float(r["hdi_0.5%"]), "hdi_99.5%": float(r["hdi_99.5%"]),
                          "ess_bulk": float(r["ess_bulk"]), "r_hat": float(r["r_hat"]), "truth": truth[name],
                          "inside_99": bool(r["hdi_0.5%"] <= truth[name] <= r["hdi_99.5%"])}
        results[backend] = {"panel": {"values": {k: (float(v) if v is not None else None) for k, v in p["values"].items()},
                                      "status": dict(p["status"])}, "population": rows, "report": fit.report()}
    write_evidence(evidence_dir, "C10_categorical_recovery.json", json.dumps(results, indent=1))
    write_evidence(evidence_dir, "C10_categorical_recovery_records.json", json.dumps(records, indent=1))
    for backend, res in results.items():
        assert res["panel"]["values"]["divergences"] == 0, (backend, res["panel"])
        assert all(r["r_hat"] <= 1.01 for r in res["population"].values()), (backend, res["population"])
        assert all(r["ess_bulk"] >= 400 for r in res["population"].values()), (backend, res["population"])
        assert all(r["inside_99"] for r in res["population"].values()), (backend, res["population"])
    comparison = ab.compare_records(records["pymc"], records["numpyro"])
    write_evidence(evidence_dir, "C10_categorical_agreement.json", json.dumps(comparison, indent=1))
    assert comparison["agree"], comparison
