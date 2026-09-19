"""One prospective NumPyro convergence diagnostic on the frozen F7 inputs (approval: reviewed-unit.json in this directory).

--check   target-free settings/identity check: head, nine input hashes, versions, x64, CPU devices, netCDF engine,
          approved settings, attempt directory absent. Builds no data and no model, fits nothing. Exit 0 or 2.
--run DIR the single child run inside the supervisor's exclusive attempt directory: identity re-check, exact seed-17
          data reconstruction with hashes, model build, ONE NumPyro fit with the approved settings, immediate netCDF
          persistence of the full InferenceData with SHA-256, then diagnostics and small summaries. Exit 0 when the
          run completed (the scientific outcome is in summary.json), 2 on refusal, 1 on an unexpected error.
No retry, no reseeding, no PyMC fit, no reference evaluation, no threshold change. Progress files are written after
every phase so a timeout leaves real partial evidence."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

UNIT = Path(__file__).resolve().parent
REPO = Path("/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3b1-varying-slope")
APPROVAL = UNIT / "reviewed-unit.json"
SETTINGS = {"backend": "numpyro", "device": "cpu", "chain_method": "parallel", "dtype": "float64", "chains": 4, "tune": 2000,
            "draws": 4000, "seed": 17, "target_accept": 0.9, "max_treedepth": 10}
EXPECTED_VERSIONS = {"numpyro": "0.21.0", "jax": "0.10.2", "jaxlib": "0.10.2", "pymc": "5.28.5", "pytensor": "2.38.3", "bambi": "0.17.2",
                     "arviz": "0.23.4", "numpy": "2.4.6"}
POPULATION = ["Intercept", "x", "1|g_sigma", "x|g_sigma", "sigma"]
TRUTH = {"Intercept": 1.0, "x": 0.5, "1|g_sigma": 0.5, "x|g_sigma": 0.3, "sigma": 0.4}
SCALES = ("1|g_sigma", "x|g_sigma", "sigma")
FORMULA = "y ~ x + (1 + x || g)"
REFERENCE_RESULT = REPO / "docs/m3b1-evidence/f7-direct-cdf-v3r1-unit-1/result.json"
THRESHOLD_TEXT = {"divergence_fraction": "== 0", "max_rhat": "<= 1.01", "min_ess_bulk": "> 400", "min_ess_tail": "> 400", "min_bfmi": "> 0.3", "treedepth_saturation": "< 0.01"}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def dump(path, obj) -> None:
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n"); tmp.replace(path)


def git_state() -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    return {"head": head, "dirty": bool(status.strip()), "status": status}


def identity(attempt_dir: Path | None) -> dict:
    """Everything that must match before any model or fit; imports the library, never the model or data."""
    approval = json.loads(APPROVAL.read_text())
    problems = []
    git = git_state()
    if git["head"] != approval["base_head"]:
        problems.append(f"head {git['head'][:12]} != approved base_head {approval['base_head'][:12]}")
    if git["dirty"]:
        problems.append("checkout is dirty")
    hashes = {k: (sha256(REPO / k) if (REPO / k).is_file() else None) for k in approval["input_sha256"]}
    bad = [k for k, v in hashes.items() if v != approval["input_sha256"][k]]
    if bad:
        problems.append(f"input hash mismatch: {bad}")
    if approval.get("settings") != SETTINGS:
        problems.append(f"approved settings {approval.get('settings')} != script settings {SETTINGS}")
    if approval.get("max_total_wall_seconds") != 300 or approval.get("retries") != 0:
        problems.append("approval allowance/retries differ from 300 s / 0")
    import autobayes as ab
    env = ab.environment()
    versions = env.get("versions", {})
    wrong = {k: versions.get(k) for k, v in EXPECTED_VERSIONS.items() if versions.get(k) != v}
    if wrong:
        problems.append(f"dependency versions differ from the records: {wrong}")
    if env.get("jax_x64") is not True:
        problems.append("jax_enable_x64 is not True")
    if env.get("python") != "3.11.0":
        problems.append(f"python {env.get('python')} != 3.11.0")
    import jax
    cpus = jax.devices("cpu")
    if len(cpus) < SETTINGS["chains"]:
        problems.append(f"only {len(cpus)} JAX CPU devices for {SETTINGS['chains']} parallel chains")
    try:
        import h5netcdf  # noqa: F401
    except ImportError:
        problems.append("h5netcdf engine unavailable: draws could not be persisted as netCDF")
    if attempt_dir is None:
        if Path(approval["attempt_dir"]).exists():
            problems.append(f"attempt directory already exists: {approval['attempt_dir']}")
    else:
        if str(attempt_dir.resolve()) != approval["attempt_dir"]:
            problems.append(f"attempt dir {attempt_dir} != approved {approval['attempt_dir']}")
        elif any(p.suffix == ".nc" or p.name == "summary.json" for p in attempt_dir.iterdir()):
            problems.append("attempt directory already holds run outputs; no second attempt")
    return {"checked_at": now(), "ok": not problems, "problems": problems, "git": git, "input_sha256": hashes, "script_sha256": sha256(__file__),
            "approval_sha256": sha256(APPROVAL), "settings": SETTINGS, "environment": env, "jax_cpu_devices": len(cpus), "cpu_count": os.cpu_count(),
            "netcdf_engine": "h5netcdf"}


def frozen_data():
    """The F7 generator of tests/test_sampling.py, reproduced line for line (seed 17, 30 groups of 20)."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(17)
    J, per = 30, 20
    g = np.repeat([f"g{j:02d}" for j in range(J)], per)
    x = rng.normal(size=J * per)
    u = rng.normal(0, 0.5, J); v = rng.normal(0, 0.3, J)
    gi = np.repeat(np.arange(J), per)
    y = 1.0 + 0.5 * x + u[gi] + x * v[gi] + rng.normal(0, 0.4, J * per)
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    from tests.reference_marginal import f7_data  # data only; no density is evaluated
    y2, x2, g2 = f7_data()
    identical = bool(np.array_equal(y, y2) and np.array_equal(x, x2) and np.array_equal(g, g2))
    digest = hashlib.sha256(np.ascontiguousarray(y).tobytes() + np.ascontiguousarray(x).tobytes() + "\n".join(g).encode()).hexdigest()
    csv_digest = hashlib.sha256(df.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    return df, {"generator": "tests/test_sampling.py::test_varying_slope_recovery_both_backends lines 260-266, seed 17, J=30, per=20",
                "identical_to_tests_reference_marginal_f7_data": identical, "n": int(len(df)), "J": J, "per": per,
                "sha256_arrays_y_x_g": digest, "sha256_csv_17g": csv_digest, "y_mean": float(y.mean()), "x_mean": float(x.mean())}


def run(attempt_dir: Path) -> int:
    import numpy as np
    t_start = time.perf_counter()
    progress = {"unit_id": "F7_numpyro_convergence_2000_4000_unit_1", "started": now(), "pid": os.getpid(), "phases": []}
    def phase(name, **extra):
        progress["phases"].append({"phase": name, "at": now(), "elapsed_s": round(time.perf_counter() - t_start, 3), **extra}); dump(attempt_dir / "progress.json", progress)
    ident = identity(attempt_dir); dump(attempt_dir / "check.json", ident); phase("identity", ok=ident["ok"])
    if not ident["ok"]:
        print("refused:", ident["problems"]); return 2
    import autobayes as ab
    import arviz as az
    import xarray as xr
    from autobayes.diagnostics import panel
    sys.path.insert(0, str(REPO))
    df, data_info = frozen_data(); dump(attempt_dir / "data.json", data_info); phase("data", **{k: data_info[k] for k in ("sha256_arrays_y_x_g", "identical_to_tests_reference_marginal_f7_data")})
    if not data_info["identical_to_tests_reference_marginal_f7_data"]:
        print("refused: reconstructed data differ from tests.reference_marginal.f7_data"); return 2
    priors = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}, sigma=ab.HalfNormal(1))
    m = ab.gaussian(FORMULA, df, priors); phase("model")
    fit = m.fit(backend=SETTINGS["backend"], device=SETTINGS["device"], chains=SETTINGS["chains"], draws=SETTINGS["draws"], tune=SETTINGS["tune"],
                seed=SETTINGS["seed"], target_accept=SETTINGS["target_accept"])
    config = {"backend": fit.backend, "device_requested": fit.device_requested, "device_actual": fit.device_actual, "platform": fit.platform, "dtype": fit.dtype,
              "chain_method": fit.chain_method, "chains": fit.chains, "draws": fit.draws, "tune": fit.tune, "target_accept": fit.target_accept, "seed": fit.seed,
              "fit_wall_seconds": float(fit.wall_seconds), "notes": list(fit.notes)}
    config["ok"] = (fit.backend == "numpyro" and fit.platform == "cpu" and fit.dtype == "float64" and fit.chain_method == "parallel" and fit.chains == 4
                    and fit.draws == 4000 and fit.tune == 2000 and fit.seed == 17 and float(fit.target_accept) == 0.9 and not fit.notes)
    phase("fit", fit_wall_seconds=config["fit_wall_seconds"], config_ok=config["ok"])
    nc = attempt_dir / "F7_numpyro_convergence_2000_4000.nc"
    fit.idata.to_netcdf(str(nc), engine="h5netcdf")
    draws = {"path": str(nc), "sha256": sha256(nc), "bytes": nc.stat().st_size, "groups": list(fit.idata.groups()),
             "posterior_variables": {k: list(v.shape) for k, v in fit.idata.posterior.data_vars.items()},
             "sample_stats_variables": list(fit.idata.sample_stats.data_vars), "written_at": now()}
    dump(attempt_dir / "draws_manifest.json", draws); phase("draws_saved", sha256=draws["sha256"], bytes=draws["bytes"])
    # -- diagnostics ------------------------------------------------------------------------------------------
    full = panel(fit.idata, max_treedepth=SETTINGS["max_treedepth"])
    pop = panel(fit.idata, max_treedepth=SETTINGS["max_treedepth"], var_names=POPULATION)
    def strict(p):
        finite = {k: bool(np.isfinite(p["values"][k])) for k in THRESHOLD_TEXT}
        return {"every_status_pass": all(p["status"][k] == "pass" for k in THRESHOLD_TEXT), "every_value_finite": all(finite.values()), "finite": finite,
                "statuses_not_pass": {k: p["status"][k] for k in THRESHOLD_TEXT if p["status"][k] != "pass"}}
    full_strict, pop_strict = strict(full), strict(pop)
    post = fit.idata.posterior
    per_parameter = {}
    rhat, essb, esst, mcse = az.rhat(post), az.ess(post, method="bulk"), az.ess(post, method="tail"), az.mcse(post, method="mean")
    for name in post.data_vars:
        per_parameter[name] = {"r_hat": np.asarray(rhat[name]).tolist(), "ess_bulk": np.asarray(essb[name]).tolist(), "ess_tail": np.asarray(esst[name]).tolist(),
                               "mcse_mean": np.asarray(mcse[name]).tolist(), "mean": np.asarray(post[name].mean(("chain", "draw"))).tolist()}
    dump(attempt_dir / "per_parameter.json", per_parameter)
    s = az.summary(fit.idata, var_names=POPULATION, hdi_prob=0.99, round_to="none")
    hdi = {name: {"mean": float(r["mean"]), "sd": float(r["sd"]), "hdi_0.5%": float(r["hdi_0.5%"]), "hdi_99.5%": float(r["hdi_99.5%"]), "mcse_mean": float(r["mcse_mean"]),
                  "ess_bulk": float(r["ess_bulk"]), "ess_tail": float(r["ess_tail"]), "r_hat": float(r["r_hat"]), "truth": TRUTH[name],
                  "truth_inside_99_hdi_observation": bool(r["hdi_0.5%"] <= TRUTH[name] <= r["hdi_99.5%"])} for name, r in s.iterrows()}
    dump(attempt_dir / "population_hdi99.json", hdi)
    record = ab.fit_record(m, fit, panel=full); dump(attempt_dir / "record.json", record)
    phase("diagnostics")
    # -- frozen reference agreement (means) and descriptive tail indicator -------------------------------------
    ref = json.loads(REFERENCE_RESULT.read_text())
    fulls = {k: v["means"] for k, v in ref["stages"].items() if k.endswith(":full")}
    ref_mean = dict(zip(SCALES, fulls["B1:r1:full"]))
    ref_spread = {n: max(m_[i] for m_ in fulls.values()) - min(m_[i] for m_ in fulls.values()) for i, n in enumerate(SCALES)}
    agreement = []
    for n in SCALES:
        diff = abs(hdi[n]["mean"] - ref_mean[n]); limit = 4 * hdi[n]["mcse_mean"] + ref_spread[n]
        agreement.append({"parameter": n, "new_mean": hdi[n]["mean"], "new_mcse": hdi[n]["mcse_mean"], "reference_mean": ref_mean[n], "reference_spread": ref_spread[n],
                          "abs_diff": diff, "diff_in_mcse": (diff / hdi[n]["mcse_mean"]) if hdi[n]["mcse_mean"] > 0 else None, "limit": limit, "within": bool(diff <= limit)})
    cdfs = {k: v for k, v in ref["cdfs"].items() if k.endswith(":cdf:0")}
    ref_cdf, ref_cdf_spread = ref["cdfs"]["B1:r1:cdf:0"], max(cdfs.values()) - min(cdfs.values())
    ind = (np.asarray(post["x|g_sigma"]) <= 0.30).astype(float)
    ds = xr.Dataset({"ind": (("chain", "draw"), ind)})
    ess_ind = float(np.asarray(az.ess(ds, method="mean")["ind"])); mcse_ind = float(np.asarray(az.mcse(ds, method="mean")["ind"]))
    p_hat = float(ind.mean()); below = ind.sum(axis=1).astype(int).tolist(); above = (ind.shape[1] - ind.sum(axis=1)).astype(int).tolist()
    informative = bool(math.isfinite(mcse_ind) and mcse_ind > 0 and math.isfinite(ess_ind) and 0 < p_hat < 1)
    tail = {"event": "I(x|g_sigma <= 0.30) on the retained draws", "p_hat": p_hat, "per_chain_below": below, "per_chain_above": above, "n_draws": int(ind.size),
            "mcse_indicator_mean_arviz_method_mean": mcse_ind, "ess_indicator_mean_arviz_method_mean": ess_ind, "informative": informative,
            "reference_cdf_B1_r1": ref_cdf, "reference_cdf_spread_four_boxes": ref_cdf_spread, "abs_diff_to_reference": abs(p_hat - ref_cdf),
            "diff_in_indicator_mcse": (abs(p_hat - ref_cdf) / mcse_ind) if informative else None,
            "descriptive_only": "not an acceptance gate; generic ess_tail(x|g_sigma) does not enter this estimate; nonfinite/zero MCSE or no variation would be uninformative"}
    prospective = bool(config["ok"] and full_strict["every_status_pass"] and full_strict["every_value_finite"])
    summary = {"unit_id": progress["unit_id"], "approval_sha256": ident["approval_sha256"], "script_sha256": ident["script_sha256"], "head": ident["git"]["head"],
               "started": progress["started"], "finished": now(), "child_wall_seconds": round(time.perf_counter() - t_start, 3), "configuration": config,
               "data": data_info, "draws": draws, "panel_full": {**full, "strict": full_strict}, "panel_population_only": {**pop, "strict": pop_strict},
               "thresholds": THRESHOLD_TEXT, "prospective_convergence_success_full_panel": prospective,
               "prospective_success_definition": "configuration as approved AND every full-panel status pass AND every value finite (n/a is not accepted through the overall aggregation)",
               "reference_mean_agreement": {"rows": agreement, "all_within": all(r["within"] for r in agreement), "rule": "|new_mean - reference| <= 4 * new MCSE + observed reference spread; an estimated diagnostic allowance, not a full-distribution error bound"},
               "tail_indicator_descriptive": tail, "population_hdi99_observations": hdi,
               "interpretation_limits": ["evidence for this configuration only; does not isolate the cause of the original 4x1000/1000 miss (two settings changed)",
                                         "not calibration, coverage or full-posterior correctness; truth-in-HDI rows are observations, not gates",
                                         "does not accept or relabel the original F7 test or its failed observation"]}
    dump(attempt_dir / "summary.json", summary)
    lines = [f"F7 NumPyro convergence diagnostic {summary['unit_id']} head {ident['git']['head'][:12]} child wall {summary['child_wall_seconds']} s fit {config['fit_wall_seconds']:.1f} s",
             f"configuration ok {config['ok']}: {config['chain_method']} {config['dtype']} chains {config['chains']} tune {config['tune']} draws {config['draws']} seed {config['seed']} notes {config['notes']}",
             f"draws {nc.name} sha256 {draws['sha256']} ({draws['bytes']} bytes)",
             "full panel: " + ", ".join(f"{k} {full['values'][k]:.6g} {full['status'][k]}" for k in THRESHOLD_TEXT) + f"; strict {full_strict['every_status_pass'] and full_strict['every_value_finite']}",
             "population panel: " + ", ".join(f"{k} {pop['values'][k]:.6g} {pop['status'][k]}" for k in THRESHOLD_TEXT),
             f"prospective convergence success (full panel, strict): {prospective}",
             "reference means: " + "; ".join(f"{r['parameter']} {r['new_mean']:.6f} vs {r['reference_mean']:.6f} diff {r['abs_diff']:.2e} = {r['diff_in_mcse']:.2f} MCSE within {r['within']}" for r in agreement),
             f"tail I(sd1<=0.30): p_hat {p_hat:.6f} per-chain below {below} above {above} mcse {mcse_ind:.3e} ess {ess_ind:.0f} informative {informative}; reference {ref_cdf:.7f} (spread {ref_cdf_spread:.1e})",
             "truth in 99% HDI (observations): " + ", ".join(f"{k} {v['truth_inside_99_hdi_observation']}" for k, v in hdi.items())]
    (attempt_dir / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); phase("finished", exit_code=0)
    return 0


def main(argv) -> int:
    if argv[:1] == ["--check"]:
        ident = identity(None); dump(UNIT / "precheck.json", ident)
        print(json.dumps({k: ident[k] for k in ("ok", "problems", "jax_cpu_devices", "cpu_count", "netcdf_engine")}, indent=1)); print("head", ident["git"]["head"], "dirty", ident["git"]["dirty"])
        return 0 if ident["ok"] else 2
    if argv[:1] == ["--run"] and len(argv) == 2:
        attempt_dir = Path(argv[1])
        try:
            return run(attempt_dir)
        except BaseException:
            (attempt_dir / "error.txt").write_text(traceback.format_exc()); traceback.print_exc(); return 1
    print(__doc__); return 2


if __name__ == "__main__":
    os.chdir(REPO); sys.path.insert(0, str(REPO))
    sys.exit(main(sys.argv[1:]))
