"""M3c-1 bounded two-fit validation (coordinator dispatch, 2026-09-19): EXACTLY two ``model.fit`` calls on the committed
example's frozen Gaussian dose-by-arm population-only model, PyMC CPU (cores=4) then NumPyro/JAX CPU (parallel chains),
identical data, 4 chains, 1000 tune, 1000 draws, seed 29, target_accept 0.9, max tree depth 10, float64.

Usage: m3c1_two_fit_validation.py OUT_DIR   (invoked once by m3c1_two_fit_supervisor.py inside its 600 s deadline)

Sequence: identity (head, hashes, environment) -> frozen data, model, ordered labels written to freeze.json BEFORE any
sampling -> fit 1 -> its InferenceData to netCDF immediately with SHA-256 -> fit 2 -> netCDF -> predeclared gates,
labelled between-backend comparison (4 combined mean MCSE), descriptive 99 % HDI membership of the example's truths,
prediction check on the example's four new rows without a grouping column, fit-record fields. Exit 0 when every gate
passes, 4 when the run completed with a failed gate, 1 on an unexpected error. No retry, no configuration change.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
SETTINGS = {"chains": 4, "draws": 1000, "tune": 1000, "seed": 29, "target_accept": 0.9, "max_treedepth": 10, "device": "cpu", "dtype": "float64"}
BACKENDS = [("pymc", {"cores": 4}), ("numpyro", {})]                       # in this order; exactly these two fits
LABELS = ["Intercept", "dose", "arm[high]", "arm[low]", "dose:arm[high]", "dose:arm[low]", "sigma"]   # frozen ordered labels: sorted non-reference levels
COLUMNS = ("dose", "arm[high]", "arm[low]", "dose:arm[high]", "dose:arm[low]")
PINS = ["examples/m3c1_population_only.py", "autobayes/__init__.py", "autobayes/formula.py", "autobayes/priors.py", "autobayes/model.py",
        "autobayes/predict.py", "autobayes/inference.py", "autobayes/session.py", "autobayes/diagnostics.py", "autobayes/likelihoods.py",
        "uv.lock", "pyproject.toml", "docs/m3c1-evidence/scripts/m3c1_two_fit_validation.py", "docs/m3c1-evidence/scripts/m3c1_two_fit_supervisor.py"]
THRESHOLDS = ("divergence_fraction", "max_rhat", "min_ess_bulk", "min_ess_tail", "min_bfmi", "treedepth_saturation")
NEW_ROWS = {"dose": [0.0, 1.0, 0.0, 1.0], "arm": ["high", "high", "control", "low"]}   # the example's four new rows, no grouping column


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def main(out: Path) -> int:
    t0 = time.perf_counter()
    progress = {"started": now(), "pid": os.getpid(), "phases": []}

    def dump(name, obj):
        (out / name).write_text(json.dumps(obj, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")

    def phase(name, **extra):
        progress["phases"].append({"phase": name, "at": now(), "elapsed_s": round(time.perf_counter() - t0, 3), **extra}); dump("progress.json", progress)

    import autobayes as ab                                                  # before any JAX touch: parallel CPU chains
    import arviz as az
    import numpy as np
    import pandas as pd
    from autobayes.diagnostics import panel
    from examples.m3c1_population_only import DOSE_MEAN, TRUTH, simulate   # the committed example: data, truth and priors

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    df = simulate(rows=240, seed=29)
    data_sha = hashlib.sha256(np.ascontiguousarray(df.y.to_numpy()).tobytes() + np.ascontiguousarray(df.dose.to_numpy()).tobytes()
                              + "\n".join(df.arm).encode()).hexdigest()
    priors = ab.Priors(intercept=ab.Normal(0, 5), b={"dose": ab.Normal(0, 2), "arm": ab.Normal(0, 2), "dose:arm": ab.Normal(0, 1)}, sigma=ab.HalfNormal(1))
    model = ab.gaussian("y ~ dose + arm + dose:arm", df, priors, categorical={"arm": "control"})
    if model.spec.columns != COLUMNS or model.spec.group is not None or not model.center_predictors or model.spec.categorical("arm").reference != "control":
        raise RuntimeError(f"model identity differs from the frozen expectation: {model.spec.columns} group={model.spec.group}")
    truth = {"Intercept": TRUTH["a"], "dose": TRUTH["b"], "arm[low]": TRUTH["low"], "arm[high]": TRUTH["high"],
             "dose:arm[low]": TRUTH["dlow"], "dose:arm[high]": TRUTH["dhigh"], "sigma": TRUTH["sigma"]}
    freeze = {"frozen_at": now(), "head": head, "dirty": dirty, "settings": SETTINGS, "backends_in_order": [b for b, _ in BACKENDS], "pymc_cores": 4,
              "labels": LABELS, "columns": list(COLUMNS), "truth_by_label": truth, "truth_note": "Intercept is the raw intercept at dose = 0 in the control arm",
              "data": {"generator": "examples.m3c1_population_only.simulate(rows=240, seed=29)", "n": int(len(df)), "sha256_y_dose_arm": data_sha,
                       "sha256_csv_17g": hashlib.sha256(df.to_csv(index=False, float_format="%.17g").encode()).hexdigest(),
                       "arm_counts": df.arm.value_counts().to_dict(), "dose_mean_by_arm": df.groupby("arm").dose.mean().round(6).to_dict(), "dose_mean_truth": DOSE_MEAN},
              "priors": {"intercept": str(priors.intercept), **{f"b:{k}": str(v) for k, v in priors.coefficient_priors(model.spec).items()}, "sigma": str(priors.sigma)},
              "formula": model.spec.formula, "reference": "control", "center_predictors": True, "coords": {k: [str(c) for c in v] for k, v in model.pymc_model.coords.items() if k != "__obs__"},
              "value_vars": sorted(v.name for v in model.pymc_model.value_vars), "describe": model.describe(),
              "source_sha256": {rel: sha(REPO / rel) for rel in PINS}, "environment": ab.environment(), "new_rows": NEW_ROWS}
    dump("freeze.json", freeze); phase("frozen_before_sampling")

    def gates(idata):
        post, ss = idata.posterior, idata.sample_stats
        finite_post = bool(all(np.isfinite(np.asarray(v)).all() for v in post.data_vars.values()))
        finite_ss = bool(all(np.isfinite(np.asarray(ss[k])).all() for k in ss.data_vars if np.issubdtype(ss[k].dtype, np.number)))
        p = panel(idata, max_treedepth=SETTINGS["max_treedepth"])
        finite_vals = {k: bool(np.isfinite(p["values"][k])) for k in THRESHOLDS}
        strict = all(p["status"][k] == "pass" for k in THRESHOLDS) and all(finite_vals.values())
        return {"panel": p, "finite_posterior": finite_post, "finite_sample_stats": finite_ss, "every_status_pass_and_finite": strict,
                "statuses_not_pass": {k: p["status"][k] for k in THRESHOLDS if p["status"][k] != "pass"}, "pass": bool(strict and finite_post and finite_ss),
                "posterior_variables": {k: list(v.shape) for k, v in post.data_vars.items()}, "sample_stats_variables": list(ss.data_vars)}

    fits, records, infos = {}, {}, {}
    for i, (backend, extra) in enumerate(BACKENDS, 1):
        phase(f"fit_{i}_{backend}_start")
        fit = model.fit(backend=backend, device="cpu", chains=SETTINGS["chains"], draws=SETTINGS["draws"], tune=SETTINGS["tune"],
                        seed=SETTINGS["seed"], target_accept=SETTINGS["target_accept"], **extra)
        nc = out / f"m3c1_two_fit_{i}_{backend}.nc"
        fit.idata.to_netcdf(str(nc), engine="h5netcdf")                       # immediately, before any diagnostic
        phase(f"fit_{i}_{backend}_saved", fit_wall_seconds=float(fit.wall_seconds), netcdf_sha256=sha(nc), bytes=nc.stat().st_size)
        rec = ab.fit_record(model, fit)
        actual_ok = (fit.backend == backend and fit.platform == "cpu" and fit.dtype == "float64" and fit.chains == 4 and fit.draws == 1000 and fit.tune == 1000
                     and fit.seed == 29 and float(fit.target_accept) == 0.9 and not fit.notes
                     and (fit.chain_method == "parallel" if backend == "numpyro" else fit.chain_method is None))
        g = gates(fit.idata)
        infos[backend] = {"backend": backend, "order": i, "report": fit.report(), "execution": rec["execution"], "requested_extra": extra,
                          "actual_configuration_ok": actual_ok, "netcdf": str(nc), "netcdf_sha256": sha(nc), "netcdf_bytes": nc.stat().st_size, "gates": g,
                          "record_group_null_J0": rec["model"]["group"] is None and rec["model"]["J"] == 0, "record": rec}
        dump(f"fit_{i}_{backend}.json", infos[backend])
        fits[backend], records[backend] = fit, rec
        phase(f"fit_{i}_{backend}_diagnosed", gates_pass=g["pass"], actual_ok=actual_ok)

    comparison = ab.compare_records(records["pymc"], records["numpyro"])       # labelled rows, 4 * hypot(mcse_a, mcse_b)
    rows_ok = set(comparison["rows"]) == set(LABELS)
    hdi, predict_checks = {}, {}
    new = pd.DataFrame(NEW_ROWS)
    for backend, fit in fits.items():
        s = az.summary(fit.idata, var_names=["Intercept", "dose", "arm", "dose:arm", "sigma"], hdi_prob=0.99, round_to="none")
        hdi[backend] = {name: {"mean": float(r["mean"]), "hdi_0.5%": float(r["hdi_0.5%"]), "hdi_99.5%": float(r["hdi_99.5%"]), "truth": truth[name],
                               "truth_inside_99_hdi_descriptive": bool(r["hdi_0.5%"] <= truth[name] <= r["hdi_99.5%"])} for name, r in s.iterrows()}
        post = fit.idata.posterior
        out_ds = model.predict(fit, newdata=new, kind="mean")
        a, b = post["Intercept"].to_numpy(), post["dose"].to_numpy()
        hand = []
        for dose, arm in zip(NEW_ROWS["dose"], NEW_ROWS["arm"]):
            mu = a + b * dose
            if arm != "control":
                mu = mu + post["arm"].sel(arm_dim=arm).to_numpy() + post["dose:arm"].sel({"dose:arm_dim": arm}).to_numpy() * dose
            hand.append(mu)
        hand = np.stack(hand, axis=-1)
        predict_checks[backend] = {"max_abs_diff": float(np.max(np.abs(out_ds["mu"].to_numpy() - hand))), "new_groups_attr": out_ds.attrs.get("new_groups"),
                                   "new_levels_attr": out_ds.attrs.get("new_levels"), "expected_y_means": np.round(out_ds["mu"].mean(("chain", "draw")).to_numpy(), 6).tolist(),
                                   "ok": bool(np.allclose(out_ds["mu"].to_numpy(), hand, atol=1e-10) and out_ds.attrs.get("new_groups") == "not_applicable")}
    result = {"unit": "m3c1-two-fit-validation", "head": head, "dirty": dirty, "started": progress["started"], "finished": now(),
              "total_child_seconds": round(time.perf_counter() - t0, 3), "settings": SETTINGS, "backends_in_order": [b for b, _ in BACKENDS],
              "fits": {b: {k: v for k, v in infos[b].items() if k != "record"} for b in infos}, "records": {b: records[b] for b in records},
              "comparison_4_combined_mcse": comparison, "comparison_rows_are_the_seven_labels": rows_ok,
              "hdi99_descriptive": hdi, "predict_new_rows_check": predict_checks,
              "gates": {"pymc_diagnostics": infos["pymc"]["gates"]["pass"], "numpyro_diagnostics": infos["numpyro"]["gates"]["pass"],
                        "pymc_configuration": infos["pymc"]["actual_configuration_ok"], "numpyro_configuration": infos["numpyro"]["actual_configuration_ok"],
                        "backends_agree_7_labels": bool(comparison["agree"] and rows_ok), "predict_pymc": predict_checks["pymc"]["ok"], "predict_numpyro": predict_checks["numpyro"]["ok"],
                        "records_group_null_J0": bool(infos["pymc"]["record_group_null_J0"] and infos["numpyro"]["record_group_null_J0"])}}
    result["all_gates_pass"] = all(result["gates"].values())
    result["interpretation"] = ("both CPU routes met every predeclared diagnostic gate on this data set and agree in the seven labelled means within 4 "
                                "combined MCSE; one realisation, not calibration; truth-in-HDI rows are descriptive only" if result["all_gates_pass"]
                                else "a predeclared gate failed; no repair fit, no changed threshold; see gates")
    dump("result.json", result)
    lines = [f"M3c-1 two-fit validation head {head[:12]} dirty {dirty}; child {result['total_child_seconds']} s; all gates pass: {result['all_gates_pass']}"]
    for b in infos:
        p = infos[b]["gates"]["panel"]; ex = infos[b]["execution"]
        lines.append(f"{b}: {ex['device_actual']} {ex['dtype']} chain_method {ex['chain_method']} chains {ex['chains']} tune {ex['tune']} draws {ex['draws']} seed {ex['seed']} "
                     f"wall {ex['wall_seconds']} s; " + ", ".join(f"{k} {p['values'][k]:.6g} {p['status'][k]}" for k in THRESHOLDS) + f"; finite {infos[b]['gates']['finite_posterior'] and infos[b]['gates']['finite_sample_stats']}; netcdf {infos[b]['netcdf_sha256'][:16]}")
    lines.append("means pymc vs numpyro (|diff| / 4*combined MCSE): " + "; ".join(f"{k} {r['mean_a']:.4f} vs {r['mean_b']:.4f} {r['diff']:.2e}/{r['tol']:.2e} {'ok' if r['agree'] else 'DISAGREE'}" for k, r in comparison["rows"].items()))
    lines.append("truth in 99% HDI (descriptive): " + "; ".join(f"{b}: " + ", ".join(f"{k} {v['truth_inside_99_hdi_descriptive']}" for k, v in hdi[b].items()) for b in hdi))
    lines.append("predict new rows vs hand: " + "; ".join(f"{b} max|diff| {predict_checks[b]['max_abs_diff']:.1e} attr {predict_checks[b]['new_groups_attr']} ok {predict_checks[b]['ok']}" for b in predict_checks))
    (out / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    phase("finished", all_gates_pass=result["all_gates_pass"])
    return 0 if result["all_gates_pass"] else 4


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    try:
        sys.exit(main(out_dir))
    except BaseException:
        (out_dir / "error.txt").write_text(traceback.format_exc()); traceback.print_exc(); sys.exit(1)
