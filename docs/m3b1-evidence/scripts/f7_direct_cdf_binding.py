"""F7 binding for the direct truncated-box CDF reference (docs/m3b1-decision.md, binding implementation 2026-09-13).

Safety contract. Importing this module, ``--help``, ``--dry-run`` and every refusal import NO F7 target, model
library or sampler and perform ZERO evaluations. The target module (``tests.reference_marginal``) is imported only
inside ``build_f7_target``, which only the explicit ``--execute`` path reaches, after every identity check passed
and a NEW unit directory was created exclusively (``os.mkdir`` is atomic; an existing directory, ledger, restart or
duplicate refuses). A dry-run creates no directory and neither acquires nor writes a ledger; it reports the stage
inventory, planned cost, observed source and implementation hashes, the saved model identity, the git state and the
ACTUAL runtime environment (interpreter, NumPy, SciPy, BLAS/LAPACK; uv.lock pins declarations, not what is
installed) so that a reviewer can write an approval. No approval file ships with the repository and none is
generated here; there is no allow-dirty, override, takeover or retry option; the comparison with stored MCMC
summaries stays empty for a separate scientific review.

Usage (repository root, locked environment). The outer wrapper ``run_logged.py`` may enclose the command with a log
path OUTSIDE the unit directory; this script never imports that wrapper and writes its own log to ``--log-dir``:
  python docs/m3b1-evidence/scripts/f7_direct_cdf_binding.py --protocol docs/m3b1-evidence/direct_cdf_v3r1_protocol.json
      --log-dir EXISTING_DIR --dry-run [--approval FILE --unit-dir DIR --attempt NAME]
  ... --log-dir EXISTING_DIR --execute --approval FILE --unit-dir NEW_DIR --attempt NAME   (future, separately reviewed)

Exit codes: 0 dry-run report produced, or every protocol condition met on a valid run; 2 refused before any
evaluation (also argparse usage errors); 3 unit stopped (including the failed counted anchor preflight), invalid or
over-budget accounting; 4 a protocol condition failed on a valid run; 1 an unexpected error (logged with traceback).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT_REL = "docs/m3b1-evidence/scripts/f7_direct_cdf_binding.py"
RECORDS_REL = "docs/m3b1-evidence/F7_varying_slope_recovery_records.json"
IMPLEMENTATION_PINS = ("tests/reference_direct_cdf.py", "tests/quadrature_budget.py", "tests/reference_marginal.py", SCRIPT_REL)
TARGET = {
    "callable": "tests.reference_marginal.GaussianSlopeMarginal.log_posterior_theta",
    "generator": "tests.reference_marginal.f7_data",
    "seed": 17, "J": 30, "per": 20,
    "prior_mean": [0.0, 0.0], "prior_sd": [5.0, 2.0], "center": True, "prior_scales": [1.0, 1.0, 1.0],
    "theta": "(log sd0, log sd1, log sigma); HalfNormal(1) priors on the natural scales and the three log-scale Jacobians included",
}
MODEL_IDENTITY = {
    "formula": "y ~ x + (1 + x || g)", "canonical_formula": "y ~ x + (1 + x || g)", "family": "gaussian", "link": "identity",
    "n": 600, "J": 30, "group": "g", "center_predictors": True, "noncentered": True, "varying_slope": "x",
    "group_effects": ["Intercept", "x"],
    "priors": {"intercept": "Normal(mu=0, sigma=5)", "b:x": "Normal(mu=0, sigma=2)", "sd:Intercept": "HalfNormal(sigma=1)",
               "sd:x": "HalfNormal(sigma=1)", "sigma": "HalfNormal(sigma=1)"},
}
ENVIRONMENT_KEYS = ("python_version", "python_implementation", "system", "machine", "numpy_version", "scipy_version",
                    "blas", "lapack", "executable")
APPROVAL_KEYS = ("approval_id", "approves_execution", "reviewer", "date", "unit_id", "attempt", "protocol_id", "protocol_sha256",
                 "planned_evaluations", "cap_evaluations", "unit_deadline_seconds", "max_batch_points", "git_head", "source_sha256",
                 "target", "model_identity", "expected_environment")
APPROVAL_CHECKS = ("approval_schema", "approval_protocol_identity", "approval_counts", "approval_source_pins", "approval_target",
                   "approval_model_identity", "approval_environment", "approval_git_head", "git_clean", "approval_unit_binding")
EXIT_REFUSED = 2


# -- small utilities (no target access) ------------------------------------------------------------------------
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    tmp.replace(path)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def observed_environment() -> dict:
    """The ACTUAL interpreter and numerical libraries of this process (NumPy and SciPy are not target modules)."""
    import numpy
    import scipy

    blas = lapack = "unavailable"
    try:
        deps = numpy.show_config(mode="dicts").get("Build Dependencies", {})
        blas = deps.get("blas", {}).get("name", "unavailable")
        lapack = deps.get("lapack", {}).get("name", "unavailable")
    except Exception:  # pragma: no cover - older NumPy without dict output
        pass
    return {"python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(), "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__, "blas": blas, "lapack": lapack, "executable": sys.executable}


def probe_git() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return {"head": None, "dirty": True, "status": f"git unavailable: {error}"}
    return {"head": head, "dirty": bool(status.strip()), "status": status}


def plan_inventory(plan: dict) -> dict:
    from tests.reference_direct_cdf import inventory  # generic injected-evaluator helper; imports no target

    inv = inventory(plan)
    return {"stages": len(inv), "planned_evaluations": int(sum(s["evaluations"] for s in inv.values())), "first_stage": next(iter(inv))}


# -- the only place that imports the F7 target ------------------------------------------------------------------
def build_f7_target(spec: dict):
    """Lazy F7 setup for the explicit --execute path only: frozen generator, analytic Gaussian marginal, log-scale
    posterior with priors and Jacobians. Constructing it generates the seed-17 data and performs no evaluation."""
    import numpy as np

    from tests.reference_marginal import GaussianSlopeMarginal, f7_data  # target import: execution path only

    y, x, g = f7_data(seed=spec["seed"], J=spec["J"], per=spec["per"])
    reference = GaussianSlopeMarginal(y, x, g, prior_mean=tuple(spec["prior_mean"]), prior_sd=tuple(spec["prior_sd"]), center=spec["center"])
    scales = tuple(spec["prior_scales"])

    def evaluate(points):
        return reference.log_posterior_theta(np.asarray(points, dtype=float), prior_scales=scales)

    return evaluate


# -- identity checks: pure reads, nothing created, no target ---------------------------------------------------------
def run_checks(protocol_path, approval_path, unit_dir, attempt, git: dict, environment: dict):
    checks, facts = {}, {}

    def add(name, ok, detail=""):
        checks[name] = {"ok": ok, "detail": detail}

    try:
        plan = json.loads(Path(protocol_path).read_text())
        facts["protocol"] = {"path": str(protocol_path), "sha256": sha256(protocol_path), "protocol_id": plan.get("protocol_id")}
        inv = plan_inventory(plan)
        facts["protocol"].update(inv)
        add("protocol_inventory", True, f"{inv['stages']} stages, {inv['planned_evaluations']} planned evaluations, first stage {inv['first_stage']}")
    except Exception as error:
        add("protocol_inventory", False, f"{type(error).__name__}: {error}")
        return checks, facts, None
    planned = inv["planned_evaluations"]
    add("planned_matches_protocol", _is_int(plan.get("planned_evaluations")) and plan["planned_evaluations"] == planned,
        f"protocol planned_evaluations {plan.get('planned_evaluations')!r} vs inventory {planned}")
    cap, deadline, batch = plan.get("proposed_unit_cap"), plan.get("proposed_wall_seconds"), plan.get("max_batch_points")
    add("protocol_caps", _is_int(cap) and cap >= planned and _is_number(deadline) and deadline > 0 and _is_int(batch) and batch >= 1,
        f"cap {cap!r} >= planned {planned}; allowance {deadline!r} s; batch {batch!r}")
    facts["protocol"].update(cap_evaluations=cap, unit_deadline_seconds=deadline, max_batch_points=batch)
    add("protocol_first_stage_is_anchor", inv["first_stage"] == "anchor:reference", inv["first_stage"])
    pins = plan.get("source_sha256") if isinstance(plan.get("source_sha256"), dict) else {}
    observed, bad = {}, []
    for rel, expected in pins.items():
        path = REPO / rel
        observed[rel] = sha256(path) if path.is_file() else None
        if observed[rel] != expected:
            bad.append(rel)
    facts["protocol_source_sha256_observed"] = observed
    add("protocol_source_pins", isinstance(plan.get("source_sha256"), dict) and not bad, "all pinned files match" if not bad else f"mismatch or missing: {bad}")
    impl = {rel: (sha256(REPO / rel) if (REPO / rel).is_file() else None) for rel in IMPLEMENTATION_PINS}
    facts["implementation_sha256"] = impl
    add("implementation_files_present", all(impl.values()), "all present" if all(impl.values()) else f"missing: {[k for k, v in impl.items() if v is None]}")
    try:
        records = json.loads((REPO / RECORDS_REL).read_text())
        subset = {be: {k: records[be]["model"].get(k) for k in MODEL_IDENTITY} for be in ("pymc", "numpyro")}
        same = all(subset[be] == MODEL_IDENTITY for be in subset)
        add("records_model_identity", same, "both saved backend records match the binding's model identity" if same else f"mismatch: {subset}")
    except Exception as error:
        add("records_model_identity", False, f"{type(error).__name__}: {error}")
    facts.update(git=git, environment=environment, target=TARGET, model_identity=MODEL_IDENTITY)
    if unit_dir is not None:
        unit_dir = Path(unit_dir)
        fresh = unit_dir.parent.is_dir() and not unit_dir.exists() and not unit_dir.is_symlink()
        add("unit_dir_new", fresh, f"{unit_dir}: parent exists and the directory is absent" if fresh else f"{unit_dir} exists, or its parent does not")
    if approval_path is None:
        for name in APPROVAL_CHECKS:
            add(name, None, "no approval supplied; a real execution would be refused")
        return checks, facts, plan
    try:
        approval = json.loads(Path(approval_path).read_text())
        if not isinstance(approval, dict):
            raise ValueError("approval must be a JSON object")
    except Exception as error:
        add("approval_schema", False, f"{type(error).__name__}: {error}")
        for name in APPROVAL_CHECKS[1:]:
            add(name, None, "approval unreadable")
        return checks, facts, plan
    facts["approval"] = {"path": str(approval_path), "sha256": sha256(approval_path), "approval_id": approval.get("approval_id")}
    missing = [k for k in APPROVAL_KEYS if k not in approval]
    add("approval_schema", not missing and approval.get("approves_execution") is True,
        "complete; approves_execution is true" if not missing and approval.get("approves_execution") is True else f"missing {missing}; approves_execution={approval.get('approves_execution')!r}")
    add("approval_protocol_identity", approval.get("protocol_sha256") == facts["protocol"]["sha256"] and approval.get("protocol_id") == plan.get("protocol_id"),
        f"approval protocol {approval.get('protocol_id')!r} {str(approval.get('protocol_sha256'))[:16]} vs file {plan.get('protocol_id')!r} {facts['protocol']['sha256'][:16]}")
    counts_ok = (_is_int(approval.get("planned_evaluations")) and approval["planned_evaluations"] == planned
                 and _is_int(approval.get("cap_evaluations")) and approval["cap_evaluations"] == cap
                 and _is_number(approval.get("unit_deadline_seconds")) and approval["unit_deadline_seconds"] == deadline
                 and _is_int(approval.get("max_batch_points")) and approval["max_batch_points"] == batch)
    add("approval_counts", counts_ok, f"approval planned/cap/allowance/batch {approval.get('planned_evaluations')!r}/{approval.get('cap_evaluations')!r}/"
        f"{approval.get('unit_deadline_seconds')!r}/{approval.get('max_batch_points')!r} vs protocol {planned}/{cap!r}/{deadline!r}/{batch!r}")
    approved_pins = approval.get("source_sha256") if isinstance(approval.get("source_sha256"), dict) else {}
    required = sorted(set(pins) | set(IMPLEMENTATION_PINS))
    missing_pins = [rel for rel in required if rel not in approved_pins]
    wrong = [rel for rel in approved_pins if (sha256(REPO / rel) if (REPO / rel).is_file() else None) != approved_pins[rel]]
    add("approval_source_pins", not missing_pins and not wrong,
        "every protocol and implementation file is pinned and matches" if not missing_pins and not wrong else f"missing {missing_pins}; mismatched {wrong}")
    add("approval_target", approval.get("target") == TARGET, "matches the binding's target" if approval.get("target") == TARGET else "target differs from the binding")
    add("approval_model_identity", approval.get("model_identity") == MODEL_IDENTITY,
        "matches" if approval.get("model_identity") == MODEL_IDENTITY else "model identity differs from the binding")
    expected = approval.get("expected_environment") if isinstance(approval.get("expected_environment"), dict) else {}
    diff = {k: {"expected": expected.get(k), "observed": environment.get(k)} for k in ENVIRONMENT_KEYS if expected.get(k) != environment.get(k)}
    add("approval_environment", isinstance(approval.get("expected_environment"), dict) and not diff, "observed environment matches the reviewed one" if not diff else f"differences: {diff}")
    add("approval_git_head", isinstance(git.get("head"), str) and approval.get("git_head") == git.get("head"),
        f"approval head {str(approval.get('git_head'))[:12]} vs checkout {str(git.get('head'))[:12]}")
    add("git_clean", git.get("dirty") is False, "clean checkout" if git.get("dirty") is False else f"dirty or unknown checkout: {git.get('status')!r}")
    bound = unit_dir is not None and attempt is not None and approval.get("unit_id") == Path(unit_dir).name and approval.get("attempt") == attempt
    add("approval_unit_binding", bound, f"approval unit {approval.get('unit_id')!r} attempt {approval.get('attempt')!r} vs requested "
        f"{None if unit_dir is None else Path(unit_dir).name!r} {attempt!r}")
    return checks, facts, plan


# -- execution path (never reached by --dry-run or a refusal) --------------------------------------------------------
def summary_text(out: dict) -> str:
    b, acc = out["budget"], out["stage_acceptance"]
    lines = [f"protocol {out['protocol_id']} plan_sha256 {out['plan_sha256']}",
             f"status {out['status']} exit_code {out['exit_code']} stopped {out['stopped']!r}",
             f"evaluations requested {b['requested']} confirmed {b['confirmed']} reserved_uncertain {b['reserved_uncertain']} "
             f"planned {out['planned_evaluations']} cap {b['cap']}; {b['seconds_spent']:.1f} s of {b['unit_deadline_seconds']} s",
             f"stages recorded {len(out['stages'])}; conditions met {len(acc['met'])} failed {len(acc['failed'])} unchecked {len(acc['unchecked'])} of {acc['n_required']}",
             f"run_validity {out['run_validity']}"]
    anchor = out["stages"].get("anchor:reference")
    if anchor:
        lines.append(f"anchor ell {anchor['ell_anchor']!r} abs_error {anchor['abs_error']!r}")
    if acc["failed"]:
        lines.append("failed: " + ", ".join(acc["failed"]))
    lines.append("comparison with stored MCMC summaries: empty (separate scientific review); no F7 acceptance, tail bound, calibration or convergence claim")
    return "\n".join(lines) + "\n"


def execute(report: dict, plan: dict, unit_dir: Path, attempt: str, target_factory) -> int:
    """The only path that creates files in a unit, touches a ledger or builds the target. The NEW unit directory is
    created first and exclusively; a failure after that point consumes the unit (no restart, retry or second allowance)."""
    try:
        os.mkdir(unit_dir)  # atomic: any existing directory (previous unit, ledger, restart, duplicate) refuses
    except FileExistsError:
        report["refusal"] = f"unit directory {unit_dir} already exists; a restarted or duplicated unit is refused"
        return EXIT_REFUSED
    record = dict(report, phase="unit_created", unit_dir=str(unit_dir), attempt=attempt, unit_created_at=utc_now())
    write_json_atomic(unit_dir / "binding_record.json", record)
    try:
        evaluate = target_factory(TARGET)  # the lazy F7 import happens inside build_f7_target; no evaluation yet
    except BaseException as error:
        record.update(phase="target_build_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json_atomic(unit_dir / "binding_record.json", record)
        report.update(phase="target_build_failed", refusal=f"target build failed after the unit directory was created; the unit is consumed: {record['error']}")
        return EXIT_REFUSED
    from tests.quadrature_budget import BudgetedEvaluator
    from tests.reference_direct_cdf import run_protocol

    batch = int(plan["max_batch_points"])
    evaluator = BudgetedEvaluator(evaluate, cap_evaluations=int(plan["proposed_unit_cap"]), unit_deadline_seconds=float(plan["proposed_wall_seconds"]),
                                  ledger_path=unit_dir / "ledger.json", attempt=attempt, batch_size=batch)  # takeover stays False; no override exists

    def checkpoint(stages, budget):
        write_json_atomic(unit_dir / "partial.json", {"written_at": utc_now(), "budget": budget, "stages": stages})

    record.update(phase="running", run_started=utc_now())
    write_json_atomic(unit_dir / "binding_record.json", record)
    output = run_protocol(evaluator, plan, checkpoint=checkpoint, batch_size=batch)
    output["binding"] = {k: record.get(k) for k in ("argv", "started", "run_started", "git", "environment", "implementation_sha256",
                                                     "protocol", "protocol_source_sha256_observed", "approval", "target", "model_identity", "unit_dir", "attempt")}
    output["binding"]["finished"] = utc_now()
    write_json_atomic(unit_dir / "result.json", output)
    (unit_dir / "summary.txt").write_text(summary_text(output))
    record.update(phase="finished", exit_code=int(output["exit_code"]), status=output["status"], stopped=output["stopped"], budget=output["budget"])
    write_json_atomic(unit_dir / "binding_record.json", record)
    report.update(phase="finished", status=output["status"], stopped=output["stopped"], budget=output["budget"])
    return int(output["exit_code"])


# -- entrypoint ------------------------------------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(prog="f7_direct_cdf_binding.py",
                                     description="F7 binding for the direct-CDF reference: dry-run identity report or an explicitly approved execution. "
                                                 "Import, --help, --dry-run and refusals never import the F7 target and never evaluate it.")
    parser.add_argument("--protocol", required=True, help="protocol JSON (e.g. docs/m3b1-evidence/direct_cdf_v3r1_protocol.json)")
    parser.add_argument("--log-dir", required=True, help="EXISTING directory outside any unit; every invocation writes its log and real exit code there")
    parser.add_argument("--approval", help="reviewed approval JSON (required for --execute; none ships with the repository)")
    parser.add_argument("--unit-dir", help="NEW unit directory to create exclusively (required for --execute)")
    parser.add_argument("--attempt", help="attempt name recorded in the new ledger (required for --execute; must match the approval)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="identity report only: creates no directory, touches no ledger, evaluates nothing")
    mode.add_argument("--execute", action="store_true", help="create the new unit and run the protocol; refused unless every identity check passes")
    return parser.parse_args(argv)


def main(argv=None, *, target_factory=None, git_state=None, environment=None) -> int:
    """Keyword injections exist only so stub tests can exercise the execution path without the F7 target; the
    command line never sets them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    started = utc_now()
    mode = "execute" if args.execute else "dry-run"
    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"refused: --log-dir {log_dir} must be an existing directory (outside the unit)", file=sys.stderr)
        return EXIT_REFUSED
    report = {"mode": mode, "argv": argv, "started": started, "repo": str(REPO), "pid": os.getpid(), "refusal": None, "phase": "checking"}
    code = EXIT_REFUSED
    try:
        git = (git_state or probe_git)()
        env = (environment or observed_environment)()
        checks, facts, plan = run_checks(args.protocol, args.approval, args.unit_dir, args.attempt, git, env)
        report.update(facts)
        report["checks"] = checks
        failed = sorted(k for k, v in checks.items() if v["ok"] is not None and v["ok"] is not True)
        not_performed = sorted(k for k, v in checks.items() if v["ok"] is None)
        report.update(failed_checks=failed, unperformed_checks=not_performed)
        if args.execute:
            missing = [flag for flag, value in (("--approval", args.approval), ("--unit-dir", args.unit_dir), ("--attempt", args.attempt)) if value is None]
            if missing:
                report["refusal"] = f"--execute requires {', '.join(missing)}"
            elif failed or not_performed:
                report["refusal"] = f"identity checks failed or not performed: {failed + not_performed}"
            else:
                code = execute(report, plan, Path(args.unit_dir), args.attempt, target_factory or build_f7_target)
        else:
            report["would_execute"] = args.approval is not None and not failed and not not_performed
            report["refusal"] = None if not failed else f"a real execution would be refused: {failed}"
            code = EXIT_REFUSED if failed else 0
    except BaseException:  # logged with the traceback; nothing is retried
        report["error"] = traceback.format_exc()
        code = 1
    finally:
        report.update(finished=utc_now(), exit_code=code)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = log_dir / f"binding-{stamp}-{mode}-{os.getpid()}.json"
        write_json_atomic(log_path, report)
        report["log_path"] = str(log_path)
        brief = {k: report.get(k) for k in ("mode", "exit_code", "refusal", "failed_checks", "unperformed_checks", "would_execute", "phase", "status", "stopped", "log_path")}
        brief["protocol"] = report.get("protocol")
        brief["environment"] = report.get("environment")
        brief["git"] = {k: report.get("git", {}).get(k) for k in ("head", "dirty")}
        print(json.dumps(brief, indent=1, default=_json_default))
    return code


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())
