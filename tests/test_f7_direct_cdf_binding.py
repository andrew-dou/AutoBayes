"""Guard/stub tests for docs/m3b1-evidence/scripts/f7_direct_cdf_binding.py (binding implementation, 2026-09-13).
The binding module is imported without running; no F7 target, model library or sampler is imported; the --execute
path is exercised only with an injected stub evaluator on a tiny synthetic protocol; approval-like data are
synthetic, written under tmp_path, routed to stubs only and never committed."""
import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests import reference_direct_cdf as q

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "docs/m3b1-evidence/scripts/f7_direct_cdf_binding.py"
WRAPPER = ROOT / "docs/m3b1-evidence/scripts/run_logged.py"
V3R1 = ROOT / "docs/m3b1-evidence/direct_cdf_v3r1_protocol.json"
TARGET_ROOTS = {"autobayes", "bambi", "pymc", "numpyro"}
GIT = {"head": "0123456789abcdef" * 2 + "01234567", "dirty": False, "status": ""}


def target_modules_loaded():
    return sorted(m for m in sys.modules if m.split(".")[0] in TARGET_ROOTS or m == "tests.reference_marginal")


@pytest.fixture(scope="module")
def binding():
    before = target_modules_loaded()
    spec = importlib.util.spec_from_file_location("f7_direct_cdf_binding", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)          # import only: main() is guarded
    assert target_modules_loaded() == before
    return mod


def clean_git():
    return dict(GIT)


def toy_protocol(tmp_path):
    plan = dict(protocol_id="toy-binding-only", domains={"B0": [[-1, 1]] * 3, "B1": [[-1.1, 1.1]] * 3},
                nodes_per_axis=[[2, 3, 2], [3, 4, 3]], cdf_points=[dict(scale="sigma", label="zero", value=1.0)],
                face_nodes=2, face_mode_anchor=[0, 0, 0], fixed_ell_reference=0.0, anchor_reference_tolerance=1e-6, source_sha256={})
    planned = sum(s["evaluations"] for s in q.inventory(plan).values())
    plan.update(planned_evaluations=planned, proposed_unit_cap=planned + 1, proposed_wall_seconds=60, max_batch_points=64)
    path = tmp_path / "toy_protocol.json"; path.write_text(json.dumps(plan, indent=1))
    return path, plan


def synthetic_approval(binding, protocol_path, unit_dir, attempt="stub-attempt"):
    """Temporary, synthetic approval-like data for stub runs only; never written into the repository."""
    plan = json.loads(protocol_path.read_text())
    pins = {rel: binding.sha256(ROOT / rel) for rel in list(plan.get("source_sha256", {})) + list(binding.IMPLEMENTATION_PINS)}
    return dict(approval_id="synthetic-stub-only", approves_execution=True, reviewer="stub", date="2026-09-13",
                unit_id=unit_dir.name, attempt=attempt, protocol_id=plan["protocol_id"], protocol_sha256=binding.sha256(protocol_path),
                planned_evaluations=plan["planned_evaluations"], cap_evaluations=plan["proposed_unit_cap"],
                unit_deadline_seconds=plan["proposed_wall_seconds"], max_batch_points=plan["max_batch_points"], git_head=GIT["head"],
                source_sha256=pins, target=copy.deepcopy(binding.TARGET), model_identity=copy.deepcopy(binding.MODEL_IDENTITY),
                expected_environment=binding.observed_environment())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=1)); return path


def stub_factory(fun, calls):
    def factory(spec):
        calls.append(spec)
        def evaluate(points):
            calls.append(len(points)); return fun(np.asarray(points, dtype=float))
        return evaluate
    return factory


def gaussian(x):
    return -np.square(x / .2).sum(axis=1) / 2


def run(binding, tmp_path, argv, **kw):
    log_dir = tmp_path / "logs"; log_dir.mkdir(exist_ok=True)
    before = set(log_dir.glob("binding-*.json"))
    code = binding.main(argv + ["--log-dir", str(log_dir)], **kw)
    new = sorted(set(log_dir.glob("binding-*.json")) - before)
    assert len(new) == 1                                                            # every invocation logs its real exit code
    return code, json.loads(new[0].read_text())


def execute_argv(protocol, approval, unit, attempt="stub-attempt"):
    return ["--protocol", str(protocol), "--approval", str(approval), "--unit-dir", str(unit), "--attempt", attempt, "--execute"]


# -- source guards ---------------------------------------------------------------------------------------------------
def test_source_has_no_module_level_target_import_and_the_lazy_import_lives_only_in_the_factory():
    src = SCRIPT.read_text(); tree = ast.parse(src); parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    def enclosing(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    found = []
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        found += [(n, enclosing(node)) for n in names if n.split(".")[0] in TARGET_ROOTS or n == "tests.reference_marginal"]
    assert found == [("tests.reference_marginal", "build_f7_target")]
    assert 'if __name__ == "__main__"' in src and "takeover=True" not in src and "run_logged" not in [n.split(".")[0] for n, _ in found]
    assert not any(isinstance(n, ast.Import) and any(a.name == "run_logged" for a in n.names) for n in ast.walk(tree))


def test_cli_option_inventory_has_no_escape_hatch_and_no_default_approval():
    tree = ast.parse(SCRIPT.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument"]
    assert sorted(c.args[0].value for c in calls) == ["--approval", "--attempt", "--dry-run", "--execute", "--log-dir", "--protocol", "--unit-dir"]
    assert not [kw for c in calls for kw in c.keywords if kw.arg == "default"]
    assert not [p for p in (ROOT / "docs").rglob("*") if "approv" in p.name.lower()]          # no approval file ships with the repository


# -- import, help, dry-run: no target, nothing created ----------------------------------------------------------------
def test_help_imports_nothing_and_creates_nothing(binding, tmp_path, capsys):
    before = target_modules_loaded()
    with pytest.raises(SystemExit) as exc:
        binding.main(["--help"])
    assert exc.value.code == 0 and "--dry-run" in capsys.readouterr().out and target_modules_loaded() == before and list(tmp_path.iterdir()) == []
    assert subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True).returncode == 0


def test_dry_run_on_the_toy_protocol_creates_no_unit_and_no_ledger(binding, tmp_path):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"; before = target_modules_loaded()
    code, log = run(binding, tmp_path, ["--protocol", str(protocol), "--unit-dir", str(unit), "--attempt", "a1", "--dry-run"], git_state=clean_git)
    assert code == 0 and log["mode"] == "dry-run" and log["exit_code"] == 0 and log["would_execute"] is False and target_modules_loaded() == before
    assert not unit.exists() and sorted(p.name for p in tmp_path.iterdir()) == ["logs", "toy_protocol.json"]
    assert log["protocol"]["stages"] == 21 and log["protocol"]["planned_evaluations"] == plan["planned_evaluations"] and log["protocol"]["first_stage"] == "anchor:reference"
    assert log["checks"]["unit_dir_new"]["ok"] is True and log["unperformed_checks"] == sorted(binding.APPROVAL_CHECKS) and log["failed_checks"] == []
    assert set(log["environment"]) == set(binding.ENVIRONMENT_KEYS) and log["git"]["head"] == GIT["head"]
    assert all(log["implementation_sha256"][rel] == binding.sha256(ROOT / rel) for rel in binding.IMPLEMENTATION_PINS)


def test_dry_run_reports_the_real_v3r1_identity_without_any_target_access(binding, tmp_path):
    before = target_modules_loaded()
    code, log = run(binding, tmp_path, ["--protocol", str(V3R1), "--dry-run"], git_state=clean_git)
    assert code == 0 and target_modules_loaded() == before
    p = log["protocol"]
    assert (p["stages"], p["planned_evaluations"], p["cap_evaluations"], p["unit_deadline_seconds"], p["max_batch_points"]) == (45, 12_038_157, 12_050_000, 720, 32768)
    assert p["protocol_id"] == "f7-direct-cdf-crossed-grid-v3r1-proposal" and p["sha256"] == binding.sha256(V3R1)
    for name in ("protocol_inventory", "planned_matches_protocol", "protocol_caps", "protocol_first_stage_is_anchor", "protocol_source_pins",
                 "implementation_files_present", "records_model_identity"):
        assert log["checks"][name]["ok"] is True, (name, log["checks"][name])
    assert log["failed_checks"] == [] and log["unperformed_checks"] == sorted(binding.APPROVAL_CHECKS) and log["would_execute"] is False
    assert not list(tmp_path.glob("**/ledger.json")) and not list(tmp_path.glob("**/*.lock"))


def test_dry_run_with_a_valid_synthetic_approval_would_execute_but_creates_nothing(binding, tmp_path):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"
    approval = write(tmp_path / "approval.json", synthetic_approval(binding, protocol, unit))
    code, log = run(binding, tmp_path, execute_argv(protocol, approval, unit)[:-1] + ["--dry-run"], git_state=clean_git)
    assert code == 0 and log["would_execute"] is True and log["failed_checks"] == [] and log["unperformed_checks"] == []
    assert not unit.exists() and log["approval"]["sha256"] == binding.sha256(approval)


# -- refusals: nothing created, nothing evaluated ----------------------------------------------------------------------
FAULTS = {"no_approval": [], "unreadable_approval": ["approval_schema"], "not_approved": ["approval_schema"], "missing_key": ["approval_schema"],
          "protocol_sha": ["approval_protocol_identity"], "protocol_id": ["approval_protocol_identity"], "planned": ["approval_counts"],
          "cap": ["approval_counts"], "deadline": ["approval_counts"], "batch": ["approval_counts"], "pin_wrong": ["approval_source_pins"],
          "pin_missing": ["approval_source_pins"], "target": ["approval_target"], "model_identity": ["approval_model_identity"],
          "environment": ["approval_environment"], "git_head": ["approval_git_head"], "dirty": ["git_clean"], "unit_id": ["approval_unit_binding"],
          "attempt": ["approval_unit_binding"], "unit_exists": ["unit_dir_new"], "parent_missing": ["unit_dir_new"]}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_execute_refusals_create_nothing_and_evaluate_nothing(binding, tmp_path, fault):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"; attempt = "stub-attempt"; git = clean_git()
    approval = synthetic_approval(binding, protocol, unit)
    if fault == "not_approved": approval["approves_execution"] = False
    if fault == "missing_key": del approval["reviewer"]
    if fault == "protocol_sha": approval["protocol_sha256"] = "0" * 64
    if fault == "protocol_id": approval["protocol_id"] = "other"
    if fault == "planned": approval["planned_evaluations"] += 1
    if fault == "cap": approval["cap_evaluations"] += 1
    if fault == "deadline": approval["unit_deadline_seconds"] = 61
    if fault == "batch": approval["max_batch_points"] = 65
    if fault == "pin_wrong": approval["source_sha256"]["tests/reference_direct_cdf.py"] = "0" * 64
    if fault == "pin_missing": del approval["source_sha256"]["tests/reference_marginal.py"]
    if fault == "target": approval["target"]["seed"] = 18
    if fault == "model_identity": approval["model_identity"]["center_predictors"] = False
    if fault == "environment": approval["expected_environment"]["numpy_version"] = "0.0.0"
    if fault == "git_head": approval["git_head"] = "f" * 40
    if fault == "dirty": git = dict(GIT, dirty=True, status=" M tests/x.py\n")
    if fault == "unit_id": approval["unit_id"] = "unit-2"
    if fault == "attempt": attempt = "other-attempt"
    if fault == "unit_exists": unit.mkdir()
    if fault == "parent_missing": unit = tmp_path / "missing" / "unit-1"
    path = None if fault == "no_approval" else write(tmp_path / "approval.json", approval)
    if fault == "unreadable_approval": path.write_text("{not json")
    calls = []
    argv = ["--protocol", str(protocol), "--unit-dir", str(unit), "--attempt", attempt, "--execute"] + ([] if path is None else ["--approval", str(path)])
    code, log = run(binding, tmp_path, argv, target_factory=stub_factory(gaussian, calls), git_state=lambda: git)
    assert code == 2 and log["exit_code"] == 2 and log["refusal"] and calls == []
    if fault == "no_approval":
        assert log["refusal"] == "--execute requires --approval"
    assert list(unit.iterdir()) == [] if fault == "unit_exists" else not unit.exists()                         # a pre-existing directory is untouched
    assert not list(tmp_path.glob("**/ledger.json")) and not list(tmp_path.glob("**/result.json")) and not list(tmp_path.glob("**/*.lock"))
    assert log["failed_checks"] == FAULTS[fault], log["checks"]
    if fault == "no_approval":
        assert log["unperformed_checks"] == sorted(binding.APPROVAL_CHECKS)


def test_cli_execute_without_approval_is_refused_in_a_real_process(tmp_path):
    protocol, _ = toy_protocol(tmp_path); unit = tmp_path / "unit-1"; log_dir = tmp_path / "logs"; log_dir.mkdir()
    proc = subprocess.run([sys.executable, str(SCRIPT), "--protocol", str(protocol), "--log-dir", str(log_dir), "--unit-dir", str(unit), "--attempt", "a1", "--execute"],
                          capture_output=True, text=True)
    assert proc.returncode == 2 and not unit.exists() and len(list(log_dir.glob("binding-*.json"))) == 1
    assert json.loads(proc.stdout)["refusal"] == "--execute requires --approval"


# -- the execution path with stubs only ---------------------------------------------------------------------------------
def test_execute_with_a_stub_target_runs_the_toy_protocol_and_writes_the_unit(binding, tmp_path):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"
    approval = write(tmp_path / "approval.json", synthetic_approval(binding, protocol, unit))
    calls = []; before = target_modules_loaded()
    code, log = run(binding, tmp_path, execute_argv(protocol, approval, unit), target_factory=stub_factory(gaussian, calls), git_state=clean_git)
    assert target_modules_loaded() == before and calls[0] == binding.TARGET and sum(calls[1:]) == plan["planned_evaluations"]
    assert sorted(p.name for p in unit.iterdir()) == ["binding_record.json", "ledger.json", "ledger.lock", "partial.json", "result.json", "summary.txt"]
    result = json.loads((unit / "result.json").read_text()); ledger = json.loads((unit / "ledger.json").read_text())
    assert code == result["exit_code"] == 4 and result["status"] == "inconclusive" and result["run_validity"]["valid"]      # toy faces fail; accounting exact
    assert result["budget"]["confirmed"] == result["planned_evaluations"] == plan["planned_evaluations"] and result["comparison"] == {}
    assert result["conditions"]["anchor_reference_consistency"] is True and result["binding"]["approval"]["sha256"] == binding.sha256(approval)
    assert result["binding"]["git"]["head"] == GIT["head"] and set(result["binding"]["environment"]) == set(binding.ENVIRONMENT_KEYS)
    assert ledger["released"] and ledger["confirmed_completed_total"] == plan["planned_evaluations"] and list(ledger["attempts"]) == ["stub-attempt"]
    partial = json.loads((unit / "partial.json").read_text())
    assert len(partial["stages"]) == 21 and partial["budget"]["confirmed"] == plan["planned_evaluations"]
    record = json.loads((unit / "binding_record.json").read_text())
    assert record["phase"] == "finished" and record["exit_code"] == 4 and log["exit_code"] == 4 and log["phase"] == "finished"
    assert "comparison with stored MCMC summaries: empty" in (unit / "summary.txt").read_text()


def test_execute_with_a_shifted_stub_stops_after_one_evaluation_and_consumes_the_unit(binding, tmp_path):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"
    approval = write(tmp_path / "approval.json", synthetic_approval(binding, protocol, unit)); calls = []
    code, log = run(binding, tmp_path, execute_argv(protocol, approval, unit), target_factory=stub_factory(lambda x: gaussian(x) - 40., calls), git_state=clean_git)
    result = json.loads((unit / "result.json").read_text()); ledger_bytes = (unit / "ledger.json").read_bytes(); ledger = json.loads(ledger_bytes)
    assert code == 3 and calls[1:] == [1] and list(result["stages"]) == ["anchor:reference"] and result["stopped"].startswith("preflight")
    assert result["budget"]["confirmed"] == 1 and ledger["confirmed_completed_total"] == 1 and ledger["released"] and log["stopped"].startswith("preflight")
    assert len(json.loads((unit / "partial.json").read_text())["stages"]) == 1 and result["conditions"]["anchor_reference_consistency"] is False
    # the same approval buys no second probe: the unit directory exists, so a re-run is refused before any evaluation
    code2, log2 = run(binding, tmp_path, execute_argv(protocol, approval, unit), target_factory=stub_factory(gaussian, calls), git_state=clean_git)
    assert code2 == 2 and log2["failed_checks"] == ["unit_dir_new"] and calls[1:] == [1] and (unit / "ledger.json").read_bytes() == ledger_bytes


def test_target_build_failure_after_unit_creation_consumes_the_unit_without_a_ledger(binding, tmp_path):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"
    approval = write(tmp_path / "approval.json", synthetic_approval(binding, protocol, unit))
    def failing(spec):
        raise ImportError("target import prohibited in this fixture")
    code, log = run(binding, tmp_path, execute_argv(protocol, approval, unit), target_factory=failing, git_state=clean_git)
    assert code == 2 and log["phase"] == "target_build_failed" and sorted(p.name for p in unit.iterdir()) == ["binding_record.json"]
    assert json.loads((unit / "binding_record.json").read_text())["phase"] == "target_build_failed"
    code2, log2 = run(binding, tmp_path, execute_argv(protocol, approval, unit), target_factory=failing, git_state=clean_git)
    assert code2 == 2 and log2["failed_checks"] == ["unit_dir_new"]


def test_interrupt_during_execution_keeps_the_reservation_and_writes_the_result(binding, tmp_path):
    protocol, plan = toy_protocol(tmp_path); unit = tmp_path / "unit-1"
    approval = write(tmp_path / "approval.json", synthetic_approval(binding, protocol, unit))
    def interrupted(x):
        raise KeyboardInterrupt("fixture interruption")
    code, log = run(binding, tmp_path, execute_argv(protocol, approval, unit), target_factory=stub_factory(interrupted, []), git_state=clean_git)
    result = json.loads((unit / "result.json").read_text()); ledger = json.loads((unit / "ledger.json").read_text())
    assert code == 3 and "KeyboardInterrupt" in result["stopped"] and result["budget"]["reserved_uncertain"] == 1 and result["budget"]["confirmed"] == 0
    assert ledger["reserved_uncertain_total"] == 1 and ledger["released"] and result["stages"] == {} and log["exit_code"] == 3


# -- outer wrapper ----------------------------------------------------------------------------------------------------
def test_run_logged_wrapper_records_the_real_child_exit_code_with_a_log_outside_any_unit(tmp_path):
    log = tmp_path / "wrapper-logs" / "nested" / "run.json"                                       # parent does not exist yet
    proc = subprocess.run([sys.executable, str(WRAPPER), str(log), "--", sys.executable, "-c", "import sys; sys.exit(3)"], capture_output=True, text=True)
    assert proc.returncode == 3 and json.loads(log.read_text())["returncode"] == 3 and json.loads(log.read_text())["status"] == "FAILED"
    ok = subprocess.run([sys.executable, str(WRAPPER), str(log), "--", sys.executable, "-c", "pass"], capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(log.read_text())["returncode"] == 0
