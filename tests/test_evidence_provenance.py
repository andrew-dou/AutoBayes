"""Deterministic checks on evidence records, separate from the stochastic sampling criteria:
schema, provenance fields, and that candidate evidence never lands in the committed history."""
import json
from pathlib import Path

import autobayes as ab

REPO = Path(__file__).resolve().parent.parent
REQUIRED_RECORD_KEYS = {"autobayes_version", "model", "execution", "environment", "diagnostics", "posterior"}
REQUIRED_EXECUTION_KEYS = {"backend", "device_requested", "device_actual", "platform", "dtype", "chain_method",
                           "chains", "draws", "tune", "target_accept", "seed", "wall_seconds", "notes"}


def test_evidence_dir_is_run_specific_and_has_provenance(evidence_dir):
    assert "docs" not in evidence_dir.resolve().parts or "evidence-runs" in evidence_dir.resolve().parts
    assert not str(evidence_dir.resolve()).startswith(str((REPO / "docs" / "m0-evidence").resolve()))
    assert not str(evidence_dir.resolve()).startswith(str((REPO / "docs" / "m1-evidence").resolve()))
    run = json.loads((evidence_dir / "run.json").read_text())
    assert {"git_head", "started_utc", "python_executable", "environment"} <= set(run)
    assert run["environment"]["versions"]["pymc"] == "5.28.5"


def test_fit_record_schema(model, small_fit, evidence_dir):
    rec = ab.fit_record(model, small_fit)
    assert REQUIRED_RECORD_KEYS <= set(rec) and REQUIRED_EXECUTION_KEYS <= set(rec["execution"])
    assert rec["model"]["family"] in {"gaussian", "bernoulli"} and rec["model"]["link"] in {"identity", "logit"}
    assert {"values", "status", "overall"} <= set(rec["diagnostics"])
    path = evidence_dir / "schema_check_record.json"
    path.write_text(json.dumps(rec, indent=1))
    assert json.loads(path.read_text())["execution"]["seed"] == 1


def test_committed_history_is_untouched_by_the_fixture(evidence_dir):
    for name in ("docs/m0-evidence/B2_recovery.json", "docs/m1-evidence/B1_recovery.json"):
        f = REPO / name
        assert f.exists() and f.resolve().parent != evidence_dir.resolve()
