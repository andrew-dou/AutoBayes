"""Evidence handling: unique run directories, refusal of history and existing runs before any
write, no overwrites, and byte-level checks that the committed history is untouched. The
allocation tests run against fake repositories in ``tmp_path`` with an injected clock, so they
are deterministic and touch nothing in this checkout."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

import autobayes as ab
from tests.conftest import REPO
from tests.evidence import (EvidenceDestinationError, allocate_run_dir, committed_bytes, git_head,
                            working_tree_bytes, write_evidence)

FIXED_NOW = datetime(2026, 9, 10, 3, 0, 0, tzinfo=timezone.utc)
ENV = {"stub": True}
REQUIRED_RECORD_KEYS = {"autobayes_version", "model", "execution", "environment", "diagnostics", "posterior"}
REQUIRED_EXECUTION_KEYS = {"backend", "device_requested", "device_actual", "platform", "dtype", "chain_method",
                           "chains", "draws", "tune", "target_accept", "seed", "wall_seconds", "notes"}


def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    hist = repo / "docs" / "history" / "m0-evidence"
    hist.mkdir(parents=True)
    (hist / "run.json").write_text("HISTORICAL-RUN")
    (hist / "B2_recovery.json").write_text("HISTORICAL-B2")
    return repo


def tree_snapshot(root: Path) -> dict:
    """Every file under root with its bytes' hash: the before/after comparison the tests use."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def alloc(repo, override=None, **kw):
    return allocate_run_dir(repo, override, head="abc123", environment=ENV, now=FIXED_NOW, argv=["t"], **kw)


# -- uniqueness -----------------------------------------------------------------------------
def test_same_instant_same_head_allocations_are_distinct_and_preserve_the_first(tmp_path):
    repo = fake_repo(tmp_path)
    d1 = alloc(repo)
    (d1 / "B1.json").write_text("first")
    before = tree_snapshot(d1)
    d2 = alloc(repo)                                   # same clock reading, same head, same pid
    assert d1 != d2 and d1.is_dir() and d2.is_dir() and d1.parent == d2.parent == repo / "evidence-runs"
    assert tree_snapshot(d1) == before                 # nothing in the first run changed
    assert json.loads((d2 / "run.json").read_text())["git_head"] == "abc123"


def test_many_allocations_never_collide(tmp_path):
    repo = fake_repo(tmp_path)
    dirs = [alloc(repo) for _ in range(20)]
    assert len(set(dirs)) == 20


# -- refusals, before any write ---------------------------------------------------------------
@pytest.mark.parametrize("target", ["docs/history/m0-evidence", "docs/history/m0-evidence/new-run", "docs",
                                    "docs/history/m1-evidence/x/y"])
def test_override_into_history_is_refused_before_any_write(tmp_path, target):
    repo = fake_repo(tmp_path)
    before = tree_snapshot(repo)
    with pytest.raises(EvidenceDestinationError, match="committed history"):
        alloc(repo, repo / target)
    assert tree_snapshot(repo) == before               # no file created or changed anywhere in the repo
    assert (repo / "docs" / "history" / "m0-evidence" / "run.json").read_text() == "HISTORICAL-RUN"


def test_symlink_into_history_is_refused(tmp_path):
    repo = fake_repo(tmp_path)
    link = tmp_path / "innocent-looking"
    link.symlink_to(repo / "docs" / "history" / "m0-evidence", target_is_directory=True)
    before = tree_snapshot(repo)
    with pytest.raises(EvidenceDestinationError, match="resolves to"):
        alloc(repo, link)
    with pytest.raises(EvidenceDestinationError):
        alloc(repo, link / "sub")                      # a not-yet-existing child of the link, still inside history
    assert tree_snapshot(repo) == before


def test_override_pointing_at_an_existing_run_is_refused(tmp_path):
    repo = fake_repo(tmp_path)
    first = alloc(repo, tmp_path / "scratch")          # a run inside the scratch parent
    (first / "B1.json").write_text("first-bytes")
    before = tree_snapshot(first)
    with pytest.raises(EvidenceDestinationError, match="existing run"):
        alloc(repo, first)                             # the run directory itself, not a parent
    assert tree_snapshot(first) == before


def test_override_is_a_parent_and_retries_do_not_reuse_a_run(tmp_path):
    repo = fake_repo(tmp_path)
    parent = tmp_path / "scratch" / "ci-fast-123-1"    # what CI passes, reused verbatim on a retry
    d1 = alloc(repo, parent)
    (d1 / "B1.json").write_text("attempt-one")
    before = tree_snapshot(d1)
    d2 = alloc(repo, parent)                           # the retry with the identical override
    assert d1.parent == d2.parent == parent and d1 != d2
    assert tree_snapshot(d1) == before and (d1 / "B1.json").read_text() == "attempt-one"


# -- no overwrites within a run ---------------------------------------------------------------
def test_write_evidence_never_overwrites(tmp_path):
    repo = fake_repo(tmp_path)
    d = alloc(repo)
    write_evidence(d, "B2.json", "original")
    with pytest.raises(FileExistsError):
        write_evidence(d, "B2.json", "replacement")
    assert (d / "B2.json").read_text() == "original"
    with pytest.raises(FileExistsError):
        write_evidence(d, "run.json", "{}")            # run.json is written once, at allocation
    for bad in ("../escape.json", "sub/dir.json", "", "."):
        with pytest.raises(ValueError):
            write_evidence(d, bad, "x")


# -- this checkout ----------------------------------------------------------------------------
def test_session_run_dir_is_fresh_and_outside_history(evidence_dir):
    resolved = evidence_dir.resolve()
    assert not resolved.is_relative_to((REPO / "docs").resolve())
    run = json.loads((evidence_dir / "run.json").read_text())
    assert {"git_head", "started_utc", "python_executable", "argv", "pid", "environment"} <= set(run)
    assert run["git_head"] == git_head(REPO) and run["pid"] == os.getpid()
    assert run["environment"]["versions"]["pymc"] == "5.28.5"


def test_fit_record_schema_and_exclusive_write(model, small_fit, evidence_dir):
    rec = ab.fit_record(model, small_fit)
    assert REQUIRED_RECORD_KEYS <= set(rec) and REQUIRED_EXECUTION_KEYS <= set(rec["execution"])
    assert rec["model"]["family"] in {"gaussian", "bernoulli"} and rec["model"]["link"] in {"identity", "logit"}
    assert {"values", "status", "overall"} <= set(rec["diagnostics"])
    path = write_evidence(evidence_dir, "schema_check_record.json", json.dumps(rec, indent=1))
    assert json.loads(path.read_text())["execution"]["seed"] == 1
    with pytest.raises(FileExistsError):
        write_evidence(evidence_dir, "schema_check_record.json", "{}")


def test_committed_history_bytes_match_head(evidence_dir):
    """The committed evidence files have exactly their committed bytes after this run has
    allocated its directory and written into it."""
    head = committed_bytes(REPO, ["docs/history/m0-evidence", "docs/history/m1-evidence"])
    assert len(head) >= 10, "expected the committed evidence files to be tracked"
    assert working_tree_bytes(REPO, list(head)) == head
