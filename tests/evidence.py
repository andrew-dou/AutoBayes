"""Where a test run may write evidence, and the rules that keep history immutable.

Rules, enforced before anything is created or written:

1. A run gets its own directory, allocated atomically with ``mkdir(exist_ok=False)`` under a
   name that carries the UTC time to the microsecond, the git head, the process id and a
   random token; a name collision is retried with a new token, never reused.
2. ``AUTOBAYES_EVIDENCE_DIR`` names the *parent* in which such a run directory is created,
   never the run directory itself, so an override cannot land two runs in one directory.
3. Nothing may resolve (symlinks followed) into the repository's ``docs/`` tree, where the
   committed evidence lives, and nothing may be created inside a directory that already
   holds a ``run.json`` (an existing run). Both are refused with an error before any write.
4. Evidence files are written with exclusive create (``open(..., "x")``); a second write to
   the same name in the same run fails instead of replacing the first.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUN_FILE = "run.json"


class EvidenceDestinationError(RuntimeError):
    """The requested evidence location would touch history or an existing run."""


def protected_roots(repo: Path) -> list[Path]:
    """Directories a test run must never write into: the committed documentation tree."""
    return [(repo / "docs").resolve()]


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def git_head(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                               timeout=10).stdout.strip() != ""
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def allocate_run_dir(repo: Path, override: str | os.PathLike | None, *, head: str, environment: dict,
                     now: datetime | None = None, argv: list[str] | None = None,
                     executable: str | None = None) -> Path:
    """Create and return a fresh, unique run directory; write its ``run.json`` first.

    Refuses, before creating anything, a parent that resolves into ``repo/docs`` or that
    already contains a ``run.json``. Two calls never return the same directory, whatever the
    clock says.
    """
    repo = Path(repo)
    parent = Path(override).expanduser() if override else repo / "evidence-runs"
    parent_resolved = parent.resolve()          # follows symlinks in the existing prefix
    for root in protected_roots(repo):
        if _inside(parent_resolved, root):
            raise EvidenceDestinationError(
                f"refusing to write evidence under {parent} (resolves to {parent_resolved}): it is inside the "
                f"committed history {root}. Point AUTOBAYES_EVIDENCE_DIR at a scratch directory instead."
            )
    if (parent_resolved / RUN_FILE).exists():
        raise EvidenceDestinationError(
            f"refusing to write into {parent_resolved}: it already holds a {RUN_FILE}, so it is an existing run's "
            "evidence. AUTOBAYES_EVIDENCE_DIR must name a parent directory in which a new run is created."
        )
    parent.mkdir(parents=True, exist_ok=True)
    when = now if now is not None else datetime.now(timezone.utc)
    stamp = when.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = None
    for _ in range(64):
        candidate = parent / f"{stamp}-{head}-p{os.getpid()}-{secrets.token_hex(3)}"
        try:
            candidate.mkdir(exist_ok=False)     # the atomic claim: exists -> retry with a new token
        except FileExistsError:
            continue
        run_dir = candidate
        break
    if run_dir is None:
        raise EvidenceDestinationError(f"could not allocate a unique run directory under {parent}")
    resolved = run_dir.resolve()
    if not _inside(resolved, parent_resolved) or any(_inside(resolved, r) for r in protected_roots(repo)):
        run_dir.rmdir()
        raise EvidenceDestinationError(f"allocated directory {run_dir} resolves outside its parent or into history")
    write_evidence(run_dir, RUN_FILE, json.dumps({
        "git_head": head, "started_utc": when.astimezone(timezone.utc).isoformat(),
        "python_executable": executable or sys.executable, "argv": argv if argv is not None else sys.argv,
        "pid": os.getpid(), "environment": environment,
    }, indent=1, default=str))
    return run_dir


def write_evidence(run_dir: Path, name: str, text: str) -> Path:
    """Write one evidence file into a run directory. Never overwrites: a second write to the
    same name raises ``FileExistsError`` and leaves the first file untouched."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"evidence file name must be a plain file name, got {name!r}")
    path = Path(run_dir) / name
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return path


def committed_bytes(repo: Path, tracked_dirs: list[str]) -> dict[str, bytes]:
    """The committed (HEAD) content of every tracked file under the given directories."""
    ls = subprocess.run(["git", "ls-files", "-z", *tracked_dirs], cwd=repo, capture_output=True, check=True)
    out = {}
    for rel in filter(None, ls.stdout.decode().split("\0")):
        out[rel] = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=repo, capture_output=True, check=True).stdout
    return out


def working_tree_bytes(repo: Path, rels: list[str]) -> dict[str, bytes]:
    return {rel: (Path(repo) / rel).read_bytes() for rel in rels}
