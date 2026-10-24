"""tools/diff_coverage.py on throwaway git repositories under tmp_path, each holding a toy `autobayes` package and its tests.

The tool runs in a subprocess (its pytest must not share this one's modules); nothing here imports PyMC.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "diff_coverage.py"
PYTEST_INI = "[pytest]\nmarkers =\n    slow: slow\n    sampling: sampling\n"
NOT_RUN = "# [not run]"                                    # marks the lines a scenario expects the tool to report


def env() -> dict:
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PYTEST_")) and k != "PYTHONPATH"}
    return {**keep, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, env=env(), check=True, capture_output=True, text=True).stdout


def write(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))


def repo_with(tmp_path: Path, base: dict[str, str], head: dict[str, str], remove=(), rename=()) -> Path:
    """A repository whose branch ``base`` holds ``base`` and whose HEAD commit renames, removes and writes on top of it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    write(repo, {"pytest.ini": PYTEST_INI, "autobayes/__init__.py": "", **base})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "branch", "base")
    for old, new in rename:
        git(repo, "mv", old, new)
    for name in remove:
        git(repo, "rm", "-q", name)
    write(repo, head)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")
    return repo


def run(repo: Path, *args: str) -> tuple[int, str, dict | None]:
    report = repo.parent / "report.json"
    report.unlink(missing_ok=True)
    proc = subprocess.run([sys.executable, str(TOOL), "--json", str(report), *args], cwd=repo, env=env(),
                          capture_output=True, text=True, timeout=300)
    data = json.loads(report.read_text()) if report.exists() else None
    return proc.returncode, proc.stdout + proc.stderr, data


def marked(repo: Path, path: str, marker: str = NOT_RUN) -> list[int]:
    return [n for n, line in enumerate((repo / path).read_text().split("\n"), 1) if line.endswith(marker)]


def line_of(repo: Path, path: str, text: str) -> int:
    lines = (repo / path).read_text().split("\n")
    return next(n for n, line in enumerate(lines, 1) if line.strip() == text)


MOVED = "".join(f"def f{i}(x):\n    return x + {i}\n\n\n" for i in range(8))

CORE_BASE = """\
def kept(x):
    return x + 1


def shrinking(x):
    y = x * 2
    y = y + 0
    return y
"""

CORE_HEAD = f'''\
"""A toy library."""
import functools


def kept(x):
    return x + 1


def shrinking(x):
    y = x * 2
    return y


def _deco(f):
    @functools.wraps(f)
    def wrapper(*a):
        return f(*a)
    return wrapper


@_deco
def added(x):
    """A docstring: no code."""
    # a comment: no code

    def inner(k):
        return k * 3

    def unused_inner():
        return 0  {NOT_RUN}
    if x > 0:
        return inner(x)
    return -x  {NOT_RUN}


def never_called():
    value = 1  {NOT_RUN}
    return value  {NOT_RUN}


def guarded(x):
    if x is None:
        raise TypeError("x")  # pragma: no cover - defensive: no caller passes None
    return x
'''

FEATURE_TESTS = """\
from autobayes import core, fresh, new_name


def test_added():
    assert core.added(2) == 6


def test_guarded():
    assert core.guarded(1) == 1


def test_moved():
    assert new_name.f3(1) == 4


def test_fresh():
    assert fresh.fresh() == 1
"""


def test_changed_lines_their_tests_and_what_did_not_run(tmp_path):
    repo = repo_with(
        tmp_path,
        base={"autobayes/core.py": CORE_BASE, "autobayes/old_name.py": MOVED, "autobayes/gone.py": "def g():\n    return 1\n",
              "tests/test_core.py": "from autobayes import core\n\n\ndef test_kept():\n    assert core.kept(1) == 2\n"},
        head={"autobayes/core.py": CORE_HEAD, "autobayes/new_name.py": MOVED + f"def extra(x):\n    return -x  {NOT_RUN}\n",
              "tests/test_feature.py": FEATURE_TESTS},
        remove=["autobayes/gone.py"], rename=[("autobayes/old_name.py", "autobayes/new_name.py")])
    with open(repo / "autobayes/core.py", "a") as f:                      # uncommitted, and an untracked module
        f.write(f"\n\ndef uncommitted(x):\n    return x  {NOT_RUN}\n")
    write(repo, {"autobayes/fresh.py": f"def fresh():\n    return 1\n\n\ndef stale():\n    return 2  {NOT_RUN}\n"})

    code, out, report = run(repo, "--base", "base")
    assert code == 1, out
    files = {f["path"]: f for f in report["files"]}
    assert sorted(files) == ["autobayes/core.py", "autobayes/fresh.py", "autobayes/new_name.py"]   # gone.py: deleted
    assert report["pytest"]["args"][-1] == "tests/test_feature.py"         # the changed test file; test_core.py is not
    assert report["pytest"]["exit"] == 0

    core = files["autobayes/core.py"]
    assert [x["line"] for x in core["not_run"]] == marked(repo, "autobayes/core.py")   # incl. the branch not taken
    assert core["exempt"] == [{"line": line_of(repo, "autobayes/core.py", 'raise TypeError("x")  # pragma: no cover - '
                                               "defensive: no caller passes None"),
                               "reason": "defensive: no caller passes None", "ran": False}]
    changed = set(core["changed_executable"])
    for text in ('"""A docstring: no code."""', "# a comment: no code", "def kept(x):", "return x + 1", "y = x * 2"):
        assert line_of(repo, "autobayes/core.py", text) not in changed, text   # no code, or not changed
    assert "" not in {(repo / "autobayes/core.py").read_text().split("\n")[n - 1].strip() for n in changed}
    for text in ("@_deco", "@functools.wraps(f)", "def inner(k):", "return k * 3", "def unused_inner():",
                 "return inner(x)", "def uncommitted(x):"):
        n = line_of(repo, "autobayes/core.py", text)
        assert n in changed and n in core["ran"], text                     # decorators and nested functions ran

    moved = files["autobayes/new_name.py"]                                 # the rename counts only the added lines
    assert moved["changed_executable"] == [line_of(repo, "autobayes/new_name.py", "def extra(x):"),
                                           *marked(repo, "autobayes/new_name.py")]
    assert [x["line"] for x in moved["not_run"]] == marked(repo, "autobayes/new_name.py")

    fresh = files["autobayes/fresh.py"]                                    # untracked: every line is new
    assert fresh["changed_executable"] == [1, 2, 5, 6] and [x["line"] for x in fresh["not_run"]] == [6]

    not_run = sum(len(f["not_run"]) for f in files.values())
    assert report["total"] - report["covered"] == not_run == 7
    assert out.rstrip().endswith(f"DIFF COVERAGE {report['covered']}/{report['total']} ({report['percent']:.1f}%)")
    assert "return value  # [not run]" in out and "  not run" in out      # every uncovered line with its source


SMALL = {"autobayes/a.py": "def one():\n    return 1\n", "tests/test_other.py": "def test_nothing():\n    pass\n",
         "tests/test_broken.py": "from autobayes import a\n\n\ndef test_broken():\n    assert a.two() == 3\n"}


def test_exit_codes_and_explicit_pytest_arguments(tmp_path):
    repo = repo_with(tmp_path, base=SMALL, head={
        "autobayes/a.py": "def one():\n    return 1\n\n\ndef two():\n    return 2\n",
        "tests/test_a.py": "from autobayes import a\n\n\ndef test_two():\n    assert a.two() == 2\n"})
    code, out, report = run(repo, "--base", "base")
    assert code == 0 and "DIFF COVERAGE 2/2 (100.0%)" in out, out
    assert report["passed"] and report["pytest"]["args"][-1] == "tests/test_a.py"

    code, out, report = run(repo, "--base", "base", "--", "tests/test_other.py")   # replaces the default selection
    assert code == 1 and "DIFF COVERAGE 0/2 (0.0%)" in out, out
    assert report["pytest"]["args"] == ["-p", "no:cacheprovider", "-m", "not slow and not sampling", "tests/test_other.py"]

    code, out, report = run(repo, "--base", "base", "--", "tests/test_broken.py")  # every line ran, but a test failed
    assert code == 1 and "DIFF COVERAGE 2/2 (100.0%)" in out and "pytest exited 1" in out, out

    assert run(repo, "--base", "no-such-branch")[0] == 2
    assert run(repo, "--base", "base", "--no-such-option")[0] == 2
    assert run(repo, "--base", "base", "--", "--no-such-pytest-option")[0] == 2


def test_a_pragma_needs_a_reason(tmp_path):
    repo = repo_with(tmp_path, base=SMALL, head={
        "autobayes/a.py": "def one():\n    return 1  # pragma: no cover\n\n\ndef two():\n    return 2  # pragma: no cover -\n",
        "tests/test_a.py": "from autobayes import a\n\n\ndef test_both():\n    assert a.one() + a.two() == 3\n"})
    code, out, report = run(repo, "--base", "base")
    assert code == 1, out                                                  # the lines ran; the bare pragmas fail it
    assert report["covered"] == report["total"] == 3
    assert [e["line"] for e in report["files"][0]["errors"]] == [2, 6] and out.count("a pragma without a reason") >= 2


def test_a_deletion_alone_needs_no_test(tmp_path):
    repo = repo_with(tmp_path, base={**SMALL, "autobayes/a.py": "def one():\n    return 1\n\n\ndef two():\n    return 2\n"},
                     head={"autobayes/a.py": "def one():\n    return 1\n"})
    code, out, report = run(repo, "--base", "base")
    assert code == 0 and "DIFF COVERAGE 0/0 (100.0%)" in out and report["pytest"]["exit"] is None, out


def test_default_selection_runs_the_gates_of_added_and_ungated_cases(tmp_path):
    case = lambda cid, **extra: json.dumps({"id": cid, "brms": {}, "autobayes": {}, **extra})
    fixture = lambda cid: {f"tests/oracle/fixtures/{cid}/fixture.json": "{}"}
    base = {"tests/oracle/test_baseline.py": "def _cases():\n    pass\n\n\ndef test_g1(bundle):\n    pass\n\n\n"
                                             "def test_g2(bundle):\n    pass\n",
            "tests/oracle/cases/kept.json": case("kept"), **fixture("kept"),
            "tests/oracle/cases/gated.json": case("gated", gate={"expect": "fail", "reason": "r"}), **fixture("gated"),
            "tests/helper.py": "X = 1\n"}
    head = {"tests/oracle/cases/gated.json": case("gated"),
            "tests/oracle/cases/new.json": case("new"), **fixture("new"),
            "tests/oracle/cases/no_fixture.json": case("no_fixture"),
            "tests/oracle/cases/brms_only.json": json.dumps({"id": "brms_only", "brms": {}}), **fixture("brms_only"),
            "tests/test_x.py": "def test_x():\n    pass\n", "tests/helper.py": "X = 2\n"}
    repo = repo_with(tmp_path, base=base, head=head)
    spec = importlib.util.spec_from_file_location("diff_coverage", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tests, cases, nodes = tool.selection(repo, git(repo, "rev-parse", "base").strip())
    assert tests == ["tests/test_x.py"] and cases == ["gated", "new"]
    assert nodes == [f"tests/oracle/test_baseline.py::{t}[{c}]" for c in cases for t in ("test_g1", "test_g2")]
