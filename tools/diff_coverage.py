#!/usr/bin/env python
"""Diff coverage: every line of library code a branch adds or changes must run under the tests.

    python tools/diff_coverage.py [--base origin/main] [--json PATH] [-- <pytest args>]

Changed lines: the lines added or modified under autobayes/**/*.py in `git diff -U0 <base>...HEAD` plus the uncommitted
changes, staged or not, and untracked files. That is one `git diff -U0 -M <merge base>` against the working tree, so the
line numbers are those of the code that runs. A deleted line is not counted; a renamed file counts only the lines its
diff adds.

Executable lines among them: the lines of the compiled code objects' line tables (`co_lines()`, recursively over nested
code objects), not a regular expression. Docstrings of functions, comments, blank lines, `else:` and `finally:` lines and
the continuation lines of a folded constant carry no code and do not count. A `def`, `class` or decorator line belongs to
the code that defines the function and counts as run when that code runs (at import, for a top-level function).

Execution: pytest runs in this process (`pytest.main`) under a `sys.settrace` and `threading.settrace` tracer that
returns a line tracer only for frames whose `co_filename` is one of the changed files. Default pytest arguments: the test
files (`test_*.py`) the diff adds or changes under tests/, plus the oracle gate tests (tests/oracle/test_baseline.py) of
every case whose file under tests/oracle/cases/ the diff adds or changes (a feature adds cases or removes their gate
field), with `-q`. `-p no:cacheprovider -m "not slow and not sampling"` always comes first; arguments after `--` replace
the default selection (a `-m` among them replaces this one). Nothing runs when no changed line needs to.

Exemptions: a changed line ending in `# pragma: no cover - <reason>`. The reason is required: a pragma without one
(`# pragma: no cover`, `# pragma: no cover -`) on a changed line is an error and fails the run. A pragma exempts its own
line only, not the block the line opens.

Output: per file, the changed executable lines, how many ran, the exempt lines with their reasons and every line that did
not run with its source text; then `DIFF COVERAGE <covered>/<total> (<pct>%)`, where total counts the non-exempt changed
executable lines and the percentage is rounded down. Exit 0 only when every one of them ran (and pytest passed or
collected nothing); 1 otherwise; 2 on a usage error (an unknown base, a bad option, a pytest usage error, an `autobayes`
imported from another checkout). `--json PATH` writes the same report as JSON.

What it cannot see:
- code that runs only in a subprocess (`subprocess`, `multiprocessing`, the fresh-interpreter tests): the tracer is in
  this process;
- JAX-compiled code after tracing: a jitted function's Python runs once, when JAX traces it, and that run is what counts
  here. The compiled program that runs afterwards has no Python lines; likewise a compiled PyTensor function, whose
  graph-building Python counts and whose C or JAX code does not;
- anything while a test replaces the trace function (a debugger, `sys.settrace` in a test), and threads started before
  the run.
On Python 3.11 an installed trace function puts every frame of the process in the interpreter's tracing mode, so the rest
of the suite does not run at full speed: the filter spares it the Python line callbacks, not the slower dispatch (a
feature's tests, those of #78, took about 1.8 times their untraced CPU time).
"""
from __future__ import annotations

import argparse
import ast
import codecs
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import tokenize
import types
from pathlib import Path

LIBRARY = "autobayes"
TESTS = "tests"
CASES = "tests/oracle/cases"
FIXTURES = "tests/oracle/fixtures"
BASELINE = "tests/oracle/test_baseline.py"
ALWAYS = ["-p", "no:cacheprovider", "-m", "not slow and not sampling"]
HUNK = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
TEST_FILE = re.compile(r"test_.*\.py|.*_test\.py")
PRAGMA = re.compile(r"#\s*pragma:\s*no\s+cover(?P<rest>.*)$")
REASON = re.compile(r"\s+-\s+(?P<reason>\S.*?)\s*")


class UsageError(Exception):
    """Exit 2: a bad base or option, or a checkout the tool cannot measure."""


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-c", "core.quotePath=false", *args], cwd=repo, capture_output=True, text=True)
    if proc.returncode:
        raise UsageError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def unquote(path: str) -> str:
    """A path as git prints it: C-quoted when it holds a quote, a backslash or a control character."""
    path = path.rstrip("\t")                               # git ends a ---/+++ name that holds a space with a tab
    if len(path) > 1 and path[0] == path[-1] == '"':
        return codecs.escape_decode(path[1:-1].encode())[0].decode("utf-8")
    return path


def untracked(repo: Path, where: str) -> list[str]:
    return [p for p in git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", where).split("\0") if p]


def changed_lines(repo: Path, merge_base: str) -> dict[str, set[int]]:
    """{library path: line numbers the diff adds or modifies, in the working tree}."""
    out = git(repo, "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-relative", "-U0", "-M",
              "--src-prefix=a/", "--dst-prefix=b/", merge_base, "--", LIBRARY)
    changed: dict[str, set[int]] = {}
    path, header = None, False
    for line in out.splitlines():
        if line.startswith("diff --git "):
            path, header = None, True
        elif header and line.startswith("+++ "):
            new = unquote(line[4:])
            path = new[2:] if new.startswith("b/") else None    # "+++ /dev/null": the file was deleted
        elif line.startswith("@@"):
            header = False
            hunk = HUNK.match(line)
            if path is not None and hunk:
                start, count = int(hunk[1]), int(hunk[2] if hunk[2] is not None else 1)
                changed.setdefault(path, set()).update(range(start, start + count))
    for path in untracked(repo, LIBRARY):
        if path.endswith(".py"):
            changed[path] = set(range(1, len(read_source(repo / path).split("\n")) + 1))
    return {p: lines for p, lines in changed.items() if p.startswith(LIBRARY + "/") and p.endswith(".py")}


def changed_paths(repo: Path, merge_base: str, where: str) -> list[str]:
    """Files under ``where`` that the diff adds, modifies, renames or copies (not deleted ones), and untracked files."""
    fields = git(repo, "diff", "--no-ext-diff", "--no-relative", "--name-status", "-z", "-M", merge_base, "--",
                 where).split("\0")
    paths, i = [], 0
    while i < len(fields) - 1:
        status = fields[i]
        if status[:1] in ("R", "C"):
            paths.append(fields[i + 2])
            i += 3
        else:
            if status[:1] != "D":
                paths.append(fields[i + 1])
            i += 2
    return sorted(set(paths + untracked(repo, where)))


def selection(repo: Path, merge_base: str) -> tuple[list[str], list[str], list[str]]:
    """(changed test files, changed oracle cases that have a fixture and an AutoBayes call, their gate-test node ids)."""
    tests = [p for p in changed_paths(repo, merge_base, TESTS)
             if TEST_FILE.fullmatch(p.rsplit("/", 1)[-1]) and (repo / p).is_file()]
    cases = []
    for p in changed_paths(repo, merge_base, CASES):
        path = repo / p
        if path.parent != repo / CASES or path.suffix != ".json" or not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        if "autobayes" in case and (repo / FIXTURES / path.stem / "fixture.json").exists():
            cases.append(path.stem)                        # the cases tests/oracle/test_baseline.py parametrizes
    nodes = []
    if cases and BASELINE not in tests and (repo / BASELINE).is_file():
        tree = ast.parse((repo / BASELINE).read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        nodes = [f"{BASELINE}::{name}[{cid}]" for cid in cases for name in names]
    return tests, cases, nodes


def read_source(path: Path) -> str:
    with tokenize.open(path) as f:                         # honours a coding cookie; universal newlines
        return f.read()


def analyse(path: Path):
    """(executable lines, {line: pragma reason}, {line: pragma without a reason}, source lines) of one file."""
    text = read_source(path)
    executable, stack = set(), [compile(text, str(path), "exec", dont_inherit=True)]
    while stack:
        code = stack.pop()
        executable.update(line for _, _, line in code.co_lines() if line)   # None: artificial; 0: the module's RESUME
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    exempt, bare = {}, {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and (pragma := PRAGMA.search(tok.string)):
            reason = REASON.fullmatch(pragma["rest"])
            if reason:
                exempt[tok.start[0]] = reason["reason"]
            else:
                bare[tok.start[0]] = tok.string
    return executable, exempt, bare, text.split("\n")


def run_pytest(repo: Path, targets: list[str], args: list[str]) -> tuple[int, dict[str, set[int]], float]:
    """Run pytest in this process, tracing lines only in ``targets``; (exit code, {path: lines run}, seconds)."""
    import pytest

    wanted = {os.path.realpath(repo / t) for t in targets}
    known: dict[str, bool] = {}
    hits: set[tuple[str, int]] = set()
    add = hits.add

    def lines(frame, event, arg):
        if event == "line":
            add((frame.f_code.co_filename, frame.f_lineno))
        return lines

    def calls(frame, event, arg):                          # called on every new frame; keep it cheap
        name = frame.f_code.co_filename
        hit = known.get(name)
        if hit is None:                                    # a relative co_filename is relative to the repository
            hit = known[name] = os.path.realpath(os.path.join(repo, name)) in wanted
        return lines if hit else None

    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={repo / '.pytensor-gate'}")
    sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    os.chdir(repo)
    t0 = time.perf_counter()
    threading.settrace(calls)
    sys.settrace(calls)
    try:
        code = int(pytest.main(args))
    finally:
        sys.settrace(None)
        threading.settrace(None)
        os.chdir(cwd)
    seconds = time.perf_counter() - t0
    library = getattr(sys.modules.get(LIBRARY), "__file__", None)
    if library and not os.path.realpath(library).startswith(os.path.realpath(repo / LIBRARY) + os.sep):
        raise UsageError(f"the tests imported {LIBRARY} from {library}, not from {repo}")
    ran: dict[str, set[int]] = {}
    for name, line in hits:
        ran.setdefault(os.path.relpath(os.path.realpath(os.path.join(repo, name)), os.path.realpath(repo)),
                       set()).add(line)
    return code, ran, seconds


def ranges(lines: list[int]) -> str:
    out, start = [], None
    for i, n in enumerate(lines):
        if start is None:
            start = n
        if i + 1 == len(lines) or lines[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    own, pytest_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="the branch the diff is taken against (default origin/main)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON to this path")
    args = ap.parse_args(own)                              # exits 2 on a bad option
    try:
        return measure(args, pytest_args)
    except UsageError as e:
        print(f"diff_coverage: {e}", file=sys.stderr)
        return 2


def measure(args, pytest_args) -> int:
    json_path = args.json.resolve() if args.json else None
    repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    try:
        git(repo, "rev-parse", "--verify", "--quiet", f"{args.base}^{{commit}}")
    except UsageError:
        raise UsageError(f"--base {args.base}: no such commit") from None
    merge_base = git(repo, "merge-base", args.base, "HEAD").strip()
    head = git(repo, "rev-parse", "--short=12", "HEAD").strip()
    dirty = bool(git(repo, "status", "--porcelain").strip())
    changed = changed_lines(repo, merge_base)

    files, errors, counted = [], [], {}
    for path in sorted(changed):
        if not (repo / path).is_file():
            continue
        try:
            executable, exempt, bare, source = analyse(repo / path)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            errors.append(f"{path}: cannot compile ({e})")
            continue
        lines = sorted(changed[path] & executable)
        entry = {"path": path, "changed_executable": lines,
                 "exempt": [{"line": n, "reason": exempt[n]} for n in lines if n in exempt], "errors": []}
        for n in sorted(changed[path] & bare.keys()):
            entry["errors"].append({"line": n, "message": f"a pragma without a reason ({bare[n].strip()}): write "
                                                          "'# pragma: no cover - <reason>'"})
            errors.append(f"{path}:{n}: a pragma without a reason")
        counted[path] = [n for n in lines if n not in exempt]
        files.append((entry, source))

    if pytest_args is None:
        tests, cases, nodes = selection(repo, merge_base)
        run_args = [*ALWAYS, "-q", *tests, *nodes] if tests or nodes else None
        shown = shlex.join([*ALWAYS, "-q", *tests]) + (f" + the oracle gates of {len(cases)} case(s) "
                                                       f"({', '.join(cases)})" if nodes else "")
    else:
        run_args = [*ALWAYS, *pytest_args]
        shown = shlex.join(run_args)
    pytest_code, ran, seconds = None, {}, 0.0
    if any(counted.values()) and run_args is not None:
        pytest_code, ran, seconds = run_pytest(repo, [p for p, lines in counted.items() if lines], run_args)
        if pytest_code == 4:
            raise UsageError(f"pytest refused its arguments: {shown}")

    print(f"\ndiff coverage of {LIBRARY}/ against {args.base} (merge base {merge_base[:12]}) at {head}"
          f"{' with uncommitted changes' if dirty else ''}")
    if pytest_code is not None:
        print(f"pytest {shown}: exit {pytest_code} in {seconds:.1f} s")
    elif any(counted.values()):
        print("no test file or oracle case changed and no pytest arguments given: no test ran")
    else:
        print("no changed line needs a test: pytest not run")
    covered = total = 0
    for entry, source in files:
        path, lines = entry["path"], entry["changed_executable"]
        did = ran.get(path, set())
        entry["ran"] = [n for n in counted[path] if n in did]
        entry["not_run"] = [{"line": n, "source": source[n - 1].strip()} for n in counted[path] if n not in did]
        for item in entry["exempt"]:
            item["ran"] = item["line"] in did
        covered, total = covered + len(entry["ran"]), total + len(counted[path])
        if not lines and not entry["errors"]:
            print(f"{path}: no changed executable lines")
            continue
        print(f"{path}: {len(lines)} changed executable line(s) ({ranges(lines)}), {len(entry['ran'])} of "
              f"{len(counted[path])} ran" + (f", {len(entry['exempt'])} exempt" if entry["exempt"] else ""))
        for item in entry["not_run"]:
            print(f"  not run  {item['line']:>5}  {item['source']}")
        for item in entry["exempt"]:
            print(f"  exempt   {item['line']:>5}  {item['reason']}" + ("  (it runs under the tests)" if item["ran"] else ""))
        for item in entry["errors"]:
            print(f"  error    {item['line']:>5}  {item['message']}")
    for e in errors:
        if ": cannot compile" in e:
            print(f"error: {e}")
    pct = 100.0 if total == 0 else math.floor(1000 * covered / total) / 10
    failed_pytest = pytest_code not in (None, 0, 5)
    if failed_pytest:
        print(f"pytest exited {pytest_code}: the gate fails whatever the coverage")
    ok = covered == total and not errors and not failed_pytest
    print(f"DIFF COVERAGE {covered}/{total} ({pct:.1f}%)")
    if json_path:
        report = {"base": args.base, "merge_base": merge_base, "head": head, "dirty": dirty,
                  "pytest": {"args": run_args, "exit": pytest_code, "seconds": round(seconds, 1)},
                  "files": [entry for entry, _ in files], "errors": errors,
                  "covered": covered, "total": total, "percent": pct, "passed": ok}
        json_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
