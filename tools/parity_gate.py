#!/usr/bin/env python
"""Run the brms-parity gates on the current checkout and, with --feature, record the result in the ledger.

    python tools/parity_gate.py [--feature ID] [--base origin/main] [--smoke] [--skip-suite]

Checks, in order (cheapest first), stopping at the first failure unless --keep-going:
  immutability  tests/oracle/fixtures/** only added since --base; tests/oracle/tolerances.py unchanged;
                changes to tests/oracle/{gates,adapter}.py reported
  oracle        pytest tests/oracle  (G1-G6)
  suite         pytest tests -m "not slow and not sampling"  (G7, local part; hosted CI runs "not slow")
  smoke         with --smoke only: one 120-second sampling smoke, advisory, zero retries (G8)

The ledger record carries the commit SHA, the gate statuses and the wall time of each step. Exit 1 on any
failing gate (the smoke never fails the run; its outcome is recorded as advisory).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "brms-parity" / "ledger.json"
PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]


def run(cmd, timeout=None):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, time.perf_counter() - t0, (proc.stdout + proc.stderr)[-4000:]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def check_immutability(base: str) -> tuple[bool, str]:
    status = git("diff", "--name-status", f"{base}...HEAD", "--", "tests/oracle")
    problems, notes = [], []
    for line in status.splitlines():
        code, _, path = line.partition("\t")
        if code.startswith("A"):
            continue                                       # new fixtures and new harness files are what a branch may add
        if path.startswith("tests/oracle/fixtures/"):
            problems.append(f"{code} {path} (fixtures are frozen: a changed case needs a new id)")
        if path == "tests/oracle/tolerances.py":
            problems.append(f"{code} {path} (tolerances are a contract)")
        if path in ("tests/oracle/gates.py", "tests/oracle/adapter.py"):
            notes.append(f"{code} {path} changed: the verifier must confirm this only extends the harness")
    text = "\n".join(problems + notes) or "no fixture, tolerance, gate or adapter changes"
    return not problems, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", help="ledger id to record the result under")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--smoke", action="store_true", help="also run the advisory 120-second sampling smoke")
    ap.add_argument("--skip-suite", action="store_true", help="skip the non-sampling suite (oracle gates only)")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args(argv)

    sha = git("rev-parse", "--short=12", "HEAD").strip()
    dirty = bool(git("status", "--porcelain").strip())
    results: dict[str, dict] = {}
    failed = False

    def record(name, ok, seconds, detail, advisory=False):
        nonlocal failed
        results[name] = {"status": "pass" if ok else ("advisory_fail" if advisory else "fail"), "seconds": round(seconds, 1)}
        print(f"[{results[name]['status']:>13}] {name} ({seconds:.0f} s)")
        if not ok:
            print(detail.strip()[-2500:])
            if not advisory:
                failed = True

    t0 = time.perf_counter()
    ok, text = check_immutability(args.base)
    record("immutability", ok, time.perf_counter() - t0, text)
    print("  " + text.replace("\n", "\n  "))
    if failed and not args.keep_going:
        return finish(args, sha, dirty, results)

    env = dict(os.environ)
    env.setdefault("PYTENSOR_FLAGS", f"base_compiledir={REPO / '.pytensor-gate'}")
    for name, cmd in (("oracle", [*PYTEST, "tests/oracle"]),
                      ("suite", [] if args.skip_suite else [*PYTEST, "tests", "-m", "not slow and not sampling"])):
        if not cmd:
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
        record(name, proc.returncode == 0, time.perf_counter() - t0, proc.stdout + proc.stderr)
        if failed and not args.keep_going:
            return finish(args, sha, dirty, results)
    if args.smoke:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([*PYTEST, "tests/test_predict.py", "-m", "sampling", "-x"], cwd=REPO,
                                  capture_output=True, text=True, env=env, timeout=120)
            ok, detail = proc.returncode == 0, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            ok, detail = False, "the 120-second smoke timed out (advisory; not retried)"
        record("smoke", ok, time.perf_counter() - t0, detail, advisory=True)
    return finish(args, sha, dirty, results)


def finish(args, sha, dirty, results) -> int:
    failed = any(r["status"] == "fail" for r in results.values())
    if args.feature:
        with open(LEDGER, encoding="utf-8") as f:
            ledger = json.load(f)
        entry = next((e for e in ledger["features"] if e["id"] == args.feature), None)
        if entry is None:
            print(f"no ledger entry {args.feature!r}; not recorded", file=sys.stderr)
            return 2
        entry["gates"] = {"sha": sha, "dirty_tree": dirty, "base": args.base,
                          "when": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "results": results}
        if not failed and entry["status"] in ("implementing", "fixtures_frozen"):
            entry["status"] = "gates_green"
        with open(LEDGER, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"recorded under {args.feature} at {sha}{' (dirty tree)' if dirty else ''}")
    print("GATES FAILED" if failed else "GATES GREEN")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
