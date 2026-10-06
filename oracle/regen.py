#!/usr/bin/env python
"""Freeze or check brms oracle fixtures (local only: needs R with brms, rstan and jsonlite; CI never runs this).

    python oracle/regen.py doctor                       # rstan compiles a brms model on this machine
    python oracle/regen.py new <id> [<id> ...] [-j N]   # freeze fixtures for cases that have none
    python oracle/regen.py check <id>|--all [-j N]      # regenerate into a temporary directory and compare

A frozen fixture is never overwritten: a changed case gets a new id, and the old fixture stays for the
history of what was compared. `check` reports any difference between the committed fixture and a fresh
generation (byte-equal inputs; numerically equal outputs) and exits 1 if there is one.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.oracle import datasets  # noqa: E402
from tests.oracle import fixture as fx  # noqa: E402

RSCRIPT = ["Rscript", str(REPO / "oracle" / "R" / "gen_fixture.R")]
IGNORED_META = {"compile_seconds"}


def write_inputs(case: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data, nd = datasets.make(case["data"])
    datasets.write_csv(data, out / "data.csv")
    if nd is not None:
        datasets.write_csv(nd, out / "newdata.csv")
    (out / "case_brms.json").write_text(fx.canonical_json(fx.frozen_section(case)), encoding="utf-8")


def generate(case_id: str, out: Path) -> None:
    write_inputs(fx.load_case(case_id), out)
    subprocess.run([*RSCRIPT, str(out)], check=True)


def freeze(case_id: str) -> str:
    out = fx.fixture_dir(case_id)
    if (out / "fixture.json").exists():
        return f"{case_id}: already frozen (a changed case needs a new id; `check` compares)"
    generate(case_id, out)
    return f"{case_id}: frozen in {out.relative_to(REPO)}"


def _diff(a, b, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if path == "meta" and k in IGNORED_META:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'committed' if k in a else 'regenerated'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > 1e-12 * max(1.0, abs(a), abs(b)):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def check(case_id: str) -> str:
    committed = fx.fixture_dir(case_id)
    if not (committed / "fixture.json").exists():
        return f"{case_id}: no frozen fixture"
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / case_id
        generate(case_id, fresh)
        problems = []
        for name in ("case_brms.json", "data.csv", "newdata.csv"):
            a, b = committed / name, fresh / name
            if a.exists() != b.exists():
                problems.append(f"{name}: present only in {'committed' if a.exists() else 'regenerated'}")
            elif a.exists() and a.read_bytes() != b.read_bytes():
                problems.append(f"{name}: bytes differ")
        with open(committed / "fixture.json", encoding="utf-8") as f:
            old = json.load(f)
        with open(fresh / "fixture.json", encoding="utf-8") as f:
            new = json.load(f)
        _diff(old, new, "", problems)
    if problems:
        return f"{case_id}: DIFFERS\n  " + "\n  ".join(problems[:40]) + ("\n  ..." if len(problems) > 40 else "")
    return f"{case_id}: identical"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p_new = sub.add_parser("new")
    p_new.add_argument("ids", nargs="*")
    p_new.add_argument("--all", action="store_true", help="every case without a fixture")
    p_new.add_argument("-j", "--jobs", type=int, default=2)
    p_check = sub.add_parser("check")
    p_check.add_argument("ids", nargs="*")
    p_check.add_argument("--all", action="store_true")
    p_check.add_argument("-j", "--jobs", type=int, default=2)
    args = ap.parse_args(argv)

    if args.command == "doctor":
        return subprocess.run(["Rscript", str(REPO / "oracle" / "R" / "doctor.R")]).returncode
    ids = list(args.ids)
    if args.all:
        ids = [c for c in fx.case_ids() if (args.command == "check") == fx.has_fixture(c)]
    if not ids:
        ap.error("give case ids or --all")
    work = freeze if args.command == "new" else check
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(work, ids))
    print("\n".join(results))
    return 1 if any("DIFFERS" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
