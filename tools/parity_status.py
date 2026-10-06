#!/usr/bin/env python
"""Print the brms-parity ledger and cross-check it against GitHub's pull requests.

    python tools/parity_status.py [--no-github] [--tier A]

Every `pr_open` entry must have an OPEN pull request and every `merged` entry a MERGED one; mismatches are
listed at the end and make the exit status 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "brms-parity" / "ledger.json"


def github_prs() -> dict[int, dict]:
    out = subprocess.run(["gh", "pr", "list", "--state", "all", "--limit", "200", "--json", "number,state,headRefName,title"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return {pr["number"]: pr for pr in json.loads(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-github", action="store_true")
    ap.add_argument("--tier")
    args = ap.parse_args(argv)
    with open(LEDGER, encoding="utf-8") as f:
        ledger = json.load(f)
    features = [e for e in ledger["features"] if not args.tier or e["tier"] == args.tier]
    prs = {} if args.no_github else github_prs()

    print(f"{'id':36} {'tier':4} {'lane':9} {'status':16} {'pr':5} {'gates':14} depends_on")
    problems = []
    for e in features:
        gates = e.get("gates") or {}
        gate_text = gates.get("sha", "") if gates else ""
        if gates.get("results"):
            gate_text += " " + "".join("." if r["status"] == "pass" else ("a" if r["status"] == "advisory_fail" else "F")
                                       for r in gates["results"].values())
        print(f"{e['id']:36} {e['tier']:4} {e['lane']:9} {e['status']:16} {str(e.get('pr') or ''):5} {gate_text:14} {','.join(e['depends_on'])}")
        if prs:
            pr = prs.get(e.get("pr") or -1)
            if e["status"] == "pr_open" and (pr is None or pr["state"] != "OPEN"):
                problems.append(f"{e['id']}: ledger says pr_open but GitHub says {pr['state'] if pr else 'no such PR'}")
            if e["status"] == "merged" and (pr is None or pr["state"] != "MERGED"):
                problems.append(f"{e['id']}: ledger says merged but GitHub says {pr['state'] if pr else 'no such PR'}")
    counts = {}
    for e in features:
        counts[e["status"].split(":")[0]] = counts.get(e["status"].split(":")[0], 0) + 1
    print("\n" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if problems:
        print("\nledger/GitHub mismatches:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
