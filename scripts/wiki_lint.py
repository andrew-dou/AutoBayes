#!/usr/bin/env python3
"""Lint the wiki: broken links, stale PENDING claims, orphans, revived dead ends.

Run before any handoff, and periodically during a long session.

    python3 scripts/wiki_lint.py
    python3 scripts/wiki_lint.py --stale-days 30
    python3 scripts/wiki_lint.py --quiet     # exit code only

Exit code 1 if anything is flagged, so it works in CI or a pre-commit hook.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
INDEX = WIKI / "index.md"
RULED_OUT = WIKI / "hypotheses" / "ruled-out.md"
DECISIONS = ROOT / "DECISIONS.md"

LINK = re.compile(r"\[\[([^\]]+)\]\]")
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
STATUS = re.compile(r"^\*\*Status:\*\*\s*(.+)$", re.M)
UPDATED = re.compile(r"^\*\*Last updated:\*\*\s*(.+)$", re.M)
PENDING = re.compile(r"\b(PENDING|TBD|TODO|pending)\b")


def pages() -> dict[str, Path]:
    """Map every resolvable page name to its file."""
    out: dict[str, Path] = {}
    for p in WIKI.rglob("*.md"):
        rel = p.relative_to(WIKI).with_suffix("")
        out[str(rel)] = p                      # hypotheses/ruled-out
        out[p.stem] = p                        # ruled-out
        out[f"wiki/{rel}"] = p                 # wiki/hypotheses/ruled-out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stale-days", type=int, default=30,
                    help="flag PENDING claims older than this (default 30)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not WIKI.is_dir():
        sys.exit(f"{WIKI} not found — run from the project root.")

    known = pages()
    md = sorted(WIKI.rglob("*.md"))
    problems: list[tuple[str, str]] = []
    today = dt.date.today()

    # --- broken cross-references --------------------------------------------
    for p in md:
        text = p.read_text()
        for target in LINK.findall(text):
            t = target.strip().split("|")[0].strip()
            if t and t not in known:
                problems.append(("broken link",
                                 f"{p.relative_to(ROOT)} -> [[{t}]]"))

    # --- pages missing from the index ---------------------------------------
    if INDEX.exists():
        index_text = INDEX.read_text()
        indexed = {t.strip().split("|")[0].strip() for t in LINK.findall(index_text)}
        for p in md:
            if p in (INDEX, WIKI / "log.md"):
                continue
            rel = str(p.relative_to(WIKI).with_suffix(""))
            if rel not in indexed and p.stem not in indexed \
                    and f"wiki/{rel}" not in indexed:
                problems.append(("not in index", str(p.relative_to(ROOT))))

    # --- stale PENDING ------------------------------------------------------
    for p in md:
        text = p.read_text()
        if not PENDING.search(text):
            continue
        m = UPDATED.search(text)
        stamp = None
        if m:
            d = DATE.search(m.group(1))
            if d:
                stamp = d.group(1)
        if stamp is None:
            problems.append(("PENDING, undated",
                             f"{p.relative_to(ROOT)} (add '**Last updated:**')"))
            continue
        age = (today - dt.date.fromisoformat(stamp)).days
        if age > args.stale_days:
            problems.append(("PENDING, stale",
                             f"{p.relative_to(ROOT)} — {age} days since {stamp}"))

    # --- missing required headers -------------------------------------------
    for p in md:
        if p in (INDEX, WIKI / "log.md") or p.name == ".gitkeep":
            continue
        text = p.read_text()
        if not STATUS.search(text):
            problems.append(("no Status header", str(p.relative_to(ROOT))))

    # --- ruled-out approaches reappearing as active proposals ---------------
    if RULED_OUT.exists() and DECISIONS.exists():
        killed = [h.strip() for h in
                  re.findall(r"^### (.+?) — ruled out", RULED_OUT.read_text(), re.M)]
        active = DECISIONS.read_text().split("## Active", 1)[-1] \
                                      .split("## Resolved", 1)[0].lower()
        for k in killed:
            key = k.lower().strip()
            if len(key) > 8 and key in active:
                problems.append(("ruled-out approach is active",
                                 f"'{k}' appears in DECISIONS.md Active"))

    if args.quiet:
        return 1 if problems else 0

    if not problems:
        n = len([p for p in md if p.name != ".gitkeep"])
        print(f"wiki clean: {n} page(s), no problems.")
        return 0

    by_kind: dict[str, list[str]] = {}
    for kind, detail in problems:
        by_kind.setdefault(kind, []).append(detail)
    print(f"{len(problems)} problem(s):\n")
    for kind in sorted(by_kind):
        print(f"  {kind}")
        for d in sorted(by_kind[kind]):
            print(f"    {d}")
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
