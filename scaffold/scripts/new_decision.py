#!/usr/bin/env python3
"""Append an entry to DECISIONS.md, refusing to write an incomplete one.

The gate is the point: an entry that does not say how you will know it worked,
and what you will do if it does not, is not an entry. This script will not let
you write one.

Usage:
    python3 scripts/new_decision.py "non-centred group scale"
    python3 scripts/new_decision.py "..." --from-file draft.md
    python3 scripts/new_decision.py --check          # validate existing entries

Interactive mode prompts for each field. Finish a field with a blank line.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "DECISIONS.md"
MARKER = "<!-- new entries are inserted directly below this line -->"

FIELDS = [
    ("What", "What is changing, concretely. Files, configuration, data."),
    ("Why", "Why you believe it will help. Name the competing explanation "
            "and what would distinguish it."),
    ("How we'll know it worked", "Specific diagnostics with thresholds, set "
                                 "BEFORE the run. Not 'we'll look at the output'."),
    ("Kill criterion", "The outcome that means this DIRECTION is dead, not just "
                       "that this run missed."),
    ("If it fails", "The concrete next action. Not 'try something else'."),
    ("Files", "New, edited, untouched."),
]

# a field whose body is only one of these is not a real answer
EMPTY_ANSWERS = {
    "", "tbd", "todo", "n/a", "na", "none", "-", "?", "unknown",
    "we'll see", "well see", "look at the output", "try something else",
}

MIN_CHARS = 25


def looks_empty(body: str) -> str | None:
    """Return a complaint if the body is not a real answer, else None."""
    s = " ".join(body.split())
    if s.strip().lower().rstrip(".") in EMPTY_ANSWERS:
        return "placeholder"
    if len(s) < MIN_CHARS:
        return f"too short ({len(s)} chars, need {MIN_CHARS})"
    return None


def prompt_field(name: str, hint: str) -> str:
    print(f"\n\033[1m{name}\033[0m")
    print(f"  {hint}")
    print("  (end with a blank line)")
    lines: list[str] = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        if line.strip() == "" and not lines:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def build_entry(title: str, bodies: dict[str, str], today: str) -> str:
    out = [f"### {title} ({today})", ""]
    for name, _ in FIELDS:
        out.append(f"**{name}.** {bodies[name]}")
        out.append("")
    out.append("**Result.** _(pending — fill in after the run)_")
    out.append("")
    return "\n".join(out)


def insert(entry: str) -> None:
    text = DECISIONS.read_text()
    if MARKER not in text:
        sys.exit(f"marker not found in {DECISIONS}; is the file intact?")
    text = text.replace(MARKER, MARKER + "\n\n" + entry, 1)
    DECISIONS.write_text(text)


def check() -> int:
    """Validate every entry in the file. Returns an exit code."""
    text = DECISIONS.read_text()
    active = text.split("## Active", 1)[-1].split("## Resolved", 1)[0]
    entries = re.split(r"^### ", active, flags=re.M)[1:]
    if not entries:
        print("no active entries.")
        return 0
    bad = 0
    for e in entries:
        title = e.splitlines()[0].strip()
        missing = [n for n, _ in FIELDS if f"**{n}.**" not in e]
        weak = []
        for n, _ in FIELDS:
            m = re.search(rf"\*\*{re.escape(n)}\.\*\*(.*?)(?=\n\*\*|\Z)", e, re.S)
            if m:
                why = looks_empty(m.group(1))
                if why:
                    weak.append(f"{n} ({why})")
        if missing or weak:
            bad += 1
            print(f"\n  {title}")
            for n in missing:
                print(f"    MISSING  {n}")
            for w in weak:
                print(f"    WEAK     {w}")
    if bad:
        print(f"\n{bad} entr{'y' if bad == 1 else 'ies'} incomplete.")
        return 1
    print(f"{len(entries)} active entr{'y' if len(entries) == 1 else 'ies'}, all complete.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("title", nargs="?", help="short title for the entry")
    ap.add_argument("--from-file", type=Path,
                    help="read the body from a file with '**Field.**' headings "
                         "instead of prompting")
    ap.add_argument("--check", action="store_true",
                    help="validate existing active entries and exit")
    args = ap.parse_args()

    if not DECISIONS.exists():
        sys.exit(f"{DECISIONS} not found — run from the project root.")
    if args.check:
        return check()
    if not args.title:
        ap.error("a title is required (or pass --check)")

    bodies: dict[str, str] = {}
    if args.from_file:
        src = args.from_file.read_text()
        for name, _ in FIELDS:
            m = re.search(rf"\*\*{re.escape(name)}\.\*\*(.*?)(?=\n\*\*|\Z)", src, re.S)
            bodies[name] = m.group(1).strip() if m else ""
    else:
        for name, hint in FIELDS:
            bodies[name] = prompt_field(name, hint)

    complaints = []
    for name, _ in FIELDS:
        why = looks_empty(bodies.get(name, ""))
        if why:
            complaints.append(f"  {name}: {why}")
    if complaints:
        print("\nRefusing to write an incomplete entry:\n" + "\n".join(complaints),
              file=sys.stderr)
        print("\nThe fields exist because writing them changes the run. "
              "Answer them and try again.", file=sys.stderr)
        return 1

    today = dt.date.today().isoformat()
    entry = build_entry(args.title, bodies, today)
    insert(entry)
    print(f"\nappended to {DECISIONS.relative_to(ROOT)}:\n")
    print(entry)
    print("Fill in **Result** once the run completes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
