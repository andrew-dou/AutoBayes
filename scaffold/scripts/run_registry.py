#!/usr/bin/env python3
"""A registry of what was actually run: config -> output directory -> wall clock.

Six months on you will be asked how long a fit took and which settings produced
it. The scheduler's accounting database will have been purged and the logs will
have rotated. This is a flat JSONL file that outlives both.

    python3 scripts/run_registry.py add --tag v2 --outdir fits/v2 \
        --chains 40 --draws 2500 --warmup 2500 --seconds 20045
    python3 scripts/run_registry.py list
    python3 scripts/run_registry.py list --sort seconds
    python3 scripts/run_registry.py show v2

The registry is append-only. Re-registering a tag adds a row; `show` reports all
of them, because two runs with the same tag and different wall clocks is exactly
the discrepancy you want surfaced rather than silently overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "runs.jsonl"

COLUMNS = [
    ("tag", 22), ("date", 11), ("chains", 7), ("draws", 7),
    ("warmup", 7), ("td", 4), ("wall", 9), ("outdir", 34),
]


def load() -> list[dict]:
    if not REGISTRY.exists():
        return []
    out = []
    for line in REGISTRY.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def hhmm(seconds) -> str:
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "?"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def row(r: dict) -> str:
    vals = {
        "tag": r.get("tag", "?"),
        "date": (r.get("registered_at") or "")[:10],
        "chains": r.get("chains", ""),
        "draws": r.get("draws", ""),
        "warmup": r.get("warmup", ""),
        "td": r.get("max_treedepth", ""),
        "wall": hhmm(r.get("seconds")),
        "outdir": r.get("outdir", ""),
    }
    return "  ".join(f"{str(vals[c]):<{w}}"[:w] for c, w in COLUMNS)


def header() -> str:
    return ("  ".join(f"{c:<{w}}" for c, w in COLUMNS) + "\n"
            + "  ".join("-" * w for _, w in COLUMNS))


def cmd_add(args) -> int:
    rec = {
        "tag": args.tag,
        "outdir": args.outdir,
        "job_id": args.job_id,
        "chains": args.chains,
        "draws": args.draws,
        "warmup": args.warmup,
        "max_treedepth": args.max_treedepth,
        "seconds": args.seconds,
        "device": args.device,
        "note": args.note,
        "registered_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    rec = {k: v for k, v in rec.items() if v is not None}
    with REGISTRY.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")

    prior = [r for r in load()[:-1] if r.get("tag") == args.tag]
    print(f"registered {args.tag}  ({hhmm(args.seconds)})")
    if prior:
        print(f"  note: {len(prior)} earlier run(s) share this tag:")
        for p in prior:
            print(f"    {hhmm(p.get('seconds'))}  {p.get('outdir', '')}")
        print("  if these should agree and do not, that is worth resolving now.")
    return 0


def cmd_list(args) -> int:
    rows = load()
    if not rows:
        print("registry empty.")
        return 0
    if args.sort == "seconds":
        rows.sort(key=lambda r: -(r.get("seconds") or 0))
    elif args.sort == "tag":
        rows.sort(key=lambda r: str(r.get("tag", "")))
    print(header())
    for r in rows:
        print(row(r))
    total = sum(r.get("seconds") or 0 for r in rows)
    print(f"\n{len(rows)} run(s), {total / 3600:.1f} GPU-hours registered.")
    return 0


def cmd_show(args) -> int:
    rows = [r for r in load() if r.get("tag") == args.tag]
    if not rows:
        print(f"no runs registered under tag {args.tag!r}.", file=sys.stderr)
        return 1
    for i, r in enumerate(rows, 1):
        print(f"--- {args.tag} [{i}/{len(rows)}] ---")
        for k in sorted(r):
            v = hhmm(r[k]) if k == "seconds" else r[k]
            print(f"  {k:16} {v}")
    if len({r.get("seconds") for r in rows}) > 1:
        print("\nwall clocks differ across runs with this tag — check the "
              "configurations before quoting one of them.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register a completed run")
    a.add_argument("--tag", required=True)
    a.add_argument("--outdir", required=True)
    a.add_argument("--job-id")
    a.add_argument("--chains", type=int)
    a.add_argument("--draws", type=int)
    a.add_argument("--warmup", type=int)
    a.add_argument("--max-treedepth", type=int)
    a.add_argument("--seconds", type=int)
    a.add_argument("--device")
    a.add_argument("--note")
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="list registered runs")
    l.add_argument("--sort", choices=["date", "seconds", "tag"], default="date")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show every run under a tag")
    s.add_argument("tag")
    s.set_defaults(func=cmd_show)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
