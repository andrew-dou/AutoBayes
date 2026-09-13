"""Run a command and record its real return code (docs/m3b1-decision.md, cap correction: the shell wrappers used
on 2026-09-13 piped Python through grep and echoed grep's status, so a Python traceback was summarised as exit 0).
Binding implementation (2026-09-13): the log's parent directory is created if missing and the log path must lie
OUTSIDE any unit directory, so a refusal or dry-run that never creates a unit still gets a record of its real exit
code; the wrapper records only the command, cwd, child pid, times and return code. Head, hashes and environment
provenance live in the binding's own log and unit record (f7_direct_cdf_binding.py never imports this wrapper).
Usage: run_logged.py RUN.json -- command args...   (exits with the command's own return code)"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
cmd = sys.argv[3:] if sys.argv[2] == "--" else sys.argv[2:]
out.parent.mkdir(parents=True, exist_ok=True)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
child = subprocess.Popen(cmd)
returncode = child.wait()
out.write_text(json.dumps({"command": cmd, "cwd": os.getcwd(), "child_pid": child.pid, "started": started,
                           "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "returncode": returncode, "status": "ok" if returncode == 0 else "FAILED"}, indent=1) + "\n")
print(f"[run_logged] returncode {returncode} -> {out}")
sys.exit(returncode)
