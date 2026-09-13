"""Run a command and record its real return code (docs/m3b1-decision.md, cap correction): the shell wrappers used
on 2026-09-13 piped Python through grep and echoed grep's status, so a Python traceback was summarised as exit 0.
Usage: run_logged.py RUN.json -- command args...   (exits with the command's own return code)"""
import datetime, json, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1]); cmd = sys.argv[3:] if sys.argv[2] == "--" else sys.argv[2:]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = subprocess.run(cmd)
out.write_text(json.dumps({"command": cmd, "started": started, "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "returncode": result.returncode, "status": "ok" if result.returncode == 0 else "FAILED"}, indent=1) + "\n")
print(f"[run_logged] returncode {result.returncode} -> {out}")
sys.exit(result.returncode)
