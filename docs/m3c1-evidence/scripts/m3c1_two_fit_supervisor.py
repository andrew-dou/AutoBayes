"""Standard-library supervisor for the M3c-1 two-fit validation: creates the exclusive attempt directory, launches ONE
child (m3c1_two_fit_validation.py) in its own process group with stdout/stderr files and a scratch PyTensor compile
directory, and enforces a single hard 600 s elapsed deadline from launch covering imports, compilation, both fits and
serialization; on expiry it terminates only that process group (SIGTERM, then SIGKILL after 10 s). Records the real
return code, timeout flag and wall time in supervisor.json. No retry, no second attempt, no configuration change.
Usage: m3c1_two_fit_supervisor.py OUT_PARENT"""
import datetime, json, os, signal, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
parent = Path(sys.argv[1]); parent.mkdir(parents=True, exist_ok=True)
attempt = parent / "attempt-1"
os.mkdir(attempt)                                       # exclusive, atomic: an existing directory refuses a second attempt
python = str(REPO / ".venv/bin/python"); deadline = 600
env = dict(os.environ, PYTENSOR_FLAGS=f"base_compiledir={attempt / 'pytensor-compiledir'}", PYTHONDONTWRITEBYTECODE="1")
record = {"attempt_dir": str(attempt), "deadline_seconds": deadline, "python": python, "child_script": str(REPO / "docs/m3c1-evidence/scripts/m3c1_two_fit_validation.py"),
          "launched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "supervisor_pid": os.getpid(), "cwd": str(REPO)}
out, err = open(attempt / "child.stdout.txt", "w"), open(attempt / "child.stderr.txt", "w")
t0 = time.monotonic()
child = subprocess.Popen([python, record["child_script"], str(attempt)], stdout=out, stderr=err, cwd=str(REPO), env=env, start_new_session=True)
record.update(child_pid=child.pid, child_pgid=os.getpgid(child.pid), timed_out=False, terminated_with=None)
(attempt / "supervisor.json").write_text(json.dumps(record, indent=1) + "\n")
try:
    rc = child.wait(timeout=deadline)
except subprocess.TimeoutExpired:
    record["timed_out"] = True
    os.killpg(record["child_pgid"], signal.SIGTERM); record["terminated_with"] = "SIGTERM"
    try:
        rc = child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(record["child_pgid"], signal.SIGKILL); record["terminated_with"] = "SIGTERM then SIGKILL"; rc = child.wait()
record.update(returncode=rc, wall_seconds=round(time.monotonic() - t0, 3), finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              status="completed" if (rc == 0 and not record["timed_out"]) else ("timeout" if record["timed_out"] else f"completed with returncode {rc}"))
out.close(); err.close()
(attempt / "supervisor.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record, indent=1)); sys.exit(0 if rc == 0 and not record["timed_out"] else 1)
