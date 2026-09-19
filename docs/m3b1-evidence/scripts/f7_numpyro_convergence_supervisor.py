"""Standard-library supervisor for the single approved run: creates the exclusive attempt directory, launches ONE child
(f7_numpyro_convergence_run.py --run DIR) in its own process group with stdout/stderr files, waits at most 300 s total
wall from launch, and on timeout terminates only that process group (SIGTERM, then SIGKILL after 10 s). Records the real
return code, timeout flag, signal and wall time in supervisor.json. No retry, no restart, no takeover."""
import datetime, json, os, signal, subprocess, sys, time
from pathlib import Path

UNIT = Path(__file__).resolve().parent
approval = json.loads((UNIT / "reviewed-unit.json").read_text())
attempt = Path(approval["attempt_dir"]); allowance = int(approval["max_total_wall_seconds"])
python = "/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3b1-varying-slope/.venv/bin/python"
os.mkdir(attempt)  # exclusive, atomic: an existing directory refuses a second attempt
record = {"attempt_dir": str(attempt), "allowance_seconds": allowance, "python": python, "child_script": str(UNIT / "f7_numpyro_convergence_run.py"),
          "launched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "supervisor_pid": os.getpid()}
out, err = open(attempt / "child.stdout.txt", "w"), open(attempt / "child.stderr.txt", "w")
t0 = time.monotonic()
child = subprocess.Popen([python, str(UNIT / "f7_numpyro_convergence_run.py"), "--run", str(attempt)], stdout=out, stderr=err,
                         cwd="/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3b1-varying-slope", start_new_session=True)
record.update(child_pid=child.pid, child_pgid=os.getpgid(child.pid), timed_out=False, terminated_with=None)
(attempt / "supervisor.json").write_text(json.dumps(record, indent=1) + "\n")
try:
    rc = child.wait(timeout=allowance)
except subprocess.TimeoutExpired:
    record["timed_out"] = True
    os.killpg(record["child_pgid"], signal.SIGTERM); record["terminated_with"] = "SIGTERM"
    try:
        rc = child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(record["child_pgid"], signal.SIGKILL); record["terminated_with"] = "SIGTERM then SIGKILL"; rc = child.wait()
record.update(returncode=rc, wall_seconds=round(time.monotonic() - t0, 3), finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              status="completed" if (rc == 0 and not record["timed_out"]) else ("timeout" if record["timed_out"] else f"failed (returncode {rc})"))
out.close(); err.close()
(attempt / "supervisor.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record, indent=1)); sys.exit(0 if rc == 0 and not record["timed_out"] else 1)
