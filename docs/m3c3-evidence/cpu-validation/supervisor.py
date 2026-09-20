"""One exclusive process-group invocation, 600 seconds plus short kill grace."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/private/tmp/autobayes-multiple-intercepts-cpu-validation-1')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c3-multiple-intercepts')

def main():
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    for row in frozen['inputs']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    config = json.loads((ROOT / 'config.json').read_text())
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == frozen['head']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    command = [str(REPO / '.venv/bin/python'), '-B', '-u', str(ROOT / 'run.py')]
    receipt = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': command,
        'cwd': str(ROOT), 'env_overrides': config['env_overrides'], 'deadline_seconds': 600,
        'termination_grace_seconds': 3, 'timeout': False, 'term_sent': False, 'kill_sent': False,
        'freeze_sha256': hashlib.sha256((ROOT / 'freeze.json').read_bytes()).hexdigest()}
    with (ROOT / 'launch.json').open('x') as out: json.dump(receipt, out, indent=2)
    started = time.monotonic()
    with (ROOT / 'stdout.txt').open('x') as stdout, (ROOT / 'stderr.txt').open('x') as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **config['env_overrides']},
                                stdout=stdout, stderr=stderr, start_new_session=True)
        receipt['pid'] = receipt['pgid'] = proc.pid
        with (ROOT / 'pid.json').open('x') as out: json.dump({'pid': proc.pid, 'pgid': proc.pid}, out)
        try: code = proc.wait(timeout=max(0., 600 - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            receipt['timeout'] = True
            try: os.killpg(proc.pid, signal.SIGTERM); receipt['term_sent'] = True
            except ProcessLookupError: pass
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(proc.pid, signal.SIGKILL); receipt['kill_sent'] = True
            except ProcessLookupError: pass
            code = proc.wait()
    try: os.killpg(proc.pid, 0); group_alive = True
    except ProcessLookupError: group_alive = False
    if group_alive:
        os.killpg(proc.pid, signal.SIGTERM); receipt['term_sent'] = True
        time.sleep(.5)
        try: os.killpg(proc.pid, signal.SIGKILL); receipt['kill_sent'] = True
        except ProcessLookupError: pass
    try: os.killpg(proc.pid, 0); group_alive = True
    except ProcessLookupError: group_alive = False
    receipt.update(child_exit_code=code, exit_code=124 if receipt['timeout'] else code,
        wall_seconds=time.monotonic() - started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        child_reaped=proc.returncode is not None, owned_group_still_live=group_alive)
    with (ROOT / 'exit-receipt.json').open('x') as out: json.dump(receipt, out, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt['exit_code']

if __name__ == '__main__': raise SystemExit(main())
