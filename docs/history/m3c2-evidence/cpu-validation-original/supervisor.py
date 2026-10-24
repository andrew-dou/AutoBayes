"""Standard-library-only, one exclusive attempt, 600s child-group deadline."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/private/tmp/autobayes-intercept-cpu-validation')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')
ATTEMPT = ROOT / 'attempt-1'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    # Refuse to overwrite or retry an existing attempt, even when it failed.
    ATTEMPT.mkdir(exist_ok=False)
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    for entry in frozen['inputs']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    cmd = [str(REPO / '.venv/bin/python'), '-B', '-u', str(ROOT / 'acceptance.py')]
    overrides = frozen['config']['env_overrides']
    receipt = {'command': cmd, 'cwd': str(ATTEMPT), 'env_overrides': overrides,
        'started': now(), 'deadline_seconds': 600, 'term_grace_seconds': 3,
        'freeze_sha256': hashlib.sha256((ROOT / 'freeze.json').read_bytes()).hexdigest(),
        'head': frozen['head'], 'timeout': False, 'term_sent': False, 'kill_sent': False}
    with (ATTEMPT / 'launch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    start = time.monotonic()
    with (ATTEMPT / 'stdout.txt').open('x') as stdout, (ATTEMPT / 'stderr.txt').open('x') as stderr:
        proc = subprocess.Popen(cmd, cwd=ATTEMPT, env={**os.environ, **overrides}, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        receipt['pid'] = proc.pid
        try:
            receipt['child_exit_code'] = proc.wait(timeout=max(0.0, 600 - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            receipt['timeout'] = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                receipt['term_sent'] = True
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # Kill any surviving descendants in this child group even if its leader already exited.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                receipt['kill_sent'] = True
            except ProcessLookupError:
                pass
            receipt['child_exit_code'] = proc.wait()
    receipt['wall_seconds'] = time.monotonic() - start
    receipt['finished'] = now()
    receipt['supervisor_exit_code'] = 124 if receipt['timeout'] else receipt['child_exit_code']
    with (ATTEMPT / 'exit-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt['supervisor_exit_code']

if __name__ == '__main__':
    raise SystemExit(main())
