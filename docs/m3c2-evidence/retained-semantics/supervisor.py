"""One read-only numerical invocation with a 120s child-group deadline."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/private/tmp/autobayes-intercept-retained-semantics')
REPO = Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')

def main():
    guard = ROOT / 'invocation-started.json'
    command = [str(REPO / '.venv/bin/python'), '-B', '-u', str(ROOT / 'review.py')]
    env = {'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(REPO),'PYTENSOR_FLAGS':'compiledir='+str(ROOT/'pytensor-cache'),
           'MPLCONFIGDIR':str(ROOT/'matplotlib-cache'),'JAX_PLATFORMS':'cpu','AUTOBAYES_HOST_DEVICES':'4',
           'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    receipt = {'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'cwd':str(ROOT),
        'env_overrides':env,'deadline_seconds':120,'timeout':False,'new_sampling_forbidden':True,
        'scripts':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['review.py','supervisor.py']}}
    with guard.open('x') as out: json.dump(receipt,out,indent=2);out.write('\n')
    start = time.monotonic()
    with (ROOT/'stdout.txt').open('x') as stdout, (ROOT/'stderr.txt').open('x') as stderr:
        proc = subprocess.Popen(command,cwd=ROOT,env={**os.environ,**env},stdout=stdout,stderr=stderr,start_new_session=True)
        receipt['pid'] = proc.pid
        try: code = proc.wait(timeout=max(0.,120-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            receipt['timeout'] = True
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            code = proc.wait()
    receipt.update(child_exit_code=code,exit_code=124 if receipt['timeout'] else code,
        wall_seconds=time.monotonic()-start,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (ROOT/'exit-receipt.json').open('x') as out: json.dump(receipt,out,indent=2);out.write('\n')
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt['exit_code']

if __name__ == '__main__': raise SystemExit(main())
