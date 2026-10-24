"""One invocation, process-group deadline, immutable receipt and aggregate budget."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c3-multiple-intercepts')
BASE=Path('/private/tmp/autobayes-m3c3-implementation')
kind,name,*args=sys.argv[1:]
assert kind in ('focused','full') and args
runs=BASE/'runs';runs.mkdir(exist_ok=True)
used=0
for d in runs.iterdir():
    launch=json.loads((d/'launch.json').read_text())
    receipt=d/'exit.json'
    if not receipt.exists(): raise SystemExit(f'Unfinished invocation: {d}')
    if kind=='full' and launch['kind']=='full': raise SystemExit('Full suite allowance already used')
    if launch['kind']=='focused':used+=json.loads(receipt.read_text())['wall_seconds']
cap=900 if kind=='full' else min(300,900-used)
if cap<=0:raise SystemExit('Focused budget exhausted')
out=runs/name;out.mkdir(exist_ok=False)
command=[str(ROOT/'.venv/bin/python'),*args]
env=os.environ.copy()
cache=BASE/'cache';cache.mkdir(exist_ok=True)
env.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',AUTOBAYES_HOST_DEVICES='4',
           MPLCONFIGDIR=str(cache/'matplotlib'),PYTENSOR_FLAGS=f'compiledir={cache}/pytensor',
           NUMBA_CACHE_DIR=str(cache/'numba'),AUTOBAYES_EVIDENCE_DIR=str(out/'fixture-evidence'),
           OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
files=list((ROOT/'autobayes').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'uv.lock',ROOT/'pyproject.toml',ROOT/'docs/m3c3-decision.md',Path(__file__)]
launch=dict(kind=kind,command=command,cwd=str(ROOT),cap_seconds=cap,focused_used_before=used,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            environment={k:env[k] for k in ['JAX_PLATFORMS','AUTOBAYES_HOST_DEVICES','PYTENSOR_FLAGS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MPLCONFIGDIR','NUMBA_CACHE_DIR','AUTOBAYES_EVIDENCE_DIR']},
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
start=time.monotonic(); timed_out=False
with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    (out/'pid.json').write_text(json.dumps({'pid':proc.pid,'pgid':proc.pid})+'\n')
    try: code=proc.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        timed_out=True
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        code=124
receipt=dict(returncode=code,timed_out=timed_out,wall_seconds=time.monotonic()-start,pid=proc.pid,
             utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'directory':str(out),**receipt}),flush=True)
raise SystemExit(code)
