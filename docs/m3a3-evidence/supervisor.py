"""Exclusive bounded numerical invocation; aggregate focused budget and one full suite."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, signal, subprocess, sys, time
ROOT=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3a3-multiple-interactions')
BASE=Path('/private/tmp/autobayes-m3a3-implementation')
kind,name,*args=sys.argv[1:]
assert kind in ('focused','full') and args
runs=BASE/'runs';runs.mkdir(exist_ok=True)
used=0.
for prior in runs.iterdir():
    launch=json.loads((prior/'launch.json').read_text())
    assert (prior/'exit.json').exists(),f'Unfinished run {prior}'
    if kind=='full' and launch['kind']=='full':raise SystemExit('Full-suite allowance already used')
    if launch['kind']=='focused':used+=json.loads((prior/'exit.json').read_text())['wall_seconds']
cap=900 if kind=='full' else min(300,900-used)
if cap<=0:raise SystemExit('Focused budget exhausted')
out=runs/name;out.mkdir(exist_ok=False)
command=[str(ROOT/'.venv/bin/python'),'-B',*args]
cache=BASE/'cache';cache.mkdir(exist_ok=True)
overrides=dict(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',AUTOBAYES_HOST_DEVICES='4',
    MPLCONFIGDIR=str(cache/'matplotlib'),PYTENSOR_FLAGS=f'compiledir={cache}/pytensor,floatX=float64',
    NUMBA_CACHE_DIR=str(cache/'numba'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
    AUTOBAYES_EVIDENCE_DIR=str(out/'fixture-evidence'))
files=list((ROOT/'autobayes').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[
    ROOT/'uv.lock',ROOT/'pyproject.toml',ROOT/'docs/m3a3-decision.md',ROOT/'README.md',ROOT/'SKILL.md']
example=ROOT/'examples/m3a3_multiple_interactions.py'
if example.exists():files.append(example)
snapshots=out/'source';snapshots.mkdir()
hashes={}
for p in files:
    target=snapshots/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
launch=dict(kind=kind,command=command,cwd=str(ROOT),cap_seconds=cap,focused_seconds_before=used,
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment=overrides,
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),source_sha256=hashes)
(out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
(out/'working.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
start=time.monotonic();timed_out=False
with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
    proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**overrides},stdout=stdout,stderr=stderr,start_new_session=True)
    (out/'pid.json').write_text(json.dumps({'pid':proc.pid,'pgid':proc.pid})+'\n')
    try:code=proc.wait(timeout=cap)
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
             finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'directory':str(out),**receipt}),flush=True)
raise SystemExit(code)
