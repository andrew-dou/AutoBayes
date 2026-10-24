from pathlib import Path
import subprocess,os,json,time,datetime,hashlib,signal
w=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/fit-binding');d=Path('/private/tmp/iqss-fit-binding-check');d.mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONPATH=str(w),PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',PYTENSOR_FLAGS='base_compiledir=/private/tmp/iqss-student-probe',MPLCONFIGDIR='/private/tmp/iqss-review-mpl',NUMBA_CACHE_DIR='/private/tmp/iqss-review-numba')
cmd=['/Users/andrewdou/iqss-research/artifacts/2026-09-25-autobayes-pr12-closeout/retained-venv/bin/python','-m','pytest','tests/test_fit_binding.py','tests/test_record_identity.py','tests/test_student.py','tests/test_poisson.py','tests/test_family_contract.py','-q','-p','no:cacheprovider']
source={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(w/'autobayes').glob('*.py'),w/'tests/test_student.py',w/'tests/test_record_identity.py',w/'tests/test_fit_binding.py']}
t=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc)
with (d/'output.txt').open('w') as f:
 p=subprocess.Popen(cmd,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 timeout=False
 while p.poll() is None:
  if max(time.monotonic()-t,(datetime.datetime.now(datetime.timezone.utc)-utc).total_seconds())>=300:
   os.killpg(p.pid,signal.SIGKILL);p.wait();timeout=True;break
  time.sleep(.5)
(d/'receipt.json').write_text(json.dumps({'command':cmd,'exit_code':p.returncode,'timeout':timeout,'elapsed':time.monotonic()-t,'started_utc':utc.isoformat(),'cap':300,'source_hashes':source},indent=2))
print((d/'output.txt').read_text()[-5000:])
