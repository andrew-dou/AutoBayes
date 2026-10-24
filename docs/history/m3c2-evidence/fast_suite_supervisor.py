from pathlib import Path
import subprocess,os,signal,json,time,datetime,hashlib
w=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')
a=Path('/Users/andrewdou/iqss-research/artifacts/2026-09-19-autobayes-intercept-validation');a.mkdir(exist_ok=False)
command=[str(w/'.venv/bin/python'),'-m','pytest','tests','-q','-m','not slow','-p','no:cacheprovider',f'--junitxml={a}/junit.xml']
env=os.environ.copy();env['AUTOBAYES_EVIDENCE_DIR']=str(a/'test-evidence')
receipt=dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),cwd=str(w),command=command,deadline_seconds=600,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),pid=None,timeout=False,policy='One full fast suite; inherited CPU sampling fixtures authorized. No new feature fit.')
start=time.monotonic()
with (a/'stdout.txt').open('w') as out,(a/'stderr.txt').open('w') as err:
 p=subprocess.Popen(command,cwd=w,env=env,stdout=out,stderr=err,start_new_session=True);receipt['pid']=p.pid;(a/'receipt.json').write_text(json.dumps(receipt,indent=2))
 try:code=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  receipt['timeout']=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
receipt.update(exit_code=code,wall_seconds=time.monotonic()-start,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
(a/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
raise SystemExit(124 if receipt['timeout'] else code)
