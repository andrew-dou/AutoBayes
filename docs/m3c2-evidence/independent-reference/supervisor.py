"""One exclusive deterministic reference invocation, 120s process-group cap."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path('/private/tmp/autobayes-intercept-reference-1')
REPO=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3c2-intercept-controls')
def main():
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for entry in frozen['inputs']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    command=[str(REPO/'.venv/bin/python'),'-B','-u',str(ROOT/'reference.py')]
    env=frozen['config']['env_overrides']
    receipt={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'cwd':str(ROOT),
        'env_overrides':env,'deadline_seconds':120,'timeout':False,'term_sent':False,'kill_sent':False,
        'freeze_sha256':hashlib.sha256((ROOT/'freeze.json').read_bytes()).hexdigest()}
    with (ROOT/'launch.json').open('x') as out: json.dump(receipt,out,indent=2);out.write('\n')
    start=time.monotonic()
    with (ROOT/'stdout.txt').open('x') as stdout,(ROOT/'stderr.txt').open('x') as stderr:
        proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**env},stdout=stdout,stderr=stderr,start_new_session=True)
        receipt['pid']=proc.pid
        try: code=proc.wait(timeout=max(0.,120-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            receipt['timeout']=True
            try: os.killpg(proc.pid,signal.SIGTERM);receipt['term_sent']=True
            except ProcessLookupError: pass
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(proc.pid,signal.SIGKILL);receipt['kill_sent']=True
            except ProcessLookupError: pass
            code=proc.wait()
    receipt.update(child_exit_code=code,exit_code=124 if receipt['timeout'] else code,
        wall_seconds=time.monotonic()-start,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (ROOT/'exit-receipt.json').open('x') as out: json.dump(receipt,out,indent=2);out.write('\n')
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt['exit_code']
if __name__=='__main__': raise SystemExit(main())
