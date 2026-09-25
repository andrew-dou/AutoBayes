from pathlib import Path
import hashlib, json, subprocess, tomllib, datetime
root=Path('/Users/andrewdou/iqss-research/projects/AutoBayes-worktrees/m3a3-multiple-interactions')
out=Path('/private/tmp/autobayes-m3a3-implementation')
base='f23beea20607cdd049b808c0c9140072fbcd6bdd'
before=tomllib.loads(subprocess.check_output(['git','show',f'{base}:uv.lock'],cwd=root,text=True))
after=tomllib.loads((root/'uv.lock').read_text())
old_self=[p for p in before['package'] if p['name']=='autobayes'];new_self=[p for p in after['package'] if p['name']=='autobayes']
assert old_self[0]['version']=='0.7.0' and new_self[0]['version']=='0.8.0'
copy_new=dict(new_self[0]);copy_new['version']='0.7.0';assert old_self==[copy_new]
old_other={**before,'package':[p for p in before['package'] if p['name']!='autobayes']}
new_other={**after,'package':[p for p in after['package'] if p['name']!='autobayes']}
assert old_other==new_other
launch=json.loads((out/'runs/full-1/launch.json').read_text())
code_paths=[p for p in launch['source_sha256'] if '/autobayes/' in p or '/tests/' in p or '/examples/' in p]
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==launch['source_sha256'][p] for p in code_paths)
metadata={}
for p in sorted((root/'.venv/lib/python3.11/site-packages').glob('*.dist-info/METADATA')):
    fields=dict(line.split(': ',1) for line in p.read_text().split('\n\n',1)[0].splitlines() if line.startswith(('Name: ','Version: ')))
    metadata[fields['Name']]=fields['Version']
receipt=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=launch['head'],
  base=base,third_party_lock_unchanged=True,lock_only_self_version_changed=True,
  code_test_example_files_match_full_launch=len(code_paths),installed_metadata_versions=metadata,
  source_version='0.8.0',github_viewer='andrew-dou',
  metadata_note='Editable environment was installed before source version bump; editable dist-info remains 0.7.0. Package __version__ and manifest/lock are 0.8.0. No reinstallation performed.')
(out/'static-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='installed_metadata_versions'},indent=2))
