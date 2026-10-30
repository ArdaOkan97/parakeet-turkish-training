"""Preserve the exact initial fork files before the new trainer reuses paths."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
LOOP=ROOT/'outputs/parakeet_turkish/runs/adaptive_20261010_continuation_01'
MANIFEST=LOOP/'fork_snapshot/fork_manifest.json'
VERIFY=LOOP/'fork_snapshot_independent_verification.json'
ARCHIVE=LOOP/'fork_snapshot/initial_fork_states'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024*1024):h.update(block)
    return h.hexdigest()

manifest_sha=sha(MANIFEST);verify_sha=sha(VERIFY)
assert manifest_sha=='ab92f0ea808b535aa8a9736556604531d75db4a57676499cee73bac9d52e8ddf'
assert verify_sha=='622407c0d08de8bcb084e7bbf8f137ff5e103d215f72f1dcc38e33821c2a6c0e'
m=json.loads(MANIFEST.read_text(encoding='utf-8'))
ARCHIVE.mkdir(exist_ok=False)
states={}
for kind,row in m['fork_checkpoints'].items():
    original=Path(row['path']);archived=ARCHIVE/original.name
    assert sha(original)==row['sha256']
    with original.open('rb') as src,archived.open('xb') as dst:
        shutil.copyfileobj(src,dst,length=8*1024*1024);dst.flush();os.fsync(dst.fileno())
    assert sha(archived)==sha(original)==row['sha256']
    states[kind]={'mutable_work_path':str(original),'mutable_work_path_initial_sha256':row['sha256'],
                  'archive_path':str(archived),'archive_sha256':sha(archived),
                  'size_bytes':archived.stat().st_size,'step':row['step'],'selected_step':row['selected_step'],
                  'parent_source_checkpoint_sha256':row['source_sha256']}
assert sha(MANIFEST)==manifest_sha and sha(VERIFY)==verify_sha
report={'schema':1,'kind':'immutable_initial_fork_states_archive_v1','passed':True,'status':'complete',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'loop_id':LOOP.name,
        'target_run_id':m['target_run_id'],'source_run_id':m['source_run_id'],
        'fork_manifest_path':str(MANIFEST),'fork_manifest_sha256':manifest_sha,
        'independent_fork_verification_path':str(VERIFY),'independent_fork_verification_sha256':verify_sha,
        'states':states,'archive_bytes':sum(x['size_bytes'] for x in states.values()),
        'purpose':'Use immutable archive aliases to verify initial optimizer/sampler/RNG state after legitimate new-run DEV saves overwrite work checkpoints.',
        'original_manifest_and_independent_verification_unchanged':True,'gpu_used':False}
destination=LOOP/'fork_initial_states_archive.json'
with destination.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({'passed':True,'path':str(destination),'sha256':sha(destination),'states':states},indent=2))
