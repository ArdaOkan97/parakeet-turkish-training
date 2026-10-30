"""Independent CPU verification of the completed continuation fork manifest."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
LOOP=OUT/'runs/adaptive_20261010_continuation_01'
PATH=LOOP/'fork_snapshot/fork_manifest.json'
torch.set_num_threads(2)
checks=0

def check(ok,message):
    global checks
    checks+=1
    if not ok:raise AssertionError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

tensor_comparisons=0

def same(left,right,path):
    global tensor_comparisons
    check(type(left)==type(right),path+' type')
    if isinstance(left,torch.Tensor):
        tensor_comparisons+=1
        check(left.device.type==right.device.type=='cpu',path+' device')
        check(left.dtype==right.dtype and left.shape==right.shape,path+' tensor type/shape')
        # Raw bytes distinguish signed zeros as well as equal numerical values.
        a=left.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        b=right.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        check(hashlib.sha256(a).digest()==hashlib.sha256(b).digest(),path+' tensor bytes')
    elif isinstance(left,dict):
        check(list(left)==list(right),path+' keys')
        for key in left:same(left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(tuple,list)):
        check(len(left)==len(right),path+' length')
        for i,(a,b) in enumerate(zip(left,right)):same(a,b,path+'.'+str(i))
    else:check(left==right,path+' value')

manifest_sha=sha(PATH); m=read(PATH)
check(m['passed'] is True and m['status']=='complete' and m['schema']==1,'manifest completion')
check(m['only_changed_path']=='run_id','fork changed path policy')
check(m['source_completed_step']==907 and m['inherited_selected_step']==801,'inherited907/801')
check(sha(m['fork_helper_path'])==m['fork_helper_sha256'],'fork source hash')
receipt=read(m['predecessor_exit_receipt_path'])
check(sha(m['predecessor_exit_receipt_path'])==m['predecessor_exit_receipt_sha256'],'owned receipt hash')
check(receipt['root_observed_exit_code']==0 and receipt['root_observed_owned_session_completed'] is True,'owned exit0')
check(receipt['predecessor_deadline_unchanged']=='2026-10-10T23:41:22.889692+00:00','old deadline unchanged')
audit=read(m['source_verification_path'])
check(sha(m['source_verification_path'])==m['source_verification_sha256']==receipt['predecessor_audit_sha256'],'old audit hash')
check(audit['passed'] is True and audit['status']=='complete' and audit['run_id']==m['source_run_id'],'old audit identity')
check(audit['passed_check_count']==495229,'actual old audit checks')

snapshot_bytes=0
for source,record in m['snapshot_files'].items():
    copied=Path(record['snapshot_path']).resolve()
    check(copied.is_relative_to(PATH.parent.resolve()),'snapshot path containment')
    check(copied.is_file() and copied.stat().st_size==record['size_bytes'],'snapshot file/size')
    check(sha(copied)==record['sha256'],'snapshot exact bytes '+str(copied))
    snapshot_bytes+=record['size_bytes']
check(snapshot_bytes==m['snapshot_bytes'] and len(m['snapshot_files'])==m['snapshot_file_count'],'snapshot accounting')
candidate=Path(m['preserved_published_export']['candidate_path'])
check(candidate.is_relative_to(ROOT/'work/models'),'immutable export candidate containment')
for name,digest in m['preserved_published_export']['files_sha256'].items():
    check(sha(candidate/name)==digest==receipt['prior_hf_files_sha256'][name],'immutable export '+name)

checkpoint_evidence={}
for kind in ['latest','best']:
    original=m['source_checkpoints'][kind];fork=m['fork_checkpoints'][kind]
    check(sha(original['path'])==original['sha256'],'source checkpoint hash')
    check(sha(fork['path'])==fork['sha256'],'fork checkpoint hash')
    with Path(original['path']).open('rb') as f:a=torch.load(f,map_location='cpu',weights_only=True)
    with Path(fork['path']).open('rb') as f:b=torch.load(f,map_location='cpu',weights_only=True)
    check(a['run_id']==m['source_run_id'] and b['run_id']==m['target_run_id'],'source/target checkpoint IDs')
    check(list(a)==list(b),'checkpoint field order')
    before=checks;before_tensors=tensor_comparisons
    for key in a:
        if key!='run_id':same(a[key],b[key],kind+'.'+key)
    check(len(b['weights'])==len(b['optimizer']['state'])==207,'complete207 checkpoint state')
    checkpoint_evidence[kind]={'source_sha256':original['sha256'],'fork_sha256':fork['sha256'],
                               'step':b['step'],'selected_step':b['selected_step'],
                               'independent_leaf_checks':checks-before,
                               'bitwise_tensor_comparisons':tensor_comparisons-before_tensors}
    del a,b

reuse={}
for step,record in m['inherited_development_reuse'].items():
    check(record['evaluation_reused_without_new_inference'] is True,'DEV reuse evidence classification')
    source=Path(record['source_path']);target=Path(record['target_path'])
    expected={'summary.json','metadata.json',*(f'validation_{lang}.jsonl' for lang in ['tr','en','es','it','fr','de'])}
    check(set(p.name for p in source.iterdir())==set(p.name for p in target.iterdir())==expected,'inherited DEV-only exact file coverage')
    for p in source.iterdir():check(sha(p)==sha(target/p.name),'inherited DEV exact bytes '+p.name)
    check(read(target/'metadata.json')['model_fingerprint']==record['model_fingerprint'],'DEV model identity preserved')
    reuse[step]={'source':str(source),'target':str(target),'files':len(expected),
                 'original_model_state_sha256':record['model_fingerprint']['identity']['model_state_sha256'],
                 'reused_source_evaluation':True}
check(not list((OUT/'results').glob(m['target_run_id']+'*test*')),'no target TEST inference copy')
check(not (OUT/'runs'/m['target_run_id']).exists(),'no target training execution during CPU fork')
check(torch.cuda.is_initialized() is False,'no CUDA initialization')
check(sha(PATH)==manifest_sha,'immutable manifest unchanged')
report={'schema':1,'passed':True,'status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'scope':'Independent CPU actual snapshot hashes + serialized tensor bytes and nested values; no training or GPU',
        'manifest_path':str(PATH),'manifest_sha256':manifest_sha,'source_run_id':m['source_run_id'],'target_run_id':m['target_run_id'],
        'snapshot_files':m['snapshot_file_count'],'snapshot_bytes':snapshot_bytes,'tensor_comparisons':tensor_comparisons,
        'checkpoints':checkpoint_evidence,'inherited_DEV_reuse':reuse,
        'old_deadline_at_utc':receipt['predecessor_deadline_unchanged'],'cuda_initialized':False,
        'published_export_preservation':'All7 existing immutable candidate files match the previous audited published export; no redundant2.5GB model copy.'}
destination=LOOP/'fork_snapshot_independent_verification.json'
with destination.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({'passed':True,'checks':checks,'tensor_comparisons':tensor_comparisons,'report':str(destination),
                  'report_sha256':sha(destination),'manifest_sha256':manifest_sha,'cuda_initialized':False},indent=2))
