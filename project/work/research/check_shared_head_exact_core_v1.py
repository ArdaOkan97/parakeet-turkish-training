"""Small pure contract/coverage checks; no actual CLOSED source/state admission."""
from pathlib import Path
import copy
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import tsc_bpe_shared_head_exact_transition_v1 as h
checks=[]
def check(ok,name):assert ok,name;checks.append(name)
def reject(callback,name):
    try:callback()
    except (ValueError,KeyError):checks.append(name);return
    raise AssertionError(name+' accepted')
r={'schema':1,'kind':h.RECEIPT_KIND,'status':'actual_CLOSED_complete_shared_head_boundary',
 'run_id':'SYNTHETIC_SAME_RUN','target_run_id':'SYNTHETIC_SAME_RUN','allow_same_phase_exact_resume':True,
 'guards_retained':True,'further_unfreeze_allowed':False,'original_step':6500,'source_acquisition_updates':500,
 'shared_head_updates':1000,'completed_steps':8000,'approved_total_shared_head_updates':37958,
 'chosen_checkpoint_role':'latest','chosen_step':8000,'latest_DEV_step':8000,
 'competitive_qualifying_latest_approved':True,'source_stop_category':'phase_bound',
 'owned_actual_exits':[{'role':'trainer_controller','exit_code':0,'session_id':'OPAQUE_SYNTHETIC_ONLY',
 'chunk_id':'NO_ACTUAL_EXIT','observed_at_utc':'2026-10-12T00:00:00Z'}]}
check(h.validate_receipt(r,'SYNTHETIC_SAME_RUN') is r,'contract_allows_declared1000_to_remaining37958_not_artificial10000_cap')
for key,bad in (('chosen_checkpoint_role','best'),('chosen_step',7500),('latest_DEV_step',7500),
 ('approved_total_shared_head_updates',37959),('shared_head_updates',999),('source_stop_category','resource'),
 ('target_run_id','HIDDEN_NEW_ID')):
    b=copy.deepcopy(r);b[key]=bad
    reject(lambda b=b:h.validate_receipt(b,'SYNTHETIC_SAME_RUN'),'reject_'+key)
check(h.full_TRAIN_coverage({})['full_unique_TRAIN_coverage_complete'] is False,'horizon_without_actual_unique_counts_does_not_prove_fullpass')
coverage={'source_groups':{name:{'pool_clips':n,'unique_clips_seen':n} for name,n in {'fleurs':2037,'cv':72846,'tsc':178915}.items()}}
check(h.full_TRAIN_coverage(coverage)['full_unique_TRAIN_coverage_complete'] is True,'actual_all253798_unique_per_source_counts_prove_coverage')
coverage['source_groups']['tsc']['unique_clips_seen']-=1
check(h.full_TRAIN_coverage(coverage)['full_unique_TRAIN_coverage_complete'] is False,'one_missing_unique_clip_keeps_fullcoverage_false')
check(not any(name in sys.modules for name in ('torch','transformers','nemo')),'stdlib_only_no_actual_state_model_or_GPU')
out=HERE/'shared_head_exact_core_contract_verification_v1.json'
value={'kind':'inactive_shared_head_exact_core_contract_verification_v1','passed':True,'checks':checks,'assertions':len(checks),
 'helper_sha256':hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
 'actual_CLOSED_source_or_fullstate_admitted':False,'Tensor_optimizer_or_hardware_execution':False,
 'scope':'receipt structural corruption and literal coverage only; actual preflight/restore/executor remain pending'}
with out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'assertions':len(checks),'helper_sha256':value['helper_sha256'],'proof_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
