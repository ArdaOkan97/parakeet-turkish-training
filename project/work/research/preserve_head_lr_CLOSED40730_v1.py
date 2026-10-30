"""Archive actually CLOSED delegated head window; byte reads only, no tensors."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/parakeet_turkish'
RUN_ID='adaptive_20261013_TR_uniform_headlr3e5_cosine_01'
RUN=OUT/'runs'/RUN_ID
LOOP=OUT/'runs'/(RUN_ID+'_loop_01_window_2_journal_01')
ARCHIVE=ROOT/'work/closed-sources/TSC_uniform_headlr3e5_cosine40730_budget_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def record(p):
    p=Path(p).resolve()
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':digest,'bytes':p.stat().st_size}
def write(p,v):
    p=Path(p);temp=p.with_suffix('.metadata.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
status=read(LOOP/'loop_status.json');selection=read(RUN/'selection.json')
assert status['status']=='train_window_ended' and status['stop_category']=='budget'
assert status['child_exit_code']==0 and status['resource_review_required'] is False
assert status['completed_steps']==selection['completed_steps']==40730
completion_path=Path(status['actual_runtime_completion']['runtime_completion']['path'])
completion=read(completion_path)
assert completion['status']=='complete' and completion['sources_unchanged'] is True
assert selection['selected_step']==36500 and selection['actual_head_lr']==3e-5
cycle=selection['Turkish_domain_coverage']['matched_cycle']
assert cycle['local_updates']==15621 and cycle['cycle_unique']==218696 and not cycle['full_cycle_complete']
assert not ARCHIVE.exists();ARCHIVE.mkdir(parents=True)
paths={'latest_checkpoint':Path(selection['latest_weights']),'best_checkpoint':Path(selection['weights']),
       'run_config':RUN/'run_config.json','training_history':RUN/'training_history.json',
       'selection':RUN/'selection.json','controller_status':LOOP/'loop_status.json',
       'initial_source_equivalence':RUN/'majority_initial_source_equivalence.json',
       'original_native_ready':Path(status['paths']['native']),'runtime_native_ready':Path(status['paths']['fresh_runtime']),
       'runtime_completion':completion_path,'owned_worker_identity':Path(status['paths']['identity'])}
artifacts={}
for key,source in paths.items():
    before=record(source);target=ARCHIVE/(key+source.suffix)
    shutil.copyfile(source,target);after=record(target)
    assert before['sha256']==after['sha256'] and before['bytes']==after['bytes']
    artifacts[key]={'original':before,'archived':after,'byte_identical':True}
summary_path=OUT/'results'/(RUN_ID+'_step_40730')/'summary.json'
summary=read(summary_path)
counts={l:summary['validation_'+l]['asr'] for l in ('tr','en','es','it','fr','de','pt')}
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt={'schema':1,'kind':'TSC_uniform_head_LR_actual_CLOSED_budget_preservation_v1','passed':True,'created_at_utc':now,
    'run_id':RUN_ID,'actual_owned_exit':{'session_id':23420,'exit_code':0,'tool_chunk':'30c661'},
    'completed_step':40730,'source_step':25109,'native_child_exit_code':0,'stop_category':'budget',
    'native_sources_unchanged':True,'retention_counter_preserved':selection['consecutive_retention_failures'],
    'selected_qualified_BEST_step':36500,'selected_qualified_BEST_TR_WER':selection['turkish_validation_wer'],
    'counts':counts,'summary':record(summary_path),'matched_cycle':cycle,'artifacts':artifacts,
    'next_action':'Fresh strict unchanged memory admission then SAME-ID fullstate exact window3; no source350 repeat.',
    'CPU_checkpoint_tensor_audit_performed':False,'byte_archive_scope':True,'counter_RNG_coverage_or_LR_reset':False,
    'pagefile_or_guard_changes':False,'TEST_used':False,'objective_complete':False}
receipt_path=OUT/'matched_headlr3e5_cosine_actual_CLOSED40730_budget_v1.json';assert not receipt_path.exists();write(receipt_path,receipt)
lease_path=ROOT/'work/gpu_training_owner.json';lease=read(lease_path)
lease.update(status='actual_head_window2_CLOSED0_preparing_exact_window3',updated_at_utc=now,running=False,actual_GPU_running=False,
             actual_exit_code=0,actual_closed_tool_chunk='30c661',actual_completed_step=40730,
             prior_head_window2_closed=record(receipt_path),source_for_next_exact_window=artifacts['latest_checkpoint']['archived'])
write(lease_path,lease)
plan_path=OUT/'experiment_plan.json';plan=read(plan_path)
for key in ('current_training','actual_current_training'):
    c=plan[key];c.update(status=lease['status'],actual_GPU_running=False,current_frontier=40730,last_complete_DEV_step=40730,
        last_DEV_counts=counts,last_complete_DEV_TR_WER=counts['tr']['wer'],last_DEV_TR_WER=counts['tr']['wer'],last_DEV_TR_word_errors=counts['tr']['word_errors'],
        last_DEV_all_six_passed=False,cycle2_unique_including_carried=cycle['cycle_unique'],local_encoder_schedule_updates=15621,
        actual_retention_counter=selection['consecutive_retention_failures'],actual_CLOSED_receipt=record(receipt_path))
plan.update(actual_GPU_running=False,updated_at_utc=now,status_authority='Head window1 actual CLOSED0/30c661 at32,976 after normal budget; cycle110,140/253,798, full cycle unfinished. Coherent fullstate and qualified BEST30,000/TR295 byte-archived. Fresh unchanged admission and immediate SAME-ID exact continuation preparing; no source350 repeat or reset.')
write(plan_path,plan)
print(json.dumps({'passed':True,'closure':record(receipt_path),'latest':artifacts['latest_checkpoint']['archived'],'retention_counter':selection['consecutive_retention_failures'],'counts':{l:v['word_errors'] for l,v in counts.items()}}))
