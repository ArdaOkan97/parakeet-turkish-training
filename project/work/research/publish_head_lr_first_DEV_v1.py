import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
RUN_ID = 'adaptive_20261013_TR_uniform_headlr3e5_cosine_01'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def record(p):
    p=Path(p).resolve()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(p,v):
    p=Path(p); t=p.with_suffix('.metadata.tmp')
    with t.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary_path=OUT/'results'/(RUN_ID+'_step_25250')/'summary.json'
summary=read(summary_path)
counts={lang:summary['validation_'+lang]['asr'] for lang in ('tr','en','es','it','fr','de','pt')}
limits={'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66}
refs={'tr':825,'en':1045,'es':1289,'it':1158,'fr':1072,'de':1004,'pt':1095}
assert all(counts[l]['reference_words']==refs[l] and counts[l]['utterances']==50 for l in refs)
gates={l:counts[l]['word_errors']<=limits[l] for l in limits}
assert counts['tr']['word_errors']==300 and not all(gates.values())
status=read(OUT/'runs'/(RUN_ID+'_loop_01')/'loop_status.json')
receipt={'kind':'TSC_uniform_head_LR_first_complete_DEV_actual_v1','observed_at_utc':now,'run_id':RUN_ID,'actual_controller_session_id':64368,'complete_DEV_step':25250,'summary':record(summary_path),'counts':counts,'strict_all_six_passed':all(gates.values()),'failed_languages':[l for l,p in gates.items() if not p],'retention_monitor_only':True,'source350_exact':True,'qualified_global_champion_unchanged':{'run_id':'adaptive_20261012_TR_uniform_encoder_cosine_01','step':42500,'TR_word_errors':279,'TR_reference_words':825},'TEST_used':False,'new_quality_superiority_claimed':False}
rp=OUT/'matched_headlr3e5_cosine_first_complete_DEV25250_v1.json';write(rp,receipt)
plan_path=OUT/'experiment_plan.json';plan=read(plan_path)
for key in ('current_training','actual_current_training'):
    current=plan[key]
    assert current['run_id']==RUN_ID
    current.update(last_complete_DEV_step=25250,last_DEV_counts=counts,last_complete_DEV_TR_WER=300/825,last_DEV_TR_WER=300/825,last_DEV_TR_word_errors=300,last_DEV_all_six_passed=False,last_DEV_scope='Actual complete head trial DEV25250; ES53 andPT67 fail strict limits',first_DEV_receipt=record(rp),current_frontier=status['published_step'],local_encoder_schedule_updates=status['published_step']-25109)
plan['updated_at_utc']=now;write(plan_path,plan)
lease_path=ROOT/'work/gpu_training_owner.json';lease=read(lease_path)
lease.update(updated_at_utc=now,actual_completed_step=status['published_step'],actual_encoder_schedule_updates=status['published_step']-25109,actual_last_complete_DEV_step=25250,current_complete_DEV_all_six_passed=False,actual_DEV_receipt=record(rp));write(lease_path,lease)
note=('Both matched encoder controls completed the same carried253,798-clip cycle. Cosine qualified BEST42,500:279/825 TR errors (33.8182%); constant qualified BEST38,500:286/825 (34.6667%). The measured single pair selects cosine without claiming significance or bitwise equivalence. A full matched head-LR3e-5 challenger is now actually running from the same coherent25,109 full207 weights/Adam/RNG/sampler/history, owned64368/native33284. Source350 transcripts/token-duration paths/counts were exact; positive optimizer updates applied head3e-5 and cosine encoder. First complete DEV25,250: TR300/825 (36.3636%), ES53 andPT67 fail strict52/66 limits, so it is not a qualifying winner. Retention remains monitoring/selection only, full18,129-update cycle to43,238 continues with unchanged memory guards and frozen TEST. The completed head1e-5 cosine arm is reused honestly as matched control; early publication cadence differed. Evidence: completed_encoder_schedule_policy_selection_v1.json and matched_headlr3e5_cosine_actual_source350_positive_updates_v1.json.')
for task in ('ASR-15','ASR-15.1'):
    result=subprocess.run([str(OUT/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=OUT,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stderr or result.stdout)
refresh=subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(OUT/'refresh_backlog.py')],cwd=ROOT,capture_output=True,text=True)
if refresh.returncode: raise RuntimeError(refresh.stderr or refresh.stdout)
print(json.dumps({'passed':True,'DEV_step':25250,'TR':300/825,'failed':receipt['failed_languages'],'receipt':record(rp),'frontier':status['published_step'],'Backlog_and_refresh_passed':True}))
