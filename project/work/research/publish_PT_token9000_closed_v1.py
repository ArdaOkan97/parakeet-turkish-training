"""Actual owned closure and bounded paired DEV changes; no ML reads."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';RUN='adaptive_20261012_TR_uniform_PTtokenKD5x_01'
DIR=HERE/'runs'/RUN;LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def record(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
state=read(LOOP/'loop_status.json');sel=read(DIR/'selection.json');queue=read(HERE/'experiment_queue_state.json')
assert state['status']=='train_window_ended' and state['child_exit_code']==0 and state['completed_steps']==9000 and state['stop_category']=='retention'
assert sel['consecutive_retention_failures']==3 and sel['selected_step']==8250
assert queue['current_run_id']==RUN
identity=read(Path(state['paths']['identity']));k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
absence=[]
for pid in (queue['pid'],25308,34476):
    ctypes.set_last_error(0);handle=k.OpenProcess(0x1000,False,pid);err=ctypes.get_last_error();assert not handle and err==87
    absence.append({'pid':pid,'OpenProcess_handle':None,'winerror':err,'absent':True})
paths=state['paths'];done=read(Path(paths['native']).with_suffix('.completion.json'));assert done['status']=='complete' and done['sources_unchanged']
dev=HERE/'results'/(RUN+'_step_9000');summary=read(dev/'summary.json');langs=('tr','en','es','it','fr','de','pt')
counts={lang:summary['validation_'+lang]['asr'] for lang in langs};assert [counts[x]['word_errors'] for x in langs]==[325,63,50,24,64,71,71]
uniform=sel['Turkish_domain_coverage']['uniform_pass'];assert uniform['unique_clips_seen']==uniform['draws']==28274 and uniform['repeat_visits']==0
artifacts={'latest_checkpoint':record(ROOT/'work'/(RUN+'_latest.pt')),'inherited_qualified_BEST':record(sel['weights']),
    'run_config':record(DIR/'run_config.json'),'training_history':record(DIR/'training_history.json'),'selection':record(DIR/'selection.json'),
    'controller_status':record(LOOP/'loop_status.json'),'initial_source_equivalence':record(DIR/'majority_initial_source_equivalence.json'),
    'runtime_completion':record(Path(paths['native']).with_suffix('.completion.json')),'runtime_native_ready':record(paths['native']),
    'owned_worker_identity':record(paths['identity']),'DEV9000_summary':record(dev/'summary.json'),'DEV9000_metadata':record(dev/'metadata.json'),
    'DEV9000_metadata_pt':record(dev/'metadata_pt.json'),**{'DEV9000_'+lang:record(dev/('validation_'+lang+'.jsonl')) for lang in langs},
    **{name:record(paths[name]) for name in ('memory','cache','io')}}
now=datetime.now(timezone.utc).isoformat()
actual={'schema':1,'kind':'PT_token5x_owned_actual_CLOSED9000_v1','status':'actual_CLOSED0_retention_review_required','observed_at_utc':now,
    'run_id':RUN,'actual_owned_exit':{'session_id':37834,'exit_code':0,'tool_chunk':'af6052'},'owned_PID_absence':absence,
    'native_child_exit_code':0,'sources_unchanged':True,'resource_failure':False,'completed_steps':9000,'new_PT_token_constraint_updates':250,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,'PT_token_source_step':8750,
    'configured_end_step':25109,'remaining_updates':16109,'uniform_unique_TR_clips':28274,'uniform_TR_pool':253798,'repeat_visits':0,'full_pass_complete':False,
    'actual_DEV9000_ASR':counts,'original_five_passed':False,'Portuguese_passed':False,'failed_languages':['de','pt'],
    'fixed_allowed_word_errors':{'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66},'source_failure_counter':2,'consecutive_retention_failures':3,
    'selected_step':8250,'selected_BEST_TR_word_errors':329,'new_grace_or_counter_reset':False,'source_coverage_rewind_clips':0,
    'peak_observed_owned_private_bytes':state['peak_observed_owned_private_bytes'],'artifacts':artifacts,
    'decision':'Stop by unchanged retention guard. PT-only token200 did not recover on the next fixed DEV; no unchanged retry or further blind scalar increase is justified by this point.',
    'goal_complete':False}
out=HERE/'PT_token5x_actual_CLOSED9000_retention_review_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(actual,f,indent=2)
sys.path.insert(0,str(HERE));from metrics import normalize_text,edit_distance
changes={}
for lang in ('de','pt'):
    before=HERE/'results'/(RUN+'_step_8750')/('validation_'+lang+'.jsonl');after=dev/('validation_'+lang+'.jsonl')
    with before.open(encoding='utf-8') as f:a=[json.loads(x) for x in f]
    with after.open(encoding='utf-8') as f:b=[json.loads(x) for x in f]
    assert len(a)==len(b)==50;items=[];total_a=total_b=0
    for aa,bb in zip(a,b):
        assert aa['source_id']==bb['source_id'] and aa['reference']==bb['reference']
        ref=normalize_text(aa['reference'],lang);ha=normalize_text(aa['hypothesis'],lang);hb=normalize_text(bb['hypothesis'],lang)
        ea=edit_distance(ref.split(),ha.split());eb=edit_distance(ref.split(),hb.split());total_a+=ea;total_b+=eb
        if ha!=hb:items.append({'source_id':aa['source_id'],'reference':ref,'source8750_hypothesis':ha,'latest9000_hypothesis':hb,'source_errors':ea,'latest_errors':eb,'delta':eb-ea})
    changes[lang]={'before_word_errors':total_a,'after_word_errors':total_b,'changed_utterances':items,'source':record(before),'latest':record(after)}
paired={'kind':'PT_token5x_DE_PT8750_to9000_saved_changes_v1','actual_fixed_DEV_only':True,'languages':changes,
    'scope':'Paired text outputs only; no acoustic/model/checkpoint/TEST read. Fixed50 per language does not establish general causality or convergence.'}
pairedpath=HERE/'PT_token5x_DE_PT8750_to9000_saved_changes_v1.json'
with pairedpath.open('x',encoding='utf-8') as f:json.dump(paired,f,indent=2,ensure_ascii=False)
lp=ROOT/'work/gpu_training_owner.json';lease=read(lp);assert lease['owner']=='/root/benchmark_design' and lease['actual_controller_session_id']==37834
lease['prior_PT_token5x_closed']={'run_id':RUN,'session_id':37834,'exit_code':0,'tool_chunk':'af6052','closure_receipt':record(out)}
for key in ('actual_controller_session_id','actual_owned_native_launcher_pid','actual_owned_worker_pid','actual_launch_tool_chunk','actual_exit_code','actual_closed_tool_chunk'):lease.pop(key,None)
lease.update(status='reserved_pending_measured_DE_PT_retention_review',updated_at_utc=now,running=False,actual_GPU_running=False,
    reserved=True,GPU_reserved=True,actual_completed_step=9000,actual_new_uniform_unique_TR_clips=28274,source_failure_counter_preserved=3,
    actual_closure_receipt=record(out),paired_saved_text_evidence=record(pairedpath))
atomic(lp,lease)
pp=HERE/'experiment_plan.json';plan=read(pp);plan.update(updated_at_utc=now,objective_complete=False,actual_current_training=actual,
    status_authority='Observed PT-token5x owned37834 CLOSED0/af6052 after250 updates at9000. DE71>70 and PT71>66 failed fixed integer gates; counter2 to3 carried. BEST8250 preserved separately;28274 unique/no repeats. No unchanged retry or new grace. Sole lease reserved pending measured review.')
item=next(x for x in plan['ordered_experiments'] if x['id']=='04');item.update(status='actual_PT_token5x_CLOSED_retention_review_required',
    run_id=RUN,actual_current_run=RUN,actual_GPU_training_running=False,actual_completed_steps=9000,actual_closed_session=37834,
    actual_closed_exit_code=0,actual_closed_tool_chunk='af6052',actual_new_uniform_unique_TR_clips=28274,actual_new_uniform_unique_clips=28274,
    current_remaining_uniform_updates=16109,current_failure_counter=3,current_all6_selected_BEST_step=8250,latest_actual_DEV_step=9000,
    latest_original_five_passed=False,latest_Portuguese_passed_by_exact_word_count=False,latest_all_six_passed=False,actual_closure_receipt=out.name,
    latest_failed_retention_languages=['de','pt'])
item['pending']=['Measured DE/PT retention review before another identity is admitted','Continue coherent full-pass state only under a defensible changed constraint; no reset or gate relaxation','Competitive quality and complete unique pass remain unmet']
atomic(pp,plan)
note='Actual PT-token5x owned37834 CLOSED0/af6052 at9000 after250 native updates; child0/native complete/sources unchanged, no resource failure. Fixed DEV: TR325/82539.3939%, EN63 ES50 IT24 FR64 pass, DE71>70 and PT71>66 fail. Genuine counter2 to3 caused unchanged guard closure; qualified BEST8250 remains separate. PT-only token200 did not recover on this point; no unchanged retry or blind scalar increase justified. Full uniform frontier28274/253798 unique,0repeat/no rewind/end25109; incomplete competitive goal. All207 latest state and failed branch retained. Sole GPU lease reserved for measured review, with no running child. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'closure':record(out),'paired':record(pairedpath),'changes':{lang:{'before':v['before_word_errors'],'after':v['after_word_errors'],'changed_utterances':len(v['changed_utterances'])} for lang,v in changes.items()},'lease_plan_Backlog_updated':True}))
