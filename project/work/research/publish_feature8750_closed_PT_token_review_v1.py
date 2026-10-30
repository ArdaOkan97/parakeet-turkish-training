"""Observed owned closure and bounded saved-text decision; no ML imports."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_featureKD5x_02';DIR=HERE/'runs'/RUN;LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def record(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
def fresh(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
now=datetime.now(timezone.utc).isoformat();s=read(LOOP/'loop_status.json');sel=read(DIR/'selection.json')
assert s['status']=='train_window_ended' and s['child_exit_code']==0 and s['completed_steps']==8750 and s['stop_category']=='retention'
k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
absence=[]
for pid in (36648,23796,18932):
    ctypes.set_last_error(0);h=k.OpenProcess(0x1000,False,pid);err=ctypes.get_last_error();assert not h and err==87
    absence.append({'pid':pid,'OpenProcess_handle':None,'winerror':err,'absent':True})
paths=s['paths'];completion=read(Path(paths['native']).with_suffix('.completion.json'))
assert completion['status']=='complete' and completion['sources_unchanged'] is True
dev=HERE/'results'/(RUN+'_step_8750');summary=read(dev/'summary.json');counts={lang:summary['validation_'+lang]['asr'] for lang in ('tr','en','es','it','fr','de','pt')}
assert [counts[x]['word_errors'] for x in counts]==[321,62,48,23,64,69,70]
artifacts={'chosen_checkpoint':record(ROOT/'work'/(RUN+'_latest.pt')),'latest_checkpoint':record(ROOT/'work'/(RUN+'_latest.pt')),
    'source_best_checkpoint':record(ROOT/'work'/(RUN+'_best.pt')),'run_config':record(DIR/'run_config.json'),
    'selection':record(DIR/'selection.json'),'training_history':record(DIR/'training_history.json'),
    'controller_status':record(LOOP/'loop_status.json'),'initial_source_equivalence':record(DIR/'majority_initial_source_equivalence.json'),
    'runtime_completion':record(Path(paths['native']).with_suffix('.completion.json')),'runtime_native_ready':record(paths['native']),
    'owned_worker_identity':record(paths['identity']),'chosen_DEV_summary':record(dev/'summary.json'),
    'chosen_DEV_metadata':record(dev/'metadata.json'),'chosen_DEV_metadata_pt':record(dev/'metadata_pt.json'),
    **{'chosen_DEV_'+lang:record(dev/('validation_'+lang+'.jsonl')) for lang in counts},
    **{x:record(paths[x]) for x in ('memory','cache','io')},
    'source_best_DEV_summary':record(HERE/'results'/(RUN+'_step_8250')/'summary.json')}
exit={'session_id':98758,'exit_code':0,'tool_chunk':'0848c0'}
closure={'schema':1,'kind':'feature5x_owned_actual_CLOSED8750_v1','status':'actual_CLOSED0_retention_review_required',
    'observed_at_utc':now,'run_id':RUN,'actual_owned_exit':exit,'owned_PID_absence':absence,'native_child_exit_code':0,
    'sources_unchanged':True,'resource_failure':False,'completed_steps':8750,'new_feature_constraint_updates':500,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16359,'uniform_unique_TR_clips':24774,'uniform_TR_pool':253798,
    'repeat_visits':0,'full_pass_complete':False,'actual_DEV8750_ASR':counts,'old_five_passed':True,'Portuguese_passed':False,
    'consecutive_retention_failures':2,'selected_step':8250,'selected_BEST_TR_word_errors':329,
    'peak_observed_owned_private_bytes':s['peak_observed_owned_private_bytes'],'artifacts':artifacts,'goal_complete':False}
out=HERE/'feature5x_actual_CLOSED8750_retention_review_v1.json'
if not out.exists():fresh(out,closure)
leasepath=ROOT/'work/gpu_training_owner.json';lease=read(leasepath)
lease['prior_feature5x_closed']={'run_id':RUN,**exit,'closure_receipt':record(out)}
for key in ('actual_controller_session_id','actual_owned_native_launcher_pid','actual_owned_worker_pid','actual_launch_tool_chunk','actual_exit_code','actual_closed_tool_chunk'):lease.pop(key,None)
lease.update(status='reserved_for_measured_PT_token_constraint_preparation',updated_at_utc=now,running=False,actual_GPU_running=False,
    reserved=True,GPU_reserved=True,actual_completed_step=8750,actual_new_uniform_unique_TR_clips=24774,
    target_run_id='adaptive_20261012_TR_uniform_PTtokenKD5x_01',source_frontier=8750,source_failure_counter_preserved=2,
    ROOT_will_not_launch_concurrent_GPU=True)
atomic(leasepath,lease)
sys.path.insert(0,str(HERE));from metrics import normalize_text,edit_distance
def rows(p):
    with p.open(encoding='utf-8') as f:return [json.loads(line) for line in f]
a=rows(HERE/'results'/(RUN+'_step_8250')/'validation_pt.jsonl');b=rows(dev/'validation_pt.jsonl')
changes=[]
for aa,bb in zip(a,b):
    assert aa['source_id']==bb['source_id'] and aa['reference']==bb['reference']
    ra=normalize_text(aa['reference'],'pt');ha=normalize_text(aa['hypothesis'],'pt');hb=normalize_text(bb['hypothesis'],'pt')
    if ha!=hb:
        ca=edit_distance(ra.split(),ha.split());cb=edit_distance(ra.split(),hb.split())
        changes.append({'source_id':aa['source_id'],'reference':ra,'BEST8250_hypothesis':ha,'latest8750_hypothesis':hb,
            'BEST8250_errors':ca,'latest8750_errors':cb,'delta_errors':cb-ca})
paired={'kind':'feature5x_PT8250_to8750_saved_text_changes_v1','actual_fixed_DEV_only':True,'utterances':50,
    'source_counts':66,'latest_counts':70,'changed_utterances':changes,'no_TEST_audio_model_or_checkpoint_read':True,
    'limitations':['Text changes do not establish acoustic or linguistic cause.','A fixed 50-utterance set is a small diagnostic sample.']}
pairedpath=HERE/'feature5x_PT8250_to8750_saved_text_changes_v1.json';fresh(pairedpath,paired)
review={'schema':1,'kind':'ROOT_delegated_measured_TSC_PT_token_constraint_review_v1','passed':True,'created_at_utc':now,
    'decision_actor':'/root/benchmark_design under ROOT explicit delegated actual diagnostic and full-pass repair authority',
    'ROOT_independent_tensor_review_claimed':False,'source_run_id':RUN,'chosen_step':8750,'completed_steps':8750,
    'chosen_checkpoint_role':'latest','source_actual_owned_exit':exit,'actual_closed_source_review':record(out),
    'actual_diagnostic':record(HERE/'retention_components_8750_actual_GPU_v3.json'),'paired_PT_saved_text':record(pairedpath),
    'decision':'Feature20 restored all old-five retention and improved Turkish; Portuguese remains 70 versus limit66. Strengthen only Portuguese token posterior preservation, without claiming a decoder-only cause or guaranteed recovery.',
    'declared_change':{'parameter':'portuguese_token_kl_multiplier','source_value':1.,'active_value':5.,
        'PT_token_KL_before':40.,'PT_token_KL_after':200.,'baseKD':40.,
        'source_Portuguese_token_KL':40.,'active_Portuguese_token_KL':200.,'old_five_token_KL_unchanged':40*5/3,
        'feature_distillation_unchanged':20.,'duration_KL_unchanged':2.,'head_LR_unchanged':1e-5,'LoRA_LR_unchanged':1e-4,
        'TR_per_update':14,'PT_replay_per_update':1,'old_five_replay_per_update':1,'fixed_integer_gate_limits_unchanged':True},
    'allow_declared_PT_token_constraint_fork':True,'source_chosen_is_all_six_count_qualified':False,
    'source_old_five_count_qualified':True,'source_PT_word_errors':70,'source_counter_preserved':2,'source_selected_BEST_preserved':8250,
    'source_unqualified_latest_used':True,'source_weights_and_Adam_from_same_complete_checkpoint':True,
    'source_moments_sampler_RNG_history_preserved':True,'renewed_Portuguese_grace':False,'initial_source350_required':True,
    'no_hybrid_training_state_used':True,'first_future_complete_DEV_must_pass_all_six_or_stop_counter3':True,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16359,'chosen_uniform_unique_clips':24774,'explicit_coverage_rewind_clips':0,
    'full_student8199_token_softmax_already_used':True,'goal_complete':False,'artifacts':artifacts}
reviewpath=HERE/'TSC_PT_token5x_measured_retention_repair_review_v1.json';fresh(reviewpath,review)
leasepath=ROOT/'work/gpu_training_owner.json';lease=read(leasepath)
assert lease['owner']=='/root/benchmark_design' and lease['prior_feature5x_closed']['session_id']==98758
lease['prior_feature5x_closed']={'run_id':RUN,**exit,'closure_receipt':record(out)}
for key in ('actual_controller_session_id','actual_owned_native_launcher_pid','actual_owned_worker_pid','actual_launch_tool_chunk','actual_exit_code','actual_closed_tool_chunk'):lease.pop(key,None)
lease.update(status='reserved_for_measured_PT_token_constraint_continuation',updated_at_utc=now,running=False,actual_GPU_running=False,
    reserved=True,GPU_reserved=True,actual_completed_step=8750,actual_new_uniform_unique_TR_clips=24774,
    target_run_id='adaptive_20261012_TR_uniform_PTtokenKD5x_01',source_frontier=8750,source_failure_counter_preserved=2,
    actual_measured_constraint_review=record(reviewpath),ROOT_will_not_launch_concurrent_GPU=True)
atomic(leasepath,lease)
planpath=HERE/'experiment_plan.json';plan=read(planpath);plan.update(updated_at_utc=now,objective_complete=False,actual_current_training=closure,
    status_authority='Observed Feature5x02 owned98758 CLOSED0/0848c0 at8750. Old five retention pass; PT70 exceeds66. BEST8250 retained. Sole GPU reserved for measured PT-only token KL40 to200 continuation from coherent latest8750; no reset or coverage rewind.')
item=next(x for x in plan['ordered_experiments'] if x['id']=='04');item.update(status='actual_feature5x_CLOSED_PT_token_repair_reserved',
    run_id=RUN,actual_current_run=RUN,actual_GPU_training_running=False,actual_completed_steps=8750,actual_closed_session=98758,
    actual_closed_exit_code=0,actual_closed_tool_chunk='0848c0',actual_new_uniform_unique_TR_clips=24774,actual_new_uniform_unique_clips=24774,
    current_failure_counter=2,current_all6_selected_BEST_step=8250,current_remaining_uniform_updates=16359,
    latest_actual_DEV_step=8750,latest_original_five_passed=True,latest_Portuguese_passed_by_exact_word_count=False,latest_all_six_passed=False,
    next_run_id=lease['target_run_id'],next_declared_repair=review['declared_change'],actual_closure_receipt=out.name)
item['pending']=['Actual source8750 archive and real-input preflight','Actual source350, coherent all207 initial serialization and positive updates','Continue full unique pass subject to unchanged strict retention and resource guards']
atomic(planpath,plan)
note='Feature5x02 observed owned98758 CLOSED0/0848c0 at8750, native child0 complete and sources unchanged. Fixed DEV: TR321/825 (38.9091%), EN62 ES48 IT23 FR64 DE69 all old-five pass; PT70>66, genuine failure counter2. BEST8250 remains qualified and separate. Full uniform frontier24774/253798 unique, no repeats; end25109/16359 remaining. Measured next repair: only PT token KL40 to200, holding feature20, duration2, all old-five coefficients, head1e-5/LoRA1e-4 and14TR1PT1old-five. Choose coherent latest8750 with all207 Adam/RNG/sampler/history, no coverage rewind or counter/grace reset. First future complete DEV must pass all six or stop at counter3. Sole GPU reserved to benchmark_design; no quality or full-pass claim.'
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'closure':record(out),'review':record(reviewpath),'lease_plan_Backlog_updated':True}))
