"""Owned actual CLOSED0 bookkeeping; no tensor/model/audio/GPU reads."""
from pathlib import Path
from datetime import datetime,timezone
import os,ctypes,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_majority_PT_focus_integer_01';DIR=HERE/'runs'/RUN;LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def record(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
state=read(LOOP/'loop_status.json');selection=read(DIR/'selection.json')
native=LOOP/(RUN+'_native_backend.completion.json');completion=read(native)
assert state['status']=='train_window_ended' and state['child_exit_code']==0 and state['stop_category']=='retention'
assert state['completed_steps']==8750 and completion['status']=='complete' and completion['sources_unchanged'] is True
assert selection['accepted'] is True and selection['selected_step']==8250 and selection['consecutive_retention_failures']==2
queue=read(HERE/'experiment_queue_state.json');assert queue['current_run_id']==RUN and queue['pid']==32248
api=ctypes.WinDLL('kernel32',use_last_error=True);api.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];api.OpenProcess.restype=ctypes.c_void_p;api.CloseHandle.argtypes=[ctypes.c_void_p]
absences=[]
for pid in (32248,17172,26792):
    ctypes.set_last_error(0);handle=api.OpenProcess(0x1000,0,pid);error=ctypes.get_last_error()
    if handle:api.CloseHandle(handle);raise RuntimeError('Previously ownedPID still present:'+str(pid))
    assert error==87,(pid,error)
    absences.append({'pid':pid,'OpenProcess_handle':None,'winerror':error,'absent':True})
with (DIR/'training_history.json').open('rb') as stream:
    stream.seek(max(0,os.fstat(stream.fileno()).st_size-350000));tail=stream.read().decode('utf-8','ignore')
i=tail.rfind('\n  {\n    "step":');assert i>=0;text=tail[i+3:].strip();assert text.endswith(']');row=json.loads(text[:-1].rstrip())
assert row['step']==8750 and row['PT_focus_updates']==750
uniform=row['Turkish_domain_coverage']['uniform_pass'];assert uniform['draws']==uniform['unique_clips_seen']==24774 and uniform['repeat_visits']==0
assert row['selection_diagnostics']['consecutive_retention_failures']==2
summary_path=HERE/'results'/(RUN+'_step_8750')/'summary.json';summary=read(summary_path)
counts={language:summary['validation_'+language]['asr'] for language in ('tr','en','es','it','fr','de','pt')}
assert {k:v['word_errors'] for k,v in counts.items()}=={'tr':329,'en':60,'es':49,'it':23,'fr':63,'de':71,'pt':71}
assert row['validation']['retention_language_passed']=={'en':True,'es':True,'it':True,'fr':True,'de':False,'pt':False}
proof=DIR/'majority_initial_source_equivalence.json';source350=read(proof)
assert source350['passed'] and source350['actual_GPU_decode'] and source350['all_six_retention_gates_passed'] is True
assert source350['source_selected_step']==8250 and source350['compared_fixed_DEV_utterances']==350
now=datetime.now(timezone.utc).isoformat()
actual={'schema':1,'kind':'PT_focus_integer_owned_actual_CLOSED8750_v1','status':'actual_CLOSED0_retention_review_required',
    'observed_at_utc':now,'run_id':RUN,'actual_owned_exit':{'session_id':43870,'exit_code':0,'tool_chunk':'df6a79'},
    'owned_PID_absence':absences,'native_child_exit_code':0,'sources_unchanged':True,'resource_failure':False,
    'completed_steps':8750,'new_integer_continuation_updates':500,'PT_focus_updates':750,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16359,'uniform_unique_TR_clips':24774,'uniform_TR_pool':253798,
    'fraction_unique_TR_seen':uniform['fraction_unique_seen'],'repeat_visits':0,'full_pass_complete':False,
    'actual_DEV8750_ASR':counts,'actual_gate':row['validation'],'retention_failures':2,
    'accepted':True,'selected_step':8250,'selected_TR_WER':selection['turkish_validation_wer'],
    'selected8250_all_six_count_qualified':True,'selected8250_TR_word_errors':329,'selected8250_PT_word_errors':66,
    'latest8750_all_six_qualified':False,'initial_source350':record(proof),
    'corrected_numeric_gate_source_accounting':{'source_step':8250,'failure5_to0_by_passing_DEV_rule':True,
        'historical_false_gate_and_None_selection_preserved':True,'no_epsilon_or_threshold_relaxation':True},
    'head_lr':row['head_lr'],'lora_lr':row['lora_lr'],'preservation_coefficients_by_language':row['preservation_coefficients_by_language'],
    'observed_peak_owned_private_bytes':state['peak_observed_owned_private_bytes'],
    'summary':record(summary_path),'selection':record(DIR/'selection.json'),'run_config':record(DIR/'run_config.json'),
    'controller_status':record(LOOP/'loop_status.json'),'runtime_completion':record(native),
    'runtime_evidence':{},'best_checkpoint':{'path':selection['weights'],'bytes':Path(selection['weights']).stat().st_size},
    'latest_checkpoint':{'path':selection['latest_weights'],'bytes':Path(selection['latest_weights']).stat().st_size},
    'checkpoint_tensor_or_bytes_read_or_hashed_by_this_report':False,'no_GPU_or_model_audio_read':True,
    'goal_complete':False,'samepass_frontier_not_reset':True,'unchanged_retry_authorized':False}
for role in ('memory','cache','io'):
    path=LOOP/(RUN+'_'+role+'.json');value=read(path);assert value['status']=='complete' and value['sources_unchanged'] is True
    actual['runtime_evidence'][role]=record(path)
cache=read(LOOP/(RUN+'_cache.json'));assert cache['cache_reclamation']['eligible_update_calls']==500
out=HERE/'PT_focus_integer_actual_CLOSED8750_retention_review_v1.json'
with out.open('x',encoding='utf-8') as stream:json.dump(actual,stream,indent=2)
leasepath=ROOT/'work/gpu_training_owner.json';lease=read(leasepath);assert lease['owner']=='/root/benchmark_design' and lease['actual_controller_session_id']==43870
lease.update(status='actual_CLOSED0_GPU_released_to_ROOT_retention_review',owner='/root',previous_owner='/root/benchmark_design',
    updated_at_utc=now,actual_exit_code=0,actual_closed_tool_chunk='df6a79',actual_GPU_running=False,GPU_reserved=False,
    ROOT_will_not_launch_concurrent_GPU=False,actual_completed_step=8750,actual_PT_focus_updates=750,
    actual_new_uniform_unique_TR_clips=24774,actual_closure_receipt=record(out))
atomic(leasepath,lease)
planpath=HERE/'experiment_plan.json';plan=read(planpath);plan['updated_at_utc']=now;plan['objective_complete']=False
plan['actual_current_training']=actual
plan['status_authority']='Actualowned43870 CLOSED0/df6a79/nativecomplete at8750 after500integercontinuationupdates. True8500+8750retentionfailures2:8750DE71>70/PT71>66,TR329/825; qualifiedBEST8250TR329/PT66 retained. Uniformunique24774/253798,no repeats; fullpassincomplete. GPUreleasedtoROOT, no unchangedretry.'
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_integer_continuation500_CLOSED_retention_review_required',run_id=RUN,actual_current_run=RUN,
    actual_GPU_training_running=False,actual_completed_steps=8750,actual_PT_focus_updates=750,actual_new_uniform_unique_TR_clips=24774,
    actual_new_uniform_unique_clips=24774,current_remaining_uniform_updates=16359,current_configured_end_step=25109,
    current_failure_counter=2,current_actual_initial_BEST_step=8250,current_all6_selected_BEST_step=8250,
    current_all6_selected_BEST_TR_WER=selection['turkish_validation_wer'],latest_actual_DEV_step=8750,
    latest_actual_DEV_counts={k:{a:v[a] for a in ('word_errors','reference_words','utterances')} for k,v in counts.items()},
    latest_original_five_passed=False,latest_Portuguese_passed_by_exact_word_count=False,latest_all_six_passed=False,
    all_six_champion_yet=True,actual_full_pass_completed=False,actual_closed_session=43870,actual_closed_exit_code=0,
    actual_closed_tool_chunk='df6a79',actual_closure_review=out.name,execution_owner='ROOT after actual43870closure/lease release')
item['pending']=['Review two realDE/PT failures; nounchangedretry','PreserveLATEST8750fullstate/24774unique separately fromqualifiedBEST8250',
    'Continueexistingfullpass through253798unique onlyvia explicitdefensible state-preserving review','Frozenheldout/competitivegoalverification']
item.setdefault('actual_DEV_points',[]).append({'run_id':RUN,'step':8750,'TR_WER':counts['tr']['wer'],
    'all_six_passed':False,'failed_languages':['de','pt'],'summary':record(summary_path),'closure':record(out)})
plan['automatic_execution_policy']['GPU_ownership']='benchmark_design actual43870 CLOSED0/df6a79/nativecomplete/ownedPIDs absent; lease releasedtoROOT. NoGPU running underbenchmark_design. Next trial requires review/soleownership andfreshmemoryadmission.'
plan['automatic_execution_policy']['PT_focused_replay']='Integerfixactual3508250qualified and500updatesexecuted. 8500/8750realDE/PTfails counter2 stopped; BEST8250 protected/LATEST8750 fullstate and24774unique retained. No unchangedretry or counter/grace/coverage reset. ROOTretentionreview required; V15solelogger preservesallmaps/sameproject.'
atomic(planpath,plan)
notes='Actualintegercontinuation owned43870 CLOSED0/df6a79/nativechild0complete/sourcesunchanged,noresourcefailure at8750 after500updates (PTfocus750). TrueDEV8750 TR329/82539.8788; EN60 ES49 IT23 FR63pass,DE71>70/PT71>66fail; consecutivefailures2 preservedguardstop. QualifiedownBEST8250TR329/PT66 retained; LATEST8750 distinctfullstate/frontier. Uniformunique24774/2537989.7613%,0repeat,end25109/16359remaining,fullpassincomplete. Numericfixworked/source350/all207initialserialization/positiveupdatesactual; historicalfalsegate5/None immutable, noepsilon/gaterelax/reset. GPUlease releasedtoROOT; reviewneeded,no unchangedretry. V15sole18970/PID4284 sameproject preserving45oldrefs. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',notes],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'closure':record(out),'owned_exit0_chunk':'df6a79','GPU_released_to_ROOT':True,'plan_Backlog_refreshed':True}))
