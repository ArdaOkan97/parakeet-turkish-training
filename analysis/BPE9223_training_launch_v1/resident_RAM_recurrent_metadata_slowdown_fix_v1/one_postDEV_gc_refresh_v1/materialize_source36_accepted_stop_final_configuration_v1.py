"""Authorized lightweight final metadata preparation only; no candidate import or launch.
All checks precede new output writes. No original journal or preserved tail read.
"""
import copy,datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;FIX=F.parent;N=FIX.parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
def pin(p):
 p=Path(p).resolve();assert p.stat().st_size<=8*1024**2;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def raw(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def prospective_pin(p,b):return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('xb') as f:f.write(raw(v));f.flush();os.fsync(f.fileno())
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);identities=[who(me),who(me.parent())]
samples=[sample_memory(w['pid'],'source36_final_metadata_materialization_only') for w in identities]
assert all(s['paired_sampling_within_one_second'] for s in samples) and sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
m=samples[-1]['system_memory'];assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
candidate=F/'candidate_source36_configuration_completed_CPU_pending_actual_stop_v1.json'
closure=F/'actual_source33_one_postDEV_refresh_checkpoint_transition_01/actual_transition_closure.json'
stop_ROOT=F/'ROOT_source33_stop178820_saved178800_one_postDEV_refresh_actual_verification_v1.json'
finite_ROOT=F/'ROOT_one_postDEV_refresh_actual_finite_CPU_review_v1.json'
typed_ROOT=F/'ROOT_source36_pre_stop_entire_selector_metadata_actual_CPU_review_v1.json'
assert pin(candidate)['sha256']=='543bb6be8041615fb50d8ef4e4f3ba8561e5d74ee59fa02a3bc8672bb394b280'
assert pin(closure)['sha256']=='824cb18cc5ad72582a026ba7273c2472061a87593b2d045df2bec75d1037301f'
assert pin(stop_ROOT)['sha256']=='625bc51d3eedfa243674f3b8217f477dd634477fbf46ae1d015ae35bfe27e7a9'
assert pin(finite_ROOT)['sha256']=='2dc84d02f35210d839dc0504993638e676d67e4719e66a7c5fbd5c8a1923e532'
assert pin(typed_ROOT)['sha256']=='92d09e9da4ac82dcce0830229522d0b3658e643669eab99a420deefac33d6421'
base=read(candidate);value=read(closure)
assert base['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'] is None
assert base['one_postDEV_gc_refresh_policy']['completed_CPU_review']==pin(finite_ROOT)
assert value['last_fully_saved_optimizer_updates']==178800 and value['actual_optimizer_updates']==value['published_journal_optimizer_updates']==178820
assert value['optimizer_vs_journal_frontier']['unpublished_optimizer_counter_advance']==0
assert value['monitor_completion']['joined'] is True and value['monitor_completion']['clean'] is True and value['monitor_completion']['error'] is None
assert value['actual_outer_TOOL_exit_code']==1 and value['normal0_claimed'] is False and value['owned_session']==65067
old_deaths=[{'identity':w,'exact_dead':not alive(w)} for w in value['all_five_runtime_identities']];assert all(x['exact_dead'] for x in old_deaths)
lease=read(P/'work/gpu_training_owner.json');assert lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None
reader={'pid':26892,'create_time':1792652904.3359847,'ppid':30128};assert alive(reader)
proper_path=N/'resident_RAM_original_interval170469_failure_actual_v1/production_configuration_source32_v1/configuration.json'
proper=read(proper_path);inverse=copy.deepcopy(base)
del inverse['one_postDEV_gc_refresh_policy'];del inverse['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'];assert inverse==proper
configuration=copy.deepcopy(base);configuration['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']=pin(closure)
back=copy.deepcopy(configuration);back['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']=None;assert back==base
pointer=value['evidence']['pointer'];assert pin(pointer['path'])==pointer
ptr=read(pointer['path']);assert ptr['committed_updates']==178800 and ptr['journal_prefix']==value['saved_journal_prefix']
latest=read(B/'latest_checkpoint.json');assert latest['pointer']==pointer and latest['committed_updates']==178800 and latest['full_state_verified'] is True
assert ptr['journal_prefix']['offset']==360796716 and ptr['journal_prefix']['sha256']=='ea1f1ee763d18c09bdcaa37830f49ae26ca3ad843ce835cbf977e755c4619ef8'
assert value['unsaved_positive_tail_count']==20 and value['preserved_unsaved_positive_tail']['bytes']==39338
# Reuse accepted tail/hash values ONLY; do not open/hash the tail or old journal.
driver=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py');wrapper=pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py')
assert driver['sha256']=='d8b9797195bf5dd1981e3a4043f86fba369fd4e7706d9535038e7f7b788e1d5a'
assert wrapper['sha256']=='178a07ebe6c6170680861f701614808003b8d2bac9287837836ec272246bd297'
runtime_kinds=['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration','resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration']
old_actual=read(B/'invocations/owner_000024/configuration.json')
assert {k:v for k,v in old_actual.items() if k not in runtime_kinds}=={k:v for k,v in proper.items() if k not in runtime_kinds}
control=N/'resident_RAM_training_actual_25';child=read(control/'current_child.json');command=child['command']
assert command.count('--latest-failure')==1;ack_path=Path(command[command.index('--latest-failure')+1]);ack=pin(ack_path)
assert read(ack_path)['kind']=='ASR18_actual_ACKed260_singlepass_control_namespace_failure_TOOL1_v1'
directory=F/'production_configuration_source36_v1';config_path=directory/'configuration.json'
runtime_path=F/'final_source36_configuration_and_future7_runtime_bindings_v1.json'
handoff_path=F/'final_source36_one_postDEV_refresh_transition_execution_handoff_v1.json'
result_path=F/'actual_source36_final_metadata_materialization_result_v1.json'
future_review=F/'ROOT_source36_one_postDEV_gc_refresh_saved178800_final_execution_review_v1.json'
assert all(not p.exists() for p in (directory,runtime_path,handoff_path,result_path,future_review))
assert not(B/'invocations/owner_000025').exists() and not(B/'journals/commits_000025.jsonl').exists() and not(N/'resident_RAM_training_actual_26').exists()
config_bytes=raw(configuration);config_pin=prospective_pin(config_path,config_bytes)
bindings={'worker_driver_source':driver,'parent_driver_source':driver,'resident_RAM_stream_configuration':config_pin,
 'resident_branch_namespace':str(B.resolve()),'checkpoint_interval_updates':200,'source_checkpoint':pointer,'source_configuration':config_pin}
assert set(bindings)==set(runtime_kinds)
validation={'restored_updates':178800,'start_updates':178820,'end_updates':181020,'next_scheduled_DEV_updates':180000,
 'formula':'start=restored+20; end=max(start+1600,nextscheduledDEV+1020)','updates':2200,'all_original20_rows':110,'natural200_saves':11,
 'actual_first_periodic_opportunity':180000,'all_pauses_realDEV_and_saves_included':True,'production_finite_stop':False,
 'comparison':'Whole/ALL200 blocks/first400/final400/two postDEV400; comparable nonDEV400 target about75+ vs original76.569475, save-excluded first-final sustained decline<=5%; no warm-only or causal guarantee.'}
assert validation['end_updates']==max(validation['start_updates']+1600,validation['next_scheduled_DEV_updates']+1020)
arguments=['--configuration',str(config_path.resolve()),'--pointer',pointer['path'],'--latest-failure',ack['path'],
 '--branch',str(B.resolve()),'--review',str(future_review.resolve()),'--output',str((N/'resident_RAM_training_actual_26').resolve()),
 '--start-invocation','25','--queue-bytes','268435456','--gc-refresh-resume',str(closure.resolve())]
assert len(arguments)==18
runtime_report={'kind':'ASR18_final_source36_configuration_and_future7_runtime_bindings_v1','source_configuration':config_pin,
 'candidate_configuration':pin(candidate),'ONLY_candidate_delta':'prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition: None -> actual accepted closure824cb18c',
 'actual_transition':pin(closure),'actual_ROOT_stop_review':pin(stop_ROOT),'proper_source33_base':pin(proper_path),
 'proper_source33_all_old_values_exact_after_only_two_operational_fields':True,'future_runtime_binding_kinds':runtime_kinds,
 'future_runtime_values':bindings,'old_actual_runtime_configuration':pin(B/'invocations/owner_000024/configuration.json'),
 'BOTH_stream_and_source_bind_actual_serialized_final_config':True,'exact_serialized_UTF8_LF_bytes_no_prewriteline_hash_guess':True}
# ALL assertions and freshness checks above precede the first output creation.
directory.mkdir()
with config_path.open('xb') as f:f.write(config_bytes);f.flush();os.fsync(f.fileno())
assert pin(config_path)==config_pin
write(runtime_path,runtime_report)
handoff={'kind':'ASR18_final_INACTIVE_source36_one_postDEV_refresh_transition_execution_handoff_v1','source_configuration':config_pin,
 'runtime7_report':pin(runtime_path),'future_seven_runtime_bindings':bindings,'driver_source':driver,'wrapper_source':wrapper,
 'CPU_source':pin(S/'fastmss_resident_feature_stream_v13.py'),'RAM_native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),
 'DEV_helper_source':pin(S/'asr18_resident_serial350_DEV_v2.py'),'gc_refresh_resume':pin(closure),
 'gc_refresh_transition_validator':pin(S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py'),
 'one_postDEV_gc_refresh_policy':configuration['one_postDEV_gc_refresh_policy'],'initial_checkpoint':pointer,'latest_failure':ack,
 'actual_ROOT_pre_stop_review':value['evidence']['ROOT_pre_stop_review'],'actual_ROOT_stop_review':pin(stop_ROOT),
 'actual_finite_CPU_ROOT_review':pin(finite_ROOT),'actual_pre_stop_typed_CPU_ROOT_review':pin(typed_ROOT),
 'candidate_configuration':pin(candidate),'proper_source33_base':pin(proper_path),
 'recovery_observer_source':pin(F/'seal_source36_dynamic_natural_recovery_v1.py'),
 'long_validation_observer_source':pin(F/'observe_source36_dynamic_durable_validation_v2.py'),
 'dynamic_observers_ROOT_source_review':pin(F/'ROOT_source36_dynamic_recovery_long_observer_inactive_source_review_v1.json'),
 'initial_owner_invocation':25,'new_journal_invocation':25,'new_control_output':str((N/'resident_RAM_training_actual_26').resolve()),
 'exact_future_wrapper_arguments':arguments,'future_final_ROOT_execution_review_path':str(future_review.resolve()),
 'predeclared_actual_production_validation':validation,
 'required_recovery':{'natural_saved_updates':179000,'postpositive_at_least':179020,'only_new_positive_updates':[178801,178821],
  'count':21,'new_journal_saved_offset':360796716,'old20_historical_must_genuinely_redo':True,
  'full_restore_207_threeAdam_fourRNG_bothsamplers_nextdraw_BEFORE_savedprefix_copy':True,'native_draws_in_copy':0},
 'old_original24_and_preserved_tail_not_reread_by_materializer':True,'separate_old_numeric_OS_exits_UNKNOWN':True,
 'historical_stop_sealer_ctime_launcher_UNRECORDED_preserved':True,
 'allstate_real_full854_event_DB_lifetime_fit_speed_or_opaque_cycle_absence_not_proved_by_CPU_controls':True,
 'original_guards_science_225_MAX2_LIVE1_FULLreservations_unchanged_except_reviewed_once_event16MiB':True,
 'production_launch_and_liveGC_admitted_by_this_handoff':False,'final_ROOT_exact_ONE_execution_required':True,'human_fix_open':True}
write(handoff_path,handoff)
result={'kind':'ASR18_actual_source36_final_metadata_materialization_result_v1','passed':True,
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':pin(__file__),'source_configuration':pin(config_path),
 'runtime7_report':pin(runtime_path),'final_handoff':pin(handoff_path),'candidate_configuration':pin(candidate),'actual_transition':pin(closure),
 'actual_ROOT_stop_review':pin(stop_ROOT),'exact_old5_deaths':old_deaths,'reader_exact_alive':reader,
 'materializer_identity':identities[0],'materializer_launcher_identity':identities[1],'materializer_pair_live_at_write':True,
 'initial_sampled_metadata_resources':samples,'no_candidate_import_GC_journal_tail_tensor_audio_draw_stop_or_launch':True}
write(result_path,result)
print(json.dumps({'configuration':pin(config_path),'runtime7':pin(runtime_path),'handoff':pin(handoff_path),'result':pin(result_path),
 'materializer_pair_live_at_write':identities,'frozen_validation':validation}),flush=True)
