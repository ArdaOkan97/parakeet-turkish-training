"""Metadata-only immutable literal successor preparation. Never import/attach it."""
import ast,copy,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;L=F.parent/'one_postDEV_gc_refresh_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025'
BEGIN=time.monotonic();ME=psutil.Process();ME.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
SCOPE=L/'ROOT_source36_post_refresh_native_readiness_attribution_inactive_preparation_scope_v1.json'
OLD_OBS=F/'observe_existing_source33_thread_CPU_and_stacks_v2.py';OLD_EXT=F/'launch_existing_source33_thread_CPU_external_v2.py'
NEW_OBS=F/'observe_existing_source36_thread_CPU_and_stacks_v1.py';NEW_EXT=F/'launch_existing_source36_thread_CPU_external_v1.py'
LEDGER=F/'current_source36_post_refresh_readiness_attribution_entire_RAW_inverse_v1.json'
HANDOFF=F/'current_source36_post_refresh_readiness_attribution_inactive_handoff_v1.json'
RESULT=F/'current_source36_readiness_inactive_preparation_outcome_v1.json'
OUTPUT=F/'existing_source36_thread_CPU_actual_01';EXTERNAL_RESULT=F/'existing_source36_thread_CPU_external_result_v1.json'
def require(v,m):
 if not v:raise ValueError(m)
def read(p):
 require(p.stat().st_size<=8*1024**2,'Compact source/metadata only');return json.loads(p.read_text(encoding='utf-8-sig'))
def data_pin(p,b):return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def pin(p):
 require(p.stat().st_size<=8*1024**2,'Bounded source/tool/metadata pin only');return data_pin(p,p.read_bytes())
def write(p,v):
 with p.open('xb')as f:f.write((json.dumps(v,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def guard():
 samples=[sample_memory(p.pid,'source36_inactive_readiness_metadata_preparation')for p in (ME,ME.parent())]
 require(all(s['paired_sampling_within_one_second']for s in samples),'Paired original metadata memory samples')
 m=samples[-1]['system_memory'];total=sum(s['worker_private_memory']['private_bytes']for s in samples)
 require(total<=536870912 and time.monotonic()-BEGIN<120 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000,'Unchanged512MiB/fullprospective512/full225/FULLprefetch/floors/E100GB/120s metadata bounds')
 return {'sampled_metadata_pair_private_bytes':total,'samples':samples}
def normalized_ast(raw):
 tree=ast.parse(raw.decode('utf-8-sig'))
 for n in ast.walk(tree):
  if isinstance(n,ast.Constant):n.value='<BOUND_LITERAL_TYPE:'+type(n.value).__name__+'>'
 return ast.dump(tree,include_attributes=False)
def transform(old_path,new_path,edits):
 raw=old_path.read_bytes();out=raw;records=[]
 for old,new,expected in edits:
  a,b=old.encode(),new.encode();count=out.count(a)
  require(count>0 and (expected is None or count==expected),'Exact literal replacement count: '+old)
  out=out.replace(a,b);records.append({'old_literal':old,'new_literal':new,'expected_count':count})
 inverse=out
 for e in reversed(records):
  a,b=e['old_literal'].encode(),e['new_literal'].encode();require(inverse.count(b)==e['expected_count'],'Unambiguous exact RAW inverse replacement');inverse=inverse.replace(b,a)
 require(inverse==raw,'FULL original RAW bytes reconstructed')
 require(normalized_ast(raw)==normalized_ast(out),'All nonliteral AST structure exact; only declared metadata literals')
 return out,{'old_source':data_pin(old_path,raw),'new_source':data_pin(new_path,out),'newline_normalization':'NONE; raw byte replacement preserves every unchanged CR/LF byte','edits_in_forward_order':records,'full_raw_forward_and_reverse_exact':True,'all_AST_structure_and_literal_types_exact_except_declared_values':True,'no_candidate_import_or_execution':True}
def main():
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'E process-local CPU-only metadata preparation environment')
 scope=read(SCOPE);require(scope['admitted_preparation_only']is True and scope['execution_or_import_of_candidate_or_profile_counter_proof_job_admitted']is False and pin(SCOPE)['sha256']=='8fdea74fe2530ca3d0ff9f7dc97913c028487b93dff731b6c226dae868e84f6c','Exact inactive preparation scope only')
 initial=guard()
 require(all(not p.exists()for p in (NEW_OBS,NEW_EXT,LEDGER,HANDOFF,RESULT,OUTPUT,EXTERNAL_RESULT,F/'existing_source36_thread_CPU_v1_stdout.log',F/'existing_source36_thread_CPU_v1_stderr.log')),'Every new immutable output and future observation namespace fresh')
 require(pin(OLD_OBS)['sha256']=='2b817f09c6456ff31aba61326055c9ba540f8edec9b133018ba015e0d7af6449' and pin(OLD_EXT)['sha256']=='1df01616efdcabe476e4c5077df7800baf518b94b4640d3bf5832d58531dfa9c','Accepted immutable source33 bases only')
 obs_edits=[
  ('source33','source36',None),
  ('invocations/owner_000024','invocations/owner_000025',1),
  ('existing_source36_thread_CPU_actual_02','existing_source36_thread_CPU_actual_01',1),
  ("TARGET={'pid':3652,'create_time':1792764340.2396317,'ppid':31268}","TARGET={'pid':21912,'create_time':1792780981.1092248,'ppid':37108}",1),
  ('eaa9da9d88cdc7086b2d77ee2fe52a8b36aa9dfcfe8ef20ed3678298c3a030b6','d8b9797195bf5dd1981e3a4043f86fba369fd4e7706d9535038e7f7b788e1d5a',1),
  ('5428362f43aeb4bda2a9591524ff5f447a9e81267dc45efe5c68c16410fce5ee','f866afb3e9cd7e8db15c341e7df5ecbc8c6cc7d6714153783c389fafc54498a7',1),
  ('PROTECTED={3652,31268,30732,26892}','PROTECTED={21912,37108,29600,26892}',1),
  ('Same exact native3652 must remain alive','Same exact native21912 must remain alive',1),
  ('Exact current source36/I24/native/runtime','Exact current source36/I25/native/runtime',1),
  ('Actual current CPU13 cohort and launcher, never historical40192 after172k restart','Actual current CPU13 cohort and launcher, exact ROOT-bound post180k identities',1),
  ("{TARGET['pid'],TARGET['ppid'],30732,26892}","{TARGET['pid'],TARGET['ppid'],29600,26892}",1),
  ('Sample only after actual scheduled172k DEV plus1020 and natural173000/post173020; no startup/postDEV-warm attribution','Sample only after fixed actual182420 and natural182400 on source36 post180k late phase; no startup/postDEV-warm attribution',1),
  ('after actual173020','after actual182420',1),
  ("['actual_optimizer_updates']>=173020","['actual_optimizer_updates']>=182420",1),
  ("['last_saved_updates']>=173000","['last_saved_updates']>=182400",1),
  ('run_ASR18_BPE9223_resident_RAM_training_v33.py','run_ASR18_BPE9223_resident_RAM_training_v36.py',2)]
 ext_edits=[
  ('source33','source36',None),
  ('observe_existing_source36_thread_CPU_and_stacks_v2.py','observe_existing_source36_thread_CPU_and_stacks_v1.py',1),
  ('existing_source36_thread_CPU_actual_02','existing_source36_thread_CPU_actual_01',1),
  ('PROTECTED={3652,31268,30732,26892}','PROTECTED={21912,37108,29600,26892}',1)]
 obs,obs_record=transform(OLD_OBS,NEW_OBS,obs_edits);ext,ext_record=transform(OLD_EXT,NEW_EXT,ext_edits)
 ledger={'kind':'ASR18_source36_readiness_attribution_entire_RAW_inverse_v1','preparation_scope':pin(SCOPE),'observer':obs_record,'external_launcher':ext_record,'same_directory_helper_imports_unchanged':True,'no_new_ThreadQuery_or_helper_controller_algorithm_change':True,'sampler_close_sole_handle_owner_and_all_bounds_unchanged':True,'new_metadata_source':pin(Path(__file__))}
 ledger_bytes=(json.dumps(ledger,indent=2)+'\n').encode();ledger_pin=data_pin(LEDGER,ledger_bytes)
 old_handoff=read(F/'current_source33_fixed60_existing_handle_live_inactive_handoff_v2.json')
 keep_names={'fixed_counter_window_and_scalar_files_v1.py','owned_thread_times_v2.py','finite_thread_counter_mapping_fixture_v2.py','finite_thread_counter_mapping_fixture_v3.py','ROOT_new_thread_counter_nativeTID_mapping_actual_CPU_review_v1.json','ROOT_fixed60_counter_controller_METHOD_STUB_actual_CPU_review_v1.json','fixed_counter_controller_actual_completion_v1.json','report.json','py-spy.exe','ROOT_pinned_py_spy_0_4_2_isolated_E_preparation_review_v1.json','fastmss_resident_feature_stream_v13.py','asr18_BPE9223_private_scalar_prefetch_v1.py','resource_watchdog_v2.py','storage_config.json','use_storage.ps1'}
 inputs=[v for v in old_handoff['input_pins']if Path(v['path']).name in keep_names]
 require(len(inputs)==len(keep_names),'Every unchanged runtime/tool/helper/proof/source/storage pin exactly once')
 # Reuse immutable closed proof metadata, never raw profile/counter/audio records.
 more=[F/'current_source33_fixed60_existing_handle_live_inactive_handoff_v2.json',F/'ROOT_current_source33_fixed60_sampler_sole_owner_inactive_source_review_v1.json',SCOPE,L/'source36_remaining_outside_native_gap_readonly_proposal_v1.json',L/'ROOT_source36_remaining_outside_native_readiness_scope_readonly_proposal_review_v1.json',L/'ROOT_source36_after181020_latest400_181520_181920_sustained_below75_observation_v1.json',L/'ROOT_source36_actual_boot_runtime7_saved178800_restore_verification_v1.json',L/'ROOT_source36_actual_first_periodic180000_idle_refresh_preservation_verification_v1.json',L/'ROOT_source36_frozen2200_178820_181020_actual_long_result_verification_v1.json',L/'ROOT_source36_one_postDEV_gc_refresh_saved178800_final_execution_review_v1.json',R/'run_ASR18_BPE9223_resident_RAM_training_v36.py',I/'configuration.json',I/'boot.json',I/'CPU_after_periodic_DEV_000180000/actual_CPU_ready.json',I/'actual_one_postDEV_gc_refresh.json',Path(__file__)]
 inputs.extend(pin(p)for p in more)
 inputs=[data_pin(NEW_OBS,obs),data_pin(NEW_EXT,ext),ledger_pin,*inputs]
 require(len({v['path']for v in inputs})==len(inputs),'No ambiguous duplicate input pins')
 require(pin(I/'configuration.json')['sha256']=='f866afb3e9cd7e8db15c341e7df5ecbc8c6cc7d6714153783c389fafc54498a7' and pin(I/'CPU_after_periodic_DEV_000180000/actual_CPU_ready.json')['sha256']=='2a953c460e4a19107206b5cfd45724cc3168c857c5dc733c02487aeb1a6e8c43','Exact current immutable runtime and phaseCPU binding')
 handoff={'kind':'ASR18_source36_post_refresh_readiness_inactive_attribution_handoff_v1','execution_admitted':False,'preparation_scope':pin(SCOPE),'requested_scope':'ASR18_existing_source36_GetThreadTimes_nonblocking60s_attribution_v1','live_observer_source':data_pin(NEW_OBS,obs),'external_live_launcher_source':data_pin(NEW_EXT,ext),'whole_RAW_inverse':ledger_pin,'input_pins':inputs,'target_identity':{'pid':21912,'create_time':1792780981.1092248,'ppid':37108},'CPU_source':pin(S/'fastmss_resident_feature_stream_v13.py'),'runtime_configuration':pin(I/'configuration.json'),'current_CPU_ready':pin(I/'CPU_after_periodic_DEV_000180000/actual_CPU_ready.json'),'minimum_actual_update':182420,'minimum_saved_update':182400,'trigger_not_rate_selected':True,'protected_production_identities':[{'role':'native','pid':21912,'create_time':1792780981.1092248,'ppid':37108},{'role':'native_launcher','pid':37108,'create_time':1792780981.1024935,'ppid':29600},{'role':'wrapper','pid':29600,'create_time':1792780979.698954,'ppid':40472},{'role':'CPU13','pid':40004,'create_time':1792783315.0228927,'ppid':21652},{'role':'CPU_launcher','pid':21652,'create_time':1792783315.0045743,'ppid':21912},{'role':'reader','pid':26892,'create_time':1792652904.3359847,'ppid':30128},{'role':'reader_launcher','pid':30128,'create_time':1792652904.283631,'ppid':10644}],'reader_identity':{'pid':26892,'create_time':1792652904.3359847,'ppid':30128},'reader_compact_paths':{name:str((Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')/file).resolve())for name,file in [('current_status','current_status.json'),('mirror_state','mirror_state.json'),('owner','owner.json')]},'completed_new_counter_CPU_review':old_handoff['completed_new_counter_CPU_review'],'completed_fixed_counter_controller_CPU_review':old_handoff['completed_fixed_counter_controller_CPU_review'],'completed_controller_gate_explicitly_consumed_before_any_attach':True,'output_path':str(OUTPUT.resolve()),'external_result_path':str(EXTERNAL_RESULT.resolve()),'exact_external_command_prefix':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe'],'exact_external_arguments':['-Xutf8','-B',str(NEW_EXT.resolve()),'--ROOT-review','__PENDING_SEPARATE_EXACT_ROOT_SCOPE__'],'limits':old_handoff['limits'],'CPU_counter_and_profile_intervals_distinct':True,'strict_nativeTID_creation_name_frame_lifetime_roles_and_UNKNOWN_semantics_unchanged':True,'total256_handle_bound_and_sampler_close_sole_owner_unchanged':True,'native_only_profile_not_feature_CPU_attach':True,'trigger_existing_last5_rows_once_only':True,'sampled_family_bounds_not_hard_allocation_claim':True,'stderr_explicit_cap_bytes':4194304,'stdout_compact_pin_cap_bytes':8388608,'no_automatic_retry_or_remedy_selection':True,'only_target_read_operations':'Nonblocking pinned profiler and fixed60 original GetThreadTimes helper; NO target signal, suspend, injection, locals, model, audio, tensor, GPU, state, journal, archive, GC or source operation','same_sole_owned_production_session':42997,'all_training_reader_environment_science_225_MAX2_LIVE1_guards_unchanged':True,'no_closed_proof_rerun':True}
 final=guard()
 # Every assertion and complete serialization occurs before creating new immutable files.
 handoff_bytes=(json.dumps(handoff,indent=2)+'\n').encode()
 for path,data in ((NEW_OBS,obs),(NEW_EXT,ext),(LEDGER,ledger_bytes),(HANDOFF,handoff_bytes)):
  with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 result={'kind':'ASR18_source36_readiness_inactive_metadata_preparation_outcome_v1','passed':True,'source':pin(Path(__file__)),'preparation_scope':pin(SCOPE),'new_observer':pin(NEW_OBS),'new_external':pin(NEW_EXT),'ledger':pin(LEDGER),'handoff':pin(HANDOFF),'initial_guard':initial,'final_guard':final,'metadata_identity':who(ME),'metadata_launcher_identity':who(ME.parent()),'candidate_or_helper_import_attach_counter_GC_model_audio_state_or_production_operation':False,'future_output_absent':not OUTPUT.exists(),'future_external_result_absent':not EXTERNAL_RESULT.exists()}
 write(RESULT,result);print(json.dumps({'outcome':pin(RESULT),'observer':pin(NEW_OBS),'external':pin(NEW_EXT),'ledger':pin(LEDGER),'handoff':pin(HANDOFF),'input_count':len(inputs),'execution_admitted':False}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  failure={'kind':'ASR18_source36_readiness_inactive_metadata_preparation_outcome_v1','passed':False,'error':{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},'metadata_identity':who(ME),'metadata_launcher_identity':who(ME.parent()),'candidate_import_or_execution':False,'no_automatic_retry':True}
  if not RESULT.exists():write(RESULT,failure)
  print(json.dumps({'failure_outcome':pin(RESULT)if RESULT.exists()else None,'error':failure['error']}),flush=True)
  raise
