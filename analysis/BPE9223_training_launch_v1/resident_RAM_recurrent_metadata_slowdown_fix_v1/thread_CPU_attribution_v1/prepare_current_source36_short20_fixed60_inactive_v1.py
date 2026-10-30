"""Immutable metadata/literal preparation only. Never import or execute the candidates."""
import ast,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025'
SCOPE=F/'ROOT_source36_short20_profile_fixed60_counter_inactive_preparation_scope_v1.json'
OLD_OBS=F/'observe_existing_source36_thread_CPU_and_stacks_v1.py'
OLD_EXT=F/'launch_existing_source36_thread_CPU_external_v1.py'
NEW_OBS=F/'observe_existing_source36_thread_CPU_and_stacks_v2.py'
NEW_EXT=F/'launch_existing_source36_thread_CPU_external_v2.py'
LEDGER=F/'current_source36_short20_fixed60_entire_RAW_inverse_v1.json'
HANDOFF=F/'current_source36_short20_fixed60_inactive_handoff_v1.json'
RESULT=F/'current_source36_short20_fixed60_inactive_preparation_result_v1.json'
OUTPUT=F/'existing_source36_thread_CPU_actual_02'
EXTERNAL_RESULT=F/'existing_source36_thread_CPU_external_result_v2.json'
BEGIN=time.monotonic();ME=psutil.Process();ME.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

def require(v,m):
 if not v:raise ValueError(m)

def read(p):
 require(p.stat().st_size<=8*1024**2,'Bounded metadata/source only');return json.loads(p.read_text(encoding='utf-8-sig'))

def data_pin(p,b):return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def pin(p):
 require(p.stat().st_size<=8*1024**2,'Compact/source pin only');return data_pin(p,p.read_bytes())

def write(p,v):
 with p.open('xb')as f:f.write((json.dumps(v,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())

def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}

def exact(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False

def guard():
 samples=[sample_memory(p.pid,'source36_short20_fixed60_inactive_metadata_preparation')for p in (ME,ME.parent())]
 require(all(s['paired_sampling_within_one_second']for s in samples),'Original paired metadata samples')
 m=samples[-1]['system_memory'];total=sum(s['worker_private_memory']['private_bytes']for s in samples)
 require(total<=536870912 and time.monotonic()-BEGIN<120 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000,'Unchanged full512MiB pair/fullprospective512/full225/FULLprefetch/floors/E100GB/120s')
 return {'sampled_metadata_pair_private_bytes':total,'samples':samples}

def normalized(raw):
 tree=ast.parse(raw.decode('utf-8-sig'))
 for n in ast.walk(tree):
  if isinstance(n,ast.Constant):n.value='<BOUND_LITERAL_TYPE:'+type(n.value).__name__+'>'
 return ast.dump(tree,include_attributes=False)

def transform(old_path,new_path,edits):
 original=old_path.read_bytes();out=original;records=[]
 for old,new,count in edits:
  a,b=old.encode(),new.encode();require(out.count(a)==count,'Exact RAW count '+old)
  out=out.replace(a,b);records.append({'old_literal':old,'new_literal':new,'expected_count':count})
 inverse=out
 for e in reversed(records):
  a,b=e['old_literal'].encode(),e['new_literal'].encode();require(inverse.count(b)==e['expected_count'],'Unambiguous inverse '+e['new_literal']);inverse=inverse.replace(b,a)
 require(inverse==original,'FULL old RAW bytes reconstructed')
 require(normalized(original)==normalized(out),'Every nonliteral AST exact; no algorithm change')
 return out,{'old_source':data_pin(old_path,original),'new_source':data_pin(new_path,out),'newline_normalization':'NONE; unchanged CR/LF bytes preserved','edits_in_forward_order':records,'full_RAW_forward_and_inverse_exact':True,'nonliteral_AST_and_literal_types_exact':True}

def main():
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'E process-local CPU metadata preparation only')
 scope=read(SCOPE);require(scope['accepted']is True and scope['reviewer']=='/root'and pin(SCOPE)['sha256']=='95855d11f5d651d142b3aaf058d20c3b76410721550358aa19a85823e33c6c66','Exact inactive preparation scope')
 initial=guard()
 future=[NEW_OBS,NEW_EXT,LEDGER,HANDOFF,RESULT,OUTPUT,EXTERNAL_RESULT,F/'existing_source36_thread_CPU_v2_stdout.log',F/'existing_source36_thread_CPU_v2_stderr.log']
 require(all(not p.exists()for p in future),'All new immutable preparation and future actual02 paths absent')
 require(pin(OLD_OBS)['sha256']=='4cd28666266c1523e40273b75bc80e5f9d8df618bbec115a0eccca7530094242'and pin(OLD_EXT)['sha256']=='92a1ddda330f2c62c50ee5a7d866ca2faedfbf8e021d60b10ad5add519be254b','Frozen consumed actual01 sources preserved')
 obs_edits=[
  ('INACTIVE one60s exact existing-process read-only attribution. No GPU or state call.','INACTIVE fixed60s CPU and nominal20s20Hz role profile. No GPU or state call.',1),
  ('existing_source36_thread_CPU_actual_01','existing_source36_thread_CPU_actual_02',1),
  ('ASR18_existing_source36_GetThreadTimes_nonblocking60s_attribution_v1','ASR18_existing_source36_GetThreadTimes_short20Hz20s_fixed60_attribution_v1',1),
  ('Actual current CPU13 cohort and launcher, exact ROOT-bound post180k identities','Actual current CPU13 cohort and launcher, exact ROOT-bound post184k identities',1),
  ('after actual182420','after actual185020',1),
  ("['actual_optimizer_updates']>=182420","['actual_optimizer_updates']>=185020",1),
  ("['last_saved_updates']>=182400","['last_saved_updates']>=185000",1),
  ('Sample only after fixed actual182420 and natural182400 on source36 post180k late phase; no startup/postDEV-warm attribution','Sample only after fixed actual185020 and natural185000 on source36 post184k late phase; no startup/postDEV-warm attribution',1),
  ("'--duration','60','--rate','100'","'--duration','20','--rate','20'",1),
  ('ASR18_existing_source36_thread_CPU_and_nonblocking_stacks_actual_v1','ASR18_existing_source36_thread_CPU_and_short20Hz20s_fixed60_stacks_actual_v2',1)]
 ext_edits=[
  ('One separately reviewed existing-source36 nonblocking observation; external guard remains active through all parsing/writing/cleanup.','One separately reviewed source36 nominal20s20Hz profile with fixed60s CPU; original external guard remains active through parsing/writing/cleanup.',1),
  ('observe_existing_source36_thread_CPU_and_stacks_v1.py','observe_existing_source36_thread_CPU_and_stacks_v2.py',1),
  ('existing_source36_thread_CPU_external_result_v1.json','existing_source36_thread_CPU_external_result_v2.json',1),
  ('existing_source36_thread_CPU_actual_01','existing_source36_thread_CPU_actual_02',1),
  ('ASR18_existing_source36_GetThreadTimes_nonblocking60s_attribution_v1','ASR18_existing_source36_GetThreadTimes_short20Hz20s_fixed60_attribution_v1',1),
  ('existing_source36_thread_CPU_v1_stdout.log','existing_source36_thread_CPU_v2_stdout.log',1),
  ('existing_source36_thread_CPU_v1_stderr.log','existing_source36_thread_CPU_v2_stderr.log',1),
  ('ASR18_existing_source36_thread_CPU_nonblocking_external_result_v1','ASR18_existing_source36_thread_CPU_short20Hz20s_fixed60_external_result_v2',1)]
 obs,obs_record=transform(OLD_OBS,NEW_OBS,obs_edits);ext,ext_record=transform(OLD_EXT,NEW_EXT,ext_edits)
 parser_path=F/'finite_thread_counter_mapping_fixture_v2.py';parser_raw=parser_path.read_bytes();tree=ast.parse(parser_raw.decode('utf-8-sig'))
 parser=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='profile_profiles')
 function=ast.get_source_segment(parser_raw.decode('utf-8-sig'),parser)
 require("p['unit']=='seconds'"in function and "math.fsum(p['weights'])"in function and not any(isinstance(n,ast.Constant)and type(n.value)in(int,float)and n.value in(100,.01)for n in ast.walk(parser)),'Existing parser actual seconds/weights; no100Hz assumption; HOLD rather than parser change')
 parser_check={'source':pin(parser_path),'function':'profile_profiles','function_AST_sha256':hashlib.sha256(ast.dump(parser,include_attributes=False).encode()).hexdigest(),'unit_seconds_required':True,'actual_weights_math_fsum':True,'no_nominal100Hz_conversion':True,'no_parser_import_or_execution_or_modification':True}
 ready_path=I/'CPU_after_periodic_DEV_000184000/actual_CPU_ready.json';ready=read(ready_path);cpu=ready['producer_identity'];launcher=who(psutil.Process(cpu['ppid']))
 require(exact(cpu)and launcher['ppid']==21912 and ready['source']['protocol']==pin(S/'fastmss_resident_feature_stream_v13.py')and ready['packet_credit']['logical_packet_maximum']==2,'Genuine current184k CPU13MAX2 ready; no old40004 current binding')
 protected=[{'role':'native','pid':21912,'create_time':1792780981.1092248,'ppid':37108},{'role':'native_launcher','pid':37108,'create_time':1792780981.1024935,'ppid':29600},{'role':'wrapper','pid':29600,'create_time':1792780979.698954,'ppid':40472},{'role':'CPU13',**cpu},{'role':'CPU_launcher',**launcher},{'role':'reader','pid':26892,'create_time':1792652904.3359847,'ppid':30128},{'role':'reader_launcher','pid':30128,'create_time':1792652904.283631,'ppid':10644}]
 require(all(exact(v)for v in protected),'Current protected7 exact identity observation only')
 ledger={'kind':'ASR18_source36_short20_fixed60_full_RAW_inverse_v1','preparation_scope':pin(SCOPE),'observer':obs_record,'external':ext_record,'generic_original_parser_source_check':parser_check,'helper_controller_parser_scope_lifetime_guard_algorithms_exact':True,'sampler_close_sole_owner_no_new_OpenThread':True,'new_metadata_writer':pin(Path(__file__))}
 ledger_bytes=(json.dumps(ledger,indent=2)+'\n').encode();ledger_pin=data_pin(LEDGER,ledger_bytes)
 old_handoff=read(F/'current_source36_post_refresh_readiness_attribution_inactive_handoff_v1.json')
 keep={'fixed_counter_window_and_scalar_files_v1.py','owned_thread_times_v2.py','finite_thread_counter_mapping_fixture_v2.py','finite_thread_counter_mapping_fixture_v3.py','ROOT_new_thread_counter_nativeTID_mapping_actual_CPU_review_v1.json','ROOT_fixed60_counter_controller_METHOD_STUB_actual_CPU_review_v1.json','report.json','py-spy.exe','ROOT_pinned_py_spy_0_4_2_isolated_E_preparation_review_v1.json','fastmss_resident_feature_stream_v13.py','asr18_BPE9223_private_scalar_prefetch_v1.py','resource_watchdog_v2.py','storage_config.json','use_storage.ps1','run_ASR18_BPE9223_resident_RAM_training_v36.py','configuration.json','boot.json'}
 inputs=[v for v in old_handoff['input_pins']if Path(v['path']).name in keep]
 require(len(inputs)==len(keep),'Carry each unchanged algorithm/proof/tool/storage/current immutable source pin once, without rereading closed raw data')
 extra=[SCOPE,F/'ROOT_source36_smaller_role_profile_readonly_proposal_review_v1.json',F/'source36_actual01_failure_least_intrusive_next_observation_readonly_proposal_v1.json',F/'ROOT_source36_fixed60_actual01_failed_wall_bound_and_closure_review_v1.json',F/'ROOT_source36_actual01_new_failed_profiler_logs_exact_once_read_v1.json',ready_path,Path(__file__)]
 inputs=[data_pin(NEW_OBS,obs),data_pin(NEW_EXT,ext),ledger_pin,*inputs,*[pin(p)for p in extra]]
 require(len({v['path']for v in inputs})==len(inputs),'No duplicate ambiguous pins')
 handoff={'kind':'ASR18_source36_short20Hz20s_role_profile_fixed60_CPU_inactive_handoff_v1','execution_admitted':False,'preparation_scope':pin(SCOPE),'requested_future_scope':'ASR18_existing_source36_GetThreadTimes_short20Hz20s_fixed60_attribution_v1','live_observer_source':data_pin(NEW_OBS,obs),'external_live_launcher_source':data_pin(NEW_EXT,ext),'whole_RAW_inverse':ledger_pin,'input_pins':inputs,'unchanged_proof_metadata_pins_carried_not_reaudited':True,'target_identity':old_handoff['target_identity'],'CPU_source':old_handoff['CPU_source'],'runtime_configuration':old_handoff['runtime_configuration'],'current_CPU_ready':pin(ready_path),'protected_production_identities':protected,'reader_identity':old_handoff['reader_identity'],'reader_compact_paths':old_handoff['reader_compact_paths'],'completed_new_counter_CPU_review':old_handoff['completed_new_counter_CPU_review'],'completed_fixed_counter_controller_CPU_review':old_handoff['completed_fixed_counter_controller_CPU_review'],'completed184000_ROOT_quality_review':None,'ROOT184k_acceptance_and_current_new_cohort_must_be_bound_before_future_execution':True,'minimum_actual_update':185020,'minimum_saved_update':185000,'trigger_not_rate_selected':True,'profile_nominal_duration_seconds':20,'profile_rate_hz':20,'requested_yielded_intervals':400,'not_a_hard20s_wall_guarantee':True,'CPU_counter_seconds':60,'profile_and_CPU_intervals_DISTINCT':True,'profiler_exit_timestamp_semantics':'EXISTING first observed by controller at/after counter60; early actual exit may precede observation. Popen/actual profile weights/exit observation are retained; no new alignment algorithm or actual20wall claim.','parser_actual_weights_seconds_source_check':parser_check,'fixed_limits_unchanged':scope['original_fixed_limits'],'limits':old_handoff['limits'],'stderr_explicit_cap_bytes':4194304,'stdout_compact_pin_cap_bytes':8388608,'output_path':str(OUTPUT.resolve()),'external_result_path':str(EXTERNAL_RESULT.resolve()),'exact_external_command_prefix':old_handoff['exact_external_command_prefix'],'exact_external_arguments':['-Xutf8','-B',str(NEW_EXT.resolve()),'--ROOT-review','__PENDING_SEPARATE_EXACT_ROOT_EXECUTION_SCOPE__'],'strict_nativeTID_creation_name_frame_and_full60s_lifetime_or_UNKNOWN_unchanged':True,'sampler_close_SOLE_handle_owner_and_TOTAL256_cap_unchanged':True,'all_metadata_families_sampled_not_hard_allocation_bound':True,'no_new_finite_proof_merely_for_elapsed_time':True,'no_automatic_retry_second_sample_or_remedy_selection':True,'same_production_session_42997_unchanged':True,'production_source_config_science_LR_GC_guards_reader_environment_unchanged':True,'no_attach_proof_import_or_production_action':True}
 handoff_bytes=(json.dumps(handoff,indent=2)+'\n').encode();final=guard()
 for path,data in ((NEW_OBS,obs),(NEW_EXT,ext),(LEDGER,ledger_bytes),(HANDOFF,handoff_bytes)):
  with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 value={'kind':'ASR18_source36_short20_fixed60_inactive_metadata_preparation_result_v1','passed':True,'writer_source':pin(Path(__file__)),'preparation_scope':pin(SCOPE),'observer':pin(NEW_OBS),'external':pin(NEW_EXT),'RAW_ledger':pin(LEDGER),'handoff':pin(HANDOFF),'initial_guard':initial,'final_guard':final,'metadata_identity':who(ME),'metadata_launcher_identity':who(ME.parent()),'candidate_import_or_CPUproof_profile_counter_or_production_action':False,'future_actual02_and_external_result_absent':not OUTPUT.exists()and not EXTERNAL_RESULT.exists()}
 write(RESULT,value)
 print(json.dumps({'result':pin(RESULT),'observer':pin(NEW_OBS),'external':pin(NEW_EXT),'ledger':pin(LEDGER),'handoff':pin(HANDOFF),'input_count':len(inputs),'execution_admitted':False}),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  value={'kind':'ASR18_source36_short20_fixed60_inactive_metadata_preparation_result_v1','passed':False,'error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},'metadata_identity':who(ME),'metadata_launcher_identity':who(ME.parent()),'no_candidate_or_production_operation':True,'no_automatic_retry':True}
  if not RESULT.exists():write(RESULT,value)
  print(json.dumps({'failure':pin(RESULT),'error':value['error']}),flush=True);raise
