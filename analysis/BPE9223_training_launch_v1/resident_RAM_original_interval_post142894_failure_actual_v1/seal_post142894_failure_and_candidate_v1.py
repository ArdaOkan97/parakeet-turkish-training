"""One owned failure seal and one streaming journal pass; no audio/model/restore."""
import copy,datetime,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000015';A=N/'resident_RAM_training_actual_16'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def guard():
 v=sample_memory(os.getpid(),'post142894_failure_metadata_seal');m=v['system_memory']
 require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Finite metadata/private/full extra256MiB/original headroom/E100GB')
 return v
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Metadata CPU only');first=guard()
deaths=json.loads((F/'actual_recorded_owned_process_deaths.json').read_text(encoding='utf-8-sig'))
require(len(deaths)==7 and all(x['actual_PID_absent']for x in deaths),'All seven actually observed owned PIDs absent; keep exact known timestamps and missing ones unknown')
failure=read_json(I/'actual_failure.json');monitor=read_json(I/'monitor_completion.json');CPU=read_json(I/'CPU/actual_CPU_failure.json')
latest=read_json(B/'latest_checkpoint.json');pointer=read_json(latest['pointer']['path']);persistence=read_json(latest['persistence']['path'])
require(failure['actual_optimizer_updates']==142894 and failure['last_checkpoint']==142800
 and 'Original interval beyond source unresolved'in failure['error']and CPU['completed_updates']==2694
 and monitor['joined']and monitor['clean']and monitor['error']is None,'Exact actual closed coverage failure, no resource-failure relabel')
require(latest['committed_updates']==142800 and latest['full_state_verified']and pointer['committed_updates']==142800
 and persistence['passed']and persistence['optimizer_updates']==142800 and persistence['weights_verified']==207
 and persistence['Adam_states_verified']==215 and persistence['named_tensors_verified']==854
 and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],'Genuine saved142800 full-state publication metadata')
saved=pointer['journal_prefix'];journal=Path(saved['path']);require(journal==B/'journals/commits_000015.jsonl'
 and saved['offset']==290088487 and saved['sha256']=='e45e1adb098b5c1fdf4388116c4db6a61feb348320a1f396f0bdb73f3cbb4598','Exact actual saved journal prefix')
whole_hash=hashlib.sha256();prefix_hash=hashlib.sha256();tail_hash=hashlib.sha256();offset=0
tail_path=F/'preserved_unsaved_positive_tail142801_142894.jsonl';require(not tail_path.exists(),'One preserved tail copy')
before_size=journal.stat().st_size
with journal.open('rb')as src,tail_path.open('xb')as dst:
 while True:
  guard();block=src.read(1024**2)
  if not block:break
  whole_hash.update(block);n=min(len(block),max(0,saved['offset']-offset))
  if n:prefix_hash.update(block[:n])
  if n<len(block):tail_hash.update(block[n:]);dst.write(block[n:])
  offset+=len(block)
 dst.flush();os.fsync(dst.fileno())
require(offset==before_size==journal.stat().st_size and prefix_hash.hexdigest()==saved['sha256'],'One actual complete whole/prefix stream; no concurrent writer or original mutation')
tail_bytes=tail_path.read_bytes();events=[json.loads(x)for x in tail_bytes.splitlines()]
require(len(events)==94 and all(e['optimizer_updates']==e['native']['update']==e['prepared_evidence']['actual_positive_native_update']==u
 for u,e in zip(range(142801,142895),events)),'All94 ordered actual positive tail events retained, not replayed or discarded')
plan_pin=record(I/'plans/plan_000135.json');future_pin=record(I/'plans/plan_000136.json');plan=read_json(plan_pin['path']);future=read_json(future_pin['path'])
initial=read_json(I/'boot.json')['source_checkpoint'];initial_updates=read_json(initial['path'])['committed_updates']
index=initial_updates+CPU['completed_updates']-plan['initial_committed_updates']
require(initial_updates==140200 and plan['initial_committed_updates']==142880 and future['initial_committed_updates']==142900
 and len(plan['updates'])==len(future['updates'])==20 and index==14,'Actual completed CPU count localizes current plan135/index14; no native+1 assumption')
remaining={'source_plan':plan_pin,'initial_updates':142880,'first_unconsumed_update_index':index,
 'candidate_global_updates':plan['updates'][index]['global_update'],'not_assumed_native_plus_one':True,
 'CPU_completed_updates':CPU['completed_updates'],'CPU_initial_frontier':initial_updates,'future_plan':future_pin,'updates':plan['updates'][index:]}
require(remaining['candidate_global_updates']==142895,'Candidate is from actual ordered plan, not observed failing row yet')
write_json(F/'remaining_current_plan_updates.json',remaining)
rows=copy.deepcopy(plan['updates'][index]['rows']);require(len(rows)==16,'Exact full16-row native candidate')
yodas=[{'row_index':i,'row':row}for i,row in enumerate(rows)if row.get('dataset')=='alania_yodas3']
require(yodas,'Exact production original dataset alias, no synthetic empty filter')
candidate={'source_plan':plan_pin,'actual_CPU_completed_updates':2694,'CPU_initial_updates':140200,'failed_candidate_update_index':14,
 'candidate_global_updates':142895,'ordered_completed_CPU_updates_localize_not_native_plus_one_assumption':True,
 'all16_original_rows':rows,'ordered_yodas3_original_rows':yodas,'number_of_ordered_yodas3_rows':len(yodas),
 'actual_failed_row_not_yet_observed':True,'no_audio_or_frontend_executed':True}
write_json(F/'exact_failed_update_candidate_rows.json',candidate)
evidence={}
for key,path in {'actual_failure':I/'actual_failure.json','CPU_failure':I/'CPU/actual_CPU_failure.json','CPU_stderr':I/'CPU/CPU_stderr.log',
 'CPU_ready':I/'CPU/actual_CPU_ready.json','monitor_completion':I/'monitor_completion.json','boot':I/'boot.json',
 'actual_restore':I/'actual_parent_restore.json','effective_configuration':I/'configuration.json','current_child':A/'current_child.json',
 'native_log':A/'native_000015.log','progress':B/'progress/latest.json','latest_checkpoint':B/'latest_checkpoint.json',
 'lease':P/'work/gpu_training_owner.json','periodic_state':B/'periodic_DEV.json','pointer':Path(latest['pointer']['path']),
 'persistence':Path(latest['persistence']['path']),'source19':P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v19.py',
 'wrapper19':P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v19.py','CPU8_source':S/'fastmss_resident_feature_stream_v8.py',
 'ROOT_execution':N/'ROOT_sixth_original_terminal_interval_saved140200_execution_review_v1.json'}.items():evidence[key]=record(path)
closure={'kind':'ASR18_actual_original_interval_post142894_failure_closed_v1','observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'owned_session':1903,'actual_outer_TOOL_exit_code':1,'actual_TOOL_chunk':'1f7856','native_numeric_OS_exit_code':None,
 'native_Popen_wait_known_nonzero_from_wrapper_assertion':True,'CPU_numeric_OS_exit_code':None,
 'actual_optimizer_updates':142894,'last_fully_saved_updates':142800,'unsaved_positive_tail_updates':94,
 'whole_journal_verified_once':{'path':str(journal),'bytes':offset,'sha256':whole_hash.hexdigest()},'saved_prefix_verified_once':saved,
 'preserved_unsaved_positive_tail':{'path':str(tail_path),'bytes':len(tail_bytes),'sha256':tail_hash.hexdigest()},'all94_ordered_positive_events_verified':True,
 'owned_process_deaths':record(F/'actual_recorded_owned_process_deaths.json'),'all7_recorded_PIDs_absent':True,
 'known_numeric_create_times_preserved':3,'known_UTC_create_times_preserved':2,'unknown_historical_create_times_not_invented':2,
 'monitor_clean_and_joined':monitor,'CPU_failure_completed_prepared_updates':2694,'actual_CPU_error':'Original interval beyond source unresolved',
 'captured_CPU_resource_telemetry_distinct_from_coverage_error':CPU['actual_resource_telemetry'],
 'remaining_current_plan':record(F/'remaining_current_plan_updates.json'),'candidate_rows':record(F/'exact_failed_update_candidate_rows.json'),
 'current_plan':plan_pin,'future_private_plan':future_pin,'diagnosis_not_assumed_native_plus_one':True,'exact_failing_row_within_candidate_not_yet_observed':True,
 'inputs':evidence,'no_tensor_payload_or_audio_read':True,'one_whole_journal_stream_only':True,'no_unchanged_retry_or_GPU_probe':True,
 'no_new_training_or_production_source_or_guard_change':True,'full_target':401072,'administrative_update_stop':None,
 'prior_sixth_recovery_and140k_quality_remain_historical':record(N/'ROOT_sixth_original_terminal_interval_saved140400_actual_verification_v1.json'),
 'initial_resource_sample':first,'final_resource_sample':guard()}
write_json(F/'actual_failure_closure.json',closure)
print(json.dumps({'closure':record(F/'actual_failure_closure.json'),'candidate':record(F/'exact_failed_update_candidate_rows.json'),
 'ordered_yodas3_rows':len(yodas),'actual_failed_row_not_yet_observed':True,'saved':142800,'actual':142894,
 'whole_journal_once':closure['whole_journal_verified_once'],'preserved94_tail':closure['preserved_unsaved_positive_tail']}),flush=True)
