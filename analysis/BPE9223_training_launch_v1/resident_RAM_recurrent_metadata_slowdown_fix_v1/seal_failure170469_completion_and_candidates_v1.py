"""Compact metadata only; no decode/query/draw/payload/original-journal operation."""
import hashlib,json,os,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');F=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1';OUT=N/'resident_RAM_original_interval170469_failure_actual_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();assert p.stat().st_size<=4*1024**2;b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def close(v):
 try:
  ct=psutil.Process(v['pid']).create_time();return {'identity':v,'exact_dead':abs(ct-v['create_time'])>=.001,'current_create_time':ct,'numeric_OS_exit':'UNKNOWN'}
 except psutil.NoSuchProcess:return {'identity':v,'exact_dead':True,'observation':'NoSuchProcess','numeric_OS_exit':'UNKNOWN'}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
s=sample_memory(os.getpid(),'source31_failure170469_candidate_metadata_completion');m=s['system_memory']
assert s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<536870912 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000
closure_path=OUT/'actual_failure_closure.json';assert pin(closure_path)['sha256']=='e2e1e2aadf1cc423067a24bba0531f5bad67995e6c54031ab882dcdcddac215e'
closure=read(closure_path);sealer_deaths=[close(closure[k])for k in ('sealer_identity','sealer_launcher_identity')];assert all(v['exact_dead']for v in sealer_deaths)
cpu=read(I/'CPU_after_periodic_DEV_000168000/actual_CPU_failure.json');ready=read(I/'CPU_after_periodic_DEV_000168000/actual_CPU_ready.json')
assert cpu['completed_updates']==2469 and ready['producer_identity']['pid']==38964 and cpu['actual_resource_telemetry']['last_observed_memory']['worker_private_memory']['pid']==38964
origin_path=I/'plans/plan_000031.json';origin=read(origin_path)
assert origin['initial_committed_updates']==168000 and len(origin['updates'])==20
assert all(v['update_index']==i for i,v in enumerate(origin['updates']))
ordinal_plan=31+cpu['completed_updates']//20;index=cpu['completed_updates']%20
plan_path=I/'plans'/('plan_'+str(ordinal_plan).zfill(6)+'.json');plan=read(plan_path)
assert ordinal_plan==154 and index==9 and plan['initial_committed_updates']==170460 and len(plan['updates'])==20
entry=plan['updates'][index];assert entry['update_index']==index
assert all(v['update_index']==i for i,v in enumerate(plan['updates']))
rows=entry['rows'];assert len(rows)==16
ordered=[]
for row_index,row in enumerate(rows):
 if row['language']=='tr':
  raw=row['broad_original'];ordered.append({'original_update_row_1based':row_index+1,'source_id':row['source_id'],'ordinal':row['ordinal'],'dataset':raw['dataset'],'raw_original_annotation':raw,'full_original_plan_row':row,'original_row_canonical_sha256':digest(row),'raw_annotation_canonical_sha256':digest(raw)})
assert len(ordered)==14
candidate={'kind':'ASR18_actual_CPU_completed_counter_original_plan154_index9_ordered14_candidates_v1','actual_failure_closure':pin(closure_path),'actual_CPU_failure':pin(I/'CPU_after_periodic_DEV_000168000/actual_CPU_failure.json'),'CPU_ready':pin(I/'CPU_after_periodic_DEV_000168000/actual_CPU_ready.json'),'CPU_producer_identity':ready['producer_identity'],'CPU_protocol_source':pin(S/'fastmss_resident_feature_stream_v11.py'),'source_based_counter_semantics':'worker_main completed=0 for this new168k cohort, increment once only AFTER successful _send, process ordered prepare_plan updates; no extra normal drain before next172kDEV','closed_168k_sameplan_origin_ROOT_review':pin(F/'ROOT_periodic168000_fixed350_and_CPU11_max2_phase_partial_verification_v1.json'),'origin_plan':pin(origin_path),'origin_initial_updates':168000,'completed_sent_packets':2469,'all_intervening_windows_in_same_regular20update_region':'168000..170480 is before next172000 DEV and first fullpass boundary; current actual plan154 initial170460/20entries independently matches completed-count arithmetic','recorded_target_plan':pin(plan_path),'recorded_target_plan_canonical_sha256':digest(plan),'next_unsent_CPU_update_index':9,'derived_original_global_update_from_plan_initial_plus_index':170470,'derivation_is_NOT_native_actual_plus_one':True,'exact_target_entry_canonical_sha256':digest(entry),'ordered14_original_TR_candidates':ordered,'exact_failed_row_or_member':'UNLOCALIZED; actualCPUstderr identifies original yodas decoder coverage error but not row/member','proposed_next_scope':'ONE bounded ordered decoder-only14-row comparison using unchanged decoder2+overlay7, stop first actual failure; baseline only same failed row. Requires exact ROOT source/scope before any decode. No frontend/model/GPU/fullmember/tolerance/skip/pad/crop/sourcechange/retry.','no_audio_frontend_tensor_model_GPU_or_authoritative_sampler_draw_operation':True,'metadata_only_plan_reads':['origin31 for counter origin, not scoring/state re-audit','target154 for exact existing planned candidates'],'all_other_closed_source31_recovery_quality_long_audits_not_repeated':True}
candidate_path=OUT/'actual_CPU_completed2469_plan154_index9_ordered14_candidates_v1.json';write(candidate_path,candidate)
value={'kind':'ASR18_source31_actual_failure170469_sealer_completion_v1','observed_UTC':datetime.now(timezone.utc).isoformat(),'actual_sealer_TOOL_exit':0,'actual_sealer_TOOL_chunk':'55948a','actual_closed_training_TOOL_exit':1,'actual_closed_training_TOOL_chunk':'fad938','failure_closure':pin(closure_path),'candidate_metadata':pin(candidate_path),'failure_sealer_exact_deaths':sealer_deaths,'all_original5production_exact_dead':closure['known_exact_production_deaths'],'all_known_dead_before_any_retry':True,'separate_native_CPU_wrapper_numeric_OS_exits':'UNKNOWN','saved170400_and_preserved69_suffix':closure['journal'],'no_original_suffix_reread_in_completion':True,'completion_process_identity':who(psutil.Process()),'completion_launcher_identity':who(psutil.Process().parent()),'completion_processes_alive_until_own_return':True,'guard':s,'performance_controller_proof_and_LIVE02_not_executed':True}
completion=OUT/'actual_failure_metadata_completion_v1.json';write(completion,value)
print(json.dumps({'candidate':pin(candidate_path),'completion':pin(completion),'sealer_deaths':sealer_deaths,'completion_process_identity':value['completion_process_identity'],'completion_launcher_identity':value['completion_launcher_identity'],'original_CPU_global_candidate':170470,'plan':ordinal_plan,'index':index,'TR_candidates':len(ordered)}))
