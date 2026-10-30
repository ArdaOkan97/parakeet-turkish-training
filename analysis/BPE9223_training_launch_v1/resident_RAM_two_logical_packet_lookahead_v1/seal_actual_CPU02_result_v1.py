import hashlib,json,os
from pathlib import Path
import psutil
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def exactdead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
out=F/'finite_transport_actual_CPU_02';report=read(out/'report.json');external=read(F/'finite_transport_actual_CPU_02_external_result.json');entry=read(F/'finite_transport_actual_CPU_02_external_entry.json')
assert external['actual_Popen_wait_code']==0 and external['external_guard_error']is None and external['targeted_cleanup']==[]
assert report['passed']is True and report['maximum_logical_packets']==2 and report['maximum_live_receiver_packets']==1
normal=report['results']['ordered_two_plans_and_periodic_drain'];again=report['results']['same_plan_resubmit']
for item in (normal,again):
 assert item['actual_Popen_returncode']==0 and item['threads_joined']and item['pipe_handles_closed']
 assert item['actual_read_exact_view_weakrefs_dead']and item['receiver_raw_read_peak']==1 and item['receiver_raw_read_active_at_closure']==0
 assert item['credit_owner']['all_owned_kernel_handles_closed']and item['credit_owner']['last_release_previous_count']==1
 assert item['normal']['CPU_actual0']['packet_credit']['all_owned_kernel_handles_closed']
forms=[]
def walk(v):
 if isinstance(v,dict):
  if isinstance(v.get('pid'),int)and isinstance(v.get('create_time'),(int,float)):forms.append({k:v[k]for k in ('pid','create_time','ppid')if k in v})
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(report);walk(external);walk(entry)
seen=set();deaths=[]
for w in forms:
 key=(w['pid'],w['create_time'])
 if key not in seen:seen.add(key);deaths.append({'identity':w,'exact_identity_dead':exactdead(w)})
assert all(x['exact_identity_dead']for x in deaths)
tool={'kind':'ASR18_actual_two_logical_packet_CPU02_TOOL_outcome_v1','owned_exec_session':10059,
 'initial_running_chunk_id':'49fdd8','actual_completion_chunk_id':'2b02e0','actual_TOOL_exit_code':0,
 'actual_external_Popen_wait_code':0,'external_result':pin(F/'finite_transport_actual_CPU_02_external_result.json'),
 'report':pin(out/'report.json'),'no_exit_inferred':True}
write(F/'finite_transport_actual_CPU_02_TOOL_outcome.json',tool)
paths=[F/'finite_transport_actual_CPU_02_TOOL_outcome.json',F/'finite_transport_actual_CPU_02_external_entry.json',
 F/'finite_transport_actual_CPU_02_external_launch.json',F/'finite_transport_actual_CPU_02_external_result.json',F/'finite_transport_actual_CPU_02_stdout.log']
paths.extend(x for x in out.rglob('*')if x.is_file())
closure={'kind':'ASR18_actual_two_logical_packet_CPU02_exact_death_closure_v1','passed':True,
 'actual_TOOL_exit_code':0,'actual_external_Popen_wait_code':0,'external_guard_error':None,'required_targeted_cleanup':[],
 'all_recorded_exact_dead':True,'recorded_identity_forms':deaths,'observed_distinct_PID_count':len({x['identity']['pid']for x in deaths}),
 'report':pin(out/'report.json'),'external_result':pin(F/'finite_transport_actual_CPU_02_external_result.json'),
 'evidence_pins':[pin(x)for x in paths],'frozen_input_pins':external['inputs_before_and_after_exact'],
 'normal40_and_sameplan20':{'normal40':normal,'same_plan20':again},
 'all_normal_actual_read_exact_views_dead':True,'parent_array_base_memoryview_refs_dead_before_posts':True,
 'normal_child_handles_closed_after_acquire_unwound_and_parent_handles_closed_after_child_wait_transport_joins':True,
 'positive_copy_claim':'Conditional admitted complete packets only; no frontend-growth/process/allocator/kernel-pipe hard bound',
 'old_failed01_preserved_and_not_promoted':True,'genuine_audio_model_GPU_fit_or_speed_claimed':False,
 'fixture_constructor_small_preparation_and_fakeclock_explicit':True,'production_candidate_completed_proof_remains_None':True}
write(F/'finite_transport_actual_CPU_02_exact_death_closure.json',closure)
print(json.dumps({'report':pin(out/'report.json'),'external_result':pin(F/'finite_transport_actual_CPU_02_external_result.json'),
 'TOOL':pin(F/'finite_transport_actual_CPU_02_TOOL_outcome.json'),'closure':pin(F/'finite_transport_actual_CPU_02_exact_death_closure.json'),
 'cases':len(report['results']),'normal40_actual_posts':normal['credit_owner']['successful_release_posts'],
 'normal40_release_histogram':normal['credit_owner']['release_previous_count_histogram'],
 'sameplan20_posts':again['credit_owner']['successful_release_posts'],'exact_identity_forms':len(deaths),
 'distinct_PIDs':len({x['identity']['pid']for x in deaths}),'peak_family':external['sampled_external_plus_owned_family_peak_bytes'],
 'elapsed':external['elapsed_seconds'],'native_or_audio':0}))
