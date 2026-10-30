import copy,hashlib,json
from pathlib import Path
import psutil
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
out=F/'typed_selector_actual_01';r=read(out/'report.json');e=read(F/'typed_selector_actual_01_external_result.json');entry=read(F/'typed_selector_actual_01_external_entry.json')
assert r['passed']is True and e['actual_Popen_wait_code']==0 and e['external_guard_error']is None and e['targeted_cleanup']==[]
forms=[]
def walk(v):
 if isinstance(v,dict):
  if isinstance(v.get('pid'),int)and isinstance(v.get('create_time'),(int,float)):forms.append({k:v[k]for k in ('pid','create_time','ppid')if k in v})
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(r);walk(e);walk(entry)
seen=set();deaths=[]
for w in forms:
 key=(w['pid'],w['create_time'])
 if key not in seen:seen.add(key);deaths.append({'identity':w,'exact_identity_dead':dead(w)})
assert all(x['exact_identity_dead']for x in deaths)
tool={'kind':'ASR18_actual_source29_entire_selector_metadata_TOOL_outcome_v1','actual_TOOL_exit_code':0,
 'actual_chunk_id':'22e776','actual_external_Popen_wait_code':0,'completed_without_yielded_exec_session':True,
 'report':pin(out/'report.json'),'external_result':pin(F/'typed_selector_actual_01_external_result.json'),'no_exit_inferred':True}
write(F/'typed_selector_actual_01_TOOL_outcome.json',tool)
closure={'kind':'ASR18_actual_source29_entire_selector_metadata_exact_death_closure_v1','passed':True,'actual_TOOL_exit_code':0,
 'actual_external_Popen_wait_code':0,'external_guard_error':None,'targeted_cleanup':[],
 'all_recorded_exact_dead':True,'recorded_identities':deaths,'input_pins_before_after_exact':e['inputs_before_and_after_exact'],
 'report':pin(out/'report.json'),'external_result':pin(F/'typed_selector_actual_01_external_result.json'),
 'TOOL':pin(F/'typed_selector_actual_01_TOOL_outcome.json'),
 'evidence_pins':[pin(p)for p in [out/'entry.json',out/'report.json',F/'typed_selector_actual_01_external_entry.json',
 F/'typed_selector_actual_01_external_launch.json',F/'typed_selector_actual_01_external_result.json',F/'typed_selector_actual_01_stdout.log']],
 'no_native_GPU_audio_NumPy_Torch_restore_prefixcopy_draw_kernelcredit_or_authority_operation':True,
 'entire_dictionary_and_forwarding_executed_with_declared_metadata_doubles':True,
 'gap0_gap1_synthetic_trace_method_stubs_are_not_actual_training_events':True}
write(F/'typed_selector_actual_01_exact_death_closure.json',closure)
h=read(F/'final_source29_actual_checkpoint_transition_execution_handoff_v1.json')
h.update(typed_metadata_actual_report=pin(out/'report.json'),typed_metadata_actual_external_result=pin(F/'typed_selector_actual_01_external_result.json'),
 typed_metadata_actual_TOOL=pin(F/'typed_selector_actual_01_TOOL_outcome.json'),typed_metadata_actual_closure=pin(F/'typed_selector_actual_01_exact_death_closure.json'),
 prior_handoff=pin(F/'final_source29_actual_checkpoint_transition_execution_handoff_v1.json'),
 actual_stop_ROOT_review=pin(F/'ROOT_source27_approved_stop158620_saved158600_actual_verification_v1.json'),
 recovery_observation={'saved':158600,'historical_positive_tail':20,'genuine_new_ordered_commits_required':[158601,158621],
 'first_natural_verified_checkpoint':158800,'postpositive_minimum':158820,
 'fixed_poststartup_checkpoint_inclusive_speed_interval':[158620,159020],'continued_full_objective':401072,'no_finite_stop':True})
write(F/'final_source29_actual_checkpoint_transition_execution_handoff_v2.json',h)
print(json.dumps({'report':pin(out/'report.json'),'external':pin(F/'typed_selector_actual_01_external_result.json'),
 'TOOL':pin(F/'typed_selector_actual_01_TOOL_outcome.json'),'closure':pin(F/'typed_selector_actual_01_exact_death_closure.json'),
 'final_handoff':pin(F/'final_source29_actual_checkpoint_transition_execution_handoff_v2.json'),
 'actual_cases':len(r['results']),'all_exact_identity_forms':len(deaths),'Popen_exit':0,'elapsed':e['elapsed_seconds'],
 'peak_family':e['sampled_external_plus_owned_family_peak_bytes']}))
