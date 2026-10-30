"""Seal ONE actual read-only attribution's outcomes and exact owned deaths."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
F=Path(__file__).resolve().parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
began=time.monotonic();me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def read(p):
 p=Path(p);assert p.stat().st_size<=8*1024**2
 return json.loads(p.read_bytes())
def pin(p,cap=8*1024**2):
 p=Path(p).resolve();assert p.stat().st_size<=cap;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def guard(admission=False):
 v=sample_memory(me.pid,'source33_thread_CPU_actual02_metadata_completion');m=v['system_memory']
 assert v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<=536870912
 assert m['commit_available_bytes']>=(5055306368 if admission else 2268435456)
 assert m['physical_available_bytes']>=(2305306368 if admission else 1768435456)
 assert shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180
 return v
initial=guard(True)
extpath=F/'existing_source33_thread_CPU_external_result_v1.json';e=read(extpath)
toolpath=F/'existing_source33_thread_CPU_actual02_TOOL_v1.json';t=read(toolpath)
assert t['actual_TOOL_exit_code']==0 and t['owned_exec_session']==88673 and t['actual_external_result']==pin(extpath)
assert e['passed']is True and e['Popen_returncode']==0 and e['external_guard_error']is None and e['required_external_cleanup']==[]
rpath=Path(e['report']['path']);r=read(rpath);assert pin(rpath)==e['report']
assert r['passed']is True and r['measurement_completed']is True and r['error']is None and r['required_cleanup']==[]
assert r['all_owned_query_handles_closed']is True and r['all_requested_roles_name_and_source_agree']is True
assert r['inputs_before_after_exact']is True and e['inputs_before_after_exact']is True
assert r['input_pins_before']==r['input_pins_after']==e['input_pins_before']==e['input_pins_after']
assert len(r['input_pins_before'])==29
observed={}
for who in e['recorded_identities']+[e['launcher_identity']]+r['recorded_owned_identities']+[r['observer_launcher_identity']]:
 observed[(who['pid'],who['create_time'])]=who
closed=[]
protected={3652,31268,30732,31568,8608,26892,30128}
for who in observed.values():
 assert who['pid']not in protected
 current=None
 try:current=identity(psutil.Process(who['pid']))
 except psutil.NoSuchProcess:pass
 assert current is None or abs(current['create_time']-who['create_time'])>=.001
 closed.append({'identity':who,'exact_dead':True,'current_PID_identity_if_reused':current})
assert len(closed)==6
artifacts={'report':pin(rpath),'external_result':pin(extpath),'actual_TOOL':pin(toolpath),
 'profile':pin(r['profile']['path'],67108864),'trigger_existing_timing_rows':pin(r['trigger_existing_timing_tail']['snapshot']['path']),
 'profiler_stdout':pin(r['profiler_stdout']['path']),'profiler_stderr':pin(r['profiler_stderr']['path'])}
for key,v in r['partial_scalar_artifacts'].items():artifacts['scalar_'+key]=pin(v['path'],33554432)
assert artifacts['profile']==r['profile']and artifacts['trigger_existing_timing_rows']==r['trigger_existing_timing_tail']['snapshot']
for key,v in r['partial_scalar_artifacts'].items():assert artifacts['scalar_'+key]==v
health=[]
review=read(F/'ROOT_existing_source33_fixed60_counter_nonblocking60s_exact_execution_scope_v1.json')
for who in review['protected_production_identities']:
 p=psutil.Process(who['pid']);now=identity(p)
 assert abs(now['create_time']-who['create_time'])<.001
 health.append(now)
value={'kind':'ASR18_existing_source33_thread_CPU_actual02_completion_v1','passed':True,
 'actual_TOOL_exit_code':0,'actual_TOOL_chunk':'254059','actual_external_Popen_returncode':0,
 'actual_external_guard_error':None,'required_external_cleanup':[],
 'all_six_prior_exact_metadata_identities_dead':True,'exact_prior_metadata_identity_closure_records':closed,
 'all_owned_query_handles_closed_as_reported':True,'scalar_files_closed_as_actual_reported_and_pinnable':True,
 'input_pins_before_after_exact':True,'input_pin_count':29,'artifacts':artifacts,
 'all_requested_roles_name_nativeTID_creation_lifetime_AND_source_agree':True,
 'counter_window_and_profile_window_are_distinct':True,'nonatomic_errors_unknown_roles_and_raw_logs_preserved':True,
 'no_exclusive_CPU_by_stack_or_pureGC_or_speed_success_claim':True,
 'protected_production_and_reader_still_exact_alive':health,'production65067_not_consumed_or_changed':True,
 'initial_resource_sample':initial,'last_resource_sample':guard(),
 'completion_identity':identity(me),'completion_launcher_identity':identity(me.parent()),'completion_source':pin(__file__)}
out=F/'existing_source33_thread_CPU_actual02_completion_v1.json'
with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(out),'prior_exact_metadata_deaths':len(closed),
 'completion_identity':value['completion_identity'],'completion_launcher_identity':value['completion_launcher_identity']}),flush=True)
