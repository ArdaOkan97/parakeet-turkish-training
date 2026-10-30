"""Read only the ONE sealed CPU/profile/timing capture; no process attachment."""
import collections,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import psutil
F=Path(__file__).resolve().parent;O=F/'existing_source33_thread_CPU_actual_02'
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
def guard(admission=False):
 s=sample_memory(os.getpid(),'source33_same_actual02_readonly_aggregation');m=s['system_memory']
 assert s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<=536870912
 assert m['commit_available_bytes']>=(5055306368 if admission else 2268435456)
 assert m['physical_available_bytes']>=(2305306368 if admission else 1768435456)
 assert shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<120
 return s
def read(p,cap):
 assert p.stat().st_size<=cap;return json.loads(p.read_bytes())
def pin(p,cap=8*1024**2):
 p=p.resolve();assert p.stat().st_size<=cap;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
initial=guard(True)
r=read(O/'report.json',8*1024**2);assert r['passed']and r['all_requested_roles_name_and_source_agree']
profile=read(O/'profile.speedscope.json',67108864);assert pin(O/'profile.speedscope.json',67108864)==r['profile']
frames=profile['shared']['frames'];roles={v['profile_name']:v for v in r['role_mapping']['mapped']}
def key(fr):return (fr.get('file',''),fr.get('name',''),fr.get('line'))
def rows(counter,limit):
 return [{'file':k[0],'function':k[1],'line':k[2],'sample_weight_seconds':v}for k,v in counter.most_common(limit)]
all_profiles=[];selected={}
for p in profile['profiles']:
 assert p['type']=='sampled'and p['unit']=='seconds'and len(p['samples'])==len(p['weights'])
 leaf=collections.Counter();inclusive=collections.Counter();stacks=collections.Counter()
 for stack,w in zip(p['samples'],p['weights']):
  assert type(w)in(int,float)and math.isfinite(w)and w>=0
  assert all(type(i)is int and 0<=i<len(frames)for i in stack)
  if not stack:continue
  ks=tuple(key(frames[i])for i in stack)
  leaf[ks[-1]]+=w
  for k in set(ks):inclusive[k]+=w
  stacks[ks]+=w
 summary={'profile_name':p['name'],'samples':len(p['samples']),
  'wall_sample_weight_seconds':math.fsum(p['weights']),'profile_start_value':p.get('startValue'),'profile_end_value':p.get('endValue')}
 all_profiles.append(summary)
 if p['name']in roles:
  role=roles[p['name']]['role'];selected[role]={**summary,'leaf_top30':rows(leaf,30),'inclusive_top40':rows(inclusive,40),
   'top15_full_python_stacks':[{'weight_seconds':v,'frames':[{'file':k[0],'function':k[1],'line':k[2]}for k in ks]}for ks,v in stacks.most_common(15)],
   'native_or_collect_leaf_callsites':[{'file':k[0],'function':k[1],'line':k[2],'sample_weight_seconds':v}for k,v in leaf.most_common()if any(x in k[0].lower()for x in ('native_transaction','collect','signature'))]}
 guard()
assert set(selected)=={'main','scalar_worker'}
counter=O/'thread_CPU_counters.jsonl';assert counter.stat().st_size<=33554432
by={};count=0
with counter.open('rb')as f:
 for line in f:
  v=json.loads(line);count+=1;assert count<=80000
  k=(v['native_tid'],v['creation_filetime_100ns'])
  a=by.setdefault(k,{'first':v,'last':v,'count':0});a['last']=v;a['count']+=1
assert pin(counter,33554432)==r['counter_snapshot']
cpu=[]
for mapped in r['role_mapping']['mapped']:
 a=by[(mapped['native_tid'],mapped['creation_filetime_100ns'])];first,last=a['first'],a['last']
 user=(last['user_100ns']-first['user_100ns'])/1e7;kernel=(last['kernel_100ns']-first['kernel_100ns'])/1e7
 wall=(last['monotonic_ns']-first['monotonic_ns'])/1e9
 assert user==mapped['user_seconds_delta']and kernel==mapped['kernel_seconds_delta']and a['count']==mapped['samples']
 assert first['creation_filetime_100ns']==last['creation_filetime_100ns']and last['exit_filetime_100ns']==0
 cpu.append({'role':mapped['role'],'profile_name':mapped['profile_name'],'native_tid':mapped['native_tid'],
  'creation_filetime_100ns':mapped['creation_filetime_100ns'],'samples':a['count'],'counter_interval_seconds':wall,
  'user_seconds':user,'kernel_seconds':kernel,'user_plus_kernel_seconds':user+kernel,
  'CPU_fraction_of_distinct_counter_interval':(user+kernel)/wall,
  'wall_minus_thread_CPU_seconds_not_attributed_to_a_cause':wall-user-kernel})
tail=r['trigger_existing_timing_tail'];ts=tail['rows'];assert len(ts)==5
timing={'initial_updates':ts[0]['initial_updates'],'committed_updates':ts[-1]['committed_updates'],
 'all5_original_rows':ts,'snapshot':tail['snapshot'],'source_range':{k:tail[k]for k in ('source_path','start_offset','end_offset','bytes','sha256')},
 'sum_native_seconds':math.fsum(v['native_seconds']for v in ts),'sum_window_pipeline_seconds':math.fsum(v['window_pipeline_seconds']for v in ts),
 'sum_source_audio_seconds':math.fsum(v['source_audio_seconds']for v in ts),'sum_checkpoint_seconds':math.fsum(v['checkpoint_seconds']for v in ts),
 'sum_worker_preparation_wall_seconds_overlapping_main':math.fsum(v['worker_preparation_seconds']for v in ts),
 'sum_original_collect_observed_wall_seconds_not_exclusive':sum(v['original_collect_observation']['elapsed_ns']for v in ts)/1e9,
 'gate_wait_seconds':math.fsum(v['native_commit_pressure_wait_seconds']for v in ts),
 'gate_waited_updates':sum(v['native_commit_pressure_waited_updates']for v in ts),
 'end_waits':sum(v['end_window_wait_count']for v in ts),'fallbacks':sum(v['serial_fallback_count']for v in ts)}
stderr_path=Path(r['profiler_stderr']['path']);assert stderr_path.stat().st_size<=4194304
stderr=stderr_path.read_text(encoding='utf-8',errors='replace')
error_lines=[line for line in stderr.splitlines()if 'error'in line.lower()or 'warn'in line.lower()]
assert len(error_lines)<=10000
value={'kind':'ASR18_current_source33_same_actual02_counter_and_profile_readonly_aggregation_v1',
 'inputs':{'report':pin(O/'report.json'),'profile':pin(O/'profile.speedscope.json',67108864),'counters':pin(counter,33554432),
  'completion':pin(F/'existing_source33_thread_CPU_actual02_completion_v1.json'),
  'final8exactdeaths':pin(F/'existing_source33_thread_CPU_actual02_final_exact_deaths_v1.json')},
 'all_profile_thread_weight_summaries_not_addable_to_process_wall':all_profiles,'requested_role_profiles':selected,
 'raw_counter_recomputed_role_CPU':cpu,'raw_counter_records':count,'raw_counter_thread_identities':len(by),
 'alignment':r['alignment'],'unknown_role_profiles_preserved':r['role_mapping']['unknown'],
 'sampler_races':r['sampler_open_or_exit_races'],'retained_lifetime_errors':r['retained_lifetime_acquire_errors'],
 'stderr':r['profiler_stderr'],'stderr_error_or_warning_line_count':len(error_lines),'stderr_error_warning_lines':error_lines,
 'trigger_existing5row_context':timing,'actual_process_IO_delta_allthreads_cache_not_phase_or_physical_IO':r['target_owned_process_IO_delta'],
 'limits':['Thread CPU is actual user+kernel inclusive C/finalizer work; counters do not identify individual function CPU',
  'Counter first/last interval is distinct from Python stack sample interval; no exclusive CPU-by-stack attribution',
  'Stack weights are sampled thread wall, inclusive entries overlap; no addition across threads or inclusive callsites',
  'Unattributed wall-minus-CPU may include GPU wait, I/O, scheduling or GIL; no single cause follows',
  'Native or collect Python callsite is not proof of pure GC, pure CUDA or exclusive CPU cost',
  'Only already captured5-row context; no new speed span, repeat attach, source/GC/guard/settings change or speed success'],
 'no_target_process_open_or_thread_query_or_profiler_or_model_audio_GPU_state_action':True,
 'resource_initial':initial,'resource_final':guard(),'analysis_source':pin(Path(__file__))}
out=F/'current_source33_actual02_same_capture_readonly_aggregation_v1.json'
with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'analysis':pin(out),'counter_roles':cpu,'profile_main':selected['main']['wall_sample_weight_seconds'],
 'main_top_leaf':selected['main']['leaf_top30'][:8],'scalar_top_leaf':selected['scalar_worker']['leaf_top30'][:5],
 'warning_error_lines':len(error_lines)}),flush=True)
