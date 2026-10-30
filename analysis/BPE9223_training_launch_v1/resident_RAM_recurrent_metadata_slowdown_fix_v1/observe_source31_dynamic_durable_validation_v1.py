"""ONE metadata observation of the frozen actual training interval; no benchmark.
Run after complete final timing append and natural standard durability/postpositive.
No metadata archive, tensor, audio, journal, model, profiler, GC or GPU operation.
"""
import argparse,hashlib,json,math,os,shutil,statistics,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023';A=F.parent/'resident_RAM_training_actual_24'
HANDOFF=F/'final_source31_signature_collect_transition_execution_handoff_v1.json';OUT=F/'actual_source31_dynamic_long_validation_01';BEGIN=time.monotonic();PEAK=0
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def live(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def guard(admission=False):
 global PEAK
 s=sample_memory(os.getpid(),'source31_frozen_long_actual_job_metadata_observer');m=s['system_memory'];PEAK=max(PEAK,s['worker_private_memory']['private_bytes']);extra=536870912 if admission else 0
 require(s['paired_sampling_within_one_second']and PEAK<536870912 and time.monotonic()-BEGIN<180
  and m['commit_available_bytes']>=2268435456+extra and m['physical_available_bytes']>=1768435456+extra
  and shutil.disk_usage('E:/').free>=100000000000,'Original512MiB/180s/full512prospective+FULLprefetch/floors/E100GB')
 return s
def snapshot(p,cap):
 p=Path(p);size=p.stat().st_size;offset=max(0,size-cap)
 with p.open('rb')as f:f.seek(offset);raw=f.read(size-offset)
 require(len(raw)==size-offset,'Exact append-only captured byte range')
 n=raw.find(b'\n')+1 if offset else 0;complete=raw[n:]
 if not complete.endswith(b'\n'):complete=complete[:complete.rfind(b'\n')+1]
 return complete,{'path':str(p.resolve()),'captured_offset':offset+n,'captured_end':size,'snapshot_bytes':len(complete),'sha256':hashlib.sha256(complete).hexdigest()}
def checkpoint(update):
 ck=B/'checkpoints'/('update_'+str(update).zfill(9));p=read(ck/'pointer.json');v=read(ck/'persistence_verified.json')
 require(p['committed_updates']==update and v['passed']is True and v['optimizer_updates']==update
  and v['named_tensors_verified']==854 and v['weights_verified']==207 and v['Adam_states_verified']==215
  and v['independent_pre_save_state_sha256']==v['saved_state_sha256']and v['saved_semantic_state_verified']['passed']is True
  and v['journal_prefix']==p['journal_prefix'],'Natural full854/207/215 PRE-SAVED semantic persistence; no payload read')
 return {'update':update,'pointer':pin(ck/'pointer.json'),'persistence':pin(ck/'persistence_verified.json'),'payload_read':False}
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
  and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
  and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Separate E CPU-only observer env; no production env change')
 review=read(a.execution_review);require(review['accepted']is True and review['reviewer']=='/root'and review['long_validation_observer_source']==pin(__file__),'Exact final ROOT binding includes this observer')
 require(review['final_execution_handoff']==pin(HANDOFF),'Exact materialized actual-stop handoff pin')
 h=read(HANDOFF);v=h['predeclared_actual_production_validation'];restored=read(h['initial_checkpoint']['path'])['committed_updates']
 start=restored+20;dev=(restored//4000+1)*4000;end=max(start+1600,dev+1020)
 require(v['restored_updates']==restored and v['start_updates']==start and v['end_updates']==end
  and v['next_scheduled_DEV_updates']==dev and v['formula']=='start=restored+20; end=max(start+1600,nextscheduledDEV+1020)'
  and (end-start)%200==0,'Entire interval frozen from actual restore frontier before launch, never performance-selected')
 source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v31.py');require(review['driver_source']==source and review['source_configuration']==h['source_configuration'],'Same exact admitted source31/config')
 initial=guard(True);require(not OUT.exists(),'ONE fresh full actual-job observation, no automatic rerun')
 progress=read(B/'progress/latest.json');boot=read(I/'boot.json')
 require(progress['source']==source and progress['producer']==boot['producer']and progress['actual_optimizer_updates']>=end
  and not progress['optimizer_poisoned']and not progress['schedule_poisoned']and progress['planned_native_updates']==401072
  and progress['administrative_update_stop']is None and not(I/'actual_failure.json').exists(),'Healthy same full401072 owner through final emitted progress')
 raw,tp=snapshot(I/'private_metadata_window_timings.jsonl',8*1024**2);rows=[];lines=[]
 for line in raw.splitlines(keepends=True):
  row=json.loads(line)
  if start<=row['initial_updates']and row['committed_updates']<=end:rows.append(row);lines.append(line)
 require([(r['initial_updates'],r['committed_updates'])for r in rows]==[(u,u+20)for u in range(start,end,20)],'ALL contiguous original20 rows, including complete final append')
 gcrows=[]
 for r in rows:
  g=r['original_collect_observation']
  require(all(type(g[k])is int and g[k]>=0 for k in ('calls','raised_calls','elapsed_ns','max_elapsed_ns','clock_errors'))
   and g['call_wall_durations_may_overlap_for_reentrant_or_concurrent_calls']is True,'Exact bounded original collect scalar schema')
  gcrows.append({'initial_updates':r['initial_updates'],'committed_updates':r['committed_updates'],**g})
 log,lp=snapshot(A/'native_000023.log',8*1024**2);required=set(range(start,end+1,200));points={};original_points={}
 for line in log.splitlines(keepends=True):
  if not line.startswith(b'{'):continue
  try:p=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if p.get('kind')=='ASR18_actual_resident_RAM_training_progress_v1'and p.get('actual_optimizer_updates')in required:
   u=p['actual_optimizer_updates'];require(u not in points,'Unique original emitted progress point');points[u]=p;original_points[u]=line
 require(set(points)==required,'ALL same-span original200 progress points plus endpoints captured')
 require(all(p['source']==source and p['producer']==boot['producer']for p in points.values()),'Every endpoint exact same producer/source')
 saves=[r for r in rows if r['checkpoint_included']];expected_saves=list(range((start//200+1)*200,end,200))
 require([r['committed_updates']for r in saves]==expected_saves,'ALL natural saves included, none selected or excluded')
 def interval(lo,hi):
  chosen=[r for r in rows if lo<=r['initial_updates']and r['committed_updates']<=hi];p,q=points[lo],points[hi]
  wall=q['active_pipeline_seconds']-p['active_pipeline_seconds'];audio=q['source_audio_seconds']-p['source_audio_seconds'];native=q['native_seconds']-p['native_seconds']
  save=math.fsum(r['checkpoint_seconds']for r in chosen if r['checkpoint_included'])
  require(len(chosen)==(hi-lo)//20 and wall>save>=0 and audio>0 and native>0,'Complete positive same-endpoint interval with labelled save components')
  collected=[r['original_collect_observation']for r in chosen]
  return {'start_updates':lo,'end_updates':hi,'updates':hi-lo,'endpoint_active_wall_seconds':wall,'source_audio_seconds':audio,'native_training_and_commit_seconds':native,
   'audio_minutes_per_wall_minute_pause_inclusive':audio/wall,'checkpoint_seconds_from_same_original_rows':save,
   'audio_minutes_per_wall_minute_save_time_excluded_derived':audio/(wall-save),
   'contains_scheduled_DEV_pause':lo<dev<hi,'source_original_timing_rows':len(chosen),
   'original_row_window_pipeline_seconds_sum_NOT_endpoint_wall':math.fsum(r['window_pipeline_seconds']for r in chosen),
   'native_commit_pressure_wait_seconds':math.fsum(r.get('native_commit_pressure_wait_seconds',0)for r in chosen),
   'native_commit_pressure_waited_updates':sum(r.get('native_commit_pressure_waited_updates',0)for r in chosen),
   'metadata_end_window_wait_count':sum(r.get('end_window_wait_count',0)for r in chosen),'serial_fallback_count':sum(r.get('serial_fallback_count',0)for r in chosen),
   'original_collect_observation':{'calls':sum(g['calls']for g in collected),'raised_calls':sum(g['raised_calls']for g in collected),
    'sum_call_wall_seconds_NOT_exclusive_or_CPU':sum(g['elapsed_ns']for g in collected)/1e9,
    'max_call_wall_seconds':max(g['max_elapsed_ns']for g in collected)/1e9,'clock_errors':sum(g['clock_errors']for g in collected),
    'scope':'All explicit original collect calls since prior20 emission, including intervening DEV/checkpoint calls. Concurrent/reentrant call wall may overlap; not pure CPU, exclusive native cost or lost-time attribution.'}}
 whole=interval(start,end);blocks=[interval(u,u+200)for u in range(start,end,200)];first=interval(start,start+400);last=interval(end-400,end)
 post=[interval(dev+220,dev+620),interval(dev+620,dev+1020)]
 decline=100*(1-last['audio_minutes_per_wall_minute_save_time_excluded_derived']/first['audio_minutes_per_wall_minute_save_time_excluded_derived'])
 cps=[checkpoint(u)for u in expected_saves];require(read(B/'latest_checkpoint.json')['committed_updates']>=end-20,'Natural final standard checkpoint and actual postpositive complete')
 periodic=read(B/'periodic_DEV.json');matched=[x for x in periodic['completed_evaluations']if x['optimizer_updates']==dev]
 require(len(matched)==1 and matched[0]['score']['full350_DEV_complete']is True,'Genuine scheduled full350 recorded in same actual interval')
 pe=matched[0];require(pin(pe['DEV']['path'])==pe['DEV']and pin(pe['frontier_preservation']['path'])==pe['frontier_preservation'],'Actual DEV and preservation compact pins')
 require(read(pe['DEV']['path'])['passed']is True,'Actual evaluation report passed completion, no quality-pass claim')
 ready_path=I/('CPU_after_periodic_DEV_'+str(dev).zfill(9))/'actual_CPU_ready.json';ready=read(ready_path);cpu=ready['producer_identity']
 current_child=read(A/'current_child.json');family=[('native',boot['producer']),('native_interpreter_launcher',current_child['launcher']),('wrapper',who(psutil.Process(current_child['launcher']['ppid']))),('CPU',cpu),('CPU_interpreter_launcher',who(psutil.Process(cpu['ppid'])))]
 require(all(live(w)for _,w in family),'Current five exact family members healthy')
 timing_bytes=b''.join(lines);point_bytes=b''.join(original_points[u]for u in sorted(original_points));final=guard()
 OUT.mkdir()
 with(OUT/'all_original_timing_rows.jsonl').open('xb')as f:f.write(timing_bytes)
 with(OUT/'all_original_200_progress_points.jsonl').open('xb')as f:f.write(point_bytes)
 report={'kind':'ASR18_source31_frozen_dynamic_durable_actual_job_observation_v1','measurement_completed':True,'fix_success_not_inferred_from_observer_alone':True,
  'observer_identity':who(psutil.Process()),'observer_launcher_identity':who(psutil.Process().parent()),'execution_review':pin(a.execution_review),'final_handoff':pin(HANDOFF),
  'predeclared_validation':v,'whole_pause_inclusive_interval':whole,'ALL_consecutive200_blocks':blocks,'first400':first,'final400':last,'two_predeclared_postDEV400_blocks':post,
  'save_excluded_first_to_final_decline_percent':decline,'comparison_criteria':{'original_checkpoint_inclusive_reference':76.569475,'near_original_engineering_floor':75.0,'maximum_sustained_save_excluded_decline_percent':5.0,
   'first_and_final_and_both_postDEV400_inclusive_near_reference':all(x['audio_minutes_per_wall_minute_pause_inclusive']>=75 for x in [first,last,*post]),
   'save_excluded_first_to_final_decline_within5percent':decline<=5,'scope':'Engineering comparisons across different original ordinals; not forecast or isolated causal proof. Partial gain/recurrent late decline keeps human fix open.'},
  'all_original_timing_rows':pin(OUT/'all_original_timing_rows.jsonl'),'all_original_200_progress_points':pin(OUT/'all_original_200_progress_points.jsonl'),
  'timing_source_snapshot':tp,'native_log_source_snapshot':lp,'all_gc_scalar_windows':gcrows,'natural_checkpoint_metadata':cps,
  'actual_scheduled_DEV':pe,'actual_normal_CPU_quiescence':pin(I/('CPU_periodic_'+str(dev).zfill(9)+'_quiescence.json')),
  'current_CPU_ready':pin(ready_path),'current_five_family':[{'role':role,**w,'exact_alive':True}for role,w in family],
  'current_progress':progress,'initial_guard':initial,'final_guard':final,'peak_private_bytes':PEAK,
  'no_metadata_archive_tensor_audio_journal_model_profiler_GC_or_GPU_operation':True,'no_production_source_environment_or_state_change':True}
 write(OUT/'report.json',report);print(json.dumps({'report':pin(OUT/'report.json'),'start':start,'end':end,'rows':len(rows),'all200_blocks':len(blocks),'pause_inclusive_rate':whole['audio_minutes_per_wall_minute_pause_inclusive'],'save_excluded_decline_percent':decline}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution-review',required=True);main(p.parse_args())
