"""Inactive ONE actual-job observer, not a benchmark or finite training stop.
Run only AFTER the final complete164620 timing row, standard164600 and postpositive.
Reads new-run bounded timing/log snapshots, compact records and ONLY the full
new163200/164600 metadata archives. No tensor/audio/journal/model/GPU operations.
"""
import argparse,gzip,hashlib,json,math,os,shutil,statistics,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000022';A=F.parent/'resident_RAM_training_actual_23'
START=163020;END=164620;BEGIN=time.monotonic();PEAK=0
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
def live(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def guard(admission=False):
 global PEAK
 v=sample_memory(os.getpid(),'actual_source30_long_production_validation_metadata_only');m=v['system_memory'];PEAK=max(PEAK,v['worker_private_memory']['private_bytes']);extra=536870912 if admission else 0
 require(v['paired_sampling_within_one_second']and PEAK<536870912 and time.monotonic()-BEGIN<180
  and m['commit_available_bytes']>=2268435456+extra and m['physical_available_bytes']>=1768435456+extra
  and shutil.disk_usage('E:/').free>=100000000000,'Original observer512MiB/180s/fullprospective/fullprefetch/floors/E100GB')
 return v
def bounded_snapshot(p,cap):
 p=Path(p);size=p.stat().st_size;offset=max(0,size-cap)
 with p.open('rb')as f:f.seek(offset);raw=f.read(size-offset)
 require(len(raw)==size-offset,'Complete captured append-only snapshot range')
 prefix=raw.find(b'\n')+1 if offset else 0;complete=raw[prefix:]
 if not complete.endswith(b'\n'):complete=complete[:complete.rfind(b'\n')+1]
 return complete,{'path':str(p.resolve()),'captured_offset':offset+prefix,'captured_end':size,'snapshot_bytes':len(complete),'sha256':hashlib.sha256(complete).hexdigest(),'no_whole_journal_read':True}
def summary(values):
 values=list(values);s=sorted(values);n=len(s);require(n>0,'Nonempty complete scalar coverage')
 return dict(count=n,sum=math.fsum(values),mean=statistics.mean(values),median=statistics.median(values),minimum=s[0],maximum=s[-1],p95_nearest_rank=s[math.ceil(.95*n)-1])
def checkpoint(update):
 ck=B/'checkpoints'/('update_'+str(update).zfill(9));pointer=ck/'pointer.json';persistence=ck/'persistence_verified.json';p=read(pointer);v=read(persistence)
 require(p['committed_updates']==update and v['passed']is True and v['optimizer_updates']==update
  and v['named_tensors_verified']==854 and v['weights_verified']==207 and v['Adam_states_verified']==215
  and v['independent_pre_save_state_sha256']==v['saved_state_sha256']and v['saved_semantic_state_verified']['passed']is True
  and v['journal_prefix']==p['journal_prefix'],'Actual natural full854/207/215 PRE-SAVED semantic checkpoint')
 return {'pointer':pin(pointer),'persistence':pin(persistence),'payload_not_read':True}
def archive_scalars(update):
 path=B/'checkpoints'/('update_'+str(update).zfill(9))/'prepared_metadata.json.gz';before=path.stat();require(before.st_size<=8*1024**2,'ONLY bounded compressed metadata archive8MiB')
 compressed=pin(path);chunks=[];length=0;sha=hashlib.sha256()
 with gzip.open(path,'rb')as f:
  while True:
   guard();chunk=f.read(min(1024**2,32*1024**2+1-length))
   if not chunk:break
   length+=len(chunk);require(length<=32*1024**2,'Hard32MiB decoded metadata cap');sha.update(chunk);chunks.append(chunk)
 blob=b''.join(chunks);del chunks;entries=json.loads(blob);del blob;guard();require(type(entries)is list and len(entries)==200,'ALL200 original full records, no sample selection')
 scalars=[]
 for index,entry in enumerate(entries):
  native=entry['native'];require(native['update']==update-199+index,'Entire ordered200 native coverage')
  groups=native['graph_boundary_memory_observations'];indices=[];seconds=[];counts=[]
  for gi,v in enumerate(groups):
   require(v['group']==gi+1 and type(v['collected_objects'])is int and v['collected_objects']>=0 and math.isfinite(v['seconds'])and v['seconds']>=0,'Original cleanup-boundary scalar schema')
   indices+=v['original_row_indices'];seconds.append(v['seconds']);counts.append(v['collected_objects'])
  require(sorted(indices)==list(range(len(native['source_ids'])))and len(indices)==len(set(indices)),'Every original native row covered once')
  scalars.append(dict(update=native['update'],cleanup_boundary_seconds=math.fsum(seconds),collected_objects=sum(counts),groups=len(groups),original_row_count=len(native['source_ids'])))
  if index%20==0:guard()
 after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Read-only archive unchanged')
 result={'compressed':compressed,'uncompressed_bytes':length,'uncompressed_sha256':sha.hexdigest(),'complete_gzip_CRC_verified':True,'all200_scalar_records':scalars,'cleanup_boundary_seconds_per_update':summary(v['cleanup_boundary_seconds']for v in scalars),
  'four_contiguous50_cleanup_summaries':[summary(v['cleanup_boundary_seconds']for v in scalars[a:a+50])for a in (0,50,100,150)],'collected_objects_per_update':summary(v['collected_objects']for v in scalars),
  'scope':'Cleanup boundary includes GC+CUDA sync+emptycache+snapshot/RNG/pointer/invariant checks; NOT pure GC time.'}
 del entries;guard();return result
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Separate E observer env, never production env')
 review=read(a.execution_review);require(review['accepted']is True and review['reviewer']=='/root'and review['long_validation_observer_source']==pin(__file__),'Exact final ROOT execution binding includes this inactive observer')
 handoff=read(F/'final_source30_actual_canonical_archive_transition_execution_handoff_v1.json');validation=handoff['predeclared_actual_production_validation'];require(validation['start_updates']==START and validation['end_updates']==END and validation['end_updates']-START==1600,'Frozen actual-restored long interval, not selected from performance')
 source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v30.py');require(review['driver_source']==source and review['source_configuration']==pin(F/'production_configuration_v1/configuration.json'),'Same exact admitted source30/config')
 out=F/'actual_source30_full_validation163020_164620_01';require(not out.exists(),'ONE fresh actual observation; no automatic rerun');initial=guard(True)
 progress=read(B/'progress/latest.json');boot=read(I/'boot.json');require(progress['source']==source and progress['producer']==boot['producer']and progress['actual_optimizer_updates']>=END and not progress['optimizer_poisoned']and not progress['schedule_poisoned']and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None and not(I/'actual_failure.json').exists(),'Same healthy full admitted job through complete frozen interval')
 raw,range_pin=bounded_snapshot(I/'private_metadata_window_timings.jsonl',2*1024**2);rows=[];original=[]
 for line in raw.splitlines(keepends=True):
  d=json.loads(line)
  if START<=d['initial_updates']and d['committed_updates']<=END:rows.append(d);original.append(line)
 require(len(rows)==80 and [(v['initial_updates'],v['committed_updates'])for v in rows]==[(u,u+20)for u in range(START,END,20)],'Every contiguous20 timing row, final completed append required')
 lograw,logpin=bounded_snapshot(A/'native_000022.log',4*1024**2);points={}
 for line in lograw.splitlines():
  if not line.startswith(b'{'):continue
  try:d=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if d.get('kind')=='ASR18_actual_resident_RAM_training_progress_v1'and d.get('actual_optimizer_updates')in(START,END):
   require(d['actual_optimizer_updates']not in points,'Unique actual endpoint');points[d['actual_optimizer_updates']]=dict(value=d,original_line=line.decode())
 require(set(points)=={START,END},'Both original emitted source30 progress endpoints captured')
 before=points[START]['value'];after=points[END]['value'];require(all(d['source']==source and d['producer']==boot['producer']for d in (before,after)),'Exact original endpoint producer/source')
 wall=after['active_pipeline_seconds']-before['active_pipeline_seconds'];audio=after['source_audio_seconds']-before['source_audio_seconds'];native=after['native_seconds']-before['native_seconds'];require(wall>0 and audio>0 and native>0,'Actual positive cumulative deltas')
 saves=[v for v in rows if v['checkpoint_included']];require([v['committed_updates']for v in saves]==list(range(163200,164601,200)),'ALL8 natural checkpoints included')
 totals={k:math.fsum(float(v[k])for v in rows)for k in ['window_pipeline_seconds','native_seconds','source_audio_seconds','checkpoint_seconds','worker_preparation_seconds','main_adoption_and_submission_seconds','main_take_wait_seconds','native_commit_pressure_wait_seconds']}
 require(abs(totals['native_seconds']-native)<.01 and abs(totals['source_audio_seconds']-audio)<.01,'Full timing native/audio deltas agree with original emitted endpoints')
 periodic=I/'CPU_periodic_000164000_quiescence.json';phase=read(periodic)
 # Actual DEV report and exact preservation paths are recorded by the driver; bind
 # the phase here without inventing a production DEV proof from CPU controls.
 require(phase['optimizer_updates']==164000 and phase['private_packets_discarded']==20 and phase['actual_native_updates']==0 and phase['authoritative_sampler_replay_or_journal_advance']is False and phase['CPU_actual0']['actual_Popen_wait_code']==0 and phase['CPU_actual0']['transport_threads_joined']is True and phase['CPU_actual0']['pipe_handles_closed']is True,'Actual production periodic CPU0/transport closure, no state advance')
 ready=I/'CPU_after_periodic_DEV_000164000/actual_CPU_ready.json';cpu=read(ready)['producer_identity'];child=read(A/'current_child.json');launcher=child['launcher'];wrapper=who(psutil.Process(launcher['ppid']));cpulauncher=who(psutil.Process(cpu['ppid']))
 family=[dict(role='native',**boot['producer']),dict(role='native_interpreter_launcher',**launcher),dict(role='wrapper',**wrapper),dict(role='CPU',**cpu),dict(role='CPU_interpreter_launcher',**cpulauncher)]
 require(all(live(v)for v in family),'All5 current exact runtime identities healthy')
 checkpoints={str(u):checkpoint(u)for u in (163200,164000,164600)};archives={str(u):archive_scalars(u)for u in (163200,164600)}
 report={'kind':'ASR18_source30_actual_long1600_through_DEV_canonical_archive_validation_v1','passed':True,'identity':who(psutil.Process()),'launcher_identity':who(psutil.Process().parent()),'source':pin(__file__),'execution_review':pin(a.execution_review),'predeclared_validation':validation,
  'original_endpoint_progress':points,'original_timing_snapshot_range':range_pin,'exact80_original_timing_rows':[x.decode().rstrip('\n')for x in original],'exact80_rows_bytes':sum(map(len,original)),'exact80_rows_sha256':hashlib.sha256(b''.join(original)).hexdigest(),'bounded_native_log_range':logpin,
  'actual_updates':END-START,'wall_seconds_checkpoint_and_DEV_inclusive':wall,'source_audio_seconds':audio,'native_training_and_commit_seconds':native,'audio_minutes_per_wall_minute_inclusive':audio/wall,'complete_original_window_totals':totals,
  'outside_native_seconds':wall-native,'outside_window_timer_seconds_including_DEV_and_other_interwindow_work':wall-totals['window_pipeline_seconds'],
  'derived_save_time_excluded_rate':audio/(wall-totals['checkpoint_seconds']),'derived_rate_still_includes_DEV_and_other_work':True,
  'all_natural_checkpoint_updates':[v['committed_updates']for v in saves],'actual_periodic_phase':pin(periodic),'post_DEV_CPU_ready':pin(ready),'natural_persistence':checkpoints,'new_first_and_post_DEV_end200_archive_cleanup_scalars':archives,
  'current_progress':progress,'current_family':family,'initial_resource_sample':initial,'last_resource_sample':guard(),'sampled_observer_private_peak_bytes':PEAK,
  'limits':['No exact80 or future fit/speed guarantee.','Boundary timing is cleanup composite,not pureGC.','No new numerical/GC/process/guard change.','All1600 updates,all8saves,realDEV included; no short-warm or selected-window claim.','Reader is left unchanged and independently checked by ROOT.']}
 out.mkdir();write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'actual_inclusive_rate':audio/wall,'updates':END-START,'all5_alive':True}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution-review',required=True);main(p.parse_args())
