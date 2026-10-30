"""Finite CPU-only exact original decoder coverage/PTS observation, no remedy."""
import gc, hashlib, inspect, json, os, shutil, sys, time
from collections import deque
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import BroadAudioReader as OriginalReader,record,require
from fastmss_bounded_original_audio_v2 import BoundedOriginalReader
from fastmss_raw_lifetime_v1 import RawLifetimeLedger,raw_plan,chunked_hashes
from resource_watchdog_v2 import sample_memory
N=Path(__file__).parent;OUT=N/'probe_actual_02';require(not OUT.exists(),'Fresh E CPU observation');OUT.mkdir()
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and sys.flags.utf8_mode==1,'CPU masked/upfront UTF8')
for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Four thread caps1')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic();peak=0;last=0.;samples=0
def guard():
 global peak,last,samples
 require(time.monotonic()-began<180,'Finite180s CPU coverage observation')
 if time.monotonic()-last>=.25:
  sample=sample_memory(os.getpid(),'original_interval_CPU_observation');last=time.monotonic();samples+=1
  private=sample['worker_private_memory']['private_bytes'];peak=max(peak,private)
  require(private<=512*1024**2 and sample['paired_sampling_within_one_second'],'Actual512MiB CPU process cap')
  require(sample['system_memory']['commit_available_bytes']>=2000000000+256*1024**2 and sample['system_memory']['physical_available_bytes']>=1500000000+256*1024**2,'Real system headroom/full256MiB extra reservation')
  require(shutil.disk_usage('E:/').free>=100000000000,'Existing E100GB reserve')
def admit(name,increment):
 guard();require(psutil.Process().memory_info().private+increment<=512*1024**2,'Actual current plus declared decoder increment fits512MiB')
def observe(reader,row,*,original=False):
 code=reader.yodas.__func__.__code__;source,start=inspect.getsourcelines(reader.yodas.__func__)
 mark=start+next(i for i,line in enumerate(source)if 'if frame_end<=begin:continue' in line)
 final=start+next(i for i,line in enumerate(source)if "require(written==count," in line)
 frames=deque(maxlen=8);first=[];summary={};function=reader.yodas
 def trace(frame,event,arg):
  if frame.f_code is not code:return None
  if event=='line' and frame.f_lineno==mark:
   v=frame.f_locals;f=v['frame'];item={'decoded':v['decoded'],'PTS':str(f.pts*f.time_base),'PTS_ticks':f.pts,'time_base':str(f.time_base),'samples':f.samples,'frame_begin':v['frame_begin'],'frame_end':v['frame_end'],'written_before':v['written']}
   if len(first)<5:first.append(item)
   frames.append(item)
  if event=='line' and frame.f_lineno==final:
   v=frame.f_locals;summary.update(written=v['written'],count=v['count'],decoded=v['decoded'],origin=str(v['origin']),begin=v['begin'],finish=v['finish'],sample_rate=v['sr'],coverage_incomplete=v['written']!=v['count'],container_duration_microseconds=v['container'].duration,stream_duration_ticks=v['stream'].duration,stream_time_base=str(v['stream'].time_base),codec=v['stream'].codec_context.name)
  return trace
 sys.settrace(trace)
 try:
  wave,info=function(row)
  f32,i16=chunked_hashes(wave,guard=guard);summary.update(passed=True,PCM_f32_sha256=f32,PCM_i16_sha256=i16,clean16k_samples=len(wave),info=info)
  del wave
 except ValueError as error:summary.update(passed=False,error=repr(error))
 finally:sys.settrace(None)
 summary.update(first_frames=first,last_frames=list(frames),frozen_reader_source=record(inspect.getfile(function.__func__)),original_baseline=original)
 return summary
from fastmss_BPE9223_cpu_common_v12 import numpy_runtime
numpy_runtime(frontend=False)
update=read_json(N/'actual_failed_update32406_plan391_index5.json');results=[];failed=None
try:
 for index,row in enumerate(update['rows']):
  if row['dataset']!='alania_yodas3':continue
  guard();raw=row['broad_original'];d=row['declared_geometry'];ledger=RawLifetimeLedger(guard=guard)
  plan=raw_plan(native_samples=d['native_samples'],native_rate=d['native_sr'],clean_samples=d['clean16k_samples'],phase=update['phase'],role=row['augmentation_role'],channels_bound=8,frame_samples_bound=16384)
  ledger.begin(str(index),plan);reader=BoundedOriginalReader(ledger=ledger,diagnostic_only=True,guard=guard,admit=admit)
  reader.active={'key':str(index),'declaration':d,'plan':plan}
  with ledger.stage(str(index),'decode_resample'):actual=observe(reader,raw)
  item={'row_index':index,'row':raw,'declared_geometry':d,'actual_decoder2':actual}
  results.append(item);write_json(OUT/('row_'+str(index)+'.json'),item)
  reader.active=None;del reader;gc.collect()
  if not actual['passed']:
   failed=index;baseline=OriginalReader(guard=guard,admit=admit)
   item['frozen_original_baseline']=observe(baseline,raw,original=True);write_json(OUT/('row_'+str(index)+'_with_baseline.json'),item)
   del baseline;break
 require('torch' not in sys.modules,'No model/GPU import')
 report={'kind':'ASR18_actual_original_interval32406_CPU_coverage_probe_v1','failure_seal':record(N/'actual_failure_closure.json'),'failed_update':record(N/'actual_failed_update32406_plan391_index5.json'),'results':results,'first_actual_failing_row_index':failed,'CPU_elapsed_seconds':time.monotonic()-began,'CPU_peak_private_bytes':peak,'guard_samples':samples,'no_model_GPU_audio_synthesis_or_source_change':True,'no_sampler_RNG_or_journal_advance':True,'diagnostic_only_no_production_admission':True}
 write_json(OUT/'report.json',report);print(json.dumps({'report':record(OUT/'report.json'),'failed_row_index':failed,'rows':[{'index':x['row_index'],'id':x['row']['unique_id'],'actual':{k:v for k,v in x['actual_decoder2'].items()if k not in('first_frames','last_frames','info')},'baseline':{k:v for k,v in x.get('frozen_original_baseline',{}).items()if k not in('first_frames','last_frames','info')}}for x in results]}),flush=True)
finally:write_json(OUT/'CPU_completion.json',{'actual_process_return_still_unobserved_inside_child':True,'elapsed_seconds':time.monotonic()-began,'guard_samples':samples,'CPU_peak_private_bytes':peak})
