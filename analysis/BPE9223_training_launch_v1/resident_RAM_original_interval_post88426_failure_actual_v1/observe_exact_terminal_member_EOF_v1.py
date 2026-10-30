"""One independent full-member EOF and exact original nonzero PTS interval observation."""
import hashlib,json,math,os,shutil,sqlite3,sys,time
from collections import deque
from fractions import Fraction
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import AV_TARGET,ROOT,Extent,BroadAudioReader,record,require,resample
from fastmss_raw_lifetime_v1 import chunked_hashes
from resource_watchdog_v2 import sample_memory
from fastmss_BPE9223_cpu_common_v12 import numpy_runtime
N=Path(__file__).parent;OUT=N/'full_member_actual_01';require(not OUT.exists(),'Fresh exact CPU observation');OUT.mkdir()
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and sys.flags.utf8_mode==1,'CPU masked/upfrontUTF8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Threadcaps1')
process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);identity={'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()};began=time.monotonic();peak=0;calls=0;last=0.
def guard():
 global peak,calls,last
 require(time.monotonic()-began<180,'Finite180s')
 if time.monotonic()-last<.25:return
 last=time.monotonic();calls+=1;s=sample_memory(os.getpid(),'one_exact_terminal_original_member_EOF');private=s['worker_private_memory']['private_bytes'];peak=max(peak,private)
 require(private<=512*1024**2 and s['paired_sampling_within_one_second'],'Actual512MiB/paired sample')
 require(s['system_memory']['commit_available_bytes']>=2000000000+256*1024**2 and s['system_memory']['physical_available_bytes']>=1500000000+256*1024**2,'Real headroom/full256MiB extra')
 require(shutil.disk_usage('E:/').free>=100000000000,'E100GB reserve')
try:
 numpy_runtime(frontend=False)
 import numpy as np
 sys.path.insert(0,str(AV_TARGET));import av
 require(av.__version__=='16.0.1','Pinned PyAV')
 evidence=read_json(N/'probe_actual_01/actual_failed_update_metadata.json');item=evidence['rows'][13];row=item['broad_original'];decl=item['declared_geometry']
 require(row['unique_id']=='alania:e4a162635e1cf1f6599a40b38b8b7a57_00004' and row['start']==45.648 and row['end']==53.151 and decl['native_samples']==360144 and decl['target_tokens']==39,'Exact failing source/interval/declaration')
 reader=BroadAudioReader(guard=guard);path,job=reader.verify_source(ROOT/'datasets/yodas3/audio/0029.tar');index=ROOT/'indexes/yodas3_tar_0029.sqlite';db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
 try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
 finally:db.close()
 name,offset,size=entry;require(name==row['source_id']+'.webm' and size<=32*1024**2,'Exact bounded original member')
 sha=hashlib.sha256()
 with path.open('rb')as f:
  f.seek(offset);left=size
  while left:
   guard();part=f.read(min(1024*1024,left));require(part,'Full exact member bytes');sha.update(part);left-=len(part)
 begin=round(Fraction(str(row['start']))*48000);finish=round(Fraction(str(row['end']))*48000)
 cursor=written=frames=packets=0;first_frames=[];last_frames=deque(maxlen=5);first_packets=[];last_packets=deque(maxlen=4);side=[];max_residual=Fraction(0);physical_interval_begin=None;physical_interval_end=None
 with Extent(path,offset,size,guard)as ext,av.open(ext,mode='r',format='webm')as c:
  require(len(c.streams.audio)==1,'One original stream');stream=c.streams.audio[0];stream.codec_context.thread_count=1;require(stream.codec_context.name=='opus' and stream.time_base==Fraction(1,1000),'Exact Opus/ms clock')
  duration=c.duration;require(duration is not None and 0<duration<=60000000,'Bounded exact source under60s, observation only')
  bound=math.ceil(Fraction(duration,1000000)*48000);require(process.memory_info().private+(bound+decl['native_samples'])*4+32*1024**2<512*1024**2,'Full member plus interval current-private allocation admission')
  pcm=np.empty(bound,dtype=np.float32);selected=np.empty(decl['native_samples'],dtype=np.float32);origin=None
  for packet in c.demux(stream):
   guard();packets+=1;p={'pts':packet.pts,'dts':packet.dts,'duration':packet.duration,'bytes':packet.size,'time_base':str(packet.time_base)}
   for sd in packet.iter_sidedata():side.append({'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc)})
   if len(first_packets)<3:first_packets.append(p)
   last_packets.append(p)
   for f in packet.decode():
    guard();require(f.pts is not None and f.sample_rate==48000 and f.format.is_planar and 0<f.samples<=16384 and len(f.layout.channels)<=8,'Full original frame envelope')
    if origin is None:origin=f.pts*f.time_base
    frame_begin=round((f.pts*f.time_base-origin)*48000);frame_end=frame_begin+f.samples;residual=(f.pts*f.time_base-origin)*48000-cursor;max_residual=max(max_residual,abs(residual));require(abs(residual)<=24,'Exact half-ms clock, no true gap')
    v={'pts':f.pts,'time_base':str(f.time_base),'samples':f.samples,'cumulative_source_sample_begin':cursor,'PTS_frame_begin':frame_begin,'PTS_frame_end':frame_end,'PTS_residual_samples':str(residual)}
    if len(first_frames)<3:first_frames.append(v)
    last_frames.append(v);frames+=1;raw=f.to_ndarray();require(raw.shape==(len(f.layout.channels),f.samples),'Full planar channels')
    if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
    else:raw=raw.astype(np.float32,copy=False)
    mono=raw.mean(axis=0,dtype=np.float32)if len(f.layout.channels)>1 else raw[0];require(cursor+len(mono)<=len(pcm),'Full source no truncation');pcm[cursor:cursor+len(mono)]=mono
    low,high=max(begin,frame_begin),min(finish,frame_end)
    if high>low:
     require(low-begin==written,'Original PTS-selected interval adjacency, no filler')
     source_low=cursor+low-frame_begin;source_high=cursor+high-frame_begin
     if physical_interval_begin is None:physical_interval_begin=source_low
     else:require(source_low==physical_interval_end,'Every selected original sample contiguous and once')
     physical_interval_end=source_high;selected[low-begin:high-begin]=mono[low-frame_begin:high-frame_begin];written=high-begin
    cursor+=len(mono);del raw,mono
  logical=ext.logical_bytes_read
 require(written==360143 and frame_end==2551247,'Independent exact original PTS EOF agrees failed seek; one absent declared sample')
 require(physical_interval_end==cursor and written==physical_interval_end-physical_interval_begin,'Actual selected interval reaches true source EOF, no dropped remainder')
 actual=selected[:written];linear=pcm[physical_interval_begin:physical_interval_end];require(np.array_equal(actual,linear),'Exact original PTS-selected samples equal independent full-source cumulative range')
 native_sha=hashlib.sha256(memoryview(actual).cast('B')).hexdigest();full_sha=hashlib.sha256(memoryview(pcm[:cursor]).cast('B')).hexdigest();clean=resample(actual,48000);f32,i16=chunked_hashes(clean,guard=guard)
 report={'kind':'ASR18_one_exact_nonzero_terminal_WebM_interval_EOF_observation_v1','source':record(__file__),'producer':identity,'failed_update':record(N/'probe_actual_01/actual_failed_update_metadata.json'),'prior_probe':record(N/'probe_actual_01/report.json'),'row':row,'declared_geometry':decl,'original_TAR':{'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'recorded_sha256':job['sha256']},'recording_index_entry':dict(zip(('member_name','offset','bytes'),entry)),'member_sha256':sha.hexdigest(),'container_duration_microseconds':duration,'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,'first_frames':first_frames,'last_frames':list(last_frames),'full_source_native_samples':cursor,'full_source_native_f32_sha256':full_sha,'decoded_frames':frames,'demuxed_packets_including_flush':packets,'max_abs_PTS_residual_samples':str(max_residual),'original_PTS_interval_begin':begin,'original_PTS_interval_declared_finish':finish,'actual_PTS_interval_finish':frame_end,'selected_physical_source_range':[physical_interval_begin,physical_interval_end],'full_source_range_equals_original_PTS_selected_PCM':True,'native_interval_samples':written,'native_interval_f32_sha256':native_sha,'clean16k_samples':len(clean),'clean_f32_sha256':f32,'clean_i16_sha256':i16,'actual_interval_geometry':{'native_sr':48000,'native_samples':written,'clean16k_samples':len(clean),'mel_frames':len(clean)//160,'encoder_frames':(len(clean)//160+7)//8,'target_tokens':decl['target_tokens']},'full_original_frames_observed_once':True,'selected_interval_original_samples_used_once':True,'synthesized_samples':0,'selected_interval_skipped_samples':0,'annotation_or_source_index_plan_sampler_journal_mutated':False,'no_model_or_GPU_import':'torch'not in sys.modules,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'guard_samples':calls,'member_logical_read_bytes':logical}
 require(report['no_model_or_GPU_import'],'No GPU/model');write_json(OUT/'report.json',report);print(json.dumps({'report':record(OUT/'report.json'),'actual_geometry':report['actual_interval_geometry'],'full_source_native_samples':cursor,'physical_range':report['selected_physical_source_range'],'side':side,'last_frames':report['last_frames']}),flush=True)
finally:write_json(OUT/'CPU_completion.json',{'producer':identity,'guard_samples':calls,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,'actual_exit_unobserved_inside_child':True})