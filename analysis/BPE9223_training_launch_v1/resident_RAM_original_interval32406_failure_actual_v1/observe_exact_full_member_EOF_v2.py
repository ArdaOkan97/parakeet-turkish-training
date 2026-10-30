"""One exact original WebM member: independent full linear EOF/sample proof."""
import gc,hashlib,json,math,os,shutil,sqlite3,sys,time
from collections import deque
from fractions import Fraction
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import AV_TARGET,ROOT,Extent,BroadAudioReader,record,require,resample
from fastmss_raw_lifetime_v1 import raw_plan,chunked_hashes
from resource_watchdog_v2 import sample_memory
from fastmss_BPE9223_cpu_common_v12 import numpy_runtime
N=Path(__file__).parent;OUT=N/'full_member_actual_02';require(not OUT.exists(),'Fresh E observation');OUT.mkdir()
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and sys.flags.utf8_mode==1,'CPU masked/upfront UTF8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Four thread caps1')
process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);identity={'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
began=time.monotonic();peak=0;calls=0;last=0

def guard():
 global peak,calls,last
 require(time.monotonic()-began<180,'Finite180s')
 if time.monotonic()-last<.25:return
 last=time.monotonic();calls+=1;s=sample_memory(os.getpid(),'one_exact_original_member_EOF')
 private=s['worker_private_memory']['private_bytes'];peak=max(peak,private)
 require(private<=512*1024**2,'Actual512MiB private')
 require(s['paired_sampling_within_one_second'] and s['system_memory']['commit_available_bytes']>=2000000000+256*1024**2 and s['system_memory']['physical_available_bytes']>=1500000000+256*1024**2,'Same real headroom/full256MiB extra')
 require(shutil.disk_usage('E:/').free>=100000000000,'E100GB reserve')

try:
 numpy_runtime(frontend=False)
 import numpy as np
 sys.path.insert(0,str(AV_TARGET));import av
 require(av.__version__=='16.0.1','Pinned PyAV')
 evidence=read_json(N/'actual_failed_update32406_plan391_index5.json');item=evidence['rows'][10];row=item['broad_original'];decl=item['declared_geometry']
 require(row['unique_id']=='alania:6619bf48d0238fefe91cea29f78c5287_00000' and row['start']==0 and decl['native_samples']==743424,'Exact failing row/declaration')
 reader=BroadAudioReader(guard=guard);path,job=reader.verify_source(ROOT/'datasets/yodas3/audio/0026.tar')
 index=ROOT/'indexes/yodas3_tar_0026.sqlite';db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
 try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
 finally:db.close()
 name,offset,size=entry;require(name==row['source_id']+'.webm' and size<=32*1024**2,'Exact bounded original extent')
 sha=hashlib.sha256()
 with path.open('rb') as f:
  f.seek(offset);left=size
  while left:
   guard();part=f.read(min(1024*1024,left));require(part,'Full exact member bytes');sha.update(part);left-=len(part)
 rp=raw_plan(native_samples=decl['native_samples'],native_rate=48000,clean_samples=decl['clean16k_samples'],phase=evidence['phase'],role=item['augmentation_role'],channels_bound=8,frame_samples_bound=16384)
 require(process.memory_info().private+rp['raw_peak_bytes']+rp['compressed_media_cap_bytes']<512*1024**2,'Same prospective raw/media envelope')
 pcm=np.empty(decl['native_samples'],dtype=np.float32);cursor=0;frames=0;packets=0;first_frames=[];last_frames=deque(maxlen=4);first_packets=[];last_packets=deque(maxlen=4);side=[];max_residual=Fraction(0)
 with Extent(path,offset,size,guard) as ext,av.open(ext,mode='r',format='webm') as c:
  require(len(c.streams.audio)==1,'One stream');stream=c.streams.audio[0];stream.codec_context.thread_count=1
  require(stream.codec_context.name=='opus' and stream.time_base==Fraction(1,1000),'Exact Opus/ms clock')
  duration=c.duration;origin=None
  for packet in c.demux(stream):
   guard();packets+=1;p={'pts':packet.pts,'dts':packet.dts,'duration':packet.duration,'bytes':packet.size,'time_base':str(packet.time_base)}
   for sd in packet.iter_sidedata():
    value={'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc)};side.append(value)
   if len(first_packets)<3:first_packets.append(p)
   last_packets.append(p)
   for f in packet.decode():
    require(f.pts is not None and f.sample_rate==48000 and f.format.is_planar and 0<f.samples<=16384,'Full native frame envelope')
    if origin is None:origin=f.pts*f.time_base
    residual=(f.pts*f.time_base-origin)*48000-cursor;max_residual=max(max_residual,abs(residual))
    require(abs(residual)<=24,'Exact half-ms clock, no true gap')
    v={'pts':f.pts,'time_base':str(f.time_base),'samples':f.samples,'cursor':cursor,'PTS_residual_samples':str(residual)}
    if len(first_frames)<3:first_frames.append(v)
    last_frames.append(v);frames+=1
    raw=f.to_ndarray();require(raw.shape==(len(f.layout.channels),f.samples) and len(f.layout.channels)<=8,'Original planar full channels')
    if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
    else:raw=raw.astype(np.float32,copy=False)
    mono=raw.mean(axis=0,dtype=np.float32) if len(f.layout.channels)>1 else raw[0]
    require(cursor+len(mono)<=len(pcm),'No clipping beyond declared native envelope')
    pcm[cursor:cursor+len(mono)]=mono;cursor+=len(mono);del raw,mono
  logical=ext.logical_bytes_read
 require(cursor==743411 and frames==775,'Independent exact native EOF observed')
 actual=pcm[:cursor];native_sha=hashlib.sha256(memoryview(actual).cast('B')).hexdigest();clean=resample(actual,48000)
 f32,i16=chunked_hashes(clean,guard=guard)
 report={'kind':'ASR18_one_exact_original_full_WebM_EOF_observation_v1','source':record(__file__),'producer':identity,'failed_update':record(N/'actual_failed_update32406_plan391_index5.json'),'prior_probe':record(N/'probe_actual_02/report.json'),'row':row,'declared_geometry':decl,'original_TAR':{'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'recorded_sha256':job['sha256']},'recording_index_entry':dict(zip(('member_name','offset','bytes'),entry)),'member_sha256':sha.hexdigest(),'container_duration_microseconds':duration,'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,'first_frames':first_frames,'last_frames':list(last_frames),'full_native_samples':cursor,'decoded_frames':frames,'demuxed_packets_including_flush':packets,'actual_full_native_seconds':str(Fraction(cursor,48000)),'declared_end_minus_actual_seconds':str(Fraction(str(row['end']))-Fraction(cursor,48000)),'max_abs_PTS_residual_samples':str(max_residual),'native_f32_sha256':native_sha,'clean16k_samples':len(clean),'clean_f32_sha256':f32,'clean_i16_sha256':i16,'actual_geometry':{'native_sr':48000,'native_samples':cursor,'clean16k_samples':len(clean),'mel_frames':len(clean)//160,'encoder_frames':(len(clean)//160+7)//8,'target_tokens':decl['target_tokens']},'full_original_frames_used_once':True,'synthesized_or_skipped_samples':0,'no_source_index_plan_sampler_or_journal_mutation':True,'no_model_or_GPU_import': 'torch' not in sys.modules,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'guard_samples':calls,'member_logical_read_bytes':logical}
 require(report['no_model_or_GPU_import'],'No model/GPU')
 write_json(OUT/'report.json',report);print(json.dumps({'report':record(OUT/'report.json'),'actual_geometry':report['actual_geometry'],'side':side,'last_frames':report['last_frames']}),flush=True)
finally:write_json(OUT/'CPU_completion.json',{'producer':identity,'guard_samples':calls,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,'actual_exit_unobserved_inside_child':True})
