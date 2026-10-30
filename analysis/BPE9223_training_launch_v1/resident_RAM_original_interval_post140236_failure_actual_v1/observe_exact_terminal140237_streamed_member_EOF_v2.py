"""One independent STREAMED full-member EOF and exact original nonzero PTS interval observation."""
import ctypes,hashlib,json,math,os,shutil,sqlite3,struct,sys,time
from ctypes import wintypes
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
N=Path(__file__).parent;OUT=N/'full_member_actual_02';require(not OUT.exists(),'Fresh exact CPU observation');OUT.mkdir()
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
packet_side_read_observations=[]
class PacketSideDataLayout(ctypes.Structure):
 _fields_=[('ob_refcnt',ctypes.c_ssize_t),('ob_type',ctypes.c_void_p),('data',ctypes.c_void_p),('size',ctypes.c_size_t),('dtype',ctypes.c_int)]
class MemoryRegion(ctypes.Structure):
 _fields_=[('BaseAddress',ctypes.c_void_p),('AllocationBase',ctypes.c_void_p),('AllocationProtect',wintypes.DWORD),('RegionSize',ctypes.c_size_t),('State',wintypes.DWORD),('Protect',wintypes.DWORD),('Type',wintypes.DWORD)]
def packet_side_bytes(sd):
 # Installed PyAV16 packet.pxd declares exactly data/size/dtype, without Buffer.
 # Read only this live, strongly held owned object; never alter its fields/data.
 require(type(sd)is av.packet.PacketSideData and ctypes.sizeof(PacketSideDataLayout)==type(sd).__basicsize__==40,'Exact source-pinned PyAV16 64-bit object layout')
 obj=PacketSideDataLayout.from_address(id(sd));size=sd.data_size
 require(obj.ob_refcnt>0 and obj.ob_type==id(type(sd)) and obj.size==size and obj.dtype==av.packet.packet_sidedata_type_from_literal(sd.data_type) and 0<size<=4096 and obj.data,'Object header/public size/type and bounded owned data pointer')
 require(ctypes.sizeof(MemoryRegion)==48,'Exact Windows64 MEMORY_BASIC_INFORMATION layout')
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);query=kernel.VirtualQuery;query.argtypes=[ctypes.c_void_p,ctypes.POINTER(MemoryRegion),ctypes.c_size_t];query.restype=ctypes.c_size_t
 region=MemoryRegion();observed=query(ctypes.c_void_p(obj.data),ctypes.byref(region),ctypes.sizeof(region));base=region.BaseAddress or 0
 require(observed==ctypes.sizeof(region) and region.State==0x1000 and not(region.Protect&0x100) and (region.Protect&0xff)in(2,4,8,0x20,0x40,0x80) and base<=obj.data<obj.data+size<=base+region.RegionSize,'VirtualQuery committed readable single region covers bounded owned side bytes')
 payload=ctypes.string_at(obj.data,size);require(len(payload)==size,'Exact bounded owned side bytes copied once')
 packet_side_read_observations.append({'type':sd.data_type,'bytes':size,'source_pinned_layout_matches_public_header_size_type':True,'VirtualQuery_committed_readable_bounds_checked':True,'no_memory_write':True});return payload
try:
 numpy_runtime(frontend=False)
 import numpy as np
 sys.path.insert(0,str(AV_TARGET));import av
 require(av.__version__=='16.0.1','Pinned PyAV')
 packet_api_provenance={'pinned_pxd': {'path': 'E:\\gpu_training_scratchpad\\scratch\\ASR18_audio_av_16_0_1\\av\\packet.pxd', 'bytes': 562, 'sha256': 'b9b437bfa3382ba37d4e5b6bdefc5e5ce5c81131af9b3fbdfca01db0c40c0082'}, 'pinned_Cython_python_source': {'path': 'E:\\gpu_training_scratchpad\\scratch\\ASR18_audio_av_16_0_1\\av\\packet.py', 'bytes': 13303, 'sha256': 'f7d474fc9ab1fc678274906d12869f0b9eea9afcef3a5f833f871dd1e9b73273'}, 'pinned_extension': {'path': 'E:\\gpu_training_scratchpad\\scratch\\ASR18_audio_av_16_0_1\\av\\packet.cp311-win_amd64.pyd', 'bytes': 90112, 'sha256': '383bdb82cba3a859d2fccaaf7ff6d4e219c9dd5908b2e66039c5351b13a3306d'}, 'exact_private_layout_source': 'av/packet.pxd: data(uint8_t*), size(size_t), dtype(AVPacketSideDataType)', 'raw_bytes_read_method': 'ctypes.string_at on strongly held owned PacketSideData.data ONLY after public fields + VirtualQuery bounds checks', 'max_copy_bytes': 4096, 'public_buffer_API_unavailable_in_installed_PyAV16': True}
 require(os.name=='nt' and sys.version_info[:3]==(3,11,13) and ctypes.sizeof(ctypes.c_void_p)==8 and av.packet.PacketSideData.__basicsize__==40,'Exact inspected Windows64 CPython3.11.13 PyAV16 ABI')
 for name in ('pinned_pxd','pinned_Cython_python_source','pinned_extension'):
  authority=packet_api_provenance[name];require(record(authority['path'])==authority,'Exact installed PyAV layout/API provenance unchanged')
 failed_update_pin={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_original_interval_post140236_failure_actual_v1\\probe_actual_01\\actual_failed_update_metadata.json', 'bytes': 72869, 'sha256': '199602c8ffcd19dc9613d84b2eb825775342d83326228190c2a9ae1f104af7d0'}
 prior_probe_pin={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_original_interval_post140236_failure_actual_v1\\probe_actual_01\\report.json', 'bytes': 37995, 'sha256': 'b0d03569cab5b93b15ef4271e5be5b628d78f33bab539d857150cb9b39ce5123'}
 require(record(failed_update_pin['path'])==failed_update_pin and record(prior_probe_pin['path'])==prior_probe_pin,'Exact only failed update and causal report bytes')
 evidence=read_json(N/'probe_actual_01/actual_failed_update_metadata.json');item=evidence['rows'][4];row=item['broad_original'];decl=item['declared_geometry']
 require(row['unique_id']=='alania:38198deb36bb10f9e58129b2c9d100fe_00003' and row['start']==1454.432 and row['end']==1473.713 and decl['native_samples']==925488 and decl['target_tokens']==121,'Exact failing source/interval/declaration')
 reader=BroadAudioReader(guard=guard);path,job=reader.verify_source(ROOT/'datasets/yodas3/audio/0016.tar');index=ROOT/'indexes/yodas3_tar_0016.sqlite';db=sqlite3.connect(index.as_uri()+'?mode=ro&immutable=1',uri=True)
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
  duration=c.duration;require(duration==1473721000,'Exact previously observed container duration; streamed source only')
  bound=math.ceil(Fraction(duration,1000000)*48000);require(0<decl['native_samples']<=1048576 and process.memory_info().private+2*decl['native_samples']*4+32*1024**2<512*1024**2,'Two bounded interval-only arrays plus decoder/resample scratch current-private admission')
  selected=np.empty(decl['native_samples'],dtype=np.float32);linear_selected=np.empty(decl['native_samples'],dtype=np.float32);streaming_full_hash=hashlib.sha256();linear_begin=None;linear_written=0;origin=None
  for packet in c.demux(stream):
   guard();packets+=1;p={'pts':packet.pts,'dts':packet.dts,'duration':packet.duration,'bytes':packet.size,'time_base':str(packet.time_base)}
   for sd in packet.iter_sidedata():
    require(len(side)<16 and sd.data_size<=4096,'Bounded actual codec side metadata');payload=packet_side_bytes(sd);require(len(payload)==sd.data_size,'Exact actual packet side-data bytes')
    observed_side={'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc),'raw_hex':payload.hex()}
    if str(sd.data_desc)=='Skip Samples' and len(payload)==10:
     skip,discard,start_reason,end_reason=struct.unpack('<IIBB',payload);observed_side.update(skip_start_samples=skip,discard_end_samples=discard,skip_reason=start_reason,discard_reason=end_reason)
    side.append(observed_side)
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
    mono=raw.mean(axis=0,dtype=np.float32)if len(f.layout.channels)>1 else raw[0];require(cursor+len(mono)<=bound,'Streamed full source stays within exact container duration bound');streaming_full_hash.update(memoryview(mono).cast('B'))
    if linear_begin is None and frame_begin<=begin<frame_end:linear_begin=cursor+begin-frame_begin
    if linear_begin is not None:
     linear_low=max(linear_begin,cursor);linear_high=min(linear_begin+decl['native_samples'],cursor+len(mono))
     if linear_high>linear_low:
      require(linear_low-linear_begin==linear_written,'Independent cumulative source-coordinate adjacency');linear_selected[linear_low-linear_begin:linear_high-linear_begin]=mono[linear_low-cursor:linear_high-cursor];linear_written=linear_high-linear_begin
    low,high=max(begin,frame_begin),min(finish,frame_end)
    if high>low:
     require(low-begin==written,'Original PTS-selected interval adjacency, no filler')
     source_low=cursor+low-frame_begin;source_high=cursor+high-frame_begin
     if physical_interval_begin is None:physical_interval_begin=source_low
     else:require(source_low==physical_interval_end,'Every selected original sample contiguous and once')
     physical_interval_end=source_high;selected[low-begin:high-begin]=mono[low-frame_begin:high-frame_begin];written=high-begin
    cursor+=len(mono);del raw,mono
  logical=ext.logical_bytes_read
 require(written==925487 and frame_end==70738223,'Exact independently observed PTS EOF agrees failed seek; cumulative physical coordinates retained separately')
 require(physical_interval_end==cursor and written==physical_interval_end-physical_interval_begin,'Actual selected interval reaches true source EOF, no dropped remainder')
 require(linear_begin==physical_interval_begin and linear_written==written,'Exact cumulative-selected count and physical begin');actual=selected[:written];linear=linear_selected[:linear_written];require(np.array_equal(actual,linear),'Exact original PTS-selected samples equal independently populated cumulative-coordinate interval')
 labels=item['tr_metadata_labels'];require(len(labels)==121 and all(type(t)is int and 0<=t<9223 for t in labels),'Exact all121 original migrated labels, no retokenization');label_sha=hashlib.sha256(b''.join(struct.pack('<I',t)for t in labels)).hexdigest()
 require(record(failed_update_pin['path'])==failed_update_pin and record(prior_probe_pin['path'])==prior_probe_pin,'Original annotation/labels/source evidence unchanged after one decode')
 native_sha=hashlib.sha256(memoryview(actual).cast('B')).hexdigest();full_sha=streaming_full_hash.hexdigest();clean=resample(actual,48000);f32,i16=chunked_hashes(clean,guard=guard)
 report={'kind':'ASR18_one_exact_nonzero_terminal_WebM_interval_EOF_observation_v1','source':record(__file__),'producer':identity,'failed_update':record(N/'probe_actual_01/actual_failed_update_metadata.json'),'prior_probe':record(N/'probe_actual_01/report.json'),'global_update':140237,'update_index':16,'row_index':4,'row':row,'declared_geometry':decl,'declared_frontend_geometry':item['declared_frontend_geometry'],'full_target_ids':labels,'full_target_ids_sha256':label_sha,'original_TAR':{'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'recorded_sha256':job['sha256']},'recording_index_entry':dict(zip(('member_name','offset','bytes'),entry)),'member_sha256':sha.hexdigest(),'container_duration_microseconds':duration,'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,'packet_side_data_API_provenance':packet_api_provenance,'packet_side_data_bounded_actual_read_controls':packet_side_read_observations,'first_frames':first_frames,'last_frames':list(last_frames),'full_source_native_samples':cursor,'full_source_native_f32_sha256':full_sha,'decoded_frames':frames,'demuxed_packets_including_flush':packets,'max_abs_PTS_residual_samples':str(max_residual),'original_PTS_interval_begin':begin,'original_PTS_interval_declared_finish':finish,'actual_PTS_interval_finish':frame_end,'selected_physical_source_range':[physical_interval_begin,physical_interval_end],'full_source_range_equals_original_PTS_selected_PCM':True,'full_source_PCM_allocation_or_temporary_file':False,'full_source_hash_computed_incrementally_once':True,'independent_cumulative_interval_array_equality':True,'actual_EOF_PTS_minus_cumulative_samples':frame_end-cursor,'selected_PTS_begin_minus_physical_begin':begin-physical_interval_begin,'native_interval_samples':written,'native_interval_f32_sha256':native_sha,'clean16k_samples':len(clean),'clean_f32_sha256':f32,'clean_i16_sha256':i16,'actual_interval_geometry':{'native_sr':48000,'native_samples':written,'clean16k_samples':len(clean),'mel_frames':len(clean)//160,'physical_feature_frames':len(clean)//160+1,'encoder_frames':(len(clean)//160+7)//8,'target_tokens':decl['target_tokens']},'full_original_frames_observed_once':True,'selected_interval_original_samples_used_once':True,'synthesized_samples':0,'selected_interval_skipped_samples':0,'annotation_or_source_index_plan_sampler_journal_mutated':False,'no_model_or_GPU_import':'torch'not in sys.modules,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'guard_samples':calls,'member_logical_read_bytes':logical}
 require(report['no_model_or_GPU_import'],'No GPU/model');write_json(OUT/'report.json',report);print(json.dumps({'report':record(OUT/'report.json'),'actual_geometry':report['actual_interval_geometry'],'full_source_native_samples':cursor,'physical_range':report['selected_physical_source_range'],'side':side,'last_frames':report['last_frames']}),flush=True)
finally:write_json(OUT/'CPU_completion.json',{'producer':identity,'guard_samples':calls,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,'actual_exit_unobserved_inside_child':True})