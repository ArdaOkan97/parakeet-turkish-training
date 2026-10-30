"""One independent full-member linear decode; no frontend or source correction."""
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
N=Path(__file__).parent;OUT=N/'full_member_actual_01';require(not OUT.exists(),'Fresh E member observation');OUT.mkdir()
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1,'CPU only/upfrontUTF8')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(k)=='1','Threadcaps1')
proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);who={'pid':proc.pid,'create_time':proc.create_time(),'ppid':proc.ppid()}
began=time.monotonic();peak=0;calls=0;last=0.
def guard():
    global peak,calls,last
    require(time.monotonic()-began<180,'Finite180s')
    if time.monotonic()-last<.25:return
    last=time.monotonic();calls+=1;s=sample_memory(os.getpid(),'one_exact_head_clip142895_EOF');peak=max(peak,s['worker_private_memory']['private_bytes'])
    require(peak<=512*1024**2 and s['paired_sampling_within_one_second'],'Actual512MiB/paired sample')
    require(s['system_memory']['commit_available_bytes']>=2000000000+256*1024**2 and s['system_memory']['physical_available_bytes']>=1500000000+256*1024**2,'Original headroom/full extra256MiB')
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
    require(av.__version__=='16.0.1','Exact pinned PyAV')
    packet_api_provenance={'pinned_pxd': {'path': 'E:\\gpu_training_scratchpad\\scratch\\ASR18_audio_av_16_0_1\\av\\packet.pxd', 'bytes': 562, 'sha256': 'b9b437bfa3382ba37d4e5b6bdefc5e5ce5c81131af9b3fbdfca01db0c40c0082'}, 'pinned_Cython_python_source': {'path': 'E:\\gpu_training_scratchpad\\scratch\\ASR18_audio_av_16_0_1\\av\\packet.py', 'bytes': 13303, 'sha256': 'f7d474fc9ab1fc678274906d12869f0b9eea9afcef3a5f833f871dd1e9b73273'}, 'pinned_extension': {'path': 'E:\\gpu_training_scratchpad\\scratch\\ASR18_audio_av_16_0_1\\av\\packet.cp311-win_amd64.pyd', 'bytes': 90112, 'sha256': '383bdb82cba3a859d2fccaaf7ff6d4e219c9dd5908b2e66039c5351b13a3306d'}, 'exact_private_layout_source': 'av/packet.pxd: data(uint8_t*), size(size_t), dtype(AVPacketSideDataType)', 'raw_bytes_read_method': 'ctypes.string_at on strongly held owned PacketSideData.data ONLY after public fields + VirtualQuery bounds checks', 'max_copy_bytes': 4096, 'public_buffer_API_unavailable_in_installed_PyAV16': True}
    require(os.name=='nt' and sys.version_info[:3]==(3,11,13) and ctypes.sizeof(ctypes.c_void_p)==8 and av.packet.PacketSideData.__basicsize__==40,'Exact inspected Windows64 CPython3.11.13 PyAV16 ABI')
    for name in ('pinned_pxd','pinned_Cython_python_source','pinned_extension'):
     authority=packet_api_provenance[name];require(record(authority['path'])==authority,'Exact installed PyAV layout/API provenance unchanged')
    failed_update_pin={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_original_interval_post142894_failure_actual_v1\\probe_actual_01\\actual_failed_update_metadata.json', 'bytes': 70464, 'sha256': '132f2df3a3a3e0d993e674094ae3b85e894b0bfd9cea505d366deb69c7bd3aa0'}
    prior_probe_pin={'path': 'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_original_interval_post142894_failure_actual_v1\\probe_actual_01\\report.json', 'bytes': 34994, 'sha256': '67105dfe6a9aec6cfeefcaddc130f2c0de29602752ab1f32f147bf39055ee8aa'}
    require(record(failed_update_pin['path'])==failed_update_pin and record(prior_probe_pin['path'])==prior_probe_pin,'Exact failed update and completed ordered probe')
    metadata=read_json(N/'probe_actual_01/actual_failed_update_metadata.json');item=metadata['rows'][6];row=item['broad_original'];decl=item['declared_geometry']
    require(metadata['global_update']==142895 and row['unique_id']=='alania:0ce0d55b73c946f80abea8bbe41f3b27_00000'
        and row['start']==0 and row['end']==14.443 and row['shard']=='0014'and decl['native_samples']==693264
        and len(item['tr_metadata_labels'])==decl['target_tokens']==61,'Exact failed interval/source/full target declaration')
    reader=BroadAudioReader(guard=guard);path,job=reader.verify_source(ROOT/'datasets/yodas3/audio/0014.tar')
    db=sqlite3.connect((ROOT/'indexes/yodas3_tar_0014.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
    finally:db.close()
    name,offset,size=entry;require(name==row['source_id']+'.webm'and 0<size<=32*1024**2,'Exact bounded original member')
    member_hash=hashlib.sha256()
    with path.open('rb')as f:
        f.seek(offset);left=size
        while left:
            guard();data=f.read(min(1024**2,left));require(data,'Complete exact member bytes');member_hash.update(data);left-=len(data)
    count=0;frames=0;packets=0;first_frames=[];last_frames=deque(maxlen=5);first_packets=[];last_packets=deque(maxlen=4);side=[];maximum_residual=Fraction(0)
    with Extent(path,offset,size,guard)as ext,av.open(ext,mode='r',format='webm')as container:
        require(len(container.streams.audio)==1,'One original audio stream');stream=container.streams.audio[0];stream.codec_context.thread_count=1
        require(stream.codec_context.name=='opus'and stream.time_base==Fraction(1,1000),'Exact Opus/millisecond clock')
        duration=container.duration;require(duration==14_461_000,'One bounded full original clip')
        bound=math.ceil(Fraction(duration,1000000)*48000)
        require(proc.memory_info().private+bound*4+decl['clean16k_samples']*8+32*1024**2<512*1024**2,'Actual current plus complete PCM/resample allocation')
        pcm=np.empty(bound,dtype=np.float32);origin=None
        for packet in container.demux(stream):
            guard();packets+=1;p={'pts':packet.pts,'dts':packet.dts,'duration':packet.duration,'bytes':packet.size,'time_base':str(packet.time_base)}
            if len(first_packets)<3:first_packets.append(p)
            last_packets.append(p)
            for sd in packet.iter_sidedata():
                require(len(side)<16 and sd.data_size<=4096,'Bounded exact codec side metadata')
                payload=packet_side_bytes(sd)
                observed={'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc),'raw_hex':payload.hex()}
                if str(sd.data_desc)=='Skip Samples'and len(payload)==10:
                    skip,discard,start_reason,end_reason=struct.unpack('<IIBB',payload);observed.update(skip_start_samples=skip,discard_end_samples=discard,skip_reason=start_reason,discard_reason=end_reason)
                side.append(observed)
            for frame in packet.decode():
                guard();require(frame.pts is not None and frame.sample_rate==48000 and frame.format.is_planar
                    and 0<frame.samples<=16384 and len(frame.layout.channels)<=8,'Original complete frame envelope')
                if origin is None:origin=frame.pts*frame.time_base
                residual=(frame.pts*frame.time_base-origin)*48000-count;maximum_residual=max(maximum_residual,abs(residual))
                require(abs(residual)<=24,'Explicit original half-tick clock, no true gap or overlap')
                description={'PTS':frame.pts,'time_base':str(frame.time_base),'samples':frame.samples,
                    'cumulative_source_sample_begin':count,'PTS_residual_samples':str(residual)}
                if len(first_frames)<3:first_frames.append(description)
                last_frames.append(description);frames+=1
                raw=frame.to_ndarray();require(raw.shape==(len(frame.layout.channels),frame.samples),'Complete planar channels')
                if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
                else:raw=raw.astype(np.float32,copy=False)
                mono=raw.mean(axis=0,dtype=np.float32)if len(frame.layout.channels)>1 else raw[0]
                require(count+len(mono)<=len(pcm),'Complete full member fits; never truncate')
                pcm[count:count+len(mono)]=mono;count+=len(mono);del raw,mono
        logical_read=ext.logical_bytes_read
    require(frames==723 and count==693255,'Independent full-member physical EOF matches observed strict failure')
    native=pcm[:count];native_hash=hashlib.sha256(memoryview(native).cast('B')).hexdigest();clean=resample(native,48000)
    f32,i16=chunked_hashes(clean,guard=guard);geometry={'native_sr':48000,'native_samples':count,'clean16k_samples':len(clean),
        'mel_frames':len(clean)//160,'physical_feature_frames':len(clean)//160+1,'encoder_frames':(len(clean)//160+7)//8,'target_tokens':len(item['tr_metadata_labels'])}
    require(len(clean)==231085 and geometry['mel_frames']==1444 and geometry['encoder_frames']==181,'Actual full source clean count and unchanged mel/encoder geometry are distinct from historical larger declaration')
    labels=item['tr_metadata_labels'];target_sha=hashlib.sha256(b''.join(struct.pack('<I',x)for x in labels)).hexdigest()
    report={'kind':'ASR18_one_complete_original_head_clip142895_EOF_observation_v1','source':record(__file__),'producer':who,
        'prior_ordered_probe':record(N/'probe_actual_01/report.json'),'failed_update':record(N/'probe_actual_01/actual_failed_update_metadata.json'),
        'global_update':142895,'update_index':14,'row_index':6,'row':row,'declared_geometry':decl,
        'declared_frontend_geometry':item['declared_frontend_geometry'],'full_target_ids':labels,'full_target_ids_sha256':target_sha,
        'original_TAR':{'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'recorded_sha256':job['sha256']},
        'recording_index_entry':dict(zip(('member_name','offset','bytes'),entry)),'member_sha256':member_hash.hexdigest(),
        'container_duration_microseconds':duration,'origin':str(origin),'decoded_frames':frames,'demuxed_packets_including_flush':packets,
        'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,'packet_side_data_API_provenance':packet_api_provenance,'packet_side_data_bounded_actual_read_controls':packet_side_read_observations,
        'first_frames':first_frames,'last_frames':list(last_frames),'maximum_abs_PTS_residual_samples':str(maximum_residual),
        'full_source_native_samples':count,'full_source_native_f32_sha256':native_hash,
        'clean16k_samples':len(clean),'clean_f32_sha256':f32,'clean_i16_sha256':i16,'actual_interval_geometry':geometry,
        'declared_native_minus_physical_native_samples':9,'all_original_physical_samples_used_once':True,
        'synthesized_or_skipped_samples':0,'ordinary_decoder_or_source_annotation_index_plan_labels_sampler_journal_mutated':False,
        'actual_frontend_execution_claimed':False,'geometry_derived_from_full_actual_PCM':True,'no_model_GPU_import':'torch'not in sys.modules,
        'member_logical_read_bytes':logical_read,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'guard_samples':calls}
    require(record(failed_update_pin['path'])==failed_update_pin and record(prior_probe_pin['path'])==prior_probe_pin,'Original labels and annotation evidence unchanged after one decode')
    require(report['no_model_GPU_import'],'No model/GPU');write_json(OUT/'report.json',report)
    print(json.dumps({'report':record(OUT/'report.json'),'geometry':geometry,'native_sha':native_hash,'clean_sha':f32,'side':side,'last_frames':list(last_frames)}),flush=True)
finally:write_json(OUT/'CPU_completion.json',{'producer':who,'guard_samples':calls,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,'actual_exit_unobserved_inside_child':True})
