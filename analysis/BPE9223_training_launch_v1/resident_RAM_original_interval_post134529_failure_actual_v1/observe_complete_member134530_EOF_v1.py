"""One independent full-member linear decode; no frontend or source correction."""
import hashlib,json,math,os,shutil,sqlite3,struct,sys,time
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
    last=time.monotonic();calls+=1;s=sample_memory(os.getpid(),'one_exact_head_clip134530_EOF');peak=max(peak,s['worker_private_memory']['private_bytes'])
    require(peak<=512*1024**2 and s['paired_sampling_within_one_second'],'Actual512MiB/paired sample')
    require(s['system_memory']['commit_available_bytes']>=2000000000+256*1024**2 and s['system_memory']['physical_available_bytes']>=1500000000+256*1024**2,'Original headroom/full extra256MiB')
    require(shutil.disk_usage('E:/').free>=100000000000,'E100GB reserve')
try:
    numpy_runtime(frontend=False)
    import numpy as np
    sys.path.insert(0,str(AV_TARGET));import av
    require(av.__version__=='16.0.1','Exact pinned PyAV')
    metadata=read_json(N/'probe_actual_01/actual_failed_update_metadata.json');item=metadata['rows'][3];row=item['broad_original'];decl=item['declared_geometry']
    require(metadata['global_update']==134530 and row['unique_id']=='alania:9105d11d7d12c9c166028f98935dc107_00000'
        and row['start']==0 and row['end']==12.504 and row['shard']=='0027'and decl['native_samples']==600192
        and len(item['tr_metadata_labels'])==decl['target_tokens']==82,'Exact failed interval/source/full target declaration')
    reader=BroadAudioReader(guard=guard);path,job=reader.verify_source(ROOT/'datasets/yodas3/audio/0027.tar')
    db=sqlite3.connect((ROOT/'indexes/yodas3_tar_0027.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
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
        duration=container.duration;require(duration is not None and 0<duration<=30_000_000,'One bounded full original clip')
        bound=math.ceil(Fraction(duration,1000000)*48000)
        require(proc.memory_info().private+bound*4+decl['clean16k_samples']*8+32*1024**2<512*1024**2,'Actual current plus complete PCM/resample allocation')
        pcm=np.empty(bound,dtype=np.float32);origin=None
        for packet in container.demux(stream):
            guard();packets+=1;p={'pts':packet.pts,'dts':packet.dts,'duration':packet.duration,'bytes':packet.size,'time_base':str(packet.time_base)}
            if len(first_packets)<3:first_packets.append(p)
            last_packets.append(p)
            for sd in packet.iter_sidedata():side.append({'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,'description':str(sd.data_desc)})
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
    require(frames==626 and count==600190,'Independent full-member physical EOF matches observed strict failure')
    native=pcm[:count];native_hash=hashlib.sha256(memoryview(native).cast('B')).hexdigest();clean=resample(native,48000)
    f32,i16=chunked_hashes(clean,guard=guard);geometry={'native_sr':48000,'native_samples':count,'clean16k_samples':len(clean),
        'mel_frames':len(clean)//160,'physical_feature_frames':len(clean)//160+1,'encoder_frames':(len(clean)//160+7)//8,'target_tokens':len(item['tr_metadata_labels'])}
    require(len(clean)==200064 and geometry['mel_frames']==1250 and geometry['encoder_frames']==157,'True full source retains historical clean/frontend encoder geometry')
    labels=item['tr_metadata_labels'];target_sha=hashlib.sha256(b''.join(struct.pack('<I',x)for x in labels)).hexdigest()
    report={'kind':'ASR18_one_complete_original_head_clip134530_EOF_observation_v1','source':record(__file__),'producer':who,
        'prior_ordered_probe':record(N/'probe_actual_01/report.json'),'failed_update':record(N/'probe_actual_01/actual_failed_update_metadata.json'),
        'global_update':134530,'update_index':9,'row_index':3,'row':row,'declared_geometry':decl,
        'declared_frontend_geometry':item['declared_frontend_geometry'],'full_target_ids':labels,'full_target_ids_sha256':target_sha,
        'original_TAR':{'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'recorded_sha256':job['sha256']},
        'recording_index_entry':dict(zip(('member_name','offset','bytes'),entry)),'member_sha256':member_hash.hexdigest(),
        'container_duration_microseconds':duration,'origin':str(origin),'decoded_frames':frames,'demuxed_packets_including_flush':packets,
        'first_packets':first_packets,'last_packets':list(last_packets),'packet_side_data':side,
        'first_frames':first_frames,'last_frames':list(last_frames),'maximum_abs_PTS_residual_samples':str(maximum_residual),
        'full_source_native_samples':count,'full_source_native_f32_sha256':native_hash,
        'clean16k_samples':len(clean),'clean_f32_sha256':f32,'clean_i16_sha256':i16,'actual_interval_geometry':geometry,
        'declared_native_minus_physical_native_samples':2,'all_original_physical_samples_used_once':True,
        'synthesized_or_skipped_samples':0,'ordinary_decoder_or_source_annotation_index_plan_labels_sampler_journal_mutated':False,
        'actual_frontend_execution_claimed':False,'geometry_derived_from_full_actual_PCM':True,'no_model_GPU_import':'torch'not in sys.modules,
        'member_logical_read_bytes':logical_read,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'guard_samples':calls}
    require(report['no_model_GPU_import'],'No model/GPU');write_json(OUT/'report.json',report)
    print(json.dumps({'report':record(OUT/'report.json'),'geometry':geometry,'native_sha':native_hash,'clean_sha':f32,'side':side,'last_frames':list(last_frames)}),flush=True)
finally:write_json(OUT/'CPU_completion.json',{'producer':who,'guard_samples':calls,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,'actual_exit_unobserved_inside_child':True})
