"""INACTIVE exact short head-member physical observation; no decoder remedy.

One fresh full member traversal, bounded by the original declared PCM capacity.
Physical EOF/count/side data are measured, not inferred from the ordinary decoder.
"""
import argparse
from collections import deque
from copy import deepcopy
import ctypes
from ctypes import wintypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import time
import traceback
import types

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
F=Path(__file__).parent;OUTPUT=F/'recorded186564_row5_full_original_member_physical_actual_01'
INPUTS=F/'recorded186564_row5_full_original_member_physical_frozen_inputs_v1.json'
SCOPE='ASR18_recorded186564_row5_full_original_member_physical_CPU_v1'
PRIVATE=8000000000;PREFETCH=268435456


def require(value,message):
    if not value:raise ValueError(message)
def pin(path):
    p=Path(path).resolve();before=p.stat();require(before.st_size<=8*1024**2,'Compact/source pin only')
    data=p.read_bytes();after=p.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Pinned source/input changed')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())


# BEGIN EXACT previously reviewed owned-side-buffer accessor (not public API).
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
# END EXACT previously reviewed owned-side-buffer accessor.


def main():
    global av
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    producer={'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
    review=read(args.ROOT_review);fixed=read(INPUTS)
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==SCOPE
        and review['fixture_source']==pin(__file__)and review['frozen_inputs']==pin(INPUTS)
        and review['input_pins']==fixed['input_pins']and Path(review['output_path']).resolve()==OUTPUT.resolve(),
        'Exact new ROOT source/input/one-output scope')
    for value in fixed['input_pins']:require(pin(value['path'])==value,'All frozen compact/source pins before physical observation')
    require(not OUTPUT.exists(),'Fresh full-member physical actual01 output')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1
        and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())
            for k in('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'Original E CPU-only masks/limits')
    identity=deepcopy(fixed['row_identity']);item=identity['original_plan_row'];row=identity['raw_original_annotation'];decl=item['declared_geometry']
    preparation_scope=read(fixed['preparation_ROOT_scope']['path'])
    require(identity==preparation_scope['exact_original_row_identity']and identity['global_update']==186564
        and identity['local_update_index']==3 and identity['original_update_row_1based']==5
        and row['ordinal']==156134 and row['unique_id']=='alania:a280321c852958fbb53e9d7a2044d1a6_00000'
        and row['source_id']=='a280321c852958fbb53e9d7a2044d1a6'and row['shard']=='0011'
        and row['start']==0 and row['end']==13.143 and decl['native_samples']==630864
        and decl['clean16k_samples']==210288 and len(item['tr_metadata_labels'])==decl['target_tokens']==66
        and item['tr_metadata_labels']==identity['full_target_ids']and item['encoder_frames']==165,
        'Exact accepted row5 original member/full66; declaration is capacity, NEVER physical count oracle')
    config=read(fixed['configuration']['path'])
    from fastmss_resident_feature_stream_v13 import AvailableMemoryPolicy,_raw_finish_without_full_heap_collection,checked_resource_profile
    import fastmss_BPE9223_cpu_common_v12 as core
    from fastmss_bounded_original_audio_v2 import BoundedOriginalReader,BoundedExtent
    from fastmss_raw_lifetime_v1 import RawLifetimeLedger,raw_plan,chunked_hashes
    from bpe_broad_audio_v1 import ROOT,AV_TARGET,resample
    require(checked_resource_profile(config)==fixed['original_resource_policy'],'Current CPU13 original resource values exact')
    raw_contract=core.verify_raw_review(config['raw_lifetime_review']);memory=AvailableMemoryPolicy(private_limit_bytes=PRIVATE)
    require(memory.prospective_reserve_bytes==2500000000 and memory.allocation_wait_seconds==300,
        'Unchanged original CPU2.5GB prospective/private8GB/allocation300s')
    began=time.monotonic();guard_error=None
    def guard():
        nonlocal guard_error
        try:
            require(time.monotonic()-began<180,'Finite180s physical observation')
            sample=memory();system=sample['system_memory']
            require(system['commit_available_bytes']>=2000000000+PREFETCH
                and system['physical_available_bytes']>=1500000000+PREFETCH
                and shutil.disk_usage('E:/').free>=100000000000,'Original floors plus FULL retained256MiB/E100GB')
            return sample
        except BaseException as error:guard_error=repr(error);raise
    initial_resource_sample=guard();core.numpy_runtime(frontend=False)
    import numpy as np
    sys.path.insert(0,str(AV_TARGET));import av
    require(av.__version__=='16.0.1'and os.name=='nt'and sys.version_info[:3]==(3,11,13)
        and ctypes.sizeof(ctypes.c_void_p)==8 and av.packet.PacketSideData.__basicsize__==40,
        'Exact installed and previously source-reviewed Windows64 CPython/PyAV private ABI')
    p=raw_plan(native_samples=decl['native_samples'],native_rate=decl['native_sr'],clean_samples=decl['clean16k_samples'],
        phase=fixed['phase'],role=item['augmentation_role'],channels_bound=raw_contract['maximum_runtime_channels'],
        frame_samples_bound=raw_contract['maximum_decoder_frame_samples'])
    work=p['workspace_incremental_envelopes']
    require(work['decoder']<=raw_contract['decoder_workspace_allowance_bytes']
        and work['weak_transform_FFT']<=raw_contract['transform_workspace_allowance_bytes']
        and work['original_frontend_spectral']<=raw_contract['frontend_workspace_allowance_bytes'],
        'Full original raw/media/transform/frontend/numerical envelope retained without frontend execution')
    actual_row_original_budget=p['raw_peak_bytes']+p['compressed_media_cap_bytes']+max(work.values())+raw_contract['numerical_runtime_reserve_bytes']
    budget=fixed['maximum_original_prospective_bytes']
    require(actual_row_original_budget==885176752 and actual_row_original_budget<=budget==901306112,
        'LARGER901306112 original proof envelope retained; no credit from smaller row/physical counts')
    admission=memory.admit(budget,guard)
    ledger=RawLifetimeLedger(guard=guard);ledger.finish=types.MethodType(_raw_finish_without_full_heap_collection,ledger)
    ledger.begin('row5',p)
    reader=BoundedOriginalReader(ledger=ledger,source_inventory_binding=fixed['source_identity_inventory'],guard=guard)
    path,job=reader.verify_source(Path(fixed['source_artifact']['path']))
    require(reader.final_source_identity.entries[str(path)]==fixed['source_artifact']and job==fixed['source_job'],
        'Original final source identity/receipt/index guards match frozen current artifact')
    db=sqlite3.connect(Path(fixed['source_artifact']['index']['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
    try:entry=db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',(row['source_id'],)).fetchone()
    finally:db.close()
    require(entry==tuple(fixed['recording_index_entry']),
        'Exact frozen original index/member extent; no other recording selection')
    name,offset,size=entry;require(0<size<=p['compressed_media_cap_bytes']and 0<=offset<offset+size<=path.stat().st_size,
        'Original compressed extent cap and actual TAR bounds')
    OUTPUT.mkdir();write_new(OUTPUT/'before_member.json',{'producer':producer,'identity':identity,
        'original_raw_plan':p,'original_full_prospective_admission':admission,'initial_resource_sample':initial_resource_sample,'recording_index_entry':list(entry),
        'no_source_decode_yet':True})
    report={'kind':SCOPE,'passed':False,'producer':producer,'ROOT_review':pin(args.ROOT_review),'input_pins_before':fixed['input_pins'],
        'row':row,'original_declaration':decl,'declared_frontend_geometry':item['declared_frontend_geometry'],
        'full_target_ids':item['tr_metadata_labels'],'full_target_ids_canonical_sha256':digest(item['tr_metadata_labels']),
        'full_target_ids_packed_u32_sha256':hashlib.sha256(b''.join(struct.pack('<I',x)for x in item['tr_metadata_labels'])).hexdigest(),
        'raw_label_text_sha256':hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
        'original_plan_row_canonical_sha256':digest(item),'raw_annotation_canonical_sha256':digest(row),
        'original_raw_plan':p,'original_full_prospective_admission':admission,'original_prospective_bytes':budget,
        'original_actual_row_arithmetic_bytes':actual_row_original_budget,'initial_resource_sample':initial_resource_sample,
        'actual_frontend_execution_claimed':False,'native_or_production_source_mutated':False}
    pcm=native=clean=raw=mono=frame=packet=None
    count=0;frames=0;packets=0;origin=None;channels=None;max_residual=Fraction(0)
    first_frames=[];last_frames=deque(maxlen=8);first_packets=[];last_packets=deque(maxlen=4);side=[]
    try:
        member_hash=hashlib.sha256()
        with path.open('rb')as file:
            file.seek(offset);left=size
            while left:
                guard();block=file.read(min(1024**2,left));require(block,'Entire exact compressed member hash, never whole TAR')
                member_hash.update(block);left-=len(block)
        del block
        with ledger.stage('row5','decode_resample'):
            with BoundedExtent(path,offset,size,guard)as extent,av.open(extent,mode='r',format='webm')as container:
                require(len(container.streams.audio)==1,'One original source stream')
                stream=container.streams.audio[0];context=stream.codec_context;context.thread_count=1
                require(context.name=='opus'and stream.time_base==Fraction(1,1000)
                    and context.sample_rate==48000,'Exact original Opus/millisecond/48k source')
                duration=container.duration
                require(type(duration)is int and 0<duration<=20000000,'ONE independently bounded short member metadata duration<=20s; not expected physical count')
                # Capacity is the unchanged larger historical declaration. An actual
                # overrun fails BEFORE writing; no crop/skip or complete-proof claim.
                pcm=np.empty(decl['native_samples'],dtype=np.float32)
                ledger.observe('row5',pcm,name='bounded_full_member_native_capacity',maximum_bytes=4*decl['native_samples'])
                for packet in container.demux(stream):
                    guard();packets+=1
                    require(packets<=fixed['maximum_demux_packets'],'Bounded original-member packet count; fail, never skip')
                    description={'pts':packet.pts,'dts':packet.dts,'duration':packet.duration,'bytes':packet.size,'time_base':str(packet.time_base)}
                    if len(first_packets)<3:first_packets.append(description)
                    last_packets.append(description)
                    for sd in packet.iter_sidedata():
                        require(len(side)<16 and sd.data_size<=4096,'Bounded codec side-data records')
                        payload=packet_side_bytes(sd)
                        observed={'packet':packets,'type':str(sd.data_type),'data_size':sd.data_size,
                            'description':str(sd.data_desc),'raw_hex':payload.hex()}
                        if str(sd.data_desc)=='Skip Samples'and len(payload)==10:
                            skip,discard,start_reason,end_reason=struct.unpack('<IIBB',payload)
                            observed.update(skip_start_samples=skip,discard_end_samples=discard,skip_reason=start_reason,discard_reason=end_reason)
                        side.append(observed)
                    for frame in packet.decode():
                        guard();require(frame.pts is not None and frame.sample_rate==48000 and frame.format.is_planar
                            and frame.format.bytes<=8 and 0<frame.samples<=raw_contract['maximum_decoder_frame_samples']
                            and 1<=len(frame.layout.channels)<=p['channels_bound'],'Every original planar frame within original preallocated raw bounds')
                        if origin is None:origin=frame.pts*frame.time_base;channels=len(frame.layout.channels)
                        require(len(frame.layout.channels)==channels,'Stable measured original channel count; original mono convention unchanged')
                        residual=(frame.pts*frame.time_base-origin)*48000-count
                        max_residual=max(max_residual,abs(residual));require(abs(residual)<=24,'Original exact half-tick clock bound, no invented gap tolerance')
                        description={'PTS':frame.pts,'time_base':str(frame.time_base),'samples':frame.samples,
                            'physical_cumulative_sample_begin':count,'physical_cumulative_sample_end':count+frame.samples,
                            'PTS_sample_begin':str((frame.pts*frame.time_base-origin)*48000),'PTS_residual_samples':str(residual)}
                        if len(first_frames)<5:first_frames.append(description)
                        last_frames.append(description);frames+=1
                        require(frames<=decl['native_samples'],'Bounded positive-frame count within original full capacity; never crop')
                        raw=frame.to_ndarray();require(raw.shape==(channels,frame.samples),'Complete original planar channels')
                        require(np.isfinite(raw).all(),'Every original physical channel sample finite')
                        if np.issubdtype(raw.dtype,np.integer):raw=raw.astype(np.float32)/float(2**(np.iinfo(raw.dtype).bits-1))
                        else:raw=raw.astype(np.float32,copy=False)
                        mono=raw.mean(axis=0,dtype=np.float32)if channels>1 else raw[0]
                        require(count+len(mono)<=len(pcm),'Complete physical member must fit original declaration; no truncation/freed credit')
                        pcm[count:count+len(mono)]=mono;count+=len(mono);del raw,mono
                complete_demux_and_decode_EOF=True;logical_read=extent.logical_bytes_read
            require(count>0 and frames>0 and origin is not None,'Actual complete positive full-member physical decode')
            native=pcm[:count];native_sha=hashlib.sha256(memoryview(native).cast('B')).hexdigest()
            clean=resample(native,48000)
            ledger.observe('row5',clean,name='whole_full_physical_clean_float32',maximum_bytes=4*decl['clean16k_samples'])
        with ledger.stage('row5','canonical_hash'):clean_f32,clean_i16=chunked_hashes(clean,guard=guard)
        heldout=read(fixed['heldout_PCM']['path'])
        require(heldout['complete']is True and heldout['reference_clips']==12273
            and len(heldout['PCM_i16_sha256'])==12273 and clean_i16 not in set(heldout['PCM_i16_sha256']),
            'Original full heldout PCM exclusion guard, no new heldout decode')
        geometry={'native_sr':48000,'native_samples':count,'clean16k_samples':len(clean),'mel_frames':len(clean)//160,
            'physical_feature_frames':len(clean)//160+1,'encoder_frames':(len(clean)//160+7)//8,'target_tokens':len(item['tr_metadata_labels'])}
        stat=path.stat();report.update(original_TAR={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'recorded_sha256':job['sha256']},
            finalized_source_inventory=fixed['source_identity_inventory'],recording_index_entry=dict(zip(('member_name','offset','bytes'),entry)),
            member_sha256=member_hash.hexdigest(),container_duration_microseconds=duration,origin=str(origin),decoded_frames=frames,
            demuxed_packets_including_flush=packets,complete_demux_and_decode_EOF=complete_demux_and_decode_EOF,
            first_packets=first_packets,last_packets=list(last_packets),packet_side_data=side,
            packet_side_data_API_provenance=fixed['packet_side_data_API_provenance'],
            packet_side_data_bounded_owned_read_controls=packet_side_read_observations,
            first_frames=first_frames,last_frames=list(last_frames),max_abs_PTS_residual_samples=str(max_residual),
            native_sample_channels_before_original_mono=channels,full_source_native_samples=count,full_source_native_f32_sha256=native_sha,
            clean16k_samples=len(clean),clean_f32_sha256=clean_f32,clean_i16_sha256=clean_i16,actual_interval_geometry=geometry,
            declared_native_minus_physical_native_samples=decl['native_samples']-count,
            all_original_physical_samples_used_once=True,synthesized_or_skipped_samples=0,
            original_float32_mono_mean_when_multichannel_and_original_resample_unchanged=True,heldout_PCM_check_performed=True,
            geometry_derived_from_full_actual_PCM_not_frontend=True,member_logical_read_bytes=logical_read)
        pcm=native=clean=raw=mono=frame=packet=None;ledger.finish('row5');report['raw_ledger']=ledger.receipt()
        report['passed']=True
    except BaseException as error:
        report['actual_exception']={'type':type(error).__name__,'repr':repr(error),'message':str(error),'traceback':traceback.format_exc()}
        report['partial_observation_NOT_complete_proof']={'physical_count_so_far':count,'frames':frames,'packets':packets,
            'origin':str(origin),'channels':channels,'max_abs_PTS_residual_samples':str(max_residual),
            'first_frames':first_frames,'last_frames':list(last_frames),'side_data':side}
        write_new(OUTPUT/'actual_failure_immediate.json',report['actual_exception'])
        raise
    finally:
        pcm=native=clean=raw=mono=frame=packet=None
        report.update(guard_error=guard_error,elapsed_seconds=time.monotonic()-began,peak_private_bytes=memory.peak_private,
            allocation_wait_count=memory.allocation_wait_count,allocation_wait_seconds_total=memory.allocation_wait_seconds_total,
            last_resource_sample=memory.last_observed,input_pins_after=[pin(v['path'])for v in fixed['input_pins']])
        report['input_pins_before_after_exact']=report['input_pins_after']==fixed['input_pins']
        require('torch'not in sys.modules,'No Torch/frontend/model/GPU import')
        write_new(OUTPUT/'report.json',report)
        write_new(OUTPUT/'CPU_completion.json',{'producer':producer,'report':pin(OUTPUT/'report.json'),
            'actual_process_return_still_unobserved_inside_child':True,'elapsed_seconds':time.monotonic()-began})
    print(json.dumps({'report':pin(OUTPUT/'report.json'),'geometry':report['actual_interval_geometry'],
        'native_sha':report['full_source_native_f32_sha256'],'clean_sha':report['clean_f32_sha256'],
        'side':report['packet_side_data'],'full_original_samples_once':report['all_original_physical_samples_used_once']}),flush=True)


if __name__=='__main__':main()
