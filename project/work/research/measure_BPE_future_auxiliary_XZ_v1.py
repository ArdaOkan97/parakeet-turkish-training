"""Read-only CLOSED20 metadata -> NEW bounded lossless XZ sizing, not adoption.

No checkpoint/payload/audio, original file write or storage-owner publication.
The frozen checkpoint gzip archive and indexed2 source stay unchanged.
"""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import struct
import sys
import time

P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
import bpe_indexed_owned_checkpoint_storage_v2 as m
from fastmss_row_serial_cpu_v6 import environment,resource_guard,admit
from bpe_broad_production_state_v1 import canonical,require
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

def guard():
    value=resource_guard();require(value['worker_private_memory']['private_bytes']<256_000_000,'XZ sizing256MB actual-private cap')
    return value

def main(output,reference):
    environment();admission=admit(256_000_000,guard=guard);began=time.monotonic();samples=[guard()]
    output=Path(output).resolve();reference=Path(reference).resolve()
    require(output.is_relative_to(m.ROOT) and not output.exists(),'New E-only sizing namespace')
    require(reference==m.ROOT/'analysis/ASR18_memory_diagnosis_v1/fullpopulation_firstwindow20_conv_actual_01','Exact closed reference only')
    output.mkdir();files=[];directories=0;raw_bytes=0
    for path in reference.rglob('*'):
        guard()
        if path.is_dir():directories+=1;continue
        if path.suffix.lower() not in ('.json','.jsonl','.log','.txt'):continue
        require(not path.is_symlink() and path.stat().st_size<=4*1024**2,'Only original small metadata, never binary payload or audio')
        raw_bytes+=path.stat().st_size;files.append(path)
        require(len(files)<=512 and raw_bytes<=32*1024**2 and directories<=128,'Frozen auxiliary raw cardinality/geometry caps')
    files.sort(key=str);pins=[record(path) for path in files]
    header={'kind':'READ_ONLY_CLOSED_METADATA_SIZING_NOT_OWNERSHIP_OR_NATIVE_PROOF','members':pins}
    encoded=canonical(header);require(len(encoded)<=512*1024,'Complete bounded header')
    target=output/'READ_ONLY_closed_metadata_sizing.xz';compression_start=time.monotonic()
    with target.open('xb') as raw:
        capped=m._CappedWriter(raw,2*1024**2)
        with lzma.LZMAFile(capped,mode='wb',format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,preset=6) as zipped:
            zipped.write(struct.pack('<Q',len(encoded)));zipped.write(encoded)
            for pin in pins:
                zipped.write(struct.pack('<Q',pin['bytes']));sha=hashlib.sha256();count=0
                with Path(pin['path']).open('rb') as source:
                    while block:=source.read(1024**2):
                        samples.append(guard());sha.update(block);count+=len(block);zipped.write(block)
                require(count==pin['bytes'] and sha.hexdigest()==pin['sha256'],'Original metadata unchanged during compression')
        raw.flush();os.fsync(raw.fileno())
    compression_seconds=time.monotonic()-compression_start;verify_start=time.monotonic()
    with lzma.open(target,'rb') as stream:
        size=struct.unpack('<Q',m._read_exact(stream,8))[0];require(size==len(encoded),'Exact complete bounded XZ header')
        require(m._read_exact(stream,size)==encoded,'Exact original header bytes')
        for pin in pins:
            size=struct.unpack('<Q',m._read_exact(stream,8))[0];require(size==pin['bytes'],'Exact member boundary')
            sha=hashlib.sha256()
            while size:
                samples.append(guard());block=m._read_exact(stream,min(1024**2,size));size-=len(block);sha.update(block)
            require(sha.hexdigest()==pin['sha256'],'Lossless original member SHA after XZ decode')
        require(stream.read(1)==b'','No omitted or extra XZ members')
    for pin in pins:require(record(pin['path'])==pin,'Every original metadata pin remains unchanged')
    sample=guard();samples.append(sample);require(time.monotonic()-began<60,'XZ measurement60s hard time bound')
    actual=record(target);require(actual['bytes']<=2*1024**2,'Same complete archive2MiB hard cap, no margin widening')
    result={'kind':'ASR18_READ_ONLY_closed20_lossless_XZ_auxiliary_sizing_v1','passed':True,'source':record(__file__),
        'bounded_stream_source':record(m.__file__),'producer':{'pid':os.getpid(),'ppid':os.getppid()},'CPU_admission':admission,
        'reference':str(reference),'member_count':len(pins),'original_directory_count':directories,'raw_original_bytes':raw_bytes,
        'complete_header_bytes':len(encoded),'archive':actual,'same_cap_bytes':2*1024**2,'spare_bytes':2*1024**2-actual['bytes'],
        'compression_seconds':compression_seconds,'verification_seconds':time.monotonic()-verify_start,'wall_seconds':time.monotonic()-began,
        'observed_private_peak_bytes':max(v['worker_private_memory']['private_bytes'] for v in samples),
        'codec':{'container':'XZ','algorithm':'LZMA2','preset':6,'integrity_check':'CRC64','Python':sys.version},
        'all_original_metadata_byte_hashes_unchanged':True,'all_members_redecoded_exact':True,
        'actual_tensor_payload_audio_or_native_operation':False,'old_or_current_file_mutated':False,
        'future_arbitrary_window_archive_fit_proved':False,'public_owner_or_production_approved':False,'production_enabled':False}
    write_json(output/'report.json',result);print(json.dumps({'passed':True,'archive':actual,'spare_bytes':result['spare_bytes'],'report':record(output/'report.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reference',required=True)
    a=p.parse_args();main(a.output,a.reference)
