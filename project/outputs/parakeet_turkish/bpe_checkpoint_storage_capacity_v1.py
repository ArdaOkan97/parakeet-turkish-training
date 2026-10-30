"""Disabled future checkpoint-capacity model and streaming codec sizing only.

No catalog/publication/retirement implementation or deletion API exists here.
Inputs are small existing JSON metadata; tensor payloads are never opened.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

from bpe_broad_production_state_v1 import canonical, require
from bpe_broad_audio_v1 import record

ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
CHUNK=64*1024
POLICY_KIND='ASR18_future_owned_packed_checkpoint_capacity_PROPOSAL_v1'


def integer(value, low=0):
    require(type(value) is int and value>=low,'Exact nonboolean proposal count/bytes')
    return value


def E_allocation_unit():
    require(os.name=='nt','Actual designated E filesystem query required')
    sectors,sector_bytes,free,total=(wintypes.DWORD() for _ in range(4))
    api=ctypes.WinDLL('kernel32',use_last_error=True).GetDiskFreeSpaceW
    api.argtypes=[wintypes.LPCWSTR,*([ctypes.POINTER(wintypes.DWORD)]*4)]
    api.restype=wintypes.BOOL
    require(api('E:\\',ctypes.byref(sectors),ctypes.byref(sector_bytes),ctypes.byref(free),ctypes.byref(total)),
            'Actual E-only allocation unit unavailable; no default/fallback')
    unit=sectors.value*sector_bytes.value
    require(512<=unit<=32*1024**2 and unit&(unit-1)==0,'Bounded actual filesystem allocation unit')
    return {'path':'E:\\','allocation_unit_bytes':unit,'sectors_per_cluster':sectors.value,'bytes_per_sector':sector_bytes.value}


def capacity_bound(policy, *, allocation_unit_bytes):
    require(policy['kind']==POLICY_KIND and policy['accepted'] is False
            and policy['production_enabled'] is False and policy['TEMPLATE_NOT_LAUNCHABLE'] is True,
            'Disabled proposal only, never storage activation')
    count=integer(policy['maximum_publications'],1)
    updates=2*((integer(policy['broad_population'],1)+13)//14)+(integer(policy['cleaner_population'],1)+13)//14
    require(count==updates==policy['planned_native_updates'] and policy['maximum_payload_publications_per_actual_positive_update']==1,
            'Full worst publication horizon, no assumed minimum window size')
    unit=integer(allocation_unit_bytes,512)
    require(unit<=32*1024**2 and unit&(unit-1)==0,'Exact allocation quantum')
    for key in ('maximum_payload_bytes','maximum_metadata_bytes','maximum_pointer_bytes','maximum_uncompressed_record_bytes',
        'maximum_compressed_record_bytes','maximum_records_per_pack','maximum_pack_index_entry_bytes',
        'maximum_history_events_per_publication','maximum_history_event_bytes','maximum_history_events_per_segment',
        'maximum_hot_state_and_control_bytes','maximum_atomic_scratch_bytes'):
        integer(policy[key],1)
    require(policy['hard_live_pair_peak']==5 and policy['steady_live_pairs']==4
            and policy['maximum_records_per_pack']<=1024
            and policy['maximum_compressed_record_bytes']<=512*1024
            and policy['maximum_records_per_pack']*policy['maximum_compressed_record_bytes']<=512*1024**2,
            'Fixed bounded live pairs/current archive pack')
    packs=math.ceil(count/policy['maximum_records_per_pack'])
    events=count*policy['maximum_history_events_per_publication']+32
    event_segments=math.ceil(events/policy['maximum_history_events_per_segment'])
    archive=count*policy['maximum_compressed_record_bytes']+packs*4096
    indexes=count*policy['maximum_pack_index_entry_bytes']+packs*4096
    history=events*policy['maximum_history_event_bytes']+event_segments*4096
    live=5*(policy['maximum_payload_bytes']+policy['maximum_metadata_bytes']+policy['maximum_pointer_bytes'])
    files=2*packs+event_segments+5*3+64
    rounding=files*(unit-1)
    namespace=archive+indexes+history+live+rounding+policy['maximum_hot_state_and_control_bytes']+policy['maximum_atomic_scratch_bytes']
    return {'kind':'ASR18_disabled_full_horizon_checkpoint_capacity_arithmetic_v1','planned_native_updates':updates,
        'maximum_publications':count,'archive_pack_count_upper_bound':packs,
        'history_event_count_upper_bound':events,'history_segment_count_upper_bound':event_segments,
        'compressed_archive_bytes_upper_bound':archive,'pack_index_bytes_upper_bound':indexes,
        'history_bytes_upper_bound':history,'live_and_inflight_pair_bytes_upper_bound':live,
        'filesystem_rounding_bytes_upper_bound':rounding,'owned_namespace_bytes_upper_bound':namespace,
        'minimum_initial_free_for_namespace_plus100GB_reserve':namespace+100_000_000_000,
        'payload_write_IO_bytes_upper_bound':count*policy['maximum_payload_bytes'],
        'strict_candidate_plus_four_live_payload_hash_IO_bytes_upper_bound':count*5*policy['maximum_payload_bytes'],
        'per_publication_parsed_history_entries':0,'maximum_hot_checkpoint_entries':5,
        'maximum_open_pack_index_entries':policy['maximum_records_per_pack'],
        'other_authoritative_journal_DEV_telemetry_DATA_growth_budgeted':False,
        'actual_full_run_capacity_or_throughput_approved':False}


def measure_lossless_bundle(components, *, guard):
    """Count/hash a prospective compressed metadata record; write no archive.

    Complete original bytes, including JSON formatting, are compressed. The
    independent decompressor consumes bounded64KiB output slices and verifies
    the uncompressed stream hash. This is sizing, not publication/retirement.
    """
    require(type(components) is list and 1<=len(components)<=8,'Small bounded JSON components')
    pins=[]
    for path in components:
        path=Path(path).resolve()
        require((path.is_relative_to(ROOT.resolve()) or path.is_relative_to(PROJECT))
                and path.suffix=='.json' and 0<path.stat().st_size<=4*1024**2,
                'Only small existing project/E JSON metadata; no payload/audio input')
        guard();pins.append(record(path))
    header=canonical({'kind':'ASR18_METADATA_CODEC_SIZING_ONLY_NOT_CHECKPOINT_ADMISSION_v1',
        'components':pins,'source':record(__file__)})+b'\n'
    require(len(header)<=32*1024 and sum(pin['bytes'] for pin in pins)+len(header)<=8*1024**2,'Explicit8MiB raw record/header bound')
    encoder=zlib.compressobj(level=6,wbits=15);decoder=zlib.decompressobj(wbits=15)
    raw_hash=hashlib.sha256();decoded_hash=hashlib.sha256();packed_hash=hashlib.sha256()
    raw_bytes=compressed_bytes=decoded_bytes=maximum_decoded=0
    def compressed(block):
        nonlocal compressed_bytes,decoded_bytes,maximum_decoded
        if not block:return
        packed_hash.update(block);compressed_bytes+=len(block)
        tail=block
        while True:
            guard();part=decoder.decompress(tail,CHUNK);tail=decoder.unconsumed_tail
            decoded_hash.update(part);decoded_bytes+=len(part);maximum_decoded=max(maximum_decoded,len(part))
            require(decoded_bytes<=8*1024**2,'Bounded decoded sizing record')
            if not tail and len(part)<CHUNK:break
    def raw(block):
        nonlocal raw_bytes
        raw_hash.update(block);raw_bytes+=len(block);compressed(encoder.compress(block))
    raw(header)
    for pin in pins:
        with Path(pin['path']).open('rb') as stream:
            while block:=stream.read(CHUNK):guard();raw(block)
        require(record(pin['path'])==pin,'Input metadata changed during sizing')
    compressed(encoder.flush())
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
            and raw_bytes==decoded_bytes and raw_hash.hexdigest()==decoded_hash.hexdigest(),'Independent streaming lossless round trip')
    return {'kind':'ASR18_actual_JSON_only_lossless_archive_sizing_v1','components':pins,'raw_bytes':raw_bytes,
        'compressed_bytes':compressed_bytes,'raw_sha256':raw_hash.hexdigest(),'compressed_sha256':packed_hash.hexdigest(),
        'lossless_roundtrip_sha256_equal':True,'maximum_decoded_slice_bytes':maximum_decoded,
        'input_read_chunk_bytes':CHUNK,'zlib_compile_version':zlib.ZLIB_VERSION,'zlib_runtime_version':zlib.ZLIB_RUNTIME_VERSION,
        'fits_proposed512KiB_record_cap':compressed_bytes<=512*1024,'archive_written':False,'checkpoint_admitted':False}


def require_record_fit(measurement):
    require(measurement['lossless_roundtrip_sha256_equal'] is True
            and type(measurement['raw_bytes']) is int and 0<measurement['raw_bytes']<=8*1024**2
            and type(measurement['compressed_bytes']) is int and 0<measurement['compressed_bytes']<=512*1024,
            'Future archive cap unresolved; no field dropping, payload retirement, or silent fallback')
    return True
