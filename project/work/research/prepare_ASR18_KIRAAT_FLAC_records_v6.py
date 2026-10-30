"""Append-only resumable original-FLAC preparation and independent all-byte audit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

from ASR18_FLAC_carrier_guard_v6 import CarrierGuard
from ASR18_FLAC_carrier_guard_v3 import PacedParquet,CACHE,AREA,metadata,signature
from cpu_preparation_guard_v2 import now
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/parakeet_turkish"))
from atomic_io_retry_v1 import write_json


def verify_source(config,guard):
    job=config['job'];source=Path(job['destination'])
    assert source.resolve().is_relative_to(CACHE.parents[1].resolve())
    assert signature(source)==config['source_signature']
    receipt=source.with_suffix(source.suffix+'.receipt.json')
    assert guard.binding(receipt)==config['source_receipt']
    value=metadata(receipt)
    assert value['sha256']==job['sha256'] and value['revision']==job['revision']
    assert source.stat().st_size==job['size']
    return source


def parquet(source,guard):
    import pyarrow.parquet as pq
    stream=PacedParquet(source,guard)
    pf=pq.ParquetFile(stream,pre_buffer=False,memory_map=False)
    for i in range(pf.metadata.num_row_groups):
        group=pf.metadata.row_group(i)
        sizes=[group.column(j).total_uncompressed_size for j in range(group.num_columns)
               if group.column(j).path_in_schema=='audio.bytes']
        assert sizes and max(sizes)<=512*1024**2,'Original audio-column bound remains unresolved, never exclude rows'
    return pf,stream


def append_verified_record(stream,media,total,guard):
    stream.seek(0,2);length=stream.tell();assert length>=total
    retained=min(len(media),length-total)
    if retained:
        stream.seek(total);actual=guard.read(stream,retained)
        assert actual==media[:retained],'Existing payload prefix differs; no deletion/truncation permitted'
    stream.seek(total+retained)
    if retained<len(media):guard.write_payload(stream,memoryview(media)[retained:])


def prepare(config,guard):
    import pyarrow as pa
    source=verify_source(config,guard);folder=CACHE/config['job']['sha256'];folder.mkdir(exist_ok=True)
    receipt_path=folder/'receipt.json';assert not receipt_path.exists(),'Never replace completed carrier'
    plan_path=folder/'plan.json'
    if plan_path.exists():
        old=metadata(plan_path);assert old['source_sha256']==config['job']['sha256']
        assert old['source_size']==config['source_signature']['bytes'] and old['source_mtime_ns']==config['source_signature']['mtime_ns']
    else:
        write_json(plan_path,{'kind':'ASR18_original_FLAC_record_preparation_v3',
            'source_path':str(source),'source_sha256':config['job']['sha256'],'revision':config['job']['revision'],
            'source_size':source.stat().st_size,'source_mtime_ns':source.stat().st_mtime_ns,
            'source_acquisition_receipt':config['source_receipt'],'builder':guard.binding(__file__),
            'no_audio_decode_or_quality_filter':True,'physical_rows_not_catalog_ordinals':True})
    plan=metadata(plan_path);pf,source_stream=parquet(source,guard)
    database=folder/'rows.sqlite';db=sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA max_page_count=15625')
    db.execute('CREATE TABLE IF NOT EXISTS records(row_index INTEGER PRIMARY KEY,clip_id TEXT NOT NULL,source_id TEXT NOT NULL,offset INTEGER NOT NULL,bytes INTEGER NOT NULL,sha256 TEXT NOT NULL)')
    payload=folder/'records.bin';stream=payload.open('r+b' if payload.exists() else 'x+b',buffering=0)
    initial=payload.stat().st_size;count=0;total=0;digest=hashlib.sha256();ordered=hashlib.sha256()
    def flush():stream.flush();os.fsync(stream.fileno());db.commit()
    guard.flush_hook=flush
    try:
        for batch in pf.iter_batches(batch_size=1,columns=['audio','id','source_id'],use_threads=False):
            guard.wait();item=batch.to_pylist()[0];media=item['audio']['bytes']
            assert isinstance(media,bytes) and media.startswith(b'fLaC') and 0<len(media)<=32*1024**2
            sha=hashlib.sha256(media).hexdigest();clip=str(item['id']);src=str(item['source_id'])
            expected=(clip,src,total,len(media),sha)
            existing=db.execute('SELECT clip_id,source_id,offset,bytes,sha256 FROM records WHERE row_index=?',(count,)).fetchone()
            if existing is not None:assert tuple(existing)==expected,'Existing row mapping differs; preserve unresolved prefix'
            append_verified_record(stream,media,total,guard)
            if existing is None:db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',(count,*expected))
            digest.update(media);ordered.update(json.dumps([count,*expected],separators=(',',':')).encode())
            total+=len(media);count+=1
            if count%128==0:
                flush();write_json(guard.run/'progress.json',{'at_utc':now(),'mode':'prepare','physical_rows':count,
                    'payload_bytes':total,'prefix_bytes_at_start':initial,'new_payload_bytes':guard.payload_written,
                    'paced_IO_bytes':guard.bytes})
            del media,item,batch
        flush();assert count==pf.metadata.num_rows==db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
        assert payload.stat().st_size==total,'Unaccounted payload tail is preserved and unresolved'
    finally:
        guard.flush_hook=lambda:None;stream.close();db.close();pf.close();source_stream.close();del pf
        gc.collect();pa.default_memory_pool().release_unused()
    verify_source(config,guard)
    receipt={'kind':'ASR18_original_compressed_FLAC_records_v1','complete':True,'source':plan,'physical_rows':count,
        'records':{'path':str(payload),'bytes':total,'sha256':digest.hexdigest(),'mtime_ns':payload.stat().st_mtime_ns},
        'index':{**guard.binding(database),'mtime_ns':database.stat().st_mtime_ns},'ordered_mapping_sha256':ordered.hexdigest(),
        'full_original_FLAC_bytes_preserved':True,'decoded_PCM_pool_created':False,'quality_language_or_disagreement_exclusions':[],
        'source_files_modified':False,'TRAIN_admitted':False}
    write_json(receipt_path,receipt)
    return {'receipt':guard.binding(receipt_path),'records_bytes':total,'physical_rows':count,
            'new_payload_bytes':guard.payload_written,'resumed_verified_prefix_bytes':initial}


def audit(config,guard):
    import pyarrow as pa
    source=verify_source(config,guard);folder=CACHE/config['job']['sha256'];receipt_path=folder/'receipt.json'
    receipt=metadata(receipt_path);assert receipt['complete'] is True
    assert receipt['source']['source_sha256']==config['job']['sha256']
    payload=receipt['records'];index=receipt['index']
    assert guard.binding(payload['path'])=={k:payload[k] for k in ('path','bytes','sha256')}
    assert guard.binding(index['path'])=={k:index[k] for k in ('path','bytes','sha256')}
    pf,source_stream=parquet(source,guard)
    db=sqlite3.connect(Path(index['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp_store=MEMORY')
    count=0;total=0;ordered=hashlib.sha256()
    try:
        with Path(payload['path']).open('rb',buffering=0) as stream:
            for batch in pf.iter_batches(batch_size=1,columns=['audio','id','source_id'],use_threads=False):
                guard.wait();item=batch.to_pylist()[0];original=item['audio']['bytes']
                row=db.execute('SELECT clip_id,source_id,offset,bytes,sha256 FROM records WHERE row_index=?',(count,)).fetchone()
                assert row is not None and row[:2]==(str(item['id']),str(item['source_id'])) and row[2]==total
                assert 0<row[3]<=32*1024**2 and original.startswith(b'fLaC')
                actual=guard.read(stream,row[3])
                assert actual==original and hashlib.sha256(actual).hexdigest()==row[4]
                ordered.update(json.dumps([count,*row],separators=(',',':')).encode())
                count+=1;total+=len(original)
                if count%128==0:write_json(guard.run/'progress.json',{'at_utc':now(),'mode':'audit','physical_rows':count,'payload_bytes':total,'paced_IO_bytes':guard.bytes})
                del original,actual,item,batch
            assert not stream.read(1)
        assert count==pf.metadata.num_rows==receipt['physical_rows']==db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
        assert total==payload['bytes'] and ordered.hexdigest()==receipt['ordered_mapping_sha256']
    finally:
        db.close();pf.close();source_stream.close();del pf;gc.collect();pa.default_memory_pool().release_unused()
    verify_source(config,guard)
    target=folder/'validation.json'
    if target.exists():target=guard.run/'validation_successor.json'
    assert not target.exists()
    proof={'kind':'ASR18_all_original_FLAC_record_bytes_validation_v1','passed':True,
        'source_sha256':config['job']['sha256'],'carrier_receipt':guard.binding(receipt_path),
        'full_physical_rows':count,'full_original_FLAC_bytes_compared':total,
        'every_compressed_byte_and_physical_row_identical':True,
        'candidate_prep_exit_not_inferred_from_payload_complete_flag':True,
        'quality_or_language_exclusions':[],'auditor':guard.binding(__file__)}
    write_json(target,proof)
    return {'validation':guard.binding(target),'physical_rows':count,'original_compressed_bytes':total}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=metadata(args.config);guard=CarrierGuard(config);code=0
    try:
        result=prepare(config,guard) if config['mode']=='prepare' else audit(config,guard)
        assert not any(name in sys.modules for name in ('torch','soundfile','av','transformers'))
        write_json(guard.run/'result.json',{**result,'mode':config['mode'],'at_utc':now(),
            'PID':os.getpid(),'actual_paced_IO_bytes':guard.bytes,'actual_new_payload_bytes':guard.payload_written})
        print(json.dumps({'PID':os.getpid(),'mode':config['mode'],**result}),flush=True)
    except BaseException as error:
        code=75 if type(error).__name__=='DataStop' else 1
        write_json(guard.run/'failure.json',{'at_utc':now(),'error_class':type(error).__name__,
            'reason':str(error),'prefix_and_originals_preserved':True,'paced_IO_bytes':guard.bytes})
        raise
    finally:guard.close(code)


if __name__=='__main__':main()
