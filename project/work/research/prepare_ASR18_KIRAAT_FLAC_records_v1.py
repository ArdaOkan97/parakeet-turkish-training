"""One completed Parquet -> original compressed FLAC record carrier, E only.

No waveform decoding, labels, quality selection, model or GPU import. Each
physical row remains addressable; originals and data workers are untouched.
"""
import argparse,hashlib,json,os,shutil,sqlite3,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import BroadAudioReader,record,require
from atomic_io_retry_v1 import write_json

def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','CPU-only compressed media preparation')
    reader=BroadAudioReader();source,job=reader.verify_source(a.source);before=source.stat()
    require(source.suffix=='.parquet' and shutil.disk_usage(ROOT).free>100_000_000_000+job['size'],'Keep E100GB reserve plus bounded one-source payload')
    folder=ROOT/'cache/KIRAAT_original_FLAC_records_v1'/job['sha256'];folder.mkdir(parents=True,exist_ok=False)
    plan={'kind':'ASR18_original_FLAC_record_preparation_v1','source_path':str(source),'source_sha256':job['sha256'],'revision':job['revision'],
        'source_size':before.st_size,'source_mtime_ns':before.st_mtime_ns,'source_acquisition_receipt':record(source.with_suffix('.parquet.receipt.json')),
        'builder':record(__file__),'no_audio_decode_or_quality_filter':True,'physical_rows_not_catalog_ordinals':True}
    write_json(folder/'plan.json',plan)
    pf=pq.ParquetFile(source);database=folder/'rows.sqlite';db=sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE records(row_index INTEGER PRIMARY KEY,clip_id TEXT NOT NULL,source_id TEXT NOT NULL,offset INTEGER NOT NULL,bytes INTEGER NOT NULL,sha256 TEXT NOT NULL)')
    total=0;count=0;digest=hashlib.sha256();ordered=hashlib.sha256()
    try:
        with (folder/'records.bin').open('xb') as stream:
            for batch in pf.iter_batches(batch_size=1,columns=['audio','id','source_id'],use_threads=False):
                item=batch.to_pylist()[0];media=item['audio']['bytes']
                require(isinstance(media,bytes) and media.startswith(b'fLaC') and 0<len(media)<=32*1024**2,'Every physical source row must retain a full original FLAC record')
                require(shutil.disk_usage(ROOT).free>100_000_000_000+len(media),'Unchanged E100GB reserve')
                sha=hashlib.sha256(media).hexdigest();stream.write(media);digest.update(media)
                clip_id=str(item['id']);source_id=str(item['source_id'])
                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)',(count,clip_id,source_id,total,len(media),sha))
                ordered.update(json.dumps([count,clip_id,source_id,total,len(media),sha],separators=(',',':')).encode());total+=len(media);count+=1
                if count%128==0:db.commit()
                del media,item,batch
            stream.flush();os.fsync(stream.fileno())
        db.commit();require(count==pf.metadata.num_rows,'All physical Parquet rows must be copied without exclusion')
    finally:db.close();pf.close();del pf;pa.default_memory_pool().release_unused()
    after=source.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Completed original source changed during preparation')
    payload=folder/'records.bin';stat=payload.stat();index=database.stat()
    receipt={'kind':'ASR18_original_compressed_FLAC_records_v1','complete':True,'source':plan,'physical_rows':count,
        'records':{'path':str(payload),'bytes':total,'sha256':digest.hexdigest(),'mtime_ns':stat.st_mtime_ns},
        'index':{**record(database),'mtime_ns':index.st_mtime_ns},'ordered_mapping_sha256':ordered.hexdigest(),
        'full_original_FLAC_bytes_preserved':True,'decoded_PCM_pool_created':False,'quality_language_or_disagreement_exclusions':[],
        'source_files_modified':False,'TRAIN_admitted':False}
    write_json(folder/'receipt.json',receipt);require('torch' not in sys.modules and 'numpy' not in sys.modules,'No audio/model runtime import')
    print(json.dumps({'passed':True,'receipt':record(folder/'receipt.json'),'records_bytes':total,'physical_rows':count}),flush=True)

if __name__=='__main__':main()
