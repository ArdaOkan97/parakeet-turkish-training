"""Independent ALL-row original-compressed-byte identity audit, E only."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
import sqlite3
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import BroadAudioReader,record,require
from atomic_io_retry_v1 import write_json

def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    source,job=BroadAudioReader().verify_source(a.source);folder=ROOT/'cache/KIRAAT_original_FLAC_records_v1'/job['sha256']
    receipt_path=folder/'receipt.json';receipt=json.loads(receipt_path.read_text());target=folder/'validation.json';require(not target.exists(),'Preserve first independent carrier validation')
    require(receipt['complete'] is True and receipt['source']['source_sha256']==job['sha256'],'Exact candidate source carrier')
    payload=receipt['records'];require(record(payload['path'])=={k:payload[k] for k in ('path','bytes','sha256')},'Whole original compressed record carrier hash')
    require(record(receipt['index']['path'])=={k:receipt['index'][k] for k in ('path','bytes','sha256')},'Whole row index hash')
    pf=pq.ParquetFile(source);db=sqlite3.connect(Path(receipt['index']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);count=0;total=0
    try:
        with Path(payload['path']).open('rb') as stream:
            for batch in pf.iter_batches(batch_size=1,columns=['audio','id','source_id'],use_threads=False):
                item=batch.to_pylist()[0];original=item['audio']['bytes']
                row=db.execute('SELECT clip_id,source_id,offset,bytes,sha256 FROM records WHERE row_index=?',(count,)).fetchone()
                require(row is not None and row[:2]==(str(item['id']),str(item['source_id'])) and row[2]==total,'ALL physical row IDs/extents/order must match')
                stream.seek(row[2]);actual=stream.read(row[3]);require(actual==original and hashlib.sha256(actual).hexdigest()==row[4],'ALL original FLAC bytes must be identical')
                count+=1;total+=len(original);del original,actual,item,batch
        require(count==pf.metadata.num_rows==receipt['physical_rows']==db.execute('SELECT COUNT(*) FROM records').fetchone()[0] and total==payload['bytes'],'Full physical row coverage; no exclusions')
    finally:db.close();pf.close();del pf;gc.collect();pa.default_memory_pool().release_unused()
    require('torch' not in sys.modules and 'soundfile' not in sys.modules and 'av' not in sys.modules,'No model/waveform decoding in byte audit')
    proof={'kind':'ASR18_all_original_FLAC_record_bytes_validation_v1','passed':True,'source_sha256':job['sha256'],'carrier_receipt':record(receipt_path),
        'full_physical_rows':count,'full_original_FLAC_bytes_compared':total,'every_compressed_byte_and_physical_row_identical':True,
        'original_prep_v1_exit1_dependency_assertion_preserved':True,'quality_or_language_exclusions':[],'auditor':record(__file__)}
    write_json(target,proof);print(json.dumps({'passed':True,'validation':record(target),'physical_rows':count,'original_compressed_bytes':total}),flush=True)

if __name__=='__main__':main()
