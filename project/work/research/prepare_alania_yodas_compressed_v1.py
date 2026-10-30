"""Pinned CPU-only Alania/YODAS staging. All new storage is on designated E:.

Downloads Alania metadata, screens it into a disk-backed candidate index, then
stages a bounded number of original compressed YODAS shards. It never imports
Torch, runs inference, reads TEST, admits TRAIN data, extracts WAVs or transcodes.
"""
import argparse, collections, concurrent.futures, datetime, hashlib, json
import pathlib, re, shutil, sqlite3, sys, threading, time, urllib.request

PROJECT = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(r'E:\gpu_training_scratchpad').resolve()
ALANIA = 'cloud0day3/alania-speech-captions-tr'
ALANIA_REV = '8b0fdd976ed9668b64f5ee0696317a8f241ce39c'
YODAS = 'espnet/yodas3'
YODAS_REV = '81d802f45a1c1e1f5331260484dacf1fbea81981'
STOP_FREE_BYTES = 100_000_000_000
MAX_STAGED_BYTES = 30_000_000_000
lock = threading.Lock()
STATE = {}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def output_path(path):
    p = pathlib.Path(path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('Output escapes designated E: root')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path, value):
    p = output_path(path)
    temporary = p.with_suffix(p.suffix+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(p)

def status(**fields):
    with lock:
        STATE.update(fields); STATE['updated_at_utc'] = now()
        write_json(ROOT/'analysis/alania_yodas_preparation_v1/status.json',STATE)
        print(json.dumps(fields,ensure_ascii=False),flush=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def get_json(url):
    with urllib.request.urlopen(url,timeout=60) as response: return json.load(response)

def download(repo, revision, record, destination):
    path=output_path(destination); size=record['size']
    expected_hash=record.get('lfs',{}).get('oid')
    receipt=path.with_suffix(path.suffix+'.receipt.json')
    if path.exists():
        if path.stat().st_size!=size: raise ValueError('Existing complete file has unexpected size: '+str(path))
        observed=digest(path)
        if expected_hash and observed!=expected_hash: raise ValueError('Existing file SHA differs: '+str(path))
        return {'path':str(path),'bytes':size,'sha256':observed,'repository':repo,'revision':revision,'cached':True}
    partial=path.with_suffix(path.suffix+'.part')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size: raise ValueError('Partial file larger than pinned source')
    if shutil.disk_usage(ROOT).free-size+offset<STOP_FREE_BYTES: raise RuntimeError('E: free-space reserve would be exceeded')
    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{record["path"]}'
    attempts=0
    while offset<size:
        attempts+=1
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'ASR-metadata-preparation/1','Range':f'bytes={offset}-'} if offset else {'User-Agent':'ASR-metadata-preparation/1'})
            with urllib.request.urlopen(request,timeout=90) as response:
                if offset and response.status!=206: raise RuntimeError('Server did not honor partial resume; preserving partial')
                if offset and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'): raise RuntimeError('Content range mismatch')
                last=time.monotonic()
                with partial.open('ab' if offset else 'wb') as output:
                    while True:
                        block=response.read(4*1024*1024)
                        if not block: break
                        if offset+len(block)>size: raise RuntimeError('Source exceeds pinned length')
                        output.write(block);offset+=len(block)
                        if time.monotonic()-last>20:
                            output.flush()
                            status(phase='downloading_compressed_audio' if record['path'].endswith('.tar') else 'downloading_metadata',active_file=record['path'],active_file_downloaded_bytes=offset,active_file_expected_bytes=size)
                            last=time.monotonic()
            if offset!=size: raise OSError('Incomplete response')
        except (OSError,TimeoutError) as e:
            if attempts>=3: raise
            status(last_download_error=str(e),retry=attempts,partial_preserved=True)
    observed=digest(partial)
    if expected_hash and observed!=expected_hash: raise RuntimeError('Pinned LFS SHA-256 mismatch; partial retained')
    partial.replace(path)
    value={'path':str(path),'bytes':size,'sha256':observed,'repository':repo,'revision':revision,'source_path':record['path'],'source_url':url,'completed_at_utc':now()}
    write_json(receipt,value)
    return value

def main(audio_shards):
    policy=json.loads((ROOT/'storage_config.json').read_text(encoding='utf-8-sig'))
    assert pathlib.Path(policy['storage_root']).resolve()==ROOT
    assert not policy['allow_fallback_outside_root']
    assert pathlib.Path(__import__('os').environ['TEMP']).resolve().is_relative_to(ROOT)
    status(kind='Alania_YODAS_compressed_preparation_v1',started_at_utc=now(),phase='pinning_metadata',TRAIN_admitted=False,GPU_used=False,TEST_read=False,audio_transcoded=False,permanent_WAV_copy=False,requested_audio_shards=audio_shards)
    source_inventory=PROJECT/'work/research/Turkish_scale_research_v1/cloud0day3_alania-speech-captions-tr_pinned_inventory.json'
    inventory=json.loads(source_inventory.read_text(encoding='utf-8'))
    assert inventory['revision']==ALANIA_REV
    write_json(ROOT/'datasets/alania/metadata/source_inventory.json',inventory)
    records=[r for r in inventory['files'] if r['type']=='file' and (r['path'].endswith('.parquet') or r['path'] in ['README.md','LICENSE.md'])]
    status(metadata_files_expected=len(records),metadata_expected_bytes=sum(r['size'] for r in records))
    def fetch(record): return download(ALANIA,ALANIA_REV,record,ROOT/'datasets/alania/metadata'/record['path'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: downloads=list(pool.map(fetch,records))
    write_json(ROOT/'datasets/alania/metadata/download_receipt.json',{'completed_at_utc':now(),'repository':ALANIA,'revision':ALANIA_REV,'files':downloads})
    status(phase='indexing_metadata',metadata_files_completed=len(downloads),metadata_downloaded_bytes=sum(r['bytes'] for r in downloads))
    sys.path.extend([str(PROJECT/'work/training-env/Lib/site-packages'),r'C:\Users\ardao\Miniconda3\envs\vlm\Lib\site-packages'])
    import pyarrow.parquet as pq
    db_path=output_path(ROOT/'indexes/alania_metadata_candidates_v1.sqlite')
    db=sqlite3.connect(db_path); db.execute('PRAGMA journal_mode=TRUNCATE');db.execute('PRAGMA cache_size=-8192')
    db.execute('CREATE TABLE IF NOT EXISTS candidates (segment_id TEXT PRIMARY KEY, recording_id TEXT, shard TEXT, start_s REAL, end_s REAL, duration_s REAL, transcript TEXT, disagrees INTEGER, lid REAL, dnsmos REAL, noise TEXT, reverb TEXT, speaker TEXT, eligible INTEGER, reasons TEXT, pilot_rank INTEGER)')
    columns=['segment_id','yodas_id','yodas_shard','start','end','duration','transcript','asr_disagrees_with_youtube','turkish_lid_p','dnsmos_ovrl','noise_label','reverb_label','speaker']
    rows_seen=0
    for record in records:
        if not record['path'].endswith('.parquet'): continue
        file=ROOT/'datasets/alania/metadata'/record['path']
        parquet=pq.ParquetFile(file)
        if not set(columns).issubset(parquet.schema_arrow.names): raise ValueError('Pinned metadata schema differs')
        for batch in parquet.iter_batches(batch_size=512,columns=columns,use_threads=False):
            values=[]
            for row in batch.to_pylist():
                sid,video,shard=row['segment_id'],row['yodas_id'],row['yodas_shard']
                if not re.fullmatch(r'[a-f0-9]{32}',video) or not re.fullmatch(r'\d{4}',shard): raise ValueError('Unsafe source identity')
                start,end,duration=row['start'],row['end'],row['duration'];text=row['transcript'] or ''
                reasons=[]
                if not 2<=duration<=20: reasons.append('duration_outside_2_20s')
                if start<0 or end<=start or abs(end-start-duration)>.1: reasons.append('invalid_span')
                if not text.strip(): reasons.append('empty_transcript')
                if row['asr_disagrees_with_youtube'] is not False: reasons.append('ASR_caption_disagreement_or_unknown')
                if row['turkish_lid_p'] is None or row['turkish_lid_p']<.95: reasons.append('LID_below_0.95_or_unknown')
                if row['dnsmos_ovrl'] is None or row['dnsmos_ovrl']<2.8: reasons.append('DNSMOS_below_2.8_or_unknown')
                rank=int.from_bytes(hashlib.sha256(('pilot20261015|'+sid).encode()).digest()[:7],'big')
                values.append((sid,video,shard,start,end,duration,text,int(bool(row['asr_disagrees_with_youtube'])),row['turkish_lid_p'],row['dnsmos_ovrl'],row['noise_label'],row['reverb_label'],row['speaker'],int(not reasons),json.dumps(reasons),rank))
            db.executemany('INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',values)
            rows_seen+=len(values)
        db.commit()
        status(metadata_rows_indexed=rows_seen,last_metadata_file=record['path'])
    total,total_seconds=db.execute('SELECT COUNT(*),SUM(duration_s) FROM candidates').fetchone()
    eligible,eligible_seconds=db.execute('SELECT COUNT(*),SUM(duration_s) FROM candidates WHERE eligible=1').fetchone()
    assert total==rows_seen==835267,'Pinned dataset row count differs; no training admission'
    db.execute('CREATE INDEX IF NOT EXISTS candidate_shard ON candidates(shard,eligible,pilot_rank)');db.commit()
    per_shard=[{'shard':s,'candidate_clips':n,'candidate_hours':seconds/3600} for s,n,seconds in db.execute('SELECT shard,COUNT(*),SUM(duration_s) FROM candidates WHERE eligible=1 GROUP BY shard ORDER BY SUM(duration_s) DESC')]
    audio_inventory=get_json(f'https://huggingface.co/api/datasets/{YODAS}/tree/{YODAS_REV}/data/tr/audio?recursive=false&limit=100')
    write_json(ROOT/'datasets/yodas3/source_audio_inventory.json',{'repository':YODAS,'revision':YODAS_REV,'files':audio_inventory})
    audio_by_shard={pathlib.PurePosixPath(f['path']).stem:f for f in audio_inventory if f['type']=='file' and f['path'].endswith('.tar')}
    # Favor many usable metadata-screened minutes per downloaded byte. This is
    # staging order only, never a final TRAIN sampling policy or quality proof.
    per_shard.sort(key=lambda s:s['candidate_hours']/audio_by_shard[s['shard']]['size'],reverse=True)
    selected=per_shard[:audio_shards]
    if sum(audio_by_shard[s['shard']]['size'] for s in selected)>MAX_STAGED_BYTES: raise RuntimeError('Initial compressed download budget exceeded')
    report={'created_at_utc':now(),'repository':ALANIA,'revision':ALANIA_REV,'source_rows':total,'source_segment_hours':total_seconds/3600,'metadata_screened_candidate_rows':eligible,'metadata_screened_candidate_hours':eligible_seconds/3600,'thresholds':{'duration_seconds':[2,20],'turkish_lid_p_min':.95,'dnsmos_ovrl_min':2.8,'ASR_caption_disagreement':False},'status':'METADATA_CANDIDATES_ONLY_NOT_ACCEPTED_TRAIN','limitations':['Source machine transcripts and labels are not independently verified.','Audio/time validation, cross-source dedup, frozen held-out exclusion and listening audit remain pending.','Speaker IDs are per recording; global speaker uniqueness is not established.','Source YODAS3 uses original compressed Opus-in-WebM. No re-encoding or full WAV extraction.'],'index_path':str(db_path),'per_shard':per_shard,'initial_staged_shards':selected,'max_initial_compressed_download_bytes':MAX_STAGED_BYTES,'minimum_free_bytes':STOP_FREE_BYTES,'Torch_imported':'torch' in sys.modules,'TEST_read':False,'GPU_used':False,'TRAIN_admitted':False}
    write_json(ROOT/'analysis/alania_yodas_preparation_v1/metadata_review.json',report)
    status(phase='metadata_complete',source_rows=total,source_segment_hours=total_seconds/3600,metadata_screened_candidate_rows=eligible,metadata_screened_candidate_hours=eligible_seconds/3600,initial_staged_shards=[s['shard'] for s in selected],accepted_TRAIN_hours=0)
    audio_receipts=[]
    for selected_shard in selected:
        shard=selected_shard['shard'];record=audio_by_shard[shard]
        meta=get_json(f'https://huggingface.co/api/datasets/{YODAS}/tree/{YODAS_REV}/data/tr/metadata?recursive=false&limit=100')
        md=next(x for x in meta if x['path']==f'data/tr/metadata/{shard}.parquet')
        download(YODAS,YODAS_REV,md,ROOT/f'datasets/yodas3/metadata/{shard}.parquet')
        status(phase='downloading_compressed_audio',active_file=record['path'],active_file_downloaded_bytes=0,active_file_expected_bytes=record['size'])
        audio_receipts.append(download(YODAS,YODAS_REV,record,ROOT/f'datasets/yodas3/audio/{shard}.tar'))
        write_json(ROOT/'datasets/yodas3/compressed_download_receipt.json',{'created_at_utc':now(),'files':audio_receipts,'no_transcode':True,'permanent_WAV_copy':False})
    db.close()
    status(phase='compressed_staging_complete',compressed_shards_completed=len(audio_receipts),compressed_audio_bytes=sum(x['bytes'] for x in audio_receipts),metadata_ready=True,accepted_TRAIN_hours=0,on_demand_decoder_status='next_CPU_validation_no_training_admission')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audio-shards',type=int,default=1);args=parser.parse_args()
    if not 0<=args.audio_shards<=3: raise ValueError('Initial bounded staging supports0:3 shards')
    try: main(args.audio_shards)
    except Exception as error:
        status(phase='failed_preserved_partial',error=repr(error),TRAIN_admitted=False)
        raise
