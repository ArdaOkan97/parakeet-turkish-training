"""One paced worker builds real accepted-audio references in bounded10k batches.

Original WAVs/carriers and staged YODAS archives remain read-only. New indexes,
features, manifests, ledgers and receipts stay on E:. No ASR/GPU/model loading.
"""
import array
import collections
import hashlib
import importlib.metadata
import io
import json
import math
import pathlib
import sqlite3
import struct
import sys
import time

# Reuse only the isolated E: CPU packages; the existing project Python supplies
# its already-installed PyArrow read API without installing or changing it.
ROOT=pathlib.Path(r'E:\gpu_training_scratchpad').resolve()
sys.path.insert(0,str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import numpy as np
import psutil
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced
from cpu_preparation_guard_v2 import PROJECT,append_json,now,scoped,sha256,write_json
from check_alania_pilot_audio_exclusion_cpu_v2 import features,retrieve
from run_paced_CPU_validation_v3 import wav_read

AREA=ROOT/'analysis/CPU_validation_v4'
base.AREA=AREA;paced.AREA=AREA
CARRIER=PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1'
COL=struct.Struct('<IIdB')
GROUPS=('fleurs','cv','tsc','worldspeech')
BATCH=10000
MAX_REFERENCES=352768

class Guard(paced.PacedGuard):
    def __init__(self):
        base.Guard.__init__(self,'accepted_audio_reference_v4')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1024*1024
        self.status(read_rate_limit_bytes_s=self.io_rate,read_block_max_bytes=65536,maximum_audio_workers=1,
            dynamic_pause_steps_mod250=[0,230,240],max_own_RSS_bytes=2_000_000_000,
            initial_reference_batch_limit=BATCH,global_reference_source_clips=MAX_REFERENCES,ASR_or_GPU_model_loaded=False)

    def status(self,**fields):
        if hasattr(self,'file_bytes_read'):fields['aggregate_paced_file_bytes_read']=self.file_bytes_read
        base.Guard.status(self,**fields)

class PacedFile(io.RawIOBase):
    """Exact-length file-like reads for projected Parquet metadata columns."""
    def __init__(self,path,guard):self.file=pathlib.Path(path).open('rb');self.guard=guard
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.file.tell()
    def seek(self,offset,whence=0):return self.file.seek(offset,whence)
    def read(self,size=-1):
        if size<0:size=self.file.seek(0,2)-self.file.tell()
        out=bytearray()
        while size>0:
            chunk=min(65536,size);self.guard.pace(chunk);data=self.file.read(chunk)
            if not data:break
            out.extend(data);size-=len(data)
            if len(data)<chunk:break
        return bytes(out)
    def close(self):
        if not self.closed:self.file.close()
        super().close()

def carrier_row(ordinal,guard):
    with (CARRIER/'offsets.u64').open('rb') as f:
        guard.pace(16);f.seek(ordinal*8);start,end=struct.unpack('<QQ',f.read(16))
    if not 0<=start<end<=732_365_239 or end-start>100000:raise ValueError('Immutable carrier extent differs')
    with (CARRIER/'rows.jsonl').open('rb') as f:
        guard.pace(end-start);f.seek(start);raw=f.read(end-start)
    row=json.loads(raw)
    if row['language']!='tr' or row.get('training_role','train')!='train':raise ValueError('Unexpected accepted role/language')
    return row

def prepare_order(guard):
    path=AREA/'accepted_reference_order_v4.u32';planpath=AREA/'accepted_reference_plan_v4.json'
    proof=json.loads((CARRIER/'index_provenance.json').read_text(encoding='utf-8'))
    if proof['rows']!=MAX_REFERENCES or not proof['complete']:raise ValueError('Wrong immutable accepted pool')
    for name,record in proof['files'].items():
        if (CARRIER/name).stat().st_size!=record['bytes']:raise ValueError('Immutable carrier file size changed')
    if path.exists():
        plan=json.loads(planpath.read_text(encoding='utf-8'))
        if plan['carrier_fingerprint']!=proof['fingerprint'] or sha256(path)!=plan['order_sha256']:raise ValueError('Reference resume plan differs')
        return np.frombuffer(guard.read_file(path,MAX_REFERENCES*4),dtype='<u4').copy(),plan
    raw=guard.read_file(CARRIER/'columns.bin',proof['files']['columns.bin']['bytes'])
    if hashlib.sha256(raw).hexdigest()!=proof['files']['columns.bin']['sha256']:raise ValueError('Accepted compact columns changed')
    dtype=np.dtype([('frames','<u4'),('tokens','<u4'),('duration','<f8'),('group','u1')]);cols=np.frombuffer(raw,dtype=dtype)
    quotas={g:int(math.floor(BATCH*int(np.count_nonzero(cols['group']==g))/MAX_REFERENCES)) for g in range(4)}
    leftovers=BATCH-sum(quotas.values())
    ranked=sorted(range(4),key=lambda g:(BATCH*np.count_nonzero(cols['group']==g)/MAX_REFERENCES-quotas[g]),reverse=True)
    for g in ranked[:leftovers]:quotas[g]+=1
    initial=[]
    for g in range(4):
        ordinals=np.flatnonzero(cols['group']==g);positions=np.linspace(0,len(ordinals)-1,quotas[g],dtype=np.int64)
        initial.extend(int(x) for x in ordinals[positions])
    priority=json.loads((ROOT/'analysis/CPU_validation_v3/WorldSpeech_overlap_bindings_v3.json').read_text(encoding='utf-8'))['recordings']
    catalog=sqlite3.connect('file:'+(CARRIER/'catalog.sqlite').as_posix()+'?mode=ro&immutable=1',uri=True);catalog.execute('PRAGMA cache_size=-1024')
    priorities=[]
    for entry in priority:
        sid=entry['accepted_rows'][0]['source_id'];guard.pace(16384)
        priorities.append(int(catalog.execute('SELECT ordinal FROM catalog WHERE source_id=?',(sid,)).fetchone()[0]))
    catalog.close()
    for ordinal in priorities:
        if ordinal not in initial:
            index=next(i for i in range(len(initial)-1,-1,-1) if cols['group'][initial[i]]==3 and initial[i] not in priorities)
            initial[index]=ordinal
    initial=priorities+[i for i in initial if i not in priorities]
    if len(initial)!=BATCH or len(set(initial))!=BATCH:raise ValueError('Initial10k order malformed')
    # Stable ordinal rank is tied to the immutable carrier fingerprint. This is
    # reference retrieval coverage only; it never changes training draws/order.
    selected=set(initial);remaining=[i for i in range(MAX_REFERENCES) if i not in selected]
    remaining.sort(key=lambda i:hashlib.blake2b((proof['fingerprint']+'|audio-reference-v4|'+str(i)).encode(),digest_size=8).digest())
    order=np.asarray(initial+remaining,dtype='<u4');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(order.tobytes())
    plan={'kind':'real_accepted_audio_reference_batched_plan_v4','created_at_utc':now(),'carrier':str(CARRIER),
        'carrier_provenance_sha256':sha256(CARRIER/'index_provenance.json'),'carrier_fingerprint':proof['fingerprint'],
        'accepted_pool_clips':MAX_REFERENCES,'accepted_pool_hours':proof['hours'],'initial_batch_clips':BATCH,'each_subsequent_batch_max_clips':BATCH,
        'bounded_batches_total':math.ceil(MAX_REFERENCES/BATCH),'initial_source_group_counts':{GROUPS[g]:quotas[g] for g in range(4)},
        'initial_reference_hours':float(sum(cols['duration'][i] for i in initial))/3600,'initial_priority_WorldSpeech_sources':3,
        'order':str(path),'order_sha256':sha256(path),'order_policy':'First10k evenly spaced proportional source strata,3 priority overlaps first; remainder deterministic immutable ordinal hash order',
        'autonomous_continuation':'Complete at most10k per durable batch, compare cached candidate features, then next batch automatically under unchanged gates',
        'existing_accepted_WAV_bytes_read_floor_estimate':proof['hours']*3600*32000,'new_index_budget_bytes':4_000_000_000,
        'audio_IO_limit_bytes_s':1048576,'maximum_audio_workers':1,'max_own_RSS_bytes':2_000_000_000,
        'all_1050_heldout_references_preserved':True,'global_audio_duplicate_exclusion_complete':False,'accepted_TRAIN_hours':0}
    write_json(planpath,plan);return order,plan

def provenance_metadata(db,guard):
    if db.execute('SELECT COUNT(*) FROM metadata_shards WHERE complete=1').fetchone()[0]==11:return
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    shards=['0027','0006','0014','0012','0056','0023','0030','0029','0002','0021','0054']
    wanted=['source_url','url','webpage_url','youtube_id','uploader','channel','creator','title','license','license_url','uploader_license']
    for shard in shards:
        if db.execute('SELECT complete FROM metadata_shards WHERE shard=?',(shard,)).fetchone():continue
        guard.wait();path=ROOT/f'datasets/yodas3/metadata/{shard}.parquet';receipt=json.loads(path.with_suffix('.parquet.receipt.json').read_text(encoding='utf-8'))
        if receipt['revision']!='81d802f45a1c1e1f5331260484dacf1fbea81981' or path.stat().st_size!=receipt['bytes']:raise ValueError('Pinned source metadata binding differs')
        with PacedFile(path,guard) as stream:
            parquet=pq.ParquetFile(stream);names=parquet.schema_arrow.names;extra=[x for x in wanted if x in names]
            columns=[x for x in ['id','shard','length','transcript_lang','locale_lang'] if x in names]+extra;count=0
            for batch in parquet.iter_batches(batch_size=128,columns=columns,use_threads=False):
                guard.wait()
                for r in batch.to_pylist():
                    rid=r['id'];available={x:r.get(x) for x in extra}
                    record={'recording_id':rid,'shard':shard,'pinned_metadata_path':str(path),'pinned_metadata_SHA256':receipt['sha256'],
                        'repository':'espnet/yodas3','revision':receipt['revision'],'published_recording_fields':list(r),
                        'length_s':r.get('length'),'transcript_lang':r.get('transcript_lang'),'locale_lang':r.get('locale_lang'),
                        'publisher_dataset_source':'https://huggingface.co/datasets/espnet/yodas3','publisher_declared_audio_license':'CC BY3.0',
                        'original_recording_attribution_fields':available,'original_creator_title_URL_license_proof_established':False,
                        'commercial_underlying_rights_verified':False,'attribution_gap':'Original creator,title,video URL and dated uploader license not present in inspected pinned public metadata fields' if not extra else 'Available fields require original-source rights validation'}
                    db.execute('INSERT OR REPLACE INTO source_provenance VALUES(?,?,?)',(rid,shard,json.dumps(record,ensure_ascii=False)));count+=1
                db.commit()
                guard.status(phase='per_recording_pinned_source_metadata_evidence',active_shard=shard,recordings_in_shard=count,
                    total_source_recordings_bound=db.execute('SELECT COUNT(*) FROM source_provenance').fetchone()[0])
        db.execute('INSERT INTO metadata_shards VALUES(?,?,?,1)',(shard,count,json.dumps({'schema':names,'attribution_fields_present':extra,'metadata_receipt':receipt},ensure_ascii=False)));db.commit()
    rows=db.execute('SELECT shard,recordings,details FROM metadata_shards ORDER BY shard').fetchall()
    write_json(AREA/'per_recording_source_provenance_report_v4.json',{'kind':'actual_pinned_staged_recording_attribution_evidence_v4','completed_at_utc':now(),
        'staged_shards_bound':len(rows),'recordings_bound':sum(x[1] for x in rows),'per_shard':[{'shard':x[0],'recordings':x[1],**json.loads(x[2])} for x in rows],
        'per_recording_evidence_index':str(ROOT/'indexes/accepted_audio_reference_v4.sqlite'),'dataset_CC_BY3_claim_verified':True,
        'per_recording_original_attribution_or_underlying_commercial_rights_established':False,
        'interpretation':'Recording-level pinned origin/shard/metadata evidence is now explicit; absent original attribution/license fields cannot be filled by the annotation license.',
        'PyArrow_version_reused_read_only':pa.__version__,'accepted_TRAIN_hours':0})

def index_reference(db,ordinal,guard):
    old=db.execute('SELECT status FROM references_audio WHERE ordinal=?',(ordinal,)).fetchone()
    if old:return
    row=carrier_row(ordinal,guard);sid=row['source_id'];path=pathlib.Path(row['audio_filepath']).resolve();before=time.monotonic()
    record={'source_id':sid,'ordinal':ordinal,'dataset':row['dataset'],'duration_s':row['duration'],'completed_at_utc':None}
    try:
        if not (path.is_relative_to(PROJECT) or path.is_relative_to(ROOT)):raise ValueError('Accepted audio outside authorized project/E: roots')
        if path.stat().st_size>4*1024*1024:raise ValueError('Accepted reference WAV exceeds declared4MiB per-clip read bound')
        pcm,digest=wav_read(row,guard);guard.wait();vec,pos=features(pcm);del pcm
        status='actual_WAV_PCM_verified';record.update(status=status,PCM_i16_sha256=digest,feature_frames=len(vec),
            current_spectral_retrieval_usable=len(vec)>=8,WAV_sha256_verified=row.get('wav_sha256'),canonical_PCM_expected=row.get('canonical_pcm_sha256'))
        blob=vec.astype('<f4').tobytes();posblob=pos.astype('<i4').tobytes();error=None
    except Exception as e:
        digest=None;blob=None;posblob=None;status='reference_unavailable_or_integrity_error';error=repr(e)
        record.update(status=status,error=error,feature_frames=0,current_spectral_retrieval_usable=False)
    record.update(completed_at_utc=now(),elapsed_s=time.monotonic()-before)
    guard.wait();db.execute('INSERT INTO references_audio VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
        (sid,ordinal,row['dataset'],row['duration'],status,digest,blob,posblob,record['feature_frames'],int(record['current_spectral_retrieval_usable']),error,json.dumps(record,ensure_ascii=False)));db.commit()
    append_json(AREA/'accepted_audio_reference_ledger_v4.jsonl',record)

def cache_candidates(db,guard):
    if db.execute('SELECT COUNT(*) FROM candidate_features').fetchone()[0]>=193:return
    candidates=[json.loads(x) for x in (ROOT/'analysis/CPU_validation_v3/new8_shards_QC_manifest_v3.jsonl').read_text(encoding='utf-8').splitlines()]
    priorities=json.loads((ROOT/'analysis/CPU_validation_v3/WorldSpeech_overlap_bindings_v3.json').read_text(encoding='utf-8'))['recordings']
    local=next(x for x in priorities if x['candidate_archive_local']);candidates.insert(0,local)
    metadata=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True)
    for r in candidates:
        sid=r['segment_id']
        if db.execute('SELECT 1 FROM candidate_features WHERE segment_id=?',(sid,)).fetchone():continue
        if 'start_s' not in r:r['start_s']=metadata.execute('SELECT start_s FROM candidates WHERE segment_id=?',(sid,)).fetchone()[0]
        guard.wait()
        try:
            pcm,info=paced.decode_paced(r,guard);canonical=np.clip(np.rint(pcm*32768),-32768,32767).astype('<i2')
            digest=hashlib.sha256(canonical.tobytes()).hexdigest();del canonical;vec,pos=features(pcm);del pcm
            blob=vec.astype('<f4').tobytes();posblob=pos.astype('<i4').tobytes();status='actual_candidate_fingerprint_cached';error=None
        except Exception as e:digest=None;blob=None;posblob=None;vec=[];status='candidate_unavailable_under_unchanged_bound';error=repr(e)
        record={'segment_id':sid,'recording_id':r['recording_id'],'shard':r['shard'],'duration_s':r['duration_s'],'status':status,'error':error,'completed_at_utc':now()}
        guard.wait();db.execute('INSERT INTO candidate_features VALUES(?,?,?,?,?,?,?,?,?)',(sid,r['recording_id'],r['duration_s'],status,digest,blob,posblob,len(vec),json.dumps(record)));db.commit()
    metadata.close()

def compare_batch(db,ordinals,batch_id,guard):
    if db.execute('SELECT 1 FROM comparisons_complete WHERE batch_id=?',(batch_id,)).fetchone():return
    rows=[]
    for ordinal in ordinals:
        row=db.execute("SELECT source_id,PCM_i16_sha256,vectors,positions,frames FROM references_audio WHERE ordinal=? AND status='actual_WAV_PCM_verified'",(int(ordinal),)).fetchone()
        if row and row[4]>=8:rows.append(row)
    if not rows:return
    allvectors=[];refids=[];refpos=[];sourceids=[];lengths=[]
    for i,(sid,digest,blob,posblob,n) in enumerate(rows):
        guard.pace(len(blob)+len(posblob));sourceids.append(sid);lengths.append(n);allvectors.append(np.frombuffer(blob,dtype='<f4').reshape(n,32));refids.extend([i]*n);refpos.extend(np.frombuffer(posblob,dtype='<i4').tolist())
    bank=np.concatenate(allvectors);ids=np.asarray(refids,np.int32);positions=np.asarray(refpos,np.int32);checked=flags=0
    for sid,digest,blob,posblob,n in db.execute("SELECT segment_id,PCM_i16_sha256,vectors,positions,frames FROM candidate_features WHERE status='actual_candidate_fingerprint_cached'"):
        if db.execute('SELECT 1 FROM candidate_batch_checks WHERE segment_id=? AND batch_id=?',(sid,batch_id)).fetchone():continue
        guard.wait();q=np.frombuffer(blob,dtype='<f4').reshape(n,32);p=np.frombuffer(posblob,dtype='<i4')
        matches=retrieve(q,p,bank,ids,positions,sourceids,lengths)
        exact=[x[0] for x in db.execute('SELECT source_id FROM references_audio WHERE PCM_i16_sha256=?',(digest,))]
        result={'candidate_segment_id':sid,'reference_batch_id':batch_id,'reference_spectral_usable_clips':len(rows),'exact_accepted_PCM_matches':exact,
            'conservative_spectral_triage_matches':matches,'completed_at_utc':now(),'accepted_TRAIN':False,
            'interpretation':'No match does not prove global uniqueness; spectral matches are review triage only'}
        guard.wait();db.execute('INSERT INTO candidate_batch_checks VALUES(?,?,?)',(sid,batch_id,json.dumps(result,ensure_ascii=False)));db.commit();checked+=1
        if matches or exact:flags+=1;append_json(AREA/'candidate_accepted_audio_overlap_triage_v4.jsonl',result)
        if checked%10==0:guard.status(phase='paced_candidate_vs_real_accepted_audio_bank',reference_batch=batch_id,candidates_checked_in_resume=checked,triage_candidates_in_resume=flags)
    db.execute('INSERT INTO comparisons_complete VALUES(?)',(batch_id,));db.commit();del bank,allvectors,ids,positions

def summary(db,guard,phase,batch_id):
    total,verified,seconds,usable=db.execute("SELECT COUNT(*),SUM(status='actual_WAV_PCM_verified'),SUM(CASE WHEN status='actual_WAV_PCM_verified' THEN duration_s ELSE 0 END),SUM(spectral_usable) FROM references_audio").fetchone()
    report={'kind':'real_accepted_audio_reference_coverage_v4','updated_at_utc':now(),'phase':phase,'current_reference_batch_id':batch_id,
        'accepted_pool_clips':MAX_REFERENCES,'accepted_pool_hours':675.0964019097223,'reference_rows_processed':total,
        'actual_WAV_PCM_verified_reference_clips':verified or 0,'actual_audio_reference_hours':(seconds or 0)/3600,
        'spectral_reference_clips_usable_under_current_min8frames':usable or 0,'unavailable_or_integrity_error_reference_rows':total-(verified or 0),
        'accepted_reference_clip_coverage_fraction':(verified or 0)/MAX_REFERENCES,'full_accepted_audio_reference_coverage':verified==MAX_REFERENCES,
        'source_group_verified_counts':dict(db.execute("SELECT dataset,COUNT(*) FROM references_audio WHERE status='actual_WAV_PCM_verified' GROUP BY dataset")),
        'candidate_fingerprints_cached':db.execute("SELECT COUNT(*) FROM candidate_features WHERE status='actual_candidate_fingerprint_cached'").fetchone()[0],
        'completed_candidate_reference_batch_checks':db.execute('SELECT COUNT(*) FROM candidate_batch_checks').fetchone()[0],
        'per_recording_staged_source_metadata_evidence_rows':db.execute('SELECT COUNT(*) FROM source_provenance').fetchone()[0],
        'aggregate_paced_read_bytes':guard.file_bytes_read,'CPU_RSS_bytes_observed':psutil.Process().memory_info().rss,
        'accepted_TRAIN_hours':0,'all1050heldout_bank_preserved':True,'GPU_used':False,'TEST_predictions_read':False,
        'commercial_underlying_rights_established':False,'global_perceptual_audio_duplicate_exclusion_complete':False,
        'limitations':['Coverage counts actual canonical accepted WAV/PCM reads, not encoded-byte or text comparisons.',
            'Short/silent clips with fewer than8 usable feature frames have exact hashes but no current conservative spectral retrieval coverage.',
            'No hash/fingerprint match is not proof of uniqueness; bank coverage remains partial until every reference is physically verified.',
            'No automatic candidate admission or transcript correction;17text-overlap clips/3longmatches remain quarantined.']}
    write_json(AREA/'accepted_audio_reference_coverage_v4.json',report)
    guard.status(phase=phase,reference_batch=batch_id,reference_rows_processed=total,actual_audio_references_verified=verified or 0,
        actual_audio_reference_hours=(seconds or 0)/3600,spectral_usable_references=usable or 0,accepted_TRAIN_hours=0)
    return report

def main():
    guard=Guard();guard.wait();order,plan=prepare_order(guard)
    path=ROOT/'indexes/accepted_audio_reference_v4.sqlite';db=sqlite3.connect(path);db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=MEMORY')
    db.executescript('''CREATE TABLE IF NOT EXISTS references_audio(source_id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,dataset TEXT,duration_s REAL,status TEXT,PCM_i16_sha256 TEXT,vectors BLOB,positions BLOB,frames INTEGER,spectral_usable INTEGER,error TEXT,details TEXT);
CREATE INDEX IF NOT EXISTS accepted_PCM_hash ON references_audio(PCM_i16_sha256);
CREATE TABLE IF NOT EXISTS source_provenance(recording_id TEXT PRIMARY KEY,shard TEXT,details TEXT);
CREATE TABLE IF NOT EXISTS metadata_shards(shard TEXT PRIMARY KEY,recordings INTEGER,details TEXT,complete INTEGER);
CREATE TABLE IF NOT EXISTS candidate_features(segment_id TEXT PRIMARY KEY,recording_id TEXT,duration_s REAL,status TEXT,PCM_i16_sha256 TEXT,vectors BLOB,positions BLOB,frames INTEGER,details TEXT);
CREATE TABLE IF NOT EXISTS candidate_batch_checks(segment_id TEXT,batch_id INTEGER,details TEXT,PRIMARY KEY(segment_id,batch_id));
CREATE TABLE IF NOT EXISTS comparisons_complete(batch_id INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS reference_batches(batch_id INTEGER PRIMARY KEY,details TEXT);''')
    for ordinal in order[:3]:index_reference(db,int(ordinal),guard);summary(db,guard,'first_priority_real_audio_reference_saved',0)
    provenance_metadata(db,guard)
    for batch_id,start in enumerate(range(0,len(order),BATCH)):
        ordinals=order[start:start+BATCH]
        for i,ordinal in enumerate(ordinals,1):
            index_reference(db,int(ordinal),guard)
            if i%10==0:summary(db,guard,'paced_real_accepted_audio_reference_indexing',batch_id)
            if path.stat().st_size>4_000_000_000:raise RuntimeError('Declared4GB new reference-index budget exceeded; preserve progress')
        report=summary(db,guard,'bounded_reference_batch_audio_reads_complete',batch_id)
        db.execute('INSERT OR REPLACE INTO reference_batches VALUES(?,?)',(batch_id,json.dumps(report)));db.commit()
        cache_candidates(db,guard);compare_batch(db,ordinals,batch_id,guard)
        report=summary(db,guard,'bounded_reference_batch_and_duplicate_triage_complete',batch_id)
        write_json(AREA/f'reference_batch_{batch_id:04d}_receipt_v4.json',report)
    summary(db,guard,'completed_all_declared_accepted_reference_batches',math.ceil(MAX_REFERENCES/BATCH)-1)
    db.close();guard.status(phase='completed_all_declared_accepted_reference_batches',actual_CLOSED0=True)

if __name__=='__main__':main()
