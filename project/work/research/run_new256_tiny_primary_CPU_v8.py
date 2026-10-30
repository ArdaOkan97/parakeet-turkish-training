"""One declared new256 batch; full-clip70% primary screen, windows audit only."""
import collections
import ctypes as c
import hashlib
import heapq
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import numpy as np
import psutil
import run_random_tiny_language_CPU_v7 as old
from whisper_cpp_cpu_bindings_v6 import Runtime,Loader,READ,EOF,CLOSE,LOG
from run_large_v3_sanity_cpu_v6 import hard_process_memory_job
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced

AREA=ROOT/'analysis/CPU_validation_v8'
base.AREA=AREA;paced.AREA=AREA
SEED=2026101608
BUDGET=1_000_000_000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readl(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]


class Guard(old.Guard):
    def __init__(self):
        base.Guard.__init__(self,'new256_tiny_primary_CPU_v8')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1048576
        self.status(CPU_threads=1,maximum_audio_workers=1,max_own_RSS_bytes=2_000_000_000,
                    read_rate_limit_bytes_s=1048576,language_threshold=.70,tiny_transcription_enabled=False,
                    batch_max_clips=256,window_audit_preselected_clips=26,maximum_paced_read_bytes=BUDGET,
                    primary_gate='full_clip_detected_tr_AND_Ptr_at_least_point70',windows_change_primary_decision=False)


def pin(guard):
    receipt=AREA/'new256_sampling_receipt_v8.json'
    if receipt.exists():
        value=json.loads(receipt.read_text());assert sha(value['manifest'])==value['manifest_sha256'];return value
    exclusions=set();exclusion_sources=[]
    for path in [ROOT/'analysis/CPU_validation_v2/independent_QC_ledger_v2.jsonl',
                 ROOT/'analysis/CPU_validation_v3/new8_shards_independent_QC_ledger_v3.jsonl',
                 ROOT/'analysis/CPU_validation_v7/tiny_LID_per_clip_ledger_v7.jsonl']:
        values=readl(path);exclusions.update(r['segment_id'] for r in values)
        exclusion_sources.append({'path':str(path),'rows':len(values),'sha256':sha(path)})
    shards=sorted(p.stem for p in (ROOT/'datasets/yodas3/audio').glob('*.tar'));assert len(shards)==11
    allowed=set()
    for shard in shards:
        db=sqlite3.connect('file:'+(ROOT/f'indexes/yodas3_tar_{shard}.sqlite').as_posix()+'?mode=ro',uri=True)
        for i,(rid,size) in enumerate(db.execute('SELECT recording_id,bytes FROM recordings')):
            if i%256==0:guard.pace(16384)
            if size<=256*1024*1024:allowed.add((shard,rid))
        db.close()
    buckets={(s,d):[] for s in shards for d in range(3)};counts=collections.Counter();scanned=0;bound_excluded=0;prior_excluded=0
    db=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA cache_size=-1024')
    query='SELECT segment_id,recording_id,shard,duration_s FROM candidates WHERE eligible=1 AND shard IN ('+','.join('?' for _ in shards)+')'
    for row in db.execute(query,shards):
        if scanned%128==0:guard.pace(65536)
        scanned+=1
        if (row['shard'],row['recording_id']) not in allowed:bound_excluded+=1;continue
        if row['segment_id'] in exclusions:prior_excluded+=1;continue
        key=(row['shard'],0 if row['duration_s']<5 else 1 if row['duration_s']<12 else 2);counts[key]+=1
        rank=int.from_bytes(hashlib.blake2b(f'{SEED}|{row["segment_id"]}'.encode(),digest_size=16).digest(),'big')
        item=(-rank,row['segment_id'])
        if len(buckets[key])<8:heapq.heappush(buckets[key],item)
        elif item>buckets[key][0]:heapq.heapreplace(buckets[key],item)
        if scanned%40000==0:guard.status(phase='pinning_new256_before_predictions',source_rows_scanned=scanned)
    keys=sorted(buckets);random.Random(SEED).shuffle(keys);selected=[];strata=[]
    for i,key in enumerate(keys):
        quota=8 if i<25 else 7;assert len(buckets[key])>=quota
        for rank,sid in sorted(buckets[key],key=lambda p:(-p[0],p[1]))[:quota]:
            row=dict(db.execute('SELECT * FROM candidates WHERE segment_id=?',(sid,)).fetchone())
            row.update(QC_group='new256_tiny_primary_v8',sampling_seed=SEED,
                       sampling_stratum={'shard':key[0],'duration_bin':['2-5s','5-12s','12-20s'][key[1]]})
            selected.append(row)
        strata.append({'shard':key[0],'duration_bin':['2-5s','5-12s','12-20s'][key[1]],'eligible_new_bounded_population':counts[key],'selected':quota})
    db.close();selected.sort(key=lambda r:r['segment_id']);assert len(selected)==256
    assert not (set(r['segment_id'] for r in selected)&exclusions)
    audit_ids=set(random.Random(SEED+26).sample([r['segment_id'] for r in selected],26))
    for row in selected:row['predeclared_window_audit']=row['segment_id'] in audit_ids
    manifest=AREA/'new256_tiny_primary_manifest_v8.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    value={'kind':'new256_preinference_seeded_manifest_v8','pinned_at_utc':base.now(),'seed':SEED,'audit_seed':SEED+26,
           'manifest':str(manifest),'manifest_sha256':sha(manifest),'clips':256,'audit_clips':26,'audit_fraction':26/256,
           'per_stratum':strata,'source_rows_scanned':scanned,'bound_excluded':bound_excluded,'prior_QC_excluded':prior_excluded,
           'all_prior_exclusion_ids':len(exclusions),'exclusion_sources':exclusion_sources,
           'new_bounded_population':sum(counts.values()),'source_labels_or_old_small_flags_used_for_ranking':False,
           'all_ids_and_26audit_ids_pinned_before_any_v8_predictions':True,'window_audit_ids':sorted(audit_ids),
           'primary_rule':'full clip toptr AND PTR>=.70; supplemental windows do not revoke a fullclip pass',
           'limits':{'max_clips':256,'max_paced_read_bytes':BUDGET,'member_max_bytes':256*1024*1024,'new_analysis_budget_bytes':20_000_000},
           'accepted_TRAIN_hours':0}
    base.write_json(receipt,value);guard.status(phase='new256_manifest_PINNED',manifest_sha256=value['manifest_sha256'])
    return value


def main():
    AREA.mkdir(parents=True,exist_ok=True)
    guard=Guard();guard.wait();job=hard_process_memory_job()
    assert psutil.Process(1912).status()==psutil.STATUS_STOPPED
    closure=ROOT/'analysis/tiny_large_review_v1/actual_CLOSED0.json'
    assert json.loads(closure.read_text())['actual_exit_code']==0
    model=ROOT/'models/whisper_cpp_multilingual_tiny/ggml-tiny.bin'
    model_receipt=json.loads(model.with_suffix('.bin.receipt.json').read_text())
    assert model.exists() and model.stat().st_size==old.TINY_BYTES and model_receipt['sha256']==old.TINY_SHA
    base.write_json(AREA/'CPU_admission_v8.json',{'at_utc':base.now(),'pid':__import__('os').getpid(),
        'source':str(Path(__file__)),'source_sha256':sha(Path(__file__)),'ROOT_QC_closed':str(closure),'ROOT_QC_closed_SHA':sha(closure),
        'prior_v7_flags_unchanged':True,'model_reused_without_download':str(model),'model_SHA':old.TINY_SHA,
        'large_CPP_partial_preserved_bytes':(ROOT/'models/whisper_cpp_full_large_v3_q5_0/ggml-large-v3-q5_0.bin.partial').stat().st_size,
        'same_resource_guards':True,'hard_process_memory_limit':2_000_000_000})
    sampling=pin(guard)
    runtime=Runtime(ROOT/'scratch/whisper_cpp_v1.8.2_CPU_x64/Release');runtime.verify_defaults()
    log_file=(AREA/'tiny_native_log_v8.txt').open('a',encoding='utf-8',buffering=1)
    @LOG
    def log(level,message,_):log_file.write(message.decode('utf-8',errors='replace'))
    runtime.log_set(log,None);stream=old.TinyStream(model,guard);errors=[]
    @READ
    def read(_,destination,length):
        try:
            done=0
            while done<length:
                block=stream.read(min(65536,length-done))
                if not block:break
                c.memmove(destination+done,block,len(block));done+=len(block)
            return done
        except BaseException as exc:errors.append(repr(exc));return 0
    @EOF
    def eof(_):return stream.position>=old.TINY_BYTES
    @CLOSE
    def close(_):pass
    loader=Loader(None,read,eof,close);cp=runtime.context_defaults().contents;cp.use_gpu=False;cp.flash_attn=True
    context=runtime.init(c.byref(loader),cp)
    if not context:raise RuntimeError('Verified tiny native init failed: '+repr(errors))
    assert not errors;stream.finish()
    dll=runtime.dll
    dll.whisper_pcm_to_mel.argtypes=[c.c_void_p,c.POINTER(c.c_float),c.c_int,c.c_int];dll.whisper_pcm_to_mel.restype=c.c_int
    dll.whisper_lang_max_id.argtypes=[];dll.whisper_lang_max_id.restype=c.c_int
    dll.whisper_lang_auto_detect.argtypes=[c.c_void_p,c.c_int,c.c_int,c.POINTER(c.c_float)];dll.whisper_lang_auto_detect.restype=c.c_int
    nlang=dll.whisper_lang_max_id()+1;languages=[runtime.language_name(i).decode() for i in range(nlang)]
    def detect(pcm,kind,begin,end):
        guard.wait();clip=np.ascontiguousarray(pcm[round(begin*16000):round(end*16000)],dtype=np.float32)
        assert dll.whisper_pcm_to_mel(context,clip.ctypes.data_as(c.POINTER(c.c_float)),len(clip),1)==0
        probs=(c.c_float*nlang)();top=dll.whisper_lang_auto_detect(context,0,1,probs);assert top>=0
        raw={lang:float(probs[i]) for i,lang in enumerate(languages)}
        assert all(0<=p<=1 for p in raw.values()) and abs(sum(raw.values())-1)<1e-4
        language=languages[top];tr=raw['tr'];confident_nontr=language!='tr' and raw[language]>=.70
        return {'scope':kind,'start_s_in_clip':begin,'end_s_in_clip':end,'detected_language':language,
                'detected_language_probability':raw[language],'Turkish_probability':tr,'raw_language_probabilities':raw,
                'TR70_pass':language=='tr' and tr>=.70,'confident_nontr':confident_nontr,
                'uncertain_under70':tr<.70 and not confident_nontr}
    ledger=AREA/'new256_tiny_primary_ledger_v8.jsonl'
    previous=readl(ledger) if ledger.exists() else [];done_ids={r['segment_id'] for r in previous}
    rows=readl(Path(sampling['manifest']))
    try:
        for index,row in enumerate(rows,1):
            if row['segment_id'] in done_ids:continue
            guard.wait();start=time.monotonic()
            if guard.file_bytes_read>BUDGET:raise RuntimeError('Declared1GB paced-read budget reached; resume receipts preserved')
            try:
                pcm,decode=paced.decode_paced(row,guard)
                full=detect(pcm,'full_clip_primary',0.,row['duration_s'])
                category='retains_primary_TR70_screen_eligibility' if full['TR70_pass'] else 'quarantine_primary_confident_nontr' if full['confident_nontr'] else 'quarantine_primary_uncertain_under70'
                audit_reason=[]
                if row['predeclared_window_audit']:audit_reason.append('predeclared_seeded26_of256_audit')
                if not full['TR70_pass']:audit_reason.append('primary_fullclip_did_not_pass')
                windows=[]
                if audit_reason and row['duration_s']>=8:
                    for begin in range(0,int(row['duration_s']),4):
                        if row['duration_s']-begin>=2:
                            windows.append(detect(pcm,'supplemental_window_audit',float(begin),min(float(begin+4),row['duration_s'])))
                del pcm
                result={'segment_id':row['segment_id'],'recording_id':row['recording_id'],'shard':row['shard'],
                    'start_s':row['start_s'],'end_s':row['end_s'],'duration_s':row['duration_s'],'transcript':row['transcript'],
                    'full_clip_primary':full,'primary_screening_category':category,'retains_primary_language_screen_eligibility':full['TR70_pass'],
                    'predeclared_window_audit':row['predeclared_window_audit'],'window_audit_reasons':audit_reason,
                    'supplemental_window_observations':windows,'supplemental_any_confident_nontr':any(w['confident_nontr'] for w in windows),
                    'supplemental_any_uncertain_under70':any(w['uncertain_under70'] for w in windows),
                    'supplemental_window_flags_change_primary_pass':False,'decode':decode,
                    'model_SHA':old.TINY_SHA,'tiny_transcript_generated':False,'accepted_TRAIN':False,'source_label_corrected':False,
                    'elapsed_s':time.monotonic()-start,'own_RSS_bytes':psutil.Process().memory_info().rss,'completed_at_utc':base.now()}
            except Exception as exc:
                result={'segment_id':row['segment_id'],'recording_id':row['recording_id'],'shard':row['shard'],
                        'primary_screening_category':'quarantine_unavailable_no_LID_decision','error':repr(exc),
                        'retains_primary_language_screen_eligibility':False,'completed_at_utc':base.now(),'accepted_TRAIN':False}
            base.append_json(ledger,result);previous.append(result);done_ids.add(row['segment_id'])
            counts=dict(collections.Counter(r['primary_screening_category'] for r in previous))
            report={'kind':'bounded_new256_primary_tiny_language_screen_v8','updated_at_utc':base.now(),'declared':256,
                    'completed':len(previous),'primary_counts':counts,
                    'fullclip_pass_but_supplemental_window_alarm':sum(r.get('retains_primary_language_screen_eligibility') and
                        (r.get('supplemental_any_confident_nontr') or r.get('supplemental_any_uncertain_under70')) for r in previous),
                    'window_decisions_are_supplemental_only':True,'manifest_SHA':sampling['manifest_sha256'],
                    'ledger':str(ledger),'model_SHA':old.TINY_SHA,'CPU_only':True,'downloads':False,'accepted_TRAIN_hours':0,
                    'last_segment_id':row['segment_id'],'last_primary_category':result['primary_screening_category'],
                    'aggregate_paced_read_bytes':guard.file_bytes_read,'max_own_RSS_at_sample_boundaries':max(r.get('own_RSS_bytes',0) for r in previous)}
            base.write_json(AREA/'new256_tiny_primary_report_v8.json',report)
            guard.status(phase='actual_new256_primary_tiny_result_durable',completed=len(previous),primary_counts=counts,
                         segment_id=row['segment_id'],primary_category=result['primary_screening_category'])
    finally:
        runtime.free(context);log_file.close()
    guard.status(phase='completed_declared_new256_tiny_primary_batch',actual_CLOSED0=True,completed=len(previous))


if __name__=='__main__':main()
