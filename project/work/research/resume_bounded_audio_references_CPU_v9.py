"""Stopped-state successor: finish original first10k real-audio references only."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import numpy as np
import psutil
import run_accepted_audio_references_cpu_v4 as impl
import run_random_tiny_language_CPU_v7 as gating
from run_large_v3_sanity_cpu_v6 import hard_process_memory_job
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced

AREA=ROOT/'analysis/CPU_validation_v9'
base.AREA=AREA;paced.AREA=AREA;impl.AREA=AREA
READ_BUDGET=3_000_000_000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Guard(gating.Guard):
    def __init__(self):
        base.Guard.__init__(self,'bounded_real_audio_references_CPU_v9')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1048576
        self.status(maximum_audio_workers=1,max_own_RSS_bytes=2_000_000_000,CPU_threads=1,
            read_rate_limit_bytes_s=1048576,bounded_total_reference_clips=10000,maximum_paced_read_bytes=READ_BUDGET,
            ASR_or_GPU_model_loaded=False,automatic_next_batch_enabled=False)

    def pace(self,count):
        if self.file_bytes_read+count>READ_BUDGET:raise RuntimeError('Explicit3GB successor I/O budget exhausted; durable progress retained')
        return super().pace(count)

    def status(self,**fields):
        if hasattr(self,'file_bytes_read'):fields['aggregate_paced_file_bytes_read']=self.file_bytes_read
        base.Guard.status(self,**fields)


def retire_owned_old_worker(db):
    status=json.loads((ROOT/'analysis/CPU_validation_v4/accepted_audio_reference_v4_status.json').read_text())
    assert status['pid']==1912 and status['started_at_utc']=='2026-10-16T06:19:03.899067+00:00'
    if not psutil.pid_exists(1912):
        assert (AREA/'owned_V4_retirement_v9.json').exists();return
    p=psutil.Process(1912);command=p.cmdline()
    assert any('run_accepted_audio_references_cpu_v4.py' in x for x in command)
    assert p.status()==psutil.STATUS_STOPPED
    assert abs(p.create_time()-dt.datetime.fromisoformat(status['started_at_utc']).timestamp())<5
    db.execute('BEGIN IMMEDIATE')
    try:
        count=db.execute('SELECT COUNT(*) FROM references_audio').fetchone()[0]
        p.terminate();actual=p.wait(timeout=10)
        base.write_json(AREA/'owned_V4_retirement_v9.json',{'at_utc':base.now(),'pid':1912,'session':40713,
             'actual_owned_process_exit':actual,'cmdline':command,'verified_stopped_before_retirement':True,
             'safe_boundary':'Writer lock acquired; no active reference transaction; all committed rows retained',
             'committed_rows_at_handoff':count,'old_running_source_edited':False})
    finally:db.rollback()


def compare_batch(db,ordinals,guard):
    if db.execute('SELECT 1 FROM comparisons_complete WHERE batch_id=0').fetchone():return
    rows=[]
    for ordinal in ordinals:
        info=db.execute("SELECT source_id,frames,length(vectors),length(positions) FROM references_audio WHERE ordinal=? AND status='actual_WAV_PCM_verified'",(int(ordinal),)).fetchone()
        if not info or info[1]<8:continue
        guard.pace(info[2]+info[3])
        rows.append(db.execute('SELECT source_id,vectors,positions,frames FROM references_audio WHERE source_id=?',(info[0],)).fetchone())
    if not rows:return
    arrays=[];refids=[];refpos=[];sourceids=[];lengths=[]
    for index,(sid,blob,posblob,n) in enumerate(rows):
        sourceids.append(sid);lengths.append(n);arrays.append(np.frombuffer(blob,dtype='<f4').reshape(n,32))
        refids.extend([index]*n);refpos.extend(np.frombuffer(posblob,dtype='<i4').tolist())
    bank=np.concatenate(arrays);ids=np.asarray(refids,np.int32);positions=np.asarray(refpos,np.int32)
    headers=db.execute("SELECT segment_id,PCM_i16_sha256,frames,length(vectors),length(positions) FROM candidate_features WHERE status='actual_candidate_fingerprint_cached'").fetchall()
    checked=0;triage=0
    for sid,digest,n,blen,plen in headers:
        if db.execute('SELECT 1 FROM candidate_batch_checks WHERE segment_id=? AND batch_id=0',(sid,)).fetchone():continue
        guard.pace(blen+plen);blob,posblob=db.execute('SELECT vectors,positions FROM candidate_features WHERE segment_id=?',(sid,)).fetchone()
        query=np.frombuffer(blob,dtype='<f4').reshape(n,32);pos=np.frombuffer(posblob,dtype='<i4')
        guard.wait();matches=impl.retrieve(query,pos,bank,ids,positions,sourceids,lengths)
        exact=[x[0] for x in db.execute('SELECT source_id FROM references_audio WHERE PCM_i16_sha256=?',(digest,))]
        result={'candidate_segment_id':sid,'reference_batch_id':0,'reference_spectral_usable_clips':len(rows),
                'exact_accepted_PCM_matches':exact,'conservative_spectral_triage_matches':matches,
                'completed_at_utc':base.now(),'accepted_TRAIN':False,
                'interpretation':'Partial reference coverage; no match does not prove global uniqueness; any match is review triage only'}
        guard.wait();db.execute('INSERT INTO candidate_batch_checks VALUES(?,?,?)',(sid,0,json.dumps(result,ensure_ascii=False)));db.commit()
        checked+=1
        if matches or exact:triage+=1;base.append_json(AREA/'candidate_accepted_audio_overlap_triage_v9.jsonl',result)
        if checked%10==0:guard.status(phase='candidate_vs_first10k_real_audio_reference_review',candidates_checked_in_resume=checked,triage_candidates_in_resume=triage)
    db.execute('INSERT INTO comparisons_complete VALUES(0)');db.commit()


def main():
    AREA.mkdir(parents=True,exist_ok=True)
    guard=Guard();guard.wait();job=hard_process_memory_job()
    prior_closed=ROOT/'analysis/CPU_validation_v8/actual_new256_CLOSED0_v8.json'
    assert json.loads(prior_closed.read_text(encoding='utf-8-sig'))['exit_code']==0
    path=ROOT/'indexes/accepted_audio_reference_v4.sqlite'
    db=sqlite3.connect(path,timeout=5);db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=MEMORY')
    retire_owned_old_worker(db)
    for name in ('accepted_reference_order_v4.u32','accepted_reference_plan_v4.json'):
        destination=AREA/name
        if not destination.exists():destination.write_bytes(guard.read_file(ROOT/'analysis/CPU_validation_v4'/name))
    order,inherited=impl.prepare_order(guard);ordinals=order[:10000]
    initial=db.execute('SELECT COUNT(*) FROM references_audio').fetchone()[0]
    plan={'kind':'explicit_bounded_reference_successor_plan_v9','admitted_at_utc':base.now(),'pid':os.getpid(),
          'source_path':str(Path(__file__)),'source_SHA':sha(Path(__file__)),
          'frozen_original_implementation':str(PROJECT/'work/research/run_accepted_audio_references_cpu_v4.py'),
          'frozen_original_implementation_SHA':sha(PROJECT/'work/research/run_accepted_audio_references_cpu_v4.py'),
          'new_isolated_process_redirects_only_its_module_output_directory':True,'old_source_or_environment_edited':False,
          'resumed_index':str(path),'already_committed_reference_rows':initial,'bounded_reference_batch_total':10000,
          'remaining_new_reference_rows_max':10000-initial,'candidate_fingerprints_max':193,'reference_order_SHA':inherited['order_sha256'],
          'inherited_full675h_order_preserved':True,'inherited36batch_auto_continuation_not_used_in_this_successor':True,
          'max_paced_read_bytes':READ_BUDGET,'max_index_bytes':4_000_000_000,'max_own_RSS_bytes':2_000_000_000,
          'CPU_only':True,'no_ASR_or_GPU_model':True,'accepted_TRAIN_hours':0,
          'actual_stopped_state_proof':str(gating.PROOF),'actual_stopped_state_proof_SHA':gating.PROOF_SHA,
          'full_accepted_audio_reference_coverage':False,'no_global_uniqueness_or_admission_claim':True}
    base.write_json(AREA/'bounded_reference_successor_plan_v9.json',plan)
    baseline=impl.summary(db,guard,'resumed_first10k_real_audio_references',0)
    guard.pace(initial*8)
    existing={r[0] for r in db.execute('SELECT ordinal FROM references_audio')}
    pending=[ordinal for ordinal in ordinals if int(ordinal) not in existing]
    verified=baseline['actual_WAV_PCM_verified_reference_clips'];hours=baseline['actual_audio_reference_hours']
    usable=baseline['spectral_reference_clips_usable_under_current_min8frames'];errors=baseline['unavailable_or_integrity_error_reference_rows']
    for i,ordinal in enumerate(pending,1):
        impl.index_reference(db,int(ordinal),guard)
        status,duration,spectral=db.execute('SELECT status,duration_s,spectral_usable FROM references_audio WHERE ordinal=?',(int(ordinal),)).fetchone()
        if status=='actual_WAV_PCM_verified':verified+=1;hours+=duration/3600;usable+=spectral
        else:errors+=1
        if i%25==0:
            report={**baseline,'updated_at_utc':base.now(),'phase':'paced_real_accepted_audio_reference_indexing',
                'reference_rows_processed':initial+i,'actual_WAV_PCM_verified_reference_clips':verified,
                'actual_audio_reference_hours':hours,'spectral_reference_clips_usable_under_current_min8frames':usable,
                'unavailable_or_integrity_error_reference_rows':errors,'accepted_reference_clip_coverage_fraction':verified/352768,
                'aggregate_paced_read_bytes':guard.file_bytes_read,'CPU_RSS_bytes_observed':psutil.Process().memory_info().rss}
            base.write_json(AREA/'accepted_audio_reference_coverage_v4.json',report)
            guard.status(phase='paced_real_accepted_audio_reference_indexing',new_reference_rows_in_this_successor=i,
                         actual_audio_references_verified=verified,actual_audio_reference_hours=hours,spectral_usable_references=usable)
        if path.stat().st_size>4_000_000_000:raise RuntimeError('Unchanged4GB reference-index cap reached')
    impl.summary(db,guard,'bounded_first10k_reference_audio_reads_complete',0)
    impl.cache_candidates(db,guard)
    compare_batch(db,ordinals,guard)
    report=impl.summary(db,guard,'completed_first10k_and_partial_duplicate_reference_review',0)
    db.execute('INSERT OR REPLACE INTO reference_batches VALUES(?,?)',(0,json.dumps(report)));db.commit();db.close()
    base.write_json(AREA/'bounded_reference_successor_completed_v9.json',report)
    guard.status(phase='completed_declared_bounded_reference_successor',actual_CLOSED0=True)


if __name__=='__main__':main()
