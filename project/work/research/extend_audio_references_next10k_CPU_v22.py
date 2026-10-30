"""One separately declared next10k reference interval; no model or new downloads."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
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

AREA=ROOT/'analysis/CPU_validation_v22'
base.AREA=AREA;paced.AREA=AREA;impl.AREA=AREA
BATCH_ID=13
TARGET=140000
READ_BUDGET=5_000_000_000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Guard(gating.Guard):
    def __init__(self):
        base.Guard.__init__(self,'next10k_real_audio_references_CPU_v22')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1048576
        self.status(maximum_audio_workers=1,max_own_RSS_bytes=2_000_000_000,CPU_threads=1,
            read_rate_limit_bytes_s=1048576,reference_interval_start=130000,reference_interval_end=TARGET,
            maximum_paced_read_bytes=READ_BUDGET,ASR_or_GPU_model_loaded=False,automatic_next_batch_enabled=False)

    def pace(self,count):
        if self.file_bytes_read+count>READ_BUDGET:raise RuntimeError('Explicit5GB next10k I/O budget reached; resume state retained')
        return super().pace(count)

    def status(self,**fields):
        if hasattr(self,'file_bytes_read'):fields['aggregate_paced_file_bytes_read']=self.file_bytes_read
        base.Guard.status(self,**fields)


def compare_batch(db,ordinals,guard):
    if db.execute('SELECT 1 FROM comparisons_complete WHERE batch_id=?',(BATCH_ID,)).fetchone():return
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
        if db.execute('SELECT 1 FROM candidate_batch_checks WHERE segment_id=? AND batch_id=?',(sid,BATCH_ID)).fetchone():continue
        guard.pace(blen+plen);blob,posblob=db.execute('SELECT vectors,positions FROM candidate_features WHERE segment_id=?',(sid,)).fetchone()
        query=np.frombuffer(blob,dtype='<f4').reshape(n,32);pos=np.frombuffer(posblob,dtype='<i4')
        guard.wait();matches=impl.retrieve(query,pos,bank,ids,positions,sourceids,lengths)
        exact=[x[0] for x in db.execute('SELECT source_id FROM references_audio WHERE PCM_i16_sha256=?',(digest,))]
        result={'candidate_segment_id':sid,'reference_batch_id':BATCH_ID,'reference_spectral_usable_clips':len(rows),
                'exact_accepted_PCM_matches':exact,'conservative_spectral_triage_matches':matches,
                'completed_at_utc':base.now(),'accepted_TRAIN':False,
                'interpretation':'Bank remains partial; no match does not prove global uniqueness; a match is review triage only'}
        guard.wait();db.execute('INSERT INTO candidate_batch_checks VALUES(?,?,?)',(sid,BATCH_ID,json.dumps(result,ensure_ascii=False)));db.commit()
        checked+=1
        if matches or exact:triage+=1;base.append_json(AREA/'candidate_accepted_audio_overlap_triage_v22.jsonl',result)
        if checked%10==0:guard.status(phase='candidate_vs_next10k_reference_review',candidates_checked_in_resume=checked,triage_candidates_in_resume=triage)
    db.execute('INSERT INTO comparisons_complete VALUES(?)',(BATCH_ID,));db.commit()


def main():
    AREA.mkdir(parents=True,exist_ok=True)
    guard=Guard();guard.wait();job=hard_process_memory_job()
    closure=ROOT/'analysis/CPU_validation_v21/actual_next10k_reference_CLOSED0_v21.json'
    previous=json.loads(closure.read_text());assert previous['exit_code']==0 and previous['actual_reference_clips']==130000
    review_path=ROOT/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'
    review=json.loads(review_path.read_text(encoding='utf-8'))
    assert review['actual_exit_code']==0 and review['ROOT_owned_GPU_exit_observed'] is True
    assert review['actual_closure_SHA256']=='3faf8936f9a416a2c9d248c32d7e635961bfaef9e10c9e9363bcc69a0ec6af9f'
    for old_pid,source in ((24292,'extend_audio_references_next10k_CPU_v21.py'),):
        if psutil.pid_exists(old_pid):
            assert not any(source in value for value in psutil.Process(old_pid).cmdline()),'Previous owned audio worker still active'
    path=ROOT/'indexes/accepted_audio_reference_v4.sqlite'
    db=sqlite3.connect(path,timeout=5);db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=MEMORY')
    for name in ('accepted_reference_order_v4.u32','accepted_reference_plan_v4.json'):
        destination=AREA/name
        if not destination.exists():destination.write_bytes(guard.read_file(ROOT/'analysis/CPU_validation_v4'/name))
    order,inherited=impl.prepare_order(guard);ordinals=order[130000:TARGET]
    assert len(ordinals)==10000 and len(set(map(int,ordinals)))==10000
    interval=AREA/'reference_interval_ordinals_v22.u32'
    raw=ordinals.astype('<u4').tobytes()
    if interval.exists():assert interval.read_bytes()==raw
    else:interval.write_bytes(raw)
    plan_path=AREA/'bounded_next10k_reference_plan_v22.json'
    if not plan_path.exists():
        plan={'kind':'separately_declared_next10k_real_audio_reference_interval_v22','admitted_at_utc':base.now(),'pid':os.getpid(),
              'source':str(Path(__file__)),'source_SHA':sha(Path(__file__)),'previous_actual_closure':str(closure),'previous_actual_closure_SHA':sha(closure),
              'fresh_storage_config_SHA256':sha(ROOT/'storage_config.json'),
              'fresh_storage_free_bytes_at_admission':shutil.disk_usage(ROOT).free,
              'fresh_system_memory_at_admission':base.memory(),
              'fresh_GPU_state_gate_at_admission':guard.gate_snapshot(),
              'frozen_reused_source_SHA256':{name:sha(PROJECT/'work/research'/name) for name in (
                  'run_accepted_audio_references_cpu_v4.py','run_random_tiny_language_CPU_v7.py',
                  'run_large_v3_sanity_cpu_v6.py','cpu_preparation_guard_v2.py','paced_CPU_audio_v3.py')},
              'index':str(path),'immutable_carrier':str(impl.CARRIER),'carrier_fingerprint':inherited['carrier_fingerprint'],
              'immutable_global_order_SHA':inherited['order_sha256'],'interval_ordinals_file':str(interval),'interval_ordinals_SHA':sha(interval),
              'exact_interval_in_pinned_order':[130000,TARGET],'new_reference_clips_max':10000,'global_reference_clips_target':TARGET,
              'reference_ID_binding':'Each sealed ordinal maps uniquely to source_id in the existing immutable352768row carrier/catalog',
              'no_new_random_selection_or_predictions':True,'cached_candidate_fingerprints_reused':186,'candidate_feature_rows_unavailable':7,
              'no_staged_audio_redecode_needed':True,'no_ASR_or_GPU_model':True,'no_downloads':True,'max_paced_read_bytes':READ_BUDGET,
              'prior_ROOT_review_release_verification':str(review_path),'prior_ROOT_review_release_verification_SHA256':sha(review_path),
              'index_cap_bytes':4_000_000_000,'own_hard_process_memory_limit':2_000_000_000,
              'minimum_E_free_bytes':100_000_000_000,'yield_commit_ratio':.90,'critical_commit_ratio':.95,'minimum_commit_headroom_bytes':2_000_000_000,
              'read_rate_limit_bytes_s':1048576,'automatic_extension_after_this_interval':False,
              'full675h_audio_exclusion_complete':False,'accepted_TRAIN_hours':0}
        base.write_json(plan_path,plan)
    else:assert json.loads(plan_path.read_text())['interval_ordinals_SHA']==sha(interval)
    baseline=impl.summary(db,guard,'next10k_reference_interval_admitted',BATCH_ID)
    initial=baseline['reference_rows_processed'];guard.pace(initial*8)
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
            report={**baseline,'updated_at_utc':base.now(),'phase':'paced_next10k_actual_reference_indexing',
                'source_group_verified_counts_are_baseline_until_phase_boundary':True,
                'reference_rows_processed':initial+i,'actual_WAV_PCM_verified_reference_clips':verified,
                'actual_audio_reference_hours':hours,'spectral_reference_clips_usable_under_current_min8frames':usable,
                'unavailable_or_integrity_error_reference_rows':errors,'accepted_reference_clip_coverage_fraction':verified/352768,
                'aggregate_paced_read_bytes':guard.file_bytes_read,'CPU_RSS_bytes_observed':psutil.Process().memory_info().rss}
            base.write_json(AREA/'accepted_audio_reference_coverage_v4.json',report)
            guard.status(phase='paced_next10k_actual_reference_indexing',new_reference_rows_in_this_interval=i,
                         actual_audio_references_verified=verified,actual_audio_reference_hours=hours,spectral_usable_references=usable)
        if path.stat().st_size>4_000_000_000:raise RuntimeError('Unchanged4GB reference-index cap reached')
    impl.summary(db,guard,'next10k_reference_audio_reads_complete',BATCH_ID)
    compare_batch(db,ordinals,guard)
    report=impl.summary(db,guard,'completed_next10k_and_partial_reference_review',BATCH_ID)
    db.execute('INSERT OR REPLACE INTO reference_batches VALUES(?,?)',(BATCH_ID,json.dumps(report)));db.commit();db.close()
    base.write_json(AREA/'bounded_next10k_reference_completed_v22.json',report)
    guard.status(phase='completed_declared_next10k_reference_interval',actual_CLOSED0=True)


if __name__=='__main__':main()
