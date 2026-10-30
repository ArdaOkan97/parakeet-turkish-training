"""Bounded full-large-v3 CPU sanity setup; owns only CPU processes and E files."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import psutil
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced

AREA = ROOT/'analysis/CPU_validation_v6'
base.AREA = AREA
paced.AREA = AREA
MODEL_REV = 'c521a4b02f422512d734391fdf08bb08c0862f68'
MODEL_SHA = 'd75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1'
MODEL_NAME = 'ggml-large-v3-q5_0.bin'
RUNTIME_SHA = 'b1514ebc099765e39fa37eb780b92a140a94c86bb0b3b3d98226b38825979732'
REF_PID = 1912
REF_START = '2026-10-16T06:19:03.899067+00:00'


class Guard(paced.PacedGuard):
    def __init__(self):
        base.Guard.__init__(self, 'full_large_v3_sanity_setup_v6')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1024*1024
        self.status(maximum_audio_workers=1, maximum_model_download_bytes=1_200_000_000,
                    CPU_threads=1, max_own_RSS_bytes=2_000_000_000, model='full original Whisper-large-v3 Q5_0')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(65536),b''):h.update(block)
    return h.hexdigest()


def pin_manifest():
    dest=AREA/'worst_samples_full_large_v3_manifest_v6.jsonl'
    if dest.exists():
        receipt=json.loads((AREA/'worst_samples_selection_receipt_v6.json').read_text())
        assert sha(dest)==receipt['manifest_sha256']
        return receipt
    inputs=[ROOT/'analysis/CPU_validation_v2/independent_QC_ledger_v2.jsonl',
            ROOT/'analysis/CPU_validation_v3/new8_shards_independent_QC_ledger_v3.jsonl']
    flagged=[]
    for source in inputs:
        for line in source.read_text(encoding='utf-8-sig').splitlines():
            row=json.loads(line)
            if row.get('triage_flags') and row.get('independent_recognizer_completed'):
                row['cached_ledger']=str(source);flagged.append(row)
    assert len(flagged)==106 and len({r['segment_id'] for r in flagged})==106
    ranked=sorted([r for r in flagged if r['duration_s']>=5],
                  key=lambda r:(-r['source_vs_independent_WER'],r['segment_id']))
    selected=ranked[:12]
    reasons={r['segment_id']:['top12_frozen_disagreement_duration_at_least5s'] for r in selected}
    extra=sorted([r for r in flagged if r['independent_language']!='tr'],key=lambda r:r['segment_id'])
    extra.append(min(flagged,key=lambda r:(r['independent_language_probability'],r['segment_id'])))
    for r in extra:
        reason='nonTR_cached_small_case' if r['independent_language']!='tr' else 'lowest_cached_language_confidence'
        if r['segment_id'] not in reasons:selected.append(r);reasons[r['segment_id']]=[]
        reasons[r['segment_id']].append(reason)
    assert len(selected)<=16
    with dest.open('w',encoding='utf-8') as f:
        for row in selected:
            row={**row,'selection_reasons':reasons[row['segment_id']]}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    receipt={'kind':'preinference_pinned_full_large_v3_worst_sample_selection_v6','pinned_at_utc':base.now(),
             'samples':len(selected),'duration_s':sum(r['duration_s'] for r in selected),
             'selection':'12 highest frozen source-vs-small WER, duration>=5s, plus nonTR and lowest LID if absent; no source correctness assumed',
             'manifest':str(dest),'manifest_sha256':sha(dest),'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs],
             'source_labels_small_hypotheses_scores_and_flags_preserved':True,'no_inference_yet':True,
             'source_ids':[r['segment_id'] for r in selected]}
    base.write_json(AREA/'worst_samples_selection_receipt_v6.json',receipt)
    return receipt


def pause_reference_worker(guard):
    process=psutil.Process(REF_PID)
    command=process.cmdline()
    if not any('run_accepted_audio_references_cpu_v4.py' in v for v in command):
        raise RuntimeError('CPU ownership mismatch; will not suspend any other PID')
    status=json.loads((ROOT/'analysis/CPU_validation_v4/accepted_audio_reference_v4_status.json').read_text())
    assert status['pid']==REF_PID and status['started_at_utc']==REF_START
    existing=AREA/'accepted_reference_worker_declared_pause_v6.json'
    if process.status()==psutil.STATUS_STOPPED:
        assert existing.exists()
        return
    database=ROOT/'indexes/accepted_audio_reference_v4.sqlite'
    db=sqlite3.connect(database,timeout=5)
    # Acquire writer lock before suspending: no reference transaction can be
    # half-committed. Any active read/decode resumes after this narrow phase.
    db.execute('BEGIN IMMEDIATE')
    try:
        count,seconds=db.execute("SELECT COUNT(*),SUM(duration_s) FROM references_audio WHERE status='actual_WAV_PCM_verified'").fetchone()
        last=db.execute('SELECT source_id,ordinal,details FROM references_audio ORDER BY rowid DESC LIMIT 1').fetchone()
        process.suspend()
        base.write_json(existing,{'kind':'owned_CPU_reference_pause_at_durable_transaction_boundary_v6',
             'paused_at_utc':base.now(),'pid':REF_PID,'session':40713,'cmdline':command,
             'process_create_time':process.create_time(),'last_committed_verified_rows':count,
             'last_committed_training_duration_h':seconds/3600,'last_committed_row':last,
             'reason':'Human priority:bounded full original large-v3 worst-case sanity; single CPU audio worker',
             'running_source_edited':False,'reference_database_complete':False,'resume_exact_process_when_sanity_finishes':True})
    finally:
        db.rollback();db.close()
    guard.status(phase='owned_reference_worker_suspended_at_durable_boundary',reference_pid=REF_PID,
                 reference_verified_rows_at_pause=count)


def download(url,path,expected_sha,guard,max_bytes,expected_bytes=None):
    partial=path.with_suffix(path.suffix+'.partial')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        # Full files are checked by paced streaming rather than materialized.
        h=hashlib.sha256()
        with path.open('rb') as f:
            while True:
                block=f.read(65536)
                if not block:break
                guard.pace(len(block));h.update(block)
        assert h.hexdigest()==expected_sha
        return
    start=partial.stat().st_size if partial.exists() else 0
    assert start<=max_bytes
    h=hashlib.sha256()
    if start:
        with partial.open('rb') as f:
            while True:
                block=f.read(65536)
                if not block:break
                guard.pace(len(block));h.update(block)
    # Short Range requests avoid holding one multi-GB CDN response across DEV
    # pauses. Each safely resumes the durable partial after a short network EOF.
    total=expected_bytes
    done=start;last=done//(4*1024*1024);failures=0
    with partial.open('ab') as f:
        while total is None or done<total:
            end=min(done+4*1024*1024-1,(total or max_bytes)-1)
            request=urllib.request.Request(url,headers={'User-Agent':'bounded-CPU-data-preparation-v6',
                                                       'Range':f'bytes={done}-{end}'})
            before=done;guard.wait()
            try:
                with urllib.request.urlopen(request,timeout=60) as response:
                    content_range=response.headers.get('Content-Range')
                    if response.status==206:
                        span,full=content_range.split(' ',1)[1].split('/')
                        actual_start,actual_end=map(int,span.split('-'))
                        if actual_start!=done:raise RuntimeError('Resume range start differs; partial preserved')
                        incoming_total=int(full)
                    elif done==0 and response.status==200:
                        incoming_total=int(response.headers['Content-Length'])
                    else:raise RuntimeError('Resume server did not honor Range; partial preserved')
                    if incoming_total>max_bytes or (total is not None and incoming_total!=total):
                        raise RuntimeError('Download size differs from pinned bound')
                    total=incoming_total
                    wanted=min(end+1,total)
                    while done<wanted:
                        guard.wait();block=response.read(min(65536,wanted-done))
                        if not block:break
                        guard.pace(len(block));f.write(block);h.update(block);done+=len(block)
                        if done>max_bytes:raise RuntimeError('Download exceeds declared budget')
                        if done//(4*1024*1024)!=last:
                            last=done//(4*1024*1024);f.flush()
                            guard.status(phase='paced_official_runtime_or_full_large_v3_model_download',active_download=path.name,
                                         downloaded_bytes=done,expected_download_bytes=total,aggregate_paced_bytes=guard.file_bytes_read)
            except (OSError,TimeoutError) as error:
                f.flush();failures+=1
                guard.status(phase='resumable_model_download_network_retry',downloaded_bytes=done,error=repr(error),retry=failures)
                if failures>20:raise
                time.sleep(min(10,failures))
            if done==before:
                failures+=1
                if failures>20:raise RuntimeError('No-progress download retry bound reached; partial preserved')
            else:failures=0
    assert done==total and h.hexdigest()==expected_sha,(done,total,h.hexdigest())
    partial.replace(path)
    base.write_json(path.with_suffix(path.suffix+'.receipt.json'),{'url':url,'bytes':done,'sha256':expected_sha,
                    'completed_at_utc':base.now(),'official_source':True,'rate_bytes_per_second':1048576})


def main():
    AREA.mkdir(parents=True,exist_ok=True)
    guard=Guard();guard.wait();selection=pin_manifest()
    fit={'kind':'full_original_large_v3_CPU_runtime_fit_preflight_v6','at_utc':base.now(),
         'choice':'Official whisper.cpp v1.8.2 CPU release; ggerganov full-large-v3 Q5_0',
         'model_repository':'ggerganov/whisper.cpp','model_revision':MODEL_REV,'model_filename':MODEL_NAME,'model_sha256':MODEL_SHA,
         'original_family':'openai/whisper-large-v3, full32 encoder+32 decoder layers; never turbo/distil',
         'cached_FP16_model':str(PROJECT/'work/models/whisper-large-v3'),
         'cached_FP16_rejected_for_this_job':'~3.1GB weights alone exceed unchanged2GB process bound; no load/conversion attempted',
         'CT2_in_memory_bytes_rejected':'Avoid multi-GB source bytes plus destination allocation duplication',
         'official_cpp_loader':'Custom supported whisper_model_loader streams <=64KiB directly into runtime destination with pacing; no whole model bytes object',
         'mmap_claim':False,'mmap_note':'Official v1.8.2 source uses streaming model-loader callbacks, not mmap; no unsupported mmap claim',
         'predicted_components_bytes':{'Q5_model_file_upper_bound':1_100_000_000,'F16_self_cross_pad_KV_approx':343_408_640,
                                       'compute_graphs_vocab_python_margin':500_000_000},
         'predicted_total_upper_bound_bytes':1_943_408_640,'fit_status':'Plausible only; actual native initialization under hard2GB job/process guards required before inference',
         'unchanged_hard_RSS_bound_bytes':2_000_000_000,'yield_commit_ratio':.90,'stop_commit_ratio':.95,
         'minimum_commit_headroom_bytes':2_000_000_000,'minimum_E_free_bytes':100_000_000_000,
         'model_download_budget_bytes':1_200_000_000,'selection':selection,
         'official_runtime_url':'https://github.com/ggml-org/whisper.cpp/releases/tag/v1.8.2',
         'official_model_url':f'https://huggingface.co/ggerganov/whisper.cpp/blob/{MODEL_REV}/{MODEL_NAME}'}
    base.write_json(AREA/'full_large_v3_CPU_fit_preflight_v6.json',fit)
    pause_reference_worker(guard)
    runtime=ROOT/'downloads/whisper_cpp_v1.8.2_CPU_x64.zip'
    download('https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.2/whisper-bin-x64.zip',runtime,RUNTIME_SHA,guard,4_000_000,3_832_432)
    runtime_dir=ROOT/'scratch/whisper_cpp_v1.8.2_CPU_x64'
    with zipfile.ZipFile(runtime) as z:
        for member in z.infolist():
            target=(runtime_dir/member.filename).resolve()
            assert target.is_relative_to(runtime_dir.resolve()) and member.file_size<30_000_000
        z.extractall(runtime_dir)
    base.write_json(AREA/'official_CPU_runtime_files_v6.json',{'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in runtime_dir.rglob('*') if p.is_file()],
                     'zip_sha256':RUNTIME_SHA,'CUDA_build':False})
    model=ROOT/'models/whisper_cpp_full_large_v3_q5_0'/MODEL_NAME
    download(f'https://huggingface.co/ggerganov/whisper.cpp/resolve/{MODEL_REV}/{MODEL_NAME}',model,MODEL_SHA,guard,1_200_000_000,1_081_140_203)
    guard.status(phase='official_full_large_v3_model_download_complete',model_path=str(model),model_bytes=model.stat().st_size)
    runner=PROJECT/'work/research/run_large_v3_sanity_cpu_v6.py'
    while not runner.exists():
        guard.status(phase='waiting_for_bounded_native_runner_source_after_setup');guard.wait();time.sleep(5)
    base.write_json(AREA/'native_runner_dispatch_source_v6.json',{'path':str(runner),'sha256':sha(runner),'dispatch_at_utc':base.now()})
    guard.status(phase='dispatching_single_full_large_v3_native_CPU_sanity_worker')
    result=subprocess.run([sys.executable,str(runner)],cwd=PROJECT,env=os.environ.copy())
    guard.status(phase='native_sanity_worker_closed',native_exit_code=result.returncode)
    if result.returncode!=0:raise RuntimeError(f'Bounded native sanity worker exit{result.returncode}; receipts preserved')


if __name__=='__main__':main()
