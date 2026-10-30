"""Full original large-v3 Q5_0,12 predeclared cached worst cases, CPU only."""
import ctypes as c
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import numpy as np
import psutil
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced
from whisper_cpp_cpu_bindings_v6 import Runtime,Loader,READ,EOF,CLOSE,LOG,ABORT
from audit_cached_pilot_text_flags_v5 import load_spoken,counts
from metrics import normalize_text

AREA=ROOT/'analysis/CPU_validation_v6'
base.AREA=AREA;paced.AREA=AREA
MODEL=ROOT/'models/whisper_cpp_full_large_v3_q5_0/ggml-large-v3-q5_0.bin'
MODEL_SHA='d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1'
MODEL_REV='c521a4b02f422512d734391fdf08bb08c0862f68'
LIMIT=2_000_000_000


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hard_process_memory_job():
    """Enforce the existing2GB ceiling in native allocation, no guard increase."""
    class IO(c.Structure):
        _fields_=[(name,c.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                                 'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
    class Basic(c.Structure):
        _fields_=[('PerProcessUserTimeLimit',c.c_longlong),('PerJobUserTimeLimit',c.c_longlong),
                  ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',c.c_size_t),
                  ('MaximumWorkingSetSize',c.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                  ('Affinity',c.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
    class Extended(c.Structure):
        _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',c.c_size_t),
                  ('JobMemoryLimit',c.c_size_t),('PeakProcessMemoryUsed',c.c_size_t),('PeakJobMemoryUsed',c.c_size_t)]
    k=c.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.restype=wintypes.HANDLE;k.CreateJobObjectW.argtypes=[c.c_void_p,wintypes.LPCWSTR]
    k.SetInformationJobObject.restype=wintypes.BOOL
    k.SetInformationJobObject.argtypes=[wintypes.HANDLE,c.c_int,c.c_void_p,wintypes.DWORD]
    k.AssignProcessToJobObject.restype=wintypes.BOOL
    k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    k.GetCurrentProcess.restype=wintypes.HANDLE
    job=k.CreateJobObjectW(None,None)
    if not job:raise c.WinError(c.get_last_error())
    info=Extended();info.BasicLimitInformation.LimitFlags=0x100;info.ProcessMemoryLimit=LIMIT
    if not k.SetInformationJobObject(job,9,c.byref(info),c.sizeof(info)):raise c.WinError(c.get_last_error())
    if not k.AssignProcessToJobObject(job,k.GetCurrentProcess()):raise c.WinError(c.get_last_error())
    # Keep the handle alive through context cleanup; no KILL_ON_JOB_CLOSE flag.
    return job


class Guard(paced.PacedGuard):
    def __init__(self):
        base.Guard.__init__(self,'full_large_v3_sanity_native_v6')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1024*1024
        self.peak_rss=psutil.Process().memory_info().rss
        self.status(maximum_audio_workers=1,max_own_RSS_bytes=LIMIT,CPU_threads=1,
                    model='full_original_Whisper_large_v3_Q5_0',read_rate_limit_bytes_s=1048576)

    def watchdog(self):
        process=psutil.Process()
        while True:
            time.sleep(.25)
            try:
                m=base.memory();rss=process.memory_info().rss
                self.peak_rss=max(getattr(self,'peak_rss',0),rss)
                if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<LIMIT or rss>LIMIT:
                    base.write_json(AREA/'full_large_v3_sanity_native_v6_resource_stop.json',
                                    {'at_utc':base.now(),'own_pid':os.getpid(),'memory':m,'own_RSS':rss,'own_worker_only':True})
                    os._exit(75)
            except Exception:
                os._exit(76)


def resume_reference_worker(reason):
    receipt=json.loads((AREA/'accepted_reference_worker_declared_pause_v6.json').read_text())
    process=psutil.Process(receipt['pid'])
    assert abs(process.create_time()-receipt['process_create_time'])<.001
    assert process.cmdline()==receipt['cmdline']
    if process.status()==psutil.STATUS_STOPPED:process.resume()
    base.write_json(AREA/'accepted_reference_worker_resume_v6.json',
                    {'resumed_at_utc':base.now(),'pid':process.pid,'session':receipt['session'],
                     'reason':reason,'source_edited':False,'exact_original_worker_resumed':True})


def score_pair(ref,hyp,spoken):
    return {'frozen':counts(normalize_text(ref,'tr'),normalize_text(hyp,'tr')),
            'existing_spoken_form_diagnostic_one_pass':counts(spoken(ref),spoken(hyp))}


def main():
    guard=Guard();guard.wait()
    assert psutil.Process(1912).status()==psutil.STATUS_STOPPED,'One audio worker requires own references suspended'
    job=hard_process_memory_job()
    guard.status(phase='hard_native_process_memory_limit_installed',process_memory_limit_bytes=LIMIT)
    selection=json.loads((AREA/'worst_samples_selection_receipt_v6.json').read_text())
    manifest=Path(selection['manifest'])
    assert sha(manifest)==selection['manifest_sha256'] and 1<=selection['samples']<=16
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==selection['samples']
    model_receipt=json.loads(MODEL.with_suffix('.bin.receipt.json').read_text())
    assert model_receipt['sha256']==MODEL_SHA and model_receipt['bytes']==MODEL.stat().st_size
    # Allocation reserve must fit below the unchanged90% yield threshold.
    while True:
        guard.wait();memory=base.memory();limit=memory['commit_headroom_bytes']/(1-memory['commit_ratio'])
        reserve=1_943_408_640
        if memory['commit_ratio']+reserve/limit<.90:break
        guard.status(phase='waiting_for_full_large_v3_projected_commit_fit',memory=memory,projected_additional_bytes=reserve)
        time.sleep(5)
    runtime_receipt=json.loads((AREA/'official_CPU_runtime_files_v6.json').read_text())
    for item in runtime_receipt['files']:
        if item['path'].lower().endswith('.dll'):
            assert sha(item['path'])==item['sha256']
    runtime=Runtime(ROOT/'scratch/whisper_cpp_v1.8.2_CPU_x64/Release')
    abi=runtime.verify_defaults()
    log_file=(AREA/'official_full_large_v3_native_log_v6.txt').open('a',encoding='utf-8',buffering=1)
    @LOG
    def log(level,message,_):
        try:log_file.write(message.decode('utf-8',errors='replace'))
        except Exception:pass
    runtime.log_set(log,None)
    spoken,fixture_count,_=load_spoken()
    # load_spoken's cached-text audit module sets its own output default; its
    # pure transform does not write anything. Restore helper guard area only.
    base.AREA=AREA;paced.AREA=AREA
    model_hash=hashlib.sha256();loader_error=[];last_report=[0]
    stream=MODEL.open('rb')
    @READ
    def read(_,destination,length):
        try:
            done=0
            while done<length:
                amount=min(65536,length-done);guard.pace(amount)
                block=stream.read(amount)
                if not block:break
                c.memmove(destination+done,block,len(block));model_hash.update(block);done+=len(block)
                position=stream.tell()
                if position//(16*1024*1024)>last_report[0]:
                    last_report[0]=position//(16*1024*1024)
                    guard.status(phase='paced_full_large_v3_native_model_loading',model_bytes_read=position,
                                 expected_model_bytes=model_receipt['bytes'],CPU_RSS_bytes=psutil.Process().memory_info().rss)
            return done
        except BaseException as exc:
            loader_error.append(repr(exc));return 0
    @EOF
    def eof(_):return stream.tell()>=model_receipt['bytes']
    @CLOSE
    def close(_):pass
    loader=Loader(None,read,eof,close)
    cp=runtime.context_defaults().contents
    cp.use_gpu=False;cp.flash_attn=True;cp.dtw_token_timestamps=False
    guard.status(phase='starting_paced_full_large_v3_native_initialization',ABI=abi,
                 model_path=str(MODEL),source_manifest_sha256=selection['manifest_sha256'])
    context=None;completed=False
    try:
        context=runtime.init(c.byref(loader),cp)
        stream.close()
        if not context:raise RuntimeError('Native initialization failed under unchanged2GB cap; see native log')
        assert not loader_error and model_hash.hexdigest()==MODEL_SHA
        dimensions={name:fn(context) for name,fn in runtime.model.items()}
        assert dimensions['audio_layer']==32 and dimensions['text_layer']==32 and dimensions['mels']==128
        native_fit={'kind':'actual_native_full_large_v3_CPU_fit_v6','at_utc':base.now(),'pid':os.getpid(),
                    'native_init_succeeded':True,'model_SHA_verified_during_paced_direct_destination_load':MODEL_SHA,
                    'model_revision':MODEL_REV,'quantization':'Q5_0','dimensions':dimensions,
                    'ABI':abi,'CPU_RSS_bytes':psutil.Process().memory_info().rss,'peak_CPU_RSS_bytes':guard.peak_rss,
                    'hard_process_memory_limit_bytes':LIMIT,'use_gpu':False,'mmap_claim':False,
                    'whole_model_bytes_copy_created':False,'CPU_threads':1}
        base.write_json(AREA/'actual_native_full_large_v3_CPU_fit_v6.json',native_fit)
        guard.status(phase='actual_full_large_v3_CPU_initialization_positive',native_fit=native_fit)
        @ABORT
        def abort(_):
            try:guard.wait();return False
            except BaseException:
                return True
        params=runtime.full_defaults(0).contents
        params.n_threads=1;params.n_max_text_ctx=0;params.no_context=True
        params.translate=False;params.language=b'auto';params.detect_language=False
        params.no_timestamps=False;params.single_segment=False
        params.print_progress=False;params.print_realtime=False;params.print_timestamps=False
        params.token_timestamps=False;params.greedy.best_of=1;params.beam_search.beam_size=1
        params.temperature=0.;params.temperature_inc=0.;params.initial_prompt=None
        params.prompt_tokens=None;params.prompt_n_tokens=0;params.max_tokens=256
        params.abort_callback=c.cast(abort,c.c_void_p).value
        ledger=AREA/'full_large_v3_sanity_ledger_v6.jsonl'
        prior=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
        done_ids={r['segment_id'] for r in prior}
        base.write_json(AREA/'full_large_v3_sanity_protocol_v6.json',
            {'pinned_at_utc':base.now(),'manifest_sha256':selection['manifest_sha256'],'model_revision':MODEL_REV,
             'model_sha256':MODEL_SHA,'model_family':'full original large-v3;32encoder/32decoder;128mels',
             'runtime':'whisper.cpp v1.8.2 official CPU','quantization':'Q5_0','greedy_best_of':1,
             'temperature':0,'fallback':False,'language':'auto','translate':False,'context_from_prior_clips':False,
             'max_tokens_per_segment':256,'CPU_threads':1,'source_text_used_as_prompt':False,
             'normalizer_fixtures_passed':fixture_count,'no_acceptance_or_correction':True})
        for index,row in enumerate(rows,1):
            if row['segment_id'] in done_ids:continue
            guard.wait();start=time.monotonic();paused_start=guard.paused_seconds
            guard.status(phase='full_large_v3_sanity_decoding_source_on_demand',sample=index,samples_total=len(rows),segment_id=row['segment_id'])
            pcm,decode=paced.decode_paced(row,guard)
            guard.wait();inference_start=time.monotonic()
            guard.status(phase='full_large_v3_sanity_actual_CPU_inference',sample=index,segment_id=row['segment_id'],CPU_RSS_bytes=psutil.Process().memory_info().rss)
            result=runtime.full(context,params,pcm.ctypes.data_as(c.POINTER(c.c_float)),len(pcm))
            if result!=0:raise RuntimeError(f'Native full-large-v3 inference failed code{result} on {row["segment_id"]}')
            segments=[{'start_s':runtime.t0(context,j)/100,'end_s':runtime.t1(context,j)/100,
                       'text':runtime.text(context,j).decode('utf-8')} for j in range(runtime.segments(context))]
            text=''.join(s['text'] for s in segments).strip()
            language=runtime.language_name(runtime.language_id(context)).decode()
            entry={'segment_id':row['segment_id'],'recording_id':row['recording_id'],'shard':row['shard'],
                   'duration_s':row['duration_s'],'selection_reasons':row['selection_reasons'],
                   'source_transcript':row['transcript'],'small_hypothesis':row['independent_hypothesis'],
                   'raw_small_WER':row['source_vs_independent_WER'],'raw_small_flags':row['triage_flags'],
                   'full_large_v3_hypothesis':text,'full_large_v3_detected_language':language,'segments':segments,
                   'source_vs_small':score_pair(row['transcript'],row['independent_hypothesis'],spoken),
                   'source_vs_full_large_v3':score_pair(row['transcript'],text,spoken),
                   'small_vs_full_large_v3':score_pair(row['independent_hypothesis'],text,spoken),
                   'decode':decode,'PCM_matches_prior_QC_decode':decode['PCM_f32le_sha256']==row.get('PCM_f32le_sha256'),
                   'model_sha256':MODEL_SHA,'model_revision':MODEL_REV,'quantization':'Q5_0','CPU_only':True,
                   'completed_at_utc':base.now(),'elapsed_wall_s':time.monotonic()-start,
                   'inference_wall_s':time.monotonic()-inference_start,'GPU_sensitive_pause_s':guard.paused_seconds-paused_start,
                   'CPU_RSS_bytes':psutil.Process().memory_info().rss,'peak_CPU_RSS_bytes':guard.peak_rss,
                   'audio_correctness_established':False,'source_label_corrected':False,'accepted_TRAIN':False}
            del pcm
            base.append_json(ledger,entry);prior.append(entry);done_ids.add(row['segment_id'])
            lower=sum(r['source_vs_full_large_v3']['frozen']['wer']<r['source_vs_small']['frozen']['wer'] for r in prior)
            report={'kind':'bounded_full_original_large_v3_CPU_sanity_progress_v6','updated_at_utc':base.now(),
                    'samples_declared':len(rows),'samples_completed':len(prior),'lower_source_disagreement_with_large':lower,
                    'same_source_disagreement_with_large':sum(r['source_vs_full_large_v3']['frozen']['wer']==r['source_vs_small']['frozen']['wer'] for r in prior),
                    'higher_source_disagreement_with_large':sum(r['source_vs_full_large_v3']['frozen']['wer']>r['source_vs_small']['frozen']['wer'] for r in prior),
                    'model_family':'full original large-v3','quantization':'Q5_0','CPU_only':True,
                    'manifest_sha256':selection['manifest_sha256'],'ledger':str(ledger),'accepted_TRAIN_hours':0,
                    'limitations':['Worst-case selected subset only, not a population error estimate.',
                                   'Neither recognizer is groundtruth; no listening/admission/correction claimed.',
                                   'Q5_0 CPU outputs can differ from full-precision decoding.'],
                    'last_segment_id':entry['segment_id'],'last_large_hypothesis':text,
                    'last_source_vs_large_frozen_WER':entry['source_vs_full_large_v3']['frozen']['wer'],
                    'last_source_vs_small_frozen_WER':entry['source_vs_small']['frozen']['wer'],
                    'peak_CPU_RSS_bytes':guard.peak_rss}
            base.write_json(AREA/'full_large_v3_sanity_report_v6.json',report)
            guard.status(phase='actual_full_large_v3_sanity_sample_durable',samples_completed=len(prior),sample=index,
                         segment_id=row['segment_id'],source_vs_large_WER=entry['source_vs_full_large_v3']['frozen']['wer'],
                         source_vs_small_WER=entry['source_vs_small']['frozen']['wer'],peak_CPU_RSS_bytes=guard.peak_rss)
        completed=True
        guard.status(phase='completed_bounded_full_large_v3_sanity',samples_completed=len(prior),actual_CLOSED0=True)
    finally:
        if context:runtime.free(context)
        if not stream.closed:stream.close()
        log_file.close()
        resume_reference_worker('bounded_full_large_v3_sanity_completed' if completed else 'bounded_native_sanity_failed_preserve_references_and_receipts')


if __name__=='__main__':main()
