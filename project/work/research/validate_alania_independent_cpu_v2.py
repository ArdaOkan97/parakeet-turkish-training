"""Resumable independent small-model triage; never admits or rewrites TRAIN."""
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import sqlite3
import sys
import tempfile
import time
import urllib.request

from cpu_preparation_guard_v2 import AREA,PROJECT,ROOT,Guard,append_json,now,scoped,sha256,write_json

sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from metrics import edit_distance,normalize_text

MODEL_REPO='Systran/faster-whisper-small'
MODEL_REV='536b0662742c02347bc0e980a01041f333bce120'
MODEL=ROOT/'cache/models/Systran_faster-whisper-small'/MODEL_REV
MAX_MEMBER=256*1024*1024
LEDGER=AREA/'independent_QC_ledger_v2.jsonl'

def get_json(url):
    with urllib.request.urlopen(url,timeout=60) as response:return json.load(response)

def model_ready(guard):
    info=get_json(f'https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REV}?blobs=true')
    if info['sha']!=MODEL_REV:raise ValueError('Recognizer revision mismatch')
    files=[]
    for r in info['siblings']:
        name=r['rfilename']
        if name not in ['config.json','model.bin','tokenizer.json','vocabulary.json','vocabulary.txt','preprocessor_config.json','README.md','.gitattributes']:continue
        p=scoped(MODEL/name);p.parent.mkdir(parents=True,exist_ok=True)
        expected=r.get('lfs',{}).get('sha256');size=r.get('size')
        if not p.exists():
            guard.status(phase='downloading_pinned_CPU_recognizer',model_file=name)
            part=p.with_suffix(p.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
            headers={'User-Agent':'independent-CPU-QC-v2'}
            if offset:headers['Range']=f'bytes={offset}-'
            req=urllib.request.Request(f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/{name}',headers=headers)
            with urllib.request.urlopen(req,timeout=60) as response:
                if offset and response.status!=206:raise ValueError('Model resume refused; partial preserved')
                with part.open('ab' if offset else 'wb') as out:
                    while True:
                        guard.wait();block=response.read(2*1024*1024)
                        if not block:break
                        out.write(block)
            if size is not None and part.stat().st_size!=size:raise ValueError('Model length mismatch')
            if expected and sha256(part)!=expected:raise ValueError('Model LFS digest mismatch')
            part.replace(p)
        actual=sha256(p)
        if expected and actual!=expected:raise ValueError('Pinned local model changed')
        files.append({'name':name,'bytes':p.stat().st_size,'sha256':actual,'LFS_sha256':expected})
    if not (MODEL/'model.bin').exists():raise ValueError('Missing pinned CPU model')
    package_names=['faster-whisper','ctranslate2','av','tokenizers','huggingface-hub','numpy','onnxruntime','psutil']
    receipt={'repository':MODEL_REPO,'revision':MODEL_REV,'files':files,
        'packages':{n:importlib.metadata.version(n) for n in package_names},'device':'cpu','compute_type':'int8',
        'cpu_threads':1,'num_workers':1,'created_at_utc':now(),'license':'MIT model card; data rights still unresolved',
        'sources':['https://github.com/SYSTRAN/faster-whisper','https://huggingface.co/Systran/faster-whisper-small']}
    write_json(AREA/'independent_recognizer_receipt_v2.json',receipt)

def manifest():
    p=AREA/'independent_QC_manifest_v2.jsonl'
    if p.exists():return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    original=ROOT/'analysis/alania_yodas_preparation_v1/stratified_QC_sample_v1.jsonl'
    chosen=[{**json.loads(x),'QC_phase':'original72_stratified_independent_semantics'} for x in original.read_text(encoding='utf-8').splitlines()]
    seen={r['recording_id'] for r in chosen}
    db=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA cache_size=-4096')
    for shard in ['0027','0006','0014']:
        n=0
        for raw in db.execute('SELECT * FROM candidates WHERE eligible=1 AND shard=? ORDER BY pilot_rank',(shard,)):
            row=dict(raw)
            if row['recording_id'] in seen:continue
            seen.add(row['recording_id']);row['QC_phase']='additional240_distinct_recording_rank_pilot';chosen.append(row);n+=1
            if n==80:break
        if n!=80:raise ValueError('Insufficient pilot recordings')
    db.close();p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen),encoding='utf-8')
    write_json(AREA/'independent_QC_plan_v2.json',{'created_at_utc':now(),'manifest':str(p),'manifest_sha256':sha256(p),
        'declared_sample_segments':len(chosen),'original_stratified72':72,'additional_distinct_recordings':240,
        'candidate_hours':sum(x['duration_s'] for x in chosen)/3600,'model_repository':MODEL_REPO,'model_revision':MODEL_REV,
        'sampling':'Original diagnostic stratified72, followed by80 new distinct recordings per existing staged shard in deterministic metadata rank order',
        'interpretation':'Independent recognizer disagreement is triage only, never transcript ground truth or automatic TRAIN acceptance',
        'accepted_TRAIN_hours':0,'maximum_compressed_member_bytes':MAX_MEMBER,'max_clip_seconds':20,
        'triage_flags':{'language_probability_min':.8,'WER_disagreement_max':.30,'source_words_per_second_max':5.5,
            'leading_word_gap_max_s':1.5,'trailing_word_gap_max_s':2.5},
        'alignment_scope':'Independent word timestamps support span plausibility only; no source-transcript forced alignment proof'})
    return chosen

def decode(row,guard):
    import av
    import numpy as np
    shard=row['shard'];rid=row['recording_id'];start=row['start_s'];duration=row['duration_s']
    if not 2<=duration<=20 or start<0:raise ValueError('Out of bounds segment')
    db=sqlite3.connect('file:'+(ROOT/f'indexes/yodas3_tar_{shard}.sqlite').as_posix()+'?mode=ro',uri=True)
    extent=db.execute('SELECT offset,bytes FROM recordings WHERE recording_id=?',(rid,)).fetchone();db.close()
    if extent is None:raise ValueError('Recording missing from compressed source')
    offset,size=extent
    if size>MAX_MEMBER:raise ValueError('Compressed recording exceeds unchanged256MiB member limit')
    temporary=None;h=hashlib.sha256();before=time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(prefix='CPU-QC-owned-',suffix='.webm',dir=ROOT/'temp',delete=False) as out:
            temporary=pathlib.Path(out.name)
            with (ROOT/f'datasets/yodas3/audio/{shard}.tar').open('rb') as archive:
                archive.seek(offset);left=size
                while left:
                    guard.wait();block=archive.read(min(2*1024*1024,left))
                    if not block:raise ValueError('Truncated compressed extent')
                    out.write(block);h.update(block);left-=len(block)
        n=round(duration*16000);pcm=np.zeros(n,dtype=np.float32);coverage=np.zeros(n,dtype=np.bool_)
        def retain(frame):
            if frame.pts is None:raise ValueError('Decoded audio lacks time binding')
            at=round((float(frame.pts*frame.time_base)-start)*16000)
            arr=frame.to_ndarray().reshape(-1)
            lo=max(0,at);hi=min(n,at+len(arr))
            if hi>lo:pcm[lo:hi]=arr[lo-at:hi-at];coverage[lo:hi]=True
        with av.open(str(temporary),'r') as container:
            stream=container.streams.audio[0];stream.codec_context.thread_count=1;stream.codec_context.thread_type='NONE'
            container.seek(max(0,round((start-.10)*1_000_000)),backward=True,any_frame=False)
            resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
            for frame in container.decode(stream):
                if time.monotonic()-before>90:raise TimeoutError('Bounded audio decoder90s exceeded')
                if frame.pts is None:raise ValueError('Source audio lacks PTS')
                if float(frame.pts*frame.time_base)>start+duration+.15:break
                for converted in resampler.resample(frame):retain(converted)
            for converted in resampler.resample(None):retain(converted)
        if not np.isfinite(pcm).all() or float(coverage.mean())<.995:raise ValueError('Insufficient finite timestamp-bound audio coverage')
        return pcm,{'compressed_member_bytes':size,'compressed_member_sha256':h.hexdigest(),
            'PCM_f32le_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),'decoded_samples':n,'sample_rate':16000,
            'timestamp_coverage_fraction':float(coverage.mean()),'peak_abs':float(np.abs(pcm).max()),
            'rms':float(np.sqrt(np.mean(pcm*pcm))),'decoder':'Pinned E:PyAV;thread_count1;original compressed source;accurate PTS slice',
            'decode_elapsed_s':time.monotonic()-before,'permanent_PCM_or_WAV_written':False}
    finally:
        # Delete only this exact newly created compressed temporary member.
        if temporary is not None:
            scoped(temporary).unlink(missing_ok=True)

def summarize(rows,guard,total):
    good=[r for r in rows if r.get('independent_recognizer_completed')]
    counts=collections.Counter(flag for r in good for flag in r['triage_flags'])
    result={'kind':'independent_small_CPU_QC_triage_v2','updated_at_utc':now(),'declared_segments':total,
        'durable_observations':len(rows),'recognizer_completed':len(good),'quarantine_triage_count':sum(bool(r['triage_flags']) for r in good),
        'no_automatic_triage_flags_count':sum(not r['triage_flags'] for r in good),'excluded_or_failed_count':len(rows)-len(good),
        'triage_reason_counts':dict(counts),'independent_detected_languages':dict(collections.Counter(r['independent_language'] for r in good)),
        'independently_processed_candidate_hours':sum(r['duration_s'] for r in good)/3600,
        'accepted_TRAIN_hours':0,'source_labels_are_not_ground_truth':True,'small_model_disagreement_is_not_correction':True,
        'full_source_audio_dedup_heldout_exclusion_and_listening_still_required':True,'GPU_used':False,'TEST_predictions_read':False,
        'manifest':str(AREA/'independent_QC_manifest_v2.jsonl'),'ledger':str(LEDGER)}
    write_json(AREA/'independent_QC_report_v2.json',result)
    guard.status(phase='independent_CPU_triage_running',completed_samples=len(rows),declared_samples=total,
        recognizer_completed=len(good),quarantine_triage_count=result['quarantine_triage_count'],triage_reason_counts=dict(counts))

def main():
    guard=Guard('independent_QC_v2');chosen=manifest();model_ready(guard)
    import numpy as np
    import psutil
    from faster_whisper import WhisperModel
    guard.wait();guard.status(phase='loading_pinned_CPU_int8_model',model=str(MODEL))
    model=WhisperModel(str(MODEL),device='cpu',compute_type='int8',cpu_threads=1,num_workers=1,local_files_only=True)
    rows=[json.loads(x) for x in LEDGER.read_text(encoding='utf-8').splitlines()] if LEDGER.exists() else []
    done={x['segment_id'] for x in rows}
    if len(done)!=len(rows) or not done.issubset({x['segment_id'] for x in chosen}):raise ValueError('Resume ledger mismatch')
    for row in chosen:
        if row['segment_id'] in done:continue
        guard.wait();before=time.monotonic()
        result={k:row[k] for k in ['segment_id','recording_id','shard','start_s','end_s','duration_s','transcript','QC_phase']}
        try:
            pcm,decoded=decode(row,guard);result.update(decoded)
            guard.status(phase='recognizing_CPU_sample',active_segment=row['segment_id'],completed_samples=len(rows),declared_samples=len(chosen))
            segments,info=model.transcribe(pcm,language=None,beam_size=1,best_of=1,temperature=0,
                condition_on_previous_text=False,word_timestamps=True,vad_filter=False)
            segments=list(segments);hyp=' '.join(s.text.strip() for s in segments).strip();del pcm
            words=[{'start':w.start,'end':w.end,'word':w.word,'probability':w.probability} for s in segments for w in (s.words or [])]
            refnorm=normalize_text(row['transcript'],'tr');hypnorm=normalize_text(hyp,'tr')
            wer=edit_distance(refnorm.split(),hypnorm.split())/max(1,len(refnorm.split()))
            cer=edit_distance(refnorm,hypnorm)/max(1,len(refnorm))
            flags=[]
            if info.language!='tr':flags.append('independent_detected_non_Turkish')
            if info.language_probability<.8:flags.append('independent_LID_uncertain')
            if wer>.30:flags.append('source_independent_ASR_disagreement')
            if len(refnorm.split())/row['duration_s']>5.5:flags.append('source_text_rate_high')
            if not words:flags.append('no_independent_words')
            elif words[0]['start']>1.5 or row['duration_s']-words[-1]['end']>2.5:flags.append('speech_span_boundary_gap')
            if any(w['start']<-.1 or w['end']>row['duration_s']+.25 or w['end']<w['start'] for w in words):flags.append('independent_word_time_out_of_bounds')
            result.update(independent_recognizer_completed=True,independent_hypothesis=hyp,independent_language=info.language,
                independent_language_probability=info.language_probability,independent_all_language_probs=info.all_language_probs,
                source_vs_independent_WER=wer,source_vs_independent_CER=cer,independent_words=words,
                source_words_per_second=len(refnorm.split())/row['duration_s'],triage_flags=flags,
                outcome='quarantine_for_further_review' if flags else 'no_automatic_flags_still_not_accepted',
                timestamp_scope='Independent hypothesis timestamps only; no forced alignment certification')
        except Exception as error:
            result.update(independent_recognizer_completed=False,outcome='excluded_untested_or_failed',error=repr(error),triage_flags=[])
        result.update(completed_at_utc=now(),elapsed_s=time.monotonic()-before,CPU_process_rss_bytes=psutil.Process().memory_info().rss,accepted_TRAIN=False)
        if result['CPU_process_rss_bytes']>3_000_000_000:raise RuntimeError('Own CPU validation3GB RSS bound exceeded; row not committed')
        append_json(LEDGER,result);rows.append(result);done.add(row['segment_id']);summarize(rows,guard,len(chosen))
    summarize(rows,guard,len(chosen));guard.status(phase='completed_triage_only',actual_CLOSED0=True)

if __name__=='__main__':main()
