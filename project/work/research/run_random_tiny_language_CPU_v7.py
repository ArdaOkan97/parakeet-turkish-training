"""Seeded random cohort and multilingual Whisper-tiny language-only screening."""
import ctypes as c
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
import urllib.request

ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import numpy as np
import psutil
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced
from whisper_cpp_cpu_bindings_v6 import Runtime,Loader,READ,EOF,CLOSE,LOG
from run_large_v3_sanity_cpu_v6 import hard_process_memory_job

AREA=ROOT/'analysis/CPU_validation_v7'
base.AREA=AREA;paced.AREA=AREA
SEED=2026101607
REV='c521a4b02f422512d734391fdf08bb08c0862f68'
TINY_SHA='be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21'
TINY_BYTES=77_691_713
PROOF=ROOT/'analysis/three_epoch_entry_v1/resource_stop_78852_v1/actual_partial_CLOSED78852_v1.json'
PROOF_SHA='10b16a25834b7df64f2f035ce95d4123773b58f0659f3d9f1a0522960e46cef1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Guard(paced.PacedGuard):
    def __init__(self):
        base.Guard.__init__(self,'random_tiny_language_CPU_v7')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=None;self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1048576
        self.status(CPU_threads=1,maximum_audio_workers=1,max_own_RSS_bytes=2_000_000_000,
                    read_rate_limit_bytes_s=1048576,language_threshold=.70,tiny_transcription_enabled=False)

    def gate_snapshot(self):
        current=time.monotonic()
        if getattr(self,'closed_snapshot',None) is not None and current-getattr(self,'closed_poll',0)<1:
            return self.closed_snapshot
        queue=json.loads((ROOT/'experiment_queue_state.json').read_text(encoding='utf-8-sig'))
        if queue.get('GPU_running') is False:
            owner=json.loads((PROJECT/'work/gpu_training_owner.json').read_text(encoding='utf-8-sig'))
            path=Path(owner['actual_controller_loop'])
            if path.suffix.lower()!='.json':path=path/'loop_status.json'
            loop=json.loads(path.read_text(encoding='utf-8-sig'))
            allowed=(PROOF.exists() and sha(PROOF)==PROOF_SHA and
                     queue.get('child_exit_code')==0 and loop.get('child_exit_code')==0 and
                     queue.get('status')=='train_window_ended' and loop.get('status')=='train_window_ended' and
                     queue.get('run_id')==loop.get('run_id') and
                     int(queue.get('completed_step',0))==78852 and int(loop.get('completed_step',0))==78852)
            self.closed_poll=current
            self.closed_snapshot={'allow':allowed,'reason':'proved_actual_CLOSED0_noGPU_fresh_system_memory_guard' if allowed else 'closed_GPU_proof_or_identity_mismatch',
                    'queue_status':queue.get('status'),'loop_status':loop.get('status'),'step':queue.get('completed_step'),
                    'closed_proof_SHA':PROOF_SHA,'fresh_system_memory':base.memory()}
            return self.closed_snapshot
        self.closed_snapshot=None
        return super().gate_snapshot()


def pin_manifests(guard):
    receipt_path=AREA/'random_sampling_receipt_v7.json'
    if receipt_path.exists():
        result=json.loads(receipt_path.read_text());assert sha(result['random64_manifest'])==result['random64_sha256']
        assert sha(result['random24_manifest'])==result['random24_sha256'];return result
    shards=sorted(p.stem for p in (ROOT/'datasets/yodas3/audio').glob('*.tar'))
    assert len(shards)==11
    allowed={};oversized=set()
    for shard in shards:
        db=sqlite3.connect('file:'+(ROOT/f'indexes/yodas3_tar_{shard}.sqlite').as_posix()+'?mode=ro',uri=True)
        for member_index,(rid,size) in enumerate(db.execute('SELECT recording_id,bytes FROM recordings')):
            if member_index%256==0:guard.pace(256*64)
            if size<=256*1024*1024:allowed[(shard,rid)]=size
            else:oversized.add((shard,rid))
        db.close()
    buckets={(s,d):[] for s in shards for d in range(3)}
    counts={key:0 for key in buckets};excluded=0
    db=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA cache_size=-1024')
    query='SELECT segment_id,recording_id,shard,duration_s FROM candidates WHERE eligible=1 AND shard IN ('+','.join('?' for _ in shards)+')'
    scanned=0
    for r in db.execute(query,shards):
        if scanned%128==0:guard.pace(128*512)
        scanned+=1
        if (r['shard'],r['recording_id']) not in allowed:excluded+=1;continue
        duration_bin=0 if r['duration_s']<5 else 1 if r['duration_s']<12 else 2
        key=(r['shard'],duration_bin);counts[key]+=1
        rank=int.from_bytes(hashlib.blake2b(f'{SEED}|{r["segment_id"]}'.encode(),digest_size=16).digest(),'big')
        item=(-rank,r['segment_id'])
        if len(buckets[key])<2:heapq.heappush(buckets[key],item)
        elif item>buckets[key][0]:heapq.heapreplace(buckets[key],item)
        if scanned%20000==0:guard.status(phase='pinning_seeded_random64_before_predictions',source_candidate_rows_scanned=scanned)
    keys=sorted(buckets);random.Random(SEED).shuffle(keys)
    selected=[];strata=[]
    for index,key in enumerate(keys):
        quota=2 if index<31 else 1
        assert len(buckets[key])>=quota
        chosen=[sid for rank,sid in sorted(buckets[key],key=lambda p:(-p[0],p[1]))[:quota]]
        for sid in chosen:
            row=dict(db.execute('SELECT * FROM candidates WHERE segment_id=?',(sid,)).fetchone())
            row.update(QC_group='seeded_stratified_random64_v7',sampling_seed=SEED,
                       sampling_stratum={'shard':key[0],'duration_bin':['2-5s','5-12s','12-20s'][key[1]]})
            selected.append(row)
        strata.append({'shard':key[0],'duration_bin':['2-5s','5-12s','12-20s'][key[1]],'population_clips':counts[key],'selected':quota})
    db.close();assert len(selected)==64 and len({r['segment_id'] for r in selected})==64
    selected.sort(key=lambda r:r['segment_id'])
    review=random.Random(SEED+24).sample(selected,24)
    ids24={r['segment_id'] for r in review}
    for row in selected:row['preselected_full_large_v3_random24']=row['segment_id'] in ids24
    path64=AREA/'random64_tiny_language_manifest_v7.jsonl';path24=AREA/'random24_full_large_v3_manifest_v7.jsonl'
    for path,rows in ((path64,selected),(path24,[next(r for r in selected if r['segment_id']==x['segment_id']) for x in review])):
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    result={'kind':'preprediction_seeded_random64_and_nested_random24_v7','pinned_at_utc':base.now(),'seed':SEED,
            'random24_seed':SEED+24,'random64_manifest':str(path64),'random64_sha256':sha(path64),
            'random24_manifest':str(path24),'random24_sha256':sha(path24),'random64_clips':64,'random24_clips':24,
            'selection':'Seeded hash ranks within33shard/duration strata;31seeded strata get2clips,2get1; random24 sampled BEFORE any tiny/large predictions',
            'population':'All source-metadata-eligible clips across11local pinned shards with unchanged locally bounded<=256MiB member API',
            'source_candidate_rows_scanned':scanned,'bounded_candidate_population':sum(counts.values()),
            'source_rows_excluded_by_existing_compressed_member_bound':excluded,'oversized_source_recordings':len(oversized),
            'per_stratum':strata,'shards':shards,'random24_includes_language_failures_without_reselection':True,
            'small_predictions_or_flags_used_for_selection':False,'predictions_read_or_generated_before_pin':False,
            'sampling_is_stratified_not_population_simple_random':True,'new_TRAIN_hours_accepted':0,
            'separate_worst12_manifest':str(ROOT/'analysis/CPU_validation_v6/worst_samples_full_large_v3_manifest_v6.jsonl')}
    base.write_json(receipt_path,result)
    guard.status(phase='random64_and_random24_manifests_PINNED',sampling_receipt=str(receipt_path),random64_sha256=result['random64_sha256'],random24_sha256=result['random24_sha256'])
    return result


class TinyStream:
    """One-pass paced download/direct-load; never copies a whole model in RAM."""
    def __init__(self,path,guard):
        self.final=path;self.partial=path.with_suffix('.bin.partial');self.guard=guard
        path.parent.mkdir(parents=True,exist_ok=True)
        self.complete=path.exists();self.path=path if self.complete else self.partial
        self.f=self.path.open('rb' if self.complete else 'r+b' if self.path.exists() else 'w+b')
        self.have=self.path.stat().st_size;self.position=0;self.response=None;self.range_end=0;self.h=hashlib.sha256();self.last=0
        self.pending=b''
        assert self.have<=TINY_BYTES

    def read(self,n):
        if self.position>=TINY_BYTES:return b''
        if self.pending:
            block=self.pending[:n];self.pending=self.pending[len(block):]
            self.position+=len(block);self.h.update(block)
            return block
        requested=n
        n=min(65536,TINY_BYTES-self.position);self.guard.pace(n)
        if self.position<self.have:
            n=min(n,self.have-self.position);self.f.seek(self.position);block=self.f.read(n)
        else:
            if self.response is None or self.position>=self.range_end:
                if self.response:self.response.close()
                end=min(self.position+4*1024*1024,TINY_BYTES)
                request=urllib.request.Request(f'https://huggingface.co/ggerganov/whisper.cpp/resolve/{REV}/ggml-tiny.bin',
                                               headers={'Range':f'bytes={self.position}-{end-1}','User-Agent':'bounded-tiny-LID-v7'})
                self.response=urllib.request.urlopen(request,timeout=60)
                cr=self.response.headers.get('Content-Range','')
                assert self.response.status==206 and cr.startswith(f'bytes {self.position}-') and cr.endswith('/'+str(TINY_BYTES)),cr
                self.range_end=end
            block=self.response.read(min(n,self.range_end-self.position))
            if not block:
                self.response.close();self.response=None;return self.read(n)
            self.f.seek(self.have);self.f.write(block);self.have+=len(block)
        self.pending=block
        if self.position//(4*1024*1024)>self.last:
            self.last=self.position//(4*1024*1024)
            self.guard.status(phase='tiny_official_multilingual_model_paced_acquisition_and_loading',model_bytes_read=self.position,total_model_bytes=TINY_BYTES)
        return self.read(requested)

    def finish(self):
        assert self.position==TINY_BYTES and self.h.hexdigest()==TINY_SHA
        if self.response:self.response.close()
        self.f.flush();self.f.close()
        if not self.complete:self.partial.replace(self.final)
        base.write_json(self.final.with_suffix('.bin.receipt.json'),{'repository':'ggerganov/whisper.cpp','revision':REV,
                 'file':'ggml-tiny.bin','sha256':TINY_SHA,'bytes':TINY_BYTES,'multilingual':True,'completed_at_utc':base.now()})


def main():
    AREA.mkdir(parents=True,exist_ok=True)
    guard=Guard();guard.wait();job=hard_process_memory_job()
    assert psutil.Process(1912).status()==psutil.STATUS_STOPPED
    assert (ROOT/'analysis/CPU_validation_v6/ROOT_clip_build_complete_sequence_receipt_v6.json').exists()
    base.write_json(AREA/'CPU_successor_admission_v7.json',{'at_utc':base.now(),'pid':os.getpid(),
       'explicit_human_new_tiny_and_random_large_direction':True,'closed_GPU_proof':str(PROOF),'closed_GPU_proof_SHA':PROOF_SHA,
       'old_waiting_setup_retired':str(ROOT/'analysis/CPU_validation_v6/owned_waiting_setup_retirement_v7.json'),
       'all_caps_unchanged':True,'large_CPU_model_download_or_load_enabled':False,
       'original_large_cpp_partial_preserved_bytes':226572871,'large_inference_owner':'ROOT, separate proposed GPU batch after this CPU audio phase closes',
       'source_path':str(Path(__file__)),'source_sha256':sha(Path(__file__))})
    sampling=pin_manifests(guard)
    runtime=Runtime(ROOT/'scratch/whisper_cpp_v1.8.2_CPU_x64/Release');abi=runtime.verify_defaults()
    log_file=(AREA/'tiny_native_log_v7.txt').open('a',encoding='utf-8',buffering=1)
    @LOG
    def log(level,message,_):log_file.write(message.decode('utf-8',errors='replace'))
    runtime.log_set(log,None)
    stream=TinyStream(ROOT/'models/whisper_cpp_multilingual_tiny/ggml-tiny.bin',guard)
    errors=[]
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
    def eof(_):return stream.position>=TINY_BYTES
    @CLOSE
    def close(_):pass
    loader=Loader(None,read,eof,close);cp=runtime.context_defaults().contents
    cp.use_gpu=False;cp.flash_attn=True;cp.dtw_token_timestamps=False
    context=runtime.init(c.byref(loader),cp)
    if not context:raise RuntimeError('Tiny native init failed: '+repr(errors))
    assert not errors;stream.finish()
    assert runtime.model['audio_layer'](context)==4 and runtime.model['mels'](context)==80
    dll=runtime.dll
    dll.whisper_pcm_to_mel.argtypes=[c.c_void_p,c.POINTER(c.c_float),c.c_int,c.c_int];dll.whisper_pcm_to_mel.restype=c.c_int
    dll.whisper_lang_max_id.argtypes=[];dll.whisper_lang_max_id.restype=c.c_int
    dll.whisper_lang_auto_detect.argtypes=[c.c_void_p,c.c_int,c.c_int,c.POINTER(c.c_float)];dll.whisper_lang_auto_detect.restype=c.c_int
    lang_count=dll.whisper_lang_max_id()+1
    languages=[runtime.language_name(i).decode() for i in range(lang_count)]
    rows=[json.loads(line) for line in Path(sampling['random64_manifest']).read_text(encoding='utf-8').splitlines()]
    random_ids={r['segment_id'] for r in rows}
    supplemental=[json.loads(line) for line in (ROOT/'analysis/CPU_validation_v6/worst_samples_full_large_v3_manifest_v6.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in supplemental:
        if row['segment_id'] not in random_ids:row['QC_group']='historical_worst12_supplement_only';rows.append(row)
    ledger=AREA/'tiny_LID_per_clip_ledger_v7.jsonl'
    prior=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    done_ids={r['segment_id'] for r in prior}
    base.write_json(AREA/'tiny_language_protocol_v7.json',{'pinned_at_utc':base.now(),'tiny_model_SHA':TINY_SHA,'tiny_revision':REV,
         'runtime':abi,'tiny_transcription':False,'API':'whisper_pcm_to_mel then whisper_lang_auto_detect; no token generation',
         'eligibility_rule':'Full clip AND every tested window detected_language=tr AND Turkish probability>=0.70',
         'quarantines':'Any top nontr with top probability>=0.70 -> confident_nontr; otherwise any Turkish probability<0.70 -> uncertain; categories stored separately',
         'windows':'For clips>=8s, nonoverlapping4s blocks (last>=2s); energy-derived speech-candidate labels only; fullclip always checked',
         'limitations':'70% is an uncalibrated screening threshold; short-window/padding sensitivity and untested boundaries can miss code-switching; no truth/admission claim',
         'random64_denominator':64,'supplemental_worst_separate':True,'large_CPP_partial_untouched':True})
    try:
        for index,row in enumerate(rows,1):
            if row['segment_id'] in done_ids:continue
            guard.wait();start=time.monotonic()
            try:
                pcm,decode=paced.decode_paced(row,guard)
                windows=[('full_clip',0.,len(pcm)/16000)]
                if row['duration_s']>=8:
                    windows += [('disjoint_window',float(begin),min(float(begin+4),row['duration_s']))
                                for begin in range(0,int(row['duration_s']),4) if row['duration_s']-begin>=2]
                observations=[]
                for kind,begin,end in windows:
                    guard.wait();clip=np.ascontiguousarray(pcm[round(begin*16000):round(end*16000)],dtype=np.float32)
                    frames=clip[:len(clip)//320*320].reshape(-1,320)
                    rms=np.sqrt(np.mean(frames*frames,axis=1)) if len(frames) else np.array([0.])
                    threshold=max(.003,float(np.percentile(rms,20))*2)
                    speech_fraction=float(np.mean(rms>=threshold))
                    assert dll.whisper_pcm_to_mel(context,clip.ctypes.data_as(c.POINTER(c.c_float)),len(clip),1)==0
                    probs=(c.c_float*lang_count)();detected=dll.whisper_lang_auto_detect(context,0,1,probs)
                    assert detected>=0
                    probabilities={lang:float(probs[j]) for j,lang in enumerate(languages)}
                    top=languages[detected];top_probability=probabilities[top];turkish=probabilities['tr']
                    observations.append({'scope':kind,'start_s_in_clip':begin,'end_s_in_clip':end,
                         'detected_language':top,'detected_language_probability':top_probability,'Turkish_probability':turkish,
                         'raw_language_probabilities':probabilities,'energy_speech_candidate_fraction':speech_fraction,
                         'low_energy_window_sensitivity':speech_fraction<.20,'Turkish_70pass':top=='tr' and turkish>=.70,
                         'confident_nontr':top!='tr' and top_probability>=.70,'uncertain_Turkish_below70':turkish<.70 and not(top!='tr' and top_probability>=.70)})
                del pcm
                nontr=any(w['confident_nontr'] for w in observations)
                uncertain=any(w['uncertain_Turkish_below70'] for w in observations)
                retained=all(w['Turkish_70pass'] for w in observations)
                category='retains_Turkish_screen_eligibility' if retained else 'quarantine_confident_nontr' if nontr else 'quarantine_uncertain_under70'
                result={'segment_id':row['segment_id'],'recording_id':row['recording_id'],'shard':row['shard'],
                        'start_s':row['start_s'],'end_s':row['end_s'],'duration_s':row['duration_s'],
                        'transcript':row['transcript'],'QC_group':row['QC_group'],'random64_member':row['segment_id'] in random_ids,
                        'preselected_full_large_v3_random24':row.get('preselected_full_large_v3_random24',False),
                        'language_observations':observations,'screening_category':category,'retains_language_screen_eligibility':retained,
                        'any_confident_nontr':nontr,'any_uncertain_under70':uncertain,'decode':decode,
                        'tiny_transcript_generated':False,'model_SHA':TINY_SHA,'CPU_only':True,
                        'completed_at_utc':base.now(),'elapsed_s':time.monotonic()-start,'own_RSS_bytes':psutil.Process().memory_info().rss,
                        'accepted_TRAIN':False,'source_text_corrected':False}
            except Exception as exc:
                result={'segment_id':row['segment_id'],'recording_id':row['recording_id'],'shard':row['shard'],
                        'QC_group':row['QC_group'],'random64_member':row['segment_id'] in random_ids,
                        'screening_category':'quarantine_unavailable_no_language_decision','error':repr(exc),
                        'completed_at_utc':base.now(),'accepted_TRAIN':False}
            base.append_json(ledger,result);prior.append(result);done_ids.add(row['segment_id'])
            primary=[r for r in prior if r['random64_member']]
            report={'kind':'random64_Whisper_tiny_language_only_progress_v7','updated_at_utc':base.now(),
                    'random64_declared':64,'random64_completed':len(primary),'total_with_worst_supplement_completed':len(prior),
                    'random64_screen_counts':{k:sum(r['screening_category']==k for r in primary) for k in
                       ('retains_Turkish_screen_eligibility','quarantine_confident_nontr','quarantine_uncertain_under70','quarantine_unavailable_no_language_decision')},
                    'supplemental_results_excluded_from_random_denominator':True,'ledger':str(ledger),
                    'random24_manifest':sampling['random24_manifest'],'tiny_transcription':False,'accepted_TRAIN_hours':0,
                    'last_result':result,'full_large_v3_owner':'ROOT; no CPU large model load/download occurred'}
            base.write_json(AREA/'tiny_LID_report_v7.json',report)
            guard.status(phase='actual_tiny_language_only_result_durable',random64_completed=len(primary),total_completed=len(prior),
                         segment_id=row['segment_id'],category=result['screening_category'],own_RSS_bytes=psutil.Process().memory_info().rss)
        base.write_json(AREA/'tiny_LID_audio_phase_closed_v7.json',{'at_utc':base.now(),'pid':os.getpid(),'actual_success':True,
                         'random64_completed':len([r for r in prior if r['random64_member']]),'total_completed':len(prior),
                         'one_CPU_audio_phase_finished':True,'large_CPP_partial_unchanged_bytes':226572871,
                         'ROOT_remux_random24_may_begin_after_actual_exit0':True,'accepted_TRAIN_hours':0})
    finally:
        runtime.free(context);log_file.close()
    guard.status(phase='completed_tiny_LID_random64_and_worst_supplement',actual_CLOSED0=True)


if __name__=='__main__':main()
