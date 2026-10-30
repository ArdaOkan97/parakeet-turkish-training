"""One paced CPU worker: suspected overlap, PT heldout audio, new8-shard QC."""
import collections
import hashlib
import io
import json
import pathlib
import sqlite3
import sys
import time
import wave

from paced_CPU_audio_v3 import AREA,PacedGuard,decode_paced
from cpu_preparation_guard_v2 import PROJECT,ROOT,append_json,now,sha256,write_json
from check_alania_pilot_audio_exclusion_cpu_v2 import features,retrieve
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from metrics import edit_distance,normalize_text

def wav_read(row,guard):
    import numpy as np
    data=guard.read_file(row['audio_filepath'])
    if row.get('wav_sha256') and hashlib.sha256(data).hexdigest()!=row['wav_sha256']:raise ValueError('Bound source WAV changed')
    with wave.open(io.BytesIO(data),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()!=16000:raise ValueError('Unexpected source WAV format')
        raw=w.readframes(w.getnframes())
    digest=hashlib.sha256(raw).hexdigest()
    if row.get('canonical_pcm_sha256') and digest!=row['canonical_pcm_sha256']:raise ValueError('Bound source PCM differs')
    return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768,digest

def waveform_overlap(a,b):
    import numpy as np
    n=1<<(len(a)+len(b)-1).bit_length();aa=a.astype(np.float64);bb=b.astype(np.float64)
    corr=np.fft.irfft(np.fft.rfft(aa,n)*np.conj(np.fft.rfft(bb,n)),n)
    lags=np.arange(-32000,32001);ia=np.maximum(lags,0);ib=np.maximum(-lags,0);count=np.minimum(len(a)-ia,len(b)-ib)
    sa=np.concatenate(([0.],np.cumsum(aa)));sb=np.concatenate(([0.],np.cumsum(bb)))
    qa=np.concatenate(([0.],np.cumsum(aa*aa)));qb=np.concatenate(([0.],np.cumsum(bb*bb)))
    suma=sa[ia+count]-sa[ia];sumb=sb[ib+count]-sb[ib]
    covariance=corr[lags%n]-suma*sumb/count
    denominator=np.sqrt(np.maximum(0,(qa[ia+count]-qa[ia]-suma*suma/count)*(qb[ib+count]-qb[ib]-sumb*sumb/count)))
    scores=np.divide(covariance,denominator,out=np.zeros_like(covariance),where=denominator>1e-12)
    index=int(np.argmax(np.abs(scores)));score=float(scores[index]);overlap=float(count[index]/min(len(a),len(b)))
    return {'maximum_absolute_normalized_waveform_correlation':abs(score),'signed_correlation':score,
        'best_lag_samples':int(lags[index]),'best_lag_seconds':float(lags[index]/16000),'overlap_seconds':float(count[index]/16000),
        'fraction_of_shorter_clip_compared':overlap,'lag_search_seconds':[-2,2],
        'strong_audio_overlap_triage':abs(score)>=.98 and overlap>=.8,
        'interpretation':'High aligned waveform correlation supports shared audio; low correlation cannot exclude transformed or differently segmented source overlap'}

def check_overlaps(guard):
    records=json.loads((AREA/'WorldSpeech_overlap_bindings_v3.json').read_text(encoding='utf-8'))['recordings']
    ledger=AREA/'WorldSpeech_audio_overlap_ledger_v3.jsonl'
    old=[json.loads(x) for x in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else [];done={x['segment_id'] for x in old}
    for row in records:
        if row['segment_id'] in done:continue
        result={k:row[k] for k in ['segment_id','recording_id','shard','transcript','duration_s']}
        if not row['candidate_archive_local']:
            result.update(outcome='candidate_archive_not_staged_no_download',audio_comparison_completed=False)
        else:
            guard.wait();guard.status(phase='paced_priority_WorldSpeech_overlap',active_segment=row['segment_id'])
            try:
                # Metadata collision receipt omitted start_s; recover only this shard/id.
                db=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True)
                start,end=db.execute('SELECT start_s,end_s FROM candidates WHERE segment_id=?',(row['segment_id'],)).fetchone();db.close()
                pcm,info=decode_paced({**row,'start_s':start,'end_s':end},guard);accepted=row['accepted_rows'][0]
                other,digest=wav_read(accepted,guard);guard.wait();comparison=waveform_overlap(pcm,other);del pcm,other
                result.update(audio_comparison_completed=True,decode=info,accepted_source_id=accepted['source_id'],
                    accepted_source_url=accepted.get('source_url'),accepted_PCM_i16_sha256=digest,comparison=comparison,
                    outcome='quarantine_recording_strong_audio_overlap' if comparison['strong_audio_overlap_triage'] else 'quarantine_recording_exact_text_audio_inconclusive')
            except Exception as e:result.update(audio_comparison_completed=False,outcome='quarantine_recording_audio_check_failed',error=repr(e))
        result.update(completed_at_utc=now(),accepted_TRAIN=False);append_json(ledger,result);old.append(result)
        write_json(AREA/'WorldSpeech_audio_overlap_report_v3.json',{'kind':'three_priority_recordings_source_audio_overlap_triage_v3','updated_at_utc':now(),
            'observations':len(old),'actual_audio_comparisons':sum(x.get('audio_comparison_completed',False) for x in old),
            'strong_waveform_overlap_recordings':sum(x.get('comparison',{}).get('strong_audio_overlap_triage',False) for x in old),
            'not_staged_no_download':sum(x['outcome']=='candidate_archive_not_staged_no_download' for x in old),
            'accepted_TRAIN_hours':0,'rows':old})
        guard.status(phase='priority_overlap_observation_saved',overlap_observations=len(old),last_overlap_outcome=result['outcome'])

def heldout_index(guard):
    import numpy as np
    old=sqlite3.connect('file:'+(ROOT/'indexes/heldout_spectral_fingerprints_v2.sqlite').as_posix()+'?mode=ro',uri=True);old.execute('PRAGMA cache_size=-1024')
    rawdb=sqlite3.connect('file:'+(ROOT/'indexes/alania_source_exclusions_v2.sqlite').as_posix()+'?mode=ro',uri=True)
    refs=rawdb.execute('SELECT source_id,PCM_i16_sha256 FROM reference_audio ORDER BY source_id').fetchall();rawdb.close()
    vectors=[];refids=[];positions=[];sourceids=[];lengths=[];hashes=collections.defaultdict(list)
    def add(sid,digest,vec,pos):
        i=len(sourceids);sourceids.append(sid);hashes[digest].append(sid);lengths.append(len(vec));vectors.append(vec);refids.extend([i]*len(vec));positions.extend(pos.tolist())
    for sid,digest in refs:
        guard.wait();blob,posblob,n=old.execute('SELECT vectors,positions,frames FROM fingerprints WHERE source_id=?',(sid,)).fetchone()
        guard.pace(len(blob)+len(posblob));add(sid,digest,np.frombuffer(blob,dtype='<f4').reshape(n,32),np.frombuffer(posblob,dtype='<i4'))
    old.close();pt=[json.loads(x) for x in (AREA/'Portuguese_heldout_source_bindings_v3.jsonl').read_text(encoding='utf-8').splitlines()]
    db=sqlite3.connect(ROOT/'indexes/Portuguese_heldout_spectral_v3.sqlite');db.execute('PRAGMA cache_size=-1024')
    db.execute('CREATE TABLE IF NOT EXISTS verified(source_id TEXT PRIMARY KEY,PCM_i16_sha256 TEXT,vectors BLOB,positions BLOB,frames INTEGER)')
    for n,row in enumerate(pt,1):
        guard.wait();existing=db.execute('SELECT PCM_i16_sha256,vectors,positions,frames FROM verified WHERE source_id=?',(row['source_id'],)).fetchone()
        if existing:
            digest,blob,posblob,size=existing;vec=np.frombuffer(blob,dtype='<f4').reshape(size,32);pos=np.frombuffer(posblob,dtype='<i4');guard.pace(len(blob)+len(posblob))
        else:
            pcm,digest=wav_read(row,guard);vec,pos=features(pcm);del pcm
            db.execute('INSERT INTO verified VALUES(?,?,?,?,?)',(row['source_id'],digest,vec.astype('<f4').tobytes(),pos.astype('<i4').tobytes(),len(vec)));db.commit()
            append_json(AREA/'Portuguese_heldout_audio_verification_v3.jsonl',{'source_id':row['source_id'],'split':row['dataset_split'],
                'WAV_sha256_verified':row['wav_sha256'],'PCM_i16_sha256_verified':digest,'completed_at_utc':now(),'prediction_read':False})
        add(row['source_id'],digest,vec,pos)
        if n%10==0:guard.status(phase='paced_Portuguese_heldout_audio_verification',PT_sources_verified=n,combined_heldout_sources=900+n)
    db.close();write_json(AREA/'seven_language_audio_reference_binding_v3.json',{'kind':'actual_seven_language_heldout_audio_hash_and_spectral_binding_v3',
        'completed_at_utc':now(),'unique_heldout_sources':len(sourceids),'prior900_actual_audio_evidence_reused':True,'new_PT_actual_WAV_PCM_rechecks':150,
        'languages':['en','es','it','fr','de','tr','pt'],'TEST_predictions_read':False,'accepted_TRAIN_hours':0,
        'complete_candidate_audio_exclusion_claimed':False})
    return np.concatenate(vectors),np.asarray(refids,np.int32),np.asarray(positions,np.int32),sourceids,lengths,hashes

def load_model(guard):
    from faster_whisper import WhisperModel
    receipt=json.loads((ROOT/'analysis/CPU_validation_v2/independent_recognizer_receipt_v2.json').read_text(encoding='utf-8'))
    modeldir=ROOT/'cache/models/Systran_faster-whisper-small'/receipt['revision'];files={}
    for name in ['config.json','vocabulary.txt','tokenizer.json','model.bin']:
        record=next(x for x in receipt['files'] if x['name']==name);guard.status(phase='paced_pinned_CPU_model_load',active_model_file=name)
        data=guard.read_file(modeldir/name,record['bytes'])
        if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Pinned model file hash differs')
        files[name]=data
    guard.wait();model=WhisperModel('pinned_CPU_small_v3',device='cpu',compute_type='int8',cpu_threads=1,num_workers=1,files=files,local_files_only=True)
    del files;guard.status(phase='pinned_CPU_model_loaded',model_revision=receipt['revision']);return model

def main():
    import numpy as np
    import psutil
    guard=PacedGuard();guard.wait();check_overlaps(guard);allvectors,refids,refpos,sourceids,lengths,hashes=heldout_index(guard);model=load_model(guard)
    chosen=[json.loads(x) for x in (AREA/'new8_shards_QC_manifest_v3.jsonl').read_text(encoding='utf-8').splitlines()]
    ledger=AREA/'new8_shards_independent_QC_ledger_v3.jsonl';rows=[json.loads(x) for x in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else [];done={r['segment_id'] for r in rows}
    for row in chosen:
        if row['segment_id'] in done:continue
        guard.wait();before=time.monotonic();out={k:row[k] for k in ['segment_id','recording_id','shard','start_s','end_s','duration_s','transcript','QC_stratum']}
        try:
            guard.status(phase='paced_new_shard_QC_decode',active_segment=row['segment_id'],completed_samples=len(rows),declared_samples=len(chosen))
            pcm,info=decode_paced(row,guard);out.update(info);guard.wait()
            canonical=np.clip(np.rint(pcm*32768),-32768,32767).astype('<i2');digest=hashlib.sha256(canonical.tobytes()).hexdigest();del canonical
            vec,pos=features(pcm);matches=retrieve(vec,pos,allvectors,refids,refpos,sourceids,lengths);guard.wait()
            segments,meta=model.transcribe(pcm,language=None,beam_size=1,best_of=1,temperature=0,condition_on_previous_text=False,word_timestamps=True,vad_filter=False)
            segments=list(segments);hyp=' '.join(x.text.strip() for x in segments).strip();del pcm
            words=[{'start':w.start,'end':w.end,'word':w.word,'probability':w.probability} for s in segments for w in (s.words or [])]
            a=normalize_text(row['transcript'],'tr');b=normalize_text(hyp,'tr');wer=edit_distance(a.split(),b.split())/max(1,len(a.split()));flags=[]
            if meta.language!='tr':flags.append('independent_detected_non_Turkish')
            if meta.language_probability<.8:flags.append('independent_LID_uncertain')
            if wer>.30:flags.append('source_independent_ASR_disagreement')
            if len(a.split())/row['duration_s']>5.5:flags.append('source_text_rate_high')
            if not words:flags.append('no_independent_words')
            elif words[0]['start']>1.5 or row['duration_s']-words[-1]['end']>2.5:flags.append('speech_span_boundary_gap')
            if any(w['start']<-.1 or w['end']>row['duration_s']+.25 or w['end']<w['start'] for w in words):flags.append('independent_word_time_out_of_bounds')
            if hashes.get(digest):flags.append('exact_seven_language_heldout_PCM_match')
            if matches:flags.append('seven_language_heldout_spectral_overlap_triage')
            out.update(independent_recognizer_completed=True,independent_hypothesis=hyp,independent_language=meta.language,
                independent_language_probability=meta.language_probability,source_vs_independent_WER=wer,independent_words=words,
                PCM_i16_sha256=digest,exact_heldout_PCM_matches=hashes.get(digest,[]),heldout_spectral_matches=matches,triage_flags=flags,
                outcome='quarantine_for_review' if flags else 'no_automatic_flags_still_unadmitted')
        except Exception as e:out.update(independent_recognizer_completed=False,error=repr(e),triage_flags=[],outcome='excluded_untested_or_failed')
        guard.wait();out.update(completed_at_utc=now(),elapsed_wall_s=time.monotonic()-before,CPU_RSS_bytes=psutil.Process().memory_info().rss,accepted_TRAIN=False)
        append_json(ledger,out);rows.append(out);done.add(row['segment_id']);good=[x for x in rows if x.get('independent_recognizer_completed')]
        report={'kind':'new8_shards_paced_independent_CPU_QC_v3','updated_at_utc':now(),'declared_samples':192,'durable_observations':len(rows),
            'independent_recognizer_completed':len(good),'triage_clips':sum(bool(x['triage_flags']) for x in good),'excluded_untested_or_failed':len(rows)-len(good),
            'independent_languages':dict(collections.Counter(x['independent_language'] for x in good)),
            'triage_reason_counts':dict(collections.Counter(f for x in good for f in x['triage_flags'])),
            'heldout_audio_sources':len(sourceids),'accepted_TRAIN_hours':0,'GPU_used':False,'TEST_predictions_read':False,
            'aggregate_audio_model_read_bytes':guard.file_bytes_read,'limitations':['Recognizer disagreement and spectral retrieval are triage only.',
                'No complete global source/audio/speaker dedup or commercial-rights clearance claimed.']}
        write_json(AREA/'new8_shards_independent_QC_report_v3.json',report);guard.status(phase='paced_new_shard_QC_observation_saved',completed_samples=len(rows),recognizer_completed=len(good),triage_clips=report['triage_clips'])
    guard.status(phase='completed_paced_declared_phase',actual_CLOSED0=True)

if __name__=='__main__':main()
