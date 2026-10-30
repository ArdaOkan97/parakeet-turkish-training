"""Bound exact source/text/audio exclusions, and useful near-text triage, on CPU."""
import collections
import hashlib
import json
import pathlib
import re
import sqlite3
import sys
import wave

from cpu_preparation_guard_v2 import AREA,PROJECT,ROOT,Guard,append_json,now,sha256,write_json
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from metrics import normalize_text

def htext(text,lang='tr'):return hashlib.sha256(normalize_text(text,lang).encode('utf-8')).hexdigest()
def grams(text):
    words=normalize_text(text,'tr').split()
    return {' '.join(words[i:i+5]) for i in range(max(0,len(words)-4))}

def main():
    import numpy as np
    guard=Guard('source_heldout_exclusions_v2')
    dbpath=ROOT/'indexes/alania_source_exclusions_v2.sqlite'
    db=sqlite3.connect(dbpath);db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE IF NOT EXISTS reference_text(text_sha256 TEXT,source_id TEXT,role TEXT,reference TEXT,PRIMARY KEY(text_sha256,source_id,role))')
    db.execute('CREATE TABLE IF NOT EXISTS reference_source(recording_id TEXT,source_id TEXT,role TEXT,PRIMARY KEY(recording_id,source_id,role))')
    db.execute('CREATE TABLE IF NOT EXISTS reference_audio(source_id TEXT PRIMARY KEY,language TEXT,wav_sha256 TEXT,PCM_i16_sha256 TEXT,sample_rate INTEGER,samples INTEGER,audio_filepath TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS completed_candidate(segment_id TEXT PRIMARY KEY,exact_heldout_text INTEGER,exact_TRAIN_text INTEGER,exact_source INTEGER,near_heldout_text INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS input_binding(path TEXT PRIMARY KEY,sha256 TEXT,role TEXT,rows INTEGER)')
    heldout_paths=sorted(p for p in (PROJECT/'work/data').glob('*.jsonl') if p.name.startswith(('validation','test')))
    # These are groundtruth/source manifests only. Predictions are never opened.
    train_names=['train.jsonl','tr_public_v3_train.jsonl','tr_expanded_train.jsonl','replay_train.jsonl']
    train_paths=[PROJECT/'work/data'/name for name in train_names if (PROJECT/'work/data'/name).exists()]
    unique_heldout={};heldout_references={};ngram_map=collections.defaultdict(set);bindings=[]
    for path,role in [(p,'heldout') for p in heldout_paths]+[(p,'existing_TRAIN_manifest_partial') for p in train_paths]:
        guard.wait();digest=sha256(path)
        old=db.execute('SELECT sha256 FROM input_binding WHERE path=?',(str(path),)).fetchone()
        if old and old[0]!=digest:raise ValueError('Bound source manifest changed; use a fresh version')
        count=0
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():continue
                r=json.loads(line);count+=1
                if any(k in r for k in ['hypothesis','prediction','predicted_text']):raise ValueError('Prediction file is forbidden in exclusion work')
                sid=r.get('source_id') or r.get('id') or f'{path.name}:{count}'
                text=r.get('reference',r.get('text',r.get('transcript','')))
                lang=r.get('language','tr')
                if role=='heldout':
                    if r.get('dataset_split',r.get('split')) not in {'validation','test'}:raise ValueError('Unexpected heldout split')
                    unique_heldout[sid]=r;heldout_references[sid]=text
                    for gram in grams(text):ngram_map[gram].add(sid)
                if not old:
                    db.execute('INSERT OR IGNORE INTO reference_text VALUES(?,?,?,?)',(htext(text),sid,role,text))
                    rid=r.get('recording_id') or r.get('yodas_id') or r.get('youtube_id')
                    if rid:db.execute('INSERT OR IGNORE INTO reference_source VALUES(?,?,?)',(str(rid),sid,role))
                if count%2000==0:guard.wait();db.commit()
        if not old:db.execute('INSERT INTO input_binding VALUES(?,?,?,?)',(str(path),digest,role,count));db.commit()
        bindings.append({'path':str(path),'sha256':digest,'role':role,'rows':count})
        guard.status(phase='source_manifest_bound',manifest=path.name,rows=count,role=role)
    for n,(sid,row) in enumerate(unique_heldout.items(),1):
        if db.execute('SELECT 1 FROM reference_audio WHERE source_id=?',(sid,)).fetchone():continue
        guard.wait();path=pathlib.Path(row['audio_filepath']).resolve()
        if not path.is_relative_to(PROJECT):raise ValueError('Heldout audio path outside authorized existing project')
        observed=sha256(path)
        if row.get('wav_sha256') and observed!=row['wav_sha256']:raise ValueError('Heldout source WAV changed')
        with wave.open(str(path),'rb') as wav:
            if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getframerate()!=16000:raise ValueError('Unexpected heldout PCM binding')
            pcm=wav.readframes(wav.getnframes())
        db.execute('INSERT INTO reference_audio VALUES(?,?,?,?,?,?,?)',(sid,row['language'],observed,hashlib.sha256(pcm).hexdigest(),16000,len(pcm)//2,str(path)))
        if n%50==0:db.commit();guard.status(phase='heldout_source_audio_hash_verified',unique_heldout_sources=len(unique_heldout),completed=n)
    db.commit()
    candidates=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True)
    candidates.row_factory=sqlite3.Row;candidates.execute('PRAGMA cache_size=-4096');candidates.execute('PRAGMA temp_store=MEMORY')
    heldtext={x[0] for x in db.execute("SELECT text_sha256 FROM reference_text WHERE role='heldout'")}
    traintext={x[0] for x in db.execute("SELECT text_sha256 FROM reference_text WHERE role='existing_TRAIN_manifest_partial'")}
    sourceids={x[0] for x in db.execute('SELECT recording_id FROM reference_source')}
    done={x[0] for x in db.execute('SELECT segment_id FROM completed_candidate')}
    guard.status(phase='scanning_all_eligible_candidate_source_and_text_exclusions',heldout_sources=len(unique_heldout),existing_TRAIN_unique_normalized_texts=len(traintext),resume_completed=len(done))
    for n,raw in enumerate(candidates.execute('SELECT segment_id,recording_id,shard,transcript,duration_s FROM candidates WHERE eligible=1'),1):
        row=dict(raw)
        if row['segment_id'] in done:continue
        digest=htext(row['transcript']);held=digest in heldtext;train=digest in traintext;source=row['recording_id'] in sourceids
        overlap=collections.Counter(sid for gram in grams(row['transcript']) for sid in ngram_map.get(gram,()))
        # At least3 distinct matching5grams AND >=half of shorter text's5grams.
        matches=[{'source_id':sid,'shared_distinct_word5grams':shared} for sid,shared in overlap.items()
                 if shared>=3 and shared/max(1,min(len(grams(row['transcript'])),len(grams(heldout_references[sid]))))>=.5]
        if held or train or source or matches:
            append_json(AREA/'candidate_source_text_collision_ledger_v2.jsonl',{**row,'exact_heldout_text':held,
                'exact_existing_TRAIN_text_partial':train,'exact_source_identity':source,'near_heldout_text_triage':matches,
                'interpretation':'Text repetition is an exclusion/triage signal, not proof of identical audio','accepted_TRAIN':False})
        db.execute('INSERT INTO completed_candidate VALUES(?,?,?,?,?)',(row['segment_id'],int(held),int(train),int(source),int(bool(matches))))
        if n%2000==0:
            db.commit();guard.wait();guard.status(phase='candidate_exclusion_scan',candidate_rows_seen=n)
    db.commit();candidates.close()
    totals=db.execute('SELECT COUNT(*),SUM(exact_heldout_text),SUM(exact_TRAIN_text),SUM(exact_source),SUM(near_heldout_text) FROM completed_candidate').fetchone()
    result={'kind':'bound_source_audio_hash_and_expanded_text_exclusion_v2','completed_at_utc':now(),
        'metadata_eligible_candidate_rows_checked':totals[0],'exact_heldout_text_collision_rows':totals[1],
        'exact_existing_TRAIN_text_collision_rows_partial_manifest_scope':totals[2],'exact_source_identity_collision_rows':totals[3],
        'near_heldout_text_triage_rows':totals[4],'unique_heldout_sources':len(unique_heldout),
        'heldout_languages':dict(collections.Counter(r['language'] for r in unique_heldout.values())),
        'actual_heldout_audio_files_sha256_and_PCM_i16_hash_verified':db.execute('SELECT COUNT(*) FROM reference_audio').fetchone()[0],
        'input_bindings':bindings,'index':str(dbpath),'accepted_TRAIN_hours':0,'TEST_predictions_read':False,'GPU_used':False,
        'limitations':['Existing TRAIN manifest bindings are partial until linked to canonical675.0964h pool.',
            'Heldout audio hashes are verified and indexed; candidate-vs-heldout audio comparison follows independently.',
            'Different text does not establish different audio. No complete source/audio exclusion or acceptance claimed.',
            'Near-text match is conservative triage; not a confirmed audio duplicate.']}
    write_json(AREA/'source_heldout_exclusion_report_v2.json',result);db.close()
    guard.status(phase='completed_declared_partial_exclusions',actual_CLOSED0=True,report=str(AREA/'source_heldout_exclusion_report_v2.json'),summary=result)

if __name__=='__main__':main()
