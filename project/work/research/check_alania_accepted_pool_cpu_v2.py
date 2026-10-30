"""Exact text/source/hash comparison to the full immutable675.0964h pool."""
import collections
import hashlib
import json
import sqlite3
import sys

from cpu_preparation_guard_v2 import AREA,PROJECT,ROOT,Guard,append_json,now,sha256,write_json
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from metrics import normalize_text

def text_sha(text):return hashlib.sha256(normalize_text(text,'tr').encode('utf-8')).hexdigest()

def main():
    guard=Guard('accepted_pool_exclusion_v2');carrier=PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1'
    provenance_path=carrier/'index_provenance.json';provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    assert provenance['complete'] and provenance['rows']==352768 and abs(provenance['hours']-675.0964019097223)<1e-8
    catalog=carrier/'catalog.sqlite';bound=provenance['files']['catalog.sqlite']
    if catalog.stat().st_size!=bound['bytes'] or sha256(catalog)!=bound['sha256']:raise ValueError('Immutable accepted source catalog binding differs')
    source=sqlite3.connect('file:'+catalog.as_posix()+'?mode=ro&immutable=1',uri=True);source.execute('PRAGMA cache_size=-4096')
    db=sqlite3.connect(ROOT/'indexes/full_accepted_pool_exclusion_v2.sqlite');db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE IF NOT EXISTS accepted(source_id TEXT PRIMARY KEY,text_sha256 TEXT,audio_sha256 TEXT,dataset TEXT,duration_s REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS candidate(segment_id TEXT PRIMARY KEY,exact_text_match INTEGER,exact_source_match INTEGER)')
    count=db.execute('SELECT COUNT(*) FROM accepted').fetchone()[0]
    if count not in [0,352768]:raise ValueError('Partial catalog import requires resume by source_id')
    if count==0:
        for n,(sid,raw) in enumerate(source.execute('SELECT source_id,fingerprint FROM catalog ORDER BY ordinal'),1):
            row=json.loads(raw)
            is_bound_training_source=(row['dataset_split']=='train' or
                (row['dataset']=='ysdede/commonvoice_17_tr_fixed' and row['dataset_split']=='validated' and row.get('training_role')=='train'))
            if row['language']!='tr' or not is_bound_training_source or row['source_id']!=sid:raise ValueError('Unexpected accepted TRAIN row')
            db.execute('INSERT INTO accepted VALUES(?,?,?,?,?)',(sid,text_sha(row['text']),row.get('audio_sha256'),row['dataset'],row['duration']))
            if n%5000==0:guard.wait();guard.status(phase='binding_current_full_accepted_TRAIN_pool',accepted_rows_bound=n)
        db.commit()
    source.close();db.execute('CREATE INDEX IF NOT EXISTS accepted_text ON accepted(text_sha256)');db.commit()
    texts={x[0] for x in db.execute('SELECT DISTINCT text_sha256 FROM accepted')};sourceids={x[0] for x in db.execute('SELECT source_id FROM accepted')}
    audiohashes={x[0] for x in db.execute('SELECT audio_sha256 FROM accepted WHERE audio_sha256 IS NOT NULL')}
    candidates=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True);candidates.row_factory=sqlite3.Row;candidates.execute('PRAGMA cache_size=-4096')
    done={x[0] for x in db.execute('SELECT segment_id FROM candidate')}
    for n,raw in enumerate(candidates.execute('SELECT segment_id,recording_id,shard,transcript,duration_s FROM candidates WHERE eligible=1'),1):
        row=dict(raw)
        if row['segment_id'] in done:continue
        digest=text_sha(row['transcript']);exact_text=digest in texts;exact_source=row['recording_id'] in sourceids or row['segment_id'] in sourceids
        if exact_text or exact_source:
            matches=[x[0] for x in db.execute('SELECT source_id FROM accepted WHERE text_sha256=? LIMIT 10',(digest,))] if exact_text else []
            append_json(AREA/'full_accepted_pool_collision_ledger_v2.jsonl',{**row,'exact_normalized_text_collision':exact_text,
                'matching_accepted_source_ids_first10':matches,'exact_source_id_collision':exact_source,'audio_duplicate_confirmed':False,'accepted_TRAIN':False})
        db.execute('INSERT INTO candidate VALUES(?,?,?)',(row['segment_id'],int(exact_text),int(exact_source)))
        if n%5000==0:db.commit();guard.wait();guard.status(phase='checking_candidate_against_full_accepted_pool',candidate_rows_seen=n)
    db.commit();candidates.close()
    # Legacy audio_sha256 binds original encoded source bytes (not float PCM).
    # Compare only original compressed member bytes, never incompatible PCM hashes.
    audio_rows=[];ledger=AREA/'independent_QC_ledger_v2.jsonl'
    for line in ledger.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row.get('compressed_member_sha256'):
            audio_rows.append({'segment_id':row['segment_id'],'exact_accepted_original_encoded_source_bytes_match':row['compressed_member_sha256'] in audiohashes})
    totals=db.execute('SELECT COUNT(*),SUM(exact_text_match),SUM(exact_source_match) FROM candidate').fetchone()
    result={'kind':'full_current_accepted_pool_exact_exclusion_v2','completed_at_utc':now(),
        'accepted_pool':{'clips':352768,'hours':provenance['hours'],'index_provenance_path':str(provenance_path),'index_provenance_sha256':sha256(provenance_path),
            'catalog_path':str(catalog),'catalog_sha256':bound['sha256'],'source_group_counts':provenance['source_group_counts'],'fingerprint':provenance['fingerprint']},
        'candidate_rows_checked':totals[0],'exact_normalized_text_collision_rows':totals[1],'exact_direct_source_id_collision_rows':totals[2],
        'unique_accepted_texts':len(texts),'accepted_original_encoded_source_byte_hashes':len(audiohashes),'QC_original_compressed_member_hashes_checked':len(audio_rows),
        'exact_original_encoded_source_byte_match_clips':sum(x['exact_accepted_original_encoded_source_bytes_match'] for x in audio_rows),
        'audio_observations':audio_rows,'accepted_TRAIN_hours':0,'TEST_predictions_read':False,'GPU_used':False,
        'limitations':['Exact normalized text repetition is conservative triage, not evidence of identical recordings.',
            'Direct source IDs are compared without claiming a universal identity mapping across corpora.',
            'Legacy audio_sha256 binds original encoded source bytes; float PCM hashes are deliberately not compared with that field.',
            'Different source-byte hashes cannot rule out segmentation/reencoding/shifted/otherwise transformed audio; full accepted-pool waveform/perceptual dedup remains pending.',
            'Source-byte hash snapshots include only independently decoded QC rows available at report time; not the entire candidate pool.']}
    write_json(AREA/'full_accepted_pool_exclusion_report_v2.json',result);db.close();guard.status(phase='completed_full_accepted_pool_exact_scope',actual_CLOSED0=True,report=str(AREA/'full_accepted_pool_exclusion_report_v2.json'),candidate_rows_checked=totals[0],exact_text_collision_rows=totals[1])

if __name__=='__main__':main()
