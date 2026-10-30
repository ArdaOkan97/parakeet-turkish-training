"""Light metadata only: PT heldout bindings and the next bounded staged pilot."""
import collections
import hashlib
import json
import pathlib
import sqlite3
import struct
import sys

import cpu_preparation_guard_v2 as base
from cpu_preparation_guard_v2 import PROJECT,ROOT,now,sha256,write_json

AREA=ROOT/'analysis/CPU_validation_v3'
base.AREA=AREA

def accepted_row(sid):
    d=PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1'
    db=sqlite3.connect('file:'+(d/'catalog.sqlite').as_posix()+'?mode=ro&immutable=1',uri=True)
    ordinal=db.execute('SELECT ordinal FROM catalog WHERE source_id=?',(sid,)).fetchone()[0];db.close()
    with (d/'offsets.u64').open('rb') as f:
        f.seek(ordinal*8);start,end=struct.unpack('<QQ',f.read(16))
    with (d/'rows.jsonl').open('rb') as f:f.seek(start);return json.loads(f.read(end-start))

def main():
    guard=base.Guard('metadata_plan_v3')
    prep_path=PROJECT/'outputs/parakeet_turkish/Portuguese_FLEURS_retention_preparation_v1.json'
    baseline_path=PROJECT/'outputs/parakeet_turkish/Portuguese_retention_baselines_v1.json'
    prep=json.loads(prep_path.read_text(encoding='utf-8'));baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if sha256(prep_path)!=baseline['pt_preparation']['sha256'] or not prep['passed']:raise ValueError('PT source-preparation binding differs')
    if baseline['fixed_PT_DEV_manifest']!=prep['files']['validation_pt']:raise ValueError('PT baseline DEV binding differs')
    evidence={r['source_id']:r for r in prep['audio_PCM_evidence']};pt=[];bindings=[]
    for name,split in [('validation_pt','validation'),('test_pt','test')]:
        record=prep['files'][name];path=pathlib.Path(record['path']).resolve()
        if not path.is_relative_to(PROJECT) or sha256(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('PT groundtruth binding differs')
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        if len(rows)!=record['utterances'] or hashlib.sha256('\n'.join(r['source_id'] for r in rows).encode()).hexdigest()!=record['source_ids_sha256']:raise ValueError('PT source IDs differ')
        for row in rows:
            if any(k in row for k in ['hypothesis','prediction','predicted_text']):raise ValueError('Predictions forbidden')
            if row['language']!='pt' or row['dataset_split']!=split or row['dataset_revision']!=prep['revision']:raise ValueError('Wrong Portuguese source')
            if row['canonical_pcm_sha256']!=evidence[row['source_id']]['canonical_pcm_sha256']:raise ValueError('PT existing PCM proof differs')
            pt.append({**row,'PCM_evidence_binding':'Existing pinned preparation artifact; actual WAV hash recheck queued in paced worker'})
        bindings.append({**record,'role':'heldout_groundtruth_source_only','split':split})
    if len({r['source_id'] for r in pt})!=150:raise ValueError('Expected PT150 unique heldout sources')
    pt_path=AREA/'Portuguese_heldout_source_bindings_v3.jsonl';pt_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pt),encoding='utf-8')
    write_json(AREA/'Portuguese_heldout_binding_receipt_v3.json',{'kind':'Portuguese_frozen_heldout_binding_v3','created_at_utc':now(),
        'preparation_receipt':{'path':str(prep_path),'sha256':sha256(prep_path)},'baseline_binding_receipt':{'path':str(baseline_path),'sha256':sha256(baseline_path)},
        'groundtruth_inputs':bindings,'unique_sources':150,'validation_sources':50,'test_sources':100,
        'source_revision':prep['revision'],'existing_canonical_PCM_evidence_rows_bound':150,
        'manifest':str(pt_path),'manifest_sha256':sha256(pt_path),'prediction_files_read':False,'GPU_used':False,
        'all_seven_language_source_manifest_binding_now_available':True,'all_seven_audio_perceptual_exclusion_complete':False,'accepted_TRAIN_hours':0})
    old=ROOT/'analysis/CPU_validation_v2';plan=json.loads((old/'additional_stage_plan_v2.json').read_text(encoding='utf-8'))
    chosen=[];seen={json.loads(x)['recording_id'] for x in (old/'independent_QC_manifest_v2.jsonl').read_text(encoding='utf-8').splitlines()}
    db=sqlite3.connect('file:'+(ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=MEMORY')
    for entry in plan['selected_shards']:
        guard.wait();shard=entry['shard'];buckets=collections.defaultdict(list)
        for raw in db.execute('SELECT * FROM candidates WHERE eligible=1 AND shard=? ORDER BY pilot_rank',(shard,)):
            row=dict(raw)
            if row['recording_id'] in seen:continue
            duration='2-6s' if row['duration_s']<6 else ('6-12s' if row['duration_s']<12 else '12-20s')
            key=(duration,'DNSMOS2.8-3.2' if row['dnsmos']<3.2 else 'DNSMOS3.2+',row['noise'],row['reverb'])
            row['QC_stratum']=key;buckets[key].append(row)
        keys=sorted(buckets,key=str);n=0
        while n<24:
            progress=False
            for key in keys:
                while buckets[key] and buckets[key][0]['recording_id'] in seen:buckets[key].pop(0)
                if not buckets[key]:continue
                row=buckets[key].pop(0);seen.add(row['recording_id']);row['QC_phase']='new8_shards_stratified192_paced_v3';chosen.append(row);n+=1;progress=True
                if n==24:break
            if not progress:raise ValueError('Insufficient distinct recordings')
    priority=json.loads((old/'long_WorldSpeech_text_overlap_priority_v2.json').read_text(encoding='utf-8'))['priority'];overlaps=[]
    for row in priority:
        accepted=[accepted_row(sid) for sid in row['matching_accepted_source_ids_first10']]
        overlaps.append({**row,'accepted_rows':accepted,'candidate_archive_local':(ROOT/f'datasets/yodas3/audio/{row["shard"]}.tar').exists(),
            'next_action':'paced bounded clip waveform comparison' if row['compressed_shard_already_local'] else 'source-text bound; no archive download authorized in this phase'})
    manifest=AREA/'new8_shards_QC_manifest_v3.jsonl';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen),encoding='utf-8')
    write_json(AREA/'WorldSpeech_overlap_bindings_v3.json',{'kind':'three_long_suspected_overlap_source_bindings_v3','created_at_utc':now(),'recordings':overlaps,'accepted_TRAIN_hours':0,'source_audio_duplicate_confirmation_pending':True})
    proof=ROOT/'analysis/three_epoch_entry_v1/telemetry_stop_75250_v1/actual_recovery_native_positive_v1.json'
    nextplan={'kind':'single_paced_CPU_QC_new8_shards_v3','created_at_utc':now(),'declared_semantic_clips':192,'different_recordings':len({r['recording_id'] for r in chosen}),
        'manifest':str(manifest),'manifest_sha256':sha256(manifest),'candidate_hours':sum(r['duration_s'] for r in chosen)/3600,'shards':[r['shard'] for r in plan['selected_shards']],
        'heldout_source_scope':{'prior_verified':900,'new_PT_bound':150,'total':1050,'languages':['en','es','it','fr','de','tr','pt']},
        'priority_overlap_recordings':3,'priority_overlap_local_audio_comparisons':sum(x['candidate_archive_local'] for x in overlaps),'bulk_downloads':False,
        'maximum_audio_workers':1,'PyAV_threads':1,'CPU_model_threads':1,'max_compressed_member_bytes':256*1024*1024,'max_clip_seconds':20,
        'aggregate_new_worker_file_read_limit_bytes_per_second':1024*1024,'read_block_max_bytes':65536,'maximum_active_decode_seconds':90,
        'pause_conditions':['ROOT native positive proof missing','queue or current loop not running','published_step <= current resume_step',
            'published_step modulo250 in{0,230,240}','worker self-telemetry or controller status older than15s','system commit>=90%','headroom<2GB'],
        'hard_stop_conditions':['system commit>=95%','headroom<2GB','own process RSS>2GB'],
        'ROOT_native_positive_proof':str(proof),'ROOT_native_positive_proof_sha256':'12bfb6244061353884c27f137903cd59dac0b05d9dc9fa1f66a66a81f0d72a9e',
        'GPU_lease_queue_and_training_sources_read_only':True,'accepted_TRAIN_hours':0,'commercial_underlying_audio_rights_not_yet_established':True}
    write_json(AREA/'paced_CPU_QC_plan_v3.json',nextplan);db.close()
    guard.status(phase='metadata_bindings_and_next_plan_complete',actual_CLOSED0=True,PT_heldout_sources_bound=150,declared_new_QC_clips=192,
        local_priority_overlap_recordings=sum(x['candidate_archive_local'] for x in overlaps),accepted_TRAIN_hours=0)

if __name__=='__main__':main()
