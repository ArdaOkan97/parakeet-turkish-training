"""Stdlib-only native loader manifests from already audited accepted TRAIN WAVs."""
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
DATA = ROOT/'work/data'
DEST = ROOT/'work/nemo_direct_trial_v1/manifests'
LANGUAGES = ['en','es','it','fr','de','tr']
SEED = 20261010


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def native(row, role):
    return {k: row[k] for k in ('audio_filepath','text','duration','source_id','language',
        'dataset','dataset_revision','dataset_split','sentence_id','audio_sha256','wav_sha256')
        if k in row} | {'training_role':role}


def write_rows(path, values):
    assert not path.exists(), 'Preserve existing prepared manifest'
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for row in values:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')


def main():
    proof_path = OUTPUT/'full_nemo_training_manifests_v1.json'
    assert not proof_path.exists()
    DEST.mkdir(parents=True,exist_ok=True)
    heldout_sources, heldout_audio, heldout_sentences = set(),set(),set()
    dev_files = []
    for language in LANGUAGES:
        original = list(rows(DATA/f'validation_{language}.jsonl'))
        assert len(original)==50
        target = DEST/f'dev_{language}.jsonl'
        write_rows(target,(native(r,'validation') for r in original))
        dev_files.append(target)
        for split in ('validation','test'):
            for row in rows(DATA/f'{split}_{language}.jsonl'):
                heldout_sources.add(row['source_id']);heldout_audio.add(row['audio_sha256'])
                if row['dataset']=='google/fleurs':heldout_sentences.add(row['sentence_id'])
    def vetted(row):
        assert row['source_id'] not in heldout_sources and row['audio_sha256'] not in heldout_audio
        assert row['dataset_split'] in ('train','validated') and .5<=row['duration']<=30
        if row['dataset']=='google/fleurs':assert row['sentence_id'] not in heldout_sentences
        return native(row,'train')
    tr_path=DATA/'tr_public_v3_train.jsonl'
    tr = {r['source_id']:vetted(r) for r in rows(tr_path)}
    assert len(tr)==74883
    # The source is the actual accepted pool; read its small replay tail only.
    pool_path=OUTPUT/'runs/adaptive_20261011_native_memory_fix_01_native_resume/training_pool.json'
    with pool_path.open('rb') as f:
        f.seek(max(0,pool_path.stat().st_size-8*1024*1024));tail=f.read()
    marker=b'\n  "replay": '
    begin=tail.index(marker)+len(marker)
    accepted_replay,_=json.JSONDecoder().raw_decode(tail[begin:].decode('utf-8'))
    replay={r['source_id']:vetted(r) for r in accepted_replay}
    assert len(replay)==1595
    del accepted_replay,tail
    plan_path=OUTPUT/'native_training_pilot_plan_v1.json'
    old_plan=json.loads(plan_path.read_text())
    prefix=[]
    for update in old_plan['updates']:
        groups=update['batches']
        assert len(groups)==2 and all(len(g)==8 for g in groups)
        assert all(r['language']=='tr' for r in groups[0]) and all(r['language']!='tr' for r in groups[1])
        prefix.append([[tr[r['source_id']] for r in groups[0]],
                       [replay[r['source_id']] for r in groups[1]]])
    assert len(prefix)==20
    used_prefix={r['source_id'] for g in prefix for r in g[0]}
    assert len(used_prefix)==160
    remaining=sorted((r for sid,r in tr.items() if sid not in used_prefix),
        key=lambda r:hashlib.sha256((str(SEED)+'|native_full_TR|'+r['source_id']).encode()).digest())
    # Deterministic length buckets keep all rows and never duplicate pool IDs.
    order=[]
    for start in range(0,len(remaining),128):
        order.extend(sorted(remaining[start:start+128],key=lambda r:(r['duration'],r['source_id'])))
    del remaining
    by_lang={lang:sorted((r for r in replay.values() if r['language']==lang),
        key=lambda r:hashlib.sha256((str(SEED)+'|replay|'+r['source_id']).encode()).digest())
        for lang in LANGUAGES[:-1]}
    rng=random.Random(SEED+1)
    positions={lang:0 for lang in by_lang}
    counts={lang:sum(r['language']==lang for groups in prefix for r in groups[1]) for lang in by_lang}
    # Continue each replay stream after its prefix exposure, with explicit repeats.
    positions.update(counts)
    def replay_group():
        result=[]
        for _ in range(8):
            lang=rng.choice(LANGUAGES[:-1]);items=by_lang[lang]
            result.append(items[positions[lang]%len(items)])
            positions[lang]+=1;counts[lang]+=1
        return result
    sequence=prefix[:]
    for start in range(0,len(order),8):
        group=order[start:start+8]
        if len(group)<8:
            padding=8-len(group)
            group=group+[dict(r,plan_padding_repeat=True) for r in order[:padding]]
        sequence.append([group,replay_group()])
    assert len(sequence)==math.ceil(74883/8)
    pilot=[r for groups in sequence[:250] for group in groups for r in group]
    assert len(pilot)==4000 and sum(r['language']=='tr' for r in pilot)==2000
    assert len({r['source_id'] for r in pilot if r['language']=='tr'})==2000
    train_path=DEST/'train_balanced_250.jsonl'
    write_rows(train_path,pilot)
    role_counts={lang:sum(r['language']==lang for r in pilot) for lang in LANGUAGES}
    proof={'schema':1,'kind':'official_NeMo_pilot_accepted_TRAIN_DEV_manifest_preparation_v1',
        'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'manifest_sha256':{str(p):sha(p) for p in [train_path,*dev_files]},
        'train_records':4000,'optimizer_updates':250,'physical_batch':8,'accumulation':2,
        'ordered_microbatch_pattern':['8Turkish','8replay'],'shuffle':False,
        'language_clip_counts':role_counts,'TR_unique_clips':2000,
        'accepted_TR_pool_clips':74883,'accepted_replay_pool_clips':1595,
        'accepted_source_manifest_path':str(tr_path),'accepted_source_manifest_sha256':sha(tr_path),
        'accepted_replay_source_pool_path':str(pool_path),
        'accepted_replay_source_pool_binding':'actual native_resume run_config input/pool hashes; only replay JSON tail read',
        'original_first20_TRAIN_plan_path':str(plan_path),'original_first20_TRAIN_plan_sha256':sha(plan_path),
        'first20_ordered_source_IDs_match_existing_training_benchmark':True,
        'sampled_source_sequence_sha256':hashlib.sha256(json.dumps([r['source_id'] for r in pilot],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        'pilot_audio_seconds_by_language':{lang:math.fsum(r['duration'] for r in pilot if r['language']==lang) for lang in LANGUAGES},
        'heldout_source_audio_global_FLEURS_sentence_collisions':0,'heldout_predictions_read':False,
        'fresh_PCM_rehash_performed':False,'audio_proof_reused':'publicTurkishv3 acceptance/full-audit receipt and actual accepted replay pool',
        'native_SentencePiece_label_acceptance':'pending actual native tokenizer/data-loader; no HF token IDs reused',
        'objective_limitations':['Official native-base decoder priming and released TDT recipe differ from the HF warmstart loop.',
            '50% supervised replay only; this manifest does not itself provide token/duration/feature KD.',
            'A whole-training timing comparison must use these identical ordered recordings; objective and model initialization differences stay explicit.',
            'Replay languages are uniform random draws after the fixed20-update prefix; finite counts need not be exactly equal.'],
        'full_continuation_proof_path':str(OUTPUT/'full_nemo_training_continuation_v1.json')}
    proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'pilot_ready':True,'proof':str(proof_path),'pilot':str(train_path),'language_counts':role_counts}),flush=True)
    # Publish continuation separately, never rewriting the seven-file pilot proof.
    full_path=DEST/'train_balanced_full_pass.jsonl'
    after_path=DEST/'train_balanced_after_pilot.jsonl'
    tr_pool_path=DEST/'train_turkish_pool.jsonl';replay_pool_path=DEST/'train_replay_pool.jsonl'
    write_rows(full_path,(r for groups in sequence for group in groups for r in group))
    write_rows(after_path,(r for groups in sequence[250:] for group in groups for r in group))
    write_rows(tr_pool_path,tr.values());write_rows(replay_pool_path,replay.values())
    full_TR=[r for groups in sequence for r in groups[0]]
    assert len({r['source_id'] for r in full_TR})==74883
    assert sum(bool(r.get('plan_padding_repeat')) for r in full_TR)==5
    continuation={'schema':1,'kind':'native_sequential_full_TR_pass_after_pilot_v1','passed':True,
        'pilot_proof_sha256':sha(proof_path),'seed':SEED,'manifest_sha256':{str(p):sha(p) for p in [full_path,after_path,tr_pool_path,replay_pool_path]},
        'full_updates':len(sequence),'pilot_completed_updates':250,'continuation_updates':len(sequence)-250,
        'full_TR_unique_clips':74883,'pilot_TR_unique_clips':2000,'new_TR_unique_clips_after_pilot':72883,
        'final_TR_batch_padding_repetitions':5,'dropped_final_TR_clips':0,'replay_repetitions_explicit':True,
        'language_draw_counts_full':counts,'full_manifest_pilot_prefix_rows':4000,
        'resume_instruction':'Use full manifest with an explicitly verified saved data position4000, or use after_pilot suffix as a declared loader-progress transition; never rerun the250-only manifest for long training.',
        'optimizer_sampler_state_limitation':'This is a preplanned ordered data stream, not a claim that Lightning automatically restores custom data-loader position.',
        'future_FLEURS_domain_sampling':'Separate source streams can use checkpointed FLEURS/CV counters without duplicating accepted pool IDs; no weighting applied in this baseline plan.'}
    (OUTPUT/'full_nemo_training_continuation_v1.json').write_text(json.dumps(continuation,indent=2)+'\n')
    print(json.dumps({'continuation_ready':True,'remaining_unique_TR':72883,'updates':len(sequence)-250}),flush=True)


if __name__=='__main__':main()
