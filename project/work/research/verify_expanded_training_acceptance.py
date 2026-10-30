"""CPU-only verification of the actual expanded trainer acceptance artifacts."""
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
from metrics import normalize_text


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


run_id='adaptive_20261010_01_screen_A';run=OUT/'runs'/run_id
config=read(run/'run_config.json');pool=read(run/'training_pool.json');vocabulary=read(run/'vocabulary_audit.json')
prep=read(OUT/'expanded_turkish_preparation.json');audit=read(OUT/'expanded_turkish_verification.json')
manifest=ROOT/'work/data/tr_expanded_train.jsonl';prepared=rows(manifest)
assert config['run_id']==run_id and config['train_limit'] is None and config['supplement'] is None
assert Path(config['tr_manifest']).resolve()==manifest.resolve()
assert prep['manifest_sha256']==audit['manifest_sha256']==sha(manifest)==config['input_sha256'][str(manifest)]
assert pool['input_sha256']==config['input_sha256'] and audit['passed'] is True
assert all(sha(Path(path))==wanted for path,wanted in config['input_sha256'].items())
assert all(sha(OUT/name)==wanted for name,wanted in config['code_sha256'].items())
assert all(sha(ROOT/name)==wanted for name,wanted in prep['protected_original_files_sha256'].items())
tokenizer=AutoTokenizer.from_pretrained(str(OUT/'checkpoint'),local_files_only=True)
semantic=hashlib.sha256(json.dumps(json.loads(tokenizer._tokenizer.to_str()),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
assert semantic==vocabulary['extended_tokenizer_sha256']==config['config_identity']['extended_tokenizer_sha256']==audit['lexical_filter']['tokenizer_semantic_sha256']
accepted=[];exclusions=Counter();label_counts=Counter();by_source={}
for record in prepared:
    ids=tokenizer(record['text'])['input_ids'];reasons=[]
    if not ids:reasons.append('empty_target')
    if len(ids)>config['max_tokens']:reasons.append('over_max_tokens')
    if any(i<0 or i>=len(tokenizer) or i==8192 for i in ids):reasons.append('invalid_vocab_or_blank')
    if normalize_text(tokenizer.decode(ids,skip_special_tokens=True),'tr')!=normalize_text(record['text'],'tr'):reasons.append('lexical_roundtrip')
    if reasons:exclusions.update(reasons)
    else:
        accepted.append(record)
        label_counts.update(ids)
        group=by_source.setdefault(record['dataset'],{'clips':0,'hours':0.,'target_tokens':0})
        group['clips']+=1;group['hours']+=record['duration']/3600;group['target_tokens']+=len(ids)
actual=pool['turkish'];actual_by_id={row['source_id']:row for row in actual}
assert len(actual_by_id)==len(actual)==len(accepted)==config['turkish_clips_used']==audit['lexical_filter']['accepted_clips']==25628
assert set(actual_by_id)=={row['source_id'] for row in accepted}
for record in accepted:
    row=actual_by_id[record['source_id']]
    assert all(row[key]==value for key,value in record.items())
    assert row['target_tokens']==len(tokenizer(row['text'])['input_ids'])
    assert isinstance(row['encoder_frames'],int) and row['encoder_frames']>0
    assert Path(row['cache_path']).is_file() and Path(row['cache_path']).parent.name==config['encoder_cache_namespace']
assert len(prepared)-len(actual)==config['excluded_training_counts']['tr']==audit['lexical_filter']['excluded_clips']==1043
assert dict(exclusions)==audit['lexical_filter']['exclusion_counts_nonexclusive']=={'lexical_roundtrip':1043}
hours=math.fsum(row['duration'] for row in actual)/3600
assert abs(hours-config['turkish_unique_pool_hours'])<1e-10 and abs(hours-audit['lexical_filter']['accepted_hours'])<1e-10
for name,stats in by_source.items():
    wanted=audit['lexical_filter']['by_source'][name]
    assert stats['clips']==wanted['clips'] and stats['target_tokens']==wanted['target_tokens'] and abs(stats['hours']-wanted['hours'])<1e-10
replay=pool['replay'];pilot=read(OUT/'lora_training_data_verification.json')
assert len(replay)==config['replay_clips_used']==1553
assert {r['source_id'] for r in replay}=={r['source_id'] for r in pilot['streams']['retention_fleurs_train']['accepted_source_ids_and_labels']}
assert config['excluded_training_counts']['replay']==len(rows(ROOT/'work/data/replay_train.jsonl'))-len(replay)==447
assert hashlib.sha256('\n'.join(sorted(actual_by_id)).encode()).hexdigest()==config['config_identity']['tr_pool_sha256']
report={'passed':True,'verified_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU actual trainer acceptance/config/pool, exact published tokenizer filter on every prepared TR text, source metadata and code/input/protected-file SHA checks; no GPU or test predictions',
        'run_id':run_id,'manifest_sha256':sha(manifest),'run_config_sha256':sha(run/'run_config.json'),'training_pool_sha256':sha(run/'training_pool.json'),
        'expanded_audit_sha256':sha(OUT/'expanded_turkish_verification.json'),'tokenizer_semantic_sha256':semantic,'encoder_cache_namespace':config['encoder_cache_namespace'],
        'prepared_clips':len(prepared),'prepared_hours':prep['prepared_hours'],'accepted_turkish_clips':len(actual),'accepted_turkish_hours':hours,'accepted_by_source':by_source,
        'prepared_to_optimizer_excluded_clips':len(prepared)-len(actual),'exclusion_counts_nonexclusive':dict(exclusions),
        'source_preparation_exclusions':{'duplicate_normalized_references':80,'commonvoice':54,'fleurs_supplement':26},
        'additional_acoustic_cache_exclusions_after_lexical_filter':0,
        'cached_original_features_note':'All accepted rows have positive actual frame/target lengths and existing original-teacher cache paths. Frozen trainer annotate_cache_lengths validates every finite cached feature tensor before these artifacts are emitted; this CPU audit checks artifact lengths/presence, not a duplicate feature-value scan.',
        'new_character_label_occurrences':{str(i):label_counts[i] for i in range(8193,8199)},
        'replay_clips':len(replay),'replay_hours':config['replay_unique_pool_hours'],'replay_excluded_clips':447,
        'training_configuration':{'planned_updates':config['steps'],'screen_turkish_epochs':config['epochs'],'effective_optimizer_batch':config['effective_batch'],'tr_clips_per_update':config['tr_clips_per_update'],'replay_clips_per_update':config['replay_clips_per_update']},
        'heldout_predictions_accessed':False,'frozen_sources_modified':False,'script_sha256':sha(Path(__file__))}
(OUT/'expanded_training_acceptance_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({key:report[key] for key in ('passed','run_id','accepted_turkish_clips','accepted_turkish_hours','accepted_by_source','prepared_to_optimizer_excluded_clips','additional_acoustic_cache_exclusions_after_lexical_filter','replay_clips')}))
