"""Read-only active-run cache/provenance audit, without loading model weights."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'parakeet_turkish'
sys.path.insert(0, str(OUT))
import torch
from tokenizers import Tokenizer
from transformers import AutoProcessor
from extend_vocab import CHARACTER_DONORS, _make_tokenizer, _hash_json
from metrics import normalize_text

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]

checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

loop_path = OUT / 'runs/adaptive_20261010_01/loop_status.json'
loop_bytes = loop_path.read_bytes()
loop = json.loads(loop_bytes)
run_id = loop['active_run_id']
run_dir = OUT / 'runs' / run_id
config = read(run_dir / 'run_config.json')
pool = read(run_dir / 'training_pool.json')
audit = read(run_dir / 'vocabulary_audit.json')
config_sha, pool_sha = digest(run_dir / 'run_config.json'), digest(run_dir / 'training_pool.json')
check('active_run_matches_saved_run_id', config['run_id'] == run_id)
for name, sha in config['code_sha256'].items():
    check('frozen_code:' + name, digest(OUT / name) == sha)
for path, sha in config['input_sha256'].items():
    check('unchanged_input:' + Path(path).name, digest(path) == sha)
check('pool_input_provenance', pool['input_sha256'] == config['input_sha256'])
check('controller_frozen_hash', digest(OUT / 'run_experiment_loop.py') == loop['controller_sha256'])
check('selected_profile_hash', digest(OUT / 'batching_speed_comparison.json') == loop['profile_sha256'])
selected = loop['selected_configuration']
for key in ('batch_size', 'accumulation', 'tdt_implementation', 'chunk_size'):
    check('profile_configuration:' + key, config[key] == selected[key])
check('profile_encoder_checkpoint', config['encoder_activation_checkpointing'] == selected['encoder_checkpointing'])
trial = next(row for row in loop['trials'] if row['run_id'] == run_id)
check('warm_start_weights_hash', digest(config['resume_weights']) == config['warm_start']['sha256'] == trial['initialization_sha256'])

# Processor/tokenizer only. No AutoModel or CUDA API is called.
base = Path((ROOT / 'work/base_model_path.txt').read_text().strip())
processor = AutoProcessor.from_pretrained(base, local_files_only=True)
old = processor.tokenizer
backend = old._tokenizer
updated = copy.deepcopy(json.loads(backend.to_str()))
vocab = backend.get_vocab()
size = len(old)
for token, index in sorted(vocab.items(), key=lambda pair: pair[1]):
    updated['model']['vocab'].setdefault(token, index)
additions = [char for char in CHARACTER_DONORS if char not in vocab]
for index, char in enumerate(additions, start=size):
    updated['model']['vocab'][char] = index
processor.tokenizer = _make_tokenizer(old, Tokenizer.from_str(json.dumps(updated, ensure_ascii=False)))
tokenizer = processor.tokenizer
check('extended_tokenizer_semantic_identity', _hash_json(json.loads(tokenizer._tokenizer.to_str())) == audit['extended_tokenizer_sha256'])
namespace = hashlib.sha256((config['base_revision'] + tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
check('teacher_cache_base_revision_and_tokenizer_namespace', namespace == config['encoder_cache_namespace'])
check('pinned_base_directory_identity', base.name == config['base_revision'])

inputs = {'turkish': jsonl(config['tr_manifest']), 'replay': jsonl(config['replay_manifest'])}
if config.get('supplement'):
    inputs['turkish'] += jsonl(config['supplement'])
heldout = [row for path in config['input_sha256'] if Path(path).name.startswith(('test_', 'validation_')) for row in jsonl(path)]
heldout_ids = {row['source_id'] for row in heldout}
heldout_sentences = {row['sentence_id'] for row in heldout if row.get('sentence_id') is not None}
heldout_audio = {row[key] for row in heldout for key in ('audio_sha256', 'wav_sha256') if row.get(key)}
heldout_refs = {(row['language'], normalize_text(row['text'], row['language'])) for row in heldout}
seen = set()
for stream in ('turkish', 'replay'):
    accepted = []
    for row in inputs[stream]:
        ids = tokenizer(row['text'])['input_ids']
        if (ids and len(ids) <= config['max_tokens'] and (stream == 'turkish' or max(ids) < size)
                and normalize_text(tokenizer.decode(ids, skip_special_tokens=True), row['language'])
                == normalize_text(row['text'], row['language'])):
            accepted.append(row)
    rows = pool[stream]
    check(stream + ':exact_ordered_lexical_acceptance', [r['source_id'] for r in accepted] == [r['source_id'] for r in rows])
    check(stream + ':configured_count', len(rows) == config['turkish_clips_used' if stream == 'turkish' else 'replay_clips_used'])
    check(stream + ':configured_pool_hash', hashlib.sha256('\n'.join(sorted(r['source_id'] for r in rows)).encode()).hexdigest()
          == config['config_identity']['tr_pool_sha256' if stream == 'turkish' else 'replay_pool_sha256'])
    check(stream + ':configured_hours', abs(math.fsum(r['duration'] for r in rows) / 3600 -
          config['turkish_unique_pool_hours' if stream == 'turkish' else 'replay_unique_pool_hours']) < 1e-12)
    for original, row in zip(accepted, rows):
        check('unchanged_pool_source:' + row['source_id'], all(row[key] == value for key, value in original.items()))
        source_id = row['source_id']
        path = Path(row['cache_path'])
        check('cache_source_key:' + source_id, path.name == hashlib.sha256(source_id.encode()).hexdigest() + '.pt'
              and path.parent.name == namespace and path.is_file())
        check('train_isolation:' + source_id, source_id not in seen and row['dataset_split'] == 'train'
              and source_id not in heldout_ids and row.get('sentence_id') not in heldout_sentences
              and not any(row.get(key) in heldout_audio for key in ('audio_sha256', 'wav_sha256'))
              and (row['language'], normalize_text(row['text'], row['language'])) not in heldout_refs)
        seen.add(source_id)

# Read a deterministic stratified subset of tensor files; the unchanged trainer
# already loaded every file and validated labels/finiteness before writing config.
samples = []
for language in ('tr', 'en', 'es', 'it', 'fr', 'de'):
    candidates = [row for stream in pool.values() if isinstance(stream, list) for row in stream if row['language'] == language]
    rng = random.Random('teacher-cache-audit-' + language)
    samples.extend(rng.sample(candidates, min(12, len(candidates))))
for row in samples:
    item = torch.load(row['cache_path'], map_location='cpu', weights_only=True)
    labels = torch.tensor(tokenizer(row['text'])['input_ids'], dtype=torch.long)
    check('sample_cache_source:' + row['source_id'], item['source_id'] == row['source_id'])
    check('sample_cache_labels:' + row['source_id'], torch.equal(labels, item['labels']))
    check('sample_cache_frames:' + row['source_id'], item['features'].shape == (row['encoder_frames'], 1024)
          and bool(torch.isfinite(item['features']).all()) and item['labels'].numel() == row['target_tokens'])
    check('sample_wav_bytes:' + row['source_id'], digest(row['audio_filepath']) == row['wav_sha256'])

source = (OUT / 'train_lora_batched.py').read_text(encoding='utf-8')
check('teacher_captured_before_extension', source.index('teacher = {name: copy.deepcopy') < source.index('vocabulary = extend_processor_and_model'))
check('cache_created_before_adapters_and_warm_start', source.index('tr = annotate_cache_lengths(make_cache') < source.index('layout = install_encoder_lora') < source.index('model.load_state_dict(resume'))
check('configured_original_teacher', 'original frozen heads' in config['teacher'] and 'original frozen singleton encoder cache' in config['teacher'])
check('replay_preservation_coefficients', config['replay_kl'] == 5 and config['duration_kl'] == .5 and config['feature_distillation'] == 1)
check('frozen_encoder_base_statistics', config['encoder_base_frozen'] and config['batchnorm_stats_frozen'] and config['encoder_dropout_and_layerdrop_disabled'])
check('config_and_pool_unchanged_during_cpu_audit', digest(run_dir / 'run_config.json') == config_sha and digest(run_dir / 'training_pool.json') == pool_sha)

report = {'schema_version': 1, 'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'CPU processor/tokenizer/source/manifest/cache audit; no model-weight load or GPU call.',
          'run_id': run_id, 'checks': len(checks), 'turkish_clips': len(pool['turkish']),
          'replay_clips': len(pool['replay']), 'replay_language_counts': dict(Counter(row['language'] for row in pool['replay'])),
          'turkish_hours': config['turkish_unique_pool_hours'], 'replay_hours': config['replay_unique_pool_hours'],
          'tensor_cache_samples_read': len(samples), 'namespace': namespace,
          'run_config_sha256': config_sha, 'training_pool_sha256': pool_sha,
          'loop_status_snapshot_sha256': hashlib.sha256(loop_bytes).hexdigest(),
          'code_sha256': config['code_sha256'], 'input_sha256': config['input_sha256'],
          'cache_key_design': {'namespace': 'SHA256(pinned base revision + exact extended HF tokenizer serialization)[:20]',
             'entry': 'SHA256(source_id)', 'reuse_validation': 'Exact cached labels plus source ID/frame/target/finiteness checks before training'},
          'preservation_design': {'teacher': 'Pinned original encoder_projector/decoder/joint copied before vocab extension; original singleton encoder cache created before LoRA/warm-start.',
             'feature_loss': 'Masked per-replay-clip cosine distance from current encoder features to original cached features; literal zero on Turkish rows; mean over all clips.',
             'token_duration_kl': 'Original teacher sees replay only with old IDs; student full extended normalization; <=8x8 valid per-clip grid.',
             'coefficients': {'token_kl': 5, 'duration_kl': .5, 'feature': 1}},
          'limitations': ['Cache entry keys do not directly bind WAV hashes or frontend/runtime code. Run input/code hashes and current prepared-WAV identity are separately audited; a future in-place audio/frontend change should use a new cache namespace.',
                          'Independently reread 72 tensor caches and WAVs, not all tensor caches. The frozen active trainer checked every accepted cache before writing this completed pool/config.'],
          'material_active_run_flaws_found': []}
target = OUT / 'runs' / run_id / 'teacher_cache_cpu_audit.json'
target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ('passed', 'checks', 'run_id', 'turkish_clips', 'replay_clips', 'turkish_hours', 'replay_hours', 'tensor_cache_samples_read')}, ensure_ascii=False))
print(str(target))
