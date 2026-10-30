"""Measure actual unchanged HF replay acceptance, not raw2000 replay labels."""
from pathlib import Path
import hashlib, importlib.util, json, math, os, sys, time
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
from prepare_HF_turkish_BPE_512_v1 import Measurements, rows, sha
from metrics import normalize_text
from tokenizers import Tokenizer

started = time.perf_counter()
source_report = OUT / 'HF_turkish_BPE_512_candidate_v1.json'
research = json.loads(source_report.read_text(encoding='utf-8'))
config_path = OUT / 'runs/adaptive_20261011_hf_quality_window_01_stronger_kd/run_config.json'
config = json.loads(config_path.read_text(encoding='utf-8'))
base = Tokenizer.from_file(str(OUT / 'checkpoint/tokenizer.json'))
candidate = Tokenizer.from_file(str(ROOT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json'))
accepted = {lang: Measurements() for lang in ['en', 'es', 'it', 'fr', 'de']}
accepted_ids = []
new_piece_rows = {lang: [] for lang in accepted}
excluded = 0
for row in rows(config['replay_manifest']):
    old = base.encode(row['text'], add_special_tokens=False)
    decoded = base.decode(old.ids, skip_special_tokens=True)
    eligible = (bool(old.ids) and len(old.ids) <= config['max_tokens'] and max(old.ids) < 8193
                and normalize_text(decoded, row['language']) == normalize_text(row['text'], row['language']))
    if not eligible:
        excluded += 1
        continue
    new = candidate.encode(row['text'], add_special_tokens=False)
    new_decoded = candidate.decode(new.ids, skip_special_tokens=True)
    accepted[row['language']].add(row, old, new, decoded, new_decoded,
                                 math.ceil(row['duration'] * 100 / 8), normalize_text)
    accepted_ids.append(row['source_id'])
    if any(identifier >= 8199 for identifier in new.ids):
        new_piece_rows[row['language']].append(row['source_id'])
if len(accepted_ids) != config['replay_clips_used']:
    raise AssertionError('Reconstructed pool differs from actual current HF run_config')
old_mean = sum(x.old_matrix / len(x.old) for x in accepted.values()) / 5
new_mean = sum(x.new_matrix / len(x.new) for x in accepted.values()) / 5
tr_matrix = research['Turkish_all_TRAIN']['matrix_arithmetic']
mixed_old = .5 * tr_matrix['baseline_mean_per_clip_bytes'] + .5 * old_mean
mixed_new = .5 * tr_matrix['candidate_mean_per_clip_bytes'] + .5 * new_mean
results = {lang: x.result() for lang, x in accepted.items()}
report = {
    'schema': 1, 'kind': 'HF_Turkish_BPE_512_actual_current_training_pool_review_v1',
    'CPU_only': True, 'Torch_Transformers_NeMo_imported': any(x == y or x.startswith(y + '.') for x in sys.modules for y in ['torch', 'transformers', 'nemo']),
    'candidate_activated': False, 'source_research_report': {'path': str(source_report), 'sha256': sha(source_report)},
    'actual_current_HF_run_config': {'path': str(config_path), 'sha256': sha(config_path), 'run_id': config['run_id']},
    'candidate_tokenizer': {'path': str(ROOT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json'),
                            'sha256': sha(ROOT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json')},
    'raw_replay_clips': len(accepted_ids) + excluded, 'accepted_actual_HF_replay_clips': len(accepted_ids),
    'already_excluded_by_unchanged_HF_baseline': excluded,
    'baseline_acceptance': {'max_target_tokens': config['max_tokens'], 'teacher_only_max_ID_exclusive': 8193,
                             'normalized_reference_roundtrip': True, 'decoded_without_CTC_grouping': True},
    'accepted_source_IDs_sha256_sorted_newline': hashlib.sha256('\n'.join(sorted(accepted_ids)).encode()).hexdigest(),
    'actual_old_language_TRAIN': results,
    'passed_normalized_roundtrip_for_actual_HF_pool': research['Turkish_all_TRAIN']['candidate_normalized_roundtrip'] == 74883
        and all(x.counts['candidate_normalized_roundtrip'] == len(x.old) for x in accepted.values()),
    'all_actual_replay_decodes_equal_baseline': all(x.counts['decode_equals_baseline'] == len(x.old) for x in accepted.values()),
    'candidate_new_piece_rows_in_actual_replay': {lang: {'count': len(ids), 'finite_source_IDs': ids[:4]} for lang, ids in new_piece_rows.items()},
    'equal_TR_replay_clip_matrix_projection': {
        'TR_fraction': .5, 'replay_language_sampling': 'uniform over5languages then each accepted pool',
        'baseline_mean_FP32_joint_bytes': mixed_old, 'candidate_mean_FP32_joint_bytes': mixed_new,
        'candidate_to_baseline_ratio': mixed_new / mixed_old,
        'scope': 'unbatched joint arithmetic only; no actual VRAM or training speed claim'},
    'future_retention_policy_required': [
        'Existing replay teacher_only check cannot blindly use candidate-tokenized new piece IDs: the frozen teacher embedding only supports old IDs.',
        'Keep all1595 replay clips and use their existing baseline label IDs/cache with the original tokenizer identity for replay objectives; candidate targets can be used for Turkish only. Old IDs decode identically through the candidate.',
        'Declare the role-specific target/cache policy; retain old teacher prefix conditioning and KL/duration masks. Do not silently drop changed-encoding replay or pretend old and new target sequences align.',
        'Inference uses candidate decoder; changed old-language encoding remains measured and all5 retention DEV gates remain necessary.'
    ],
    'not_a_quality_or_memory_validation': True, 'DEV_TEST_read': False,
    'finalizer_source_sha256': sha(__file__), 'elapsed_seconds': time.perf_counter() - started}
p = OUT / 'HF_turkish_BPE_512_actual_training_pool_review_v1.json'
p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'report': str(p), 'accepted': len(accepted_ids),
                  'passed_roundtrip': report['passed_normalized_roundtrip_for_actual_HF_pool'],
                  'old_language_encoding_changed': {lang: x.counts['encoding_changed'] for lang, x in accepted.items()},
                  'mixed_matrix_ratio': mixed_new / mixed_old}))
