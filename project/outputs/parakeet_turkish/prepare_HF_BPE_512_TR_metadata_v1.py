"""Regenerate inert TRAIN-only targets; reuse audited PCM/frontend dimensions.

The existing manifest, baseline metadata and both tokenizer assets are read-only.
No model, audio API, DEV/TEST input, token fitting or training helper is imported.
"""
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import ctypes
import datetime
import hashlib
import json
import math
import os
import sys
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['RAYON_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
DATA = ROOT / 'work/data'
DEST = DATA / 'tr_public_v3_frame_target_metadata_HF_BPE_512_v1.jsonl'
PROOF = OUT / 'HF_BPE_512_TR_target_metadata_v1.json'
KIND = 'TR_frame_target_metadata_HF_BPE_512_v1'
MANIFEST_SHA = 'd2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704'
BASE_METADATA_SHA = '95eeb218ef88faaec4b2f04b438527f9442dd4ab8d9de165880aaf38fd764837'
CANDIDATE_SHA = '6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327'
BASE_SEMANTIC_SHA = 'b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
CANDIDATE_SEMANTIC_SHA = '10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607'
COPIED_ROW_KEYS = ('audio_filepath', 'duration', 'sample_rate', 'audio_sha256',
                   'wav_sha256', 'language', 'dataset', 'dataset_revision',
                   'dataset_config', 'dataset_split', 'dataset_file')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False).encode('utf-8')


def semantic_sha(value):
    # Exact convention used by the original HF tokenizer/frame proof.
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   ensure_ascii=False).encode('utf-8')).hexdigest()


def label_sha(ids):
    # Exact label convention in tr_frame_metadata_v1.py.
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode('utf-8')).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summary(values):
    ordered = sorted(values)
    n = len(ordered)
    return {'clips': n, 'sum': sum(ordered), 'min': ordered[0],
            'mean': sum(ordered) / n, 'median': ordered[n // 2],
            'p90': ordered[min(n - 1, int(.9 * n))], 'max': ordered[-1]}


def main():
    if DEST.exists() or PROOF.exists():
        raise FileExistsError('Existing inert metadata/proof must not be overwritten')
    started = time.perf_counter()
    priority = False
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        priority = bool(api.SetPriorityClass(api.GetCurrentProcess(), 0x4000))
    import tokenizers
    from metrics import normalize_text
    paths = {
        'accepted_TR_manifest': DATA / 'tr_public_v3_train.jsonl',
        'audited_baseline_frame_target_metadata': DATA / 'tr_public_v3_frame_metadata_v1.jsonl',
        'audited_baseline_frame_proof': OUT / 'TR_frame_metadata_cpu_verification_v1.json',
        'actual_HF_baseline_tokenizer': OUT / 'checkpoint/tokenizer.json',
        'inert_candidate_tokenizer': ROOT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json',
        'candidate_freeze': ROOT / 'work/hf_turkish_bpe_candidate_512_v1/candidate_frozen_before_measurement.json',
        'candidate_TRAIN_measurement': OUT / 'HF_turkish_BPE_512_candidate_v1.json',
        'frozen_frame_helper_source': OUT / 'tr_frame_metadata_v1.py',
        'frozen_batch_helper_source': OUT / 'batched_lora_tr_metadata_v1.py',
        'shared_normalizer_source': OUT / 'metrics.py',
    }
    inputs = {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()}
    require(inputs['accepted_TR_manifest']['sha256'] == MANIFEST_SHA, 'Accepted TRAIN manifest differs')
    require(inputs['audited_baseline_frame_target_metadata']['sha256'] == BASE_METADATA_SHA,
            'Audited baseline dimensions/targets differ')
    require(inputs['inert_candidate_tokenizer']['sha256'] == CANDIDATE_SHA, 'Frozen candidate differs')
    frame_proof = json.loads(paths['audited_baseline_frame_proof'].read_text(encoding='utf-8'))
    require(frame_proof.get('passed') is True and frame_proof.get('CPU_only') is True
            and frame_proof.get('clips') == 74883 and frame_proof.get('manifest_sha256') == MANIFEST_SHA
            and frame_proof.get('metadata_sha256') == BASE_METADATA_SHA
            and frame_proof.get('cache_files_written') == 0, 'Completed audited frame proof required')
    candidate_measurement = json.loads(paths['candidate_TRAIN_measurement'].read_text(encoding='utf-8'))
    require(candidate_measurement.get('fit_performed') is False
            and candidate_measurement.get('DEV_or_TEST_text_read') is False
            and candidate_measurement['candidate_files_sha256'][str(paths['inert_candidate_tokenizer'])] == CANDIDATE_SHA,
            'Existing TRAIN-only candidate measurement is not bound')
    freeze = json.loads(paths['candidate_freeze'].read_text(encoding='utf-8'))
    require(freeze.get('fit_performed') is False and freeze.get('DEV_or_TEST_text_read') is False
            and freeze['files_sha256'][str(paths['inert_candidate_tokenizer'])] == CANDIDATE_SHA,
            'Prior candidate freeze differs')
    base_json = json.loads(paths['actual_HF_baseline_tokenizer'].read_text(encoding='utf-8'))
    candidate_json = json.loads(paths['inert_candidate_tokenizer'].read_text(encoding='utf-8'))
    require(semantic_sha(base_json) == BASE_SEMANTIC_SHA, 'Actual 8199 baseline identity differs')
    require(semantic_sha(candidate_json) == CANDIDATE_SEMANTIC_SHA, 'Candidate semantic identity differs')
    require(len(base_json['model']['vocab']) == 8199 and len(candidate_json['model']['vocab']) == 8711,
            'Required 8199-to-8711 vocabulary sizes differ')
    require(all(candidate_json['model']['vocab'].get(piece) == identifier
                for piece, identifier in base_json['model']['vocab'].items()), 'Original IDs moved')
    require(candidate_json['model']['merges'][:len(base_json['model']['merges'])] == base_json['model']['merges'],
            'Original merge priorities changed')
    require({key: value for key, value in candidate_json.items() if key != 'model'}
            == {key: value for key, value in base_json.items() if key != 'model'},
            'Normalizer, pretokenizer, decoder or special token settings differ')
    baseline = tokenizers.Tokenizer.from_str(json.dumps(base_json, ensure_ascii=False))
    candidate = tokenizers.Tokenizer.from_str(json.dumps(candidate_json, ensure_ascii=False))
    temporary = DEST.with_suffix('.pending.jsonl')
    if temporary.exists():
        raise FileExistsError('An incomplete prior metadata output exists; preserve it for inspection')
    old_lengths, new_lengths, durations = [], [], []
    counts, source_ids, old_frame_digest, new_frame_digest = Counter(), set(), hashlib.sha256(), hashlib.sha256()
    ordered_ids, manifest_rows, examples = hashlib.sha256(), hashlib.sha256(), []
    with temporary.open('x', encoding='utf-8', newline='\n') as output:
        for index, (row, parent) in enumerate(zip_longest(rows(paths['accepted_TR_manifest']),
                                                        rows(paths['audited_baseline_frame_target_metadata']))):
            require(row is not None and parent is not None, 'Manifest/metadata row counts differ')
            require(row['source_id'] == parent['source_id'] and row['source_id'] not in source_ids,
                    'Ordered source identity differs or duplicates')
            require(row['language'] == 'tr' and row['dataset_split'] in {'train', 'validated'},
                    'Only the prior accepted Turkish TRAIN pool may be read')
            require(row['sample_rate'] == 16000 and math.isfinite(row['duration']) and row['duration'] > 0,
                    'Prior positive 16kHz dimensions required')
            frames = parent['actual_PCM_frames']
            mel, T = frames // 160, (frames // 160 + 7) // 8
            require(int(round(row['duration'] * 16000)) == frames
                    and parent['valid_mel_frames'] == mel and parent['encoder_frames'] == T,
                    'Copied manifest duration/PCM/mel/encoder dimensions disagree')
            old = baseline.encode(row['text'], add_special_tokens=False).ids
            new = candidate.encode(row['text'], add_special_tokens=False).ids
            require(len(old) == parent['target_tokens'] and label_sha(old) == parent['label_ids_sha256'],
                    'Rust baseline does not reproduce the audited original HF target labels')
            require(1 <= len(new) <= 180 and min(new) >= 0 and max(new) < 8711 and 8192 not in new,
                    'Candidate target cannot retain this accepted row; no cropping/dropping is permitted')
            decoded = candidate.decode(new, skip_special_tokens=True)
            require(normalize_text(decoded, 'tr') == normalize_text(row['text'], 'tr'),
                    'Candidate fails the original scored TRAIN roundtrip')
            require(decoded == baseline.decode(old, skip_special_tokens=True), 'Candidate decoded TRAIN reference differs')
            copied = {key: value for key, value in parent.items() if key not in {'target_tokens', 'label_ids_sha256'}}
            target = {**parent, 'target_tokens': len(new), 'label_ids_sha256': label_sha(new),
                      'kind': KIND, 'manifest_row_index': index, 'label_ids': new,
                      'training_manifest_row_sha256': hashlib.sha256(canonical(row)).hexdigest(),
                      'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                      **{key: row[key] for key in COPIED_ROW_KEYS}}
            require(all(target[key] == value for key, value in copied.items()), 'An audited non-target field changed')
            old_frame_digest.update(canonical(copied) + b'\n')
            new_frame_digest.update(canonical({key: target[key] for key in copied}) + b'\n')
            source_ids.add(row['source_id']); ordered_ids.update((row['source_id'] + '\n').encode('utf-8'))
            manifest_rows.update(canonical(row) + b'\n')
            counts['clips'] += 1; counts['candidate_roundtrip_passed'] += 1
            counts['original_target_label_hashes_reproduced'] += 1
            counts['encoding_changed'] += old != new
            counts['target_length_changed'] += len(old) != len(new)
            counts['contains_new_512_piece'] += any(identifier >= 8199 for identifier in new)
            counts['encoder_frames_total'] += T; counts['actual_PCM_frames_total'] += frames
            old_lengths.append(len(old)); new_lengths.append(len(new)); durations.append(row['duration'])
            if len(examples) < 4 and old != new:
                examples.append({'source_id': row['source_id'], 'encoder_frames': T,
                                 'baseline_target_tokens': len(old), 'candidate_target_tokens': len(new),
                                 'baseline_label_ids_sha256': label_sha(old),
                                 'candidate_label_ids_sha256': target['label_ids_sha256']})
            output.write(json.dumps(target, ensure_ascii=False, separators=(',', ':')) + '\n')
            if (index + 1) % 10000 == 0:
                print(json.dumps({'phase': 'inert_HF_BPE_512_TRAIN_target_regeneration', 'clips': index + 1}), flush=True)
    require(counts['clips'] == 74883 and old_frame_digest.digest() == new_frame_digest.digest(),
            'The complete accepted ordered dimension pool was not retained')
    require(sum(new_lengths) == candidate_measurement['Turkish_all_TRAIN']['candidate_U']['sum']
            and sum(old_lengths) == candidate_measurement['Turkish_all_TRAIN']['baseline_U']['sum'],
            'Candidate target lengths differ from the prior TRAIN-only measurement')
    # Validate persisted IDs and dimensions rather than trusting only writer counters.
    saved = 0
    for target, parent, row in zip_longest(rows(temporary), rows(paths['audited_baseline_frame_target_metadata']),
                                         rows(paths['accepted_TR_manifest'])):
        require(target is not None and parent is not None and row is not None, 'Serialized metadata counts differ')
        require(target['manifest_row_index'] == saved and target['source_id'] == row['source_id'] == parent['source_id'],
                'Serialized order/source identity differs')
        require(all(target[key] == row[key] for key in COPIED_ROW_KEYS), 'Serialized path/duration/provenance differs')
        require(all(target[key] == value for key, value in parent.items()
                    if key not in {'target_tokens', 'label_ids_sha256'}), 'Serialized prior dimensions differ')
        require(target['target_tokens'] == len(target['label_ids'])
                and target['label_ids_sha256'] == label_sha(target['label_ids'])
                and target['training_manifest_row_sha256'] == hashlib.sha256(canonical(row)).hexdigest(),
                'Serialized target hash/manifest identity differs')
        saved += 1
    require(saved == 74883, 'Serialized whole TRAIN pool required')
    for key, path in paths.items():
        require(sha(path) == inputs[key]['sha256'], 'An input/source changed during regeneration: ' + key)
    require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules
                    for prefix in ('torch', 'transformers', 'nemo', 'soundfile', 'scipy', 'torchaudio')),
            'Prohibited model/audio API import occurred')
    temporary.rename(DEST)
    report = {
        'schema': 1, 'kind': 'inert_HF_BPE_512_TR_target_metadata_proof_v1', 'passed': True,
        'CPU_only': True, 'candidate_activated': False, 'active_files_modified': False,
        'fit_performed': False, 'DEV_or_TEST_inputs_read': False, 'audio_files_or_APIs_read': False,
        'Torch_Transformers_NeMo_imported': False, 'cache_files_written': 0,
        'inputs': inputs, 'script_path': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
        'metadata_path': str(DEST), 'metadata_sha256': sha(DEST), 'metadata_kind': KIND,
        'manifest_sha256': MANIFEST_SHA, 'parent_metadata_sha256': BASE_METADATA_SHA,
        'baseline_tokenizer_semantic_sha256': BASE_SEMANTIC_SHA,
        'candidate_tokenizer_semantic_sha256': CANDIDATE_SEMANTIC_SHA,
        'candidate_tokenizer_file_sha256': CANDIDATE_SHA,
        'baseline_vocab_size': 8199, 'candidate_vocab_size': 8711, 'blank_id_unchanged': 8192,
        'teacher_original_vocab_size_unchanged': 8193, 'max_target_tokens_policy_unchanged': 180,
        'counts': dict(counts), 'clips': saved,
        'actual_PCM_hours_copied': counts['actual_PCM_frames_total'] / 16000 / 3600,
        'manifest_duration_hours_copied': math.fsum(durations) / 3600,
        'ordered_source_ids_sha256': ordered_ids.hexdigest(),
        'canonical_manifest_rows_sha256': manifest_rows.hexdigest(),
        'copied_non_target_parent_fields_sha256': old_frame_digest.hexdigest(),
        'regenerated_non_target_parent_fields_sha256': new_frame_digest.hexdigest(),
        'source_ID_order_and_all_prior_frame_values_preserved': True,
        'paths_durations_dataset_provenance_and_text_reference_unchanged': True,
        'clips_dropped_or_cropped': 0, 'serialized_rows_rechecked': saved,
        'baseline_target_tokens': summary(old_lengths), 'candidate_target_tokens': summary(new_lengths),
        'target_token_sum_reduction_fraction': 1 - sum(new_lengths) / sum(old_lengths),
        'finite_examples': examples,
        'future_helper_incompatibilities': [
            'tr_frame_metadata_v1.py pins baseline tokenizer b91d500..., metadata95eeb218..., vocab8199 and parentproof kind; it must not consume this new proof/metadata.',
            'batched_lora_tr_metadata_v1.py imports the old KIND/load_training_item; a future isolated helper must read label_ids from this metadata and bind candidate8711/tokenizer identity.',
            'Existing replay cache labels and original8193 teacher remain unchanged; never re-tokenize replay targets with the candidate.',
            'Future student durations move to8711:8716; six-character and512-piece optimizer row policies must remain distinct.'
        ],
        'can_current_frozen_trainer_consume_metadata': False,
        'scope': 'Inert actual Rust-tokenizer TRAIN targets only. No model inference, training, GPU memory or WER improvement claim.',
        'runtime': {'tokenizers_version': tokenizers.__version__, 'TOKENIZERS_PARALLELISM': 'false',
                    'RAYON_NUM_THREADS': '2', 'below_normal_priority_applied': priority,
                    'elapsed_seconds': time.perf_counter() - started,
                    'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    }
    PROOF.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'clips': saved, 'metadata': str(DEST), 'metadata_sha256': report['metadata_sha256'],
                      'proof': str(PROOF), 'baseline_U_sum': sum(old_lengths), 'candidate_U_sum': sum(new_lengths),
                      'elapsed_seconds': report['runtime']['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
