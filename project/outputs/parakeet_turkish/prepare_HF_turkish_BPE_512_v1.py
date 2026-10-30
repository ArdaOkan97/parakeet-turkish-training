"""Inert HF merge translation; existing TRAIN-learned pieces, no model/GPU.

No DEV/TEST manifests are read, no merges are fitted, and no active assets change.
"""
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import copy, ctypes, datetime, hashlib, json, math, os, sys, time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
DEST = ROOT / 'work/hf_turkish_bpe_candidate_512_v1'
REPORT = OUT / 'HF_turkish_BPE_512_candidate_v1.json'
SEED = 20261010


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def summary(values):
    values = sorted(values)
    n = len(values)
    return {'clips': n, 'sum': sum(values), 'mean': sum(values) / n if n else None,
            'median': values[n // 2] if n else None,
            'p90': values[min(n - 1, int(.9 * n))] if n else None,
            'max': values[-1] if n else None}


class Measurements:
    def __init__(self):
        self.old, self.new = [], []
        self.counts, self.new_tokens = Counter(), Counter()
        self.examples = []
        self.old_matrix = self.new_matrix = 0
        self.worst = []

    def add(self, row, old, new, old_text, new_text, T, normalize):
        self.old.append(len(old.ids)); self.new.append(len(new.ids))
        lang = row['language']; text = row['text']
        self.counts['encoding_changed'] += old.ids != new.ids
        self.counts['baseline_raw_roundtrip'] += old_text == text
        self.counts['candidate_raw_roundtrip'] += new_text == text
        self.counts['baseline_normalized_roundtrip'] += normalize(old_text, lang) == normalize(text, lang)
        self.counts['candidate_normalized_roundtrip'] += normalize(new_text, lang) == normalize(text, lang)
        self.counts['decode_equals_baseline'] += new_text == old_text
        self.counts['baseline_unknown_clips'] += 0 in old.ids
        self.counts['candidate_unknown_clips'] += 0 in new.ids
        self.counts['baseline_unknown_tokens'] += old.ids.count(0)
        self.counts['candidate_unknown_tokens'] += new.ids.count(0)
        self.counts['candidate_target_over180'] += len(new.ids) > 180
        self.new_tokens.update(i for i in new.ids if i >= 8199)
        before, after = T * (len(old.ids) + 1) * 8204 * 4, T * (len(new.ids) + 1) * 8716 * 4
        self.old_matrix += before; self.new_matrix += after
        self.worst.append({'source_id': row['source_id'], 'language': lang, 'T': T,
                           'baseline_U': len(old.ids), 'candidate_U': len(new.ids),
                           'baseline_joint_FP32_bytes': before, 'candidate_joint_FP32_bytes': after})
        self.worst.sort(key=lambda value: value['candidate_joint_FP32_bytes'], reverse=True)
        del self.worst[8:]
        if len(self.examples) < 4 and (old.ids != new.ids or new_text != old_text):
            self.examples.append({'source_id': row['source_id'], 'reference': text,
                                  'baseline_U': len(old.ids), 'candidate_U': len(new.ids),
                                  'baseline_decode': old_text, 'candidate_decode': new_text,
                                  'baseline_tokens': old.tokens, 'candidate_tokens': new.tokens})

    def result(self):
        n = len(self.old)
        return {'baseline_U': summary(self.old), 'candidate_U': summary(self.new),
                'token_count_reduction_fraction': 1 - sum(self.new) / sum(self.old) if sum(self.old) else None,
                **dict(self.counts), 'new_piece_types_used': len(self.new_tokens),
                'new_piece_token_occurrences': sum(self.new_tokens.values()),
                'matrix_arithmetic': {'baseline_joint_FP32_bytes_sum': self.old_matrix,
                                      'candidate_joint_FP32_bytes_sum': self.new_matrix,
                                      'reduction_fraction': 1 - self.new_matrix / self.old_matrix if self.old_matrix else None,
                                      'baseline_mean_per_clip_bytes': self.old_matrix / n if n else None,
                                      'candidate_mean_per_clip_bytes': self.new_matrix / n if n else None},
                'worst8_candidate_joint_singletons': self.worst, 'finite_examples': self.examples}


def main():
    if DEST.exists() or REPORT.exists():
        raise FileExistsError('New inert paths required; existing candidates are immutable.')
    started = time.perf_counter()
    priority = False
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        priority = bool(api.SetPriorityClass(api.GetCurrentProcess(), 0x4000))
    import sentencepiece as spm
    import tokenizers
    from metrics import normalize_text
    paths = {
        'original_HF_tokenizer': ROOT / 'work/hf-cache/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/tokenizer.json',
        'actual_HF_extended_tokenizer': OUT / 'checkpoint/tokenizer.json',
        'native_baseline_SP': ROOT / 'work/nemo_direct_trial_v1/tokenizer/tokenizer.model',
        'native_candidate_SP': ROOT / 'work/turkish_bpe_candidates_v1/add_512/tokenizer.model',
        'TRAIN_learned_merges': ROOT / 'work/turkish_bpe_candidates_v1/add_512/added_pieces.json',
        'learning_partition': ROOT / 'work/turkish_bpe_candidates_v1/learning_partition.json',
        'native_candidate_freeze': ROOT / 'work/turkish_bpe_candidates_v1/candidate_files_frozen_before_DEV_read.json',
        'TR_TRAIN': ROOT / 'work/data/tr_public_v3_train.jsonl',
        'replay_TRAIN': ROOT / 'work/data/replay_train.jsonl',
        'TR_exact_frame_target_metadata': ROOT / 'work/data/tr_public_v3_frame_metadata_v1.jsonl',
        'TR_frame_proof': OUT / 'TR_frame_metadata_cpu_verification_v1.json',
    }
    inputs = {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()}
    if inputs['original_HF_tokenizer']['sha256'] != 'bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91':
        raise ValueError('Pinned original HF tokenizer changed')
    base = json.loads(paths['actual_HF_extended_tokenizer'].read_text(encoding='utf-8'))
    original = json.loads(paths['original_HF_tokenizer'].read_text(encoding='utf-8'))
    semantic = hashlib.sha256(json.dumps(base, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if semantic != 'b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75':
        raise ValueError('Actual current extended HF tokenizer identity differs')
    for piece, identifier in original['model']['vocab'].items():
        if base['model']['vocab'][piece] != identifier:
            raise ValueError('Original token ID changed')
    if base['model']['merges'] != original['model']['merges']:
        raise ValueError('Original merge priority changed')
    additions = json.loads(paths['TRAIN_learned_merges'].read_text(encoding='utf-8'))
    if len(additions) != 512 or base['model']['vocab'].get('<blank>') != 8192:
        raise ValueError('Expected original blank and exactly512 learned pieces')
    candidate = copy.deepcopy(base)
    translated = []
    for index, addition in enumerate(additions):
        piece, left, right = addition['piece'], addition['left'], addition['right']
        vocab = candidate['model']['vocab']
        if (piece in vocab or left not in vocab or right not in vocab or left + right != piece
                or addition['native_id'] != 8198 + index
                or (index and addition['score'] >= additions[index - 1]['score'])):
            raise ValueError('TRAIN merge dependency or priority does not translate')
        identifier = 8199 + index
        translated.append({**addition, 'HF_id': identifier, 'left_HF_id': vocab[left],
                           'right_HF_id': vocab[right], 'HF_merge_rank': len(base['model']['merges']) + index})
        vocab[piece] = identifier
        candidate['model']['merges'].append([left, right])
    if set(candidate['model']['vocab'].values()) != set(range(8711)):
        raise ValueError('Candidate token IDs are not contiguous')
    baseline = tokenizers.Tokenizer.from_str(json.dumps(base, ensure_ascii=False))
    updated = tokenizers.Tokenizer.from_str(json.dumps(candidate, ensure_ascii=False))
    native = spm.SentencePieceProcessor(model_file=str(paths['native_baseline_SP']))
    native_candidate = spm.SentencePieceProcessor(model_file=str(paths['native_candidate_SP']))
    partition = json.loads(paths['learning_partition'].read_text(encoding='utf-8'))
    if partition['learning_input_sha256'] != inputs['TR_TRAIN']['sha256'] or partition['seed'] != SEED:
        raise ValueError('Existing TRAIN-only learning partition is not bound to this pool')
    DEST.mkdir(parents=True)
    write(DEST / 'tokenizer.json', candidate)
    write(DEST / 'added_pieces.json', translated)
    migration = {'kind': 'inert_HF_Turkish_512_row_layout_v1', 'active_weights_modified': False,
                 'original_HF_lexical_and_blank_and_six_rows_copied': [0, 8199],
                 'blank_token_id_unchanged': 8192, 'new_HF_piece_rows': [8199, 8711],
                 'decoder_embedding_before_after': [8199, 8711],
                 'joint_head_before_after': [8204, 8716],
                 'duration_row_pairs': [[8199 + i, 8711 + i] for i in range(5)],
                 'generation_suppression_old_new': [[8199 + i, 8711 + i] for i in range(5)],
                 'additional_trainable_parameters_if_hidden640': 512 * (640 + 640 + 1),
                 'additional_FP32_parameter_gradient_Adam_moments_bytes': 512 * (640 + 640 + 1) * 16,
                 'future_initialization': 'Preserve all learned old rows. New embedding/head donors may use recursive TRAIN merge operands, with a declared low-probability bias; no numerical/function equivalence claim.',
                 'native_SP_asset_not_interchangeable_with_this_HF_JSON': True}
    write(DEST / 'row_migration_plan.json', migration)
    write(DEST / 'candidate_frozen_before_measurement.json', {
        'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'fit_performed': False, 'merges_read_from_prior_TRAIN_only_candidate': True,
        'DEV_or_TEST_text_read': False,
        'files_sha256': {str(DEST / name): sha(DEST / name) for name in ['tokenizer.json', 'added_pieces.json', 'row_migration_plan.json']}})
    tr = Measurements(); heldout = Measurements(); learning = Measurements()
    digests = {'learning': hashlib.sha256(), 'heldout': hashlib.sha256()}
    counts = Counter(); label_hash_mismatches = 0
    native_old_U, native_new_U = [], []
    with (DEST / 'TRAIN_projection.jsonl').open('w', encoding='utf-8') as projection:
        for index, (row, meta) in enumerate(zip_longest(rows(paths['TR_TRAIN']), rows(paths['TR_exact_frame_target_metadata']))):
            if row is None or meta is None or row['source_id'] != meta['source_id']:
                raise ValueError('Exact ordered TRAIN frame metadata differs')
            if row['language'] != 'tr' or row['dataset_split'] not in {'train', 'validated'}:
                raise ValueError('Candidate measurements are limited to accepted TRAIN roles')
            text = row['text']
            old, new = baseline.encode(text, add_special_tokens=False), updated.encode(text, add_special_tokens=False)
            old_text, new_text = baseline.decode(old.ids, skip_special_tokens=True), updated.decode(new.ids, skip_special_tokens=True)
            label_hash_mismatches += hashlib.sha256(json.dumps(old.ids, sort_keys=True).encode()).hexdigest() != meta['label_ids_sha256']
            is_heldout = int.from_bytes(hashlib.sha256((str(SEED) + '|' + native.normalize(text)).encode()).digest()[:8], 'big') % 10 == 0
            name = 'heldout' if is_heldout else 'learning'
            digests[name].update((row['source_id'] + '\n').encode()); counts[name] += 1
            T = meta['encoder_frames']
            tr.add(row, old, new, old_text, new_text, T, normalize_text)
            (heldout if is_heldout else learning).add(row, old, new, old_text, new_text, T, normalize_text)
            if is_heldout:
                native_old_U.append(len(native.encode(text))); native_new_U.append(len(native_candidate.encode(text)))
            projection.write(json.dumps({'source_id': row['source_id'], 'partition': name,
                                         'encoder_frames': T, 'baseline_HF_U': len(old.ids),
                                         'candidate_HF_U': len(new.ids)}, ensure_ascii=False) + '\n')
            if (index + 1) % 10000 == 0:
                print(json.dumps({'phase': 'TRAIN_only_HF_measurement', 'clips': index + 1}), flush=True)
    for name, key in [('learning', 'learning_source_ID_order_sha256'), ('heldout', 'heldout_TRAIN_source_ID_order_sha256')]:
        if digests[name].hexdigest() != partition[key]:
            raise ValueError('TRAIN normalized-text partition differs from original learning proof')
    replay = {lang: Measurements() for lang in ['en', 'es', 'it', 'fr', 'de']}
    for row in rows(paths['replay_TRAIN']):
        if row['language'] not in replay or row['dataset_split'] != 'train':
            raise ValueError('Replay measurements must use genuine TRAIN only')
        old, new = baseline.encode(row['text'], add_special_tokens=False), updated.encode(row['text'], add_special_tokens=False)
        # Existing replay duration metadata only; conservative arithmetic, no fresh WAV read.
        T = math.ceil(float(row['duration']) * 100 / 8)
        replay[row['language']].add(row, old, new, baseline.decode(old.ids, skip_special_tokens=True),
                                    updated.decode(new.ids, skip_special_tokens=True), T, normalize_text)
    if label_hash_mismatches:
        raise ValueError('Rust baseline does not reproduce actual HF TRAIN label IDs')
    report = {'schema': 1, 'kind': 'inert_HF_Turkish_BPE_512_candidate_v1',
              'CPU_only': True, 'active_assets_modified': False, 'candidate_activated': False,
              'fit_performed': False, 'DEV_or_TEST_text_read': False,
              'libraries': {'tokenizers': tokenizers.__version__, 'sentencepiece': spm.__version__},
              'below_normal_priority_applied': priority, 'TOKENIZERS_PARALLELISM': 'false',
              'Torch_Transformers_NeMo_imported': any(x == y or x.startswith(y + '.') for x in sys.modules for y in ['torch', 'transformers', 'nemo']),
              'inputs': inputs, 'script_sha256': sha(__file__), 'HF_baseline_semantic_sha256': semantic,
              'candidate_files_sha256': {str(path): sha(path) for path in DEST.iterdir() if path.is_file()},
              'prefix_identity': {'original8192_HF_IDs_unchanged': True, 'blank8192_unchanged': True,
                                  'six_existing_chars': {char: base['model']['vocab'][char] for char in 'ğĞıİşŞ'},
                                  'six_chars_in_AddedTokens': [x for x in base['added_tokens'] if x['content'] in 'ğĞıİşŞ'],
                                  'all8199_existing_BPE_core_IDs_unchanged': True,
                                  'existing13476_merges_order_exact_prefix': True,
                                  'normalizer_pretokenizer_decoder_special_tokens_unchanged': True},
              'merge_translation': {'method': 'Append each explicit left/right TRAIN merge pair in existing learned order after all original HF merge ranks.',
                                    'new_pieces': 512, 'native_id_to_HF_id_offset': 1,
                                    'new_HF_ids': [8199, 8711], 'SP_and_HF_encoding_not_assumed_equal': True},
              'learning_partition': {**partition, 'actual_learning_clips': counts['learning'],
                                     'actual_heldout_TRAIN_clips': counts['heldout'], 'source_order_hashes_recomputed_equal': True},
              'actual_HF_baseline_TRAIN_label_hash_mismatches': label_hash_mismatches,
              'Turkish_all_TRAIN': tr.result(), 'Turkish_learning_partition': learning.result(),
              'Turkish_text_hash_heldout_TRAIN': heldout.result(),
              'native_SP_same_heldout_comparison': {'baseline_U': summary(native_old_U), 'candidate_U': summary(native_new_U),
                                                    'reduction_fraction': 1 - sum(native_new_U) / sum(native_old_U)},
              'old_language_TRAIN': {lang: value.result() for lang, value in replay.items()},
              'row_migration_plan': migration,
              'passed_normalized_roundtrip': tr.counts['candidate_normalized_roundtrip'] == len(tr.old)
                  and all(value.counts['candidate_normalized_roundtrip'] == len(value.old) for value in replay.values()),
              'limitations': ['Preserving old IDs does not preserve old-language encoding sequences or probabilities; changed TRAIN encodings are measured per language.',
                              'BPE translation was measured in Rusttokenizers, not through Transformers or model inference.',
                              'TR T uses prior actual PCM frame proof; replay T is conservative duration arithmetic. No PCM/header rereads.',
                              'FP32 joint arithmetic excludes weights, hidden activations, native contiguous copies/gradients, checkpoint graphs, allocator and workspaces; not actual VRAM or speed proof.',
                              'Added512 logits widen softmax even with unchanged old rows; inference quality/retention and GPU memory remain unmeasured.',
                              'Future use requires a new target/cache namespace, tokenizer/model config changes and all5 retention DEV gates; no activation performed.'],
              'elapsed_CPU_process_seconds': time.perf_counter() - started}
    write(REPORT, report)
    print(json.dumps({'report': str(REPORT), 'candidate_tokenizer': str(DEST / 'tokenizer.json'),
                      'TRAIN_rows': len(tr.old), 'heldout_TRAIN_rows': len(heldout.old),
                      'HF_TRAIN_reduction': report['Turkish_all_TRAIN']['token_count_reduction_fraction'],
                      'HF_heldout_reduction': report['Turkish_text_hash_heldout_TRAIN']['token_count_reduction_fraction'],
                      'normalized_roundtrip_passed': report['passed_normalized_roundtrip']}), flush=True)


if __name__ == '__main__':
    main()
