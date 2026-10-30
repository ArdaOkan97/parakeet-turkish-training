"""Small balanced FLEURS TRAIN vocabulary-use measurement; no model/audio."""
from collections import Counter
from pathlib import Path
import ctypes
import datetime
import hashlib
import itertools
import json
import os
import sys
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['RAYON_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
REPORT = OUT / 'HF_shared_vocabulary_TRAIN_comparison_v1.json'
LANGS = ['en', 'es', 'it', 'fr', 'de', 'tr']
SEED = 20261011


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


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':')).encode('utf-8')).hexdigest()


def main():
    if REPORT.exists():
        raise FileExistsError('Preserve prior comparison')
    started = time.perf_counter()
    priority = False
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetCurrentProcess.restype = ctypes.c_void_p
        api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        priority = bool(api.SetPriorityClass(api.GetCurrentProcess(), 0x4000))
    sys.path.insert(0, str(OUT))
    import tokenizers
    from metrics import normalize_text
    paths = {
        'accepted_TR_manifest': ROOT / 'work/data/tr_public_v3_train.jsonl',
        'local_replay_TRAIN_manifest': ROOT / 'work/data/replay_train.jsonl',
        'current_six_character_HF_tokenizer': OUT / 'checkpoint/tokenizer.json',
        'inert_512_piece_HF_tokenizer': ROOT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json',
    }
    inputs = {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()}
    base_json = json.loads(paths['current_six_character_HF_tokenizer'].read_text(encoding='utf-8'))
    candidate_json = json.loads(paths['inert_512_piece_HF_tokenizer'].read_text(encoding='utf-8'))
    semantic = hashlib.sha256(json.dumps(base_json, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if semantic != 'b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75':
        raise ValueError('Required actual six-character HF baseline differs')
    if inputs['inert_512_piece_HF_tokenizer']['sha256'] != '6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327':
        raise ValueError('Frozen inactive candidate differs')
    base = tokenizers.Tokenizer.from_str(json.dumps(base_json, ensure_ascii=False))
    candidate = tokenizers.Tokenizer.from_str(json.dumps(candidate_json, ensure_ascii=False))
    pool = {lang: [] for lang in LANGS}
    raw_counts, accepted_counts = Counter(), Counter()
    raw_sentences = {lang: set() for lang in LANGS}
    for row in rows(paths['local_replay_TRAIN_manifest']):
        lang = row['language']
        if lang not in LANGS[:-1] or row['dataset'] != 'google/fleurs' or row['dataset_split'] != 'train':
            raise ValueError('Only actual FLEURS TRAIN replay is allowed')
        raw_counts[lang] += 1; raw_sentences[lang].add(row['sentence_id'])
        ids = base.encode(row['text'], add_special_tokens=False).ids
        if (1 <= len(ids) <= 180 and min(ids) >= 0 and max(ids) < 8193 and 8192 not in ids
                and normalize_text(base.decode(ids), lang) == normalize_text(row['text'], lang)):
            pool[lang].append(row); accepted_counts[lang] += 1
    for row in rows(paths['accepted_TR_manifest']):
        if row['dataset'] == 'google/fleurs':
            if row['language'] != 'tr' or row['dataset_split'] != 'train':
                raise ValueError('Only accepted Turkish FLEURS TRAIN allowed')
            raw_counts['tr'] += 1; accepted_counts['tr'] += 1
            raw_sentences['tr'].add(row['sentence_id']); pool['tr'].append(row)
    unique = {}
    accepted_sentences = {}
    for lang, values in pool.items():
        accepted_sentences[lang] = {row['sentence_id'] for row in values}
        chosen = {}
        # Stable speaker/source tie breaking; duplicate transcript contributes once.
        for row in sorted(values, key=lambda item: item['source_id']):
            chosen.setdefault(normalize_text(row['text'], lang), row)
        unique[lang] = list(chosen.values())
    common_raw = sorted(set.intersection(*raw_sentences.values()))
    common_accepted = sorted(set.intersection(*accepted_sentences.values()))
    N = min(300, *(len(values) for values in unique.values()))
    if N < 100:
        raise ValueError('Insufficient balanced unique TRAIN comparison')
    selected = {}
    for lang, values in unique.items():
        selected[lang] = sorted(values, key=lambda row: hashlib.sha256(
            (str(SEED) + '|' + lang + '|' + normalize_text(row['text'], lang)).encode()).hexdigest())[:N]
    variants = {}
    for variant, tokenizer in [('current_6_characters', base), ('inert_plus_512', candidate)]:
        pieces, result = {}, {}
        for lang in LANGS:
            original, six, added, occurrence = set(), Counter(), Counter(), Counter()
            token_sum = words = chars = changed = scored_roundtrip = 0
            lengths = []
            for row in selected[lang]:
                normalized = normalize_text(row['text'], lang)
                ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
                if normalize_text(tokenizer.decode(ids), lang) != normalized:
                    raise ValueError('Selected target has a scored roundtrip mismatch')
                scored_roundtrip += 1; token_sum += len(ids); lengths.append(len(ids))
                words += len(normalized.split()); chars += sum(not char.isspace() for char in normalized)
                original.update(identifier for identifier in ids if identifier < 8192)
                six.update(identifier for identifier in ids if 8193 <= identifier < 8199)
                added.update(identifier for identifier in ids if identifier >= 8199)
                occurrence.update(ids)
                changed += ids != base.encode(row['text'], add_special_tokens=False).ids
            pieces[lang] = original
            result[lang] = {
                'unique_original_8192_pieces_observed': len(original),
                'current_six_character_pieces_observed': len(six),
                'current_six_character_token_occurrences': sum(six.values()),
                'inert_512_pieces_observed': len(added),
                'inert_512_token_occurrences': sum(added.values()),
                'tokens': token_sum, 'scored_words': words, 'scored_nonspace_characters': chars,
                'tokens_per_scored_word': token_sum / words,
                'tokens_per_100_scored_nonspace_characters': 100 * token_sum / chars,
                'mean_target_tokens_per_sentence': token_sum / N,
                'max_target_tokens': max(lengths), 'scored_roundtrip_clips': scored_roundtrip,
                'encoding_changed_from_current_clips': changed,
                'original_piece_ID_set_sha256': digest(sorted(original)),
            }
        prevalence = Counter(identifier for values in pieces.values() for identifier in values)
        for lang in LANGS:
            result[lang]['original_pieces_shared_with_at_least_one_other_sample_language'] = sum(
                prevalence[identifier] >= 2 for identifier in pieces[lang])
            result[lang]['original_pieces_seen_only_in_this_language_sample_NOT_exclusive_ownership'] = sum(
                prevalence[identifier] == 1 for identifier in pieces[lang])
        shared = {f'{left}_{right}': len(pieces[left] & pieces[right])
                  for left, right in itertools.combinations(LANGS, 2)}
        variants[variant] = {
            'per_language': result, 'pairwise_original_piece_intersection_counts': shared,
            'original_piece_union_across_six_samples': len(prevalence),
            'original_pieces_observed_in_all_six_samples': len(set.intersection(*pieces.values())),
            'original_pieces_observed_in_two_or_more_samples': sum(count >= 2 for count in prevalence.values()),
            'original_piece_number_of_languages_histogram': dict(sorted(Counter(prevalence.values()).items())),
        }
    for lang in LANGS:
        old = variants['current_6_characters']['per_language'][lang]['tokens']
        new = variants['inert_plus_512']['per_language'][lang]['tokens']
        variants['inert_plus_512']['per_language'][lang]['token_count_reduction_fraction'] = 1 - new / old
    if any(name == prefix or name.startswith(prefix + '.') for name in sys.modules
           for prefix in ['torch', 'transformers', 'nemo', 'soundfile', 'torchaudio']):
        raise ValueError('Prohibited model/audio import')
    report = {
        'schema': 1, 'kind': 'HF_shared_vocabulary_FLEURS_TRAIN_usage_v1', 'passed': True,
        'CPU_only': True, 'active_training_or_assets_modified': False, 'candidate_activated': False,
        'DEV_or_TEST_read': False, 'audio_read': False, 'fit_performed': False,
        'official_model_card': {'url': 'https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3',
            'documented_shared_training_tokenizer_vocab': 8192, 'supported_languages': 25,
            'documented_statement_paraphrase': 'One unified SentencePiece vocabulary optimized across all25 supported languages, fitted to training transcripts.'},
        'actual_HF_layout': {'original_shared_pieces': [0, 8192], 'blank_id': 8192,
                            'original_teacher_vocab_size_including_blank': 8193,
                            'six_character_ids': {piece: base_json['model']['vocab'][piece] for piece in 'ğĞıİşŞ'},
                            'current_model_vocab_size': 8199, 'inert_candidate_vocab_size': 8711,
                            'inert_512_new_piece_ids': [8199, 8711]},
        'inputs': inputs, 'script_sha256': sha(__file__),
        'sampling': {'seed': SEED, 'raw_clips_by_language': dict(raw_counts),
                     'current_original_teacher_replay_accepted_clips_by_language': dict(accepted_counts),
                     'unique_normalized_TRAIN_texts_available': {lang: len(values) for lang, values in unique.items()},
                     'raw_common_sentence_IDs_all_six': common_raw,
                     'accepted_common_sentence_IDs_all_six': common_accepted,
                     'matched_translation_comparison_used': False,
                     'reason': 'Only a tiny local common sentence-ID intersection; equal-N unique TRAIN text samples provide more coverage.',
                     'unique_TRAIN_sentences_per_language': N,
                     'language_samples_have_unmatched_content': True,
                     'selected_sentence_ids': {lang: [row['sentence_id'] for row in values] for lang, values in selected.items()},
                     'selected_ordered_source_IDs_sha256': {lang: digest([row['source_id'] for row in values]) for lang, values in selected.items()},
                     'method': 'Deduplicate normalized TRAIN text per language, stable source-ID tie-break, select smallest SHA256(seed|language|normalized text).'},
        'measurement_definition': 'Encode original raw TRAIN labels; divide token totals by words/nonspace characters after the existing ASR scorer normalization. Raw punctuation/case tokens remain in the numerator.',
        'variants': variants,
        'interpretation_limits': [
            'Observed original token use is not an exclusive per-language vocabulary allocation; the same original pieces support multiple languages.',
            'Sample-only unique pieces may occur in other languages or texts not sampled, including the other19 supported languages.',
            'Sentence content, morphology and word length differ; token/word or character ratios describe fragmentation, not an exact fairness or ASR-quality estimate.',
            'The512 new TRAIN-derived pieces can also encode text in other languages; they are not Turkish-exclusive.',
            'No inference quality or speed result is measured here; the candidate remains inactive.'
        ],
        'runtime': {'TOKENIZERS_PARALLELISM': 'false', 'RAYON_NUM_THREADS': '2',
                    'tokenizers_version': tokenizers.__version__, 'below_normal_priority_applied': priority,
                    'elapsed_seconds': time.perf_counter() - started,
                    'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(REPORT), 'N': N, 'common_accepted_sentence_IDs': common_accepted,
                      'current': variants['current_6_characters'], 'candidate': variants['inert_plus_512'],
                      'elapsed_seconds': report['runtime']['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
