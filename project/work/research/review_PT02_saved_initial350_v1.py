"""Read-only streaming saved-DEV review; no model/checkpoint/DB access."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(HERE))
from metrics import score_transcriptions

RUN = 'adaptive_20261012_TR_uniform_full_PT_02'
FOLDER = HERE / 'results' / (RUN + '_step_6500')
PROOF = HERE / 'runs' / RUN / 'uniform_full_initial_source_equivalence.json'
OUTPUT = HERE / 'TSC_uniform_full_PT02_initial350_independent_saved_evidence_v1.json'
RECORDS = {}


def record(path, expected=None):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    value = {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': after.st_size}
    if expected is not None:
        assert value == {key: expected[key] for key in value}, 'Changed evidence: ' + str(path)
    RECORDS[str(path)] = value
    return value


def read(path, expected=None):
    record(path, expected)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(stream):
    for line in stream:
        if line.strip():
            yield json.loads(line)


def main():
    assert not OUTPUT.exists(), 'Fresh review output required'
    proof = read(PROOF)
    receipt = read(proof['source_receipt']['path'], proof['source_receipt'])
    artifacts = receipt['artifacts']
    assert proof['kind'] == 'TSC_uniform_full_before_update_source_equivalence_v1'
    assert proof['passed'] is True and proof['run_id'] == RUN
    assert proof['source_selected_step'] == receipt['selected_step'] == 6500
    assert proof['compared_fixed_DEV_utterances'] == 300 and proof['Portuguese_fixed_DEV_measured_utterances'] == 50
    for key in ('actual_GPU_decode', 'before_first_uniform_update', 'all_raw_transcripts_and_emitted_paths_exact',
                'all_six_ASR_counts_exact', 'Portuguese50_raw_paths_and_ASR_exact',
                'RNG_restored_after_added_evaluation', 'module_training_modes_restored_after_added_evaluation'):
        assert proof[key] is True, key
    old_summary = read(artifacts['DEV_summary']['path'], artifacts['DEV_summary'])
    new_summary = read(proof['candidate_summary']['path'], proof['candidate_summary'])
    old_metadata = read(artifacts['DEV_metadata']['path'], artifacts['DEV_metadata'])
    new_metadata = read(proof['candidate_metadata']['path'], proof['candidate_metadata'])
    assert old_metadata['model_fingerprint'] == new_metadata['model_fingerprint']
    assert old_metadata['manifests'] == new_metadata['manifests']
    languages = ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt')
    assert set(old_metadata['manifests']) == {'validation_' + lang for lang in languages[:-1]}
    pt_proof = read(receipt['portuguese_baseline_proof']['path'], receipt['portuguese_baseline_proof'])
    assert pt_proof['no_TEST_prediction_or_score_used'] is True
    source_pt = pt_proof['results']['PT_source6500_retention_v1']
    old_pt_metadata = read(source_pt['metadata']['path'], source_pt['metadata'])
    new_pt_metadata = read(FOLDER / 'metadata_pt.json')
    assert old_pt_metadata['model_fingerprint'] == new_pt_metadata['model_fingerprint']
    assert set(old_pt_metadata['manifests']) == set(new_pt_metadata['manifests']) == {'validation_pt'}
    a, b = old_pt_metadata['manifests']['validation_pt'], new_pt_metadata['manifests']['validation_pt']
    assert {k: v for k, v in a.items() if k != 'limit'} == {k: v for k, v in b.items() if k != 'limit'}
    assert a['limit'] == 50 and b['limit'] == 0
    assert not any(key.startswith('test_') for key in new_summary)
    assert set(key for key in new_summary if key.startswith('validation_')) == {'validation_' + lang for lang in languages}
    identity = ('source_id', 'sentence_id', 'language', 'split', 'reference', 'dataset_revision', 'audio_sha256', 'wav_sha256', 'duration')
    path_keys = ('token_id', 'duration_id', 'piece', 'start_frame', 'duration_frames')
    count_keys = ('word_errors', 'reference_words', 'character_errors', 'reference_characters')
    results = {}
    for language in languages:
        source = artifacts['DEV_' + language] if language != 'pt' else source_pt['predictions']
        source_record = record(source['path'], source)
        target = FOLDER / ('validation_' + language + '.jsonl')
        candidate_record = record(target)
        totals = {key: 0 for key in count_keys}
        count = tokens = 0
        seen = set()
        full_tokens_equal = True
        with Path(source['path']).open(encoding='utf-8') as before_stream, target.open(encoding='utf-8') as after_stream:
            for before, after in itertools.zip_longest(rows(before_stream), rows(after_stream)):
                assert before is not None and after is not None
                assert all(key in before and key in after and before[key] == after[key] for key in identity)
                assert before['language'] == language and before['split'] == after['split'] == 'validation'
                assert after['source_id'] not in seen
                seen.add(after['source_id'])
                assert before['hypothesis'] == after['hypothesis'] == after['confidence']['text']
                bt, at = before['confidence']['tokens'], after['confidence']['tokens']
                assert [tuple(token[key] for key in path_keys) for token in bt] == [tuple(token[key] for key in path_keys) for token in at]
                assert all(0 <= t['token_id'] < 8199 and t['token_id'] != 8192 and 0 <= t['duration_id'] <= 4
                           and t['duration_frames'] == t['duration_id'] for t in at)
                full_tokens_equal = full_tokens_equal and bt == at
                score = score_transcriptions((after,), language=language)
                for key in count_keys:
                    totals[key] += score[key]
                count += 1
                tokens += len(at)
        assert count == 50
        score = {**totals, 'utterances': count, 'wer': totals['word_errors'] / totals['reference_words'],
                 'cer': totals['character_errors'] / totals['reference_characters']}
        source_summary = old_summary if language != 'pt' else pt_proof['champion6500_PT']
        assert all(score[key] == source_summary['validation_' + language]['asr'][key]
                   == new_summary['validation_' + language]['asr'][key]
                   == proof['comparisons'][language]['asr'][key] for key in score)
        assert tokens == proof['comparisons'][language]['tokens_compared']
        assert proof['comparisons'][language]['exact_raw_transcripts'] == proof['comparisons'][language]['exact_token_duration_paths'] == 50
        results[language] = {**score, 'WER_percent': 100 * score['wer'], 'CER_percent': 100 * score['cer'],
            'exact_raw_transcripts': 50, 'exact_token_duration_paths': 50, 'emitted_tokens_compared': tokens,
            'full_saved_token_records_exact': full_tokens_equal,
            'prediction_files_byteidentical': source_record['sha256'] == candidate_record['sha256']}
    assert results['pt']['word_errors'] == 96 and results['pt']['reference_words'] == 1095
    assert pt_proof['Portuguese_gate']['allowed_word_errors'] == 66
    assert proof['all_five_original_retention_gates_passed'] is True and proof['all_six_retention_gates_passed'] is False
    assert 'torch' not in sys.modules and 'trackio' not in sys.modules
    report = {'schema': 1, 'kind': 'independent_actual_saved_PT02_initial350_source_equivalence_review_v1',
        'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_id': RUN, 'source_selected_step': 6500,
        'original_DEV_recordings_compared': 300, 'Portuguese_DEV_recordings_compared': 50,
        'source_original6_model_fingerprint_and_manifest_metadata_exact': True,
        'Portuguese_source_model_and_manifest_identity_exact': True,
        'Portuguese_metadata_limit_only_difference': {'source': 50, 'candidate': 0,
            'interpretation': 'Both decode the same complete fixed50 manifest; actual rows are paired and counted below.'},
        'all_350_raw_transcripts_and_token_duration_paths_exact': True,
        'all_350_full_saved_token_records_exact': all(v['full_saved_token_records_exact'] for v in results.values()),
        'all_350_prediction_files_byteidentical': all(v['prediction_files_byteidentical'] for v in results.values()),
        'all_seven_independent_corpus_ASR_rescores_match_saved_summaries_and_proof': True,
        'language_results': results, 'Portuguese_gate': pt_proof['Portuguese_gate'],
        'all_five_original_retention_passed': True, 'all_six_retention_passed': False,
        'initial_all_six_champion_claimed': False, 'DEV_only_saved_records_and_metadata_verified': True,
        'TEST_predictions_or_scores_read': False,
        'RNG_and_module_restoration': 'Producer runtime proof records successful assertions; no tensor/RNG state independently loaded by this metadata-only review.',
        'later_full_state_save_and_Adam_runtime_verified': False,
        'limitations': ['Saved JSON/JSONL streaming comparison and independent text rescore only; no model/checkpoint/audio/GPU activity.',
            'PT03 initialization or training progression is not certified by PT02 evidence.',
            'OS closure, training fit, full-state save and Adam restore remain ROOT-owned runtime evidence.'],
        'records': list(RECORDS.values()), 'maximum_decoded_prediction_rows_resident': 2, 'Torch_or_Trackio_imported': False}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'path': str(OUTPUT), 'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'TR_word_errors': results['tr']['word_errors'], 'PT_word_errors': results['pt']['word_errors'], 'PT_allowed': 66}))


if __name__ == '__main__':
    main()
