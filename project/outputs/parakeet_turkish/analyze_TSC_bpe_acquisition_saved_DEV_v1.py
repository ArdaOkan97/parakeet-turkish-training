"""Read saved phase1 DEV evidence only; never load checkpoints or run inference.

Requires an explicit run/cumulative step, actual completed source300 proof,
six complete saved DEV sets, and a matching published acquisition history row.
Confidence arrays are discarded after counting emitted BPE IDs. Positive JSON
update counters are reported separately from emissions and are not Adam audits.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from metrics import score_transcriptions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PHASE = 'HF_TSC_BPE512_new_rows_only_acquisition_v1'
PROOF_NAME = 'TSC_BPE_acquisition_initial_source_equivalence.json'
PROOF_KIND = 'TSC_BPE512_acquisition_before_update_source_equivalence_v1'
LANGUAGES = ('tr', 'en', 'es', 'it', 'fr', 'de')
IDENTITY_KEYS = ('source_id', 'sentence_id', 'language', 'split', 'reference',
                 'dataset_revision', 'audio_sha256', 'wav_sha256', 'duration')
TOKEN_KEYS = ('token_id', 'duration_id', 'piece', 'start_frame', 'duration_frames')
GEOMETRY = {'student_vocab_size': 8711, 'joint_width': 8716, 'blank_id': 8192,
    'teacher_vocab_size': 8193, 'new_BPE_rows': [8199, 8711],
    'protected_six_rows': [8193, 8199], 'duration_rows': [8711, 8716]}


def require(ok, message):
    if not ok: raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Evidence:
    """Only small JSON/JSONL evidence; checkpoint extensions are never admitted."""
    def __init__(self): self.records = {}

    def path(self, value):
        path = Path(value).resolve()
        require((path.is_relative_to(HERE) or path.is_relative_to(ROOT / 'work/closed-sources'))
                and path.suffix in ('.json', '.jsonl'),
                'Only experiment JSON/JSONL evidence is allowed: ' + str(path))
        return path

    def remember(self, path, digest, size):
        value = {'path': str(path), 'sha256': digest, 'bytes': size}
        require(str(path) not in self.records or self.records[str(path)] == value,
                'Evidence changed during this read: ' + str(path))
        self.records[str(path)] = value
        return value

    def read(self, value, bound=16_000_000):
        path = self.path(value)
        require(path.stat().st_size <= bound, 'Oversized small evidence: ' + str(path))
        raw = path.read_bytes()
        self.remember(path, hashlib.sha256(raw).hexdigest(), len(raw))
        return json.loads(raw)

    def checked(self, record):
        path = self.path(record['path'])
        value = self.read(path)
        actual = self.records[str(path)]
        require(actual['sha256'] == record['sha256'] and
            ('bytes' not in record or actual['bytes'] == record['bytes']),
            'Hash-bound JSON evidence differs: ' + str(path))
        return value

    def scan(self, value, language, vocab):
        path = self.path(value); digest = hashlib.sha256(); size = 0
        rows, hits, pieces, utterances = [], Counter(), {}, []
        with path.open('rb') as stream:
            for raw in stream:
                size += len(raw); digest.update(raw)
                require(len(raw) <= 4_000_000 and len(rows) < 50, 'Oversized/incomplete fixed DEV schema')
                require(raw.strip(), 'Blank line in fixed DEV evidence')
                row = json.loads(raw)
                require(all(key in row for key in IDENTITY_KEYS) and row['language'] == language
                    and row['split'] == 'validation' and isinstance(row['hypothesis'], str),
                    'Prediction identity/language/split is invalid')
                confidence = row.pop('confidence')
                require(confidence['text'] == row['hypothesis'] and
                    integer(confidence['blank_steps']) and integer(confidence['total_steps']),
                    'Saved confidence text/step schema differs')
                per_row = Counter(); token_path = []
                for token in confidence['tokens']:
                    values = [token[key] for key in TOKEN_KEYS]; token_id, duration, piece, frame, frames = values
                    require(integer(token_id) and token_id < vocab and token_id != 8192
                        and integer(duration) and duration <= 4 and integer(frame)
                        and frames == duration and isinstance(piece, str), 'Unsafe saved emitted token/duration')
                    token_path.append(values)
                    if 8199 <= token_id < 8711:
                        require(token_id not in pieces or pieces[token_id] == piece, 'ID/piece mismatch')
                        pieces[token_id] = piece; per_row[token_id] += 1; hits[token_id] += 1
                if per_row:
                    utterances.append({'source_id': row['source_id'], 'sentence_id': row['sentence_id'],
                        'emitted_new_tokens': sum(per_row.values()), 'unique_new_ids': sorted(per_row)})
                rows.append({key: row[key] for key in IDENTITY_KEYS} | {
                    'hypothesis': row['hypothesis'], 'token_path_sha256': canonical(token_path),
                    'blank_steps': confidence['blank_steps'], 'total_steps': confidence['total_steps'],
                    'new_tokens': sum(per_row.values())})
                del row, confidence, token_path
        self.remember(path, digest.hexdigest(), size)
        require(len(rows) == 50 and len({r['source_id'] for r in rows}) == 50,
                'Exactly 50 unique saved fixed DEV rows required: ' + str(path))
        return rows, {'emitted_count': sum(hits.values()), 'unique_ids': sorted(hits),
            'unique_id_count': len(hits), 'utterance_count': len(utterances),
            'by_id': [{'token_id': token, 'piece': pieces[token], 'emissions': count}
                      for token, count in sorted(hits.items())], 'utterances': utterances}

    def checked_scan(self, record, language, vocab):
        result = self.scan(record['path'], language, vocab)
        actual = self.records[str(self.path(record['path']))]
        require(actual['sha256'] == record['sha256'] and
            ('bytes' not in record or actual['bytes'] == record['bytes']), 'Saved source predictions changed')
        return result

    def history(self, value, source_step, target_step):
        """Stream the JSON array; retain only parent/target rows, not old history."""
        path = self.path(value); decoder = json.JSONDecoder(); digest = hashlib.sha256()
        buffer = ''; count = size = 0; started = ended = False; need_value = True
        parent = target = last = None
        with path.open('rb') as stream:
            # Worker write_json emits UTF-8; an incremental decoder handles split codepoints.
            import codecs
            utf8 = codecs.getincrementaldecoder('utf-8')()
            while True:
                raw = stream.read(65536); size += len(raw); digest.update(raw)
                buffer += utf8.decode(raw, final=not raw)
                require(len(buffer) <= 4_000_000, 'Oversized individual history record')
                while True:
                    buffer = buffer.lstrip()
                    if not started:
                        if not buffer: break
                        require(buffer[0] == '[', 'History must be an array')
                        buffer = buffer[1:]; started = True
                        continue
                    if ended:
                        require(not buffer, 'Trailing history content'); break
                    if not buffer: break
                    if not need_value:
                        if buffer[0] == ',': buffer = buffer[1:]; need_value = True; continue
                        require(buffer[0] == ']', 'History delimiter missing')
                    if buffer[0] == ']':
                        require(not need_value or count == 0, 'Trailing history comma')
                        buffer = buffer[1:]; ended = True; continue
                    try: row, stop = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if raw: break
                        raise
                    count += 1; require(isinstance(row, dict) and row.get('step') == count, 'History step gap')
                    if count > source_step:
                        require(row.get('acquisition_updates_completed') == count - source_step,
                                'Acquisition local frontier is missing or inconsistent')
                    if count == source_step: parent = row
                    if count == target_step: target = row
                    last = row; buffer = buffer[stop:]; need_value = False
                if not raw: break
        require(started and ended and parent is not None and target is not None,
                'Published history has not reached the requested acquisition checkpoint')
        self.remember(path, digest.hexdigest(), size)
        return parent, target, last, count

    def stable(self):
        # Small evidence only; this does not open any path from opaque checkpoint records.
        for record in self.records.values():
            path = Path(record['path'])
            with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            require(digest == record['sha256'] and path.stat().st_size == record['bytes'],
                    'Evidence changed; retry after its atomic publication: ' + str(path))


def rescore(rows, summary, language):
    score = score_transcriptions(rows, language=language); score.pop('details')
    require(score == summary['validation_' + language]['asr'], 'Independent saved-row ASR counts differ')
    return score


def analyze(args):
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) is not None and integer(args.step, 1),
            'Explicit safe run ID and cumulative positive step required')
    evidence = Evidence(); run = HERE / 'runs' / args.run_id
    config = evidence.read(run / 'run_config.json'); identity = config['config_identity']
    phase = identity['acquisition_phase']; source_step = config['source_selected_step']
    require(config['run_id'] == args.run_id and config['phase'] == identity['phase'] == PHASE
        and integer(source_step, 1) and config['inherited_history_steps'] == source_step
        and phase['kind'] == PHASE and phase['source_selected_step'] == source_step
        and phase['active_optimizer_tables'] == 3 and phase['acquisition_lr'] == 1e-4
        and phase['source_Adam_retained_dormant'] is True and phase['unfreeze_allowed'] is False
        and all(identity['geometry'][key] == value for key, value in GEOMETRY.items())
        and args.step > source_step, 'Actual row-only phase geometry/source/frontier required')
    proof = evidence.read(run / PROOF_NAME)
    require(proof['kind'] == PROOF_KIND and proof['run_id'] == args.run_id
        and proof['source_selected_step'] == source_step and proof['compared_fixed_DEV_utterances'] == 300
        and all(proof.get(key) is True for key in ('passed', 'actual_GPU_decode', 'before_optimizer_creation',
            'unsuppressed_new512_ids', 'all_raw_transcripts_and_emitted_paths_exact', 'all_six_ASR_counts_exact',
            'all_five_retention_gates_passed', 'RNG_restored_after_added_evaluation',
            'module_training_modes_restored_after_added_evaluation'))
        and config['initial_source_equivalence'] == proof
        and identity['initial_source_equivalence_sha256'] == evidence.records[str((run / PROOF_NAME).resolve())]['sha256'],
        'Completed actual source300 proof is missing or inconsistent')
    receipt = evidence.checked(proof['source_receipt'])
    require(receipt['kind'] == 'ROOT_owned_closed_TSC_best_for_BPE512_acquisition_v1'
        and receipt['status'] == 'actual_CLOSED_qualified_best' and receipt['target_run_id'] == args.run_id
        and receipt['selected_step'] == source_step, 'Actual archived selected-source receipt required')
    source_summary = evidence.checked(receipt['artifacts']['DEV_summary'])
    source_metadata = evidence.checked(receipt['artifacts']['DEV_metadata'])
    initial = HERE / 'results' / (args.run_id + '_step_' + str(source_step))
    target = HERE / 'results' / (args.run_id + '_step_' + str(args.step))
    require(evidence.path(proof['candidate_summary']['path']) == initial / 'summary.json'
        and evidence.path(proof['candidate_metadata']['path']) == initial / 'metadata.json', 'Wrong initial result namespace')
    initial_summary = evidence.checked(proof['candidate_summary']); initial_metadata = evidence.checked(proof['candidate_metadata'])
    target_summary = evidence.read(target / 'summary.json'); target_metadata = evidence.read(target / 'metadata.json')
    require(source_metadata['manifests'] == initial_metadata['manifests'] == target_metadata['manifests'],
            'Fixed DEV manifest metadata differs')
    base = evidence.read(HERE / 'results/base_v3/summary.json')
    parent, current, newest, history_count = evidence.history(run / 'training_history.json', source_step, args.step)
    local = args.step - source_step
    require(current['development_selection_only'] is True and current['acquisition_lr'] == 1e-4
        and current['head_lr'] == current['lora_lr'] == 0. and current['new_token_update_multiplier'] == 1.
        and current['old_six_character_update_multiplier_applied'] is False,
        'Requested saved row lacks completed DEV/acquisition policy')
    findings = {}; total_hits = Counter(); old_hits = Counter()
    for language in LANGUAGES:
        before, before_use = evidence.checked_scan(receipt['artifacts']['DEV_' + language], language, 8199)
        initial_rows, initial_use = evidence.scan(initial / ('validation_' + language + '.jsonl'), language, 8711)
        after, use = evidence.scan(target / ('validation_' + language + '.jsonl'), language, 8711)
        source_asr = rescore(before, source_summary, language)
        require(rescore(initial_rows, initial_summary, language) == source_asr and
            initial_use['emitted_count'] == before_use['emitted_count'] == 0
            and proof['comparisons'][language]['asr'] == source_asr
            and proof['comparisons'][language]['exact_raw_transcripts'] == 50
            and proof['comparisons'][language]['exact_token_duration_paths'] == 50,
            'Initial source300 baseline differs')
        for a, b, c in zip(before, initial_rows, after):
            require(all(a[key] == b[key] == c[key] for key in IDENTITY_KEYS), 'Saved fixed DEV IDs/order/reference differ')
            require(a['hypothesis'] == b['hypothesis'] and a['token_path_sha256'] == b['token_path_sha256']
                and a['blank_steps'] == b['blank_steps'] and a['total_steps'] == b['total_steps'],
                'Saved source300 transcript/path equality no longer reproduces')
        require([r['source_id'] for r in after] == target_metadata['manifests']['validation_' + language]['source_ids'],
                'Predictions do not match saved manifest order')
        after_asr = rescore(after, target_summary, language)
        delta = {key: after_asr[key] - source_asr[key] for key in ('word_errors', 'character_errors')}
        delta.update(wer_percentage_points=100 * (after_asr['wer'] - source_asr['wer']),
            cer_percentage_points=100 * (after_asr['cer'] - source_asr['cer']),
            relative_word_error_reduction=(source_asr['word_errors'] - after_asr['word_errors']) / source_asr['word_errors']
                if source_asr['word_errors'] else None)
        changed = sum(a['hypothesis'] != b['hypothesis'] for a, b in zip(before, after))
        findings[language] = {'source_ASR': source_asr, 'requested_ASR': after_asr,
            'delta_from_actual_source300': delta, 'raw_transcripts_changed': changed, 'new_BPE_emission': use}
        for item in use['by_id']:
            total_hits[item['token_id']] += item['emissions']
            if language != 'tr': old_hits[item['token_id']] += item['emissions']
        if language != 'tr':
            baseline_wer = base['validation_' + language]['asr']['wer']
            limit = baseline_wer + max(.005, .1 * baseline_wer)
            maximum = math.floor(limit * after_asr['reference_words'] + 1e-9)
            findings[language]['retention'] = {'passed': after_asr['wer'] <= limit,
                'WER_limit': limit, 'max_word_errors': maximum,
                'word_error_margin': maximum - after_asr['word_errors']}
    gate = current['validation']
    require(gate['turkish_wer'] == findings['tr']['requested_ASR']['wer']
        and gate['retention_gate_passed'] == all(findings[l]['retention']['passed'] for l in LANGUAGES[1:]),
        'History DEV gate differs from saved independently scored predictions')
    markers = {key: current[key] for key in ('step', 'acquisition_updates_completed', 'acquisition_lr',
        'head_lr', 'lora_lr', 'new_token_update_multiplier', 'old_six_character_update_multiplier_applied',
        'turkish_epoch', 'turkish_clips_seen', 'turkish_unique_clips_seen', 'Turkish_domain_coverage',
        'replay_clips_seen', 'replay_language_clip_counts', 'selection_diagnostics') if key in current}
    result = {'kind': 'saved_TSC_BPE512_acquisition_DEV_evidence_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': args.run_id, 'step': args.step, 'source_selected_step': source_step,
        'acquisition_updates_at_requested_DEV': local, 'saved_rows_independently_rescored': 300,
        'per_language': findings, 'TR_new_BPE_emitted': findings['tr']['new_BPE_emission'],
        'five_old_languages_new_BPE_emitted': {'emitted_count': sum(old_hits.values()),
            'unique_ids': sorted(old_hits), 'unique_id_count': len(old_hits),
            'utterance_count': sum(findings[l]['new_BPE_emission']['utterance_count'] for l in LANGUAGES[1:])},
        'all_six_new_BPE_emitted': {'emitted_count': sum(total_hits.values()), 'unique_id_count': len(total_hits)},
        'copied_JSON_frontier': markers,
        'declared_three_table_Adam_identity': identity['acquisition_optimizer'],
        'declared_three_table_Adam_clock_from_JSON_update_counter': local,
        'per_table_actual_Adam_clocks_in_JSON': None,
        'published_history_frontier_at_read': {'completed_steps': history_count,
            'acquisition_updates_completed': newest.get('acquisition_updates_completed')},
        'scope': {'saved_DEV_only': True, 'checkpoint_or_tensor_read': False,
            'new_inference_or_training': False, 'TEST_evaluated': False,
            'confidence_probability_analysis': False, 'temperature_fit': False},
        'interpretation_limits': ['Emissions establish saved decoder use, not correct recognition or a training cause.',
            'Fresh update counts and optimizer identity are copied JSON evidence, not three moment/clock audits.',
            'No checkpoint is read; actual protected weights/deltas and GPU fit require separate smoke evidence.',
            'Fixed50 per language is an early DEV diagnostic, not final winner or generalization evidence.'],
        'scorer': {'path': str(HERE / 'metrics.py'), 'sha256': hashlib.sha256((HERE / 'metrics.py').read_bytes()).hexdigest()},
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    evidence.stable(); result['input_evidence'] = list(evidence.records.values())
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--step', required=True, type=int, help='Actual cumulative saved DEV step, source S plus local K')
    p.add_argument('--output', required=True, type=Path, help='Fresh JSON report under outputs/parakeet_turkish')
    return p


def main(args):
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and output.suffix == '.json' and not output.exists(),
            'Fresh experiment JSON output required; existing files are never overwritten')
    result = analyze(args)
    require(str(output) not in {r['path'] for r in result['input_evidence']}, 'Output must not be input evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'report': str(output), 'run_id': args.run_id, 'step': args.step,
        'TR_new_tokens': result['TR_new_BPE_emitted']['emitted_count'],
        'old_languages_new_tokens': result['five_old_languages_new_BPE_emitted']['emitted_count']}))


if __name__ == '__main__': main(parser().parse_args())
