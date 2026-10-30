"""Saved phase2 DEV only: rescoring, new-BPE emissions and JSON clock evidence.

Never opens checkpoints, tensors, audio or a model. All six saved fixed DEV
sets and a published phase2 history gate are required. History is streamed;
confidence arrays are discarded by the frozen phase1 Evidence.scan helper.
"""
from __future__ import annotations
import argparse
import codecs
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from analyze_TSC_bpe_acquisition_saved_DEV_v1 import (
    Evidence, HERE, ROOT, LANGUAGES, IDENTITY_KEYS, GEOMETRY,
    require, integer, canonical, rescore,
)
from metrics import normalize_text, edit_distance

PHASE = 'HF_TSC_BPE512_new_rows_plus_encoder_LoRA_v1'
PROOF_NAME = 'TSC_BPE_LoRA_initial_source_equivalence.json'
PROOF_KIND = 'TSC_BPE512_LoRA_before_update_source_equivalence_v1'
BASE_ANALYZER_SHA = 'd135829c6eae053b0bc10ba0bacb0d1f7b863c33b866c476ee41dbf5c564c716'
WORD_LIMITS = {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70}


def word_error_types(reference, hypothesis):
    """One optimal word alignment; diagonal/delete/insert tie order is explicit."""
    table = [list(range(len(hypothesis) + 1))]
    for i, token in enumerate(reference, 1):
        row = [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, table[-1][j] + 1,
                table[-1][j - 1] + (token != other)))
        table.append(row)
    i, j = len(reference), len(hypothesis)
    counts = {'substitutions': 0, 'deletions': 0, 'insertions': 0}
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            counts['substitutions'] += reference[i - 1] != hypothesis[j - 1]; i -= 1; j -= 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            counts['deletions'] += 1; i -= 1
        else:
            counts['insertions'] += 1; j -= 1
    require(sum(counts.values()) == table[-1][-1], 'Optimal alignment count differs')
    return counts


def recognition_diagnostic(before, after, language):
    before_total = Counter(); after_total = Counter(); examples = []
    for old, new in zip(before, after):
        ref = normalize_text(new['reference'], language)
        old_hyp = normalize_text(old['hypothesis'], language)
        new_hyp = normalize_text(new['hypothesis'], language)
        old_counts = word_error_types(ref.split(), old_hyp.split())
        new_counts = word_error_types(ref.split(), new_hyp.split())
        before_total.update(old_counts); after_total.update(new_counts)
        old_error, new_error = sum(old_counts.values()), sum(new_counts.values())
        examples.append({'source_id': new['source_id'], 'sentence_id': new['sentence_id'],
            'reference': new['reference'], 'source_hypothesis': old['hypothesis'],
            'requested_hypothesis': new['hypothesis'], 'source_word_errors': old_error,
            'requested_word_errors': new_error, 'word_error_delta': new_error - old_error,
            'source_character_errors': edit_distance(ref, old_hyp),
            'requested_character_errors': edit_distance(ref, new_hyp),
            'requested_error_types': new_counts, 'new_BPE_emissions': new['new_tokens']})
    return {'source_word_error_types': dict(before_total), 'requested_word_error_types': dict(after_total),
        'error_type_delta': {key: after_total[key] - before_total[key] for key in after_total},
        'optimal_alignment_ties': 'diagonal then deletion then insertion; category split may not be unique',
        'normalized_empty_hypotheses': sum(not normalize_text(row['hypothesis'], language) for row in after),
        'raw_empty_hypotheses': sum(not row['hypothesis'].strip() for row in after),
        'largest_regressions': sorted(examples, key=lambda row: row['word_error_delta'], reverse=True)[:5],
        'largest_requested_errors': sorted(examples, key=lambda row: row['requested_word_errors'], reverse=True)[:5]}


def phase2_history(evidence, value, source_step, target_step, original_step, acquired):
    """Retain only source/target/latest rows, enforcing every phase2 N/clock."""
    path = evidence.path(value)
    decoder = json.JSONDecoder(); utf8 = codecs.getincrementaldecoder('utf-8')()
    digest = hashlib.sha256(); buffer = ''; count = size = 0
    started = ended = False; need_value = True
    parent = target = last = None
    with path.open('rb') as stream:
        while True:
            raw = stream.read(65536); size += len(raw); digest.update(raw)
            buffer += utf8.decode(raw, final=not raw)
            require(len(buffer) <= 4_000_000, 'Oversized individual phase2 history record')
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer: break
                    require(buffer[0] == '[', 'History must be an array')
                    buffer = buffer[1:]; started = True; continue
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
                count += 1
                require(isinstance(row, dict) and row.get('step') == count, 'History step gap')
                if count > source_step:
                    n = count - source_step
                    require(row.get('source_phase1_step') == source_step
                        and row.get('original_source_selected_step') == original_step
                        and row.get('selected_acquisition_updates') == acquired
                        and row.get('adaptation_updates_completed') == n
                        and row.get('row_Adam_clock') == acquired + n
                        and row.get('LoRA_Adam_clock') == original_step + n,
                        'Phase2 JSON S/K/N and independent clocks are inconsistent')
                if count == source_step: parent = row
                if count == target_step: target = row
                last = row; buffer = buffer[stop:]; need_value = False
            if not raw: break
    require(started and ended and parent is not None and target is not None,
        'Published complete history has not reached the requested phase2 DEV')
    evidence.remember(path, digest.hexdigest(), size)
    return parent, target, last, count


def common_decode_identity(metadata):
    value = metadata['model_fingerprint']['identity']
    return {key: value[key] for key in ('generation_config_sha256', 'tokenizer_sha256',
        'processor_sha256', 'evaluation_code_sha256', 'inference_dtype')}


def analyze(args):
    require(hashlib.sha256((HERE / 'analyze_TSC_bpe_acquisition_saved_DEV_v1.py').read_bytes()).hexdigest()
        == BASE_ANALYZER_SHA, 'Frozen Evidence/scorer helper changed')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) is not None and integer(args.step, 1),
        'Explicit safe run and cumulative DEV step required')
    evidence = Evidence(); run = HERE / 'runs' / args.run_id
    config = evidence.read(run / 'run_config.json'); identity = config['config_identity']
    phase = identity['adaptation_phase']; optimizers = identity['adaptation_optimizers']
    s, k, source_step = (config[key] for key in ('original_source_selected_step',
        'selected_acquisition_updates', 'source_phase1_step'))
    require(integer(s, 1) and integer(k, 1) and source_step == s + k
        and (s, k, source_step) == (args.expected_original_step, args.expected_acquisition_updates,
            args.expected_source_step) and args.step > source_step
        and config['run_id'] == args.run_id and config['phase'] == identity['phase'] == phase['kind'] == PHASE
        and config['inherited_history_steps'] == source_step
        and phase['source_phase1_step'] == source_step and phase['original_source_selected_step'] == s
        and phase['selected_acquisition_updates'] == k and phase['active_parameter_tensors'] == 195
        and phase['row_lr'] == 1e-4 and phase['lora_lr'] == 3e-4
        and phase['old15_head_moments_active'] is False and phase['further_unfreeze_allowed'] is False
        and optimizers['active_parameter_tensors'] == 195 and optimizers['fixed_shared_head_tensors'] == 12
        and optimizers['source_selected_step'] == s and optimizers['selected_acquisition_updates'] == k
        and optimizers['mask_before_single_global_clip'] is True
        and all(identity['geometry'][key] == value for key, value in GEOMETRY.items()),
        'Actual phase2 source S/K,195/12 policy or candidate geometry differs')
    proof = evidence.read(run / PROOF_NAME)
    proof_record = evidence.records[str((run / PROOF_NAME).resolve())]
    require(proof['kind'] == PROOF_KIND and proof['run_id'] == args.run_id
        and proof['source_selected_step'] == source_step and proof['original_source_step'] == s
        and proof['selected_acquisition_updates'] == k and proof['compared_fixed_DEV_utterances'] == 300
        and all(proof.get(key) is True for key in ('passed', 'actual_GPU_decode', 'before_phase2_updates',
            'unsuppressed_new512_ids', 'all_raw_transcripts_and_emitted_paths_exact', 'all_six_ASR_counts_exact',
            'all_five_retention_gates_passed', 'RNG_restored_after_added_evaluation',
            'module_training_modes_restored_after_added_evaluation'))
        and config['initial_source_equivalence'] == proof
        and identity['initial_source_equivalence_sha256'] == proof_record['sha256'],
        'Hash-bound actual300 source equality/RNG/mode proof is missing')
    receipt = evidence.checked(proof['source_receipt'])
    require(receipt['kind'] == 'ROOT_owned_closed_TSC_BPE_phase1_for_LoRA_v1'
        and receipt['status'] == 'actual_CLOSED_qualified_phase1' and receipt['target_run_id'] == args.run_id
        and receipt['original_step'] == s and receipt['acquisition_updates'] == k
        and receipt['chosen_step'] == source_step and receipt['retention_gate_passed_at_chosen_step'] is True,
        'Actual chosen phase1 CLOSED point must be paired, not substituted by old BEST')
    source_summary = evidence.checked(receipt['artifacts']['DEV_summary'])
    source_metadata = evidence.checked(receipt['artifacts']['DEV_metadata'])
    initial = HERE / 'results' / (args.run_id + '_step_' + str(source_step))
    target = HERE / 'results' / (args.run_id + '_step_' + str(args.step))
    require(evidence.path(proof['candidate_summary']['path']) == initial / 'summary.json'
        and evidence.path(proof['candidate_metadata']['path']) == initial / 'metadata.json',
        'Initial300 result namespace is not the actual phase2 source point')
    initial_summary = evidence.checked(proof['candidate_summary'])
    initial_metadata = evidence.checked(proof['candidate_metadata'])
    target_summary = evidence.read(target / 'summary.json'); target_metadata = evidence.read(target / 'metadata.json')
    require(source_metadata['manifests'] == initial_metadata['manifests'] == target_metadata['manifests']
        and common_decode_identity(source_metadata) == common_decode_identity(initial_metadata)
            == common_decode_identity(target_metadata), 'Fixed DEV identity/tokenizer/decoding protocol changed')
    base = evidence.read(HERE / 'results/base_v3/summary.json')
    parent, current, newest, history_count = phase2_history(evidence, run / 'training_history.json',
        source_step, args.step, s, k)
    local = args.step - source_step
    require(current.get('development_selection_only') is True and 'validation' in current
        and 'selection_diagnostics' in current and current.get('row_lr') == 1e-4
        and current.get('head_lr') == 0. and current.get('lora_lr') == 3e-4
        and current.get('new_token_update_multiplier') == 1.
        and current.get('old_six_character_update_multiplier_applied') is False,
        'Requested point is not a completed own-phase2 DEV/update policy')
    findings = {}; hits = Counter(); old_hits = Counter(); pieces = {}
    for language in LANGUAGES:
        before, before_use = evidence.checked_scan(receipt['artifacts']['DEV_' + language], language, 8711)
        initial_rows, initial_use = evidence.scan(initial / ('validation_' + language + '.jsonl'), language, 8711)
        after, use = evidence.scan(target / ('validation_' + language + '.jsonl'), language, 8711)
        source_asr = rescore(before, source_summary, language)
        require(rescore(initial_rows, initial_summary, language) == source_asr
            and before_use == initial_use and proof['comparisons'][language]['asr'] == source_asr
            and proof['comparisons'][language]['exact_raw_transcripts'] == 50
            and proof['comparisons'][language]['exact_token_duration_paths'] == 50,
            'Saved initial300 source comparison does not reproduce')
        for a, b, c in zip(before, initial_rows, after):
            require(all(a[key] == b[key] == c[key] for key in IDENTITY_KEYS), 'Fixed DEV ID/reference/order changed')
            require(a['hypothesis'] == b['hypothesis'] and a['token_path_sha256'] == b['token_path_sha256']
                and a['blank_steps'] == b['blank_steps'] and a['total_steps'] == b['total_steps'],
                'Initial300 source transcript/token-duration path equality changed')
        require([r['source_id'] for r in after] == target_metadata['manifests']['validation_' + language]['source_ids'],
            'Saved predictions do not follow the declared fixed manifest')
        after_asr = rescore(after, target_summary, language)
        delta = {key: after_asr[key] - source_asr[key] for key in ('word_errors', 'character_errors')}
        delta.update(wer_percentage_points=100 * (after_asr['wer'] - source_asr['wer']),
            cer_percentage_points=100 * (after_asr['cer'] - source_asr['cer']))
        findings[language] = {'source_ASR': source_asr, 'requested_ASR': after_asr,
            'delta_from_chosen_phase1_source': delta,
            'raw_transcripts_changed': sum(a['hypothesis'] != b['hypothesis'] for a, b in zip(before, after)),
            'source_new_BPE_emission': before_use, 'new_BPE_emission': use}
        if language == 'tr':
            findings[language]['recognition_diagnostic'] = recognition_diagnostic(before, after, language)
            require(sum(findings[language]['recognition_diagnostic']['requested_word_error_types'].values())
                == after_asr['word_errors'], 'TR word-error decomposition differs from independent WER')
        for item in use['by_id']:
            token, piece = item['token_id'], item['piece']
            require(token not in pieces or pieces[token] == piece, 'Cross-language new ID/piece mismatch')
            pieces[token] = piece; hits[token] += item['emissions']
            if language != 'tr': old_hits[token] += item['emissions']
        if language != 'tr':
            baseline_wer = base['validation_' + language]['asr']['wer']
            allowed_delta = max(.005, .1 * baseline_wer); limit = baseline_wer + allowed_delta
            maximum = math.floor(limit * after_asr['reference_words'] + 1e-9)
            require(maximum == WORD_LIMITS[language], 'Original retention word limit changed')
            findings[language]['retention'] = {'passed': after_asr['wer'] <= limit,
                'WER_limit': limit, 'max_word_errors': maximum, 'word_error_margin': maximum - after_asr['word_errors']}
            require(math.isclose(current['validation']['retention_wer_deltas'][language],
                    after_asr['wer'] - baseline_wer, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(current['validation']['retention_wer_limits'][language], allowed_delta,
                    rel_tol=1e-12, abs_tol=1e-12), 'History retention delta/limit differs from rescoring')
    gate = current['validation']
    require(gate['turkish_wer'] == findings['tr']['requested_ASR']['wer']
        and gate['retention_gate_passed'] == all(findings[l]['retention']['passed'] for l in LANGUAGES[1:]),
        'Actual published history gate differs from independently rescored saved predictions')
    markers = {key: current[key] for key in ('step', 'adaptation_updates_completed', 'source_phase1_step',
        'original_source_selected_step', 'selected_acquisition_updates', 'row_Adam_clock', 'LoRA_Adam_clock',
        'row_lr', 'head_lr', 'lora_lr', 'new_token_update_multiplier', 'old_six_character_update_multiplier_applied',
        'turkish_epoch', 'turkish_clips_seen', 'turkish_unique_clips_seen', 'Turkish_domain_coverage',
        'replay_clips_seen', 'replay_language_clip_counts', 'selection_diagnostics') if key in current}
    result = {'schema': 1, 'kind': 'saved_TSC_BPE512_LoRA_DEV_evidence_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id, 'step': args.step,
        'original_source_selected_step': s, 'selected_acquisition_updates': k, 'source_phase1_step': source_step,
        'adaptation_updates_at_requested_DEV': local, 'JSON_row_Adam_clock': k + local,
        'JSON_LoRA_Adam_clock': s + local, 'saved_rows_independently_rescored': 300,
        'per_language': findings, 'TR_new_BPE_emitted': findings['tr']['new_BPE_emission'],
        'five_old_languages_new_BPE_emitted': {'emitted_count': sum(old_hits.values()),
            'unique_ids': sorted(old_hits), 'unique_id_count': len(old_hits),
            'utterance_count': sum(findings[l]['new_BPE_emission']['utterance_count'] for l in LANGUAGES[1:])},
        'all_six_new_BPE_emitted': {'emitted_count': sum(hits.values()), 'unique_ids': sorted(hits),
            'unique_id_count': len(hits), 'utterance_count': sum(findings[l]['new_BPE_emission']['utterance_count'] for l in LANGUAGES),
            'by_id': [{'token_id': token, 'piece': pieces[token], 'emissions': count} for token, count in sorted(hits.items())]},
        'copied_JSON_frontier': markers, 'history_retention_gate': gate,
        'declared_two_Adam_identity': optimizers, 'actual_Adam_moment_or_clock_tensor_audit_performed': False,
        'published_history_frontier_at_read': {'completed_steps': history_count,
            'adaptation_updates_completed': newest.get('adaptation_updates_completed')},
        'actual_initial300_proof_record': proof_record,
        'scope': {'saved_DEV_only': True, 'checkpoint_or_tensor_read': False, 'new_inference_or_training': False,
            'TEST_evaluated': False, 'Backlog_or_Trackio_mutation': False, 'OS_process_closure_inferred': False},
        'interpretation_limits': ['Saved emissions show decoder use, not correctness or a causal training explanation.',
            'JSON clocks/195 identity do not audit actual moments, protected weights, gradients or GPU stability.',
            'Published later history points show saved progress only; actual owned process status comes from ROOT.',
            'Fixed50 per language is DEV selection evidence, not final generalization or goal achievement.'],
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'Evidence_helper_sha256': BASE_ANALYZER_SHA,
        'scorer_sha256': hashlib.sha256((HERE / 'metrics.py').read_bytes()).hexdigest()}
    evidence.stable(); result['input_evidence'] = list(evidence.records.values())
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--step', required=True, type=int)
    p.add_argument('--expected-original-step', type=int, default=6500)
    p.add_argument('--expected-acquisition-updates', type=int, default=500)
    p.add_argument('--expected-source-step', type=int, default=7000)
    p.add_argument('--output', type=Path, required=True)
    return p


def main(args):
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and output.suffix == '.json' and not output.exists(),
        'Fresh experiment JSON output required; no existing report is overwritten')
    result = analyze(args)
    require(str(output) not in {r['path'] for r in result['input_evidence']}, 'Report path cannot be input evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'report': str(output), 'run_id': args.run_id, 'step': args.step,
        'TR_WER': result['per_language']['tr']['requested_ASR']['wer'],
        'TR_new_tokens': result['TR_new_BPE_emitted']['emitted_count'],
        'old_languages_new_tokens': result['five_old_languages_new_BPE_emitted']['emitted_count']}))


if __name__ == '__main__': main(parser().parse_args())
