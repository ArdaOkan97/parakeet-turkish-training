"""INACTIVE ROOT-owned byte archive of a competitive CLOSED shared-head latest.

No checkpoint or ML bytes are read on import. The actual owned exit0, complete
runtime, positive localN and qualifying latest gates precede .pt hashing. SAME
run identity, original ready/source300 paths and separate BEST are preserved.
This writer does not validate tensors, launch training or infer OS closure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import shutil
import sys
import json

import record_TSC_BPE_phase2_exact_source_v1 as archive_ops

require, scoped, read, canonical = (archive_ops.require, archive_ops.scoped,
                                  archive_ops.read, archive_ops.canonical)
record, write_new, controller_kind = (archive_ops.record, archive_ops.write_new,
                                     archive_ops.controller_kind)
HERE, ROOT = archive_ops.HERE, archive_ops.ROOT
ARCHIVE_OPS_SHA = '696188f1e0652cad8489c7fc010583069d8e4bdcabf282fb66fe092489e422a7'
PHASE = 'HF_TSC_BPE512_new_rows_plus_shared_heads_v1'
FIRST_CONTROLLER = 'run_TSC_bpe_shared_head_memory_v2.py'
FIRST_CONTROLLER_SHA = 'c599a08848c3476da6e3c073ef7c0afd6fadd441cdf9498b99f449762395daed'
EXACT_CONTROLLER = 'run_TSC_bpe_shared_head_exact_v1.py'
RECEIPT_KIND = 'ROOT_owned_closed_TSC_BPE_shared_heads_exact_v1'
LANGUAGES = ('tr', 'en', 'es', 'it', 'fr', 'de')
RETENTION_MAX = {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70}
IMMUTABLE_TOKENS = ('original_native_ready', 'initial_source_equivalence')


def option(arguments, flag):
    require(type(arguments) is list and arguments.count(flag) == 1,
            'One original scientific worker argument required: ' + flag)
    index = arguments.index(flag)
    require(index+1 < len(arguments), 'Missing worker argument: ' + flag)
    return arguments[index+1]


def validate_closure_header(controller):
    """Pure metadata: reject live/failed/manual/smoke before checkpoint bytes."""
    require(controller.get('status') == 'train_window_ended'
        and controller.get('child_exit_code') == 0
        and controller.get('resource_review_required') is False,
        'Successful CLOSED own shared-head boundary required; no checkpoint bytes read')
    require(controller.get('stop_category') in ('budget', 'phase_bound')
        and controller.get('finished_at_utc') and controller.get('actual_runtime_completion'),
        'Only complete budget/phase-bound windows may continue; no failure/quality/manual retry')
    require(controller.get('smoke_updates') in (None, 0), 'Separate one-update smoke is not a scientific source')
    run_id = controller.get('run_id')
    require(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}', run_id),
            'Safe SAME scientific run ID required')
    return run_id


def validate_frontier(config, selection, history, controller):
    """JSON frontier only; tensor/Adam restore belongs to the exact worker."""
    run_id = controller['run_id']; identity = config['config_identity']; phase = identity['adaptation_phase']
    s, k, n = (phase.get('original_source_selected_step'), phase.get('selected_acquisition_updates'),
               selection.get('adaptation_updates_completed'))
    require(all(type(value) is int and value > 0 for value in (s, k, n)), 'Positive actual S/K/N required')
    step = s+k+n; selected = selection.get('selected_step')
    require(config.get('run_id') == selection.get('run_id') == run_id
        and config.get('phase') == identity.get('phase') == selection.get('phase') == PHASE
        and config.get('smoke_updates') is None and config.get('steps') == 44958
        and type(selected) is int and s+k <= selected <= step
        and selection.get('completed_steps') == controller.get('completed_steps') == step
        and controller.get('selected_step') == selected and controller.get('adaptation_updates_completed') == n
        and selection.get('row_Adam_clock') == k+n and selection.get('shared_Adam_clock') == s+n
        and selection.get('source_phase1_step') == phase.get('source_phase1_step') == s+k
        and selection.get('original_source_selected_step') == s and selection.get('selected_acquisition_updates') == k,
        'Keep latest ownN/rowK+N/sharedS+N and separate selectedBEST')
    require(phase.get('active_parameter_tensors') == 15 and phase.get('active_shared_head_tensors') == 12
        and phase.get('frozen_encoder_LoRA_tensors') == 192
        and phase.get('row_lr') in (1e-3, 3e-4) and phase.get('shared_lr') in (1e-5, 3e-6)
        and phase.get('further_unfreeze_allowed') is False
        and len(history) == step and all(row.get('step') == i for i, row in enumerate(history, 1))
        and all(row.get('adaptation_updates_completed') == row['step']-s-k
            and row.get('row_Adam_clock') == k+row['adaptation_updates_completed']
            and row.get('shared_Adam_clock') == s+row['adaptation_updates_completed']
            and row.get('row_lr') == phase['row_lr'] and row.get('shared_lr') == phase['shared_lr']
            and row.get('lora_lr') == 0. for row in history[s+k:]),
        'Immutable active15/frozen192 recipe and complete history/clocks required')
    return s, k, n, step, selected


def competitive_latest(summary, last, initial_wer):
    scores = {lang: summary['validation_'+lang]['asr'] for lang in LANGUAGES}
    require(type(initial_wer) in (int, float) and math.isfinite(initial_wer)
        and all(value.get('utterances') == 50 and type(value.get('word_errors')) is int
            and value.get('reference_words', 0) > 0 for value in scores.values())
        and all(scores[lang]['word_errors'] <= cap for lang, cap in RETENTION_MAX.items())
        and last.get('retention_gate_passed') is True
        and last.get('turkish_wer') == scores['tr']['wer'] < initial_wer,
        'Actual latest must pass original five gates and improve its startingchampion')
    return scores


def inspect_closed(source_loop):
    status_path = scoped(source_loop)/'loop_status.json'; controller = read(status_path)
    run_id = validate_closure_header(controller)
    require(record(archive_ops.__file__)['sha256'] == ARCHIVE_OPS_SHA, 'Borrowed immutable archive operations changed')
    executor = HERE/(FIRST_CONTROLLER if controller.get('controller_sha256') == FIRST_CONTROLLER_SHA else EXACT_CONTROLLER)
    require(executor.is_file() and record(executor)['sha256'] == controller.get('controller_sha256')
        and controller.get('kind') == controller_kind(executor), 'Actual CLOSED executor source/hash/kind differs')
    run = HERE/'runs'/run_id
    config, selection, history = (read(run/name) for name in ('run_config.json', 'selection.json', 'training_history.json'))
    s, k, n, step, selected = validate_frontier(config, selection, history, controller)
    paths = controller['paths']
    original_ready = scoped(option(controller['worker_arguments'], '--native-backend-provenance'))
    runtime_ready = scoped(paths['native']); completion_path = runtime_ready.with_suffix('.completion.json')
    ready, actual_ready, done, owned = (read(path) for path in
        (original_ready, runtime_ready, completion_path, scoped(paths['identity'])))
    require(ready.get('run_id') == actual_ready.get('run_id') == done.get('run_id') == run_id
        and ready.get('phase') == done.get('phase') == PHASE
        and ready.get('launcher_sha256') == FIRST_CONTROLLER_SHA
        and done.get('status') == 'complete' and done.get('sources_unchanged') is True and not done.get('error')
        and done.get('pid') == actual_ready.get('pid') == owned.get('pid')
        and done.get('actual_dispatch_source_sha256') == controller['controller_sha256']
        and done.get('backend_ready_sha256') == record(runtime_ready)['sha256'],
        'Current-window completion/freshPID and ORIGINAL native-ready must agree')
    runtime = {}
    for name, kind in (('memory', 'TSC_BPE_adaptation_memory_runtime_v1'),
                       ('cache', 'TSC_BPE_adaptation_cache_runtime_v1'), ('io', 'TSC_BPE_adaptation_atomic_IO_v1')):
        value = read(paths[name])
        require(value.get('kind') == kind and value.get('status') == 'complete'
            and value.get('sources_unchanged') is True and not value.get('error'), 'Actual runtime failed: '+name)
        runtime[name] = value
    start_n = controller.get('shared_updates_at_window_start', 0); memory = runtime['memory']
    require(type(start_n) is int and 0 <= start_n < n
        and runtime['cache']['cache_reclamation']['eligible_update_calls'] == n-start_n
        and memory.get('cap_bytes') == 8_000_000_000 and memory.get('cleanup_threshold_bytes') == 3_000_000_000
        and memory.get('allocator_cap_applied') is True
        and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True,
        'Actual suffix hook count/cap8/end3/telemetry closure required')
    proof_path = run/'TSC_BPE_shared_head_initial_source_equivalence.json'; proof = read(proof_path)
    require(proof.get('kind') == 'TSC_BPE512_shared_heads_before_update_source_equivalence_v1'
        and proof.get('run_id') == run_id and proof.get('source_selected_step') == s+k
        and proof.get('compared_fixed_DEV_utterances') == 300
        and all(proof.get(key) is True for key in ('passed', 'actual_GPU_decode', 'before_shared_head_updates',
            'all_raw_transcripts_and_emitted_paths_exact', 'all_six_ASR_counts_exact', 'all_five_retention_gates_passed',
            'RNG_restored_after_added_evaluation', 'module_training_modes_restored_after_added_evaluation'))
        and config['config_identity']['initial_source_equivalence_sha256'] == record(proof_path)['sha256'],
        'Keep original actual source300 scientific token; never repeat it')
    release_path = run/'source_admission_release.json'; release = read(release_path); compact = release.get('training_row_compaction', {})
    require(release.get('run_id') == run_id and release.get('passed') is True and compact.get('passed') is True
        and compact.get('rows_verified') == 253798 and compact.get('all_rows_canonical_equal') is True
        and compact.get('all_row_key_orders_equal') is True and compact.get('same_row_list_and_row_dict_objects') is True
        and compact.get('same_label_list_objects') is True, 'Actual complete original startup compaction proof required')
    latest_folder = HERE/'results'/f'{run_id}_step_{step}'
    latest_scores = competitive_latest(read(latest_folder/'summary.json'), history[-1].get('validation', {}),
        config['initial_adaptation_turkish_wer'])
    selected_folder = HERE/'results'/f'{run_id}_step_{selected}'
    selected_summary = read(selected_folder/'summary.json')
    selected_scores = {lang: selected_summary['validation_'+lang]['asr'] for lang in LANGUAGES}
    require(all(score.get('utterances') == 50 for score in selected_scores.values())
        and all(selected_scores[lang]['word_errors'] <= cap for lang, cap in RETENTION_MAX.items()), 'Separate selectedBEST actual all-five evidence required')
    sources = {'latest_checkpoint': scoped(selection['latest_weights']), 'best_checkpoint': scoped(selection['weights']),
        'run_config': run/'run_config.json', 'training_history': run/'training_history.json', 'selection': run/'selection.json',
        'controller_status': status_path, 'original_native_ready': original_ready, 'runtime_native_ready': runtime_ready,
        'initial_source_equivalence': proof_path, 'runtime_completion': completion_path,
        'fresh_runtime': scoped(paths['identity']), 'owned_worker_identity': scoped(paths['identity']),
        'startup_source_release': release_path, **{name: scoped(paths[name]) for name in ('memory', 'cache', 'io')}}
    for label, folder in (('selected', selected_folder), ('latest', latest_folder)):
        sources.update({label+'_DEV_summary': folder/'summary.json', label+'_DEV_metadata': folder/'metadata.json',
            **{label+'_DEV_'+lang: folder/f'validation_{lang}.jsonl' for lang in LANGUAGES}})
    require(sources['latest_checkpoint'] != sources['best_checkpoint'], 'Latest/BEST paths must be independently preserved')
    return {'controller': controller, 'config': config, 'selection': selection, 'history': history, 'executor': executor,
        'sources': sources, 'original_step': s, 'source_acquisition_updates': k, 'shared_head_updates': n,
        'completed_steps': step, 'selected_step': selected, 'latest_DEV_asr': latest_scores, 'selected_DEV_asr': selected_scores}


def make_receipt(args):
    if not args.validate_only:
        require(args.archive_dir and args.receipt_output, 'Fresh archive/receipt paths required')
        archive = scoped(args.archive_dir)
        require(archive.is_relative_to(ROOT/'work/closed-sources') and not archive.exists(), 'Fresh closed-source archive required')
        require(not scoped(args.receipt_output).exists(), 'Fresh receipt output required')
    context = inspect_closed(args.source_loop); controller, config = context['controller'], context['config']
    observed = datetime.fromisoformat(args.owned_exit_observed_at.replace('Z', '+00:00'))
    finished = datetime.fromisoformat(controller['finished_at_utc'].replace('Z', '+00:00'))
    require(type(args.owned_session_id) is int and args.owned_session_id > 0
        and re.fullmatch(r'[0-9a-f]+', args.owned_exit_chunk or '') and observed.tzinfo is not None
        and finished.tzinfo is not None and observed >= finished, 'ROOT actual owned exit0 must follow terminal publication')
    s, k, n = (context[key] for key in ('original_step', 'source_acquisition_updates', 'shared_head_updates'))
    total = args.approved_total_shared_head_updates; initial_bound = config['config_identity']['adaptation_phase']['effective_max_updates']
    require(type(total) is int and n < total <= 44958-s-k and type(initial_bound) is int and initial_bound > 0
        and (total <= initial_bound or args.allow_stopping_bound_extension), 'Explicit remaining totalN/extension approval required')
    # All actual closure/science/runtime/ROOT observation gates precede the first .pt byte.
    original = {name: record(path) for name, path in context['sources'].items()}
    from tsc_bpe_shared_head_exact_transition_v1 import validate_receipt
    receipt = {'schema': 1, 'kind': RECEIPT_KIND, 'status': 'actual_CLOSED_complete_shared_head_boundary',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_id': controller['run_id'], 'target_run_id': controller['run_id'],
        'allow_same_phase_exact_resume': True, 'guards_retained': True, 'further_unfreeze_allowed': False,
        'original_step': s, 'source_acquisition_updates': k, 'shared_head_updates': n,
        'completed_steps': context['completed_steps'], 'selected_step': context['selected_step'],
        'approved_total_shared_head_updates': total, 'allow_stopping_bound_extension': args.allow_stopping_bound_extension,
        'immutable_initial_shared_head_bound': initial_bound, 'chosen_checkpoint_role': 'latest',
        'chosen_step': context['completed_steps'], 'latest_DEV_step': context['completed_steps'],
        'competitive_qualifying_latest_approved': True, 'latest_DEV_asr': context['latest_DEV_asr'],
        'selected_DEV_asr': context['selected_DEV_asr'], 'existing_retention_max_word_errors': RETENTION_MAX,
        'scientific_config_identity_sha256': canonical(config['config_identity']), 'training_history_sha256': canonical(context['history']),
        'source_controller': record(context['executor']), 'source_stop_category': controller['stop_category'],
        'source_stop_reason': context['selection'].get('stop_early_reason'),
        'source_window_start_shared_head_updates': controller.get('shared_updates_at_window_start', 0),
        'artifacts': original, 'original_artifact_records': original,
        'original_artifact_paths': {key: value['path'] for key, value in original.items()},
        'owned_actual_exits': [{'role': 'trainer_controller', 'session_id': args.owned_session_id, 'exit_code': 0,
            'tool_chunk': args.owned_exit_chunk, 'observed_at_utc': observed.isoformat()}],
        'receipt_creator': record(__file__), 'archive_operations_source': record(archive_ops.__file__),
        'checkpoint_tensors_audited_here': False, 'actual_exact_utility_restore_and_tensor_frontier_validation_pending': True,
        'original_native_ready_and_initial300_paths_preserved': True, 'latest_own_frontier_and_selected_BEST_not_mixed': True,
        'torch_imported': 'torch' in sys.modules, 'actual_training_activation': False}
    validate_receipt(receipt, controller['run_id'])
    if args.validate_only:
        return {'kind': 'ROOT_closed_shared_head_exact_source_preflight_v1', 'passed': True, 'archive_created': False,
            'run_id': controller['run_id'], 'shared_head_updates': n, 'approved_total_shared_head_updates': total,
            'original_artifacts': original, 'checkpoint_tensors_audited': False}
    archive.mkdir(parents=True, exist_ok=False); archived, backups, copies = {}, {}, {}
    for name, source in context['sources'].items():
        if source in copies: bound = copies[source]
        else:
            destination = archive/(name+source.suffix); shutil.copy2(source, destination); bound = record(destination)
            require((bound['sha256'], bound['bytes']) == (original[name]['sha256'], original[name]['bytes'])
                and record(source) == original[name], 'Source/archive changed: '+name)
            copies[source] = bound
        if name in IMMUTABLE_TOKENS: archived[name], backups[name] = original[name], bound
        else: archived[name] = bound
    require(record(context['sources']['controller_status']) == original['controller_status'], 'Source terminal status changed during archive')
    for name in IMMUTABLE_TOKENS: require(record(context['sources'][name]) == original[name], 'Immutable scientific token changed: '+name)
    receipt.update(artifacts=archived, archived_immutable_scientific_tokens=backups,
        archive_copies_byte_identical=True, archive_directory=str(archive))
    validate_receipt(receipt, controller['run_id']); write_new(args.receipt_output, receipt)
    return {'receipt': record(args.receipt_output), 'run_id': controller['run_id'], 'shared_head_updates': n,
        'approved_total_shared_head_updates': total, 'archive_created': True, 'original_scientific_tokens_paths_preserved': True,
        'actual_exact_resume_launched': False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop', type=Path, required=True)
    p.add_argument('--approved-total-shared-head-updates', type=int, required=True)
    p.add_argument('--allow-stopping-bound-extension', action='store_true')
    p.add_argument('--owned-session-id', type=int, required=True)
    p.add_argument('--owned-exit-chunk', required=True)
    p.add_argument('--owned-exit-observed-at', required=True)
    p.add_argument('--archive-dir', type=Path); p.add_argument('--receipt-output', type=Path)
    p.add_argument('--validate-only', action='store_true'); p.add_argument('--validation-report', type=Path)
    return p


def main(args):
    try:
        result = make_receipt(args); result.update(passed=True, checkpoint_byte_reads=archive_ops.CHECKPOINT_BYTE_READS,
            torch_imported='torch' in sys.modules); code = 0
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as error:
        result = {'kind': 'ROOT_shared_head_exact_archive_rejection_v1', 'passed': False, 'error': str(error),
            'checkpoint_byte_reads': archive_ops.CHECKPOINT_BYTE_READS, 'torch_imported': 'torch' in sys.modules,
            'archive_or_receipt_admitted': False}; code = 2
    if args.validation_report: write_new(args.validation_report, result)
    print(json.dumps(result, ensure_ascii=False)); return code


if __name__ == '__main__':
    raise SystemExit(main(parser().parse_args()))
