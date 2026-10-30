"""Restore, merge and benchmark one development-selected encoder LoRA model.

The trainer never evaluates held-out test data. This separate step rebuilds the
selected model from its pinned base, checks the complete trainable state, merges
in FP32, saves and reloads an ordinary HF candidate, and rechecks development
gates before publishing a checkpoint or running the final held-out benchmark.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark import LANGUAGES, evaluate, read_jsonl, write_json
from encoder_lora import encoder_lora_metadata, install_encoder_lora, merge_encoder_lora
from extend_vocab import extend_processor_and_model
from metrics import paired_bootstrap_delta
from runtime import DATA, OUTPUT, ROOT, load_model


BASE_REVISION = '541d1f99c6b0c3cd0b11a95167540bb8edefd82b'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def tokenizer_sha256(processor) -> str:
    return sha256_json(json.loads(processor.tokenizer._tokenizer.to_str()))


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise RuntimeError(f'Path leaves the intended workspace directory: {path}')
    return resolved


def hf_file_manifest(path: Path) -> dict:
    """Hash the complete HF export; the separate provenance marker is excluded."""
    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob('*'))
        if file.is_file() and file.name != 'finalization_provenance.json'
    }


def load_selected_trainables(model, state: dict, *, run_id: str, selected_step: int,
                            layout: dict) -> dict:
    """Reject missing/extra trainables, incompatible layout, dtype or NaNs."""
    if state.get('run_id') != run_id or state.get('step') != selected_step:
        raise RuntimeError('Selected checkpoint run ID or step differs from selection.json')
    if state.get('lora_metadata') != layout:
        raise RuntimeError('Selected checkpoint LoRA layout differs from the reconstructed layout')
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    weights = state.get('weights')
    if not isinstance(weights, dict) or set(weights) != set(expected):
        provided = set(weights) if isinstance(weights, dict) else set()
        raise RuntimeError('Selected trainable key set is incomplete or incompatible: '
                           f'missing={sorted(set(expected)-provided)}, extra={sorted(provided-set(expected))}')
    for name, parameter in expected.items():
        tensor = weights[name]
        if not torch.is_tensor(tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise RuntimeError(f'Selected parameter shape/dtype differs: {name}')
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f'Selected parameter contains nonfinite values: {name}')
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys or set(result.missing_keys) & set(expected):
        raise RuntimeError('Selected trainables were not fully loaded')
    return {
        'trainable_key_count': len(expected),
        'trainable_parameter_count': sum(parameter.numel() for parameter in expected.values()),
        'trainable_keys_sha256': sha256_json(sorted(expected)),
        'all_trainable_keys_loaded': True,
        'frozen_state_loaded_from_pinned_base': True,
    }


def development_gate(development: dict, baseline: dict) -> dict:
    deltas = {}
    tolerances = {}
    for language in LANGUAGES:
        key = f'validation_{language}'
        if key not in development or key not in baseline:
            raise RuntimeError(f'Incomplete development summary: {key}')
        current = development[key]['asr']
        original = baseline[key]['asr']
        if current['utterances'] != original['utterances']:
            raise RuntimeError(f'Incomplete merged development predictions: {key}')
        if not all(torch.isfinite(torch.tensor(float(result['wer']))) for result in (current, original)):
            raise RuntimeError(f'Nonfinite development WER: {key}')
        if language != 'tr':
            deltas[language] = current['wer'] - original['wer']
            tolerances[language] = max(.005, .1 * original['wer'])
    retention = all(deltas[language] <= tolerances[language] for language in deltas)
    turkish = development['validation_tr']['asr']['wer']
    original_turkish = baseline['validation_tr']['asr']['wer']
    return {
        'passed': retention and turkish < original_turkish,
        'retention_gate_passed': retention,
        'turkish_improved': turkish < original_turkish,
        'turkish_wer': turkish,
        'baseline_turkish_wer': original_turkish,
        'retention_wer_deltas': deltas,
        'retention_wer_tolerances': tolerances,
        'policy': 'each dev WER <= base + max(0.5 percentage point,10% relative base WER); Turkish WER improves',
    }


def validate_baseline(baseline: dict) -> None:
    for language in LANGUAGES:
        for split in ('validation', 'test'):
            key = f'{split}_{language}'
            expected = read_jsonl(DATA/f'{key}.jsonl')
            predictions = read_jsonl(OUTPUT/f'results/base_v3/{key}.jsonl')
            identities = [row['source_id'] for row in predictions]
            if len(set(identities)) != len(identities) or set(identities) != {row['source_id'] for row in expected}:
                raise RuntimeError(f'Baseline does not cover the complete fixed manifest: {key}')
            if key not in baseline or baseline[key]['asr']['utterances'] != len(expected):
                raise RuntimeError(f'Incomplete baseline summary: {key}')
            references = {row['source_id']: row['text'] for row in expected}
            if any(row['reference'] != references[row['source_id']] for row in predictions):
                raise RuntimeError(f'Baseline references differ from the fixed manifest: {key}')


def publish_checkpoint(candidate: Path, destination: Path, manifest: dict) -> dict:
    """Copy a complete clean export, preserving a previous recommendation."""
    ensure_inside(candidate, ROOT/'work/models')
    ensure_inside(destination, OUTPUT)
    if destination.is_symlink():
        raise RuntimeError('Recommended checkpoint path must not be a symbolic link')
    if destination.exists() and not destination.is_dir():
        raise RuntimeError('Recommended checkpoint path is not a directory')
    if destination.exists() and hf_file_manifest(destination) == manifest:
        shutil.copy2(candidate/'finalization_provenance.json', destination/'finalization_provenance.json')
        return {'checkpoint': str(destination), 'reused_identical_export': True, 'previous_checkpoint_backup': None}
    stage = OUTPUT/f'checkpoint.stage_{uuid.uuid4().hex[:12]}'
    ensure_inside(stage, OUTPUT)
    shutil.copytree(candidate, stage)
    if hf_file_manifest(stage) != manifest:
        raise RuntimeError(f'Copied checkpoint differs from the verified candidate: {stage}')
    backup = None
    if destination.exists():
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup = OUTPUT/f'checkpoint.previous_{timestamp}_{uuid.uuid4().hex[:8]}'
        # Check absolute source and destination before moving the directory.
        ensure_inside(destination, OUTPUT)
        ensure_inside(backup, OUTPUT)
        destination.rename(backup)
    ensure_inside(stage, OUTPUT)
    ensure_inside(destination, OUTPUT)
    stage.rename(destination)
    return {'checkpoint': str(destination), 'reused_identical_export': False,
            'previous_checkpoint_backup': str(backup) if backup else None}


def seed_validation_predictions(source: Path, destination: Path) -> None:
    """Reuse the exact reloaded merged dev predictions for final calibration."""
    source_metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination/'metadata.json'
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding='utf-8'))
        if existing.get('model_fingerprint') != source_metadata['model_fingerprint']:
            raise RuntimeError('Existing final results belong to a different model')
    else:
        if any(destination.glob('*.jsonl')) or (destination/'summary.json').exists():
            raise RuntimeError('Unverified final prediction files already exist')
        write_json(metadata_path, copy.deepcopy(source_metadata))
    for language in LANGUAGES:
        source_path = source/f'validation_{language}.jsonl'
        target = destination/source_path.name
        if not target.exists():
            shutil.copy2(source_path, target)
        elif sha256_file(target) != sha256_file(source_path):
            raise RuntimeError('Existing final development predictions differ from the merged development pass')


def run(args) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', args.run_id):
        raise ValueError('run-id must be a short filename-safe identifier')
    run_dir = OUTPUT/'runs'/args.run_id
    config_path = run_dir/'run_config.json'
    configuration = json.loads(config_path.read_text(encoding='utf-8'))
    training_selection_path = run_dir/'training_selection.json'
    selection_path = training_selection_path if training_selection_path.exists() else run_dir/'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if not selection.get('accepted') or selection.get('selected_step') is None:
        raise RuntimeError('The run has no development-selected checkpoint to finalize')
    if selection.get('run_id') != args.run_id or configuration.get('run_id') != args.run_id:
        raise RuntimeError('Run configuration and selection IDs disagree')
    if configuration.get('base_revision') != BASE_REVISION:
        raise RuntimeError('This finalizer requires the pinned v3 base revision')
    for filename in ('encoder_lora.py', 'extend_vocab.py'):
        if configuration['code_sha256'][filename] != sha256_file(OUTPUT/filename):
            raise RuntimeError(f'Reconstruction code differs from the training run: {filename}')
    best_path = Path(selection['weights'])
    if not best_path.exists():
        raise RuntimeError(f'Selected checkpoint is absent: {best_path}')
    best_sha256 = sha256_file(best_path)
    selected_step = int(selection['selected_step'])
    baseline = json.loads((OUTPUT/'results/base_v3/summary.json').read_text(encoding='utf-8'))
    validate_baseline(baseline)
    base_path = Path((ROOT/'work/base_model_path.txt').read_text(encoding='utf-8').strip())
    if base_path.name != BASE_REVISION:
        raise RuntimeError('The configured local base path does not identify the pinned snapshot')
    base_manifest = hf_file_manifest(base_path)
    state = torch.load(best_path, map_location='cpu', weights_only=True)
    model, processor = load_model(base_path)
    extension = extend_processor_and_model(processor, model)
    trained_extension = json.loads((run_dir/'vocabulary_audit.json').read_text(encoding='utf-8'))
    if extension['extended_tokenizer_sha256'] != trained_extension['extended_tokenizer_sha256']:
        raise RuntimeError('Reconstructed tokenizer differs from the training tokenizer')
    layout = install_encoder_lora(model, rank=configuration['rank'], alpha=configuration['alpha'],
                                 last_n_layers=configuration.get('last_n_layers'))
    if layout != configuration['lora_layout']:
        raise RuntimeError('Reconstructed adapter layout differs from run_config.json')
    loaded = load_selected_trainables(model, state, run_id=args.run_id,
                                     selected_step=selected_step, layout=layout)
    if loaded['trainable_parameter_count'] != configuration['trainable_parameters']:
        raise RuntimeError('Reconstructed trainable parameter count differs from the trainer')
    model.config.encoder_lora_tracking = {
        'layout': layout, 'implementation_sha256': configuration['code_sha256']['encoder_lora.py'],
        'kind': 'merged', 'run_id': args.run_id, 'selected_step': selected_step,
    }
    model.eval()
    # Explicitly disable any surrounding autocast so adapter deltas merge FP32.
    with torch.autocast('cuda', enabled=False):
        merged_layout = merge_encoder_lora(model)
    if merged_layout != layout or encoder_lora_metadata(model)['adapter_parameter_count']:
        raise RuntimeError('Adapter merge was incomplete')
    candidate = ROOT/'work/models'/f'{args.run_id}_merged'
    marker_path = candidate/'finalization_provenance.json'
    if candidate.exists():
        previous = json.loads(marker_path.read_text(encoding='utf-8')) if marker_path.exists() else {}
        if previous.get('selected_state_sha256') != best_sha256:
            candidate = ROOT/'work/models'/f'{args.run_id}_step{selected_step}_{best_sha256[:12]}_merged'
    ensure_inside(candidate, ROOT/'work/models')
    candidate.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(candidate)
    processor.save_pretrained(candidate)
    candidate_manifest = hf_file_manifest(candidate)
    provenance = {
        'run_id': args.run_id, 'selected_step': selected_step,
        'base_revision': BASE_REVISION, 'base_path': str(base_path),
        'base_files_sha256': base_manifest,
        'run_config_sha256': sha256_file(config_path),
        'training_vocabulary_audit_sha256': sha256_file(run_dir/'vocabulary_audit.json'),
        'selected_state_sha256': best_sha256,
        'selected_state_path': str(best_path),
        'extended_tokenizer_sha256': extension['extended_tokenizer_sha256'],
        'adapter_layout': layout, 'adapter_layout_sha256': sha256_json(layout),
        'merged_fp32': True, 'reconstruction': loaded,
        'candidate': str(candidate), 'candidate_files_sha256': candidate_manifest,
        'evaluation_config_name_or_path': f'nvidia/parakeet-tdt-0.6b-v3@{BASE_REVISION}',
        'canonical_model_name_or_path': f'nvidia/parakeet-tdt-0.6b-v3@{BASE_REVISION}',
        'training_code_sha256': configuration['code_sha256'],
        'finalizer_sha256': sha256_file(Path(__file__)),
        'heldout_test_policy': 'One selected merged checkpoint after reloaded development gates; no test-driven tuning',
    }
    write_json(candidate/'finalization_provenance.json', provenance)
    # A fresh standard HF reload verifies export compatibility and is the actual
    # object used for all development and held-out predictions below.
    del model, processor, state
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_model(candidate)
    model.eval()
    if model.config.vocab_size != extension['extended_vocab_size'] or model.config.blank_token_id != extension['blank_token_id']:
        raise RuntimeError('Reloaded export token/blank configuration differs')
    if tokenizer_sha256(processor) != extension['extended_tokenizer_sha256']:
        raise RuntimeError('Reloaded export tokenizer differs')
    if model.config.encoder_lora_tracking['kind'] != 'merged':
        raise RuntimeError('Reloaded export lacks merged-adapter provenance')
    # HF reloads put the local directory into config._name_or_path, which the
    # existing benchmark hashes. Normalize this provenance-only field so a
    # candidate directory is recorded above but is not part of model identity.
    model.config.name_or_path = provenance['evaluation_config_name_or_path']
    merged_tag = f'{args.run_id}_merged_dev'
    development = evaluate(model, processor, merged_tag, splits=('validation',))
    gate = development_gate(development, baseline)
    write_json(run_dir/'merged_development_gate.json', gate)
    write_json(run_dir/'finalization_provenance.json', provenance)
    if not training_selection_path.exists():
        write_json(training_selection_path, selection)
    final_selection = {
        **selection, 'merged': True, 'accepted': gate['passed'],
        'candidate': str(candidate), 'merged_development_tag': merged_tag,
        'merged_development_gate': gate, 'heldout_tests_evaluated': False,
        'reason': 'Merged, saved and reloaded model passes Turkish improvement and every development retention gate.'
                  if gate['passed'] else 'Merged export failed a development gate; candidate retained for investigation.',
    }
    if not gate['passed']:
        write_json(run_dir/'selection.json', final_selection)
        write_json(OUTPUT/'selection.json', final_selection)
        print(json.dumps(final_selection), flush=True)
        return final_selection
    published = publish_checkpoint(candidate, OUTPUT/'checkpoint', candidate_manifest)
    trainables_path = run_dir/'selected_trainables.pt'
    if not trainables_path.exists() or sha256_file(trainables_path) != best_sha256:
        shutil.copy2(best_path, trainables_path)
    final_selection.update({**published, 'selected_trainables': str(trainables_path)})
    write_json(run_dir/'selection.json', final_selection)
    write_json(OUTPUT/'selection.json', final_selection)
    adapted_tag = f'{args.run_id}_adapted'
    # The merged development predictions are already from this reloaded model.
    # Reusing them avoids a second identical 300-clip pass while preserving the
    # full 900-clip final result layout and independent held-out test inference.
    seed_validation_predictions(OUTPUT/'results'/merged_tag, OUTPUT/'results'/adapted_tag)
    adapted = evaluate(model, processor, adapted_tag)
    comparisons = {}
    for language in LANGUAGES:
        before = read_jsonl(OUTPUT/f'results/base_v3/test_{language}.jsonl')
        after = read_jsonl(OUTPUT/f'results/{adapted_tag}/test_{language}.jsonl')
        after_by = {row['source_id']: row for row in after}
        if len(after_by) != len(after) or set(after_by) != {row['source_id'] for row in before}:
            raise RuntimeError(f'Paired final test source IDs differ: {language}')
        after = [after_by[row['source_id']] for row in before]
        if any(left['reference'] != right['reference'] or left.get('sentence_id') != right.get('sentence_id')
               for left, right in zip(before, after)):
            raise RuntimeError(f'Paired final references/sentence clusters differ: {language}')
        comparisons[language] = paired_bootstrap_delta(
            [row['reference'] for row in before], [row['hypothesis'] for row in before],
            [row['hypothesis'] for row in after], language=language,
            groups=[row.get('sentence_id') for row in before],
        )
    calibration = {language: adapted[f'calibration_{language}'] for language in LANGUAGES}
    results_path = {'tag': adapted_tag, 'summary': str(OUTPUT/f'results/{adapted_tag}/summary.json'),
                    'checkpoint': str(OUTPUT/'checkpoint')}
    for folder in (run_dir, OUTPUT):
        write_json(folder/'benchmark_comparison.json', comparisons)
        write_json(folder/'confidence_calibration.json', calibration)
        write_json(folder/'adapted_results_path.json', results_path)
    final_selection.update({'heldout_tests_evaluated': True, 'final_results_tag': adapted_tag})
    write_json(run_dir/'selection.json', final_selection)
    write_json(OUTPUT/'selection.json', final_selection)
    print(json.dumps(final_selection), flush=True)
    return final_selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='lora_20261010_01')
    run(parser.parse_args())
