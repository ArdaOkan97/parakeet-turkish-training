"""Matched effective-batch-16 correctness and solo-GPU training profiles.

All physical batch sizes see the same eight Turkish and eight replay TRAIN clips.
The original experiment files are imported, never modified. --plan-only uses the
CPU. Every GPU job is a separate invocation controlled by the parent experiment.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from runtime import ROOT, OUTPUT
from batched_lora import collate_training_batch, batch_objective
from efficient_tdt import chunked_tdt_loss as original_loss
from efficient_tdt_v2 import chunked_tdt_loss as v2_loss
from profile_batched_training import initialize, original_objective
from diagnose_batch_objective_math import compare_tensors, strict_gate

WORK = ROOT / 'work' / 'large_batches'
SOURCE_NAMES = ('profile_large_batches.py', 'profile_batched_training.py',
                'diagnose_batch_objective_math.py', 'batched_lora.py', 'efficient_tdt.py',
                'efficient_tdt_v2.py', 'runtime.py', 'encoder_lora.py', 'extend_vocab.py',
                'train_lora.py', 'train.py', 'train_lora_batched.py')
LANGUAGES = ('en', 'es', 'it', 'fr', 'de')
COMPONENTS = ('tdt', 'token_kl', 'duration_kl', 'feature_distillation')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    return {name: digest(OUTPUT / name) for name in SOURCE_NAMES}


def grouping(batch_size):
    if batch_size == 16:
        return [list(range(16))]
    if batch_size == 1:
        return [[index] for index in range(16)]
    tr, replay = list(range(0, 16, 2)), list(range(1, 16, 2))
    groups = []
    for start in range(0, 8, batch_size):
        groups.extend((tr[start:start + batch_size], replay[start:start + batch_size]))
    if sorted(index for group in groups for index in group) != list(range(16)):
        raise ValueError('Microbatch groups do not cover each clip exactly once.')
    return groups


def choose_rows(pool, case):
    """Deterministic T/U-near-median typical cases and maximum-lattice stress."""
    def ordered(rows):
        if case == 'long':
            return sorted(rows, key=lambda row: (-row['encoder_frames'] * (row['target_tokens'] + 1),
                                                 hashlib.sha256(row['source_id'].encode()).hexdigest()))
        mt = statistics.median(row['encoder_frames'] for row in rows)
        mu = statistics.median(row['target_tokens'] for row in rows)
        return sorted(rows, key=lambda row: (abs(math.log(row['encoder_frames'] / mt)) +
                                            abs(math.log(row['target_tokens'] / mu)),
                                            hashlib.sha256(row['source_id'].encode()).hexdigest()))
    tr = ordered(pool['turkish'])[:8]
    by_language = {language: ordered([row for row in pool['replay'] if row['language'] == language])
                   for language in LANGUAGES}
    replay, cursors = [], dict.fromkeys(LANGUAGES, 0)
    for language in (*LANGUAGES, 'en', 'es', 'it'):
        replay.append(by_language[language][cursors[language]])
        cursors[language] += 1
    rows = [row for pair in zip(tr, replay) for row in pair]
    if len(rows) != 16 or len({row['source_id'] for row in rows}) != 16:
        raise ValueError('Sixteen unique fixed TRAIN clips are required.')
    if any(row['dataset_split'] != 'train' for row in rows):
        raise ValueError('Large-batch profiles may use TRAIN only.')
    return rows


def inspect_rows(rows):
    items, specifications = [], []
    for row in rows:
        cache = Path(row['cache_path'])
        if cache.name != hashlib.sha256(row['source_id'].encode()).hexdigest() + '.pt':
            raise ValueError('Teacher cache source key differs.')
        info = sf.info(row['audio_filepath'])
        if info.samplerate != 16000 or info.channels != 1 or abs(info.frames / 16000 - row['duration']) > 1e-5:
            raise ValueError('WAV format or exact audio duration differs.')
        wav_sha = digest(row['audio_filepath'])
        if wav_sha != row['wav_sha256']:
            raise ValueError('Prepared WAV bytes differ from accepted training pool.')
        item = torch.load(cache, map_location='cpu', weights_only=True)
        if (item['source_id'] != row['source_id'] or item['features'].shape != (row['encoder_frames'], 1024)
                or item['labels'].shape != (row['target_tokens'],)
                or not torch.isfinite(item['features']).all()):
            raise ValueError('Cache metadata/finiteness differs.')
        items.append(item)
        specifications.append({**row, 'wav_sha256': wav_sha, 'cache_sha256': digest(cache),
                               'actual_pcm_seconds': info.frames / 16000,
                               'labels': item['labels'].tolist(),
                               'features_dtype': str(item['features'].dtype)})
    return items, specifications


def padding_from_rows(rows, groups):
    valid = sum(row['encoder_frames'] * (row['target_tokens'] + 1) for row in rows)
    padded = sum(len(group) * max(rows[i]['encoder_frames'] for i in group) *
                 (max(rows[i]['target_tokens'] for i in group) + 1) for group in groups)
    return {'valid_lattice_states': valid, 'padded_lattice_states': padded,
            'lattice_padding_factor': padded / valid,
            'encoder_padding_frames': sum(len(group) * max(rows[i]['encoder_frames'] for i in group) -
                                         sum(rows[i]['encoder_frames'] for i in group) for group in groups),
            'decoder_padding_tokens': sum(len(group) * max(rows[i]['target_tokens'] for i in group) -
                                         sum(rows[i]['target_tokens'] for i in group) for group in groups)}


def backend_flags():
    return {'tf32_matmul_allowed': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn_allowed': torch.backends.cudnn.allow_tf32,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'sdpa_flash_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'sdpa_math_enabled': torch.backends.cuda.math_sdp_enabled(),
            'sdpa_mem_efficient_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'sdpa_cudnn_enabled': torch.backends.cuda.cudnn_sdp_enabled()}


def gpu_process_evidence():
    command = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {'self_pid': os.getpid(), 'command': command, 'returncode': result.returncode,
                'stdout': result.stdout.strip(), 'stderr': result.stderr.strip(),
                'scope': 'Windows driver process inventory; parent coordinates the paused training window.'}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'self_pid': os.getpid(), 'inventory_error': str(error)}


def batch_invariants(batch, indices):
    feature = batch.inputs['input_features']
    mask = batch.inputs['attention_mask'].bool()
    lengths = mask.sum(-1).cpu().tolist()
    padded_input_zero = padded_cache_zero = padded_labels_blank = True
    for row_index, (mel, frames, targets) in enumerate(zip(lengths, batch.frame_lengths.tolist(), batch.target_lengths.tolist())):
        if not bool(mask[row_index, :mel].all()) or bool(mask[row_index, mel:].any()):
            raise ValueError('Frontend attention mask is not right padded.')
        padded_input_zero &= bool((feature[row_index, mel:] == 0).all())
        padded_cache_zero &= bool((batch.base_features[row_index, frames:] == 0).all())
        padded_labels_blank &= bool((batch.labels[row_index, targets:] == batch.blank_id).all())
    if not padded_input_zero or not padded_cache_zero or not padded_labels_blank:
        raise ValueError('Input/cache/target padding invariant failed.')
    return {'indices': indices, 'batch_size': batch.batch_size,
            'input_features_shape': list(feature.shape), 'attention_mask_shape': list(mask.shape),
            'input_mask_sums': lengths, 'right_padded_mask': True,
            'input_padding_zero': padded_input_zero, 'cache_padding_zero': padded_cache_zero,
            'target_padding_blank': padded_labels_blank,
            'encoder_frames': batch.frame_lengths.tolist(), 'target_tokens': batch.target_lengths.tolist(),
            'labels_shape': list(batch.labels.shape), 'decoder_input_ids_shape': list(batch.decoder_input_ids.shape),
            'replay_indices': list(batch.replay_indices),
            'input_padding_frames': int(mask.numel() - mask.sum()),
            'lattice_padding_factor': batch.lattice_padding_factor}


def register_matrix(args, report):
    path = args.matrix_output
    matrix = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'schema_version': 1, 'status': 'partial', 'effective_batch': 16, 'cases': []}
    entry = {'case_id': report['case_id'], 'family': args.case, 'mode': args.mode,
             'batch_size': args.batch_size, 'precision': report['precision'],
             'backend': args.backend, 'tdt': report['tdt_implementation'],
             'report_path': str(args.output), 'report_sha256': digest(args.output),
             'evidence_path': report.get('evidence_path'), 'evidence_sha256': report.get('evidence_sha256'),
             'status': report['status']}
    if any(row['case_id'] == entry['case_id'] for row in matrix['cases']):
        raise ValueError('Matrix case already exists; choose a fresh output/matrix rather than overwrite.')
    matrix['cases'].append(entry)
    matrix['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    matrix['notes'] = ['Actual measured cases only; an incomplete matrix remains partial.',
                       'Physical-batch timings do not authorize adoption in the current training loop.']
    write(path, matrix)


def run(args):
    hashes = source_hashes()
    pool_sha = digest(args.pool)
    pool = json.loads(args.pool.read_text(encoding='utf-8'))
    rows = choose_rows(pool, args.case)
    items, specs = inspect_rows(rows)
    groups = grouping(args.batch_size)
    case_id = args.output.stem
    precision = 'fp32' if args.mode.startswith('fp32') else 'bf16'
    tdt = 'original' if args.mode in ('fp32-reference', 'bf16-reference') else 'v2'
    padding = padding_from_rows(rows, groups)
    if args.plan_only:
        value = {'status': 'planned_not_run', 'gpu_jobs_started': 0, 'case_id': case_id,
                 'family': args.case, 'pool_sha256': pool_sha, 'source_sha256': hashes,
                 'effective_batch': 16, 'source_ids': [row['source_id'] for row in rows],
                 'source_clips': specs, 'groups': groups, 'padding': padding,
                 'audio_seconds_per_update': math.fsum(row['duration'] for row in rows),
                 'batch_size': args.batch_size, 'accumulation': len(groups)}
        write(args.output, value)
        print(json.dumps({'saved': str(args.output), 'status': value['status']}), flush=True)
        return 0
    if args.output.exists():
        raise ValueError('Output already exists; do not overwrite a measured case.')
    if args.gpu_context != 'solo':
        raise ValueError('Large-batch gradient/timing jobs require the parent-coordinated solo GPU window.')
    if not torch.cuda.is_available():
        raise RuntimeError('Local GPU unavailable.')
    if precision == 'fp32' and args.backend != 'math':
        raise ValueError('Strict FP32 gradient oracles require forced SDPA MATH.')
    if args.mode == 'fp32-reference' and args.batch_size != 1:
        raise ValueError('The FP32 reference is native original B1x16.')
    if args.mode == 'timing' and (args.warmup < (1 if args.case == 'long' else 2) or args.updates < (3 if args.case == 'long' else 10)):
        raise ValueError('Use >=2 warmups/10 updates for typical, >=1 warmup/3 updates for long stress.')
    WORK.mkdir(parents=True, exist_ok=True)
    evidence_path = WORK / (case_id + '.pt')
    if evidence_path.exists():
        raise ValueError('Gradient evidence already exists; use a fresh case ID.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(min(args.max_gpu_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory, .98))
    context = sdpa_kernel(SDPBackend.MATH) if args.backend == 'math' else contextlib.nullcontext()
    with context:
        model, processor, teacher, old_size, snapshot = initialize(SimpleNamespace(weights=args.weights, no_encoder_checkpoint=False))
        parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if len(parameters) != 207:
            raise ValueError('Expected the exact trained full-24-layer rank8 207-tensor layout.')
        heads = [p for name, p in parameters if not name.endswith(('lora_A', 'lora_B'))]
        adapters = [p for name, p in parameters if name.endswith(('lora_A', 'lora_B'))]
        optimizer = torch.optim.AdamW([{'params': heads, 'lr': args.head_lr, 'weight_decay': .01},
                                       {'params': adapters, 'lr': args.lora_lr, 'weight_decay': 0.}],
                                      betas=(.9, .999), eps=1e-8)
        row_names = ('decoder.embedding.weight', 'joint.head.weight', 'joint.head.bias')
        named = dict(parameters)
        new_parameters = [named[name] for name in row_names]
        cpu_batches = [collate_training_batch(processor, [rows[i] for i in group], [items[i] for i in group],
                        blank_id=model.config.blank_token_id, old_size=old_size, device='cpu') for group in groups]
        invariants = [batch_invariants(batch, group) for batch, group in zip(cpu_batches, groups)]
        for row, item in zip(rows, items):
            expected = torch.tensor(processor.tokenizer(row['text'])['input_ids'], dtype=torch.long)
            if not torch.equal(expected, item['labels']):
                raise ValueError('Cached labels differ from the actual extended tokenizer.')
            if row['language'] != 'tr' and int(expected.max()) >= old_size:
                raise ValueError('Replay contains an appended tokenizer ID.')
        identity = {'weights_sha256': digest(args.weights), 'precision': precision,
                    'source_ids': [row['source_id'] for row in rows],
                    'wav_sha256': [row['wav_sha256'] for row in specs],
                    'cache_sha256': [row['cache_sha256'] for row in specs],
                    'pool_sha256': pool_sha, 'source_sha256': hashes,
                    'base_revision': (ROOT / 'work/base_model_path.txt').read_text().strip().split('\\')[-1],
                    'cache_namespace': Path(rows[0]['cache_path']).parent.name}
        native = args.mode == 'fp32-reference'
        loss_fn = original_loss if tdt == 'original' else v2_loss
        flags = backend_flags()
        if args.backend == 'math' and (not flags['sdpa_math_enabled'] or any(flags[key] for key in
                ('sdpa_flash_enabled', 'sdpa_mem_efficient_enabled', 'sdpa_cudnn_enabled'))):
            raise ValueError('Forced MATH flags are not exclusive.')
        initial_digest = state_digest(snapshot['weights'])

        def backward(prefetched, audit=False):
            optimizer.zero_grad(set_to_none=True)
            component_values = [0., 0., 0., 0.]
            per_clip = {name: [None] * 16 for name in (*COMPONENTS, 'total')}
            features = [None] * 16
            for group_index, group in enumerate(groups):
                if native:
                    index = group[0]
                    batch = cpu_batches[group_index].to('cuda') if prefetched else collate_training_batch(
                        processor, [rows[index]], blank_id=model.config.blank_token_id, old_size=old_size, device='cuda')
                    item = items[index] if prefetched else torch.load(rows[index]['cache_path'], map_location='cpu', weights_only=True)
                    loss, values, feature = original_objective(model, processor, teacher, old_size, rows[index], item,
                        16, inputs=batch.inputs, debug=audit, use_autocast=precision == 'bf16')
                    (loss / 16).backward()
                    values = [float(value.detach()) for value in values]
                    component_values = [a + b / 16 for a, b in zip(component_values, values)]
                    if audit:
                        for name, value in zip(COMPONENTS, values):
                            per_clip[name][index] = value
                        per_clip['total'][index] = float(loss.detach())
                        features[index] = feature
                    del batch, loss
                else:
                    batch = cpu_batches[group_index].to('cuda') if prefetched else collate_training_batch(
                        processor, [rows[index] for index in group], blank_id=model.config.blank_token_id,
                        old_size=old_size, device='cuda')
                    result = batch_objective(model, teacher, batch, old_size=old_size, replay_kl=5,
                        duration_kl=.5, feature_distillation=1, chunk_size=16, tdt_loss_fn=loss_fn,
                        tdt_host_lengths=tdt == 'v2', use_autocast=precision == 'bf16',
                        return_debug_features=audit)
                    scaled = result.loss / len(groups)
                    if not torch.isfinite(scaled):
                        raise ValueError('Nonfinite profiled objective.')
                    scaled.backward()
                    metrics = result.detached_metrics()  # Include actual trainer scalar synchronization.
                    values = [metrics[key] for key in ('tdt_loss', 'replay_kl', 'replay_duration_kl', 'replay_feature_distillation')]
                    component_values = [a + b / len(groups) for a, b in zip(component_values, values)]
                    if audit:
                        for local_index, index in enumerate(group):
                            for key, name in (('tdt', 'tdt'), ('token_kl', 'token_kl'), ('duration_kl', 'duration_kl'),
                                              ('feature_distillation', 'feature_distillation'), ('total', 'total')):
                                per_clip[name][index] = float(result.per_clip[key][local_index].detach())
                            features[index] = result.debug_encoder_features[local_index:local_index + 1,
                                               :rows[index]['encoder_frames']].detach().cpu()
                    del result, scaled, batch
            if audit:
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for _, parameter in parameters):
                    raise ValueError('Missing or nonfinite trainable gradient.')
                for name in ('token_kl', 'duration_kl', 'feature_distillation'):
                    if any(per_clip[name][index] != 0 for index in range(0, 16, 2)):
                        raise ValueError('Turkish preservation terms are not literal zeros.')
            return component_values, per_clip, features

        def adam_update(record=False):
            initial_state_empty = not bool(optimizer.state)
            before = {name: named[name].detach().cpu().clone() for name in row_names} if record else None
            unclipped = {name: named[name].grad.detach().cpu().clone() for name in row_names} if record else None
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
            clipped = {name: named[name].grad.detach().cpu().clone() for name in row_names} if record else None
            previous = [p[old_size:model.config.vocab_size].detach().clone() for p in new_parameters]
            optimizer.step()
            intermediate = {name: named[name].detach().cpu().clone() for name in row_names} if record else None
            with torch.no_grad():
                for p, old in zip(new_parameters, previous):
                    new = p[old_size:model.config.vocab_size]
                    new.copy_(old + args.new_token_lr_multiplier * (new - old))
            if record:
                final = {name: named[name].detach().cpu().clone() for name in row_names}
                return {'optimizer': 'AdamW', 'initial_state_empty': initial_state_empty,
                        'gradient_norm_before_clip': float(norm), 'unclipped_global_norm': float(norm),
                        'maximum_gradient_norm': 1., 'max_grad_norm': 1.,
                        'old_size': old_size, 'vocab_size': model.config.vocab_size,
                        'new_token_lr_multiplier': args.new_token_lr_multiplier, 'row_multiplier': args.new_token_lr_multiplier,
                        'lr': args.head_lr, 'weight_decay': .01,
                        'parameters': {name: {'before': before[name], 'unclipped_gradient': unclipped[name],
                            'clipped_gradient': clipped[name], 'after_adam_before_row_scaling': intermediate[name],
                            'after_adam': intermediate[name], 'after_row_scaling': final[name], 'after_multiplier': final[name],
                            'lr': args.head_lr, 'weight_decay': .01,
                            'optimizer_step': float(optimizer.state[named[name]]['step']),
                            'exp_avg': optimizer.state[named[name]]['exp_avg'].detach().cpu().clone(),
                            'exp_avg_sq': optimizer.state[named[name]]['exp_avg_sq'].detach().cpu().clone()}
                            for name in row_names}, 'betas': [.9, .999], 'eps': 1e-8}
            return None

        def reset():
            model.load_state_dict(snapshot['weights'], strict=False)
            optimizer.zero_grad(set_to_none=True)
            # Preserve warmed allocations, but reset all moments and step counts.
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
            torch.manual_seed(20261010)
            torch.cuda.manual_seed_all(20261010)
            if state_digest({name: p for name, p in parameters}) != initial_digest:
                raise ValueError('Warmup reset changed the starting trainable state.')
            if any(bool(value.any()) for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)):
                raise ValueError('Optimizer moments/step reset failed.')

        torch.cuda.reset_peak_memory_stats()
        components, per_clip, encoder_features = backward(True, audit=True)
        gradient_artifact = {'identity': identity, 'reference_batch_size': args.batch_size,
                             'components': components, 'per_clip': per_clip,
                             'gradients': {name: p.grad.detach().cpu().clone() for name, p in parameters},
                             'encoder_features': encoder_features, 'microbatches': invariants,
                             'microbatch_tensors': [{
                                 'indices': group, 'labels': batch.labels.cpu().clone(),
                                 'decoder_input_ids': batch.decoder_input_ids.cpu().clone(),
                                 'target_lengths': batch.target_lengths.cpu().clone(),
                                 'frame_lengths': batch.frame_lengths.cpu().clone(),
                                 'input_attention_mask': batch.inputs['attention_mask'].cpu().clone(),
                                 'input_features': batch.inputs['input_features'].cpu().clone(),
                                 'base_features': batch.base_features.cpu().clone(),
                                 'replay_indices': list(batch.replay_indices)}
                                 for group, batch in zip(groups, cpu_batches)],
                             'backend_flags': flags}
        parity = {'scope': 'Reference capture; no candidate comparison.', 'passed': None}
        if args.mode in ('fp32-parity', 'bf16-parity') or (args.mode == 'timing' and args.reference_path):
            if not args.reference_path or not args.reference_path.exists():
                raise ValueError('A matching full-gradient reference path is required.')
            reference = torch.load(args.reference_path, map_location='cpu', weights_only=True)
            comparison = compare_tensors(reference, gradient_artifact)
            if args.mode == 'fp32-parity':
                gate = strict_gate(comparison)
                component_passed = (max(comparison['component_absolute_deltas'].values()) < 1e-4
                                    and comparison['weighted_objective_absolute_delta'] < 1e-4)
                gate['component_absolute_delta_max'] = 1e-4
                gate['component_and_weighted_objective_passed'] = component_passed
                gate['passed'] &= component_passed
                parity = {**comparison, 'strict_gate': gate, 'passed': gate['passed'],
                          'scope': 'FP32 forced-MATH native B1x16 versus physical batch with the same per-clip objective.'}
            else:
                if reference['reference_batch_size'] != args.batch_size or reference['backend_flags'] != flags:
                    raise ValueError('BF16 original-v2 reference must use identical physical batch and backend.')
                gate = strict_gate(comparison, same_shape=True)
                parity = {**comparison, 'strict_gate': gate, 'passed': gate['passed'],
                          'scope': 'Same-shape/backend BF16 original TDT versus v2; encoder activations must be bitwise equal.'}
            del reference
        adam = adam_update(record=True)
        gradient_artifact['adam_update'] = adam
        torch.save(gradient_artifact, evidence_path)
        numerical_peak = {'allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                          'reserved_gb': torch.cuda.max_memory_reserved() / 1e9}
        timing = {}
        if args.mode == 'timing':
            if parity['passed'] is False and not args.exploratory_timing:
                raise ValueError('Same-shape original-v2 parity failed; refusing timing.')
            if not args.reference_path:
                raise ValueError('Timing requires the matching BF16 original-TDT full-gradient reference.')
            for prefetched in ((False, True) if args.timing_category == 'both' else (args.timing_category == 'prefetched',)):
                reset()
                for _ in range(args.warmup):
                    backward(prefetched)
                    adam_update()
                torch.cuda.synchronize()
                reset()
                torch.cuda.synchronize()
                process_before = gpu_process_evidence()
                torch.cuda.reset_peak_memory_stats()
                start_utc = datetime.now(timezone.utc).isoformat()
                seconds, losses = [], []
                for update in range(args.updates):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    values, _, _ = backward(prefetched)
                    adam_update()
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter() - started)
                    losses.append(values)
                    print(json.dumps({'case_id': case_id, 'category': 'prefetched' if prefetched else 'end_to_end',
                                      'update': update + 1, 'seconds': seconds[-1]}), flush=True)
                ended_utc = datetime.now(timezone.utc).isoformat()
                allocated, reserved = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
                if allocated > args.max_gpu_gb:
                    raise ValueError('GPU process peak exceeded the declared memory ceiling.')
                audio_per_update = math.fsum(row['duration'] for row in rows)
                category = 'prefetched_cpu_inputs' if prefetched else 'end_to_end'
                timing[category] = {'updates': args.updates, 'warmup_updates': args.warmup,
                    'seconds': seconds, 'component_losses': losses,
                    'median_seconds_per_update': statistics.median(seconds),
                    'p90_seconds_per_update': sorted(seconds)[min(len(seconds) - 1, int(.9 * len(seconds)))],
                    'audio_seconds_seen': audio_per_update * args.updates,
                    'audio_min_per_wall_min': audio_per_update * args.updates / math.fsum(seconds),
                    'clips_per_second': 16 * args.updates / math.fsum(seconds),
                    'peak_allocated_vram_gb': allocated, 'peak_reserved_vram_gb': reserved,
                    'started_at_utc': start_utc, 'ended_at_utc': ended_utc,
                    'solo_gpu_operator_declared': True, 'process_inventory_before': process_before,
                    'process_inventory_after': gpu_process_evidence(),
                    'reset_starting_trainable_sha256': initial_digest,
                    'reset_optimizer_moments_and_step': True,
                    'reset_reuses_warmed_optimizer_buffers': True,
                    'includes': ['H2D', 'encoder/decoder/joint/loss/backward', 'scalar logging synchronization', 'gradient clipping', 'AdamW', 'new-token row update scaling'] +
                        ([] if prefetched else ['WAV reads', 'teacher-cache tensor reads', 'frontend', 'CPU collate']),
                    'excludes': ['development/test inference', 'checkpoint/report serialization', 'warmup', 'CUDA synchronization before timer', 'initial model/teacher loading']}
        if source_hashes() != hashes or digest(args.pool) != pool_sha or digest(args.weights) != identity['weights_sha256']:
            raise ValueError('Frozen source/pool/weight provenance changed during profiling.')
        if args.mode == 'timing':
            status = 'measured_with_numerical_gate_failure' if parity['passed'] is False else 'measured'
        else:
            status = 'strict_parity_failed' if parity['passed'] is False else 'reference_saved'
        report = {'schema_version': 1, 'status': status, 'case_id': case_id, 'family': args.case,
            'mode': args.mode, 'batch_size': args.batch_size, 'accumulation': len(groups), 'effective_batch': 16,
            'precision': precision, 'tdt_implementation': tdt, 'backend': args.backend,
            'backend_flags': flags, 'encoder_checkpointing': True, 'chunk_size': 16,
            'source_ids': identity['source_ids'], 'source_clips': specs, 'groups': groups,
            'microbatches': invariants, 'padding': padding, 'identity': identity,
            'audio_seconds_per_update': math.fsum(row['duration'] for row in rows),
            'objective_coefficients': {'tdt': 1, 'token_kl': 5, 'duration_kl': .5, 'feature_distillation': 1},
            'turkish_preservation_terms_zero': True, 'parity': parity, 'timing': timing,
            'evidence_path': str(evidence_path), 'evidence_sha256': digest(evidence_path),
            'reference_path': str(args.reference_path) if args.reference_path else None,
            'reference_sha256': digest(args.reference_path) if args.reference_path else None,
            'gradient_tensors': len(parameters), 'gradient_elements': sum(p.numel() for _, p in parameters),
            'numerical_gpu_peak': numerical_peak, 'source_sha256': hashes,
            'initial_trainable_sha256': initial_digest,
            'optimizer': {'head_lr': args.head_lr, 'lora_lr': args.lora_lr, 'betas': [.9, .999], 'eps': 1e-8,
                          'head_weight_decay': .01, 'lora_weight_decay': 0., 'clip_norm': 1.,
                          'new_token_lr_multiplier': args.new_token_lr_multiplier, 'old_size': old_size,
                          'extended_vocab_size': model.config.vocab_size, 'optimizer_step_before_audit': 0},
            'gpu_context': args.gpu_context, 'hardware': torch.cuda.get_device_name(0),
            'max_gpu_gb': args.max_gpu_gb,
            'torch_version': str(torch.__version__), 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'pause_provenance': {'path': str(args.pause_provenance), 'sha256': digest(args.pause_provenance)} if args.pause_provenance else None,
            'eligibility': 'Numerical/speed evidence only; no adoption in the current loop.',
            'adoption_eligible': False, 'exploratory_timing_authorized': args.exploratory_timing,
            'limitations': ['16 fixed TRAIN clips repeatedly presented; effective16 differs from current effective4 optimizer experiment.',
                            'BF16 batch-shape differences must be assessed separately from same-shape original-v2 parity.',
                            'Timing is valid only for this parent-coordinated solo GPU window and recorded backend.']}
        write(args.output, report)
        register_matrix(args, report)
        print(json.dumps({'saved': str(args.output), 'status': status, 'timing': {
            name: {'audio_min_per_wall_min': value['audio_min_per_wall_min'],
                   'peak_allocated_vram_gb': value['peak_allocated_vram_gb']} for name, value in timing.items()}}), flush=True)
        return 2 if parity['passed'] is False and args.mode != 'timing' else 0


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--pool', type=Path, default=OUTPUT / 'runs/adaptive_20261010_01_screen_A/training_pool.json')
    value.add_argument('--weights', type=Path, default=ROOT / 'work/lora_20261010_01_best.pt')
    value.add_argument('--case', choices=('typical', 'long'), default='typical')
    value.add_argument('--batch-size', type=int, choices=(1, 2, 4, 8, 16), default=1)
    value.add_argument('--mode', choices=('fp32-reference', 'fp32-parity', 'bf16-reference', 'bf16-parity', 'timing'), default='timing')
    value.add_argument('--backend', choices=('default', 'math'), default='math')
    value.add_argument('--updates', type=int, default=10)
    value.add_argument('--warmup', type=int, default=2)
    value.add_argument('--reference-path', type=Path)
    value.add_argument('--head-lr', type=float, default=2e-5)
    value.add_argument('--lora-lr', type=float, default=3e-4)
    value.add_argument('--new-token-lr-multiplier', type=float, default=15.)
    value.add_argument('--max-gpu-gb', type=float, default=14.5)
    value.add_argument('--gpu-context', choices=('solo', 'concurrent'), default='concurrent')
    value.add_argument('--timing-category', choices=('end_to_end', 'prefetched', 'both'), default='end_to_end')
    value.add_argument('--exploratory-timing', action='store_true', help='Measure finite cases despite a documented numerical gate failure; never authorizes adoption.')
    value.add_argument('--pause-provenance', type=Path)
    value.add_argument('--plan-only', action='store_true')
    value.add_argument('--output', type=Path)
    value.add_argument('--matrix-output', type=Path, default=OUTPUT / 'large_batch_matrix.json')
    return value


if __name__ == '__main__':
    args = parser().parse_args()
    if args.output is None:
        args.output = OUTPUT / f'large_batch_{args.case}_{args.mode.replace("-", "_")}_{args.backend}_b{args.batch_size}.json'
    try:
        raise SystemExit(run(args))
    except Exception as error:
        # Preserve completed cases and an explicit failed/OOM row in bounded runs.
        if not args.plan_only and not args.output.exists():
            error_report = {'schema_version': 1, 'status': 'error_oom' if isinstance(error, torch.OutOfMemoryError) else 'error',
                'case_id': args.output.stem, 'family': args.case, 'mode': args.mode,
                'batch_size': args.batch_size, 'effective_batch': 16,
                'precision': 'fp32' if args.mode.startswith('fp32') else 'bf16',
                'tdt_implementation': 'original' if args.mode in ('fp32-reference', 'bf16-reference') else 'v2',
                'backend': args.backend, 'timing': {}, 'error': str(error), 'traceback': traceback.format_exc()}
            write(args.output, error_report)
            register_matrix(args, error_report)
        raise
