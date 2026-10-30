"""Correctness-only full-objective gradients with forced SDPA MATH.

The default command writes a CPU-only plan. --run is reserved for a coordinated
GPU window. Each case runs the existing, unmodified profiler in a fresh process
with parity_only=True, saves all trainable gradients, and records backend flags.
There are no optimizer updates, speed measurements, or throughput claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WORK = ROOT / 'work' / 'batch_math_objective'
SOURCES = ('profile_batched_training.py', 'batched_lora.py', 'efficient_tdt.py',
           'efficient_tdt_v2.py', 'runtime.py', 'encoder_lora.py', 'extend_vocab.py',
           'train_lora.py', 'train_lora_batched.py')
CASES = {
    'fp32_b1_original': (1, 'fp32', 'original', True),
    'fp32_b2_original': (2, 'fp32', 'original', False),
    'fp32_b2_v2': (2, 'fp32', 'v2', False),
    'bf16_b1_original': (1, 'bf16', 'original', True),
    'bf16_b2_original': (2, 'bf16', 'original', False),
    'bf16_b2_v2': (2, 'bf16', 'v2', False),
}
CORE = ('fp32_b1_original', 'fp32_b2_v2', 'bf16_b2_original', 'bf16_b2_v2')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    return {name: digest(HERE / name) for name in SOURCES}


def case_paths(case):
    return {'reference': WORK / f'{case}_gradients.pt',
            'metadata': HERE / f'batch_math_{case}_metadata.json',
            'profile': HERE / f'batch_math_{case}_profile.json',
            'log': WORK / f'{case}.log'}


def flags():
    return {'tf32_matmul_allowed': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn_allowed': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'sdpa_flash_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'sdpa_math_enabled': torch.backends.cuda.math_sdp_enabled(),
            'sdpa_mem_efficient_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'sdpa_cudnn_enabled': torch.backends.cuda.cudnn_sdp_enabled()}


def execute_case(args):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import profile_batched_training as profiler
    if not torch.cuda.is_available():
        raise RuntimeError('This case requires the coordinated local GPU window.')
    start_hashes = source_hashes()
    wrapper_sha = digest(__file__)
    WORK.mkdir(parents=True, exist_ok=True)
    path = case_paths(args.job)
    if any(path[key].exists() for key in ('reference', 'metadata', 'profile')):
        raise RuntimeError(f'Existing case artifacts must be reviewed rather than overwritten: {args.job}')
    free, total = torch.cuda.mem_get_info()
    if free < 4_500_000_000:
        raise RuntimeError(f'Insufficient free GPU memory for this correctness case: {free} bytes.')
    # Enforced allocator ceiling, including activations; this is not a training
    # memory estimate. Fresh child processes release their allocations on exit.
    torch.cuda.set_per_process_memory_fraction(min(args.max_gpu_gb * 1e9 / total, .95))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    batch, precision, tdt, native = CASES[args.job]
    job_args = SimpleNamespace(batch_size=batch, precision=precision, tdt=tdt,
                               quartet=args.quartet, chunk_size=16,
                               no_encoder_checkpoint=False, reference=native,
                               reference_batched=not native, reference_path=path['reference'],
                               parity_only=True, weights=args.weights,
                               warmup=0, updates=0, output=path['profile'].name)
    with sdpa_kernel(SDPBackend.MATH):
        backend = flags()
        if not backend['sdpa_math_enabled'] or any(backend[key] for key in
                ('sdpa_flash_enabled', 'sdpa_mem_efficient_enabled', 'sdpa_cudnn_enabled')):
            raise RuntimeError('Forced MATH backend flags are not exclusive.')
        profiler.run(job_args)
        backend_after = flags()
    if source_hashes() != start_hashes or digest(__file__) != wrapper_sha:
        raise RuntimeError('A frozen source changed during the correctness case.')
    peak = torch.cuda.max_memory_allocated() / 1e9
    if peak > args.max_gpu_gb:
        raise RuntimeError('Diagnostic allocation exceeded the declared process ceiling.')
    profile = json.loads(path['profile'].read_text(encoding='utf-8'))
    if profile['timing'] or not profile['parity']['reference_created']:
        raise RuntimeError('Unexpected timing or missing full-gradient artifact.')
    metadata = {'schema_version': 1, 'status': 'complete', 'case': args.job,
                'correctness_only': True, 'optimizer_updates': 0, 'timing_measured': False,
                'concurrent_gpu_context': args.gpu_context, 'batch_size': batch,
                'accumulation': 4 // batch, 'effective_batch': 4, 'precision': precision,
                'tdt_implementation': tdt, 'native_single_clip_objective': native,
                'forced_backend': 'SDPBackend.MATH', 'backend_flags': backend,
                'backend_flags_after': backend_after, 'torch_version': torch.__version__,
                'hardware': torch.cuda.get_device_name(0),
                'gpu_peak_allocated_gb_diagnostic_only': peak,
                'gpu_peak_reserved_gb_diagnostic_only': torch.cuda.max_memory_reserved() / 1e9,
                'max_gpu_gb': args.max_gpu_gb, 'wrapper_sha256': wrapper_sha,
                'source_sha256': start_hashes, 'profile_report_path': str(path['profile']),
                'profile_report_sha256': digest(path['profile']),
                'gradients_path': str(path['reference']), 'gradients_sha256': digest(path['reference']),
                'source_ids': profile['source_ids'], 'weights_sha256': digest(args.weights)}
    write(path['metadata'], metadata)
    print(json.dumps({'case': args.job, 'status': 'complete', 'metadata': str(path['metadata']),
                      'diagnostic_gpu_peak_gb': peak}), flush=True)


def norm_stats(norm2, candidate2, delta2, dot):
    relative = math.sqrt(delta2 / max(norm2, 1e-30))
    cosine = dot / max(math.sqrt(norm2 * candidate2), 1e-30)
    if norm2 == candidate2 == delta2 == 0:
        relative, cosine = 0., 1.
    return {'reference_norm': math.sqrt(norm2), 'candidate_norm': math.sqrt(candidate2),
            'delta_norm': math.sqrt(delta2), 'relative_l2_difference': relative,
            'cosine_similarity': cosine}


def compare_tensors(reference, candidate, allow_precision_difference=False):
    left_identity, right_identity = dict(reference['identity']), dict(candidate['identity'])
    if allow_precision_difference:
        left_identity.pop('precision')
        right_identity.pop('precision')
    if left_identity != right_identity:
        raise ValueError('Gradient identity, weights, TRAIN source, WAV, cache, or frozen code differs.')
    if set(reference['gradients']) != set(candidate['gradients']):
        raise ValueError('Trainable gradient layout differs.')
    totals = [0., 0., 0., 0.]
    groups = {'encoder_lora': [0., 0., 0., 0.], 'heads': [0., 0., 0., 0.]}
    per_parameter, exactly_equal = [], True
    for name, a in reference['gradients'].items():
        b = candidate['gradients'][name]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'Gradient shape/finite check failed: {name}')
        equal = torch.equal(a, b)
        exactly_equal &= equal
        a, b = a.double(), b.double()
        values = [float(a.square().sum()), float(b.square().sum()),
                  float((a - b).square().sum()), float((a * b).sum())]
        group = 'encoder_lora' if name.endswith(('lora_A', 'lora_B')) else 'heads'
        totals = [x + y for x, y in zip(totals, values)]
        groups[group] = [x + y for x, y in zip(groups[group], values)]
        per_parameter.append({'name': name, 'elements': a.numel(), 'bitwise_equal': equal,
                              **norm_stats(*values)})
    features = []
    if len(reference['encoder_features']) != len(candidate['encoder_features']):
        raise ValueError('Encoder feature clip count differs.')
    for index, (a, b) in enumerate(zip(reference['encoder_features'], candidate['encoder_features'])):
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Encoder valid feature dimensions or finiteness differ.')
        equal = torch.equal(a, b)
        a, b = a.double(), b.double()
        features.append({'source_id': reference['identity']['source_ids'][index],
                         'shape': list(a.shape), 'bitwise_equal': equal,
                         'relative_l2_difference': float((a - b).norm() / a.norm().clamp_min(1e-30)),
                         'max_absolute_difference': float((a - b).abs().max())})
    names = ('tdt', 'token_kl', 'duration_kl', 'feature_distillation')
    coefficients = (1., 5., .5, 1.)
    component_deltas = [abs(a - b) for a, b in zip(reference['components'], candidate['components'])]
    if len(component_deltas) != 4 or not all(math.isfinite(x) for x in
            reference['components'] + candidate['components']):
        raise ValueError('Objective components missing or nonfinite.')
    return {'trainable_tensors': len(per_parameter),
            'trainable_elements': sum(row['elements'] for row in per_parameter),
            'all_gradients_bitwise_equal': exactly_equal, 'global_gradient': norm_stats(*totals),
            'gradient_groups': {name: norm_stats(*values) for name, values in groups.items()},
            'per_parameter': per_parameter, 'encoder_features': features,
            'reference_components': dict(zip(names, reference['components'])),
            'candidate_components': dict(zip(names, candidate['components'])),
            'component_absolute_deltas': dict(zip(names, component_deltas)),
            'weighted_objective_absolute_delta': abs(sum(a * c for a, c in
                zip(reference['components'], coefficients)) - sum(a * c for a, c in
                zip(candidate['components'], coefficients)))}


def load_case(case):
    path = case_paths(case)
    metadata = json.loads(path['metadata'].read_text(encoding='utf-8'))
    if metadata['case'] != case or metadata['status'] != 'complete':
        raise ValueError(f'Incomplete case: {case}')
    if metadata['profile_report_sha256'] != digest(path['profile']) or metadata['gradients_sha256'] != digest(path['reference']):
        raise ValueError(f'Case report/reference hash differs: {case}')
    if metadata['source_sha256'] != source_hashes() or metadata['wrapper_sha256'] != digest(__file__):
        raise ValueError(f'Case code provenance differs: {case}')
    backend = metadata['backend_flags']
    if not backend['sdpa_math_enabled'] or any(backend[key] for key in
            ('sdpa_flash_enabled', 'sdpa_mem_efficient_enabled', 'sdpa_cudnn_enabled',
             'tf32_matmul_allowed', 'tf32_cudnn_allowed')):
        raise ValueError(f'Case backend flags differ: {case}')
    profile = json.loads(path['profile'].read_text(encoding='utf-8'))
    if profile['timing']:
        raise ValueError('Correctness artifact unexpectedly contains timed updates.')
    return torch.load(path['reference'], map_location='cpu', weights_only=True), metadata


def strict_gate(value, same_shape=False):
    limits = {'gradient_relative_l2': .0005 if same_shape else .002,
              'gradient_cosine_min': .9999995 if same_shape else .99999,
              'encoder_feature_relative_l2': 1e-4,
              'component_absolute_delta': 1e-4 if same_shape else None}
    groups = [value['global_gradient'], *value['gradient_groups'].values()]
    checks = [group['relative_l2_difference'] < limits['gradient_relative_l2'] and
              group['cosine_similarity'] > limits['gradient_cosine_min'] for group in groups]
    checks.append(all(row['relative_l2_difference'] < limits['encoder_feature_relative_l2']
                      for row in value['encoder_features']))
    if same_shape:
        # Identical encoder shape/precision must produce exactly the same encoder
        # activations. TDT DP accumulation is audited numerically as well as by
        # the explicit bitwise equality field; it need not have identical order.
        checks.append(all(row['bitwise_equal'] for row in value['encoder_features']))
        checks.append(max(value['component_absolute_deltas'].values()) < 1e-4)
    return {'passed': all(checks), 'tolerances': limits,
            'same_shape_encoder_bitwise_equal_required': same_shape,
            'per_check_passed': checks}


def aggregate(args):
    pairs = [('fp32_b1_original', 'fp32_b2_v2', 'fp32_b1_original_vs_b2_v2', True, False),
             ('bf16_b2_original', 'bf16_b2_v2', 'bf16_b2_original_vs_v2_same_shape', True, True)]
    if not args.core_only:
        pairs += [('fp32_b1_original', 'fp32_b2_original', 'fp32_batch_shape_original', True, False),
                  ('fp32_b2_original', 'fp32_b2_v2', 'fp32_b2_original_vs_v2_same_shape', True, True),
                  ('fp32_b1_original', 'bf16_b1_original', 'b1_amp_vs_matching_fp32_oracle', False, False),
                  ('fp32_b2_original', 'bf16_b2_original', 'b2_amp_vs_matching_fp32_oracle', False, False),
                  ('bf16_b1_original', 'bf16_b2_original', 'bf16_batch_shape_original', False, False)]
    comparisons, artifacts = {}, {}
    for left, right, label, gated, same_shape in pairs:
        reference, rm = load_case(left)
        candidate, cm = load_case(right)
        result = compare_tensors(reference, candidate, allow_precision_difference=not gated and 'amp' in label)
        if result['trainable_tensors'] != 207:
            raise ValueError('Expected exact 207-tensor trained pilot layout.')
        result.update({'reference_case': left, 'candidate_case': right, 'gated': gated})
        if gated:
            result['strict_gate'] = strict_gate(result, same_shape)
        else:
            result['scope'] = 'Measured AMP or batch-shape diagnostic; no correctness waiver or automatic gate.'
        comparisons[label] = result
        artifacts[left], artifacts[right] = rm, cm
        del reference, candidate
    passed = all(row['strict_gate']['passed'] for row in comparisons.values() if row['gated'])
    value = {'schema_version': 1, 'status': 'passed' if passed else 'strict_parity_failed',
             'passed': passed, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'correctness_only': True, 'timing_measured': False, 'optimizer_updates': 0,
             'concurrent_gpu_context': args.gpu_context, 'forced_backend': 'SDPBackend.MATH',
             'wrapper_sha256': digest(__file__), 'source_sha256': source_hashes(),
             'effective_batch': 4, 'clip_mean_objective': {'tdt': 1, 'replay_token_kl': 5,
                 'replay_duration_kl': .5, 'replay_feature_distillation': 1,
                 'turkish_preservation_terms': 0},
             'comparisons': comparisons, 'case_artifacts': artifacts,
             'encoder_diagnostic_path': str(HERE / 'encoder_gpu_batching_diagnostic.json'),
             'encoder_diagnostic_sha256': digest(HERE / 'encoder_gpu_batching_diagnostic.json'),
             'limitations': ['Four fixed TRAIN clips at the same pilot state; no optimizer update.',
                 'No timing or throughput claim. BF16 shape error is reported separately from v2 loss parity.',
                 'Successful FP32 parity does not itself establish an acceptable BF16 training difference.']}
    write(args.output, value)
    print(json.dumps({'saved': str(args.output), 'status': value['status']}), flush=True)
    return 0 if passed else 2


def plan(args):
    import profile_batched_training as profiler
    rows, items = profiler.examples(SimpleNamespace(quartet=args.quartet))
    selected = CORE if args.core_only else tuple(CASES)
    value = {'schema_version': 1, 'status': 'planned_not_run', 'gpu_jobs_started': 0,
             'correctness_only': True, 'timing_measured': False,
             'wrapper_sha256': digest(__file__), 'source_sha256': source_hashes(),
             'weights': str(args.weights), 'weights_sha256': digest(args.weights),
             'cases': {name: {'batch_size': CASES[name][0], 'precision': CASES[name][1],
                             'tdt': CASES[name][2], 'native_original_b1': CASES[name][3],
                             'artifacts': {key: str(path) for key, path in case_paths(name).items()}}
                       for name in selected},
             'train_clips': [{'source_id': row['source_id'], 'language': row['language'],
                 'audio_seconds': row['duration'], 'encoder_frames': len(item['features']),
                 'target_tokens': len(item['labels']), 'wav_sha256': digest(row['audio_filepath']),
                 'cache_sha256': digest(row['cache_path'])} for row, item in zip(rows, items)],
             'strict_fp32_gate': {'group_and_global_relative_l2_max': .002,
                                 'group_and_global_cosine_min': .99999,
                                 'valid_encoder_feature_relative_l2_max': 1e-4},
             'gpu_process_memory_limit_gb': args.max_gpu_gb,
             'run_command': f'"{sys.executable}" "{Path(__file__).resolve()}" --run --gpu-context solo',
             'notes': ['Default invocation is CPU-only. Run only in a coordinated GPU window.',
                       'The original profiler/helper/trainer source files remain unchanged.']}
    write(HERE / 'batch_math_objective_plan.json', value)
    print(json.dumps({'saved': str(HERE / 'batch_math_objective_plan.json'), 'status': value['status']}), flush=True)


def main(args):
    if args.job:
        execute_case(args)
        return 0
    if args.compare_existing:
        return aggregate(args)
    if not args.run:
        plan(args)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    selected = CORE if args.core_only else tuple(CASES)
    for case in selected:
        path = case_paths(case)
        command = [sys.executable, str(Path(__file__).resolve()), '--job', case,
                   '--quartet', args.quartet, '--weights', str(args.weights),
                   '--max-gpu-gb', str(args.max_gpu_gb), '--gpu-context', args.gpu_context]
        with path['log'].open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        print(json.dumps({'case': case, 'returncode': completed.returncode,
                          'log': str(path['log'])}), flush=True)
        if completed.returncode:
            raise RuntimeError(f'Correctness case failed; inspect {path["log"]}')
    return aggregate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--compare-existing', action='store_true')
    parser.add_argument('--job', choices=tuple(CASES))
    parser.add_argument('--core-only', action='store_true')
    parser.add_argument('--quartet', default='median')
    parser.add_argument('--weights', type=Path, default=ROOT / 'work/lora_20261010_01_best.pt')
    parser.add_argument('--max-gpu-gb', type=float, default=6.)
    parser.add_argument('--gpu-context', choices=('solo', 'concurrent_teacher_cache', 'concurrent_development'), default='solo')
    parser.add_argument('--output', type=Path, default=HERE / 'batch_math_objective_gradient_diagnostic.json')
    raise SystemExit(main(parser.parse_args()))
