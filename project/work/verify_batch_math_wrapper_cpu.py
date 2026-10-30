"""CPU-only negative guards for the numerical comparison wrapper."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'parakeet_turkish'
spec = importlib.util.spec_from_file_location('math_wrapper', OUT / 'diagnose_batch_objective_math.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []

def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

def rejects(name, left, right):
    try:
        module.compare_tensors(left, right)
    except ValueError:
        checks.append(name)
    else:
        raise AssertionError(name)

torch.manual_seed(77)
reference = {'identity': {'precision': 'fp32', 'source_ids': ['tr:a', 'de:b', 'tr:c', 'en:d'],
                         'weights_sha256': 'unchanged', 'wav_sha256': ['a', 'b', 'c', 'd']},
             'gradients': {f'encoder.layer{i}.lora_B': torch.randn(5, 3) for i in range(96)},
             'encoder_features': [torch.randn(1, 4 + i, 3) for i in range(4)],
             'components': [2., .03, .02, .001]}
reference['gradients'].update({f'head.{i}.weight': torch.randn(2, 4) for i in range(111)})
same = module.compare_tensors(reference, copy.deepcopy(reference))
check('all_207_gradients_compared', same['trainable_tensors'] == 207)
check('all_233?elements_exact', same['trainable_elements'] == 96 * 15 + 111 * 8)
check('identical_global_relative_zero', same['global_gradient']['relative_l2_difference'] == 0)
check('identical_global_cosine_one', abs(same['global_gradient']['cosine_similarity'] - 1) < 1e-12)
check('identical_gradients_bitwise', same['all_gradients_bitwise_equal'])
check('identical_features_bitwise', all(row['bitwise_equal'] for row in same['encoder_features']))
check('identical_strict_fp32_passes', module.strict_gate(same)['passed'])
check('identical_strict_same_shape_passes', module.strict_gate(same, same_shape=True)['passed'])

small = copy.deepcopy(reference)
for value in small['gradients'].values():
    value.mul_(1.00001)
near = module.compare_tensors(reference, small)
check('small_gradient_numerical_delta_passes', module.strict_gate(near)['passed'])
check('small_gradient_not_bitwise', not near['all_gradients_bitwise_equal'])
bad = copy.deepcopy(reference)
for value in bad['gradients'].values():
    value.mul_(1.01)
check('one_percent_gradient_delta_rejected', not module.strict_gate(module.compare_tensors(reference, bad))['passed'])
bad = copy.deepcopy(reference)
bad['encoder_features'][1].add_(.1)
check('feature_drift_rejected', not module.strict_gate(module.compare_tensors(reference, bad))['passed'])
bad = copy.deepcopy(reference)
bad['encoder_features'][1].add_(1e-7)
check('same_shape_requires_bitwise_encoder', not module.strict_gate(module.compare_tensors(reference, bad), True)['passed'])
bad = copy.deepcopy(reference)
bad['components'][2] += .002
check('same_shape_component_error_rejected', not module.strict_gate(module.compare_tensors(reference, bad), True)['passed'])
bad = copy.deepcopy(reference)
bad['components'][0] = float('nan')
rejects('nonfinite_components_rejected', reference, bad)
bad = copy.deepcopy(reference)
bad['gradients']['head.0.weight'][0, 0] = float('nan')
rejects('nonfinite_gradient_rejected', reference, bad)
bad = copy.deepcopy(reference)
bad['gradients'].pop('head.0.weight')
rejects('missing_gradient_rejected', reference, bad)
bad = copy.deepcopy(reference)
bad['gradients']['head.0.weight'] = torch.randn(4, 2)
rejects('gradient_shape_rejected', reference, bad)
bad = copy.deepcopy(reference)
bad['encoder_features'][0] = torch.randn(1, 9, 3)
rejects('valid_feature_shape_rejected', reference, bad)
bad = copy.deepcopy(reference)
bad['identity']['source_ids'][0] = 'wrong'
rejects('source_identity_rejected', reference, bad)
bad = copy.deepcopy(reference)
bad['identity']['precision'] = 'bf16'
rejects('precision_rejected_for_semantic_parity', reference, bad)
check('explicit_amp_comparison_allows_only_precision_change',
      module.compare_tensors(reference, bad, allow_precision_difference=True)['all_gradients_bitwise_equal'])

plan = json.loads((OUT / 'batch_math_objective_plan.json').read_text(encoding='utf-8'))
check('cpu_plan_claims_no_gpu', plan['gpu_jobs_started'] == 0 and plan['status'] == 'planned_not_run')
check('plan_has_six_distinct_gradient_cases', len(plan['cases']) == 6)
check('plan_uses_four_train_source_ids', len(plan['train_clips']) == 4)
check('plan_source_hashes_unchanged', plan['source_sha256'] == module.source_hashes())
check('wrapper_matches_plan_hash', plan['wrapper_sha256'] == module.digest(module.__file__))
report = {'passed': True, 'checks': len(checks), 'check_names': checks,
          'scope': 'CPU synthetic positive/negative numerical guards and actual TRAIN/protected-source plan; no GPU execution.',
          'wrapper_sha256': module.digest(module.__file__), 'source_sha256': module.source_hashes()}
module.write(OUT / 'batch_math_objective_cpu_verification.json', report)
print(json.dumps({'passed': True, 'checks': len(checks)}))
