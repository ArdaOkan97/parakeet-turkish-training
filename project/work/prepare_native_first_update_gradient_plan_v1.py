"""Stdlib-only exact207 first-update capture plan, no model/GPU execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
PROFILER = OUT / 'profile_native_training_v1.py'
PROFILER_SHA = '8d2e7f0eb19db6b62fc12f4d3a38fae28a89fe07f1261abcaec2c3ce86f697f6'
SOURCE_PLAN = OUT / 'native_training_pilot_plan_v1_02.json'
PLAN_SHA = '96b6877f2ec83a4fa22feff208e179eb852e33939c1caf57372e8db9e94b9af9'
PYTHON = ROOT / 'work/nemo-native-env-v2/Scripts/python.exe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, text):
    if not value:
        raise ValueError(text)


require(sha(PROFILER) == PROFILER_SHA and sha(SOURCE_PLAN) == PLAN_SHA,
        'Preserve successful frozen profiler/exact TRAIN plan')
source = json.loads(SOURCE_PLAN.read_text(encoding='utf-8'))
for name, digest in source['code_sha256'].items():
    require(sha(OUT / name) == digest, 'Source changed since actual20-update pilot: ' + name)
first = source['updates'][0]
require(len(first['batches']) == 2 and all(len(b) == 8 for b in first['batches']), 'Exact B8x2 update required')
require(all(row['dataset_split'] == 'train' for b in first['batches'] for row in b), 'No heldout data permitted')
outputs = {arm: OUT / ('native_gradient_' + arm + '_profile_v1.json') for arm in ('custom', 'native')}
comparison = OUT / 'native_first_update_gradient_comparison_v1.json'
commands = {}
for arm, path in outputs.items():
    require(not path.exists() and not path.with_suffix('.first_update.pt').exists(), 'New capture artifacts required')
    commands[arm] = [str(PYTHON), '-X', 'utf8', str(PROFILER), '--arm', arm,
                    '--plan', str(SOURCE_PLAN), '--output', str(path), '--updates', '1',
                    '--warmup', '2', '--capture-gradients', '--cpu-threads', '2',
                    '--max-wall-seconds', '600', '--gpu-window-authorized',
                    '--gpu-context', 'root_owned_exclusive_after_control250_gradient_check']
require(not comparison.exists(), 'New comparison artifact required')
commands['compare'] = [str(ROOT / 'work/training-env/Scripts/python.exe'), '-X', 'utf8', str(PROFILER),
                       '--compare', '--custom-report', str(outputs['custom']),
                       '--native-report', str(outputs['native']), '--output', str(comparison)]
report = {'schema': 1, 'kind': 'queued_exact207_native_first_update_plan_v1', 'status': 'CPU_prepared',
          'preparation_script_sha256': sha(__file__),
          'profiler_path': str(PROFILER), 'profiler_sha256': PROFILER_SHA,
          'TRAIN_plan_path': str(SOURCE_PLAN), 'TRAIN_plan_sha256': PLAN_SHA,
          'checkpoint': source['weights'], 'physical_batch': 8, 'accumulation': 2, 'effective_batch': 16,
          'TRAIN_first_update_source_ids': first['source_ids'], 'audio': first['audio'],
          'runtime': source['runtime'], 'code_sha256': source['code_sha256'], 'commands': commands,
          'expected_full_gradient_tensors': 207, 'expected_trainable_elements': 19709068,
          'artifacts': {arm: {'report_path': str(path), 'gradient_path': str(path.with_suffix('.first_update.pt'))}
                        for arm, path in outputs.items()}, 'comparison_path': str(comparison),
          'strict_same_shape_BF16_gate': {'global_and_group_gradient_relative_l2_less_than': .0005,
                                        'global_and_group_gradient_cosine_greater_than': .9999995,
                                        'encoder_valid_features_bitwise_equal': True,
                                        'maximum_component_delta_less_than': 1e-4},
          'notes': ['Queued only: no GPU or Torch import in this preparation.',
                    'Both arms restart from identical1871 tensors with fresh Adam and identical RNG, not the current live250-step candidate.',
                    'Each command captures every trainable gradient plus per-clip valid encoder features/components during the first cold update before clipping/Adam; comparison applies existing frozen strict gate.',
                    'The existing profiler also performs two warmup updates, resets, and one measured update. These repeats are required by its frozen CLI; no new speed claim is intended.',
                    'Run arms sequentially only after parent confirms the production worker is closed; compiler preload is already inside the successful frozen profiler.',
                    'Do not waive a failed BF16 gate. Raw official-kernel full-logit parity and finite20-step loss do not themselves prove model-gradient equivalence.',
                    'A subsequent NEW metadata7 NeMo-loss trial still needs fixed DEV Turkish accuracy plus all five retention gates before promotion. Full NeMo trainer migration remains separate.'],
          'Torch_imported': False, 'CUDA_executed': False, 'adoption_authorized': False}
path = OUT / 'native_first_update_gradient_plan_v1.json'
require(not path.exists(), 'Preserve existing gradient plan')
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'status': report['status'], 'plan': str(path), 'sha256': sha(path),
                  'expected_full_gradient_tensors': 207, 'Torch_imported': False, 'CUDA_executed': False}))
