"""ROOT-only closed checkpoint metadata inspection; no CUDA or tensor dump."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(HERE))
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
assert kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
from tsc_bpe_lora_transition_v1 import _context, canonical, PHASE1_STATE_KIND
receipt = HERE / 'TSC_BPE512_phase1_actual_CLOSED_for_LoRA_main_v1.json'
context = _context(str(receipt), 'adaptive_20261011_hf_TSC_BPE512_lora_01', hashlib.sha256(receipt.read_bytes()).hexdigest())
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
state = torch.load(context['receipt']['artifacts']['chosen_checkpoint']['path'], map_location='cpu', weights_only=True)
runtime = state.get('runtime_state', {})
checks = {
    'schema': (state.get('schema'), 1),
    'kind': (state.get('kind'), PHASE1_STATE_KIND),
    'state_run': (state.get('run_id'), context['receipt']['source_run_id']),
    'runtime_run': (runtime.get('run_id'), context['receipt']['source_run_id']),
    'state_step': (state.get('step'), context['source_step']),
    'runtime_step': (runtime.get('step'), context['source_step']),
    'state_source': (state.get('source_selected_step'), context['original_step']),
    'runtime_source': (runtime.get('source_selected_step'), context['original_step']),
    'acquisition_clock': (state.get('acquisition_optimizer_updates'), context['acquisition_updates']),
    'state_identity_sha': (canonical(state.get('config_identity')), canonical(context['parent_identity'])),
    'runtime_identity_sha': (canonical(runtime.get('config_identity')), canonical(context['parent_identity'])),
}
report = {'kind': 'actual_CLOSED_phase1_frontier_metadata_diagnostic_v1',
          'checkpoint_artifact': context['receipt']['artifacts']['chosen_checkpoint'],
          'actual_checkpoint_loaded_CPU_only': True, 'CUDA_initialized': torch.cuda.is_initialized(),
          'checks': {key: {'actual': pair[0], 'expected': pair[1], 'passed': pair[0] == pair[1]} for key, pair in checks.items()},
          'state_keys': list(state), 'runtime_keys': list(runtime)}
def representation_differences(left, right, prefix='config_identity'):
    differences = []
    if type(left) is not type(right):
        differences.append({'path': prefix, 'actual_type': type(left).__name__, 'expected_type': type(right).__name__,
                            'canonical_equal': canonical(left) == canonical(right)})
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys():
            differences.extend(representation_differences(left[key], right[key], prefix + '.' + str(key)))
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(representation_differences(a, b, prefix + '[' + str(index) + ']'))
    return differences
report['direct_state_identity_equal'] = state['config_identity'] == context['parent_identity']
report['direct_runtime_identity_equal'] = runtime['config_identity'] == context['parent_identity']
report['representation_differences'] = representation_differences(state['config_identity'], context['parent_identity'])
path = HERE / 'TSC_BPE512_actual_phase1_frontier_metadata_diagnostic_v2.json'
assert not path.exists()
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
