"""Check the actual closed acquisition checkpoint before GPU startup."""
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
source = HERE / 'tsc_bpe_lora_transition_v2.py'
expected = '22407adf7301fae61c823cf391c87c486bc60e48d748935210b2ed91c8a77469'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
from tsc_bpe_lora_transition_v2 import _context, validate_phase1_state
receipt = HERE / 'TSC_BPE512_phase1_actual_CLOSED_for_LoRA_main_v1.json'
context = _context(str(receipt), 'adaptive_20261011_hf_TSC_BPE512_lora_01', hashlib.sha256(receipt.read_bytes()).hexdigest())
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
state = torch.load(context['receipt']['artifacts']['chosen_checkpoint']['path'], map_location='cpu', weights_only=True)
result = validate_phase1_state(state, context)
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
report = {'kind': 'ROOT_actual_CLOSED_phase1_corrected_helper_CPU_validation_v1',
          'passed': True, 'checkpoint_artifact': context['receipt']['artifacts']['chosen_checkpoint'],
          'actual_checkpoint_loaded_CPU_only': True, 'CUDA_initialized': torch.cuda.is_initialized(),
          'helper_sha256': expected, 'boundary': result, 'weights_optimizer_tensor_audits_pending': True}
path = HERE / 'TSC_BPE512_actual_phase1_corrected_helper_CPU_validation_v1.json'
assert not path.exists()
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
