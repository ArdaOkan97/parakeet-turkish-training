"""Verify every pinned candidate TRAIN row without loading audio or ML libraries."""
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
import hf_bpe512_tr_metadata_v1 as helper

if sys.platform == 'win32':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.SetPriorityClass.restype = ctypes.c_int
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
started = time.perf_counter()
context = helper.read_candidate_context(OUT / 'HF_BPE_512_TR_target_metadata_v1.json')
count = tokens = maximum = bpe_rows = 0
for row in helper.iter_candidate_training_rows(context):
    item = helper.load_candidate_tr_item(row)
    count += 1
    tokens += len(item['labels'])
    maximum = max(maximum, len(item['labels']))
    bpe_rows += any(helper.BPE_ROWS[0] <= i < helper.BPE_ROWS[1] for i in item['labels'])
assert count == 74883 and tokens == 1131420 and maximum == 119
assert not {'torch', 'nemo', 'transformers'} & sys.modules.keys()
report = {
    'kind': 'ROOT_HF_BPE512_full_pinned_TRAIN_reader_verification_v1',
    'passed': True, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'clips': count, 'target_tokens_total': tokens, 'maximum_target_tokens': maximum,
    'rows_using_new_BPE_ids': bpe_rows, 'elapsed_seconds': time.perf_counter() - started,
    'helper_sha256': helper.sha_file(helper.__file__),
    'script_sha256': helper.sha_file(__file__), 'metadata_sha256': helper.METADATA_SHA,
    'all_pinned_file_hashes_and_row_order_verified': True,
    'audio_or_teacher_cache_read': False, 'ML_imports_or_GPU_work': False,
    'candidate_training_activated': False,
}
path = OUT / 'HF_BPE512_TR_full_reader_ROOT_verification_v1.json'
with path.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report))
