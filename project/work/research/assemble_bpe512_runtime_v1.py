"""One-time stdlib assembly: new routes only, assert frozen originals."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ORIGINALS = {
    'launch_with_io_retry_v9.py': '8d49db792cafa88341a64d893eeae4f09233edde36cdca3b53954f6da102cd3e',
    'launch_with_cache_reclaim_v12.py': '9febf980a1eb51dd35e5d7768afe73664774ca0445763f6cf1e2caccc5db802b',
    'launch_with_memory_guard_v15.py': '00213ae0a1af7ec5a24c9a2912bbab43a173c40a48448885cc971e7c4d77e3aa',
    'launch_native_training_v5.py': '562717de152de43976e9c7b3a72884dd0cadc93bd7910f749375259bd4fc723a'}
for name, digest in ORIGINALS.items(): assert sha(OUTPUT / name) == digest
WORKER = 'train_lora_batched_bpe512_v1.py'
WORKER_SHA = '9dca06b398a028d1f8fdc84c984f297f3874cb58917150a5199d6f6dfeb52f11'
assert sha(OUTPUT / WORKER) == WORKER_SHA
def save(name, text):
    with (OUTPUT / name).open('x', encoding='utf-8', newline='\n') as target: target.write(text)
def template(name):
    return (OUTPUT / name).read_text(encoding='utf-8').replace('train_lora_batched_data_v11.py', WORKER)

IO = 'launch_with_io_retry_bpe512_v1.py'
text = template('launch_with_io_retry_v9.py').replace('launch_with_io_retry_v9.py', IO).replace('atomic_io_retry_v9', 'atomic_io_retry_bpe512_v1')
text = text.replace("ALLOWED_WORKERS = ('train_lora_batched_bpe512_v1.py', 'finalize_lora.py')",
    f"FROZEN_TRAINER_SHA256 = '{WORKER_SHA}'\nALLOWED_WORKERS = ('{WORKER}',)")
text = text.replace('    worker_sha = sha(worker_path)',
    "    worker_sha = sha(worker_path)\n    if worker_sha != FROZEN_TRAINER_SHA256:\n        raise RuntimeError('Isolated BPE worker source changed')\n    if provenance.exists():\n        raise FileExistsError('Use new immutable BPE provenance')")
save(IO, text)

CACHE = 'launch_with_cache_reclaim_bpe512_v1.py'
text = template('launch_with_cache_reclaim_v12.py').replace('launch_with_io_retry_v9', IO.removesuffix('.py'))
text = text.replace('from cuda_cache_reclaim_v2 import CudaCacheReclaimer, file_sha',
    'from cuda_cache_caller_bpe512_v1 import CudaCacheReclaimer, file_sha')
text = text.replace('4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20', WORKER_SHA)
text = text.replace(ORIGINALS['launch_with_io_retry_v9.py'], sha(OUTPUT / IO))
text = text.replace('cache_reclamation_training_launcher_v12', 'cache_reclamation_training_launcher_bpe512_v1')
text = text.replace("'cache_helper_sha256': file_sha(OUTPUT/'cuda_cache_reclaim_v2.py'),",
    "'cache_helper_sha256': file_sha(OUTPUT/'cuda_cache_reclaim_v2.py'),\n               'caller_binding_sha256': file_sha(OUTPUT/'cuda_cache_caller_bpe512_v1.py'),")
text = text.replace("default=8.)", "default=4.)")
text = text.replace('NEW-process data-v3 trainer dispatch', 'Isolated BPE512 NEW-process dispatch')
save(CACHE, text)

MEMORY = 'launch_with_memory_guard_bpe512_v1.py'
text = template('launch_with_memory_guard_v15.py').replace('launch_with_cache_reclaim_v12', CACHE.removesuffix('.py'))
text = text.replace('4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20', WORKER_SHA)
text = text.replace(ORIGINALS['launch_with_cache_reclaim_v12.py'], sha(OUTPUT / CACHE))
text = text.replace('run:544', 'run:588').replace('memory_guard_training_launcher_v15', 'memory_guard_training_launcher_bpe512_v1')
text = text.replace('cap_gb not in (7., 8., 9.)', 'cap_gb != 8.').replace('args.cap_gb not in (7., 8., 9.)', 'args.cap_gb != 8.')
text = text.replace('default=9.', 'default=8.').replace('default=9.)', 'default=8.)')
text = text.replace('explicit7/8/9GB', 'explicit8GB').replace('Explicit7/8/9GB', 'Explicit8GB').replace('cap7/8/9GB', 'cap8GB').replace('declared7/8/9GB', 'declared8GB')
extras = ['cuda_cache_caller_bpe512_v1.py', 'hf_bpe512_training_v1.py', 'batched_lora_bpe512_v1.py',
    'hf_bpe512_tr_metadata_v1.py', 'expand_trainable_bpe_checkpoint_v1.py']
insert = "('native_tdt_checkpointed_v3.py', NATIVE_ADAPTER_SHA256)]"
replacement = "('native_tdt_checkpointed_v3.py', NATIVE_ADAPTER_SHA256),\n" + ',\n'.join(
    f"            ('{name}', '{sha(OUTPUT / name)}')" for name in extras) + ']'
assert insert in text; text = text.replace(insert, replacement)
save(MEMORY, text)

OUTER = 'launch_native_training_bpe512_v1.py'
text = template('launch_native_training_v5.py').replace('launch_with_memory_guard_v15.py', MEMORY)
text = text.replace("in ('7','8','9')", "== '8'").replace('root-selected7/8/9GB', 'root-selected8GB')
text = text.replace('Only reviewed new identity-only native memory15 dispatch allowed', 'Only reviewed isolated BPE memory dispatch allowed')
text = text.replace(' require(sha(wrapper)==args.memory_launcher_sha256 and sha(trainer)==args.trainer_sha256,',
    f" require(sha(wrapper)=='{sha(OUTPUT / MEMORY)}' and sha(trainer)=='{WORKER_SHA}','Pinned isolated BPE source changed')\n require(sha(wrapper)==args.memory_launcher_sha256 and sha(trainer)==args.trainer_sha256,")
text = text.replace(" run_id=option(worker,'--run-id')", " require('--resume-exact' not in worker and '--resume-runtime-migration' not in worker,'BPE arm must use fresh model-only warmstart')\n for flag in ('--resume-weights','--bpe-source-checkpoint','--bpe-source-selection','--bpe-source-closure','--bpe-conversion-report'):\n  scoped(option(worker,flag))\n run_id=option(worker,'--run-id')")
text = text.replace(' from preload_nemo_linker_v1 import preload', " require('torch' not in sys.modules,'Actual preload must precede Torch')\n from preload_nemo_linker_v1 import preload")
text = text.replace(' result=preload()  # No Torch/Numba/memory wrapper import may precede this call.',
    " result=preload()  # No Torch/Numba/memory wrapper import may precede this call.\n require(result.get('kind')=='coherent_native_cuda12_9_preload_v1' and result.get('actual_binding_linker',{}).get('version')==[12,9] and result.get('cuda_initialized') is False,'Actual coherent pre-Torch linkage required')")
save(OUTER, text)
for name, digest in ORIGINALS.items(): assert sha(OUTPUT / name) == digest
for name in [IO, CACHE, MEMORY, OUTER, 'cuda_cache_caller_bpe512_v1.py']: print(name, sha(OUTPUT / name))
