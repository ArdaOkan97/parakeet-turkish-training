"""New dispatch versions for the data-v2 trainer; frozen prior wrappers unchanged."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2]
directory=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
trainer=directory/'train_lora_batched_data_v2.py'
policy=directory/'public_turkish_data_policy_v2.py'
trainer_sha=sha(trainer);policy_sha=sha(policy)
if trainer_sha!='9f158449e181768489c64dfbe9060c81c4166ec4a8999340616124bf4e8f6f5a':raise ValueError('Reviewed data trainer differs')
if policy_sha!='bd0525a260edf6d4de16cc8b74c5e8ed180d00466a47b2adb063d776a907cd1c':raise ValueError('Reviewed data policy differs')
old_cache=directory/'launch_with_cache_reclaim_v1.py'
old_memory=directory/'launch_with_memory_guard_v3.py'
if sha(old_cache)!='db1ad326f26a89c5163a34a839cc781e457793b83d625a7c99b8366db9301459':raise ValueError('Frozen cache launcher differs')
if sha(old_memory)!='28e51998bbb481f89a1106ad55fa6579f6ce7475826a8229d615ff973dccc2d1':raise ValueError('Frozen memory v3 differs')
cache_text=old_cache.read_text(encoding='utf-8')
cache_text=cache_text.replace('from cuda_cache_reclaim_v1 import CudaCacheReclaimer, FROZEN_TRAINER_SHA256, file_sha',
    f"from cuda_cache_reclaim_v1 import CudaCacheReclaimer, file_sha\nFROZEN_TRAINER_SHA256 = '{trainer_sha}'")
cache_text=cache_text.replace("worker_path = OUTPUT/'train_lora_batched.py'","worker_path = OUTPUT/'train_lora_batched_data_v2.py'")
cache_text=cache_text.replace('max_events=args.max_events)','max_events=args.max_events, expected_sha=FROZEN_TRAINER_SHA256)')
cache_text=cache_text.replace("worker='train_lora_batched.py'","worker='train_lora_batched_data_v2.py'")
cache_text=cache_text.replace("'kind': 'cache_reclamation_training_launcher_v1'","'kind': 'cache_reclamation_training_launcher_v2'")
cache_text=cache_text.replace('"""NEW-process cache hook around the frozen versioned I/O training launcher.',
    '"""NEW-process data-v2 trainer dispatch through unchanged scoped cache/I/O helpers.')
new_cache=directory/'launch_with_cache_reclaim_v2.py'
if new_cache.exists():raise FileExistsError(new_cache)
new_cache.write_text(cache_text,encoding='utf-8',newline='\n')
new_cache_sha=sha(new_cache)
memory_text=old_memory.read_text(encoding='utf-8')
memory_text=memory_text.replace('from cuda_cache_reclaim_v1 import file_sha, FROZEN_TRAINER_SHA256',
    f"from cuda_cache_reclaim_v1 import file_sha\nFROZEN_TRAINER_SHA256 = '{trainer_sha}'\nDATA_POLICY_SHA256 = '{policy_sha}'")
memory_text=memory_text.replace('launch_with_cache_reclaim_v1','launch_with_cache_reclaim_v2')
memory_text=memory_text.replace('db1ad326f26a89c5163a34a839cc781e457793b83d625a7c99b8366db9301459',new_cache_sha)
memory_text=memory_text.replace("('train_lora_batched.py', FROZEN_TRAINER_SHA256)","('train_lora_batched_data_v2.py', FROZEN_TRAINER_SHA256),\n            ('public_turkish_data_policy_v2.py', DATA_POLICY_SHA256)")
memory_text=memory_text.replace("'frozen_worker_sha256': FROZEN_TRAINER_SHA256}","'frozen_worker_sha256': FROZEN_TRAINER_SHA256, 'data_policy_sha256': DATA_POLICY_SHA256}")
memory_text=memory_text.replace("file_sha(OUTPUT/'train_lora_batched.py')","file_sha(OUTPUT/'train_lora_batched_data_v2.py')")
memory_text=memory_text.replace("'memory_guard_training_launcher_v3'","'memory_guard_training_launcher_v4'")
memory_text=memory_text.replace('"""NEW-process 9GB allocator cap, 4GB cache cleanup and owned-worker telemetry.',
    '"""NEW data-v2 process:9GB allocator cap,4GB cache cleanup and owned-worker telemetry.')
new_memory=directory/'launch_with_memory_guard_v4.py'
if new_memory.exists():raise FileExistsError(new_memory)
new_memory.write_text(memory_text,encoding='utf-8',newline='\n')
print({'cache_launcher_sha256':sha(new_cache),'memory_launcher_sha256':sha(new_memory),
       'trainer_sha256':trainer_sha,'data_policy_sha256':policy_sha})
