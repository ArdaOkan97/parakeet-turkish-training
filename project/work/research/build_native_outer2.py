from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/parakeet_turkish'
s=(out/'launch_native_training_v1.py').read_text(encoding='utf-8')
for old,new in {
 'train_lora_batched_data_v7.py':'train_lora_batched_data_v8.py',
 'launch_with_memory_guard_v11.py':'launch_with_memory_guard_v12.py',
 'native_tdt_checkpointed_v1.py':'native_tdt_checkpointed_v2.py',
 'f41d19bc88b82e6a03b394e8e42d48820314aa7a3a3d189b959c5503c2e57a7c':'24e6f01a63c482d8451d72af9e21994332c0d0be53cbefbdf4c295f659090fd5',
 'native_tdt_training_backend_v1':'native_tdt_training_backend_v2',
 'native_tdt_training_backend_completion_v1':'native_tdt_training_backend_completion_v2',
 "option(argv[:argv.index('--')],'--cap-gb')=='9'":"option(argv[:argv.index('--')],'--cap-gb') in ('6','7')",
 'reviewed9GB/4GB memory policy':'declared6GB-or7GB/4GB memory policy',
 'reviewed new native memory11 dispatch':'reviewed new native memory12 dispatch',
 "'worker_arguments_sha256':argument_sha(worker),\n   'native_full":"'worker_arguments_sha256':argument_sha(worker),\n   'declared_allocator_policy':{'cap_gb':int(option(argv[:argv.index('--')],'--cap-gb')),'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},\n   'native_full",
}.items():
 if old not in s:raise ValueError(old)
 s=s.replace(old,new)
(out/'launch_native_training_v2.py').write_text(s,encoding='utf-8',newline='\n')
