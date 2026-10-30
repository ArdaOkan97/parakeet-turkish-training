from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];folder=ROOT/'work/research'
text=(folder/'test_data_v4_io_dispatch_cpu.py').read_text()
for a,b in [('train_lora_batched_data_v4','train_lora_batched_data_v6'),('launch_with_io_retry_v2','launch_with_io_retry_v4'),('launch_with_memory_guard_v8','launch_with_memory_guard_v10'),('launch_with_cache_reclaim_v5','launch_with_cache_reclaim_v7'),('atomic_io_retry_v2','atomic_io_retry_v4'),('data_v4_actual_io_dispatch_cpu_verification','data_v6_actual_io_dispatch_cpu_verification')]:text=text.replace(a,b)
target=folder/'test_data_v6_io_dispatch_cpu.py'
if target.exists():raise FileExistsError(target)
target.write_text(text,encoding='utf-8',newline='\n')
