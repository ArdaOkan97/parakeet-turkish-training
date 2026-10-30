"""Narrowly extend all training dispatch branches; immutable older launchers."""
from pathlib import Path
import hashlib,ast,json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(name,text):
    p=D/name
    if p.exists():raise FileExistsError(p)
    ast.parse(text);p.write_text(text,encoding='utf-8',newline='\n');return sha(p)
text=(D/'launch_with_io_retry_v1.py').read_text(encoding='utf-8')
text=text.replace('launch_with_io_retry_v1.py','launch_with_io_retry_v2.py').replace("ALLOWED_WORKERS = ('train_lora_batched.py', 'finalize_lora.py')","ALLOWED_WORKERS = ('train_lora_batched_data_v4.py', 'finalize_lora.py')")
if text.count("if args.worker == 'train_lora_batched.py':")!=2:raise ValueError('Both trainer dispatch and save adapter branches required')
text=text.replace("if args.worker == 'train_lora_batched.py':","if args.worker == 'train_lora_batched_data_v4.py':").replace("'kind': 'atomic_io_retry_v1'","'kind': 'atomic_io_retry_v2'")
text=text.replace('parsed = worker.parser().parse_args(arguments)','parsed = worker.validate_arguments(worker.parser().parse_args(arguments))')
io=create('launch_with_io_retry_v2.py',text)
text=(D/'launch_with_cache_reclaim_v4.py').read_text(encoding='utf-8').replace('launch_with_io_retry_v1','launch_with_io_retry_v2').replace('98032d4f56e56c6441b5ab3551cbd52f0b6681d9b2f312fb307cbc9b285ef00f',io).replace('cache_reclamation_training_launcher_v4','cache_reclamation_training_launcher_v5')
cache=create('launch_with_cache_reclaim_v5.py',text)
text=(D/'launch_with_memory_guard_v7.py').read_text(encoding='utf-8').replace('launch_with_cache_reclaim_v4','launch_with_cache_reclaim_v5').replace('58879af4dded6e4909f436b2db57c74bc2b24ae173b45d83a92eff5493a79986',cache).replace('memory_guard_training_launcher_v7','memory_guard_training_launcher_v8')
memory=create('launch_with_memory_guard_v8.py',text)
print(json.dumps(dict(io=io,cache=cache,memory=memory)))
