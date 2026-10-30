"""One-shot stdlib generation of NEW identity-routing-only HF continuation files."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,text):
    p=O/name
    with p.open('x',encoding='utf-8') as f:f.write(text)
    ast.parse(text)
    return sha(p)

trainer=(O/'train_lora_batched_data_v10.py').read_text(encoding='utf-8')
trainer=trainer.replace('"train_lora_batched_data_v10.py", "native_tdt_checkpointed_v3.py"',
    '"train_lora_batched_data_v11.py", "hf_exact_runtime_continuation_v1.py", "native_tdt_checkpointed_v3.py"')
trainer=trainer.replace('"launch_native_training_v4.py", "public_turkish_data_policy_v3.py"',
    '"launch_native_training_v5.py", "public_turkish_data_policy_v3.py"')
needle="    if args.resume_runtime_migration is None:\n        return old == current\n"
assert trainer.count(needle)==1
trainer=trainer.replace(needle,needle+"    receipt_kind = json.loads(args.resume_runtime_migration.read_text(encoding='utf-8')).get('kind')\n    if receipt_kind == 'HF_already_native3_exact_runtime_continuation_v1':\n        from hf_exact_runtime_continuation_v1 import exact_identity_matches\n        return exact_identity_matches(saved, args, current, OUTPUT, native_loss_identity)\n")
pins={'train_lora_batched_data_v11.py':put('train_lora_batched_data_v11.py',trainer)}
io=(O/'launch_with_io_retry_v8.py').read_text(encoding='utf-8').replace('launch_with_io_retry_v8','launch_with_io_retry_v9').replace('atomic_io_retry_v8','atomic_io_retry_v9').replace('train_lora_batched_data_v10.py','train_lora_batched_data_v11.py')
pins['launch_with_io_retry_v9.py']=put('launch_with_io_retry_v9.py',io)
cache=(O/'launch_with_cache_reclaim_v11.py').read_text(encoding='utf-8').replace('train_lora_batched_data_v10.py','train_lora_batched_data_v11.py').replace('launch_with_io_retry_v8','launch_with_io_retry_v9').replace('cache_reclamation_training_launcher_v11','cache_reclamation_training_launcher_v12')
cache=cache.replace('4581e9510d7fa31d4b40ab795cd3a1c8af2f2df34ef729703607d97a37c0eb3c',pins['train_lora_batched_data_v11.py']).replace('2b99a23868e8e793df3ccf782536cbb91b33e212306422dc6bb6b3edba99cce9',pins['launch_with_io_retry_v9.py'])
pins['launch_with_cache_reclaim_v12.py']=put('launch_with_cache_reclaim_v12.py',cache)
memory=(O/'launch_with_memory_guard_v14.py').read_text(encoding='utf-8').replace('train_lora_batched_data_v10.py','train_lora_batched_data_v11.py').replace('launch_with_cache_reclaim_v11','launch_with_cache_reclaim_v12').replace('memory_guard_training_launcher_v14','memory_guard_training_launcher_v15')
memory=memory.replace('4581e9510d7fa31d4b40ab795cd3a1c8af2f2df34ef729703607d97a37c0eb3c',pins['train_lora_batched_data_v11.py']).replace('03e02fbeb706cab907fd95a9f8d89c5fc01b31b4b9e668460973f8f0bb589a4e',pins['launch_with_cache_reclaim_v12.py'])
pins['launch_with_memory_guard_v15.py']=put('launch_with_memory_guard_v15.py',memory)
outer=(O/'launch_native_training_v4.py').read_text(encoding='utf-8').replace('train_lora_batched_data_v10.py','train_lora_batched_data_v11.py').replace('launch_with_memory_guard_v14.py','launch_with_memory_guard_v15.py').replace('new native memory14 dispatch','new identity-only native memory15 dispatch')
pins['launch_native_training_v5.py']=put('launch_native_training_v5.py',outer)
pins['hf_exact_runtime_continuation_v1.py']=sha(O/'hf_exact_runtime_continuation_v1.py')
old=ast.parse((O/'train_lora_batched_data_v10.py').read_text(encoding='utf-8'))
new=ast.parse(trainer)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
runold,runnew=fn(old,'run'),fn(new,'run')
# Only the recorded source_names literals differ in run().
def strip_source_names(node):
    return ast.Module(body=[n for n in node.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source_names' for t in n.targets))],type_ignores=[])
assert ast.dump(strip_source_names(runold),include_attributes=False)==ast.dump(strip_source_names(runnew),include_attributes=False)
assert trainer.splitlines()[543].strip()=='torch.cuda.synchronize()'
assert all(ast.dump(fn(old,n),include_attributes=False)==ast.dump(fn(new,n),include_attributes=False) for n in ('save_training_state','parser','validate_arguments'))
(O/'HF_862_runtime_chain_pins_v1.json').write_text(json.dumps({'schema':1,'passed':True,'source_sha256':pins,'training_update_AST_unchanged':True,'optimizer_end_line':544,'CUDA_initialized':False},indent=2),encoding='utf-8')
print(json.dumps(pins,indent=2))
