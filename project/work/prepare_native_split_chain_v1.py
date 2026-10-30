"""Create a narrow immutable runtime version; stdlib only, no model imports."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text, old, new, count=None):
    actual = text.count(old)
    if actual == 0 or (count is not None and actual != count):
        raise ValueError(f'Unexpected occurrence count {old!r}: {actual}')
    return text.replace(old, new)


PINS = {
    'train_lora_batched_data_v9.py':'3e19e11c3a90ac4bf11688b7198df1f2ad65a1dbbb22d18414220c703024b287',
    'launch_with_io_retry_v7.py':'b097504dd3c202bd9f56d959b8f0efe075660855deb12a2c5b405558ca9b33e1',
    'launch_with_cache_reclaim_v10.py':'dc15a4fd119d04adafce1ec5d18324d0f28804b42121debced3c9b43bac10098',
    'launch_with_memory_guard_v13.py':'48837767a924912173095792e4d7f694d1100b06ad8b373c4b99adae6457fad8',
    'native_tdt_checkpointed_v3.py':'839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494',
}
for name, digest in PINS.items():
    if sha(OUT/name) != digest:
        raise ValueError('Frozen source changed: '+name)

created = {}


def save(name, text):
    path = OUT/name
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError('Existing generated file differs; refusing overwrite: '+str(path))
        created[name] = sha(path)
        return
    ast.parse(text, filename=name)
    path.write_text(text, encoding='utf-8', newline='\n')
    created[name] = sha(path)


trainer = (OUT/'train_lora_batched_data_v9.py').read_text(encoding='utf-8')
trainer = replace(trainer, 'train_lora_batched_data_v9.py', 'train_lora_batched_data_v10.py')
trainer = replace(trainer, 'native_tdt_checkpointed_v2', 'native_tdt_checkpointed_v3')
trainer = replace(trainer, 'launch_native_training_v3.py', 'launch_native_training_v4.py')
trainer = replace(trainer, 'native_tdt_training_backend_v2', 'native_tdt_training_backend_v3')
trainer = replace(trainer, '        loss_function = native_loss_function(args, tr, replay)', '        loss_function = native_loss_function(args, tr, replay)\n        from native_tdt_checkpointed_v3 import reclaim_inactive_cuda_cache', 1)
trainer = replace(trainer, '            del result, scaled, batch', '            del result, scaled, batch\n            if args.tdt_implementation == "nemo":\n                reclaim_inactive_cuda_cache(torch.device("cuda"), "after_whole_microbatch_backward")', 1)
trainer = replace(trainer, '        torch.cuda.synchronize()\n        train_wall +=', '        if args.tdt_implementation == "nemo":\n            reclaim_inactive_cuda_cache(torch.device("cuda"), "after_adam_and_new_rows")\n        torch.cuda.synchronize()\n        train_wall +=', 1)
cleanup_identity = '{"after_completed_microbatch_backward":True,"after_adam_and_new_rows":True,"inactive_threshold_bytes":750000000,"scope":"Only unused CUDA allocator blocks; no tensor/optimizer/RNG changes"}'
trainer = replace(trainer, '    return {**configuration_identity(), "max_joint_bytes":args.native_max_joint_bytes,', '    return {**configuration_identity(), "max_joint_bytes":args.native_max_joint_bytes,\n            "trainer_inactive_cuda_cache_cleanup":'+cleanup_identity+',', 1)
trainer = replace(trainer, "receipt.get('kind') != 'native_runtime_state_transition_v1'", "receipt.get('kind') != 'native_runtime_state_transition_v2'", 1)
start = trainer.index("    mappings = {'train_lora_batched_data_v7.py'")
end = trainer.index('    approved = receipt[', start)
trainer = trainer[:start] + "    mappings = {'train_lora_batched_data_v9.py':'train_lora_batched_data_v10.py',\n                'native_tdt_checkpointed_v2.py':'native_tdt_checkpointed_v3.py',\n                'launch_native_training_v3.py':'launch_native_training_v4.py'}\n" + trainer[end:]
trainer = replace(trainer, "expected['native_loss']['kind'] = 'native_nemo_checkpointed_per_clip_v2'", "expected['native_loss']['kind'] = 'native_nemo_checkpointed_per_clip_v3'\n    expected['native_loss']['native_wrapper_input_policy'] = 'Split joint output before FP32 casting; contiguous FP32 token logits and FP32 duration log_softmax passed to SAME official backend.loss'\n    expected['native_loss']['combined_fp32_joint_retained'] = False", 1)
trainer = replace(trainer, "    expected['native_loss']['combined_fp32_joint_retained'] = False", "    expected['native_loss']['combined_fp32_joint_retained'] = False\n    expected['native_loss']['trainer_inactive_cuda_cache_cleanup'] = "+cleanup_identity, 1)
trainer = replace(trainer, "policy.get('cap_gb') != 9", "policy.get('cap_gb') not in (7, 8, 9)", 1)
trainer = replace(trainer, 'Only declared native cache/runtime sources and ready provenance change; optimizer/sampler/RNG/history remain exact.', 'Only declared split-before-FP32 native runtime sources, allocator cap and ready provenance change; optimizer/sampler/RNG/history remain exact.', 1)
save('train_lora_batched_data_v10.py', trainer)
sync_lines = [node.lineno for node in ast.walk(ast.parse(trainer)) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.cuda.synchronize']
if len(sync_lines) != 1:
    raise ValueError('Expected one unchanged optimizer-end synchronization')
optimizer_sync_line = sync_lines[0]

io = (OUT/'launch_with_io_retry_v7.py').read_text(encoding='utf-8')
io = replace(io, 'launch_with_io_retry_v7', 'launch_with_io_retry_v8')
io = replace(io, 'train_lora_batched_data_v9.py', 'train_lora_batched_data_v10.py')
io = replace(io, 'atomic_io_retry_v7', 'atomic_io_retry_v8')
save('launch_with_io_retry_v8.py', io)

cache = (OUT/'launch_with_cache_reclaim_v10.py').read_text(encoding='utf-8')
cache = replace(cache, 'train_lora_batched_data_v9.py', 'train_lora_batched_data_v10.py')
cache = replace(cache, 'launch_with_io_retry_v7', 'launch_with_io_retry_v8')
cache = replace(cache, 'cache_reclamation_training_launcher_v10', 'cache_reclamation_training_launcher_v11')
cache = replace(cache, PINS['train_lora_batched_data_v9.py'], created['train_lora_batched_data_v10.py'])
cache = replace(cache, PINS['launch_with_io_retry_v7.py'], created['launch_with_io_retry_v8.py'])
save('launch_with_cache_reclaim_v11.py', cache)

memory = (OUT/'launch_with_memory_guard_v13.py').read_text(encoding='utf-8')
memory = replace(memory, 'train_lora_batched_data_v9.py', 'train_lora_batched_data_v10.py')
memory = replace(memory, 'launch_with_cache_reclaim_v10', 'launch_with_cache_reclaim_v11')
memory = replace(memory, 'native_tdt_checkpointed_v2', 'native_tdt_checkpointed_v3')
memory = replace(memory, 'memory_guard_training_launcher_v13', 'memory_guard_training_launcher_v14')
memory = replace(memory, PINS['train_lora_batched_data_v9.py'], created['train_lora_batched_data_v10.py'])
memory = replace(memory, PINS['launch_with_cache_reclaim_v10.py'], created['launch_with_cache_reclaim_v11.py'])
memory = replace(memory, '24e6f01a63c482d8451d72af9e21994332c0d0be53cbefbdf4c295f659090fd5', PINS['native_tdt_checkpointed_v3.py'])
memory = replace(memory, 'NEW data-v3 process:9GB allocator cap with native per-clip reclamation,4GB cache cleanup and owned-worker telemetry.', 'NEW data-v3 process:declared7/8/9GB allocator cap with split-native reclamation,4GB cache cleanup and owned-worker telemetry.', 1)
memory = replace(memory, "if cap_gb != 9.:\n        raise ValueError('This reviewed version requires exactly 9GB allocator cap')", "if cap_gb not in (7., 8., 9.):\n        raise ValueError('This version permits explicit7/8/9GB caps; root must select a hardware-validated cap')", 1)
memory = replace(memory, "if args.cap_gb != 9. or args.threshold_gb != 4.:\n        raise ValueError('Reviewed policy is cap9GB; optimizer-end cache4GB unchanged')", "if args.cap_gb not in (7., 8., 9.) or args.threshold_gb != 4.:\n        raise ValueError('Explicit7/8/9GB cap required; optimizer-end cache4GB unchanged')", 1)
memory = replace(memory, 'Allocator cap9GB; original endcleanup4GB/criticalWindowsguards unchanged; nativeadapter2 owns perclip inactivecache cleanup', 'Declared allocator cap7/8/9GB; original endcleanup4GB/criticalWindowsguards unchanged; nativeadapter3 splits operands before FP32 and owns perclip inactivecache cleanup', 1)
memory = replace(memory, 'run:539', 'run:'+str(optimizer_sync_line))
save('launch_with_memory_guard_v14.py', memory)

outer = (OUT/'launch_native_training_v3.py').read_text(encoding='utf-8')
outer = replace(outer, 'train_lora_batched_data_v9.py', 'train_lora_batched_data_v10.py')
outer = replace(outer, 'launch_with_memory_guard_v13.py', 'launch_with_memory_guard_v14.py')
outer = replace(outer, 'native memory13', 'native memory14')
outer = replace(outer, 'native_tdt_checkpointed_v2.py', 'native_tdt_checkpointed_v3.py')
outer = replace(outer, '24e6f01a63c482d8451d72af9e21994332c0d0be53cbefbdf4c295f659090fd5', PINS['native_tdt_checkpointed_v3.py'])
outer = replace(outer, 'native_tdt_training_backend_v2', 'native_tdt_training_backend_v3')
outer = replace(outer, 'native_tdt_training_backend_completion_v2', 'native_tdt_training_backend_completion_v3')
outer = replace(outer, "option(argv[:argv.index('--')],'--cap-gb') =='9'", "option(argv[:argv.index('--')],'--cap-gb') in ('7','8','9')", 1)
outer = replace(outer, 'Retain the hardware-verified9GB/4GB memory policy', 'Explicit root-selected7/8/9GB cap and unchanged4GB endcleanup required; efficacy needs the chosen hardware proof', 1)
save('launch_native_training_v4.py', outer)

receipt = {
    'schema':1,'kind':'native_split_runtime_deployment_readiness_v1','CPU_only':True,
    'Torch_or_NeMo_imported':False,'GPU_execution':False,'frozen_originals_sha256':PINS,
    'new_sources_sha256':created,'actual_GPU_resource_and_gradient_validation_pending':True,
    'allowed_declared_allocator_caps_gb':[7,8,9],'end_cache_threshold_gb':4,
    'exact_resume_receipt_contract':{
        'kind':'native_runtime_state_transition_v2','status':'CPU_prepared','passed':True,
        'target_run_id':'NEW fork run ID','source_run_id':'parent extension run ID',
        'source_step':358,'inherited_selected_step':'exact saved selected_step',
        'source_config_identity_sha256':'canonical JSON SHA of loaded source config_identity',
        'fork_latest':{'path':'immutable full-state fork latest.pt','sha256':'actual full-file SHA','step':358},
        'source_run_config':{'path':'immutable source run_config.json','sha256':'actual full-file SHA'},
        'approved_source_changes':{
            old:{'new_filename':new,'old_sha256':sha(OUT/old),'new_sha256':sha(OUT/new)}
            for old,new in [('train_lora_batched_data_v9.py','train_lora_batched_data_v10.py'),('native_tdt_checkpointed_v2.py','native_tdt_checkpointed_v3.py'),('launch_native_training_v3.py','launch_native_training_v4.py')]},
        'native_backend_transition':{'old_path':'immutable source ready file','old_sha256':'actual file SHA','new_path':'new outer4 ready file','expected_new_kind':'native_tdt_training_backend_v3'},
        'declared_allocator_policy':{'cap_gb':'root-selected7/8/9 integer','end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
    },
    'optimizer_end_synchronize_source_line':optimizer_sync_line,
    'trainer_inactive_cuda_cache_cleanup':json.loads(cleanup_identity.replace('True','true')),
    'unchanged_scientific_scope':'Same original loss kernel/full label normalization, sigma0/omega0, per-target averaging, accepted data/IDs/blank/durations, LR/KD/LoRA, optimizer/sampler/RNG/history. Only source routing, split-before-FP32 operand representation, post-microbatch/Adam inactive cache reclamation and declared per-process allocator cap change.'
}
path = OUT/'native_split_runtime_deployment_readiness_v1.json'
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(created,indent=2))
