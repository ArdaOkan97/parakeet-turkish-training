from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'work/research/test_native_split_chain_cpu_v1.py').read_text(encoding='utf-8')
start=source.index("oldtree=tree(");end=source.index('# Execute only exact native exporter',start)
source=source[:start]+'''oldtree=tree('train_lora_batched_data_v10.py');newtree=tree('train_lora_batched_data_v11.py')
def cleanrun(tree):
    return ast.Module(body=[n for n in function(tree,'run').body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source_names' for t in n.targets))],type_ignores=[])
check('run_math_sampler_optimizer_eval_save_AST_identical',ast.dump(cleanrun(oldtree))==ast.dump(cleanrun(newtree)))
for name in ('save_training_state','parser','validate_arguments','BalancedBucketSampler','native_loss_identity','native_loss_function'):
    selector=lambda tree:next(n for n in tree.body if getattr(n,'name',None)==name)
    check(name+'_AST_identical',ast.dump(selector(oldtree))==ast.dump(selector(newtree)))
''' +source[end:]
for old,new in (('train_lora_batched_data_v10','train_lora_batched_data_v11'),('launch_with_io_retry_v8','launch_with_io_retry_v9'),
    ('launch_with_cache_reclaim_v11','launch_with_cache_reclaim_v12'),('launch_with_memory_guard_v14','launch_with_memory_guard_v15'),
    ('launch_native_training_v4','launch_native_training_v5')):
    # Preserve the one original AST comparison path explicitly afterward.
    source=source.replace(old,new)
source=source.replace("oldtree=tree('train_lora_batched_data_v11.py')","oldtree=tree('train_lora_batched_data_v10.py')")
source=source.replace("cap_gb=7.","cap_gb=8.").replace("me['cap_bytes']==7e9","me['cap_bytes']==8e9")
start=source.index('    # Exact migration uses actual source config');end=source.index('    outer_ready=',start)
new='''    # Use the actual closed862 source identity with a tiny synthetic state file.
    receipt=json.loads((OUT/'runs/adaptive_20261011_hf_return_window_01/runtime_identity_continuation_v1.json').read_text(encoding='utf-8'))
    parent=json.loads(Path(receipt['source_run_config']['path']).read_text(encoding='utf-8'))
    old=parent['config_identity'];fork=temporary/'fork_latest.pt';fork.write_bytes(b'CPU_full_state_identity_fixture')
    receipt_path=temporary/'runtime_identity.json';new_ready=temporary/'migration_ready.json'
    policy={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}
    ready_value['declared_allocator_policy']=policy;new_ready.write_text(json.dumps(ready_value),encoding='utf-8')
    receipt['target_run_id']=run_id
    receipt['fork_latest']={'path':str(fork),'sha256':sha(fork),'step':862}
    receipt['native_backend_transition']['new_path']=str(new_ready)
    receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
    current=copy.deepcopy(old)
    for oldname,item in receipt['approved_source_changes'].items():
        current['source_sha256'].pop(oldname);current['source_sha256'][item['new_filename']]=item['new_sha256']
    added=receipt['added_identity_validator'];current['source_sha256'][added['filename']]=added['sha256']
    current['input_sha256'].pop(str(Path(receipt['replaced_previous_migration']['path']).resolve()))
    current['input_sha256'].pop(str(Path(parent['native_backend_provenance']).resolve()))
    current['input_sha256'][str(new_ready.resolve())]=sha(new_ready)
    current['input_sha256'][str(receipt_path.resolve())]=sha(receipt_path)
    current['native_loss']['native_backend_provenance_sha256']=sha(new_ready)
    args=types.SimpleNamespace(run_id=run_id,tdt_implementation='nemo',resume_exact=fork,resume_runtime_migration=receipt_path,native_backend_provenance=new_ready,native_max_joint_bytes=1700000000)
    state={'step':862,'selected_step':0,'config_identity':old}
    check('actual_checker_accepts_only_identity_paths_and_routing_changes',worker.exact_resume_identity_matches(state,args,current))
    for key,value in [('head_lr',.0001),('replay_fraction',.25),('rank',16),('schedule','cosine')]:
        bad=copy.deepcopy(current);bad['training_arguments'][key]=value
        try:worker.exact_resume_identity_matches(state,args,bad)
        except ValueError:checks.append('exact_migration_rejects_changed_'+key)
        else:raise AssertionError('Scientific change silently accepted')
    bad=copy.deepcopy(current);bad['native_loss']['sigma']=.02
    try:worker.exact_resume_identity_matches(state,args,bad)
    except ValueError:checks.append('exact_migration_rejects_changed_native_sigma')
    else:raise AssertionError('Native loss changed')
'''
source=source[:start]+new+source[end:]
source=source.replace('native_split_chain_stdlib_dispatch_cpu_verification_v1.json','HF_862_chain_stdlib_dispatch_cpu_verification_v1.json')
with (ROOT/'work/research/test_hf862_dispatch_cpu_v1.py').open('x',encoding='utf-8') as f:f.write(source)
