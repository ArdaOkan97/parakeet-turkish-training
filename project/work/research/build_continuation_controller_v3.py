from pathlib import Path

base = Path(__file__).resolve().parents[2] / 'outputs/parakeet_turkish'
target = base / 'run_experiment_loop_continuation_v3.py'
if target.exists():
    raise FileExistsError(target)
source = (base / 'run_experiment_loop_continuation_v2.py').read_text(encoding='utf-8')
source = source.replace("PRIOR_LOOP='adaptive_20261010_continuation_01'", "PRIOR_LOOP='adaptive_20261010_continuation_02'")
source = source.replace('SOURCE_STEP=1773', 'SOURCE_STEP=1821').replace('SOURCE_BEST_STEP=1773', 'SOURCE_BEST_STEP=1821')
source = source.replace("d2dc4fed6a4f0a105ac981ba1759a95656636f3e177ddaedac44ef55b8d278a3", "54aeab34e637e4aed5a572a80d1f3c6479069f9ce03644e14926300ee4a581b6")
source = source.replace(" 'resource_watchdog_v1.py':", " 'run_experiment_loop_continuation_v2.py':'7c0f689c6886f8cbd221acd7e8d6b0e9ab9b9b0157ffe25150bbe1bbe39e8f58',\n 'launch_with_memory_guard_v3.py':'28e51998bbb481f89a1106ad55fa6579f6ce7475826a8229d615ff973dccc2d1',\n 'resource_watchdog_v1.py':", 1)
source = source.replace("('fork_training_state_v2.py','resource_watchdog_v1.py'", "('fork_training_state_v3.py','resource_watchdog_v1.py'")
start = source.index('def validate_cache_gpu(')
end = source.index('\ndef validate_entry(', start)
source = source[:start] + '''def validate_cache_gpu(path, config):
 # Historical907 13GB/6GB causal and production-hook evidence remains historical.
 # The new9GB/4GB policy requires its own measured validation below.
 import run_experiment_loop_continuation_v2 as old
 return old.validate_cache_gpu(path, config)

''' + source[end:]
source = source.replace('full1773', 'full1821').replace('1773/latest and1773/best', '1821/latest and1821/best')
source = source.replace('exact1773/1773', 'exact1821/1821').replace('through1773', 'through1821').replace('exact1773optimizer', 'exact1821optimizer')
source = source.replace("receipt.get('root_owned_exec_session')==7641", "receipt.get('root_owned_exec_session')==3634")
source = source.replace('Parent-owned7641', 'Parent-owned3634')
source = source.replace("prior_status.get('resource_stop_reason')=='system_commit_above90percent'", "prior_status.get('resource_stop_reason')=='system_commit_at_or_above95percent'")
source = source.replace("'exact_training_state_run_id_fork_v2'", "'exact_training_state_run_id_fork_v3'")
source = source.replace('args.cache_threshold_gb==6', 'args.cache_threshold_gb==4')
source = source.replace("validation=validate_cache_gpu(args.cache_validation,config)", "validation=validate_cache_gpu(args.cache_validation,config)\n policy_validation=validate_memory_policy(args.memory_policy_validation,config)")
source = source.replace("'cache_validation':validation,'fork_manifest_path'", "'cache_validation':validation,'memory_policy_validation':policy_validation,'fork_manifest_path'")
source = source.replace("'prior_runtime_cache_evidence':cache_evidence(", "'prior_runtime_cache_evidence':historical_cache_evidence(")
source = source.replace("Path(receipt['runtime_provenance']['io_provenance_path']['path']),SOURCE_RUN,907,", "Path(receipt['runtime_provenance']['io_provenance_path']['path']),SOURCE_RUN,1773,")
source = source.replace("'continuation_v2_preflight.json'", "'continuation_v3_preflight.json'")
source = source.replace("'renewed_finite_B8_continuation_v2'", "'renewed_finite_B8_continuation_v3'")
source = source.replace("'threshold_reserved_gb':6,'allocator_cap_gb':13", "'threshold_reserved_gb':4,'allocator_cap_gb':9")
source = source.replace("FROZEN['launch_with_memory_guard_v2.py'],'strict_BF16", "FROZEN['launch_with_memory_guard_v3.py'],'strict_BF16")
source = source.replace("str(OUTPUT/'launch_with_memory_guard_v2.py')", "str(OUTPUT/'launch_with_memory_guard_v3.py')")
source = source.replace("'--cap-gb','13','--threshold-gb','6','--',*argv", "'--worker-identity',str(identity_path),'--cap-gb','9','--threshold-gb','4','--',*argv")
source = source.replace("memory_path=loop_dir/(run_id+'_memory_guard_provenance.json')", "memory_path=loop_dir/(run_id+'_memory_guard_provenance.json')\n   identity_path=loop_dir/(run_id+'_worker_identity.json')")
source = source.replace("'parameters':{k:source_changes.get", "'worker_identity_path':str(identity_path),'worker_arguments_sha256':argument_sha(argv),\n     'parameters':{k:source_changes.get")
source = source.replace("snapshot=sample_memory(child.pid,phase)", "worker_identity=read_worker_identity(identity_path,run_id,argv,child.pid,entry['started_at_utc'])\n      actual_pid=worker_identity['pid'] if worker_identity is not None else None\n      if worker_identity is not None:\n       entry.update(actual_worker_pid=actual_pid,worker_identity_sha256=sha(identity_path),launcher_pid=child.pid)\n      snapshot=sample_memory(actual_pid,phase)\n      snapshot['worker_private_memory'].update(attribution='validated_self_reported_frozen_worker' if actual_pid else 'worker_identity_not_yet_available',launcher_pid=child.pid)")
source = source.replace("p.add_argument('--loop-id',default='adaptive_20261010_continuation_02')", "p.add_argument('--loop-id',default='adaptive_20261010_continuation_03')")
source = source.replace("p.add_argument('--cache-validation',type=Path,required=True)", "p.add_argument('--cache-validation',type=Path,required=True)\n p.add_argument('--memory-policy-validation',type=Path,required=True)")
source = source.replace("p.add_argument('--cache-threshold-gb',type=float,default=6.)", "p.add_argument('--cache-threshold-gb',type=float,default=4.)")
source = source.replace("p.add_argument('--audit-worker',default='verify_continued_loop_results_v2.py')", "p.add_argument('--audit-worker',default='verify_continued_loop_results_v3.py')")
start = source.index('def cache_evidence(')
end = source.index('\ndef remaining_for_child(',start)
source = source[:start] + '''def historical_cache_evidence(path,io_path,run_id,start_step,memory_path=None):
 import run_experiment_loop_continuation_v2 as old
 return old.cache_evidence(path,io_path,run_id,start_step,memory_path)

''' + source[end:]
target.write_text(source, encoding='utf-8')
print(target)
