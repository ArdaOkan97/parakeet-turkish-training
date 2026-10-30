"""Create a separate reviewed v4 probe; never modify the measured v3 source."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
directory = ROOT/'outputs/parakeet_turkish'
old = directory/'profile_cuda_cache_reclaim_v3.py'
if hashlib.sha256(old.read_bytes()).hexdigest() != '53c4ef0a4cb40e4e6a19b0f33fdc7b452360c030a1b848aabfdf94122b8ededc':
    raise ValueError('Frozen v3 differs')
text = old.read_text(encoding='utf-8')
def once(left, right):
    global text
    if text.count(left) != 1: raise ValueError('Expected unique source span: '+left[:80])
    text = text.replace(left, right)

once('Only the intervention clears unused CUDA cache at synchronized update end above\n6GB.',
     'Control uses the preceding13GB-cap/6GB-cleanup policy; intervention uses9GB/4GB.\nBoth branches collect actual self-PID host telemetry and enforce the new commit guard.')
once('from train_lora_batched import BalancedBucketSampler, ids_sha',
     "from train_lora_batched import BalancedBucketSampler, ids_sha\nfrom resource_watchdog_v1 import ResourceGuard, sample_memory, POLICY as RESOURCE_POLICY\nfrom launch_with_memory_guard_v3 import creation_identity, validate_capacity")
start = text.index('def system_memory():')
end = text.index('\n\ndef driver_memory():', start)
text = text[:start] + '''def memory_sample(phase):
    value=sample_memory(os.getpid(),phase)
    system=value['system_memory']
    if system.get('available'):
        system['committed_bytes']=system['commit_used_bytes']  # legacy report alias
    return value


def system_memory():
    return memory_sample('probe_auxiliary_read')['system_memory']
''' + text[end:]
start = text.index('def safety_reason(')
end = text.index('\n\ndef summarize_snapshot(',start)
text = text[:start] + '''def safety_reason(memory,reserved_bytes,*,resource_guard=None,observed_monotonic=None,
                  worker_private_memory=None,reserved_limit=12_000_000_000):
    guard=ResourceGuard() if resource_guard is None else resource_guard
    reason,_=guard.observe(memory,worker_private_memory,observed_monotonic=observed_monotonic)
    if reason:return reason
    if reserved_bytes>=reserved_limit:return 'allocator_reservation_guard'
    return None
''' + text[end:]
once("'probe_sha256':sha(__file__),'helper_sha256':sha(OUTPUT/'cuda_cache_reclaim_v1.py'),",
     "'probe_sha256':sha(__file__),'resource_helper_sha256':sha(OUTPUT/'resource_watchdog_v1.py'),\n              'memory_launcher_v3_sha256':sha(OUTPUT/'launch_with_memory_guard_v3.py'),\n              'helper_sha256':sha(OUTPUT/'cuda_cache_reclaim_v1.py'),")
once("'probe_version':3,'cleanup_threshold_gb':6.", "'probe_version':4,'cleanup_threshold_gb':4.")
once("'maximum_reserved_gb_guard':12.,'allocator_hard_cap_gb':13.,'maximum_commit_fraction':.90,",
     "'maximum_reserved_gb_guard':12.,'allocator_hard_cap_gb':9.,'maximum_commit_fraction':.95,\n            'resource_policy':dict(RESOURCE_POLICY),\n            'branch_policies':{'control':{'allocator_hard_cap_gb':13.,'cleanup_threshold_gb':6.},\n                               'reclaim':{'allocator_hard_cap_gb':9.,'cleanup_threshold_gb':4.}},\n            'actual_worker_pid':os.getpid(),'actual_worker_ppid':os.getppid(),\n            'actual_worker_creation_identity':creation_identity(os.getpid()),")
once("'safety_cap_limit':'Both branches use a13GB per-process allocator cap; this can trigger native automatic reclaim/allocation retries. Control is a guarded reproduction, not the untouched31GB fallback run. num_alloc_retries is recorded.',",
     "'safety_cap_limit':'Control13GB and intervention9GB per-process allocator caps can trigger native autoreclaim/OOM; control is the preceding6GB-cleanup policy, not the untouched31GB fallback run. Allocation retries/OOMs and critical/sustained commit guards are recorded.',")
once("'harness_cleanup_policy':'After full Adam/newrow update+synchronize, empty unused cache above6GB; timed overhead included.',",
     "'harness_cleanup_policy':'After full Adam/newrow update+synchronize, control clears unused cache above6GB; intervention above4GB; timing includes cleanup.',")
once('    memory=system_memory()\n    if safety_reason(memory,0):raise RuntimeError(\'System already above conservative commit guard\')',
     "    initial_sample=memory_sample('before_GPU_initialization')\n    if safety_reason(initial_sample['system_memory'],0,observed_monotonic=initial_sample['observed_monotonic']):\n        raise RuntimeError('System already above the new critical/telemetry guard')\n    report['owned_memory_before_GPU_initialization']=initial_sample")
once('    capacity=torch.cuda.get_device_properties(0).total_memory\n    torch.cuda.set_per_process_memory_fraction(min(13e9/capacity,.98))',
     '    capacity=torch.cuda.get_device_properties(0).total_memory\n    validate_capacity(capacity,9.)\n    torch.cuda.set_per_process_memory_fraction(13e9/capacity)')
once("        for policy,requested in [('control',args.control_updates),('reclaim',args.intervention_updates)]:\n            optimizer.zero_grad",
     "        for policy,requested in [('control',args.control_updates),('reclaim',args.intervention_updates)]:\n            branch_cap_gb=13. if policy=='control' else 9.\n            cleanup_threshold_bytes=6_000_000_000 if policy=='control' else 4_000_000_000\n            resource_guard=ResourceGuard()\n            optimizer.zero_grad")
once("            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()",
     "            torch.cuda.synchronize();torch.cuda.empty_cache()\n            torch.cuda.set_per_process_memory_fraction(branch_cap_gb*1e9/capacity)\n            torch.cuda.reset_peak_memory_stats()")
once("            branch={'requested_updates':requested,'updates':[],'stop_reason':None,'initial_allocator':allocator_stats(),\n                    'initial_system':system_memory(),'initial_driver':driver_memory(),'snapshots':[]}",
     "            initial_owned_sample=memory_sample('branch_'+policy+'_initial')\n            branch={'requested_updates':requested,'updates':[],'stop_reason':None,'initial_allocator':allocator_stats(),\n                    'initial_system':initial_owned_sample['system_memory'],\n                    'initial_owned_memory_telemetry':initial_owned_sample,\n                    'allocator_hard_cap_gb':branch_cap_gb,'cleanup_threshold_gb':cleanup_threshold_bytes/1e9,\n                    'initial_driver':driver_memory(),'snapshots':[],'capture_memory_events':[]}")
once("            def capture():\n                return {'gradients':",
     "            def capture():\n                captured={'gradients':")
once("                        'optimizer_sha256':tree_sha(optimizer.state_dict())}\n            snapshot('initial')",
     "                        'optimizer_sha256':tree_sha(optimizer.state_dict())}\n                branch['capture_memory_events'].append(memory_sample('branch_'+policy+'_after_grad_weight_capture'))\n                return captured\n            snapshot('initial')")
once("                before_memory=system_memory();before_stats=allocator_stats()\n                reason=safety_reason(before_memory,before_stats['reserved_bytes.all.current'])",
     "                before_sample=memory_sample('branch_'+policy+'_before_update');before_memory=before_sample['system_memory'];before_stats=allocator_stats()\n                reason=safety_reason(before_memory,before_stats['reserved_bytes.all.current'],resource_guard=resource_guard,\n                    observed_monotonic=before_sample['observed_monotonic'],worker_private_memory=before_sample['worker_private_memory'])")
once("                if policy=='reclaim' and pre_cleanup['reserved_bytes.all.current']>6_000_000_000:",
     "                if pre_cleanup['reserved_bytes.all.current']>cleanup_threshold_bytes:")
once("                stats=allocator_stats();seconds=time.perf_counter()-began\n                row=", 
     "                stats=allocator_stats();seconds=time.perf_counter()-began\n                post_sample=memory_sample('branch_'+policy+'_after_update_and_cache_cleanup')\n                row=")
once("                    'system':system_memory()}",
     "                    'system':post_sample['system_memory'],'owned_memory_telemetry':post_sample,\n                    'before_owned_memory_telemetry':before_sample}")
once("                reason=safety_reason(row['system'],stats['reserved_bytes.all.current'])",
     "                reason=safety_reason(row['system'],stats['reserved_bytes.all.current'],resource_guard=resource_guard,\n                    observed_monotonic=post_sample['observed_monotonic'],worker_private_memory=post_sample['worker_private_memory'])\n                row['moderate_commit_streak_after_observation']=resource_guard.moderate_streak")
once("            branch.update(completed_updates=completed,final_driver=driver_memory(),final_system=system_memory(),",
     "            final_owned_sample=memory_sample('branch_'+policy+'_final')\n            branch.update(completed_updates=completed,final_driver=driver_memory(),final_system=final_owned_sample['system_memory'],\n                    final_owned_memory_telemetry=final_owned_sample,")
once("            production_hook_validation_pending=True,", "            production_hook_validation_pending=True,\n            actual_self_PID_private_telemetry_recorded=True,")
once("    run_id='adaptive_20261010_recovery_b8_01_b8_quality'", "    run_id='adaptive_20261010_continuation_02_resume_B8'")
once("default=OUTPUT/'cuda_cache_reclaim_gpu_probe_v3.json'", "default=OUTPUT/'cuda_cache_reclaim_gpu_probe_v4.json'")
once("value.add_argument('--control-updates',type=int,default=64)", "value.add_argument('--control-updates',type=int,default=10)")
target=directory/'profile_cuda_cache_reclaim_v4.py'
if target.exists():raise FileExistsError(target)
target.write_text(text,encoding='utf-8',newline='\n')
print(hashlib.sha256(target.read_bytes()).hexdigest())
