"""Independent small JSON-only guards for the new continuation audit."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
import verify_continued_loop_results_v1 as audit
import launch_with_memory_guard_v2 as guard


def main():
    destination = OUTPUT / 'continued_loop_memory_evidence_cpu_verification_v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    before = audit.file_sha(audit.__file__)
    names = []
    positive_checks = 0
    cases = []
    with tempfile.TemporaryDirectory(prefix='continued_memory_guard_', dir=ROOT/'work') as temporary:
        directory = Path(temporary)
        cache_path = directory/'cache.json'
        io_path = directory/'io.json'
        memory_path = directory/'memory.json'
        io_path.write_text('{}', encoding='utf-8')
        for name, value in (
            ('launch_with_memory_guard_v2.py', 'memory_launcher_sha256'),
            ('launch_with_cache_reclaim_v1.py', 'cache_launcher_sha256'),
            ('cuda_cache_reclaim_v1.py', 'cache_helper_sha256')):
            names.append((name, value))
        memory = {'kind':'memory_guard_training_launcher_v2', 'status':'complete',
            'cap_bytes':13_000_000_000, 'allocator_cap_applied':True,
            'cleanup_threshold_bytes':6_000_000_000,
            'physical_cuda_device_bytes':17_179_344_896,
            'per_process_allocator_fraction':13_000_000_000/17_179_344_896,
            'cache_provenance_path':str(cache_path), 'io_provenance_path':str(io_path),
            **{key:False for key in ('global_driver_settings_changed','training_math_changed',
                'backend_flags_changed','RNG_sampler_optimizer_changed')}}
        for name, key in names:
            memory[key] = memory[key+'_after'] = audit.file_sha(OUTPUT/name)
        cache = {'status':'complete','active_process_attachment':False}
        for before_key, after_key, name in (
            ('launcher_sha256','launcher_sha256_after','launch_with_cache_reclaim_v1.py'),
            ('cache_helper_sha256','cache_helper_sha256_after','cuda_cache_reclaim_v1.py'),
            ('frozen_io_launcher_sha256','frozen_io_launcher_sha256_after','launch_with_io_retry_v1.py'),
            ('frozen_io_helper_sha256','frozen_io_helper_sha256_after','atomic_io_retry_v1.py')):
            cache[before_key] = cache[after_key] = audit.file_sha(OUTPUT/name)
        worker = audit.file_sha(OUTPUT/'train_lora_batched.py')
        cache['cache_reclamation'] = {'worker_sha256':worker,'worker_sha256_after':worker,
            'caller_function':'run','caller_line':539,'threshold_reserved_bytes':6_000_000_000,
            'eligible_update_calls':3,'failed_reclamations':0,'reclamations':1,'events_omitted':0,
            'cleanup_seconds_in_training_timing':.025,
            **{key:False for key in ('changes_training_math','changes_backend_flags',
                'changes_rng_or_sampler','changes_optimizer_state')},
            'events':[{'status':'complete','before':{'reserved_bytes':7_000_000_000,
                'allocated_bytes':3_000_000_000},'after':{'reserved_bytes':3_250_000_000,
                'allocated_bytes':3_000_000_000}}]}

        def run_memory(value):
            memory_path.write_text(json.dumps(value), encoding='utf-8')
            entry={'cache_evidence':{'memory_guard_provenance_path':str(memory_path),
                'memory_guard_provenance_sha256':audit.file_sha(memory_path)},
                'cache_provenance_path':str(cache_path),'io_provenance_path':str(io_path)}
            check=audit.CompactChecks()
            audit.validate_memory_evidence(entry,check)
            return check.count

        def run_cache(value):
            cache_path.write_text(json.dumps(value), encoding='utf-8')
            check=audit.CompactChecks()
            audit.validate_cache_evidence(cache_path,check,completed=910,initial_step=907)
            return check.count

        positive_checks += run_memory(memory)
        positive_checks += run_cache(cache)
        cases.append('Valid v2 memory proof and three-update cache proof accepted')
        bad_memory = [('kind','memory_guard_training_launcher_v1'),
            ('cleanup_threshold_bytes',8_000_000_000), ('cap_bytes',12_000_000_000),
            ('allocator_cap_applied',False), ('status','running'),
            ('physical_cuda_device_bytes',24_000_000_000),
            ('per_process_allocator_fraction',.75), ('backend_flags_changed',True),
            ('memory_launcher_sha256_after','0'*64), ('cache_helper_sha256','0'*64),
            ('cache_provenance_path',str(directory/'wrong.json'))]
        for key, bad in bad_memory:
            value=copy.deepcopy(memory);value[key]=bad
            try:
                run_memory(value)
            except audit.AuditError:
                cases.append('Rejected memory '+key)
            else:
                raise AssertionError('Accepted corrupt memory '+key)
        bad_cache = [('threshold_reserved_bytes',8_000_000_000),('eligible_update_calls',2),
            ('failed_reclamations',1),('caller_line',538),('changes_optimizer_state',True),
            ('worker_sha256_after','0'*64)]
        for key, bad in bad_cache:
            value=copy.deepcopy(cache);value['cache_reclamation'][key]=bad
            try:
                run_cache(value)
            except audit.AuditError:
                cases.append('Rejected cache '+key)
            else:
                raise AssertionError('Accepted corrupt cache '+key)
        for label, mutate in (
            ('release below trigger',lambda event:event['before'].update(reserved_bytes=6_000_000_000)),
            ('live tensor allocation changed',lambda event:event['after'].update(allocated_bytes=2_999_999_999)),
            ('reservation increased',lambda event:event['after'].update(reserved_bytes=7_000_000_001)),
            ('failed release',lambda event:event.update(status='failed'))):
            value=copy.deepcopy(cache);mutate(value['cache_reclamation']['events'][0])
            try:
                run_cache(value)
            except audit.AuditError:
                cases.append('Rejected cache event '+label)
            else:
                raise AssertionError('Accepted corrupt cache event '+label)
        cap=guard.validate_capacity(17_179_344_896,13)
        assert cap['cap_bytes']==13_000_000_000 and cap['per_process_allocator_fraction']==memory['per_process_allocator_fraction']
        cases.append('Frozen v2 exact decimal cap independently recomputed')
        for capacity, value in ((8e9,13),(24e9,13),(17_179_344_896,6),(17_179_344_896,13.01)):
            try:
                guard.validate_capacity(capacity,value)
            except ValueError:
                cases.append('Rejected unreviewed device/cap '+str((capacity,value)))
            else:
                raise AssertionError('Accepted unsupported capacity')
    assert audit.file_sha(audit.__file__)==before
    import torch
    assert not torch.cuda.is_initialized()
    result={'schema':1,'passed':True,'audit_source_sha256':before,
        'memory_wrapper_sha256':audit.file_sha(guard.__file__),
        'positive_assertions':positive_checks,'case_count':len(cases),
        'corruptions_rejected':len(bad_memory)+len(bad_cache)+4,
        'cases':cases,'CUDA_initialized':False,
        'scope':'CPU-only temporary JSON fixtures and real immutable source hashes; no allocation/cap API, training, GPU inference, or old artifact writes.'}
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('passed','positive_assertions','case_count','corruptions_rejected','audit_source_sha256')}))


if __name__=='__main__':
    main()
