"""Explicit lifetime-bounded orchestrator; original7 owns actual leaf execution.

Full original source, labels, geometry, frontend and provisional row protocols.
This parent owns reference and frontend handles; per-update reports never
pretend that a still-live aggregate worker has exited.
"""
import argparse
from copy import deepcopy
import gc
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastmss_BPE9223_cpu_common_v12 as core
import fastmss_owned_process_closure_v2 as process
from asr18_owned_family_observation_v4 import (owned_family_sample,
    wait_successful_launcher_after_disappearance, terminate_known_family, wait_captured_dead)
import produce_FastMSS_BPE9223_reference_CPU_v12 as reference
import fastmss_BPE9223_feature_protocol_v12 as public

import run_ASR18_BPE9223_CPU_leaf_observer_v12 as legacy_leaf_executor
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility

KIND = 'ASR18_actual_BPE9223_recorded_room_merged_CPU_task_window_v1'
PROPOSAL = 'ASR18_BPE9223_aggregate_CPU_window_configuration_v1'
OWNER = '/root/cpu_data_preparation'
PARENT_BYTES = 128_000_000
SHIM_BYTES = 32 * 1024**2
LEAF_BYTES = 1_800_000_000
INNER_BYTES = PARENT_BYTES + SHIM_BYTES + LEAF_BYTES
FAMILY_BYTES = INNER_BYTES + SHIM_BYTES
LOGICAL_BYTES = 16 * 1024**2
PHYSICAL_BYTES = 64 * 1024**2
ALLOCATION = 512 * 1024
MAXIMUM_FILES = 64
MAXIMUM_DIRS = 12



class TaskFamilyMonitor:
    """Joins only this task's family observer; the main parent guard stays live."""
    def __init__(self,parent,folder):
        import threading
        self.parent=parent;self.folder=folder;self.active=None;self.done=threading.Event();self.error=None;self.count=0;self.peak=0;self.last=None
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        deadline=time.monotonic()+3
        while self.count<3:
            core.require(self.error is None and self.thread.is_alive() and time.monotonic()<deadline,'Live task-family observer');time.sleep(.02)
    def enter(self,child,launcher,known):
        self.check();core.require(self.active is None,'One actual CPU leaf at a time');self.active=(child,launcher,known)
    def leave(self,child,known):
        core.require(self.active is not None and self.active[0] is child and child.returncode==0,'Actual normal leaf handle0')
        core.require(all(process.dead(item) for item in list(known.values())),'All captured CPU leaf/shim identities dead')
        self.active=None;self.check()
    def _run(self):
        from resource_watchdog_v2 import sample_memory
        try:
            while not self.done.is_set():
                value=sample_memory(os.getpid(),'merged_CPU_task_family');own=value['worker_private_memory']
                core.require(own['available'] and own['private_bytes']<=PARENT_BYTES,'Actual merged parent128M')
                total=own['private_bytes'];active=self.active
                if active is not None:
                    for item in list(active[2].values()):
                        if process.dead(item):continue
                        child=sample_memory(item['pid'],'merged_CPU_actual_leaf');part=child['worker_private_memory']
                        if not part['available'] and process.dead(item):continue
                        core.require(part['available'],'Available actual live CPU family');total+=part['private_bytes']
                core.require(total<=FAMILY_BYTES,'Whole merged CPU family below1,995,108,864')
                self.peak=max(self.peak,total);self.count+=1;self.last=time.monotonic();self.done.wait(.05)
        except BaseException:
            self.error=__import__('traceback').format_exc()
    def check(self):
        if self.done.is_set():
            core.require(not self.thread.is_alive() and self.error is None and self.active is None and self.count>=3, 'Joined task observer/no CPU inflight; main parent guard remains fresh')
            return
        core.require(self.error is None and self.thread.is_alive() and self.last is not None and time.monotonic()-self.last<2,'Clean fresh live task-family observer')
    def fail_close(self):
        self.done.set();self.thread.join(2)
        active=self.active;retirement=None
        if active is not None:
            terminate_known_family(active[0],active[1],active[2])
            retirement=wait_captured_dead(active[2],timeout=5)
        core.require(not self.thread.is_alive(),'Failed task observer actually joined')
        path=self.folder/'task_family_monitor_failure.json'
        if not path.exists():core.write_new(path,{'kind':'ASR18_failed_merged_CPU_task_family_observer_v1',
            'source':core.record(__file__),'producer':core.identity(),'monitor_joined':True,
            'monitor_error':self.error,'normal_task_completion_claimed':False,'captured_CPU_family_retirement':retirement})

    def close(self,code):
        self.done.set();self.thread.join(2)
        core.require(not self.thread.is_alive() and self.error is None and self.active is None and self.count>=3,'Actual task observer joined; no CPU leaf inflight')
        core.write_new(self.folder/'task_family_monitor_completion.json',{'kind':'ASR18_actual_merged_CPU_task_family_monitor_joined_v1',
            'source':core.record(__file__),'producer':core.identity(),'task_return_code':code,'monitor_joined':True,'monitor_clean':True,
            'samples':self.count,'actual_sampled_whole_family_private_peak_bytes':self.peak,'CPU_leaf_inflight_at_join':False,
            'main_parent_monitor_stopped':False,'parent_process_exit0_claimed':False,'whole_allowance_bytes':FAMILY_BYTES})


def checked_task_completion(task_pin,report_pin,configuration_pin,compatibility_pin,*,guard):
    core.require(not (Path(task_pin['path']).parent/'task_family_monitor_failure.json').exists(),'Failed task marker forbids a clean completion')
    task=core.read_bound(task_pin,guard);report=core.read_bound(report_pin,guard);proposal=core.read_bound(configuration_pin,guard)
    review=core.read_bound(compatibility_pin,guard);original=core.read_bound(review['original_configuration'],guard)
    checked_compatibility(compatibility_pin,original['full_run_identity'],guard)
    core.require(task['kind']=='ASR18_completed_merged_CPU_task_no_inflight_v1' and task['passed'] is True
        and report['kind']==KIND and report['passed'] is True and report['worker_source']==task['source']
        and report['aggregate_process_exit0_claimed'] is False and report['merged_parent_remains_live'] is True
        and report['actual_optimizer_updates']==0 and report['sampler_journal_frontier_advanced'] is False
        and task['owner']=='/root/memory_diagnosis' and task['source']==core.record(__file__,guard)
        and task['source']==review['operational_sources']['CPU_orchestrator']
        and task['parent_source']==report['parent_source']==review['operational_sources']['parent']
        and task['configuration']==configuration_pin and task['report']==report_pin and report['proposal']==configuration_pin
        and task['operational_compatibility']==report['operational_compatibility']==compatibility_pin
        and task['task_nonce']==report['task_nonce'] and len(task['task_nonce'])==64
        and task['producer']==report['producer'] and task['owner_binding']==report['owner'] and task['manifest']==report['manifest']
        and task['ordered_prepared_payloads_sha256']==core.digest(report['complete_ordered_prepared_payloads'])
        and task['CPU_leaf_inflight_at_task_return'] is False and task['actual_all_leaf_processes_closed0'] is True
        and task['parent_process_exit0_claimed'] is False and task['parent_remains_live_at_task_return'] is True,
        'Only a typed completed live-parent CPU task, never aggregate process0')
    for field,role in [('actual_reference_process0','reference'),('actual_frontend_process0','frontend')]:
        proof=core.read_bound(task[field],guard);component=report[role+'_actual_aggregate0']
        core.require(component['actual0']==task[field] and proof['parent_source']==proposal['sources']['parent']
            ==core.record(legacy_leaf_executor.__file__,guard) and proof['worker_source']==proposal['sources'][role],
            'Actual new leaf observer/Popen source and exact numeric worker')
        process.verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner=OWNER,report=component['report'],guard=guard)
    monitor=core.read_bound(task['task_family_monitor'],guard)
    core.require(monitor['source']==task['source'] and monitor['producer']==task['producer'] and monitor['task_return_code']==0
        and monitor['monitor_joined'] is True and monitor['monitor_clean'] is True and monitor['samples']>=3
        and monitor['CPU_leaf_inflight_at_join'] is False and monitor['main_parent_monitor_stopped'] is False
        and monitor['parent_process_exit0_claimed'] is False and monitor['whole_allowance_bytes']==1995108864,
        'Actual joined task-family observer only; main supervisor lifetime stays charged')
    return task


def run_merged_task(proposal_path, ROOT_review, output, operational_compatibility, *, parent_guard, actual_parent_source):
    began = time.monotonic(); core.environment()
    import psutil
    core.require(psutil.Process().nice()==psutil.BELOW_NORMAL_PRIORITY_CLASS,'Existing admitted BelowNormal parent, no environment/priority mutation')
    proposal_pin = core.record(proposal_path); proposal = core.read_bound(proposal_pin)
    review_pin = core.record(ROOT_review); review = core.read_bound(review_pin, core.resource_guard)
    core.require(proposal['kind'] == PROPOSAL and proposal['production_enabled'] is False
        and proposal['native_benchmark_only'] is False and proposal['vocabulary'] == 9223
        and review['accepted'] is True and review['reviewer'] == '/root'
        and review['scope'] == 'BPE9223_recorded_noise_16room_bounded_CPU_window_v1',
        'Exact ROOT-reviewed generic9223 bounded CPU configuration')
    from asr18_BPE9223_source_context_v9 import verify_CPU_configuration, verify_CPU_prior_boundary
    reviewed_configuration = verify_CPU_configuration(proposal_pin, review_pin, guard=core.resource_guard)
    core.require(core.same(reviewed_configuration, review['configuration']), 'Exact reviewed direct configuration or authorized static template')
    root = legacy_leaf_executor.private_output(output)
    compatibility_pin = core.record(operational_compatibility)
    original = core.read_bound(core.read_bound(compatibility_pin)['original_configuration'])
    compatibility = checked_compatibility(compatibility_pin, original['full_run_identity'], guard=core.resource_guard)
    del original
    driver_pin = core.record(__file__); parent_pin = core.record(legacy_leaf_executor.__file__)
    core.require(core.same(compatibility['operational_sources']['CPU_orchestrator'], driver_pin)
        and core.same(compatibility['CPU_leaf_execution_delegate_source'], parent_pin)
        and Path(inspect.getsourcefile(legacy_leaf_executor.run_leaf)).resolve() == Path(parent_pin['path']),
        'Actual new main8 and explicitly delegated unchanged original7 Popen/observer')
    core.require(str(root) == proposal['output_root'] and not root.exists(),
        'Exact fresh private output, or separately authorized preserved reference-only continuation')
    sources = proposal['sources']
    core.require(core.same(sources['parent'], parent_pin)
        and core.same(sources['reference'], core.record(reference.__file__))
        and core.same(sources['core'], core.record(core.__file__))
        and type(proposal['freeze_after_setup']) is bool, 'Exact new finite sources and explicit GC mode')
    for pin in sources.values(): core.require(core.same(core.record(pin['path']), pin), 'Bound source differs')
    python_pin = proposal['python']; guard_pin = sources['guard']
    core.require(Path(python_pin['path']).resolve() == core.CPU_ENV / 'Scripts/python.exe'
        and core.same(core.record(python_pin['path']), python_pin)
        and core.same(guard_pin, core.record(PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py')),
        'Exact unchanged E CPU interpreter and guard3')
    root.mkdir(parents=True, exist_ok=False)
    monitor=None
    def guard():
        core.require(time.monotonic() - began <= 480, 'Bounded whole CPU window480s')
        value = parent_guard.check()
        if monitor is not None:monitor.check()
        core.require(value['worker_private_memory']['private_bytes'] <= PARENT_BYTES, 'One actual merged128M parent domain')
        return value
    contract_pin = proposal['source_contract']
    failure_pin = None
    reference_proof = None
    core.require(core.same(actual_parent_source,compatibility['operational_sources']['parent']), 'Actual merged native-owner source')
    task_returned=False
    try:
        monitor=TaskFamilyMonitor(parent_guard,root)
        task_nonce=__import__('secrets').token_hex(32)
        source, plan = public.verify_source_contract(contract_pin, guard=guard)
        numerical = core.read_bound(source['numerical_CPU_contract'], guard)
        rebuilt = core.bridge.bridge_plan(plan['coordinator_plan'], DATA_evidence=plan['DATA_evidence'],
            session_binding=plan['session'], window_index=plan['window_index'],
            prior_window_boundary=plan['prior_window_boundary'],
            verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard), guard=guard)
        core.require(core.same(rebuilt, plan), 'Independent source/frontier/phase/full-row revalidation before any reference audio')
        summary = {'phase': plan['phase'], 'window_index': plan['window_index'],
            'row_counts': [len(update['rows']) for update in plan['updates']]}
        del rebuilt, plan
        gc.collect()
        core.require(core.same(source['sources']['controller'], parent_pin)
            and core.same(source['sources']['reference'], sources['reference'])
            and core.same(source['sources']['frontend'], sources['frontend'])
            and core.same(source['sources']['core'], sources['core'])
            and core.same(source['owner_validation_only'], proposal['owner_validation_only'])
            and core.same(source['checkpoint_diagnostic_only'], proposal['checkpoint_diagnostic_only'])
            and source['freeze_after_setup'] == proposal['freeze_after_setup'], 'Exact reviewed source contract')
        outer_parent = proposal['outer_parent_source']
        core.require(core.same(outer_parent, numerical['outer_parent_source'])
            and core.same(core.record(outer_parent['path'], guard), outer_parent), 'Actual external native-owner source binding')
        core.require(core.same(outer_parent,actual_parent_source),'Truthful merged process parent, no code6 stamp')
        numerical_request = root / 'numerical_request.json'
        numerical_value = {'kind': core.REQUEST_KIND, 'production_enabled': False,
            'bridge': source['bridge'], 'contract': source['numerical_CPU_contract'], 'update_index': 0}
        if reference_proof is None: core.write_new(numerical_request, numerical_value)
        else: core.require(core.same(core.read_bound(core.record(numerical_request, guard), guard), numerical_value),
            'Preserved numerical request unchanged; no overwrite')
        common_request = {'kind': public.REQUEST, 'mode': public.MODE, 'production_enabled': False,
            'freeze_after_setup': proposal['freeze_after_setup'], 'source_contract': contract_pin,
            'parent_source': parent_pin, 'bridge': source['bridge'], 'numerical_request': core.record(numerical_request, guard),
            'output_root': str(root), 'ROOT_review': review_pin, 'configuration': reviewed_configuration,
            'window_configuration': proposal_pin}
        target = root / 'reference_request.json'
        if reference_proof is None:
            core.write_new(target, common_request)
            _, reference_proof = legacy_leaf_executor.run_leaf('reference', core.record(target, guard), sources['reference'], root,
                                          python_pin, guard_pin, parent_pin, guard, task_monitor=monitor)
        else:
            core.require(core.same(core.read_bound(core.record(target, guard), guard), common_request),
                'Exact original source-bound reference request reused read only')
        legacy_leaf_executor.inventory(root)
        public.verify_reference_window(reference_proof['report'], reference_proof['actual0'], contract_pin=contract_pin, guard=guard)
        gc.collect()
        owner = public.create_window(contract_pin, window_id=proposal['window_id'], frontend=numerical['frontend'],
            guard=guard, verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard))
        frontend_request = {**common_request, 'owner': owner, 'reference_window_report': reference_proof['report'],
                            'reference_window_actual0': reference_proof['actual0']}
        target = root / 'frontend_request.json'; core.write_new(target, frontend_request)
        frontend_child, frontend_proof = legacy_leaf_executor.run_leaf('frontend', core.record(target, guard), sources['frontend'], root,
                                                 python_pin, guard_pin, parent_pin, guard, task_monitor=monitor)
        legacy_leaf_executor.inventory(root)
        closure = public.observe_feature_window(frontend_child, owner_pin=owner, actual_process0=frontend_proof['actual0'],
            report=frontend_proof['report'], output=root / 'actual_frontend_window_closure.json', guard=guard)
        manifest = public.finalize_window(owner, closure, guard=guard)
        reader = public.FeatureWindowReader(owner, manifest, guard=guard)
        frontend = core.read_bound(frontend_proof['report'], guard)
        total_rows = sum(summary['row_counts'])
        core.require(frontend['passed'] is True and frontend['all_updates_reported'] is True
            and frontend['full_rows'] == total_rows and frontend['individual_array_raw_hashes'] == 2 * total_rows
            and frontend['new_NPY_files_written'] == 2 * total_rows and frontend['native_features_ready'] is False,
            'Every full row and both original numeric arrays from the actual aggregate frontend0')
        rows, payloads, operations = [], [], []
        for index, row_count in enumerate(summary['row_counts']):
            item = core.read_bound(reader.update_record(index), guard)
            core.require(item['prepared_payload']['state'] == 'prepared_not_committed'
                and item['per_update_child0_claimed'] is False and len(item['rows']) == row_count,
                'All full copied provisional rows; no fake per-update child0')
            rows.extend({'update_index': index, 'row_index': j, **row} for j, row in enumerate(item['rows']))
            payloads.append(item['prepared_payload'])
            operations.extend({'draw': event['draw'], **operation}
                for event in item['prepared_payload']['rows'] for operation in event['augmentation'].get('operations', []))
        new_owner, new_folder, _ = public.owner_folder(owner, guard)
        store = public.engine._Store(new_folder, new_owner, guard, public.capacity_lock)
        logical = store.inventory(); actual_files = list(new_folder.iterdir())
        physical = ALLOCATION + sum(((p.stat().st_size + ALLOCATION - 1) // ALLOCATION) * ALLOCATION for p in actual_files)
        core.require(logical <= public.CAPACITY and len(actual_files) == len(new_owner['owned_names'])
            and physical <= 512 * 1024**2, 'Exact complete owned logical and cluster-rounded physical512MiB reservation')
        result = {'kind': KIND, 'passed': True, 'producer': core.identity(), 'worker_source': driver_pin,
            'legacy_leaf_execution_source': parent_pin, 'operational_compatibility': compatibility_pin,
            'reference_reuse_failure': None, 'new_aggregate_children_this_invocation': 2,
            'task_nonce':task_nonce,'merged_parent_remains_live':True,'aggregate_process_exit0_claimed':False,
            'parent_source': outer_parent, 'proposal': proposal_pin, 'source_contract': contract_pin,
            'owner': owner, 'manifest': manifest, 'actual_frontend_window_closure': closure,
            'bridge': source['bridge'], 'phase': summary['phase'], 'window_index': summary['window_index'],
            'reference_actual_aggregate0': reference_proof, 'frontend_actual_aggregate0': frontend_proof,
            'real_actual_child0_count': 2, 'per_update_child0_count_claimed': 0,
            'complete_raw_array_hash_rows': rows, 'complete_ordered_prepared_payloads': payloads,
            'updates': len(summary['row_counts']), 'full_rows': total_rows, 'individual_array_raw_hashes': 2 * total_rows,
            'new_reference_worker_timings': core.read_bound(reference_proof['report'], guard)['timings'],
            'new_frontend_worker_timings': frontend['timings'], 'freeze_after_setup': proposal['freeze_after_setup'],
            'recorded_room_operations': operations, 'recorded_room_operation_counts': {kind: sum(v['kind'] == kind for v in operations) for kind in ('noise', 'rir')},
            'actual_NEW_feature_logical_bytes': logical, 'actual_NEW_feature_cluster_rounded_bytes': physical,
            'private_output_inventory': legacy_leaf_executor.inventory(root), 'native_features_ready': False,
            'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
            'production_enabled': False, 'native_benchmark_only': False,
            'owner_validation_only': source['owner_validation_only'],
            'checkpoint_diagnostic_only': source['checkpoint_diagnostic_only'],
            'production_source_eligible': source['production_source_eligible'],
            'actual_full_CPU_preparation_speedup_claimed': False, 'wall_seconds': time.monotonic() - began}
        core.write_new(root / 'report.json', result)
        legacy_leaf_executor.inventory(root)
        monitor.close(0)
        report_pin=core.record(root/'report.json',guard)
        task={'kind':'ASR18_completed_merged_CPU_task_no_inflight_v1','passed':True,'owner':'/root/memory_diagnosis',
            'source':driver_pin,'producer':core.identity(),'parent_source':actual_parent_source,'task_nonce':task_nonce,
            'configuration':proposal_pin,'source_contract':contract_pin,'operational_compatibility':compatibility_pin,
            'report':report_pin,'owner_binding':owner,'manifest':manifest,
            'ordered_prepared_payloads_sha256':core.digest(payloads),
            'actual_reference_process0':reference_proof['actual0'],'actual_frontend_process0':frontend_proof['actual0'],
            'actual_all_leaf_processes_closed0':True,'CPU_leaf_inflight_at_task_return':False,
            'parent_process_exit0_claimed':False,'parent_remains_live_at_task_return':True,
            'task_family_monitor':core.record(root/'task_family_monitor_completion.json',guard),
            'whole_owned_CPU_allowance_bytes':1995108864,'merged_parent_current_cap_bytes':128000000}
        task_path=root/'merged_task_completion.json';core.write_new(task_path,task)
        task_pin=core.record(task_path,guard)
        checked_task_completion(task_pin,report_pin,proposal_pin,compatibility_pin,guard=guard)
        legacy_leaf_executor.inventory(root)
        task_returned=True
        return task_pin,report_pin
    finally:
        if monitor is not None and not task_returned:monitor.fail_close()
