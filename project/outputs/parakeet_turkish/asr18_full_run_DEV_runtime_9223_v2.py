"""Inactive real-pass-only optimizer-free serial350 DEV worker.

Uses the closed diagnostic runtime's model/restore/evaluation recipe, with the
unchanged public evaluator and a static whole-run executor review. Future
checkpoint/PID identities are validated when real, never fabricated in advance.
The outer sole GPU owner supplies actual admission, monitoring and process0.
"""
import gc
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/'work/research'))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity

STORAGE = Path(r'E:\gpu_training_scratchpad')
OWNER = '/root/memory_diagnosis'


def same(a, b): return a == b and digest(a) == digest(b)


def resource_proposal():
    return {'kind': 'ASR18_optimizer_free_serial350DEV_resource_proposal_v1',
        'maximum_pre_restore_private_bytes': 3_000_000_000,
        'weight_RNG_restore_increment_bytes': 350_000_000,
        'serial350DEV_increment_bytes': 2_100_000_000,
        'whole_worker_private_forecast_bytes': 5_450_000_000,
        'readonly_base_increment_bytes': 900_000_000,
        'BPE_LoRA_structure_increment_bytes': 250_000_000,
        'CUDA_allocator_limit_bytes': 8_000_000_000, 'capacity_argument_GB_decimal': 8.0, 'torch_threads': 4,
        'system_projected_commit_limit_percent': 90.0,
        'consecutive_fresh_projected_samples': 3,
        'system_critical_commit_limit_percent': 95.0,
        'minimum_commit_headroom_bytes': 2_000_000_000,
        'minimum_E_free_bytes': 100_000_000_000,
        'optimizer_objects': 0, 'Adam_tensor_views': 0,
        'post_restore_private_release_assumed': False,
        'actual_new_bootstrap_or_full350_fit_proved': False}


def check_bindings(config):
    pin = config['source_manifest']; require(record(pin['path']) == pin, 'Frozen full source manifest')
    manifest = read_json(pin['path'])
    for item in manifest['bindings']: require(record(item['path']) == item, 'Admitted active dependency changed')
    require(record(__file__) == config['worker_driver_source'], 'Exact actual DEV entrypoint source')


def checked_configuration(config, *, parent_source):
    from asr18_full_run_serial_DEV_9223_v2 import read_bound, checked_request
    from bpe_full_run_lifecycle_v2 import milestones
    require(config['kind'] == 'ASR18_optimizer_free_serial350DEV_worker_configuration_v1'
            and config['worker_mode'] == 'serial350DEV_optimizer_free'
            and config['production_enabled'] is False and config['diagnostic_only'] is False
            and same(config['parent_driver_source'], parent_source)
            and same(config['resource_policy'], resource_proposal())
            and type(config['maximum_GPU_child_wall_seconds']) is int
            and 0 < config['maximum_GPU_child_wall_seconds'] <= 7200,
            'Exact original full-run optimizer-free DEV configuration')
    request = read_bound(config['request'])
    _, _, _, sampler = checked_request(request)  # Genuine current native0/recovery/prefix/pass boundary.
    execution = request['execution_sources']
    require(same(execution['worker_source'], record(__file__))
            and same(execution['bootstrap_source'], record(HERE/'asr18_optimizer_free_DEV_structure_9223_v1.py'))
            and same(execution['parent_source'], parent_source), 'Exact runtime/structure/sole owner sources')
    review = read_bound(config['ROOT_execution_review'])
    resource = read_bound(config['ROOT_resource_review'])
    require(review['kind'] == 'ROOT_full_run_serial350DEV_static_executor_review_v1'
            and review['accepted'] is True and review['reviewer'] == '/root'
            and review['scope'] == 'static_original_full_run_pass_boundaries_only'
            and review['run_identity_sha256'] == digest(request['identity'])
            and same(review['full_run_source_contract'], request['identity']['full_run_source_contract'])
            and same(review['declared_boundaries'], milestones(sampler))
            and same(review['execution_sources'], execution)
            and same(review['implementation'], record(HERE/'asr18_full_run_serial_DEV_9223_v2.py'))
            and same(review['base_assets'], config['base_assets'])
            and same(review['source_bindings'], config['source_bindings'])
            and same(review['resource_review'], config['ROOT_resource_review'])
            and same(review['scorer_policy'], request['scorer_policy'])
            and same(review['ROOT_policy_review'], request['ROOT_policy_review'])
            and resource['kind'] == 'ROOT_optimizer_free_serial350DEV_resource_review_v1'
            and resource['accepted'] is True and resource['reviewer'] == '/root'
            and same(resource['resource_policy'], resource_proposal())
            and same(resource['execution_sources'], execution),
            'Static exact whole-run source/policy/resource authority; no future checkpoint request pin')
    for pin in config['source_bindings'].values(): require(same(record(pin['path']), pin), 'DEV active source changed')
    for name in ('asr18_full_run_DEV_runtime_9223_v2.py', 'asr18_optimizer_free_DEV_structure_9223_v1.py',
                 'asr18_full_run_serial_DEV_9223_v2.py', 'asr18_DEV_weight_only_mapper_9223_v1.py',
                 'stream_frozen_encoder_matrices_v1.py', 'streamed_generation_device_v1.py',
                 'train_lora_batched_tsc_bpe_full_decoder_v1.py', 'metrics.py',
                 'tdt_joint_canonical_2d_fp32_9223_v1.py', 'asr18_temporal_tiled_subsampling_v2.py'):
        require(same(config['source_bindings'][name], record(HERE/name)), 'Required active DEV source omitted')
    base = Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
    require(base.is_relative_to(PROJECT.resolve()) or base.is_relative_to(STORAGE.resolve()), 'Authorized existing original base')
    require(set(config['base_assets']) == {'config.json', 'generation_config.json', 'processor_config.json',
                                         'tokenizer_config.json', 'tokenizer.json'}, 'Complete original frontend/tokenizer assets')
    for name, pin in config['base_assets'].items(): require(same(pin, record(base/name)), 'Original base asset changed')
    return request


def run(config_path, folder):
    from asr18_memory_diagnosis_v2 import storage, stage, admit_stage
    storage(); began = time.monotonic(); folder = Path(folder).resolve()
    require(folder.is_relative_to(STORAGE.resolve()) and folder.is_dir(), 'Parent-created owned E DEV directory')
    config = read_json(config_path); producer = identity(); parent_source = config['parent_driver_source']
    boot = {'producer': producer, 'source': record(__file__), 'configuration': record(config_path),
            'parent_source': parent_source}
    write_json(folder/'entrypoint_boot.json', boot)  # Before source scans or ML imports.
    check_bindings(config); request = checked_configuration(config, parent_source=parent_source)
    proposal = resource_proposal(); write_json(folder/'worker_boot_identity.json', boot)
    admission_path = Path(config['owned_worker_admission_path']).resolve()
    require(admission_path == folder/'owned_worker_admission.json', 'Exact actual-PID child admission path')
    lease_path = PROJECT/'work/gpu_training_owner.json'
    def guard():
        require(not (folder/'stop.json').exists() and time.monotonic()-began < config['maximum_GPU_child_wall_seconds'],
                'Owned DEV stop or finite wall budget')
        lease = read_json(lease_path)
        require(lease['owner'] == OWNER and lease['actual_GPU_running'] is True
                and lease['actual_owned_worker_pid'] == os.getpid()
                and lease['actual_training_optimizer_running'] is False, 'Actual sole optimizer-free GPU lease')
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        lease = read_json(lease_path)
        if (lease.get('owner') == OWNER and lease.get('actual_GPU_running') is True
                and lease.get('actual_owned_worker_pid') == os.getpid() and admission_path.exists()): break
        time.sleep(.1)
    else: raise ValueError('Actual lease/PID/admission handshake missing')
    guard(); admission_pin = record(admission_path); admission = read_json(admission_path)
    require(same(admission['producer'], producer) and same(admission['request'], config['request'])
            and same(admission['ROOT_execution_review'], config['ROOT_execution_review'])
            and same(admission['ROOT_resource_review'], config['ROOT_resource_review'])
            and admission['whole_worker_fresh_three_sample_admission_passed'] is True
            and admission['whole_worker_forecast_bytes'] == proposal['whole_worker_private_forecast_bytes'],
            'Actual independently observed whole-worker admission, no invented future PID')
    require('torch' not in sys.modules, 'Coherent CUDA preload precedes Torch')
    from preload_nemo_linker_v1 import preload
    coherent = preload(); require(coherent['actual_binding_linker']['version'] == [12, 9]
                                 and not coherent['cuda_initialized'], 'Unchanged CUDA12.9 preload')
    import torch
    torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity = validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)
    require(capacity['cap_bytes'] == proposal['CUDA_allocator_limit_bytes'] == 8000000000, 'Exact decimal8GB CUDA capacity')
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'], device=0)
    from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
    phase = None; phase_records = []
    def emit(name, **extra):
        guard(); value = stage(folder, name, torch, **extra)
        require(value['worker_private_memory']['private_bytes'] <= proposal['whole_worker_private_forecast_bytes']
                and value['worker_private_peak_bytes_lifetime'] <= proposal['whole_worker_private_forecast_bytes']
                and value['cuda']['peak_allocated_bytes'] <= 8000000000, 'Actual current+OS-lifetime/CUDA whole DEV guard')
        if name == 'full_run_optimizer_free_serial350DEV_ready': finish_phase()
        return value
    def private(label):
        value = emit(label)
        require(value['paired_sampling_within_one_second'] and value['system_memory']['available']
                and value['worker_private_memory']['available'], 'Fresh paired DEV telemetry')
        return value['worker_private_memory']['private_bytes']
    def finish_phase():
        nonlocal phase
        if phase is None: return
        current = phase; phase = None; observed = current['monitor'].finish()
        covered = observed['private_peak_bytes'] <= current['before_private_bytes']+current['increment_bytes']
        path = folder/(current['name']+'_memory.json')
        write_json(path, {'name': current['name'], 'before_private_bytes': current['before_private_bytes'],
            'increment_bytes': current['increment_bytes'], 'actual': observed, 'covered': covered,
            'wall_seconds': time.monotonic()-current['began'],
            'margin_bytes': current['before_private_bytes']+current['increment_bytes']-observed['private_peak_bytes']})
        phase_records.append(record(path)); require(covered, 'Prospective DEV phase missed; no unchanged retry')
    def admit(name, budget):
        nonlocal phase
        finish_phase(); before = private(name+'_before')
        expected = {'optimizer_free_DEV_readonly_base': 900000000, 'optimizer_free_DEV_BPE_LoRA_structure': 250000000,
            'full_run_DEV_weight_only_restore_and_hash': 350000000,
            'full_run_optimizer_free_serial350DEV_original_route': 2100000000}
        require(name in expected and type(budget) is int and budget == expected[name]
                and before+budget <= proposal['whole_worker_private_forecast_bytes'], 'Exact original four phase budgets')
        if name == 'full_run_DEV_weight_only_restore_and_hash': require(before <= 3000000000, 'Unchanged3GB pre-restore bound')
        admit_stage(folder, name, budget); monitor = PhasePrivateMonitor(); monitor.start()
        phase = {'name': name, 'before_private_bytes': before, 'increment_bytes': budget,
                 'monitor': monitor, 'began': time.monotonic()}
    model = processor = None
    try:
        from native_tdt_minimal_import_v1 import prepare
        minimal = prepare(); emit('optimizer_free_DEV_runtime_ready', coherent=coherent, capacity=capacity, minimal=minimal)
        admit('optimizer_free_DEV_readonly_base', 900000000)
        from transformers import AutoProcessor
        from stream_frozen_encoder_matrices_v1 import load
        from streamed_generation_device_v1 import apply
        base = Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
        processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        model, loader = load(base); logical = apply(model)
        from asr18_optimizer_free_DEV_structure_9223_v1 import prepare_structure
        structure = prepare_structure(model, processor, torch=torch, guard=guard, admit=admit); finish_phase()
        from bpe_broad_production_v3 import prohibit_model_transforms
        prohibit_model_transforms(model, torch)
        from tdt_joint_canonical_2d_fp32_9223_v1 import apply as canonical_joint
        joint = canonical_joint(model.joint)
        from asr18_temporal_tiled_subsampling_v2 import apply as conv_only
        conv = conv_only(model, torch=torch, guard=guard)
        from asr18_full_run_serial_DEV_9223_v2 import evaluate_boundary_checkpoint, OPTIMIZER_FREE_PROTOCOL
        worker_report = evaluate_boundary_checkpoint(model, processor, config['request'], folder,
            owned_worker_admission=admission_pin, torch=torch, stage=emit, admit=admit, stop_check=guard)
        finish_phase(); check_bindings(config)
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        require(torch.cuda.memory_reserved() <= 3000000000, 'Coherent completed evaluation endcache3GB')
        emit('full_run_DEV_evaluation_coherent_endcache')
    finally:
        finish_phase(); model = processor = None; gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        require(torch.cuda.memory_reserved() <= 3000000000, 'Final model-release endcache3GB')
    emit('optimizer_free_DEV_model_released')
    report = {'kind': 'ASR18_optimizer_free_serial350DEV_runtime_completion_v1', 'passed': True,
        'optimizer_free_protocol': OPTIMIZER_FREE_PROTOCOL, 'producer': producer, 'source': record(__file__),
        'worker_source': request['execution_sources']['worker_source'], 'parent_source': parent_source,
        'request': config['request'], 'worker_report': worker_report, 'owned_worker_admission': admission_pin,
        'resource_policy': proposal, 'phase_memory_records': phase_records, 'structure': structure, 'loader': loader,
        'logical_device': logical, 'canonical_joint': joint, 'conv_only_CNN': conv, 'diagnostic_only': False,
        'optimizer_objects_constructed': 0, 'Adam_tensor_views_created': 0, 'native_updates': 0,
        'sampler_or_replay_objects_constructed_or_advanced': False, 'wall_seconds': time.monotonic()-began,
        'actual_process_exit_pending': True, 'production_approval_claimed': False}
    write_json(folder/'worker_completion.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--folder', required=True)
    args = parser.parse_args(); run(args.config, args.folder)
