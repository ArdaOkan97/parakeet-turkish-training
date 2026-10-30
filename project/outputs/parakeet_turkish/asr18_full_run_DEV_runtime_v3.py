"""Inactive optimizer-free sole-worker serial350 DEV runtime, no launcher/CLI.

The separately reviewed GPU owner must implement actual process launch, fresh
whole-worker admission, continuous guards and actual process0 sealing. Neither
this module nor its policy proposal authorizes a run or promotes diagnostics.
"""
import gc
import inspect
import os
from pathlib import Path
import sys
import time

from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
STORAGE = Path(r'E:\gpu_training_scratchpad')
OWNER = '/root/memory_diagnosis'


def same(a, b):
    return a == b and digest(a) == digest(b)


def resource_proposal():
    """Conservative arithmetic only, never a measured or accepted worker fit."""
    return {'kind': 'ASR18_optimizer_free_serial350DEV_resource_proposal_v1',
        'maximum_pre_restore_private_bytes': 3_000_000_000,
        'weight_RNG_restore_increment_bytes': 350_000_000,
        'serial350DEV_increment_bytes': 2_100_000_000,
        # Deliberately retain the entire restore allowance during evaluation.
        'whole_worker_private_forecast_bytes': 5_450_000_000,
        'readonly_base_increment_bytes': 900_000_000,
        'BPE_LoRA_structure_increment_bytes': 250_000_000,
        'CUDA_allocator_limit_GiB': 8.0, 'torch_threads': 4,
        'system_projected_commit_limit_percent': 90.0,
        'consecutive_fresh_projected_samples': 3,
        'system_critical_commit_limit_percent': 95.0,
        'minimum_commit_headroom_bytes': 2_000_000_000,
        'minimum_E_free_bytes': 100_000_000_000,
        'optimizer_objects': 0, 'Adam_tensor_views': 0,
        'post_restore_private_release_assumed': False,
        'actual_new_bootstrap_or_full350_fit_proved': False}


def checked_configuration(config, *, parent_source):
    from asr18_full_run_serial_DEV_v6 import read_bound, checked_request
    require(config['kind'] == 'ASR18_optimizer_free_serial350DEV_worker_configuration_v1'
            and config['worker_mode'] == 'serial350DEV_optimizer_free'
            and config['production_enabled'] is False
            and same(config['parent_driver_source'], parent_source)
            and same(config['resource_policy'], resource_proposal())
            and type(config['maximum_GPU_child_wall_seconds']) is int
            and 0 < config['maximum_GPU_child_wall_seconds'] <= 7200,
            'Exact reviewed inactive sole-worker DEV configuration')
    request = read_bound(config['request'])
    checked_request(request)
    execution = request['execution_sources']
    require(same(execution['worker_source'], record(__file__))
            and same(execution['bootstrap_source'], record(HERE/'asr18_optimizer_free_DEV_structure_v2.py'))
            and same(execution['parent_source'], parent_source), 'Exact NEW runtime/structure/owner sources')
    source_review = read_bound(config['ROOT_execution_review'])
    resource_review = read_bound(config['ROOT_resource_review'])
    require(source_review['kind'] == 'ROOT_optimizer_free_serial350DEV_execution_configuration_review_v1'
            and source_review['accepted'] is True and source_review['reviewer'] == '/root'
            and same(source_review['request'], config['request'])
            and same(source_review['execution_sources'], execution)
            and same(source_review['base_assets'], config['base_assets'])
            and same(source_review['source_bindings'], config['source_bindings'])
            and same(source_review['resource_review'], config['ROOT_resource_review'])
            and resource_review['kind'] == 'ROOT_optimizer_free_serial350DEV_resource_review_v1'
            and resource_review['accepted'] is True and resource_review['reviewer'] == '/root'
            and same(resource_review['resource_policy'], resource_proposal())
            and same(resource_review['execution_sources'], execution),
            'Separate ROOT exact execution and numeric resource acceptance; source existence is insufficient')
    for pin in config['source_bindings'].values():
        require(same(record(pin['path']), pin), 'Owned source changed after review')
    for name in ('asr18_full_run_DEV_runtime_v3.py','asr18_optimizer_free_DEV_structure_v2.py',
                 'asr18_full_run_serial_DEV_v6.py','asr18_DEV_weight_only_mapper_v1.py',
                 'stream_frozen_encoder_matrices_v1.py','streamed_generation_device_v1.py',
                 'train_lora_batched_tsc_bpe_full_decoder_v1.py','metrics.py','tdt_joint_canonical_2d_fp32_v2.py'):
        require(same(config['source_bindings'][name],record(HERE/name)), 'Required active source omitted')
    base = Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
    require(base.is_relative_to(PROJECT.resolve()) or base.is_relative_to(STORAGE.resolve()),
            'Existing original base must stay in authorized project or E storage')
    require(set(config['base_assets']) == {'config.json','generation_config.json','processor_config.json',
                                          'tokenizer_config.json','tokenizer.json'},
            'Complete original tokenizer/frontend/config asset set')
    for name,pin in config['base_assets'].items():
        require(same(pin,record(base/name)), 'Original base tokenizer/frontend asset changed')
    return request


def run(config_path, folder, *, parent_source, check_bindings, storage, stage, admit_stage):
    """Fresh child entrypoint; callbacks all belong to the reviewed owner parent.

    Parent writes a SEPARATE actual-PID admission record after worker_boot_identity
    exists. The immutable pre-worker request never contains invented future PIDs.
    Parent must keep stop/lease/resource monitoring active until actual death.
    """
    began = time.monotonic(); storage()
    folder = Path(folder).resolve()
    require(folder.is_relative_to(STORAGE.resolve()) and folder.is_dir(), 'Fresh owner-created E output only')
    for function in (check_bindings, storage, stage, admit_stage):
        require(callable(function) and str(Path(inspect.getsourcefile(function)).resolve()) == parent_source['path'],
                'Actual source-bound owner callbacks, no unreviewed lambda bypass')
    config = read_json(config_path); check_bindings(config)
    request = checked_configuration(config, parent_source=parent_source)
    proposal = resource_proposal(); producer = identity()
    write_json(folder/'worker_boot_identity.json', {'producer': producer, 'source': record(__file__),
        'request': config['request'], 'actual_optimizer_objects': 0})
    admission_path = Path(config['owned_worker_admission_path']).resolve()
    require(admission_path == folder/'owned_worker_admission.json', 'Actual future admission path is exact E child path')
    lease_path = PROJECT/'work/gpu_training_owner.json'
    def guard():
        require(not (folder/'stop.json').exists() and time.monotonic()-began < config['maximum_GPU_child_wall_seconds'],
                'Owned DEV stop or finite wall limit')
        lease = read_json(lease_path)
        require(lease['owner'] == OWNER and lease['actual_GPU_running'] is True
                and lease['actual_owned_worker_pid'] == os.getpid()
                and lease['actual_training_optimizer_running'] is False,
                'Actual sole-owned optimizer-free GPU lease must remain current')
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
            'Actual native-owner fresh whole-worker admission with real worker identity')
    require('torch' not in sys.modules, 'Coherent CUDA preload before any Torch import')
    from preload_nemo_linker_v1 import preload
    coherent = preload()
    require(coherent['actual_binding_linker']['version'] == [12,9] and not coherent['cuda_initialized'],
            'Frozen CUDA12.9 preload')
    import torch
    torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity = validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'], device=0)
    from resource_watchdog_v2 import sample_memory
    from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
    phase = None; phase_records = []
    def private(label):
        guard(); value = sample_memory(os.getpid(), label)
        require(value['paired_sampling_within_one_second'] and value['system_memory']['available']
                and value['worker_private_memory']['available'], 'Fresh paired DEV telemetry')
        measured = value['worker_private_memory']['private_bytes']
        require(measured <= proposal['whole_worker_private_forecast_bytes'], 'Whole DEV-worker forecast exceeded')
        return measured
    def finish_phase():
        nonlocal phase
        if phase is None: return
        current = phase; phase = None
        observed = current['monitor'].finish()
        covered = observed['private_peak_bytes'] <= current['before_private_bytes']+current['increment_bytes']
        path = folder/(current['name']+'_memory.json')
        write_json(path, {'name':current['name'], 'before_private_bytes':current['before_private_bytes'],
            'increment_bytes':current['increment_bytes'], 'actual':observed, 'covered':covered,
            'wall_seconds':time.monotonic()-current['began'],
            'margin_bytes':current['before_private_bytes']+current['increment_bytes']-observed['private_peak_bytes']})
        phase_records.append(record(path))
        require(covered, 'Prospective DEV phase allowance missed; no unchanged retry')
    def admit(name, budget):
        nonlocal phase
        finish_phase(); before = private(name+'_before')
        expected = {'optimizer_free_DEV_readonly_base':900_000_000,
            'optimizer_free_DEV_BPE_LoRA_structure':250_000_000,
            'full_run_DEV_weight_only_restore_and_hash':350_000_000,
            'full_run_optimizer_free_serial350DEV_original_route':2_100_000_000}
        require(name in expected and type(budget) is int and budget == expected[name]
                and before+budget <= proposal['whole_worker_private_forecast_bytes'],
                'Exact unchanged component and whole-worker prospective budgets')
        if name == 'full_run_DEV_weight_only_restore_and_hash':
            require(before <= proposal['maximum_pre_restore_private_bytes'], 'Actual pre-restore private must be <=3GB')
        admit_stage(folder, name, budget)
        monitor = PhasePrivateMonitor(); monitor.start()
        phase = {'name':name,'before_private_bytes':before,'increment_bytes':budget,
                 'monitor':monitor,'began':time.monotonic()}
    def emit(name, **extra):
        guard(); stage(folder, name, torch, **extra)
        if name == 'full_run_optimizer_free_serial350DEV_ready': finish_phase()
    model = processor = None
    try:
        # Same narrow namespace path as actual closed350; never calls TDT loss,
        # bootstrap optimizers, training samplers, replay or any model backward.
        from native_tdt_minimal_import_v1 import prepare
        minimal = prepare(); emit('optimizer_free_DEV_runtime_ready', coherent=coherent, capacity=capacity, minimal=minimal)
        admit('optimizer_free_DEV_readonly_base',900_000_000)
        from transformers import AutoProcessor
        from stream_frozen_encoder_matrices_v1 import load
        from streamed_generation_device_v1 import apply
        base = Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
        # Loader's exact original archive/bank and local-only processor; never downloads.
        processor = AutoProcessor.from_pretrained(base,local_files_only=True)
        model, loader = load(base); logical = apply(model)
        from asr18_optimizer_free_DEV_structure_v2 import prepare_structure
        structure = prepare_structure(model,processor,torch=torch,guard=guard,admit=admit)
        finish_phase()
        from bpe_broad_production_v3 import prohibit_model_transforms
        prohibit_model_transforms(model,torch)
        from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
        joint = canonical_joint(model.joint)
        from asr18_full_run_serial_DEV_v6 import evaluate_boundary_checkpoint, OPTIMIZER_FREE_PROTOCOL
        worker_report = evaluate_boundary_checkpoint(model,processor,config['request'],folder,
            owned_worker_admission=admission_pin,torch=torch,stage=emit,admit=admit,stop_check=guard)
        finish_phase(); check_bindings(config)
    finally:
        finish_phase()
        model = processor = None; gc.collect(); torch.cuda.empty_cache()
    emit('optimizer_free_DEV_model_released')
    report = {'kind':'ASR18_optimizer_free_serial350DEV_runtime_completion_v1','passed':True,
        'optimizer_free_protocol':OPTIMIZER_FREE_PROTOCOL,
        'producer':producer,'source':record(__file__),'worker_source':request['execution_sources']['worker_source'],
        'parent_source':parent_source,'request':config['request'],'worker_report':worker_report,
        'owned_worker_admission':admission_pin,'resource_policy':proposal,'phase_memory_records':phase_records,
        'structure':structure,'loader':loader,'logical_device':logical,'canonical_joint':joint,
        'optimizer_objects_constructed':0,'Adam_tensor_views_created':0,'native_updates':0,
        'sampler_or_replay_objects_constructed_or_advanced':False,'wall_seconds':time.monotonic()-began,
        'actual_process_exit_pending':True,'production_approval_claimed':False}
    write_json(folder/'worker_completion.json',report)
    return report
