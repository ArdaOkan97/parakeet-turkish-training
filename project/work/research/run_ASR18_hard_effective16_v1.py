"""NEW opt-in finite two-window caller, resident-first, production disabled.

All sources are frozen before admission. Genuine final DATA plus geometry and
ROOT's complete caller review are mandatory. This file supplies concrete CPU,
GPU, checkpoint, idle, retirement and next-window operations; preparation alone
never advances a sampler. No command line can launch long production.
"""
import argparse, copy, gc, json, os, secrets, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'outputs/parakeet_turkish'; sys.path.insert(0, str(SOURCE))
ROOT = Path(r'E:\gpu_training_scratchpad'); OWNER = '/root/memory_diagnosis'
LEASE = PROJECT/'work/gpu_training_owner.json'
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from asr18_memory_diagnosis_v2 import storage, now, stage, admit_stage, concurrent_data_observation, terminate_runtime_worker
from resource_watchdog_v2 import ResourceGuard, sample_memory


def scoped(path):
    path = Path(path).resolve(); require(path.is_relative_to(ROOT.resolve()), 'All new outputs E-only'); return path


def check_bindings(config):
    manifest = read_json(config['source_manifest']['path'])
    require(record(config['source_manifest']['path']) == config['source_manifest'], 'Reviewed source manifest changed')
    for pin in manifest['bindings']: require(record(pin['path']) == pin, 'Admitted transitive source changed: '+pin['path'])
    return manifest


def fresh_admit(increment, *, pid=None):
    require(type(increment) is int and increment > 0, 'Positive declared incremental budget')
    sample = sample_memory(os.getpid() if pid is None else pid, 'full_DATA_prospective_admission')
    require(sample['paired_sampling_within_one_second'] and sample['system_memory']['available']
            and sample['worker_private_memory']['available'], 'Fresh available paired prospective telemetry required')
    system = sample['system_memory']
    passed = system['commit_used_bytes']+increment < .90*system['commit_limit_bytes'] and system['commit_available_bytes']-increment > 2_000_000_000
    return {'passed': passed, 'incremental_private_forecast_bytes': increment, 'sample': sample,
            'system90_three_fresh_95critical_2GB_headroom_unchanged': True}


class ParentGuard:
    """The parent is CPU-only and bounded even while no GPU process exists."""
    def __init__(self, folder):
        self.folder = folder; self.watch = ResourceGuard(); self.done = threading.Event(); self.error = None
        self.count = 0; self.last = None; self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
        deadline = time.monotonic()+3
        while self.count < 3:
            require(self.thread.is_alive() and self.error is None and time.monotonic() < deadline, 'Live parent guard before work')
            time.sleep(.05)

    def _run(self):
        try:
            while not self.done.is_set():
                sample = sample_memory(os.getpid(), 'full_DATA_CPU_parent_guard'); worker = sample['worker_private_memory']
                require(sample['paired_sampling_within_one_second'] and worker['available'] and sample['system_memory']['available'], 'Fresh paired parent observation')
                reason, detail = self.watch.observe(sample['system_memory'], worker)
                require(reason is None and worker['private_bytes'] < 2_000_000_000, 'Parent resource stop: '+str(reason))
                require(shutil.disk_usage(ROOT).free > 100_000_000_000, 'Unchanged E100GB free reserve')
                self.count += 1; self.last = time.monotonic(); self.done.wait(.2)
        except BaseException: self.error = traceback.format_exc()

    def check(self):
        require(self.error is None and self.thread.is_alive() and self.last is not None and time.monotonic()-self.last < 2,
                'Parent monitor unavailable/stale/failed; stop, no child retry')
        return sample_memory(os.getpid(), 'full_DATA_parent_callback')

    def close(self):
        self.done.set(); self.thread.join(timeout=2)
        require(not self.thread.is_alive() and self.error is None and self.count >= 3, 'Clean joined parent guard required')


def worker(config_path, folder):
    worker_began = time.monotonic(); storage(); config = read_json(config_path); config['_path'] = config_path; check_bindings(config)
    require(config['worker_mode'] == 'resident_training', 'No recovery commands in a training worker')
    from fastmss_owned_process_closure_v2 import identity
    producer = identity(); write_json(folder/'worker_boot_identity.json', {'producer': producer, 'source': record(__file__)})
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        lease = read_json(LEASE)
        if lease['owner'] == OWNER and lease['actual_GPU_running'] and lease['actual_owned_worker_pid'] == os.getpid(): break
        time.sleep(.1)
    else: raise ValueError('Actual exclusive GPU-worker lease handshake required')
    require('torch' not in sys.modules, 'CUDA12.9 preload must precede Torch')
    from preload_nemo_linker_v1 import preload
    coherent = preload(); require(coherent['actual_binding_linker']['version'] == [12, 9] and not coherent['cuda_initialized'], 'Coherent CUDA12.9 binding')
    import torch
    torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity = validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'], device=0)
    emit = lambda name, **extra: stage(folder, name, torch, **extra)
    admit = lambda name, budget: admit_stage(folder, name, budget)
    def guard(): require(not (folder/'stop.json').exists(), 'Owned parent cooperative stop')
    from native_tdt_minimal_import_v1 import prepare
    minimal = prepare(); emit('full_DATA_native_runtime_ready', capacity=capacity, coherent=coherent)
    admit('readonly_streamed_base_and_original_teacher', 900_000_000)
    from asr18_token_only_carrier_v1 import load as load_token_carrier
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply
    base = Path((PROJECT/'work/base_model_path.txt').read_text().strip())
    model, loader = load(base); logical = apply(model); processor = load_token_carrier(base)
    teacher = {name: copy.deepcopy(getattr(model, name)).eval().requires_grad_(False) for name in ('encoder_projector', 'decoder', 'joint')}
    from bpe_broad_streamed_bootstrap_v2 import bootstrap
    optimizer, provenance, replay = bootstrap(model, processor, torch=torch, stage=emit, admit=admit)
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    provenance.update(frozen_loader=record(SOURCE/'stream_frozen_encoder_matrices_v1.py'), bootstrap=record(SOURCE/'bpe_broad_streamed_bootstrap_v2.py'),
        minimal_native=minimal, logical_device=logical, NEW_joint_numerical_policy=canonical_joint(model.joint), frozen_loader_policy=loader)
    from asr18_hard_effective16_native_v1 import NativeWindowEngine
    engine = NativeWindowEngine(model, processor, teacher, optimizer, provenance, replay, folder, config,
                               torch=torch, stage=emit, admit=admit, guard=guard)
    sequence = 0; windows = []; began = time.monotonic()
    try:
        if config.get('restore_parent_pointer') is not None:
            engine.restore_parent(config['restore_parent_pointer'], augmentation_identity=config['augmentation_identity'])
        write_json(folder/'worker_ready.json', {'passed': True, 'producer': producer, 'optimizer_updates': optimizer.updates,'worker_startup_and_initial_restore_seconds':time.monotonic()-worker_began})
        while True:
            guard(); require(time.monotonic()-began < config['maximum_GPU_child_wall_seconds'], 'Finite native child time budget')
            path = folder/'command.json'
            if not path.exists(): time.sleep(.1); continue
            command = read_json(path)
            if command['sequence'] <= sequence: time.sleep(.1); continue
            require(command['sequence'] == sequence+1 and type(command['nonce']) is str and len(command['nonce']) >= 16, 'Exact serial command sequence/nonce')
            sequence = command['sequence']; check_bindings(config)
            if command['kind'] == 'run_prepared_window':
                complete = engine.run_window(command['window']); windows.append(complete)
                idle = engine.coherent_idle(nonce=command['nonce'], window_completion=complete)
                idle_path = folder/('idle_ack_'+str(sequence)+'.json'); write_json(idle_path, idle)
                response = {'passed': True, 'sequence': sequence, 'nonce': command['nonce'], 'window_completion': complete, 'idle_ack': record(idle_path)}
            elif command['kind'] == 'refresh_coherent_idle':
                require(windows, 'No idle state before a completed actual native window')
                idle = engine.coherent_idle(nonce=command['nonce'], window_completion=windows[-1])
                idle_path = folder/('idle_ack_'+str(sequence)+'.json'); write_json(idle_path, idle)
                response = {'passed': True, 'sequence': sequence, 'nonce': command['nonce'], 'window_completion': windows[-1], 'idle_ack': record(idle_path)}
            elif command['kind'] == 'finish_coherent_native_process':
                require(windows or config.get('restore_parent_pointer') is not None, 'No invented completed native boundary')
                write_json(folder/'response.json', {'passed': True, 'sequence': sequence, 'nonce': command['nonce'], 'actual_process_exit_pending': True}); break
            else: raise ValueError('Unsupported finite native command')
            write_json(folder/'response.json', response)
    finally: engine.close()
    actual_updates = optimizer.updates; actual_process_updates = optimizer.updates-engine.process_initial_updates
    del engine, optimizer, teacher, model, processor; gc.collect(); torch.cuda.empty_cache()
    emit('full_DATA_native_models_released')
    result = {'passed': True, 'producer': producer, 'worker_source': record(__file__), 'parent_source': config['parent_driver_source'],
              'windows': windows, 'coherent_chain_new_optimizer_updates': actual_updates,
              'actual_native_updates_this_process': actual_process_updates, 'wall_seconds': time.monotonic()-began,
              'all_sources_unchanged': True, 'production_enabled': False, 'qualified_checkpoint': False,
              'all_exact_GPU_checkpoint_restores_deferred': True,
              'deferred_flag_scope': 'All checkpoint captures produced in this training process; startup migration restore is separately recorded',
              'startup_fallback_restore_performed': config.get('restore_parent_pointer') is not None,
              'overall_diagnostic_pass_before_separate_recovery': False}
    check_bindings(config); write_json(folder/'worker_completion.json', result)


class GPUService:
    @classmethod
    def launch(cls, *arguments):
        candidate = object.__new__(cls)
        try:
            cls.__init__(candidate, *arguments)
            return candidate
        except BaseException:
            child = getattr(candidate, 'child', None)
            if child is not None:
                from asr18_owned_family_observation_v4 import terminate_known_family
                known = getattr(candidate, 'known', {})
                producer = getattr(candidate, 'producer', None)
                if producer is not None:
                    known[(producer['pid'], producer['create_time'])] = producer
                # Even if the launcher's handle already closed, native identity
                # retirement must precede releasing a possibly acquired lease.
                terminate_known_family(child, getattr(candidate, 'launcher', None), known)
            done = getattr(candidate, 'done', None)
            if done is not None: done.set()
            monitor = getattr(candidate, 'monitor', None)
            if monitor is not None: monitor.join(timeout=2)
            log = getattr(candidate, 'log', None)
            if log is not None: log.close()
            if getattr(candidate, 'pid', None) is not None: candidate._release_lease()
            raise

    def __init__(self, config_path, folder, parent_guard):
        from fastmss_owned_process_closure_v2 import identity
        self.folder = folder; self.parent_guard = parent_guard; self.config = read_json(config_path)
        from asr18_full_population_geometry_forecast_v1 import CalibratedGeometryForecast
        from asr18_full_population_separated_forecast_v1 import resident_worker_forecast, restore_only_worker_forecast
        require(self.config['worker_mode'] in ('resident_training', 'restore_only'), 'Exact separated GPU worker mode')
        if self.config['worker_mode'] == 'resident_training':
            calibration = CalibratedGeometryForecast(**self.config['forecast_calibration'], guard=parent_guard.check)
            first_inputs = read_json(self.config['first_feature_update']['path'])
            require(record(self.config['first_feature_update']['path']) == self.config['first_feature_update'], 'Actual first closed feature input changed')
            expected = resident_worker_forecast(calibration.budget(rows=first_inputs['rows'], current_private_bytes=3_000_000_000, update_in_process=1))
        else:
            from asr18_hard_effective16_recovery_v1 import verified_requests
            verified_requests(self.config['deferred_recovery_requests'], session=self.config['session'],
                parent_source=record(__file__), guard=parent_guard.check)
            expected = restore_only_worker_forecast()
        require(digest(expected) == digest(self.config['entire_worker_forecast']), 'Full worker forecast must match actual source-bound policy, not arbitrary config bytes')
        require(record(self.config['GPU_python']['path']) == self.config['GPU_python'], 'Existing GPU runtime executable changed')
        full = self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']
        admission = fresh_admit(full); write_json(folder.parent/(folder.name+'_prelaunch_full_forecast.json'), admission)
        require(admission['passed'], 'Entire GPU worker prospective forecast does not fit; no GPU launch')
        lease = read_json(LEASE); require(lease['owner'] == OWNER and not lease['actual_GPU_running'] and not lease.get('actual_owned_worker_pid'), 'Exclusive actual idle GPU lease')
        folder.mkdir(exist_ok=False); self.log = (folder/'worker.log').open('xb')
        environment = dict(os.environ); environment['CUDA_VISIBLE_DEVICES'] = '0'; environment['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'] = '16'
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): environment[key] = '1'
        self.child = subprocess.Popen([self.config['GPU_python']['path'], '-X', 'utf8', '-B', str(Path(__file__).resolve()),
            '--worker' if self.config['worker_mode'] == 'resident_training' else '--recovery-worker',
            '--config', str(config_path), '--folder', str(folder)], env=environment, cwd=PROJECT,
            stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.launcher = identity(self.child.pid)
        self.known = {(self.launcher['pid'], self.launcher['create_time']): self.launcher}
        deadline = time.monotonic()+15
        while not (folder/'worker_boot_identity.json').exists() and self.child.poll() is None and time.monotonic() < deadline:
            from asr18_owned_family_observation_v4 import owned_family_sample
            owned_family_sample(self.child, self.launcher, known=self.known, guard=parent_guard.check)
            time.sleep(.1)
        boot = read_json(folder/'worker_boot_identity.json'); producer = boot['producer']
        require(boot['source'] == record(__file__) and producer == identity(producer['pid'])
                and (producer['pid'] == self.child.pid or producer['ppid'] == self.child.pid), 'Actual owned native/shim process identity')
        self.producer = producer; self.pid = producer['pid']
        self.known[(self.producer['pid'], self.producer['create_time'])] = self.producer
        lease.update(actual_GPU_running=True, actual_owned_worker_pid=self.pid, actual_owned_launcher_pid=self.child.pid,
            actual_training_optimizer_running=False, status='bounded_full_DATA_multiwindow_diagnostic', updated_at_utc=now()); write_json(LEASE, lease)
        self.exit_transition = None; self.exit_transition_began = None; self.sequence = 0; self.idle_proof = None; self.command_state = 'native_startup'; self.guard = ResourceGuard()
        self.samples = 0; self.peak = 0; self.last = None; self.monitor_error = None; self.reason = None; self.stop_at = None; self.admitted = set()
        self.done = threading.Event(); self.monitor = threading.Thread(target=self._watch, daemon=True); self.monitor.start()
        deadline = time.monotonic()+5
        while self.samples < 3:
            parent_guard.check()
            require(self.monitor.is_alive() and self.monitor_error is None and self.child.poll() is None
                    and time.monotonic() < deadline, 'Three fresh owned native samples before commands')
            time.sleep(.05)
        print(json.dumps({'actual_GPU_native_PID': self.pid, 'launcher_PID': self.child.pid, 'folder': str(folder)}), flush=True)

    def _watch(self):
        try:
            while not self.done.is_set() and self.child.poll() is None:
                self.parent_guard.check(); sample = sample_memory(self.pid, 'full_DATA_GPU_parent_fresh')
                if not sample['worker_private_memory'].get('available'):
                    from asr18_owned_family_observation_v4 import reaped_native_success_pending
                    self.exit_transition = reaped_native_success_pending(self.producer,
                        completion_path=self.folder/'worker_completion.json',worker_source=record(__file__),
                        parent_source=record(__file__),launcher=self.child)
                    if self.exit_transition_began is None:self.exit_transition_began=time.monotonic()
                    require(time.monotonic()-self.exit_transition_began<=10,'Reaped native launcher wait exceeded bounded transition')
                    fresh=sample_memory(os.getpid(),'native_reaped_launcher_pending_parent')
                    require(fresh['paired_sampling_within_one_second'] and fresh['system_memory']['available']
                        and fresh['worker_private_memory']['available'],'Fresh parent/system guard during actual launcher wait')
                    reason,_=self.guard.observe(fresh['system_memory'])
                    require(reason is None,'System guard failed during native-reaped launcher wait')
                    write_json(self.folder/'native_exit_wait_transition.json',{'transition':self.exit_transition,'fresh_parent':fresh,
                        'actual_launcher_exit_NOT_claimed':True})
                    self.done.wait(.1);continue
                require(sample['paired_sampling_within_one_second'] and sample['system_memory']['available']
                        and sample['worker_private_memory']['available'], 'Fresh available paired GPU-parent observation')
                reason, detail = self.guard.observe(sample['system_memory'], sample['worker_private_memory'])
                private = sample['worker_private_memory']['private_bytes']; self.samples += 1; self.peak = max(self.peak, private); self.last = time.monotonic()
                if private > self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']: reason = reason or 'actual_private_above_prospective_entire_worker_forecast'
                if reason is not None: raise ValueError(reason)
                parent = sample_memory(os.getpid(), 'full_DATA_native_observer_fresh')
                write_json(self.folder/'parent_live.json', {'sample': sample, 'parent': parent, 'guard': detail,
                    'command_state': self.command_state, 'concurrent_DATA': concurrent_data_observation()})
                if self.idle_proof is not None and self.command_state == 'coherent_idle_CPU_preparation':
                    from fastmss_owned_process_closure_v2 import identity
                    write_json(self.folder/'resident_live_witness.json', {'kind': 'ASR18_resident_native_live_guard_witness_v1',
                        'owner': OWNER,
                        'idle_proof': self.idle_proof, 'parent_source': record(__file__), 'producer': self.producer, 'observer': identity(),
                        'request_nonce': read_json(self.idle_proof['path'])['request_nonce'], 'command_state': self.command_state,
                        'resume_command_issued': False, 'parent_monitor_live': True, 'observed_monotonic': time.monotonic(),
                        'parent_guard_observation': parent, 'native_guard_observation': sample})
                request = self.folder/'stage_request.json'
                if request.exists():
                    value = read_json(request)
                    if value['name'] not in self.admitted:
                        result = fresh_admit(value['incremental_private_budget_bytes'], pid=self.pid)
                        observed_private = result['sample']['worker_private_memory']['private_bytes']
                        whole_projection = observed_private + value['incremental_private_budget_bytes']
                        passed = result['passed'] and whole_projection <= self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']
                        write_json(self.folder/'stage_admission.json', {'name': value['name'], 'passed': passed,
                            'fresh_sample': result['sample'], 'incremental_budget_bytes': value['incremental_private_budget_bytes'],
                            'projected_worker_private_bytes': whole_projection,
                            'whole_worker_private_allowance_bytes': self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']})
                        self.admitted.add(value['name']); lease = read_json(LEASE)
                        lease.update(actual_training_optimizer_running=passed and self.config['worker_mode'] == 'resident_training'
                                     and value['name'].startswith('full_DATA_native_update_'),
                                     diagnostic_phase=value['name'], updated_at_utc=now()); write_json(LEASE, lease)
                self.done.wait(1.)
        except BaseException:
            self.monitor_error = traceback.format_exc(); self.reason = 'native_parent_watchdog_failed'
            write_json(self.folder/'stop.json', {'reason': self.reason, 'error': self.monitor_error})

    def check(self):
        observer_sample = self.parent_guard.check()
        require(self.child.poll() is None and self.monitor_error is None and self.monitor.is_alive()
                and self.last is not None and time.monotonic()-self.last < 2.5, 'Owned native monitor/process failed; stop, no retry')
        return observer_sample

    def command(self, kind, *, window=None):
        self.check(); self.sequence += 1; nonce = secrets.token_hex(24); self.idle_proof = None
        self.command_state = 'native_running' if kind == 'run_prepared_window' else 'native_command_pending'
        value = {'sequence': self.sequence, 'nonce': nonce, 'kind': kind}
        if window is not None: value['window'] = window
        write_json(self.folder/'command.json', value); began = time.monotonic()
        while True:
            path = self.folder/'response.json'
            if path.exists():
                response = read_json(path)
                if response['sequence'] == self.sequence:
                    require(response['passed'] is True and response['nonce'] == nonce, 'Actual serial native response identity')
                    return response
            self.check()
            require(time.monotonic()-began < self.config['maximum_GPU_child_wall_seconds'], 'Finite native response timeout')
            time.sleep(.1)

    def seal_idle(self, response):
        from fastmss_owned_process_closure_v2 import identity
        self.check(); ack = read_json(response['idle_ack']['path'])
        require(record(response['idle_ack']['path']) == response['idle_ack'] and ack['producer'] == self.producer
                and ack['request_nonce'] == response['nonce'], 'Current live native nonce acknowledgment required')
        value = {**ack, 'observer': identity(), 'launcher_PID': self.child.pid, 'request_nonce_acknowledged': True,
            'parent_guard_observation': sample_memory(os.getpid(), 'seal_native_idle_parent'),
            'actual_GPU_process_closed': False, 'window_manifest': read_json(response['window_completion']['path'])['feature_manifest'],
            'live_witness_path': str(self.folder/'resident_live_witness.json')}
        path = self.folder/('resident_boundary_'+str(self.sequence)+'.json'); write_json(path, value)
        self.idle_proof = record(path); self.command_state = 'coherent_idle_CPU_preparation'
        deadline = time.monotonic()+3
        while not (self.folder/'resident_live_witness.json').exists() or read_json(self.folder/'resident_live_witness.json')['idle_proof'] != self.idle_proof:
            self.check(); require(time.monotonic() < deadline, 'Actual fresh parent idle witness required'); time.sleep(.05)
        return self.idle_proof

    def finish(self):
        from fastmss_owned_process_closure_v2 import observe
        self.command('finish_coherent_native_process')
        deadline = time.monotonic()+60
        while self.child.poll() is None:
            self.parent_guard.check(); require(time.monotonic() < deadline, 'Owned native exit timeout'); time.sleep(.1)
        code = self.child.wait()
        from asr18_owned_family_observation_v4 import wait_captured_dead
        retired = wait_captured_dead(self.known, timeout=15)
        self.done.set(); self.monitor.join(timeout=2); self.log.close()
        require(code == 0 and not self.monitor.is_alive() and self.monitor_error is None and self.samples >= 3,
                'Actual native exit0 and clean joined parent guard required')
        footer = self.folder/'native_guard_completion.json'
        write_json(footer, {'pid': self.pid, 'actual_exit_code': code, 'monitor_clean': True, 'monitor_joined': True,
            'observed': {'samples': self.samples, 'peak_private_bytes': self.peak, 'monitor_error': None},
            'guard_source': record(__file__), 'sampling_interval_seconds': 1.,
            'all_captured_GPU_identities_observed_dead': retired,
            'native_reaped_launcher_pending_transition':self.exit_transition})
        proof = observe(self.child, kind='ASR18_owned_GPU_process_completion_v1', owner=OWNER,
            report=record(self.folder/'worker_completion.json'), worker_source=record(__file__), parent_source=record(__file__),
            guard_source=record(__file__), guard_closure=record(footer), launcher_identity=self.launcher,
            parent_guard_observation=sample_memory(os.getpid(), 'native_actual_wait_observer'),
            output=self.folder/'actual_owned_native_process0.json', guard=self.parent_guard.check)
        self._release_lease(); return proof

    def _release_lease(self):
        lease = read_json(LEASE); require(lease['owner'] == OWNER and lease.get('actual_owned_worker_pid') in (None, self.pid), 'Exact owned lease release')
        lease.update(actual_GPU_running=False, actual_owned_worker_pid=None, actual_owned_launcher_pid=None,
            actual_training_optimizer_running=False, status='memory_diagnostic_idle', updated_at_utc=now()); write_json(LEASE, lease)

    def abort(self):
        write_json(self.folder/'stop.json', {'reason': 'owned_parent_failure_preserve_partial'})
        from asr18_owned_family_observation_v4 import same_process
        deadline=time.monotonic()+60
        while (self.child.poll() is None or same_process(self.producer)) and time.monotonic()<deadline:time.sleep(.1)
        if same_process(self.producer):
            import psutil
            try:psutil.Process(self.pid).terminate()
            except psutil.NoSuchProcess:pass
        if self.child.poll() is None:self.child.terminate()
        self.child.wait(timeout=15)
        from asr18_owned_family_observation_v4 import wait_captured_dead
        observed_dead=wait_captured_dead({(self.producer['pid'],self.producer['create_time']):self.producer},timeout=15)
        write_json(self.folder/'failed_native_actual_identity_retirement.json',observed_dead)
        self.done.set(); self.monitor.join(timeout=2); self.log.close(); self._release_lease()


def run_CPU(config, script, arguments, folder, *, guard, report_relative='report.json'):
    """One actual bounded CPU controller/cleanup child, source-bound Popen proof."""
    from fastmss_owned_process_closure_v2 import identity, observe
    folder.mkdir(exist_ok=False); monitor = folder/'guard'; log = folder/'worker.log'; phase_began=time.monotonic(); known={}; family_peak=0; family_samples=0; outer_seen=False
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '-1'
    require(record(config['CPU_python']['path']) == config['CPU_python']
            and record(config['CPU_guard_source']['path']) == config['CPU_guard_source'], 'Pinned CPU interpreter/guard before launch')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
    admission = fresh_admit(config['CPU_owned_allowance_bytes']); write_json(folder/'admission.json', admission)
    require(admission['passed'], 'Fresh CPU allowance does not fit; preserve, no DATA hold')
    with log.open('xb') as stream:
        child = subprocess.Popen([config['CPU_python']['path'], '-X', 'utf8', '-B', config['CPU_guard_source']['path'],
            '--folder', str(monitor), '--seconds', str(config['maximum_CPU_phase_seconds']), '--script', str(script), '--', *arguments],
            env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = identity(child.pid); began = time.monotonic()
        known[(launcher['pid'],launcher['create_time'])] = launcher
        exit_transition = None
        try:
            from asr18_owned_family_observation_v4 import owned_family_sample
            from asr18_outer_CPU_exit_transition_v1 import wait_actual_CPU_family_exit
            while child.poll() is None:
                guard(); require(time.monotonic()-began < config['maximum_CPU_phase_seconds']+10, 'Finite CPU controller timeout')
                try:family=owned_family_sample(child,launcher,known=known,guard=guard)
                except ProcessLookupError:
                    # A launcher syscall can race normal exit before poll sees0.
                    # Drain captured children under their own unchanged cap;
                    # the outer native parent retains its separate ParentGuard.
                    exit_transition = wait_actual_CPU_family_exit(child,launcher,known,guard=guard,
                        CPU_family_private_allowance_bytes=config['CPU_owned_allowance_bytes'])
                    break
                family_samples+=1;family_peak=max(family_peak,family['private_bytes_sum'])
                outer_seen=outer_seen or family['outer_launcher_included']
                write_json(folder/'owned_CPU_family_live.json',family)
                require(family['private_bytes_sum']<=config['CPU_owned_allowance_bytes'],'Whole CPU family private allowance exceeded, including outer launcher')
                time.sleep(.2)
            require(child.wait() == 0, 'Actual CPU phase failed; artifacts preserved, no unchanged retry')
            if exit_transition is None:
                exit_transition = wait_actual_CPU_family_exit(child,launcher,known,guard=guard,
                    CPU_family_private_allowance_bytes=config['CPU_owned_allowance_bytes'])
            family_peak=max(family_peak,exit_transition['captured_CPU_family_private_peak_bytes'])
            write_json(folder/'owned_CPU_family_exit_transition.json',exit_transition)
        except BaseException:
            from asr18_owned_family_observation_v4 import terminate_known_family
            monitor.mkdir(exist_ok=True); write_json(monitor/'stop.json', {'reason': 'owned_parent_failed'})
            if child.poll() is None:
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            # Even an exited outer launcher can leave captured descendants.
            # Observe their termination before considering this family retired.
            terminate_known_family(child,launcher,known)
            raise
    require(family_samples>=1 and outer_seen,'Actual outer launcher+CPU-family telemetry required; no guessed zero overhead')
    write_json(folder/'owned_CPU_family_completion.json',{'passed':True,'measured_family_peak_private_bytes':family_peak,
        'family_samples':family_samples,'reviewed_entire_CPU_allowance_bytes':config['CPU_owned_allowance_bytes'],
        'inner_controller_family_allowance_bytes':config['CPU_inner_owned_allowance_bytes'],
        'additional_outer_launcher_reserve_bytes':config['CPU_outer_controller_launcher_reserve_bytes'],
        'outer_launcher_included':True,'actual_parent_phase_wall_seconds':time.monotonic()-phase_began,
        'launcher_disappearance_or_normal_exit_drain':exit_transition,
        'only_actual_owned_ancestry_observed':True})
    report = record(folder/report_relative)
    return observe(child, kind='ASR18_owned_CPU_process_completion_v1', owner=OWNER, report=report,
        worker_source=record(script), parent_source=record(__file__), guard_source=config['CPU_guard_source'],
        guard_closure=record(monitor/'closure.json'), launcher_identity=launcher,
        parent_guard_observation=sample_memory(os.getpid(), 'CPU_actual_wait_observer'), output=folder/'actual_owned_process0.json', guard=guard)


def cleanup_child(request_path, output):
    storage(); request = read_json(request_path)
    from fastmss_owned_process_closure_v2 import identity, verify_resident_idle, verify
    from fastmss_phase_aware_feature_window_v6 import cleanup_window, read_bound
    def guard():
        sample = sample_memory(os.getpid(), 'owned_cleanup_callback'); require(sample['paired_sampling_within_one_second'], 'Cleanup fresh telemetry'); return sample
    boundary = request['boundary']; pointer = read_bound(boundary['coherent_checkpoint_pointer'], guard)
    def verify_boundary(value):
        require(digest(value) == digest(boundary), 'Exact coherent cleanup boundary')
        if 'resident_native_boundary' in value:
            verify_resident_idle(read_bound(value['resident_native_boundary'], guard), pointer=pointer,
                manifest=request['manifest'], phase=value['phase'], guard=guard, proof_binding=value['resident_native_boundary'])
        else:
            proof = read_bound(value['actual_native_closed'], guard)
            verify(proof, kind='ASR18_owned_GPU_process_completion_v1', owner=OWNER, guard=guard)
        require(pointer['saved_semantic_state_verified']['passed'] is True, 'Actual saved scalar state verified')
        require(record(pointer['payload']['path']) == pointer['payload'], 'Final checkpoint payload unchanged before owned cleanup')
        return True
    result = cleanup_window(request['owner'], closures=request['closures'], manifest_binding=request['manifest'],
        boundary=boundary, verify_boundary=verify_boundary, evidence_path=request['archive_path'], guard=guard)
    result.update(producer=identity(), worker_source=record(__file__), parent_source=request['parent_source'])
    write_json(output/'report.json', result)


def controller(config_path,review_path,folder):
    storage();config=read_json(config_path);check_bindings(config)
    require(config['kind']=='ASR18_hard_effective16_CPU8_native_diagnostic_config_v1'
            and config['finite_diagnostic_execution_enabled'] is True and config['TEMPLATE_NOT_LAUNCHABLE'] is False
            and config['production_enabled'] is False and config['diagnostic_only'] is True
            and config['maximum_windows']==config['execute_window_limit']==config['maximum_updates_per_window']==config['maximum_total_native_updates']==1
            and config['worker_driver_source']==str(Path(__file__).resolve()) and config['parent_driver_source']==record(__file__),
            'One exact original-source hard effective16 diagnostic, never production')
    from asr18_full_population_separated_forecast_v1 import restore_only_worker_forecast
    require(config['worker_mode']=='resident_training' and config['all_captured_checkpoints_require_separate_final_GPU_recovery'] is True
            and config['entire_worker_forecast']['native_private_forecast_bytes']==6_684_646_272
            and config['entire_worker_forecast']['checkpoint_write_increment_bytes']==64_000_000
            and config['entire_worker_forecast']['entire_worker_private_forecast_bytes']==6_748_646_272
            and digest(config['final_recovery_worker_forecast'])==digest(restore_only_worker_forecast()),'Conservative full hard forecast; no component/private credit')
    review=read_json(review_path)
    require(review['accepted'] is True and review['scope']=='finite_hard_effective16_CPU8_native_fresh_recovery_diagnostic_v1'
            and review['production_approval_claimed'] is False and review['source_manifest']==config['source_manifest']
            and review['configuration']==record(config_path),'Exact ROOT source/resource/configuration review')
    require(not folder.exists(),'Fresh hard effective16 namespace');folder.mkdir()
    write_json(folder/'launch.json',{'config':record(config_path),'review':record(review_path),'production_enabled':False})
    parent=ParentGuard(folder);gpu=None;actuals=[];windows=[];began=time.monotonic()
    from fastmss_phase_aware_plan_bridge_v7 import bridge_plan,read_bound
    from fastmss_owned_process_closure_v2 import verify_resident_idle,verify as verify_process
    from fastmss_row_serial_cpu_v8 import verify_raw_review,verify_contract
    try:
        from asr18_hard_envelope_schedule_context_v1 import HardEnvelopeContext
        context=HardEnvelopeContext(config['planning_arguments'],config['hard_row_provenance'],guard=parent.check)
        try:
            initial=context.new_schedule();plan=context.plan(initial,maximum_updates=1)
            plan['hard_envelope_diagnostic_context']=config['hard_envelope_diagnostic_context']
            require(initial.committed_updates()==0 and initial.broad.draws==0 and digest(initial.state_dict())==config['initial_sampler_sha256'],
                    'Original0 schedule/RNG and honest explicit hard diagnostic cursor')
        finally:context.close()
        require(digest(plan)==digest(read_bound(config['provisional_plan'],parent.check)),'Exact pre-audio metadata-bridged plan')
        session=read_bound(config['session'],parent.check);contract=verify_contract(session['CPU_preparation_contract'],session,guard=parent.check)
        CPU_review=read_bound(config['actual_ROOT_CPU_resource_review'],parent.check)
        require(CPU_review['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and CPU_review['reviewer']=='/root'
                and CPU_review['accepted'] is True and CPU_review['scope']=='CPU_preparation_only'
                and CPU_review['outer_parent_source']==record(__file__) and CPU_review['CPU_source_manifest']==check_bindings(config)['CPU_owner_bundle']
                and CPU_review['production_enabled'] is False,'Exact new CPU8 source/resource binding')
        verify_raw_review(contract['raw_lifetime_review'])
        require(contract['actual_ROOT_CPU_resource_review']==config['actual_ROOT_CPU_resource_review']
                and digest(contract['resource_review'])==digest(CPU_review['resource_review'])
                and digest(contract['raw_lifetime_review'])==digest(CPU_review['raw_lifetime_review'])
                and config['CPU_owned_allowance_bytes']==1_934_217_728
                and config['CPU_owned_allowance_bytes']==contract['resource_review']['whole_owned_private_allowance_bytes']+contract['resource_review']['outer_controller_launcher_reserve_bytes']
                and contract['outer_parent_source']==record(__file__), 'No changed CPU/raw/margin allowance')
        require(session['mode']=='one_hard_envelope_diagnostic' and session['initial_optimizer_updates']==0
                and session['hard_envelope_diagnostic_context']==config['hard_envelope_diagnostic_context']
                and session['initial_sampler_sha256']==plan['initial_sampler_sha256'],'One unqualified hard diagnostic session')
        pp=folder/'coordinator_plan_0.json';write_json(pp,plan)
        bridge=bridge_plan(record(pp),DATA_evidence=config['planning_arguments']['DATA_evidence'],session_binding=config['session'],
            window_index=0,prior_window_boundary=None,guard=parent.check)
        bp=folder/'bridge_0.json';write_json(bp,bridge);ba=folder/'bridge_0_binding.json';write_json(ba,record(bp))
        cpu_folder=folder/'CPU_window_0'
        CPU0=run_CPU(config,config['CPU_controller_source']['path'],['--bridge',str(ba),'--output',str(cpu_folder/'payload')],
            cpu_folder,guard=parent.check,report_relative='payload/report.json');actuals.append(CPU0)
        CPU=read_json(cpu_folder/'payload/report.json');require(CPU['passed'] is True and CPU['all_children_actually_closed0'] is True
            and CPU['bridge']==record(bp),'Actual complete CPU8 independent references/frontend/finalization0')
        manifest=read_bound(CPU['manifest'],parent.check);prepared=read_bound(manifest['updates'][0]['update'],parent.check)
        hard=prepared['rows'][13]
        require(len(prepared['rows'])==16 and hard['source_id']=='kiraat:src01792-00990' and hard['encoder_frames']==1991
                and len(hard['full_target_ids'])==574 and hard['samples']==2547840
                and hard['input_features']['shape']==[1,15925,128]
                and prepared['prepared_payload']['rows'][13]['augmentation']['applied'] is False,
                'Complete hard original audio/targets/physical frames clean over120s plus13TR/exact2replay')
        child={**config,'restore_parent_pointer':None,'first_feature_update':manifest['updates'][0]['update']}
        cp=folder/'GPU_config_0.json';write_json(cp,child)
        gpu=GPUService.launch(cp,folder/'GPU_process_0',parent)
        while not(gpu.folder/'worker_ready.json').exists():gpu.check();time.sleep(.1)
        response=gpu.command('run_prepared_window',window={'coordinator_plan':record(pp),'feature_owner':CPU['owner'],
            'feature_manifest':CPU['manifest'],'session':config['session']})
        idle=gpu.seal_idle(response);complete=read_bound(response['window_completion'],parent.check);windows.append(complete)
        pointer_pin=complete['latest_pointer'];pointer=read_bound(pointer_pin,parent.check)
        require(complete['actual_new_updates']==complete['global_new_updates']==pointer['committed_updates']==1
                and complete['hard_envelope_effective16_diagnostic'] is True and complete['all_checkpoint_persistence_verified'] is True
                and complete['exact_GPU_restores_deferred'] is True,'Actual one effective16 commit/persistence, not overall recovery')
        verify_resident_idle(read_bound(idle,parent.check),pointer=pointer,manifest=CPU['manifest'],phase=plan['phase'],guard=parent.check,proof_binding=idle)
        native0=gpu.finish();actuals.append(native0);gpu=None
        boundary={'kind':'coherent_checkpoint_boundary','owner_sha256':CPU['owner']['sha256'],'session':config['session'],
            'phase':plan['phase'],'committed_window_updates':1,'window_manifest_sha256':CPU['manifest']['sha256'],
            'coherent_checkpoint_pointer':pointer_pin,'checkpoint':pointer['checkpoint'],
            'deferred_restore_required_before_overall_diagnostic_pass':True,'actual_native_closed':native0}
        cr=folder/'cleanup_request_0.json';write_json(cr,{'owner':CPU['owner'],'manifest':CPU['manifest'],'closures':CPU['closures'],
            'boundary':boundary,'archive_path':str(folder/'window_0_durable_feature_metadata.json'),'parent_source':record(__file__)})
        cleanup_folder=folder/'cleanup_0';cleanup0=run_CPU({**config,'maximum_CPU_phase_seconds':config['maximum_cleanup_phase_seconds']},
            Path(__file__).resolve(),['--cleanup','--request',str(cr),'--folder',str(cleanup_folder)],cleanup_folder,guard=parent.check)
        actuals.append(cleanup0)
        request={'kind':'ASR18_hard_effective16_all_checkpoint_GPU_recovery_requests_v1','owner':OWNER,'production_enabled':False,
            'session':config['session'],'parent_source':record(__file__),'native_engine_source':record(SOURCE/'asr18_hard_effective16_native_v1.py'),
            'expected_global_new_updates':1,'native_process_completions':[native0],'native_window_completions':[response['window_completion']],
            'checkpoints':complete['checkpoints'],'hard_row_provenance':config['hard_row_provenance']}
        rp=folder/'all_actual_native_checkpoints_pending_recovery.json';write_json(rp,request)
        recovery_config={**config,'worker_mode':'restore_only','deferred_recovery_requests':record(rp),
            'entire_worker_forecast':restore_only_worker_forecast(),'maximum_GPU_child_wall_seconds':config['maximum_final_recovery_seconds']}
        rc=folder/'GPU_final_recovery_config.json';write_json(rc,recovery_config)
        gpu=GPUService.launch(rc,folder/'GPU_recovery_process',parent)
        while not(gpu.folder/'worker_ready.json').exists():gpu.check();time.sleep(.1)
        recovered=gpu.command('restore_all_deferred_checkpoints',window={'request':record(rp)})
        all_recovery=read_bound(recovered['recovery_report'],parent.check)
        require(all_recovery['passed'] is True and all_recovery['all_captured_checkpoints_recovered'] is True
                and all_recovery['actual_native_updates_in_recovery_process']==0 and all_recovery['actual_draws_consumed_in_recovery']==0
                and all_recovery['actual_global_native_updates']==1 and len(all_recovery['recovered_checkpoints'])==1,'Every new hard capture exact fresh recovery')
        recovery0=gpu.finish();actuals.append(recovery0);gpu=None
        for proof_pin in actuals:
            proof=read_bound(proof_pin,parent.check);verify_process(proof,kind=proof['kind'],owner=OWNER,guard=parent.check)
        result={'kind':'ASR18_actual_hard_effective16_CPU8_native_fresh_recovery_diagnostic_v1','passed':True,
            'actual_native_updates':1,'actual_global_updates':1,'final_mature_clocks':[64251,2001,62251],
            'hard_original_ordinal':1051133,'hard_native_microbatch_index':14,'hard_T':1991,'hard_U':574,'hard_samples':2547840,
            'windows':windows,'actual_process_completions':actuals,'actual_CPU_process0':CPU0,'actual_native_process0':native0,
            'actual_cleanup_process0':cleanup0,'actual_recovery_process0':recovery0,'all_captured_checkpoints_actual_fresh_GPU_recovery':recovered['recovery_report'],
            'all_deferred_recovery_obligations_discharged':True,'current_checkpoint_pointer':pointer_pin,
            'actual_languages':[row['language'] for row in prepared['rows']],'hard_row_provenance':config['hard_row_provenance'],
            'full_original_source_schedule_retained_as_immutable_ancestry':True,'production_phase_advance_claimed':False,
            'production_enabled':False,'diagnostic_only':True,'qualified_checkpoint':False,'wall_seconds':time.monotonic()-began}
        write_json(folder/'report.json',result);print(json.dumps({'passed':True,'actual_updates':1,'report':record(folder/'report.json')}),flush=True)
    except BaseException:
        write_json(folder/'failure.json',{'passed':False,'error':traceback.format_exc(),'completed_windows':windows,
            'actual_process_completions':actuals,'actual_native_progress':read_json(gpu.folder/'native_progress.json')
                if gpu is not None and(gpu.folder/'native_progress.json').exists() else None,
            'production_enabled':False,'overall_diagnostic_success_claimed':False})
        raise
    finally:
        if gpu is not None:gpu.abort()
        parent.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--review');p.add_argument('--folder',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--recovery-worker',action='store_true');p.add_argument('--cleanup',action='store_true');p.add_argument('--request')
    a=p.parse_args();folder=scoped(a.folder)
    if a.worker:worker(a.config,folder)
    elif a.recovery_worker:
        from asr18_hard_effective16_restore_runtime_v1 import run
        run(a.config,folder,parent_source=record(__file__),check_bindings=check_bindings,storage=storage,stage=stage,admit_stage=admit_stage)
    elif a.cleanup:cleanup_child(a.request,folder)
    else:controller(a.config,a.review,folder)
