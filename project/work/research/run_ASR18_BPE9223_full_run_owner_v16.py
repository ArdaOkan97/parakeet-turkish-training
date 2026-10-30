"""Inactive full-run owner supervision; exact reviewed bounded validation only.
The native runtime, metadata phases and real process identities are distinct.
No CLI arms long production or promotes a diagnostic state.
"""
import argparse, copy, ctypes, gc, json, os, secrets, shutil, subprocess, sys, threading, time, traceback
from ctypes import wintypes
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'outputs/parakeet_turkish'; sys.path.insert(0, str(SOURCE))
ROOT = Path(r'E:\gpu_training_scratchpad'); OWNER = '/root/memory_diagnosis'
LEASE = PROJECT/'work/gpu_training_owner.json'
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from asr18_memory_diagnosis_v2 import storage, now, stage, admit_stage, concurrent_data_observation, terminate_runtime_worker
from resource_watchdog_v2 import ResourceGuard, sample_memory, PROCESS_MEMORY_COUNTERS_EX


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


def private_lifetime_limit(cap):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    value=PROCESS_MEMORY_COUNTERS_EX();value.cb=ctypes.sizeof(value)
    require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb),'Owned current/lifetime telemetry')
    require(value.PrivateUsage<=cap and value.PeakPagefileUsage<=cap,'Owned current AND lifetime private limit')
    return {'current_private_bytes':int(value.PrivateUsage),'OS_lifetime_private_peak_bytes':int(value.PeakPagefileUsage)}

def checked_stage(folder,name,torch,*,cap,**extra):
    value=stage(folder,name,torch,**extra)
    require(value['worker_private_memory']['private_bytes']<=cap and value['worker_private_peak_bytes_lifetime']<=cap,
            'Actual current/lifetime private must fit separately admitted whole worker')
    require(value.get('cuda',{}).get('peak_allocated_bytes',0)<=12_000_000_000,'Human-authorized decimal12GB CUDA ceiling')
    return value


CONTROL_POLICY={'kind':'ASR18_full_run_nonjournal_control_log_quota_v1',
    'allocation_cluster_bytes':524288,'total_allocated_cap_bytes':4294967296,
    'fixed_control_allocation_cap_bytes':1073741824,'maximum_GPU_worker_directories':48,
    'reserved_each_GPU_worker_allocation_bytes':67108864,'maximum_worker_metadata_files':32,
    'maximum_worker_directories':8,'ordinary_metadata_file_cap_bytes':1048576,
    'stage_log_cap_bytes':8388608,'stdout_log_cap_bytes':8388608,
    'prospective_atomic_replace_reserve_bytes':2097152}

def allocated(size):
    return ((max(int(size),1)+524287)//524288)*524288

def worker_control_guard(folder):
    # Only fixed startup/footer/current IPC files live here. Every native row
    # stage and window ACK/boundary lives in the separately reserved AUX tree.
    total=0;files=0;directories=0
    for root,dirs,names in os.walk(folder):
        for name in dirs:
            path=Path(root)/name;require(not path.is_symlink(),'No control directory aliases')
            directories+=1;total+=524288
        require(directories<=CONTROL_POLICY['maximum_worker_directories'],'Bounded worker control directories')
        for name in names:
            path=Path(root)/name
            try:info=path.lstat()
            except FileNotFoundError:
                require('.atomic-v1-' in name and name.endswith('.tmp'),'Only exact atomic temporary may disappear during observation');continue
            require(__import__('stat').S_ISREG(info.st_mode) and not path.is_symlink(),'Only exact regular worker control files')
            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.tmp'),
                    'No tensor/audio/model/unknown files in control-log domain')
            size=info.st_size;limit=(8388608 if name in ('stages.jsonl','worker.log') else 1048576)
            require(size<=limit,'Source-declared per-file control bound')
            files+=1;total+=allocated(size)
        require(files<=CONTROL_POLICY['maximum_worker_metadata_files'],'Bounded source control file count')
    require(total+CONTROL_POLICY['prospective_atomic_replace_reserve_bytes']<=67108864,
            'Cluster-rounded entire worker control reservation, including transient replacement')
    return total

# The existing worker pool owner binds the original parent2 authority.
# Inherit its actual code objects and callbacks; no authority/source rewrite.
import run_ASR18_BPE9223_full_run_owner_v2 as frozen_parent
import run_ASR18_BPE9223_full_run_owner_v6 as frozen_CPU_executor
class ParentGuard(frozen_parent.ParentGuard):
    def configure_control_quota(self,control,policy,*,identity,create,operational_compatibility):
        require(policy==CONTROL_POLICY,'Exact non-journal quota, no budget substitution')
        self.control=Path(control).resolve()
        require(self.control.is_relative_to(ROOT/'runs'),'Exact original owned sibling control root')
        from asr18_worker_control_pool_v2 import WorkerControlPool
        self.control_pool=WorkerControlPool(self.control,identity=identity,policy=policy,
            authority_source=record(frozen_parent.__file__),guard=self.pool_guard,metadata_phase=self.metadata_phase,
            admit_control=self.control_guard,operational_compatibility=operational_compatibility,create=create)
        self.worker_slots=list(self.control_pool.live_folders())
        self.control_guard()

    def configure_merged_CPU(self,policy,operational_compatibility):
        from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility
        review=checked_compatibility(operational_compatibility,self.control_pool.identity,self.check)
        expected={'kind':'ASR18_merged_supervisor_CPU_orchestration_resource_domain_v1',
            'merged_parent_current_cap_bytes':128000000,'leaf_maximum_private_bytes':1800000000,
            'each_shim_private_reserve_bytes':33554432,'inner_including_parent_bytes':1961554432,
            'child_subtree_private_allowance_bytes':1867108864,'whole_owned_CPU_allowance_bytes':1995108864,
            'separate_CPU_orchestrator_process_exists':False,'original_global_CPU_hard_cap_bytes':2000000000}
        require(policy==expected and review['merged_CPU_topology']==expected
            and review['operational_sources']['parent']==record(__file__),
            'Explicit ROOT reviewed role merge under same2GB; no leaf/workspace cap change')
        self.merged_CPU_policy=copy.deepcopy(policy);self.merged_CPU_active=False

    @__import__('contextlib').contextmanager
    def metadata_phase(self,phase):
        if getattr(self,'merged_CPU_active',False):
            require(phase=='read_index' and self.phase=='merged_CPU' and self.current_cap==128000000
                and self.phase_depth>=1,'Only nested readonly metadata at SAME merged128M; no archive expansion')
            self.phase_depth+=1
            try:yield self.check()
            finally:self.phase_depth-=1;self.check()
        else:
            with super().metadata_phase(phase) as value:yield value

    @__import__('contextlib').contextmanager
    def CPU_phase(self):
        require(hasattr(self,'merged_CPU_policy') and self.phase=='base' and self.phase_depth==0,
            'Exact reviewed merged-role CPU phase, no overlapping metadata compressor')
        before=self.check();require(before['worker_private_memory']['private_bytes']<=128000000,
            'Actual merged parent<=128M; separate64MiB CPU process eliminated, total<2GB')
        self.phase='merged_CPU';self.current_cap=128000000;self.phase_depth=1;self.merged_CPU_active=True
        self.check()
        try:yield
        finally:
            self.check();self.merged_CPU_active=False;self.phase_depth=0;self.phase='base';self.current_cap=self.CAPS['base'];self.check()



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
                early=read_json(candidate.folder/'entrypoint_boot.json')
                if early is not None:
                    require(early['source']==candidate.config['worker_driver_source'] and early['configuration']==candidate.configuration_binding
                        and early['parent_source']==record(__file__),'Exact early startup provenance before failure retirement')
                    entry=early['producer'];require(entry['pid']==child.pid or entry['ppid']==child.pid,'Owned early startup ancestry')
                    known[(entry['pid'],entry['create_time'])]=entry
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
            sink=getattr(candidate,'stdout_thread',None)
            if sink is not None:sink.join(timeout=2)
            log = getattr(candidate, 'log', None)
            if log is not None: log.close()
            if getattr(candidate, 'pid', None) is not None: candidate._release_lease()
            raise

    def __init__(self, config_path, folder, parent_guard):
        from fastmss_owned_process_closure_v2 import identity
        self.folder = folder; self.parent_guard = parent_guard; self.config = read_json(config_path)
        require(self.config['control_log_policy']==CONTROL_POLICY,'Same source-bound reserved control quota')
        self.configuration_binding = record(config_path)
        self.completion_path=folder/('serial350_worker_report.json' if self.config['worker_mode']=='serial350DEV_optimizer_free' else
            'actual_current_recovery/current_recovery_report.json' if self.config['worker_mode']=='restore_only' else 'worker_completion.json')
        from asr18_BPE9223_geometry_forecast_v1 import CalibratedGeometryForecast
        from asr18_BPE9223_geometry_forecast_v1 import resident_worker_forecast
        from asr18_full_population_separated_forecast_v1 import restore_only_worker_forecast
        require(self.config['worker_mode'] in ('resident_training', 'restore_only','serial350DEV_optimizer_free'), 'Exact separated GPU worker mode')
        if self.config['worker_mode'] == 'resident_training':
            calibration = CalibratedGeometryForecast(**self.config['forecast_calibration'], guard=parent_guard.check)
            first_inputs = read_json(self.config['first_feature_update']['path'])
            require(record(self.config['first_feature_update']['path']) == self.config['first_feature_update'], 'Actual first closed feature input changed')
            expected = resident_worker_forecast(calibration.budget(rows=first_inputs['rows'], current_private_bytes=3_000_000_000, update_in_process=1))
        elif self.config['worker_mode']=='serial350DEV_optimizer_free':
            from asr18_full_run_DEV_runtime_9223_v3 import resource_proposal,checked_configuration
            checked_configuration(self.config,parent_source=record(__file__))
            require(self.config['worker_driver_source']==record(SOURCE/'asr18_full_run_DEV_runtime_9223_v3.py'),
                    'DEV launches the actual standalone runtime4 entrypoint')
            expected={'entire_worker_private_forecast_bytes':resource_proposal()['whole_worker_private_forecast_bytes']}
        else:
            from asr18_BPE9223_current_recovery_v5 import checked_request
            checked_request(self.config['current_recovery_request'],guard=parent_guard.check)
            expected = restore_only_worker_forecast()
        require(digest(expected) == digest(self.config['entire_worker_forecast']), 'Full worker forecast must match actual source-bound policy, not arbitrary config bytes')
        require(record(self.config['GPU_python']['path']) == self.config['GPU_python'], 'Existing GPU runtime executable changed')
        full = self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']
        own=parent_guard.check()['worker_private_memory']['private_bytes']
        admission = fresh_admit(full+max(384_000_000-own,0)+64_000_000); write_json(folder.parent/(folder.name+'_prelaunch_full_forecast.json'), admission)
        require(admission['passed'], 'Entire GPU worker prospective forecast does not fit; no GPU launch')
        lease = read_json(LEASE); require(lease['owner'] == OWNER and not lease['actual_GPU_running'] and not lease.get('actual_owned_worker_pid'), 'Exclusive actual idle GPU lease')
        parent_guard.reserve_worker_control(folder,configuration=self.configuration_binding,
            worker_source=self.config['worker_driver_source'],parent_source=record(__file__))
        folder.mkdir(exist_ok=False); self.log = (folder/'worker.log').open('xb');self.log_error=None
        environment = dict(os.environ); environment['CUDA_VISIBLE_DEVICES'] = '0'; environment['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'] = '16'
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): environment[key] = '1'
        require(record(self.config['worker_driver_source']['path'])==self.config['worker_driver_source'],'Exact separate GPU runtime source')
        self.child = subprocess.Popen([self.config['GPU_python']['path'], '-X', 'utf8', '-B', self.config['worker_driver_source']['path'],
            '--config', str(config_path), '--folder', str(folder)], env=environment, cwd=PROJECT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.stdout_bytes=0
        def bounded_stdout():
            try:
                while True:
                    block=self.child.stdout.read(16384)
                    if not block:break
                    require(self.stdout_bytes+len(block)<=CONTROL_POLICY['stdout_log_cap_bytes'],
                            'Prospective stdout chunk quota; no dropping or truncation')
                    self.log.write(block);self.log.flush();self.stdout_bytes+=len(block)
            except BaseException:
                self.log_error=traceback.format_exc()
                write_json(folder/'stop.json',{'reason':'owned_stdout_control_quota_failed','error':self.log_error})
            finally:self.child.stdout.close()
        self.stdout_thread=threading.Thread(target=bounded_stdout,daemon=True);self.stdout_thread.start()
        self.launcher = identity(self.child.pid)
        self.known = {(self.launcher['pid'], self.launcher['create_time']): self.launcher}
        deadline = time.monotonic()+15
        while not (folder/'entrypoint_boot.json').exists() and self.child.poll() is None and time.monotonic() < deadline:
            from asr18_owned_family_observation_v4 import owned_family_sample
            startup_family=owned_family_sample(self.child, self.launcher, known=self.known, guard=parent_guard.check)
            require(startup_family['private_bytes_sum']<=full+64_000_000 and all(item['private']['private_bytes']<=full for item in startup_family['measured_processes']),
                    'Native startup domain unresolved: whole family bounded without mislabeling native as shim')
            time.sleep(.1)
        boot = read_json(folder/'entrypoint_boot.json'); producer = boot['producer']
        require(boot['source'] == self.config['worker_driver_source'] and boot['configuration']==self.configuration_binding and boot['parent_source']==record(__file__) and producer == identity(producer['pid'])
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
        deadline=time.monotonic()+30
        while not (folder/'worker_boot_identity.json').exists():
            self.check();require(time.monotonic()<deadline,'Bound runtime identity after source validation');time.sleep(.1)
        runtime_boot=read_json(folder/'worker_boot_identity.json')
        require(runtime_boot==boot,'Early entrypoint and runtime identity/configuration/source must agree')
        if self.config['worker_mode']=='serial350DEV_optimizer_free':
            require(Path(self.config['owned_worker_admission_path']).resolve()==folder/'owned_worker_admission.json',
                    'Same actual source/PID-bound standalone DEV admission path')
            admissions=[]
            for _ in range(3):
                context=self._family_guard();observed=sample_memory(self.pid,'DEV_actual_preimport_admission')
                require(observed['paired_sampling_within_one_second'] and observed['worker_private_memory']['available'],
                        'Actual DEV private before prospective remaining whole-worker budget')
                remaining=max(full-observed['worker_private_memory']['private_bytes'],0)
                admitted=fresh_admit(max(remaining+context['remaining_parent_and_allshim_allowance_bytes'],1),pid=self.pid)
                require(admitted['passed'],'Three fresh whole DEV-family admissions before ML import')
                admissions.append(admitted);time.sleep(.2)
            write_json(folder/'owned_worker_admission.json',{
                'kind':'ASR18_sole_owned_optimizer_free_serialDEV_worker_admission_v1','owner':OWNER,
                'sole_owned_GPU_child':True,'actual_training_optimizer_running':False,
                'worker_source':self.config['worker_driver_source'],
                'optimizer_objects_constructed_or_loaded_in_DEV_worker':False,
                'producer':self.producer,'request':self.config['request'],
                'ROOT_execution_review':self.config['ROOT_execution_review'],'ROOT_resource_review':self.config['ROOT_resource_review'],
                'whole_worker_fresh_three_sample_admission_passed':True,'whole_worker_forecast_bytes':full,
                'configuration':self.configuration_binding,'parent_source':record(__file__),'samples':admissions})
            self.command_state='autonomous_serial350DEV'
        print(json.dumps({'actual_GPU_native_PID': self.pid, 'launcher_PID': self.child.pid, 'folder': str(folder)}), flush=True)

    def _family_guard(self):
        from asr18_owned_family_observation_v4 import owned_family_sample,same_process,reaped_native_success_pending
        own=self.parent_guard.check();worker_control_guard(self.folder)
        require(self.log_error is None,'Bounded stdout sink failed; no silent dropped output')
        try:
            family=owned_family_sample(self.child,self.launcher,known=self.known,guard=self.parent_guard.check)
        except ProcessLookupError:
            require(not same_process(self.producer),'Live native orphan cannot use a launcher-disappearance success transition')
            if self.child.poll() is None:
                reaped_native_success_pending(self.producer,completion_path=self.completion_path,
                    worker_source=self.config['worker_driver_source'],parent_source=record(__file__),launcher=self.child)
            else:
                require(type(self.child.returncode) is int and self.child.returncode==0,'Failed launcher cannot become a normal family exit')
                report=read_json(self.completion_path)
                require(report['passed'] is True and report['producer']==self.producer
                        and report['worker_source']==self.config['worker_driver_source'] and report['parent_source']==record(__file__),
                        'Source-bound actual native report at real handle0; final wait/guard/dead proof is still mandatory')
            family={'measured_processes':[],'launcher_disappearance_is_not_actual_wait0':True}
        measured={(part['identity']['pid'],part['identity']['create_time']):part for part in family['measured_processes']}
        # Count ALL captured still-live identities, including a reparented/draining shim.
        for key,identity in list(self.known.items()):
            if key in measured or not same_process(identity):continue
            value=sample_memory(identity['pid'],'hard_current_captured_family')['worker_private_memory']
            if not value['available'] or not same_process(identity):
                require(not same_process(identity),'Live captured family memory unavailable');continue
            measured[key]={'identity':identity,'private':value}
        native=(self.producer['pid'],self.producer['create_time'])
        shim=0
        for key,part in measured.items():
            require(part['private']['available'] and type(part['private']['private_bytes']) is int
                    and part['private']['private_bytes']>0,'Actual available exact captured family bytes')
            if key==native:
                require(part['private']['private_bytes']<=self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes'],
                        'Actual native current private exceeds its whole worker')
            else:shim+=part['private']['private_bytes']
        require(shim<=64_000_000,'Continuously summed ALL exact GPU shim private cap64MB')
        own_private=own['worker_private_memory']['private_bytes'];require(own_private<=self.parent_guard.current_cap,'Exact current parent metadata phase cap')
        result={'parent':own,'all_captured_GPU_shim_private_bytes':shim,
            'remaining_parent_and_allshim_allowance_bytes':max(self.parent_guard.current_cap-own_private,0)+max(64_000_000-shim,0),
            'native_identity':self.producer,'measured_processes':list(measured.values()),
            'source':record(__file__),'configuration':self.configuration_binding,
            'observed_monotonic':time.monotonic(),'native_excluded_from_shim_sum':True}
        write_json(self.folder/'exact_GPU_family_current.json',result)
        return result

    def _watch(self):
        try:
            while not self.done.is_set() and self.child.poll() is None:
                family_context=self._family_guard(); sample = sample_memory(self.pid, 'full_DATA_GPU_parent_fresh')
                if not sample['worker_private_memory'].get('available'):
                    from asr18_owned_family_observation_v4 import reaped_native_success_pending
                    self.exit_transition = reaped_native_success_pending(self.producer,
                        completion_path=self.completion_path,worker_source=self.config['worker_driver_source'],
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
                        family_context=self._family_guard()
                        require(time.monotonic()-family_context['observed_monotonic']<=1,'Fresh exact owned-family context')
                        result = fresh_admit(value['incremental_private_budget_bytes']+family_context['remaining_parent_and_allshim_allowance_bytes'], pid=self.pid)
                        observed_private = result['sample']['worker_private_memory']['private_bytes']
                        whole_projection = observed_private + value['incremental_private_budget_bytes']
                        passed = result['passed'] and whole_projection <= self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']
                        write_json(self.folder/'stage_admission.json', {'name': value['name'], 'passed': passed,
                            'fresh_sample': result['sample'], 'exact_family_context':family_context, 'incremental_budget_bytes': value['incremental_private_budget_bytes'],
                            'projected_worker_private_bytes': whole_projection,
                            'whole_worker_private_allowance_bytes': self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes']})
                        self.admitted.add(value['name']); lease = read_json(LEASE)
                        lease.update(actual_training_optimizer_running=passed and self.config['worker_mode'] == 'resident_training'
                                     and __import__('re').fullmatch(r'native_[1-9][0-9]*',value['name']) is not None,
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
        window=read_json(response['window_completion']['path'])
        path=Path(response['window_completion']['path']).parent/'resident_boundary.json'
        require(path.parent.is_relative_to(Path(self.config['checkpoint_storage_namespace']).resolve()/'transient'),
                'Per-window sealed boundary only in exact registered current AUX/native')
        write_json(path,value)
        self.idle_proof = record(path); self.command_state = 'coherent_idle_CPU_preparation'
        deadline = time.monotonic()+3
        while not (self.folder/'resident_live_witness.json').exists() or read_json(self.folder/'resident_live_witness.json')['idle_proof'] != self.idle_proof:
            self.check(); require(time.monotonic() < deadline, 'Actual fresh parent idle witness required'); time.sleep(.05)
        return self.idle_proof

    def finish(self):
        from fastmss_owned_process_closure_v2 import observe
        autonomous=self.config['worker_mode']=='serial350DEV_optimizer_free'
        if not autonomous:self.command('finish_coherent_native_process')
        # Standalone DEV has no worker_ready/command/finish protocol. Its real
        # process0 names the inner report; the release envelope is separately required.
        deadline=time.monotonic()+(self.config['maximum_GPU_child_wall_seconds'] if autonomous else 60)
        while self.child.poll() is None:
            self.parent_guard.check(); require(time.monotonic() < deadline, 'Owned native exit timeout'); time.sleep(.1)
        code = self.child.wait()
        if autonomous and code==0:
            envelope=read_json(self.folder/'worker_completion.json')
            require(envelope is not None and envelope['passed'] is True and envelope['producer']==self.producer
                and envelope['worker_source']==self.config['worker_driver_source'] and envelope['source']==self.config['worker_driver_source']
                and envelope['parent_source']==record(__file__) and envelope['worker_report']==record(self.completion_path)
                and envelope['request']==self.config['request'] and len(envelope['phase_memory_records'])==4,
                'Autonomous DEV actual release envelope plus SAME canonical inner report before process0')
        from asr18_owned_family_observation_v4 import wait_captured_dead
        retired = wait_captured_dead(self.known, timeout=15)
        self.done.set(); self.monitor.join(timeout=2);self.stdout_thread.join(timeout=2); self.log.close()
        require(not self.stdout_thread.is_alive() and self.log_error is None,'Actual stdout EOF and clean joined bounded sink')
        worker_control_guard(self.folder)
        require(code == 0 and not self.monitor.is_alive() and self.monitor_error is None and self.samples >= 3,
                'Actual native exit0 and clean joined parent guard required')
        footer = self.folder/'native_guard_completion.json'
        write_json(footer, {'pid': self.pid, 'actual_exit_code': code, 'monitor_clean': True, 'monitor_joined': True,
            'observed': {'samples': self.samples, 'peak_private_bytes': self.peak, 'monitor_error': None},
            'guard_source': record(__file__), 'sampling_interval_seconds': 1.,
            'all_captured_GPU_identities_observed_dead': retired,
            'native_reaped_launcher_pending_transition':self.exit_transition})
        proof = observe(self.child, kind='ASR18_owned_GPU_process_completion_v1', owner=OWNER,
            report=record(self.completion_path), worker_source=self.config['worker_driver_source'], parent_source=record(__file__),
            guard_source=record(__file__), guard_closure=record(footer), launcher_identity=self.launcher,
            parent_guard_observation=sample_memory(os.getpid(), 'native_actual_wait_observer'),
            output=self.folder/'actual_owned_native_process0.json', guard=self.parent_guard.check)
        self._release_lease(); return proof

    def _release_lease(self):
        lease = read_json(LEASE); require(lease['owner'] == OWNER and lease.get('actual_owned_worker_pid') in (None, self.pid), 'Exact owned lease release')
        lease.update(actual_GPU_running=False, actual_owned_worker_pid=None, actual_owned_launcher_pid=None,
            actual_training_optimizer_running=False, status='memory_diagnostic_idle', updated_at_utc=now()); write_json(LEASE, lease)

    def abort(self):
        write_json(self.folder/'stop.json',{'reason':'owned_parent_failure_preserve_partial'})
        from asr18_owned_family_observation_v4 import terminate_known_family
        self.done.set();self.monitor.join(timeout=2)
        retired=terminate_known_family(self.child,self.launcher,self.known)
        write_json(self.folder/'failed_all_captured_identity_retirement.json',retired)
        self.stdout_thread.join(timeout=2);self.log.close();self._release_lease()


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
    from fastmss_BPE9223_feature_protocol_v8 import cleanup_window, read_bound
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
    result = cleanup_window(request['owner'], closure_pin=request['closure'], manifest_pin=request['manifest'],
        boundary=boundary, verify_boundary=verify_boundary, evidence_path=request['archive_path'], guard=guard)
    require(result['passed'] is True and result['original_shared_oracle_untouched'] is True
        and result['sampler_journal_frontier_advanced'] is False,'Actual exact aggregate owned retirement only')
    # New aggregate cleanup verifies and preserves the complete manifest/closure
    # before exact owner.owned_names retirement. These legacy handoff fields
    # describe that actual source-bound operation, not a new authority.
    result.update(manifest=request['manifest'],feature_files_only_deleted=True,
        original_audio_source_checkpoint_journal_untouched=True,
        producer=identity(), worker_source=record(__file__), parent_source=request['parent_source'])
    write_json(output/'report.json', result)


_ACTIVE_OWNER=None

def install_owner(owner):
    global _ACTIVE_OWNER
    require(_ACTIVE_OWNER is None,'One actual owner per process');_ACTIVE_OWNER=owner
    frozen_parent.install_owner(owner)

def execute_owned_DEV(request_pin):
    require(_ACTIVE_OWNER is not None,'Actual owner installed before DEV callback')
    return _ACTIVE_OWNER.execute_DEV(request_pin)

def native_boundary_DEV(pointer):
    require(_ACTIVE_OWNER is not None,'Actual owner installed before native boundary callback')
    return _ACTIVE_OWNER.make_DEV_boundary(pointer)

def verify_worker_control_protection(request):
    require(_ACTIVE_OWNER is not None,'Actual owner installed before control protection callback')
    return _ACTIVE_OWNER.verify_worker_control_protection(request)

def verify_closed_worker_control(request):
    require(_ACTIVE_OWNER is not None,'Actual owner installed before closed control callback')
    return _ACTIVE_OWNER.verify_closed_worker_control(request)

def controller(config_path,review_path,folder,resume_request=None):
    from asr18_BPE9223_full_run_controller_v15 import run
    return run(config_path,review_path,folder,parent_source=record(__file__),ParentGuard=ParentGuard,
        GPUService=GPUService,run_CPU=frozen_CPU_executor.run_CPU,cleanup_script=Path(__file__).resolve(),storage=storage,fresh_admit=fresh_admit,
        execute_owned_DEV=execute_owned_DEV,native_boundary_DEV=native_boundary_DEV,install_owner=install_owner,
        verify_worker_control_protection=frozen_parent.verify_worker_control_protection,
        verify_closed_worker_control=frozen_parent.verify_closed_worker_control,resume_request=resume_request)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--review');p.add_argument('--folder',required=True)
    p.add_argument('--cleanup',action='store_true');p.add_argument('--request');p.add_argument('--resume-request')
    a=p.parse_args();folder=scoped(a.folder)
    if a.cleanup:cleanup_child(a.request,folder)
    else:controller(a.config,a.review,folder,None if a.resume_request is None else record(a.resume_request))
