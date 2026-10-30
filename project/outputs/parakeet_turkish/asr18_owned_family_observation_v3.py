"""Exact owned-family private accounting and reaped-native exit transition.

No model/CUDA imports or unrelated-process enumeration. Known identities are
captured only through the actual Popen ancestry. A source-bound reaped native
report can explain a transient launcher wait; it NEVER supplies wait0 itself.
"""
import json, subprocess, time
from pathlib import Path
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from resource_watchdog_v2 import sample_memory


def same_process(identity):
    import psutil
    require(type(identity['pid']) is int and identity['pid'] > 0
            and type(identity['create_time']) in (int, float) and identity['create_time'] > 0, 'Exact captured process identity')
    try:
        process=psutil.Process(identity['pid'])
        return process.is_running() and abs(process.create_time()-identity['create_time']) <= .001
    except psutil.NoSuchProcess: return False


def owned_family_sample(child, launcher_identity, *, known, guard):
    import psutil
    guard(); require(isinstance(child, subprocess.Popen) and child.pid == launcher_identity['pid'], 'Exact owned Popen launcher')
    if not same_process(launcher_identity):
        # The Windows Popen handle can still be wait-pending. This exception is
        # a transition signal only: the caller must obtain real wait0 and check
        # actual clean child/report closure before any success claim.
        raise ProcessLookupError('Captured owned launcher absent during family sampling; actual handle wait required')
    try:
        launcher = psutil.Process(child.pid)
        require(launcher.create_time() == launcher_identity['create_time']
                and launcher.ppid() == launcher_identity['ppid'], 'Owned launcher ancestry/create-time changed')
        processes = [launcher, *launcher.children(recursive=True)]
    except psutil.NoSuchProcess as error:
        # Resolve/ctime/ppid/children are separate syscalls. A normal launcher
        # may exit between any pair; do not let a bare psutil exception bypass
        # the controlled handle-wait path. AccessDenied and identity mismatches
        # still fail closed and trigger captured-family retirement.
        raise ProcessLookupError('Captured owned launcher disappeared during ancestry snapshot') from error
    identities = []
    for process in processes:
        try:
            identity = {'pid': process.pid, 'ppid': process.ppid(), 'create_time': process.create_time()}
            identities.append(identity)
        except psutil.NoSuchProcess: continue
    current = {item['pid']: item for item in identities}
    for identity in identities:
        parent = identity
        for _ in range(len(identities)+1):
            if parent['pid'] == child.pid: break
            require(parent['ppid'] in current, 'Captured process is outside exact current owned family')
            parent = current[parent['ppid']]
        else: raise ValueError('Cyclic/unknown process ancestry')
        known[(identity['pid'], identity['create_time'])] = identity
    private = 0; measured = []; exited = []
    for identity in identities:
        observation = sample_memory(identity['pid'], 'owned_CPU_family_private')['worker_private_memory']
        if not observation['available'] or not same_process(identity):
            require(not same_process(identity), 'Live owned family private telemetry unavailable')
            exited.append(identity); continue
        require(type(observation['private_bytes']) is int and observation['private_bytes'] > 0, 'Exact actual private bytes')
        private += observation['private_bytes']; measured.append({'identity': identity, 'private': observation})
    return {'kind': 'ASR18_actual_owned_Popen_family_private_sample_v1', 'launcher': launcher_identity,
            'private_bytes_sum': private, 'measured_processes': measured, 'exited_during_sample': exited,
            'outer_launcher_included': any(item['identity']['pid'] == child.pid for item in measured),
            'only_actual_owned_ancestry_observed': True, 'observed_monotonic': time.monotonic()}


def wait_successful_launcher_after_disappearance(child, launcher_identity, known, *, guard, timeout=2):
    """Narrow bounded transition, never a replacement for native/report proof.

    Only an absent exact captured launcher may enter. Every actual handle poll
    is accompanied by the unchanged parent/global guard. Reused PIDs are not
    touched. A live captured orphan or a nonzero handle cannot become CLOSED0.
    """
    require(isinstance(child, subprocess.Popen) and child.pid == launcher_identity['pid'], 'Actual captured Popen handle required')
    require(type(timeout) in (int, float) and 0 < timeout <= 2, 'Bounded at-most2s launcher exit transition')
    require(not same_process(launcher_identity), 'A live captured launcher cannot use disappearance handling')
    began = time.monotonic(); samples = 0
    while True:
        guard(); samples += 1
        code = child.poll()
        if code is not None:
            require(type(code) is int and code == 0, 'Failed owned launcher cannot become a normal completion')
            actual = child.wait(timeout=0)
            require(type(actual) is int and actual == 0, 'Real owned Popen wait0 required')
            require(not any(same_process(identity) for identity in known.values()),
                    'Captured owned descendant remains live after launcher0; failure retirement required')
            return {'kind': 'ASR18_owned_launcher_disappearance_actual_wait0_transition_v1',
                'launcher': launcher_identity, 'actual_Popen_wait_code': actual,
                'all_captured_owned_identities_observed_dead': True, 'guard_samples': samples,
                'wall_seconds': time.monotonic()-began, 'normal_native_exit0_claimed': False,
                'clean_guard_or_report_validated_here': False, 'tool_CLOSED0_claimed': False}
        require(time.monotonic()-began < timeout, 'Absent launcher handle failed to close within bounded2s transition')
        time.sleep(.02)


def terminate_known_family(child, launcher_identity, known):
    """Failure only: verify captured creation times even after reparenting."""
    import psutil
    for identity in reversed(list(known.values())):
        if same_process(identity):
            try: psutil.Process(identity['pid']).terminate()
            except psutil.NoSuchProcess: pass
    if child.poll() is None: child.terminate()
    child.wait(timeout=15)
    wait_captured_dead(known,timeout=15)


def wait_captured_dead(known, *, timeout):
    deadline=time.monotonic()+timeout
    while any(same_process(identity) for identity in known.values()):
        require(time.monotonic()<deadline,'Captured owned process still live; no family retirement or lease release')
        time.sleep(.05)
    return {'all_captured_owned_identities_observed_dead':True,'captured_identities':list(known.values()),
            'observed_monotonic':time.monotonic(),'PID_reuse_skipped':True}


def reaped_native_success_pending(known_native, *, completion_path, worker_source, parent_source, launcher):
    """Read a completed report ONLY after the expected native identity is dead.

    The caller must still wait the actual launcher handle, demand wait0 and
    validate the final clean monitor/source/scientific proof before success.
    """
    require(isinstance(launcher, subprocess.Popen), 'Actual Popen launcher handle required')
    require(not same_process(known_native), 'A live native with unavailable telemetry must fail closed')
    path = Path(completion_path)
    require(path.exists() and 0 < path.stat().st_size <= 16*1024**2, 'Bounded completed native report required after native reap')
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    require(report['passed'] is True and digest(report['producer']) == digest(known_native)
            and report['worker_source'] == worker_source and report['parent_source'] == parent_source,
            'Expected reaped native must have completed its exact source-bound report')
    require(record(worker_source['path']) == worker_source and record(parent_source['path']) == parent_source,
            'Native/parent source changed during exit transition')
    code = launcher.poll()
    require(code is None or type(code) is int and code == 0, 'Failed launcher cannot become a successful pending exit')
    return {'kind': 'ASR18_source_bound_reaped_native_launcher_wait_transition_v1',
            'expected_native': known_native, 'completed_report': record(path), 'launcher_PID': launcher.pid,
            'launcher_wait_pending': code is None, 'actual_Popen_wait_observed_here': False,
            'GPU_or_tool_CLOSED0_claimed': False, 'observed_monotonic': time.monotonic()}
