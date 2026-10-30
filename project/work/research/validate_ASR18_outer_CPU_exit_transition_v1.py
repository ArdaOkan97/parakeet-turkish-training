"""Finite process-only outer-parent exit-race control; no audio/cache/GPU."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[2]
S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
import asr18_owned_family_observation_v4 as family
from asr18_outer_CPU_exit_transition_v1 import wait_actual_CPU_family_exit
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from fastmss_owned_process_closure_v1 import identity, verify
from fastmss_row_serial_cpu_v4 import environment, resource_guard

PYTHON = Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
GUARD = P/'work/research/ASR18_bounded_CPU_phase_v3.py'
PARENT = P/'work/research/run_ASR18_full_DATA_multiwindow_v6.py'


def leaf(request_path, folder):
    request = read_json(request_path)
    write_json(request['ready'], {'producer': identity()})
    deadline = time.monotonic()+10
    while not Path(request['release']).exists():
        resource_guard(); require(time.monotonic() < deadline, 'Finite controlled exit release')
        time.sleep(.02)
    write_json(folder/'report.json', {'passed': True, 'producer': identity(),
        'worker_source': record(__file__), 'parent_source': record(PARENT),
        'purpose': 'outer_CPU_exit_process_control_only'})


def main(output):
    environment(); require(not output.exists(), 'Fresh E-only process control')
    require(output.resolve().is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve()), 'E-only controls')
    output.mkdir(); checks = []
    admission = resource_guard(); system = admission['system_memory']; increment = 160_000_000
    require(system['commit_used_bytes']+increment < .90*system['commit_limit_bytes']
            and system['commit_available_bytes']-increment > 2_000_000_000,
            'Fresh 160MB process-only incremental forecast before any child')
    write_json(output/'prospective_admission.json', {'sample': admission, 'incremental_bytes': increment, 'passed': True})
    def check(name, value): require(value, name); checks.append(name)
    def reject(name, call):
        try: call()
        except (ValueError, ProcessLookupError): checks.append(name); return
        raise AssertionError('Expected rejection: '+name)
    spec = importlib.util.spec_from_file_location('actual_outer_parent6', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    ready, release = output/'ready.json', output/'release.json'
    request = output/'request.json'; write_json(request, {'ready': str(ready), 'release': str(release)})
    original = family.owned_family_sample; injected = {'done': False, 'actual_wait': None}
    def race(child, launcher, *, known, guard):
        if ready.exists() and not injected['done']:
            # Capture actual ancestry while live, then reproduce the normal
            # launcher disappearance with a REAL owned wait0, never a fake code.
            original(child, launcher, known=known, guard=guard)
            injected['done'] = True; write_json(release, {'release': True})
            injected['actual_wait'] = child.wait(timeout=10)
            raise ProcessLookupError('Deterministic launcher snapshot disappearance after real exit')
        return original(child, launcher, known=known, guard=guard)
    config = {'CPU_python': record(PYTHON), 'CPU_guard_source': record(GUARD),
        'CPU_owned_allowance_bytes': 160_000_000, 'CPU_inner_owned_allowance_bytes': 126_445_568,
        'CPU_outer_controller_launcher_reserve_bytes': 33_554_432, 'maximum_CPU_phase_seconds': 25}
    child_folder = output/'positive_guarded_leaf'
    family.owned_family_sample = race
    try:
        actual0 = parent.run_CPU(config, Path(__file__).resolve(),
            ['--leaf', '--request', str(request), '--output', str(child_folder)],
            child_folder, guard=resource_guard)
    finally: family.owned_family_sample = original
    actual = read_json(actual0['path']); proof = read_json(child_folder/'owned_CPU_family_completion.json')
    transition = proof['launcher_disappearance_or_normal_exit_drain']
    verify(actual, kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis',
           report=record(child_folder/'report.json'))
    check('Actual parent6 run_CPU source executed', actual['parent_source'] == record(PARENT))
    check('Real wait0 precedes deterministic disappeared-launcher signal', injected['done'] and injected['actual_wait'] == 0)
    check('Normal process0 independently verifies exact report/source/guard', actual['actual_native_exit_code'] == 0 and actual['actual_Popen_wait_code'] == 0)
    check('All captured identities actually dead', transition['all_captured_owned_identities_observed_dead'] is True)
    check('Parent and CPU family retain separate unchanged resource domains',
          transition['separate_parent_private_limit_bytes'] == 2_000_000_000
          and transition['CPU_family_private_allowance_bytes'] == config['CPU_owned_allowance_bytes'])
    check('Transition itself claims no native or tool proof', transition['normal_native_exit0_claimed'] is False
          and transition['clean_guard_or_report_validated_here'] is False and transition['tool_CLOSED0_claimed'] is False)
    check('Actual outer launcher accounted in ordinary family samples', proof['outer_launcher_included'] is True and proof['family_samples'] >= 1)
    closed = subprocess.Popen([str(PYTHON), '-B', '-c', 'pass'], creationflags=subprocess.CREATE_NO_WINDOW)
    closed_id = identity(closed.pid); require(closed.wait(timeout=10) == 0, 'Actual control handle0')
    known = {(closed_id['pid'], closed_id['create_time']): closed_id}
    closed_result = wait_actual_CPU_family_exit(closed, closed_id, known, guard=resource_guard,
        CPU_family_private_allowance_bytes=160_000_000)
    check('Normal handle0 drain also validates independent parent guard', closed_result['guard_samples'] >= 1)
    reject('Boolean allowance rejected', lambda: wait_actual_CPU_family_exit(closed,closed_id,known,
        guard=resource_guard,CPU_family_private_allowance_bytes=True))
    reject('Unknown original launcher rejected', lambda: wait_actual_CPU_family_exit(closed,closed_id,{},
        guard=resource_guard,CPU_family_private_allowance_bytes=160_000_000))
    reject('Non-Popen handle rejected', lambda: wait_actual_CPU_family_exit(object(),closed_id,known,
        guard=resource_guard,CPU_family_private_allowance_bytes=160_000_000))
    reject('Timeout expansion rejected', lambda: wait_actual_CPU_family_exit(closed,closed_id,known,
        guard=resource_guard,CPU_family_private_allowance_bytes=160_000_000,timeout=3))
    bad = copy.deepcopy(resource_guard()); bad['worker_private_memory']['pid'] = -1
    reject('Wrong observer PID rejected', lambda: wait_actual_CPU_family_exit(closed,closed_id,known,
        guard=lambda:bad,CPU_family_private_allowance_bytes=160_000_000))
    bad = copy.deepcopy(resource_guard()); bad['worker_private_memory']['private_bytes'] = 2_000_000_000
    reject('Independent parent2GB ceiling enforced', lambda: wait_actual_CPU_family_exit(closed,closed_id,known,
        guard=lambda:bad,CPU_family_private_allowance_bytes=160_000_000))
    failed = subprocess.Popen([str(PYTHON), '-B', '-c', 'raise SystemExit(7)'], creationflags=subprocess.CREATE_NO_WINDOW)
    failed_id = identity(failed.pid); require(failed.wait(timeout=10) == 7, 'Actual failure7')
    reject('Actual failure cannot become normal completion', lambda: wait_actual_CPU_family_exit(failed,failed_id,
        {(failed_id['pid'],failed_id['create_time']):failed_id},guard=resource_guard,CPU_family_private_allowance_bytes=160_000_000))
    live = subprocess.Popen([str(PYTHON), '-B', '-c', 'import time;time.sleep(10)'], creationflags=subprocess.CREATE_NO_WINDOW)
    live_id = identity(live.pid); live_known = {(live_id['pid'],live_id['create_time']):live_id}
    try:
        reject('Live waiting launcher cannot enter normal drain', lambda: wait_actual_CPU_family_exit(live,live_id,live_known,
            guard=resource_guard,CPU_family_private_allowance_bytes=160_000_000))
        orphans = {**known, **live_known}
        reject('Captured live orphan cannot be treated dead', lambda: wait_actual_CPU_family_exit(closed,closed_id,orphans,
            guard=resource_guard,CPU_family_private_allowance_bytes=160_000_000,timeout=.08))
        reject('CPU-family ceiling applies while draining', lambda: wait_actual_CPU_family_exit(closed,closed_id,orphans,
            guard=resource_guard,CPU_family_private_allowance_bytes=1))
    finally: family.terminate_known_family(live, live_id, live_known)
    check('Negative captured child actually retired', not family.same_process(live_id))
    check('No numerical/audio/frontend/model imports', not any(module in sys.modules for module in
        ('numpy','scipy','torch','transformers','av','soundfile','pyarrow')))
    result = {'kind': 'ASR18_outer_CPU_exit_process_controls_v1', 'passed': True,
        'check_count': len(checks), 'checks': checks, 'source': record(__file__),
        'parent': record(PARENT), 'outer_transition': record(S/'asr18_outer_CPU_exit_transition_v1.py'),
        'family_helper': record(S/'asr18_owned_family_observation_v4.py'),
        'actual_positive_process0': actual0, 'actual_positive_family_completion': record(child_folder/'owned_CPU_family_completion.json'),
        'actual_original_audio_frontend_jobs': 0, 'GPU_launched': False,
        'shared_cache_changed': False, 'Adam_sampler_frontier_advanced': False}
    write_json(output/'report.json', result)
    print(json.dumps({'passed':True,'checks':len(checks),'actual_guarded_process0':True}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--leaf',action='store_true'); parser.add_argument('--request')
    args=parser.parse_args()
    leaf(args.request,Path(args.output)) if args.leaf else main(Path(args.output))
