"""Tiny process-only controller race regression; no DATA/audio/cache work."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'outputs/parakeet_turkish'
sys.path.insert(0, str(SOURCE))
import psutil
import asr18_owned_family_observation_v3 as family
import fastmss_row_serial_cpu_v3 as core
import fastmss_owned_process_closure_v1 as process

PYTHON = Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
GUARD = PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py'
CONTROLLER = PROJECT/'work/research/prepare_FastMSS_phase_window_CPU_v3.py'


def child(request_path, output):
    request = json.loads(Path(request_path).read_text())
    Path(request['ready']).write_text('ready', encoding='utf-8')
    deadline = time.monotonic()+15
    while not Path(request['release']).exists():
        core.resource_guard()
        core.require(time.monotonic() < deadline, 'Finite controlled leaf release')
        time.sleep(.02)
    core.write_new(output, {'passed': True, 'producer': process.identity(),
        'worker_source': core.record(__file__), 'parent_source': core.record(CONTROLLER),
        'purpose': 'synthetic_process_supervision_only', 'actual_native_updates': 0})


def main(output):
    core.environment(); output = core.scoped(output)
    core.require(not output.exists(), 'Fresh E-only process fixture'); output.mkdir()
    spec = importlib.util.spec_from_file_location('fixture_controller3', CONTROLLER)
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    checks = []
    def check(name, value): core.require(value, name); checks.append(name)
    def reject(name, call):
        try: call()
        except (ValueError, ProcessLookupError, subprocess.TimeoutExpired): checks.append(name); return
        raise AssertionError('Expected rejection: '+name)
    original_sample = family.owned_family_sample; original_Process = psutil.Process
    injected = {'done': False, 'phase': None, 'actual_wait': None, 'native_before_exit': None}
    ready, release = output/'ready.txt', output/'release.txt'
    request = {'ready': str(ready), 'release': str(release)}
    core.write_new(output/'request.json', request)

    def race_sample(owned, launcher, *, known, guard):
        if not injected['done'] and ready.exists():
            # The actual guarded leaf is live at sampling entry. Emulate exactly
            # the failing inter-syscall timing, not a fake successful child:
            # same_process resolves live, then ppid waits the REAL handle0 and
            # raises the same psutil.NoSuchProcess seen in the failed run.
            injected['done'] = True; calls = {'n': 0}
            class VanishingLauncher:
                def create_time(self): return launcher['create_time']
                def ppid(self):
                    core.write_new(release, {'release': True})
                    injected['actual_wait'] = owned.wait(timeout=10)
                    injected['phase'] = 'ppid_after_live_identity_before_ancestry_snapshot'
                    raise psutil.NoSuchProcess(launcher['pid'])
            def resolve(pid=None):
                if pid == launcher['pid']:
                    calls['n'] += 1
                    if calls['n'] == 2: return VanishingLauncher()
                return original_Process(pid)
            psutil.Process = resolve
            try: return original_sample(owned, launcher, known=known, guard=guard)
            finally: psutil.Process = original_Process
        return original_sample(owned, launcher, known=known, guard=guard)

    contract = {'python': core.record(PYTHON), 'maximum_child_seconds': 25,
        'sources': {'controller': core.record(CONTROLLER), 'guard': core.record(GUARD)},
        'resource_review': {'reference_incremental_bytes': 120_000_000,
            'controller_private_reserve_bytes': 64*1024**2,
            'whole_owned_private_allowance_bytes': 220_000_000}}
    family.owned_family_sample = race_sample
    try:
        handle, report, closure, actual0, admission = controller.run_child(str(Path(__file__).resolve()),
            core.record(output/'request.json'), output/'leaf_report.json', output/'positive_child',
            contract, 'reference', core.resource_guard)
    finally:
        family.owned_family_sample = original_sample; psutil.Process = original_Process
    actual = core.read_bound(actual0); footer = core.read_bound(closure)
    check('Actual same inter-syscall ppid disappearance injected', injected['done'] and injected['phase'] is not None)
    check('Injected race waited a REAL owned handle0', type(injected['actual_wait']) is int and injected['actual_wait'] == 0)
    check('Controller3 tolerated disappeared launcher and still required actual process0',
          actual['actual_Popen_wait_code'] == 0 and actual['actual_native_exit_code'] == 0)
    check('Normal child clean joined >=3 samples independently checked',
          footer['actual_exit_code'] == 0 and footer['monitor_clean'] and footer['monitor_joined'] and footer['observed']['samples'] >= 3)
    transition = admission['launcher_disappearance_transition']
    check('Transition remains distinct from normal native/report/tool proof',
          transition['actual_Popen_wait_code'] == 0 and transition['normal_native_exit0_claimed'] is False
          and transition['clean_guard_or_report_validated_here'] is False and transition['tool_CLOSED0_claimed'] is False)
    check('All captured child identities are actually dead', transition['all_captured_owned_identities_observed_dead'])
    process.verify(actual, kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis', report=report)
    check('Exact source/report/guard/ancestry process0 passes independent verifier', True)
    check('Whole family/controller reserve remained bounded', admission['actual_controller_leaf_shim_private_peak_bytes'] < 220_000_000)

    # These are fresh tiny CPU-only processes, never audio/model/worker jobs.
    live = subprocess.Popen([str(PYTHON), '-B', '-c', 'import time;time.sleep(10)'], creationflags=subprocess.CREATE_NO_WINDOW)
    live_id = process.identity(live.pid); live_known = {(live_id['pid'],live_id['create_time']):live_id}
    try:
        reject('Live captured launcher cannot use disappearance transition', lambda:
            family.wait_successful_launcher_after_disappearance(live,live_id,live_known,guard=core.resource_guard))
        failed = subprocess.Popen([str(PYTHON), '-B', '-c', 'raise SystemExit(7)'], creationflags=subprocess.CREATE_NO_WINDOW)
        failed_id = process.identity(failed.pid); code = failed.wait(timeout=10)
        check('Actual negative owned handle failed with7', type(code) is int and code == 7)
        reject('Nonzero handle cannot be converted to successful transition', lambda:
            family.wait_successful_launcher_after_disappearance(failed,failed_id,
                {(failed_id['pid'],failed_id['create_time']):failed_id},guard=core.resource_guard))
        orphan_known = {(actual['launcher']['pid'],actual['launcher']['create_time']):actual['launcher'],
                        (live_id['pid'],live_id['create_time']):live_id}
        reject('Captured live orphan after real launcher0 is rejected', lambda:
            family.wait_successful_launcher_after_disappearance(handle,actual['launcher'],orphan_known,guard=core.resource_guard))
        reject('Arbitrary non-Popen handle rejected', lambda:
            family.wait_successful_launcher_after_disappearance(object(),actual['launcher'],{},guard=core.resource_guard))
    finally:
        family.terminate_known_family(live,live_id,live_known)
    check('Captured failure identity actually retired', not family.same_process(live_id))
    check('No numeric/frontend/model/audio runtime imports', not any(name in sys.modules for name in
        ('numpy','torch','scipy','transformers','pyarrow','soundfile','av')))
    result = {'kind':'FastMSS_actual_launcher_disappearance_CPU_controls_v1','passed':True,
        'check_count':len(checks),'checks':checks,'source':core.record(__file__),
        'helper':core.record(SOURCE/'asr18_owned_family_observation_v3.py'),'controller':core.record(CONTROLLER),
        'common':core.record(SOURCE/'fastmss_row_serial_cpu_v3.py'),
        'positive_process0':actual0,'positive_guard':closure,'positive_report':report,'positive_admission':admission,
        'race_timing_deterministically_injected':True,'actual_guarded_child_handle0_observed':True,
        'original_failed_audio_trial_rerun':False,'GPU_audio_frontend_or_DATA_job_started':False,
        'real_shared_CACHE_changed':False,'sampler_Adam_draw_frontier_advanced':False}
    core.write_new(output/'report.json', result)
    print(json.dumps({'passed':True,'checks':len(checks),'real_guarded_child0':True}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--request'); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    child(args.request,core.scoped(args.output)) if args.request else main(core.scoped(args.output))
