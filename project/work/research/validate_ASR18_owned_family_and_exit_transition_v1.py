"""Real CPU Popen family/race controls, never a GPU or DATA proof."""
import argparse, copy, json, os, subprocess, sys, time
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from fastmss_owned_process_closure_v1 import identity
from asr18_owned_family_observation_v1 import owned_family_sample, reaped_native_success_pending, terminate_known_family
from resource_watchdog_v2 import sample_memory, ResourceGuard
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
PYTHON = Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
GUARD = P/'work/research/ASR18_bounded_CPU_phase_v3.py'


def leaf(folder):
    folder.mkdir(exist_ok=False)
    write_json(folder/'report.json', {'passed': True, 'producer': identity(), 'worker_source': record(__file__),
        'parent_source': record(__file__), 'kind': 'ASR18_CPU_only_fixture_marker_report_v1', 'GPU_used': False})


def outer(folder):
    folder.mkdir(exist_ok=False)
    command = [str(PYTHON), '-B', str(GUARD), '--folder', str(folder/'leaf_guard'), '--seconds', '30',
        '--script', str(Path(__file__).resolve()), '--', '--mode', 'leaf', '--folder', str(folder/'leaf')]
    with (folder/'leaf.log').open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = identity(child.pid)
        require(child.wait(timeout=20) == 0, 'Actual CPU marker child must close0')
    report = read_json(folder/'leaf/report.json'); closed = read_json(folder/'leaf_guard/closure.json')
    require(report['producer']['pid'] == closed['pid'] and closed['actual_exit_code'] == 0 and closed['monitor_clean']
        and closed['monitor_joined'] and closed['observed']['samples'] >= 3, 'Actual guarded CPU leaf closure')
    write_json(folder/'ready.json', {'producer': identity(), 'marker_report': record(folder/'leaf/report.json'),
        'marker_native': report['producer'], 'actual_marker_Popen_wait_code': 0, 'marker_launcher': launcher,
        'outer_handle_intentionally_still_alive': True, 'GPU_used': False})
    deadline = time.monotonic()+20
    while not (folder/'release.json').exists():
        require(time.monotonic() < deadline, 'Bounded CPU outer wait'); time.sleep(.05)


def main(folder):
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'Fresh E fixture')
    folder.mkdir(); checks = []; known = {}; peak = 0; samples = []; watch = ResourceGuard()
    def guard():
        sample = sample_memory(os.getpid(), 'owned_family_CPU_fixture')
        require(sample['paired_sampling_within_one_second'] and sample['worker_private_memory']['available'], 'Fresh fixture sample')
        reason, _ = watch.observe(sample['system_memory'], sample['worker_private_memory'])
        require(reason is None, 'Unchanged system guard'); return sample
    def check(name, value): require(value, name); checks.append(name)
    def reject(name, action):
        try: action()
        except (ValueError, KeyError): checks.append(name); return
        raise ValueError('Expected rejection: '+name)
    command = [str(PYTHON), '-B', str(GUARD), '--folder', str(folder/'outer_guard'), '--seconds', '40',
        '--script', str(Path(__file__).resolve()), '--', '--mode', 'outer', '--folder', str(folder/'outer')]
    with (folder/'outer.log').open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = identity(child.pid); deadline = time.monotonic()+20
        try:
            while not (folder/'outer/ready.json').exists():
                guard(); require(child.poll() is None and time.monotonic() < deadline, 'Bounded healthy actual CPU outer')
                sample = owned_family_sample(child, launcher, known=known, guard=guard)
                peak = max(peak, sample['private_bytes_sum']); samples.append(sample)
                require(peak < 400_000_000, 'Finite400MB fixture family bound'); time.sleep(.05)
            ready = read_json(folder/'outer/ready.json')
            check('Actual guarded CPU leaf0 while real outer Popen remains alive', ready['actual_marker_Popen_wait_code'] == 0 and child.poll() is None)
            check('Outer Windows launcher private bytes are explicitly included', any(item['outer_launcher_included'] for item in samples))
            check('Actual nested controller/leaf family was observed', max(len(item['measured_processes']) for item in samples) >= 2)
            check('All known identities derive only from actual Popen ancestry', all(item['only_actual_owned_ancestry_observed'] for item in samples))
            transition = reaped_native_success_pending(ready['marker_native'], completion_path=ready['marker_report']['path'],
                worker_source=record(__file__), parent_source=record(__file__), launcher=child)
            check('Reaped source-bound CPU marker/outer-pending transition recognized', transition['launcher_wait_pending'] is True)
            check('Pending transition never claims actual Popen wait or GPU/tool closure', transition['actual_Popen_wait_observed_here'] is False and transition['GPU_or_tool_CLOSED0_claimed'] is False)
            reject('Live native with missing telemetry stays fail-closed', lambda: reaped_native_success_pending(ready['producer'],
                completion_path=ready['marker_report']['path'], worker_source=record(__file__), parent_source=record(__file__), launcher=child))
            reject('Changed source report cannot explain an exit transition', lambda: reaped_native_success_pending(ready['marker_native'],
                completion_path=ready['marker_report']['path'], worker_source=record(GUARD), parent_source=record(__file__), launcher=child))
            bad = copy.deepcopy(ready['marker_native']); bad['pid'] = True
            reject('Boolean native PID rejected', lambda: reaped_native_success_pending(bad,
                completion_path=ready['marker_report']['path'], worker_source=record(__file__), parent_source=record(__file__), launcher=child))
            reject('Fake launcher handle cannot supply a transition', lambda: reaped_native_success_pending(ready['marker_native'],
                completion_path=ready['marker_report']['path'], worker_source=record(__file__), parent_source=record(__file__), launcher=object()))
            write_json(folder/'outer/release.json', {'fixture_release': True})
            check('Actual outer handle0 observed only after separate wait', child.wait(timeout=10) == 0)
        except BaseException:
            if child.poll() is None:
                (folder/'outer').mkdir(exist_ok=True); write_json(folder/'outer/release.json', {'fixture_failure_release': True})
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: terminate_known_family(child, launcher, known)
            raise
    closed = read_json(folder/'outer_guard/closure.json')
    check('Actual outer clean joined guard0 and>=3 samples', closed['actual_exit_code'] == 0 and closed['monitor_clean'] and closed['monitor_joined'] and closed['observed']['samples'] >= 3)
    require(not any(name in sys.modules for name in ('torch','numpy','scipy','transformers','pyarrow')), 'No numeric/model imports')
    write_json(folder/'report.json', {'passed': True, 'checks': checks, 'check_count': len(checks), 'producer': identity(),
        'source': record(__file__), 'helper': record(S/'asr18_owned_family_observation_v1.py'),
        'actual_family_private_peak_bytes': peak, 'family_samples': samples, 'transition': transition,
        'outer_guard': record(folder/'outer_guard/closure.json'), 'leaf_guard': record(folder/'outer/leaf_guard/closure.json'),
        'actual_CPU_outer_Popen_exit_code': 0, 'actual_CPU_leaf_Popen_exit_code': 0,
        'actual_GPU_or_ASR_updates': 0, 'GPU_lease_changed': False, 'real_DATA_or_CACHE_prepared': False})
    print(json.dumps({'passed': True, 'checks': len(checks), 'actual_family_private_peak_bytes': peak, 'actual_GPU_updates': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('main','outer','leaf'), default='main'); parser.add_argument('--folder', required=True)
    args = parser.parse_args(); globals()[args.mode](Path(args.folder).resolve())
