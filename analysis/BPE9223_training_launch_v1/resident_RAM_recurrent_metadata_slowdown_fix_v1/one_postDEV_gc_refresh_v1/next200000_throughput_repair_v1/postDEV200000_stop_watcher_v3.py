"""Inactive, fail-closed observer for the original200k post-DEV startup pause.

Only main() may publish the original supported stop request. Scientific metadata
is observed, never generated here. Pure helpers also support an isolated CPU
fixture whose layout is a fresh E-only METHOD_STUB tree, never the live run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import psutil

TARGET = 200000
MAX_JSON = 32 * 1024**2
SELF_CAP = 512 * 1024**2
COMMIT_ADMISSION = 5055306368
PHYSICAL_ADMISSION = 2305306368
COMMIT_RUNTIME = 2268435456
PHYSICAL_RUNTIME = 1768435456
P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
B = Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
I = B / 'invocations/owner_000026'
SOURCE = P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'
CPU_SOURCE = P / 'outputs/parakeet_turkish/fastmss_resident_feature_stream_v14.py'
CONFIG = I / 'configuration.json'
NATIVE = {'pid': 19320, 'create_time': 1792803517.7236164, 'ppid': 39856}
NATIVE_LAUNCHER = {'pid': 39856, 'create_time': 1792803517.7165897, 'ppid': 21928}
WRAPPER = {'pid': 21928, 'create_time': 1792803516.8916032, 'ppid': 41376}
OLD_CPU = {'pid': 13472, 'create_time': 1792822495.866724, 'ppid': 39752}
OLD_CPU_LAUNCHER = {'pid': 39752, 'create_time': 1792822495.8054123, 'ppid': 19320}
READER = {'pid': 26892, 'create_time': 1792652904.3359847, 'ppid': 30128}
READER_LAUNCHER = {'pid': 30128, 'create_time': 1792652904.283631, 'ppid': 10644}
SOURCE_SHA = '4dc49abfb06fdad61bf04c75dccf261d71e84a260a829b54a112103a4673500b'
CONFIG_SHA = '39dc6a3aa4b7139d6209751de769d5aef63a507e7122f56a6720afd85bdb966d'
CPU_SHA = 'e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e'


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def pin(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_JSON, 'Bounded source/metadata pin')
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_JSON, 'Bounded JSON metadata only')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_once(path, value):
    raw = canonical(value) + b'\n'
    require(len(raw) <= MAX_JSON, 'Bounded receipt')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(path)


def identity(pid=None):
    proc = psutil.Process(os.getpid() if pid is None else pid)
    return {'pid': proc.pid, 'create_time': proc.create_time(), 'ppid': proc.ppid()}


def exact_alive(who):
    try:
        proc = psutil.Process(who['pid'])
        return (abs(proc.create_time() - who['create_time']) < .001
                and proc.ppid() == who['ppid'] and proc.is_running())
    except psutil.NoSuchProcess:
        return False


def layout(root, invocation):
    root, invocation = Path(root), Path(invocation)
    ck = root / 'checkpoints/update_000200000'
    startup = invocation / 'CPU_after_periodic_DEV_000200000'
    return {'root': root, 'invocation': invocation, 'checkpoint': ck,
            'progress': root / 'progress/latest.json',
            'latest': root / 'latest_checkpoint.json',
            'periodic': root / 'periodic_DEV.json',
            'quiescence': invocation / 'CPU_periodic_000200000_quiescence.json',
            'failure': invocation / 'actual_failure.json',
            'startup': startup, 'startup_log': startup / 'CPU_stderr.log',
            'ready': startup / 'actual_CPU_ready.json', 'stop': root / 'stop.json'}


def capture_native_descendants(native):
    """Read-only exact identity evidence; no termination or method-close inference."""
    require(exact_alive(native), 'Exact native PIDctime/PPID before descendant capture')
    root = psutil.Process(native['pid'])
    children = root.children(recursive=True)
    require(len(children) <= 8, 'Bounded current native descendants, never truncate')
    records = []
    for child in children:
        who = identity(child.pid)
        require(exact_alive(who), 'Same current descendant before ancestry read')
        chain = []
        for ancestor in psutil.Process(who['pid']).parents():
            parent = identity(ancestor.pid)
            chain.append(parent)
            require(len(chain) <= 16, 'Bounded original ancestry walk')
            if parent == native:
                break
        require(chain and chain[-1] == native and exact_alive(native) and exact_alive(who),
                'Exact live ancestry to unchanged native, no automatic rebind')
        records.append({'identity': who, 'ancestry_to_native': chain})
    require(exact_alive(native), 'Same native after bounded descendant capture')
    return {'native': native, 'descendants': records, 'count': len(records),
            'read_only_identity_capture_not_close_or_control': True,
            'captured_monotonic_ns': time.monotonic_ns()}


def expected_current():
    return {'source': {'path': str(SOURCE), 'bytes': 132567, 'sha256': SOURCE_SHA},
            'configuration': {'path': str(CONFIG), 'bytes': 226990, 'sha256': CONFIG_SHA},
            'CPU_source': {'path': str(CPU_SOURCE), 'bytes': 36718, 'sha256': CPU_SHA},
            'native': NATIVE, 'native_launcher': NATIVE_LAUNCHER, 'wrapper': WRAPPER,
            'old_CPU': OLD_CPU, 'old_CPU_launcher': OLD_CPU_LAUNCHER,
            'reader': READER, 'reader_launcher': READER_LAUNCHER,
            'cohort_start': 196000}


def collect_stage(paths, expected, alive=exact_alive):
    """No plan/journal/prediction/tensor/audio read; successful report is evidence.

    Progress emits BEFORE checkpoint and can retain last_saved199800 at200000.
    Authoritative latest+persistence, not that stale field, bind the saved state.
    Pending partial publication is WAIT. Invalid completed metadata is rejection.
    """
    require(not paths['stop'].exists(), 'Foreign/existing stop target: no overwrite')
    require(not paths['failure'].exists(), 'No current invocation failure')
    for key in ('native', 'native_launcher', 'wrapper', 'reader', 'reader_launcher'):
        require(alive(expected[key]), 'Exact live protected identity ' + key)
    progress = read(paths['progress'])
    require(progress['producer'] == expected['native']
            and progress['source'] == expected['source']
            and progress['planned_native_updates'] == 401072
            and progress['administrative_update_stop'] is None
            and progress['optimizer_poisoned'] is False
            and progress['schedule_poisoned'] is False, 'Unchanged healthy source37 progress')
    actual = progress['actual_optimizer_updates']
    require(type(actual) is int, 'Integer actual frontier')
    if actual < TARGET:
        return {'decision': 'WAIT', 'reason': 'before200000', 'actual': actual}
    if actual > TARGET:
        return {'decision': 'MISSED', 'reason': 'ordinary training already resumed', 'actual': actual}
    require(pin(expected['source']['path']) == expected['source']
            and pin(expected['configuration']['path']) == expected['configuration']
            and pin(expected['CPU_source']['path']) == expected['CPU_source'],
            'Exact current source/config/CPU source')
    if paths['ready'].exists():
        return {'decision': 'MISSED', 'reason': 'newCPU ready already published', 'actual': actual}
    latest = read(paths['latest'])
    if latest['committed_updates'] < TARGET:
        return {'decision': 'WAIT', 'reason': 'original checkpoint publication pending', 'actual': actual}
    require(latest['committed_updates'] == TARGET and latest['full_state_verified'] is True
            and latest['source'] == expected['source'], 'Exact authoritative200000 checkpoint')
    ck = paths['checkpoint']
    pointer = pin(ck / 'pointer.json')
    persistence_pin = pin(ck / 'persistence_verified.json')
    require(latest['pointer'] == pointer and latest['persistence'] == persistence_pin,
            'Latest points to original verified200000 metadata')
    persistence = read(persistence_pin['path'])
    require(persistence['passed'] is True and persistence['optimizer_updates'] == TARGET
            and persistence['named_tensors_verified'] == 854
            and persistence['weights_verified'] == 207 and persistence['Adam_states_verified'] == 215
            and persistence['independent_pre_save_state_sha256'] == persistence['saved_state_sha256']
            and persistence['saved_semantic_state_verified']['passed'] is True,
            'Original full854/207/215 semantic PRE-SAVED persistence, metadata only')
    entries = [entry for entry in read(paths['periodic'])['completed_evaluations']
               if entry['optimizer_updates'] == TARGET]
    if not entries:
        return {'decision': 'WAIT', 'reason': 'original DEV has not returned/recorded', 'actual': actual}
    require(len(entries) == 1, 'Exactly one original periodic record')
    entry = entries[0]
    require(entry['checkpoint_pointer'] == pointer
            and entry['shared_with_full_pass_DEV'] is False
            and entry['periodic_record_advances_full_pass_or_winner'] is False,
            'Genuine original periodic-only record after DEV returned')
    require(pin(entry['DEV']['path']) == entry['DEV']
            and pin(entry['frontier_preservation']['path']) == entry['frontier_preservation'],
            'Exact returned DEV and preservation metadata')
    report, preservation = read(entry['DEV']['path']), read(entry['frontier_preservation']['path'])
    langs = {'tr', 'en', 'es', 'it', 'fr', 'de', 'pt'}
    require(report['kind'] == 'ASR18_actual_resident_optimizer_idle_serial350_DEV_v1'
            and report['passed'] is True and report['optimizer_updates'] == TARGET
            and report['checkpoint_pointer'] == pointer
            and report['actual_native_updates_in_DEV'] == 0
            and report['all207_threeAdam_all4RNG_unchanged'] is True
            and report['TEST_used'] is False and report['retention_early_stop'] is False
            and set(report['predictions']) == langs and set(report['scores']) == {'validation_' + lang for lang in langs}
            and all(report['scores']['validation_' + lang]['asr']['utterances'] == 50 for lang in langs)
            and report['official_saved_text_score']['full350_DEV_complete'] is True
            and entry['score'] == report['official_saved_text_score'],
            'Original returned full7x50 DEV, no prediction read/new inference')
    require(preservation['optimizer_updates'] == TARGET
            and preservation['DEV'] == entry['DEV']
            and preservation['actual_native_updates_in_DEV'] == 0
            and preservation['actual_sampler_or_replay_advance'] is False
            and all(preservation[key] is True for key in
                    ('all207_threeAdam_all4RNG_unchanged', 'both_cursors_journal_and_nextdraw_unchanged',
                     'fullpass_and_winner_state_unchanged_here')),
            'Original DEV returned with all original preservation checks')
    q = read(paths['quiescence'])
    closed = q['CPU_actual0']
    drained = q['private_packets_discarded']
    require(type(drained) is int and 0 < drained <= 20, 'Original bounded actual drain')
    completed = TARGET - expected['cohort_start'] + drained
    require(q['kind'] == 'ASR18_actual_periodic_DEV_private_CPU_quiescence_v1'
            and q['optimizer_updates'] == TARGET and q['actual_native_updates'] == 0
            and q['authoritative_sampler_replay_or_journal_advance'] is False
            and q['private_plan_regenerated_or_reordered'] is False
            and q['same_exact_plan_must_be_resubmitted_after_DEV'] is True
            and q['full_prefetch_and_DEV_reservations_unchanged'] is True
            and q['only_original_two_carried_RNG_representation_paths_normalized'] is True
            and q['unconsumed_private_plan'] == preservation['pending_private_plan_retained']
            and q['unconsumed_private_plan'] is not None,
            'Original q and exact same published pending-plan binding, no planbody read')
    require(closed['producer_identity'] == expected['old_CPU']
            and not alive(expected['old_CPU']) and not alive(expected['old_CPU_launcher'])
            and closed['actual_Popen_wait_code'] == 0
            and closed['transport_threads_joined'] is True and closed['pipe_handles_closed'] is True
            and closed['completed_updates'] == completed
            and closed['source']['protocol'] == expected['CPU_source']
            and closed['queue_capacity_bytes'] == 268435456
            and closed['maximum_live_receiver_packets'] == 1
            and closed['peak_live_receiver_packets'] == 1, 'Exact old normal CPU close/cohort/queue')
    child, parent = closed['packet_credit'], closed['packet_credit_owner']
    require(child['all_owned_kernel_handles_closed'] is True
            and parent['all_owned_kernel_handles_closed'] is True
            and child['remaining_owned_handle_count'] == parent['remaining_owned_handle_count'] == 0
            and child['successful_preparation_acquires'] == parent['successful_release_posts'] == completed
            and child['descriptor'] == parent['descriptor']
            and child['descriptor']['logical_packet_maximum'] == 2
            and parent['last_release_previous_count'] == 1
            and sum(parent['release_previous_count_histogram'].values()) == completed
            and parent['cancelled'] is False, 'Both original owned credit sets closed, final1to2, no cancellation')
    if not (paths['startup'].is_dir() and paths['startup_log'].is_file()):
        return {'decision': 'WAIT', 'reason': 'newCPU constructor has not opened startup folder/log', 'actual': actual}
    if paths['ready'].exists():
        return {'decision': 'MISSED', 'reason': 'newCPU became ready during observation', 'actual': actual}
    return {'decision': 'READY', 'actual': actual, 'progress': progress, 'latest': latest,
            'periodic_entry': entry, 'DEV': entry['DEV'], 'preservation': entry['frontier_preservation'],
            'persistence': persistence_pin, 'quiescence': pin(paths['quiescence']),
            'actual_drain': drained, 'actual_expected_completed': completed,
            'same_pending_plan_binding': q['unconsumed_private_plan'],
            'new_startup_folder': str(paths['startup']), 'startup_log_present': True,
            'ready_absent': True, 'current_native_descendants': capture_native_descendants(expected['native']),
            'full_science_assertions_source_bound_not_independently_recomputed': True}


def decisive_recheck(paths, expected, alive=exact_alive):
    """Repeat ALL decisive metadata predicates immediately before publication."""
    result = collect_stage(paths, expected, alive)
    require(result['decision'] == 'READY', 'No stop when the postDEV startup stage was missed')
    require(read(paths['progress'])['actual_optimizer_updates'] == TARGET
            and not paths['ready'].exists() and not paths['failure'].exists()
            and not paths['stop'].exists() and alive(expected['native']),
            'Immediate final original frontier/ready/failure/foreignstop/native recheck')
    return result


def publish_marker(target, temporary, payload, *, before_publish,
                   fsync=os.fsync, rename=os.rename):
    """Windows SAME-volume exclusive rename; NEVER replace a foreign stop.

    Serialize before opening. Preserve any failed owned temp. On Windows rename
    refuses an existing target; an exclusive precheck alone is not the guarantee.
    """
    target, temporary = Path(target), Path(temporary)
    require(os.name == 'nt', 'Windows-exclusive rename semantics required')
    require(target.drive.lower() == temporary.drive.lower() == 'e:', 'Same E volume only')
    require(not target.exists() and not temporary.exists(), 'Fresh target and owned temporary')
    raw = canonical(payload) + b'\n'
    require(len(raw) <= 65536, 'Small supported stop request')
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        fsync(stream.fileno())
    evidence = before_publish()
    rename(str(temporary), str(target))
    return {'target': pin(target), 'prepublish': evidence,
            'atomic_same_E_volume_exclusive_rename': True,
            'directory_flush_or_race_free_pause_not_claimed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out = Path(args.output)
    require(out.drive.lower() == 'e:' and not out.exists(), 'One fresh E-only observer namespace')
    out.mkdir()
    began = time.monotonic()
    report = {'kind': 'ASR18_postDEV200000_supported_stop_observation_v1',
              'observer': identity(), 'launcher': identity(os.getppid()),
              'source': pin(__file__), 'stop_published': False,
              'production_session_consumed': False, 'inner_OS_exits': 'UNKNOWN',
              'passed': False, 'last_stage': None, 'polls': 0}
    sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
    from resource_watchdog_v2 import sample_memory

    def sample(label, prospective=False):
        value = sample_memory(os.getpid(), label)
        # Persist the ACTUAL numeric value before either prospective predicate.
        if prospective:
            write_once(out / (label + '.json'), value)
        mem = value['system_memory']
        require(value['paired_sampling_within_one_second'] is True
                and value['worker_private_memory']['private_bytes'] < SELF_CAP
                and mem['commit_available_bytes'] >= (COMMIT_ADMISSION if prospective else COMMIT_RUNTIME)
                and mem['physical_available_bytes'] >= (PHYSICAL_ADMISSION if prospective else PHYSICAL_RUNTIME)
                and shutil.disk_usage('E:/').free >= 100000000000
                and time.monotonic() - began < 3600, 'Original full metadata prospective/runtime/E/wall guards')
        return value

    try:
        report['initial_sample'] = sample('initial_full_prospective', True)
        scope = read(args.ROOT_review)
        require(scope['kind'] == 'ROOT_postDEV200000_stop_watcher_exact_ONCE_execution_scope_v1'
                and scope['accepted'] is True and scope['reviewer'] == '/root'
                and scope['watcher_source'] == pin(__file__)
                and scope['expected_current'] == expected_current()
                and Path(scope['output']) == out, 'Separate exact ROOT production watcher admission')
        for item in scope['input_pins']:
            require(pin(item['path']) == item, 'Exact admitted watcher input')
        expected = expected_current()
        paths = layout(B, I)
        while True:
            report['last_sample'] = sample('runtime')
            stage = collect_stage(paths, expected)
            report['last_stage'] = stage
            report['polls'] += 1
            if stage['decision'] == 'MISSED':
                report['passed'] = True
                report['no_write_reason'] = stage['reason']
                break
            if stage['decision'] == 'READY':
                report['prestop_evidence'] = write_once(out / 'prestop_complete_stage_evidence.json', stage)
                report['final_sample'] = sample('final_full_prospective', True)
                # Deliberate fsync-then-exclusive-rename refinement; no direct partial stop file.
                payload = {'kind': 'ASR18_human_authorized_postDEV200000_cooperative_checkpoint_stop_v1',
                           'reason': 'Reviewed throughput repair after original200k DEV/persistence return',
                           'ROOT_review': pin(args.ROOT_review), 'observer': report['observer'],
                           'evidence': report['prestop_evidence'], 'optimizer_updates_observed': TARGET,
                           'normal_exit_or_no_tail_race_claimed': False}
                report['publication'] = publish_marker(paths['stop'], out / 'owned_stop_request.tmp', payload,
                    before_publish=lambda: decisive_recheck(paths, expected))
                report['stop_published'] = True
                report['passed'] = True
                break
            time.sleep(.25)
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - began
        report['final_identity'] = identity()
        result = write_once(out / 'result.json', report)
        print(json.dumps({'result': result, 'stop_published': report['stop_published'],
                          'passed': report['passed']}), flush=True)


if __name__ == '__main__':
    main()
