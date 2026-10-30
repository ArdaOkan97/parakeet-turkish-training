"""Finite actual predicate/atomic-file proof; all scientific metadata METHOD_STUB.
No live run path is used for a stop marker, no training control or data read.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil
import postDEV204000_stop_watcher_v2 as w


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(w.canonical(value) + b'\n')


def build_case(folder, expected, changes=None):
    paths = w.layout(folder / 'run', folder / 'run/invocations/owned_stub')
    ck = paths['checkpoint']
    ck.mkdir(parents=True)
    paths['startup'].mkdir(parents=True)
    paths['startup_log'].write_bytes(b'METHOD_STUB startup log, no real producer\n')
    put(ck / 'pointer.json', {'METHOD_STUB': True, 'updates': 204000})
    pointer = w.pin(ck / 'pointer.json')
    persistence = {'passed': True, 'optimizer_updates': 204000, 'named_tensors_verified': 854,
                   'weights_verified': 207, 'Adam_states_verified': 215,
                   'independent_pre_save_state_sha256': 'METHOD_STUB_same',
                   'saved_state_sha256': 'METHOD_STUB_same',
                   'saved_semantic_state_verified': {'passed': True}}
    report = {'kind': 'ASR18_actual_resident_optimizer_idle_serial350_DEV_v1', 'passed': True,
              'optimizer_updates': 204000, 'checkpoint_pointer': pointer,
              'actual_native_updates_in_DEV': 0, 'all207_threeAdam_all4RNG_unchanged': True,
              'TEST_used': False, 'retention_early_stop': False,
              'predictions': {lang: {'METHOD_STUB_unread_prediction': True}
                              for lang in ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt')},
              'scores': {'validation_' + lang: {'asr': {'utterances': 50}} for lang in ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt')},
              'official_saved_text_score': {'full350_DEV_complete': True}}
    pending = {'path': str(folder / 'NO_PLAN_BODY_EXISTS.json'), 'bytes': 19,
               'sha256': '0' * 64}
    preservation = {'optimizer_updates': 204000, 'actual_native_updates_in_DEV': 0,
                    'actual_sampler_or_replay_advance': False,
                    'all207_threeAdam_all4RNG_unchanged': True,
                    'both_cursors_journal_and_nextdraw_unchanged': True,
                    'fullpass_and_winner_state_unchanged_here': True,
                    'pending_private_plan_retained': pending}
    descriptor = {'logical_packet_maximum': 2, 'METHOD_STUB_credit': True}
    closed = {'producer_identity': expected['old_CPU'], 'actual_Popen_wait_code': 0,
              'transport_threads_joined': True, 'pipe_handles_closed': True,
              'completed_updates': 4020, 'source': {'protocol': expected['CPU_source']},
              'queue_capacity_bytes': 268435456, 'maximum_live_receiver_packets': 1,
              'peak_live_receiver_packets': 1,
              'packet_credit': {'all_owned_kernel_handles_closed': True,
                                'remaining_owned_handle_count': 0,
                                'successful_preparation_acquires': 4020, 'descriptor': descriptor},
              'packet_credit_owner': {'all_owned_kernel_handles_closed': True,
                                      'remaining_owned_handle_count': 0,
                                      'successful_release_posts': 4020, 'descriptor': descriptor,
                                      'last_release_previous_count': 1,
                                      'release_previous_count_histogram': {'0': 4019, '1': 1},
                                      'cancelled': False}}
    q = {'kind': 'ASR18_actual_periodic_DEV_private_CPU_quiescence_v1',
         'optimizer_updates': 204000, 'private_packets_discarded': 20, 'actual_native_updates': 0,
         'authoritative_sampler_replay_or_journal_advance': False,
         'private_plan_regenerated_or_reordered': False,
         'same_exact_plan_must_be_resubmitted_after_DEV': True,
         'full_prefetch_and_DEV_reservations_unchanged': True,
         'only_original_two_carried_RNG_representation_paths_normalized': True,
         'unconsumed_private_plan': pending, 'CPU_actual0': closed}
    progress = {'producer': expected['native'], 'source': expected['source'],
                'planned_native_updates': 401072, 'administrative_update_stop': None,
                'optimizer_poisoned': False, 'schedule_poisoned': False,
                'actual_optimizer_updates': 204000, 'last_saved_optimizer_updates': 203800}
    data = {'persistence': persistence, 'report': report, 'preservation': preservation,
            'q': q, 'progress': progress, 'entries_present': True, 'latest_update': 204000}
    if changes:
        changes(data)
    put(ck / 'persistence_verified.json', data['persistence'])
    put(folder / 'DEV.json', data['report'])
    dev_pin = w.pin(folder / 'DEV.json')
    data['preservation']['DEV'] = dev_pin
    put(folder / 'preservation.json', data['preservation'])
    entry = {'optimizer_updates': 204000, 'DEV': dev_pin,
             'frontier_preservation': w.pin(folder / 'preservation.json'),
             'checkpoint_pointer': pointer, 'shared_with_full_pass_DEV': False,
             'periodic_record_advances_full_pass_or_winner': False,
             'score': data['report']['official_saved_text_score']}
    put(paths['periodic'], {'completed_evaluations': [entry] if data['entries_present'] else []})
    put(paths['latest'], {'committed_updates': data['latest_update'], 'full_state_verified': True,
                         'source': expected['source'], 'pointer': pointer,
                         'persistence': w.pin(ck / 'persistence_verified.json')})
    put(paths['progress'], data['progress'])
    put(paths['quiescence'], data['q'])
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    w.require(out.drive.lower() == 'e:' and not out.exists(), 'Fresh isolated E fixture only')
    out.mkdir()
    began = time.monotonic()
    report = {'kind': 'ASR18_postDEV204000_stop_watcher_finite_CPU_v1', 'passed': False,
              'cases': [], 'METHOD_STUB_scientific_metadata': True,
              'real_training_state_or_GPU_or_stop_operation': False,
              'fixture_identity': w.identity(), 'fixture_launcher': w.identity(os.getppid()),
              'source': w.pin(__file__), 'watcher_source': w.pin(w.__file__),
              'owned_children': [], 'cleanup': []}
    children = []
    try:
        scope = w.read(args.ROOT_review)
        w.require(scope['kind'] == 'ROOT_postDEV204000_correct_route_watcher_finite_CPU_exact_ONCE_execution_scope_v1'
                  and scope['accepted'] is True and scope['reviewer'] == '/root'
                  and scope['fixture_source'] == w.pin(__file__)
                  and scope['watcher_source'] == w.pin(w.__file__)
                  and Path(scope['fixture_output']) == out, 'Separate exact finite CPU ROOT scope')
        before = [w.pin(item['path']) for item in scope['input_pins']]
        w.require(before == scope['input_pins'], 'All frozen finite input pins')
        ready = out / 'owned_target_ready.json'
        done = out / 'owned_target_done.json'
        toy_source = "import argparse,json,time,psutil,os\nfrom pathlib import Path\nparser=argparse.ArgumentParser()\nparser.add_argument('--worker',action='store_true')\nparser.add_argument('--configuration',required=True)\nparser.add_argument('--private-limit',required=True)\nargs=parser.parse_args()\nassert args.worker and args.private_limit=='8000000000'\nconfig=json.loads(Path(args.configuration).read_text())\np=psutil.Process()\nready=Path(config['ready']);tmp=Path(str(ready)+'.owned.tmp');f=tmp.open('x');json.dump({'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()},f);f.flush();os.fsync(f.fileno());f.close();os.rename(tmp,ready)\nstart=time.monotonic()\nwhile not Path(config['done']).exists() and time.monotonic()-start<60:time.sleep(.02)\n"
        (out / 'CPU_stub.py').write_bytes(toy_source.encode())
        put(out / 'config_stub.json', {'ready': str(ready), 'done': str(done), 'METHOD_STUB': True})
        child = subprocess.Popen([sys.executable, '-Xutf8', '-B', str(out / 'CPU_stub.py'),
            '--worker', '--configuration', str(out / 'config_stub.json'), '--private-limit', '8000000000'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(child)
        launch_id = w.identity(child.pid)
        deadline = time.monotonic() + 10
        while not ready.exists():
            w.require(child.poll() is None and time.monotonic() < deadline, 'Owned CPU target readiness')
            time.sleep(.02)
        target = w.read(ready)
        w.require(target['pid'] != 19320 and w.exact_alive(target), 'Actual owned native PIDctime, never training')
        proc = psutil.Process(target['pid'])
        ancestry = [w.identity(parent.pid) for parent in proc.parents()]
        w.require(target == launch_id or launch_id in ancestry, 'Actual target same launcher or exact descendant')
        report['owned_children'].append({'launcher': launch_id, 'actual_native': target, 'ancestry': ancestry})
        report['cases'].append({'name': 'actual_owned_nativePID_ctime_ancestry', 'passed': True})
        # Native-and-launcher handshake closes the actual old-child scope;
        # Popen wait0 alone is NOT proof that a Windows venv native body exited.
        old_ready = out / 'owned_old_native_ready.json'
        old_code = ("import os,json,time,psutil;from pathlib import Path;"
            "p=psutil.Process();v={'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()};"
            "ready=Path(" + repr(str(old_ready)) + ");tmp=Path(str(ready)+'.owned.tmp');"
            "f=tmp.open('x');json.dump(v,f);f.flush();os.fsync(f.fileno());f.close();os.rename(tmp,ready);time.sleep(.2)")
        deadproc = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-c', old_code],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dead_id = w.identity(deadproc.pid);children.append(deadproc)
        old_deadline = time.monotonic() + 10
        while not old_ready.exists():
            w.require(deadproc.poll() is None and time.monotonic() < old_deadline, 'Actual old-native handshake')
            time.sleep(.01)
        dead_native = w.read(old_ready)
        w.require(dead_native['pid'] != 19320 and w.exact_alive(dead_native), 'Actual owned old-native identity')
        old_ancestry = [w.identity(parent.pid) for parent in psutil.Process(dead_native['pid']).parents()]
        w.require(dead_native == dead_id or dead_id in old_ancestry, 'Exact owned old-native launcher ancestry')
        old_wait = deadproc.wait(timeout=10)
        old_exit_samples = [];exit_deadline = time.monotonic() + 2
        while True:
            old_exit_samples.append({'native_exact_alive': w.exact_alive(dead_native), 'launcher_exact_alive': w.exact_alive(dead_id)})
            if not old_exit_samples[-1]['native_exact_alive'] and not old_exit_samples[-1]['launcher_exact_alive']:break
            w.require(len(old_exit_samples) < 201 and time.monotonic() < exit_deadline, 'Bounded exact native ANDlauncher exit')
            time.sleep(.01)
        w.require(old_wait == 0, 'Actual owned prior launcher wait0; native OS code separately UNKNOWN')
        report['owned_children'].append({'launcher': dead_id, 'actual_native': dead_native,
            'ancestry': old_ancestry, 'Popen_wait': old_wait, 'native_OS_exit': 'UNKNOWN_UNRECORDED',
            'native_and_launcher_exit_observations': old_exit_samples})
        for name in ('source_stub.py',):
            (out / name).write_bytes(b'"METHOD_STUB immutable metadata"\n')
        expected = {'source': w.pin(out / 'source_stub.py'),
                    'configuration': w.pin(out / 'config_stub.json'),
                    'CPU_source': w.pin(out / 'CPU_stub.py'), 'native': w.identity(),
                    'native_launcher': w.identity(os.getppid()), 'wrapper': w.identity(),
                    'reader': w.identity(), 'reader_launcher': w.identity(os.getppid()),
                    'old_CPU': dead_native, 'old_CPU_launcher': dead_id, 'cohort_start': 200000}
        captured = w.capture_native_descendants(expected['native'])
        actual_chain = w.match_current_starting_CPU_chain(captured, expected)
        w.require(actual_chain is not None and actual_chain['native_body'] == target
                  and actual_chain['launcher'] == launch_id, 'Actual owned two-level CPU argv/ancestry chain')
        report['cases'].append({'name': 'actual_two_level_owned_worker_chain', 'passed': True,
                                'capture': captured, 'chain': actual_chain})
        zero = {'native': expected['native'], 'descendants': [], 'count': 0}
        w.require(w.match_current_starting_CPU_chain(zero, expected) is None, 'Zero descendants WAIT, never READY')
        launcher_only = {'native': expected['native'], 'descendants': [item for item in captured['descendants']
                         if item['identity'] == launch_id], 'count': 1}
        w.require(w.match_current_starting_CPU_chain(launcher_only, expected) is None, 'Launcher-only WAIT')
        report['cases'].append({'name': 'zero_and_launcher_only_are_ineligible', 'passed': True})
        for label in ('wrong_ctime', 'wrong_ancestry'):
            wrong_capture = copy.deepcopy(captured)
            record = next(item for item in wrong_capture['descendants'] if item['identity'] == target)
            if label == 'wrong_ctime':
                record['identity']['create_time'] += 1
            else:
                record['ancestry_to_native'][0]['pid'] += 1000000
            try:
                w.match_current_starting_CPU_chain(wrong_capture, expected)
            except ValueError as exc:
                report['cases'].append({'name': label + '_worker_chain_rejected', 'passed': True,
                                        'type': type(exc).__name__, 'message': str(exc)})
            else:
                raise AssertionError('Wrong exact worker identity/ancestry must reject')
        paths = build_case(out / 'positive', expected)
        decision = w.collect_stage(paths, expected)
        w.require(decision['decision'] == 'READY' and decision['actual_expected_completed'] == 4020,
                  'Actual full predicate positive with stale progress.last_saved203800')
        w.require(not Path(decision['same_pending_plan_binding']['path']).exists(), 'No planbody needed/read')
        # Execute ONLY the actual original Monitor.check AST, never its
        # thread-starting constructor or the training module/import/main.
        import ast
        sys.path.insert(0, str(w.P / 'outputs/parakeet_turkish'))
        from bpe_broad_audio_v1 import require as original_require
        driver_path = w.P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'
        driver_tree = ast.parse(driver_path.read_bytes())
        monitor_class = next(n for n in driver_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Monitor')
        check_node = next(n for n in monitor_class.body if isinstance(n, ast.FunctionDef) and n.name == 'check')
        check_namespace = {'require': original_require}
        exec(compile(ast.Module(body=[check_node], type_ignores=[]), '<ACTUAL_original_Monitor_check_AST_ONLY>', 'exec'), check_namespace)
        owner = type('METHOD_STUB_Monitor_owner_no_thread', (), {})()
        owner.error = None;owner.folder = paths['invocation']
        check_namespace['check'](owner)
        report['cases'].append({'name': 'actual_original_Monitor_check_before_invocation_marker_passes', 'passed': True})
        put(paths['root'] / 'stop.json', {'unrelated_root_marker_METHOD_STUB': True})
        check_namespace['check'](owner)
        report['cases'].append({'name': 'unrelated_root_marker_does_not_stop_original_invocation_Monitor', 'passed': True})
        report['actual_original_Monitor_check_AST_sha256'] = __import__('hashlib').sha256(ast.dump(check_node, include_attributes=False).encode()).hexdigest()
        report['actual_original_Monitor_source'] = w.pin(driver_path)
        result = w.publish_marker(paths['stop'], out / 'positive_owned.tmp', {'finite': True, 'updates': 204000},
                                  before_publish=lambda: w.decisive_recheck(paths, expected))
        w.require(paths['stop'].exists() and not (out / 'positive_owned.tmp').exists(), 'Actual exclusive atomic E publication')
        report['cases'].append({'name': 'full_predicate_then_real_atomic_fixture_marker', 'passed': True, 'result': result})
        try:check_namespace['check'](owner)
        except ValueError as error:
            w.require(str(error) == 'Actual owner cooperative stop', 'Actual ORIGINAL supported cooperative exception')
        else:raise AssertionError('Invocation stop marker must trigger actual original Monitor.check')
        report['cases'].append({'name': 'actual_invocation_marker_triggers_original_Monitor_check', 'passed': True,
            'target': str(paths['stop']), 'thread_or_production_import_executed': False})
        changes = {
            'before_DEV': lambda d: d.update(entries_present=False),
            'partial350': lambda d: d['report']['scores']['validation_tr']['asr'].update(utterances=49),
            'bad_preservation': lambda d: d['preservation'].update(both_cursors_journal_and_nextdraw_unchanged=False),
            'checkpoint_mismatch': lambda d: d['persistence'].update(optimizer_updates=203800),
            'checkpoint_pending': lambda d: d.update(latest_update=203800),
            'missed_actual': lambda d: d['progress'].update(actual_optimizer_updates=204020),
            'before_actual': lambda d: d['progress'].update(actual_optimizer_updates=203980),
            'bad_wait': lambda d: d['q']['CPU_actual0'].update(actual_Popen_wait_code=1),
            'bad_credit': lambda d: d['q']['CPU_actual0']['packet_credit_owner'].update(remaining_owned_handle_count=1),
            'bad_source_progress': lambda d: d['progress'].update(source={'wrong': True}),
        }
        expected_nonready = {'before_DEV': 'WAIT', 'checkpoint_pending': 'WAIT',
                             'missed_actual': 'MISSED', 'before_actual': 'WAIT'}
        for name, change in changes.items():
            case_paths = build_case(out / name, expected, change)
            try:
                answer = w.collect_stage(case_paths, expected)
            except ValueError as exc:
                w.require(name not in expected_nonready, 'Unexpected rejection of pending stage')
                observed = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                w.require(name in expected_nonready and answer['decision'] == expected_nonready[name],
                          'Expected actual nonready result, never a swallowed assertion')
                observed = answer
            w.require(not case_paths['stop'].exists(), 'Negative cannot publish stop')
            report['cases'].append({'name': name, 'passed': True, 'observed': observed})
        for name in ('already_ready', 'missing_startup', 'wrong_ctime', 'wrong_config_pin', 'foreign_stop'):
            case_paths = build_case(out / name, expected)
            changed_expected = copy.deepcopy(expected)
            if name == 'already_ready':
                put(case_paths['ready'], {'METHOD_STUB': True})
            elif name == 'missing_startup':
                case_paths['startup_log'].unlink()  # Own fixture file only.
            elif name == 'wrong_ctime':
                changed_expected['native']['create_time'] += 1
            elif name == 'wrong_config_pin':
                changed_expected['configuration']['sha256'] = 'f' * 64
            else:
                case_paths['stop'].write_bytes(b'foreign-owned-marker')
            original_marker = case_paths['stop'].read_bytes() if case_paths['stop'].exists() else None
            try:
                answer = w.collect_stage(case_paths, changed_expected)
            except ValueError as exc:
                observed = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                w.require(name in ('already_ready', 'missing_startup')
                          and answer['decision'] in ('MISSED', 'WAIT'), 'No READY for changed stage')
                observed = answer
            if original_marker is not None:
                w.require(case_paths['stop'].read_bytes() == original_marker, 'Foreign marker untouched')
            else:
                w.require(not case_paths['stop'].exists(), 'Negative no stop')
            report['cases'].append({'name': name, 'passed': True, 'observed': observed})
        for name in ('serialization_failure', 'fsync_failure', 'ready_race', 'foreign_target_race'):
            case_paths = build_case(out / name, expected)
            temporary = out / (name + '_owned.tmp')
            payload = {'value': float('nan')} if name == 'serialization_failure' else {'finite': True}
            def fsync(fd):
                if name == 'fsync_failure':
                    raise OSError('METHOD_STUB injected fsync failure before publication')
                os.fsync(fd)
            def prepublish():
                if name == 'ready_race':
                    put(case_paths['ready'], {'METHOD_STUB': True})
                if name == 'foreign_target_race':
                    # Deliberately after the actor's target precheck: Windows rename must refuse it.
                    case_paths['stop'].write_bytes(b'late-foreign-marker')
                    return {'METHOD_STUB_race': True}
                return w.decisive_recheck(case_paths, expected)
            try:
                w.publish_marker(case_paths['stop'], temporary, payload, before_publish=prepublish, fsync=fsync)
            except (ValueError, OSError) as exc:
                observed = {'type': type(exc).__name__, 'message': str(exc),
                            'failed_owned_temp_preserved': temporary.exists()}
            else:
                raise AssertionError('Expected prepublication/exclusive-rename failure')
            if name == 'foreign_target_race':
                w.require(case_paths['stop'].read_bytes() == b'late-foreign-marker', 'Late foreign target unchanged')
            else:
                w.require(not case_paths['stop'].exists(), 'No partial stop file on failed publication')
            w.require(temporary.exists() == (name != 'serialization_failure'), 'Honest failed-temp preservation')
            report['cases'].append({'name': name, 'passed': True, 'observed': observed})
        done.write_bytes(b'done')
        w.require(child.wait(timeout=10) == 0, 'Owned target exits0 after fixture-only done')
        report['owned_children'][0]['Popen_wait'] = 0
        report['inputs_before'] = before
        report['inputs_after'] = [w.pin(item['path']) for item in scope['input_pins']]
        w.require(report['inputs_after'] == before, 'All frozen inputs unchanged')
        report['passed'] = True
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        # Only direct owned Popen descendants, never any production/reader identity.
        for child in children:
            if child.poll() is None:
                owned = next((item['launcher'] for item in report['owned_children'] if item['launcher']['pid'] == child.pid), None)
                if owned and w.exact_alive(owned) and child.pid not in (19320, 39856, 21928, 42068, 40536, 26892, 30128):
                    root = psutil.Process(child.pid)
                    captured = [w.identity(p.pid) for p in root.children(recursive=True)]
                    for who in reversed(captured):
                        if w.exact_alive(who) and child.pid in [p.pid for p in psutil.Process(who['pid']).parents()]:
                            psutil.Process(who['pid']).terminate()
                    if w.exact_alive(owned):
                        child.terminate()
                    report['cleanup'].append({'owned': owned, 'termination_after_failure_only': True})
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    report['cleanup'].append({'pid': child.pid, 'wait': 'UNKNOWN_timeout'})
        report['wall_seconds'] = time.monotonic() - began
        result = w.write_once(out / 'report.json', report)
        print(json.dumps({'report': result, 'passed': report['passed'], 'cases': len(report['cases'])}), flush=True)


if __name__ == '__main__':
    main()
