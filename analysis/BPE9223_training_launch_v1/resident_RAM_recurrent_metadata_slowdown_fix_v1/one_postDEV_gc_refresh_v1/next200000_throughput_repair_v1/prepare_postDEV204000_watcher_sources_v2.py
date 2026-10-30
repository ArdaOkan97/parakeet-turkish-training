"""INACTIVE204k watcher derivation/RAW/AST only; no candidate import or arm."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil

P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
W = Path(__file__).parent
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from resource_watchdog_v2 import sample_memory


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def who(pid):
    proc = psutil.Process(pid)
    return {'pid': proc.pid, 'create_time': proc.create_time(), 'ppid': proc.ppid()}


def write(path, value):
    with Path(path).open('xb') as file:
        file.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())
        file.flush()
        os.fsync(file.fileno())
    return pin(path)


started = time.monotonic()
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
initial = sample_memory(os.getpid(), 'postDEV204000_watcher_source_preparation')
initial_pin = write(W / 'postDEV204000_watcher_source_preparation_initial_v2.json',
    {'sample': initial, 'preparer': who(os.getpid()), 'launcher': who(os.getppid()),
     'actual_numeric_before_original_full_prospective_predicate': True})
assert initial['paired_sampling_within_one_second']
assert initial['worker_private_memory']['private_bytes'] < 536870912
assert initial['system_memory']['commit_available_bytes'] >= 5055306368
assert initial['system_memory']['physical_available_bytes'] >= 2305306368
assert shutil.disk_usage('E:/').free >= 100000000000
ledgers = []


def derive(base_name, new_name, edits):
    base = W / base_name
    before = base.read_bytes()
    text = before.decode()
    steps = []
    for old, new, expected_count in edits:
        count = text.count(old)
        assert count == expected_count if expected_count is not None else count > 0, (old, count)
        text = text.replace(old, new)
        steps.append({'old': old, 'new': new, 'count': count})
    tree = ast.parse(text)
    inverse = text
    for step in reversed(steps):
        assert inverse.count(step['new']) == step['count'], step['new']
        inverse = inverse.replace(step['new'], step['old'])
    assert inverse.encode() == before
    path = W / new_name
    if path.exists():
        assert path.read_bytes() == text.encode(), 'Only exact already-frozen source reused after metadata failure'
    else:
        with path.open('xb') as file:
            file.write(text.encode())
            file.flush()
            os.fsync(file.fileno())
    ledgers.append({'base': pin(base), 'successor': pin(path), 'ordered_RAW_steps': steps,
                    'whole_forward_inverse_exact': True,
                    'AST_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
                    'source_only_no_import_execution_or_compile': True})
    return pin(path)


chain_helper = '''def match_current_starting_CPU_chain(capture, expected):
    """Only an ACTUAL two-level native->launcher->CPU14 body is eligible.

    Zero descendants/launcher-only are WAIT, never a fabricated closure. Every
    captured current descendant remains in the evidence; no truncation or PID
    rank inference. No claim about future descendants after this observation.
    """
    native = expected['native']
    require(capture['native'] == native and capture['count'] == len(capture['descendants']) <= 8
            and exact_alive(native), 'Exact bounded unchanged current-native capture')
    suffix = ['-Xutf8', '-B', str(Path(expected['CPU_source']['path']).resolve()),
              '--worker', '--configuration', expected['configuration']['path'],
              '--private-limit', '8000000000']
    matches = []
    for record in capture['descendants']:
        who = record['identity']
        ancestry = record['ancestry_to_native']
        require(exact_alive(who) and ancestry and ancestry[-1] == native
                and who['ppid'] == ancestry[0]['pid']
                and all(ancestry[index]['ppid'] == ancestry[index + 1]['pid']
                        for index in range(len(ancestry) - 1)),
                'Fresh identity and exact linked ancestry, never a guessed child')
        if record['argv'][1:] == suffix:
            matches.append(record)
    chains = []
    for parent in matches:
        if parent['identity']['ppid'] != native['pid']:
            continue
        for body in matches:
            if body['identity']['ppid'] == parent['identity']['pid']:
                chains.append({'launcher': parent['identity'], 'native_body': body['identity'],
                               'launcher_argv': parent['argv'], 'native_body_argv': body['argv']})
    require(len(chains) <= 1, 'Unique actual CPU14 worker chain; ambiguity is no-write failure')
    return None if not chains else chains[0]


'''
capture_old = "        records.append({'identity': who, 'ancestry_to_native': chain})"
capture_new = """        argv = psutil.Process(who['pid']).cmdline()
        require(type(argv) is list and 0 < len(argv) <= 16
                and all(type(value) is str for value in argv)
                and sum(len(value) for value in argv) <= 32768
                and exact_alive(who), 'Bounded actual descendant argv and same identity')
        records.append({'identity': who, 'ancestry_to_native': chain, 'argv': argv})"""
ready_old = "    return {'decision': 'READY', 'actual': actual, 'progress': progress, 'latest': latest,"
ready_new = """    captured = capture_native_descendants(expected['native'])
    starting_chain = match_current_starting_CPU_chain(captured, expected)
    if starting_chain is None:
        return {'decision': 'WAIT', 'reason': 'actual two-level newCPU14 body not yet captured',
                'actual': actual, 'current_native_descendants': captured}
    return {'decision': 'READY', 'actual': actual, 'progress': progress, 'latest': latest,"""
arm_old = "        paths = layout(B, I)\n        while True:"
arm_new = """        paths = layout(B, I)
        arm_progress = read(paths['progress'])
        require(scope['arm_frontier_band'] == [203000, 203980]
                and 203000 <= arm_progress['actual_optimizer_updates'] <= 203980
                and arm_progress['producer'] == expected['native']
                and exact_alive(expected['old_CPU']) and exact_alive(expected['old_CPU_launcher']),
                'Arm only in reviewed203000..203980 band with actual200k cohort; no early/late retry')
        report['arm_actual_progress'] = arm_progress['actual_optimizer_updates']
        report['arm_frontier_band'] = [203000, 203980]
        while True:"""
watcher = derive('postDEV200000_stop_watcher_v3.py', 'postDEV204000_stop_watcher_v1.py', [
    ('200000', '204000', None),
    ("OLD_CPU = {'pid': 13472, 'create_time': 1792822495.866724, 'ppid': 39752}",
     "OLD_CPU = {'pid': 42068, 'create_time': 1792830275.0700862, 'ppid': 40536}", 1),
    ("OLD_CPU_LAUNCHER = {'pid': 39752, 'create_time': 1792822495.8054123, 'ppid': 19320}",
     "OLD_CPU_LAUNCHER = {'pid': 40536, 'create_time': 1792830275.0545292, 'ppid': 19320}", 1),
    ("'cohort_start': 196000", "'cohort_start': 200000", 1),
    (capture_old, capture_new, 1),
    ('def expected_current():', chain_helper + 'def expected_current():', 1),
    (ready_old, ready_new, 1),
    ("'current_native_descendants': capture_native_descendants(expected['native']),",
     "'current_native_descendants': captured, 'actual_starting_CPU_chain': starting_chain,", 1),
    (arm_old, arm_new, 1)])

# The old21 science/atomic cases are inherited; real captured owned launcher/body
# now also tests the new exact command/ancestry requirement. No production read.
base_fixture = (W / 'verify_postDEV200000_stop_watcher_CPU_v2.py').read_text()
child_start = base_fixture.index('        child_code = (')
child_end = base_fixture.index('        children.append(child)', child_start)
old_child = base_fixture[child_start:child_end]
toy_source = """import argparse,json,time,psutil
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--worker',action='store_true')
parser.add_argument('--configuration',required=True)
parser.add_argument('--private-limit',required=True)
args=parser.parse_args()
assert args.worker and args.private_limit=='8000000000'
config=json.loads(Path(args.configuration).read_text())
p=psutil.Process()
Path(config['ready']).write_text(json.dumps({'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}))
start=time.monotonic()
while not Path(config['done']).exists() and time.monotonic()-start<60:time.sleep(.02)
"""
new_child = """        toy_source = """ + repr(toy_source) + """
        (out / 'CPU_stub.py').write_bytes(toy_source.encode())
        put(out / 'config_stub.json', {'ready': str(ready), 'done': str(done), 'METHOD_STUB': True})
        child = subprocess.Popen([sys.executable, '-Xutf8', '-B', str(out / 'CPU_stub.py'),
            '--worker', '--configuration', str(out / 'config_stub.json'), '--private-limit', '8000000000'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
"""
proof_needle = "        paths = build_case(out / 'positive', expected)"
proof_insert = """        captured = w.capture_native_descendants(expected['native'])
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
        paths = build_case(out / 'positive', expected)"""
fixture = derive('verify_postDEV200000_stop_watcher_CPU_v2.py', 'verify_postDEV204000_stop_watcher_CPU_v1.py', [
    ('200000', '204000', None),
    ('postDEV204000_stop_watcher_v2', 'postDEV204000_stop_watcher_v1', 1),
    ('200020', '204020', None), ('199980', '203980', None),
    ('199800', '203800', None), ('196000', '200000', None),
    # old_child is transformed by preceding literal target substitutions.
    (old_child, new_child, 1),
    ("for name in ('source_stub.py', 'CPU_stub.py', 'config_stub.json'):",
     "for name in ('source_stub.py',):", 1),
    ("'CPU_source': w.pin(out / 'CPU_stub.py'), 'native': target,\n                    'native_launcher': launch_id, 'wrapper': w.identity(),",
     "'CPU_source': w.pin(out / 'CPU_stub.py'), 'native': w.identity(),\n                    'native_launcher': w.identity(os.getppid()), 'wrapper': w.identity(),", 1),
    (proof_needle, proof_insert, 1),
    ('(19320, 39856, 21928, 13472, 39752, 26892, 30128)',
     '(19320, 39856, 21928, 42068, 40536, 26892, 30128)', 1)])
external = derive('run_postDEV200000_stop_watcher_CPU_external_v1.py',
                  'run_postDEV204000_stop_watcher_CPU_external_v1.py', [
    ('200000', '204000', None),
    ('{19320, 39856, 21928, 13472, 39752, 26892, 30128}',
     '{19320, 39856, 21928, 42068, 40536, 26892, 30128}', 1)])
ledger_pin = write(W / 'postDEV204000_watcher_fixture_external_full_RAW_AST_v1.json',
    {'kind': 'ASR18_postDEV204000_source37_exact_derivation_v1', 'sources': ledgers,
     'no_existing_production_source_or_policy_changed': True})
scope = W.parent / 'ROOT_next204000_eval_recurrent_throughput_repair_INACTIVE_preparation_scope_v1.json'
cohort = W / 'ROOT_actual200000_normal_CPU_binding_and_missed_stop_for_next204000_v1.json'
inputs = [watcher, fixture, external, ledger_pin, pin(Path(__file__)), pin(scope), pin(cohort),
    pin(P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'),
    pin(S / 'fastmss_resident_feature_stream_v14.py'), pin(S / 'resource_watchdog_v2.py'),
    pin(Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000026\configuration.json')),
    pin(Path(r'E:\gpu_training_scratchpad\storage_config.json')), pin(Path(r'E:\gpu_training_scratchpad\use_storage.ps1'))]
review = W / 'ROOT_postDEV204000_stop_watcher_finite_CPU_exact_ONCE_execution_scope_v1.json'
handoff = write(W / 'postDEV204000_watcher_INACTIVE_source_CPU_handoff_v1.json',
    {'kind': 'ASR18_postDEV204000_watcher_INACTIVE_source_CPU_handoff_v1', 'execution_admitted': False,
     'watcher_source': watcher, 'fixture_source': fixture, 'external_source': external,
     'input_pins': inputs, 'full_RAW_AST': ledger_pin, 'actual200000_cohort_ROOT': pin(cohort),
     'proposed_CPU_prefix': [r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe', '-X', 'utf8', '-B'],
     'proposed_CPU_arguments': [external['path'], '--ROOT-review', str(review)],
     'fixture_output': str(W / 'watcher204000_finite_CPU_actual_01'),
     'external_output': str(W / 'watcher204000_finite_CPU_external_actual_01'),
     'proposed_live_arm_band': [203000, 203980], 'watcher_wall_seconds_unchanged': 3600,
     'original_prospective_commit_bytes': 5055306368, 'original_prospective_physical_bytes': 2305306368,
     'private_bytes': 536870912, 'E_free_bytes': 100000000000,
     'CPU_fixture_external_wall_seconds': 120, 'poll_seconds': .25,
     'new_chain_requirement': 'Actual fresh native->launcher->CPU14body, exact --worker/config/private8000000000 argv, allcurrentdescendants<=8 captured, zero/launcheronly WAIT; no methodclose/futurechild/racefree claim',
     'scientific_owners_in_finite_fixture': 'METHOD_STUB generated metadata; actual owned process chain and atomic E fixture actor',
     'production_action_NOW': False, 'no_prior200000_retry_or_ordinary_stop': True,
     'preserved_source_preparation_failure': {'actual_TOOL_exit': 1, 'chunk': '68cb7a',
        'source': pin(W / 'prepare_postDEV204000_watcher_sources_v1.py'),
        'initial_numeric': pin(W / 'postDEV204000_watcher_source_preparation_initial_v1.json'),
        'already_frozen_watcher_source_reused_only_after_full_byte_equality': True}})
assert [pin(item['path']) for item in inputs] == inputs
assert time.monotonic() - started < 60
result = write(W / 'postDEV204000_watcher_source_only_preparation_result_v1.json',
    {'watcher': watcher, 'fixture': fixture, 'external': external, 'handoff': handoff,
     'full_RAW_AST': ledger_pin, 'initial_full_numeric': initial_pin,
     'preparer': who(os.getpid()), 'launcher': who(os.getppid()),
     'no_candidate_import_execution_or_arm': True, 'wall_seconds': time.monotonic() - started})
print(json.dumps({'result': result, 'handoff': handoff, 'watcher': watcher,
                  'fixture': fixture, 'external': external,
                  'preparer': who(os.getpid()), 'launcher': who(os.getppid())}), flush=True)
