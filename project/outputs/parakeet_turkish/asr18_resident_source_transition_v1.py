"""Truthful cooperative source transition; never converts a failed exit to normal0."""
import argparse, ctypes, hashlib, json, os, time
from ctypes import wintypes
from pathlib import Path
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require

ROOT = Path(r'E:\gpu_training_scratchpad')


def check_saved_relation(saved, whole, actual_updates, saved_updates):
    require(whole['path'] == saved['path'] and whole['bytes'] >= saved['offset'], 'Same retained journal, complete saved prefix')
    require(actual_updates >= saved_updates, 'Actual update counter cannot precede saved frontier')
    if whole['bytes'] == saved['offset']:
        require(whole['sha256'] == saved['sha256'] and actual_updates == saved_updates,
                'No-tail transition requires every actual live update saved')
    return whole['bytes'] - saved['offset']


def checked_source_transition_resume(pin, *, base, folder, pointer_pin):
    from fastmss_owned_process_closure_v2 import dead
    require(record(pin['path']) == pin, 'Exact actual transition closure')
    value = read_json(pin['path'])
    intent_pin = base['resident_source_transition_intent']
    require(value['intent'] == intent_pin and record(intent_pin['path']) == intent_pin, 'Reviewed explicit transition intent')
    intent = read_json(intent_pin['path'])
    require(intent['kind'] == 'ASR18_authorized_private_prefetch_source_transition_intent_v1'
            and intent['branch'] == value['branch'] == str(folder)
            and value['kind'] == 'ASR18_actual_cooperative_prefetch_source_transition_TOOL1_v1'
            and value['intentional_source_transition'] is True and value['normal_native0_claimed'] is False
            and value['actual_native_exit_code'] == value['actual_wrapper_exit_code'] == 1,
            'Actual intentional abnormal exit, never resource failure or normal0 relabel')
    require(value['checkpoint_pointer'] == pointer_pin and value['saved_optimizer_updates'] >= intent['minimum_saved_updates']
            and all(dead(w) for w in intent['captured_identities']), 'Exact saved frontier and every old identity dead')
    require([x['identity'] for x in value['actual_process_exit_codes']] == intent['captured_identities']
            and all(x['actual_exit_code'] != 259 for x in value['actual_process_exit_codes']), 'All captured process handles actually exited')
    for key in ('old_driver_source', 'old_wrapper_source', 'old_configuration', 'old_invocation_configuration', 'old_boot'):
        require(record(intent[key]['path']) == intent[key], 'Preserve old source/configuration/owner evidence')
    for key in ('stop_request', 'original_failure', 'original_monitor', 'original_native_log',
                'checkpoint_pointer', 'checkpoint_persistence', 'whole_journal_including_preserved_tail'):
        require(record(value[key]['path']) == value[key], 'Preserved actual stop/failure/state/journal evidence')
    failure = read_json(value['original_failure']['path'])
    require(failure['error'] == "ValueError('Actual owner cooperative stop')" and failure['producer'] == intent['old_native']
            and failure['actual_optimizer_updates'] == value['actual_optimizer_updates_before_exit']
            and failure['last_checkpoint'] == value['saved_optimizer_updates'],
            'This entry only accepts the exact requested cooperative guard exit')
    monitor = read_json(value['original_monitor']['path'])
    require(monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,
            'No concurrent memory-monitor failure hidden as intentional transition')
    pointer = read_json(pointer_pin['path'])
    require(pointer['committed_updates'] == value['saved_optimizer_updates']
            and pointer['journal_prefix'] == value['saved_journal_prefix'], 'Exact coherent saved journal authority')
    stop = read_json(value['stop_request']['path']); proof = read_json(value['checkpoint_persistence']['path'])
    require(stop['intent'] == intent_pin and stop['observed_verified_checkpoint']['pointer'] == pointer_pin
            and proof['named_tensors_verified'] == 854 and proof['Adam_states_verified'] == 215,
            'Requested verified checkpoint is the actual all-state restore frontier')
    tail = check_saved_relation(value['saved_journal_prefix'], value['whole_journal_including_preserved_tail'],
                                value['actual_optimizer_updates_before_exit'], value['saved_optimizer_updates'])
    require(tail == value['unsaved_tail_bytes'], 'Truthful actual tail length')
    from bpe_bounded_production_journal_v4 import prefix
    require(prefix(Path(value['saved_journal_prefix']['path']), value['saved_journal_prefix']['offset'], lambda: None).hexdigest()
            == value['saved_journal_prefix']['sha256'], 'Exact saved prefix, never replay later tail')
    if tail:
        require(record(value['preserved_unsaved_tail']['path']) == value['preserved_unsaved_tail']
                and value['preserved_unsaved_tail']['bytes'] == tail, 'Actual tail kept separately and in original journal')
    return value


def watch(intent_path, review_path, output):
    """Request the existing guard only after a fresh verified checkpoint publication."""
    import psutil
    from asr18_memory_diagnosis_v2 import storage
    storage(); require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only transition observer')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    intent_pin = record(intent_path); intent = read_json(intent_path); review = read_json(review_path)
    require(review['accepted'] is True and review['reviewer'] == '/root'
            and review['cooperative_transition_authorized'] is True
            and review['transition_source'] == record(__file__) and review['transition_intent'] == intent_pin,
            'Exact ROOT-reviewed transition source and intent before any stop')
    output = Path(output).resolve(); require(output.is_relative_to(ROOT) and not output.exists(), 'Fresh E transition observer output')
    output.mkdir(); branch = Path(intent['branch']); invocation = Path(intent['old_invocation'])
    require(invocation.parent == branch / 'invocations' and not (invocation / 'stop.json').exists(), 'Exact live owner, no prior stop')
    for key in ('old_driver_source', 'old_wrapper_source', 'old_configuration', 'old_invocation_configuration', 'old_boot'):
        require(record(intent[key]['path']) == intent[key], 'Unchanged old admitted inputs')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.restype = wintypes.BOOL
    handles = []
    try:
        for who in intent['captured_identities']:
            process = psutil.Process(who['pid'])
            require(abs(process.create_time() - who['create_time']) < .001, 'Exact old live PID/create-time')
            handle = kernel.OpenProcess(0x1000 | 0x100000, False, who['pid'])
            require(bool(handle), 'Real old-process wait/query handle before stop')
            handles.append((who, handle))
        initial = read_json(branch / 'latest_checkpoint.json'); began = time.monotonic()
        write_json(output / 'observer_ready.json', {'intent': intent_pin, 'initial_checkpoint': initial,
                   'actual_process_handles': len(handles), 'source': record(__file__)})
        last_guard = 0.
        while True:
            require(time.monotonic() - began < 1800, 'Bounded wait for a new actual saved frontier')
            require(psutil.Process().memory_info().private < 268435456, 'Bounded CPU-only observer private memory')
            if time.monotonic() - last_guard >= 1:
                from resource_watchdog_v2 import sample_memory
                sample = sample_memory(os.getpid(), 'CPU_only_checkpoint_transition_observer')
                require(sample['paired_sampling_within_one_second']
                        and sample['system_memory']['commit_available_bytes'] >= 2000000000 + 67108864
                        and sample['system_memory']['physical_available_bytes'] >= 1500000000 + 67108864,
                        'Unchanged real system headroom plus full64MiB observer charge')
                require(__import__('shutil').disk_usage(ROOT).free >= 100000000000 + 67108864, 'Unchanged E100GB reserve')
                last_guard = time.monotonic()
            latest = read_json(branch / 'latest_checkpoint.json')
            if latest['full_state_verified'] and latest['committed_updates'] >= intent['minimum_saved_updates'] \
                    and latest['committed_updates'] > initial['committed_updates']:
                pointer_pin = latest['pointer']; pointer = read_json(pointer_pin['path'])
                # Whole-file identity is the last gate before requesting the existing cooperative stop.
                saved = pointer['journal_prefix']; whole = record(saved['path'])
                if whole == {'path': saved['path'], 'sha256': saved['sha256'], 'bytes': saved['offset']}:
                    write_json(invocation / 'stop.json', {'kind': 'ASR18_explicit_cooperative_prefetch_source_transition_request_v1',
                               'intent': intent_pin, 'ROOT_review': record(review_path), 'observed_verified_checkpoint': latest,
                               'observed_whole_journal': whole, 'normal0_requested_or_claimed': False})
                    break
            time.sleep(.002)
        requested = time.monotonic(); codes = []
        while True:
            codes = []
            for who, handle in handles:
                code = wintypes.DWORD(); require(kernel.GetExitCodeProcess(handle, ctypes.byref(code)), 'Actual process exit code')
                codes.append({'identity': who, 'actual_exit_code': code.value})
            if all(x['actual_exit_code'] != 259 for x in codes): break
            require(time.monotonic() - requested < 180, 'Old family must actually close; no forced kill or unchanged retry')
            time.sleep(.1)
        failure = read_json(invocation / 'actual_failure.json'); latest = read_json(branch / 'latest_checkpoint.json')
        pointer_pin = latest['pointer']; pointer = read_json(pointer_pin['path']); saved = pointer['journal_prefix']; whole = record(saved['path'])
        tail_bytes = check_saved_relation(saved, whole, failure['actual_optimizer_updates'], latest['committed_updates'])
        tail_pin = None; tail_updates = []
        if tail_bytes:
            destination = output / 'preserved_unsaved_tail.jsonl'
            with Path(saved['path']).open('rb') as src:
                src.seek(saved['offset']); tail = src.read()
            destination.write_bytes(tail); tail_pin = record(destination)
            tail_updates = [json.loads(line)['optimizer_updates'] for line in tail.splitlines() if line.strip()]
        lookup = {x['identity']['pid']: x['actual_exit_code'] for x in codes}
        result = {'kind': 'ASR18_actual_cooperative_prefetch_source_transition_TOOL1_v1', 'intent': intent_pin,
                  'intentional_source_transition': True, 'normal_native0_claimed': False, 'branch': str(branch),
                  'actual_native_exit_code': lookup[intent['old_native']['pid']],
                  'actual_wrapper_exit_code': lookup[intent['old_wrapper']['pid']], 'actual_process_exit_codes': codes,
                  'all_old_processes_observed_closed': True, 'checkpoint_pointer': pointer_pin,
                  'checkpoint_persistence': latest['persistence'], 'saved_optimizer_updates': latest['committed_updates'],
                  'actual_optimizer_updates_before_exit': failure['actual_optimizer_updates'], 'saved_journal_prefix': saved,
                  'whole_journal_including_preserved_tail': whole, 'unsaved_tail_bytes': tail_bytes,
                  'preserved_unsaved_tail': tail_pin, 'preserved_tail_positive_updates': tail_updates,
                  'stop_request': record(invocation / 'stop.json'), 'original_failure': record(invocation / 'actual_failure.json'),
                  'original_monitor': record(invocation / 'monitor_completion.json'),
                  'original_native_log': record(intent['old_native_log']), 'source': record(__file__),
                  'actual_exec_tool_exit_claimed_by_observer': False}
        path = output / 'actual_transition_closure.json'; write_json(path, result)
        checked_source_transition_resume(record(path), base={'resident_source_transition_intent': intent_pin},
                                         folder=branch, pointer_pin=pointer_pin)
        print(json.dumps({'actual_transition_closure': record(path), 'saved_updates': latest['committed_updates'],
                          'unsaved_tail_bytes': tail_bytes}), flush=True)
    finally:
        for _, handle in handles: kernel.CloseHandle(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--intent', required=True); parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); watch(args.intent, args.review, args.output)
