"""Normalize observed metadata provenance only; never infer exit0 from payload."""
import hashlib
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


def binding(path):
    path = Path(path).resolve()
    stat = path.stat()
    assert stat.st_size < 8_000_000, 'Only bounded metadata/source bindings; no payload reads'
    data = path.read_bytes()
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    return {'path': str(path), 'bytes': stat.st_size, 'sha256': hashlib.sha256(data).hexdigest(),
            'mtime_ns': stat.st_mtime_ns}


def checked(pin):
    actual = binding(pin['path'])
    assert all(actual[key] == value for key, value in pin.items())
    return read_json(pin['path'])


def write_once(path, value):
    path = Path(path)
    if path.exists():
        assert read_json(path) == value, 'Preserve existing different normalized provenance'
    else:
        write_json(path, value)
    return binding(path)


def guard_facts(path):
    raw = read_json(path)
    if raw.get('kind') in ('ASR18_guarded_FLAC_native_closure_v3', 'ASR18_guarded_FLAC_native_closure_v5'):
        assert raw['native_return_code'] == 0 and raw['watchdog_clean'] is True
        facts = (raw['PID'], raw['samples'], raw['peak_private_bytes'])
        mapping = 'FLAC_guard_v3_or_v5_watchdog_clean_asserts_joined_monitor_before_footer'
    else:
        assert raw['actual_exit_code'] == 0 and raw['monitor_clean'] is True and raw['monitor_joined'] is True
        assert raw['observed']['monitor_error'] is None
        facts = (raw['pid'], raw['observed']['samples'], raw['observed']['peak_private_bytes'])
        mapping = 'bounded_CPU_guard_v2_explicit_actual_exit_and_monitor_clean_and_joined'
    assert facts[0] > 0 and facts[1] >= 3 and facts[2] < 2_000_000_000
    assert raw['CPU_private_cap_bytes'] == 2_000_000_000 and raw['CUDA_masked'] is True
    return facts, mapping


def emit(folder, *, owner, source_sha256, mode, launcher_PID, original_wait_path,
         original_footer_path, result_receipt_path, worker_source_path, semantics,
         parent_actual_tool_closure=None, PID_correction_provenance=None):
    assert owner in ('/root/cpu_data_preparation', '/root/memory_diagnosis')
    assert mode in ('prepare', 'completion_verify', 'audit')
    folder = Path(folder).resolve()
    assert folder.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve())
    folder.mkdir(parents=True, exist_ok=True)
    (pid, samples, peak), mapping = guard_facts(original_footer_path)
    footer = {'kind': 'ASR18_normalized_FLAC_native_guard_completion_v1',
              'native_PID': pid, 'actual_exit_code': 0, 'watchdog_clean': True,
              'monitor_joined': True, 'samples': samples, 'peak_private_bytes': peak,
              'CPU_private_cap_bytes': 2_000_000_000, 'CUDA_masked': True,
              'original_guard_footer': binding(original_footer_path),
              'normalization_mapping': mapping}
    footer_pin = write_once(folder / f'{source_sha256}_{mode}_guard.json', footer)
    proof = {'kind': 'ASR18_actual_owned_FLAC_stage_completion_v1', 'schema_version': 1,
             'owner': owner, 'source_sha256': source_sha256, 'mode': mode,
             'actual_exit_code': 0, 'native_PID': pid, 'launcher_PID': launcher_PID,
             'parent_observed_wait': True, 'original_parent_wait_proof': binding(original_wait_path),
             'original_guard_footer': binding(original_footer_path), 'native_guard_closure': footer_pin,
             'result_receipt': binding(result_receipt_path), 'worker_source': binding(worker_source_path),
             'original_PID_field_semantics': semantics, 'normalizer_source': binding(__file__),
             'parent_actual_tool_closure': binding(parent_actual_tool_closure) if parent_actual_tool_closure else None,
             'PID_correction_provenance': binding(PID_correction_provenance) if PID_correction_provenance else None}
    proof_pin = write_once(folder / f'{source_sha256}_{mode}_actual_exit.json', proof)
    return {'mode': mode, 'actual_exit_code': 0, 'owner': owner, 'native_PID': pid,
            'parent_observed_wait': True, 'actual_exit_proof': proof_pin,
            'native_guard_closure': footer_pin, 'result_receipt': proof['result_receipt'],
            'worker_source': proof['worker_source']}


def normalize_owned(proof_path, result_receipt_path, folder, *, parent_actual_tool_closure,
                    PID_correction_provenance=None):
    raw = read_json(proof_path)
    assert raw['kind'] == 'ASR18_actual_owned_FLAC_child_exit_v1'
    assert raw['actual_exit_code'] == 0
    assert raw.get('parent_observed_wait', True) is True
    stage_folder = Path(proof_path).parent
    config = read_json(stage_folder / 'config.json')
    assert config['job']['sha256'] == raw['source_sha256'] and config['mode'] == raw['mode']
    assert Path(result_receipt_path).resolve() == (stage_folder / 'result.json').resolve()
    footer_path = raw['native_closure']['path']
    checked(raw['native_closure'])
    footer = read_json(footer_path)
    (native_pid, _, _), _ = guard_facts(footer_path)
    actual_outer = read_json(parent_actual_tool_closure)
    assert actual_outer['owner'] == '/root/cpu_data_preparation'
    assert actual_outer['actual_exit_code'] in (0, 1)  # A later supported outer stop does not relabel the observed child.
    assert actual_outer.get('actual_tool_chunk_id') or actual_outer.get('actual_tool_chunk')
    if raw['native_PID'] != native_pid:
        assert PID_correction_provenance is not None
        correction = read_json(PID_correction_provenance)
        matches = [item for item in correction['all_native_children']
                   if item['native_closure_path'] == footer_path and item['real_native_PID'] == native_pid
                   and item['actual_native_return_code'] == 0
                   and item['native_closure_SHA256'] == raw['native_closure']['sha256']
                   and item['legacy_actual_owned_exit_native_PID_field_is_launcher'] is True]
        assert len(matches) == 1
        semantics = 'launcher_mislabelled_in_original_native_PID_field_preserved_with_bound_correction'
        launcher = raw['native_PID']
    else:
        semantics = 'native_PID_matches_original_guard_footer'
        launcher = raw.get('launcher_PID')
    checked(raw['worker_source'])
    result = read_json(result_receipt_path)
    assert result.get('mode', raw['mode']) == raw['mode']
    return emit(folder, owner='/root/cpu_data_preparation', source_sha256=raw['source_sha256'],
                mode=raw['mode'], launcher_PID=launcher, original_wait_path=proof_path,
                original_footer_path=footer_path, result_receipt_path=result_receipt_path,
                worker_source_path=raw['worker_source']['path'], semantics=semantics,
                parent_actual_tool_closure=parent_actual_tool_closure,
                PID_correction_provenance=PID_correction_provenance)


def normalize_delegated_fixture(report_path, source_path, source_sha256, mode, result_receipt_path, folder):
    report = read_json(report_path)
    matches = [item for item in report['actual_owned_results']
               if item['source'] == str(source_path) and item.get('phase') == mode]
    assert len(matches) == 1
    record = matches[0]
    assert record['actual_exit_code'] == 0 and record['parent_observed_wait'] is True
    checked(record['guard_closure'])
    checked(record['source_file'])
    (native_pid, _, _), _ = guard_facts(record['guard_closure']['path'])
    assert native_pid == record['actual_worker_PID']
    result = read_json(result_receipt_path)
    if mode == 'prepare':
        assert result['source']['source_sha256'] == source_sha256 and result['complete'] is True
    else:
        assert result['source_sha256'] == source_sha256 and result['passed'] is True
    return emit(folder, owner='/root/memory_diagnosis', source_sha256=source_sha256,
                mode=mode, launcher_PID=record['owned_handle_PID'], original_wait_path=report_path,
                original_footer_path=record['guard_closure']['path'], result_receipt_path=result_receipt_path,
                worker_source_path=record['source_file']['path'],
                semantics='delegated_parent_report_actual_worker_PID_matches_native_guard')
