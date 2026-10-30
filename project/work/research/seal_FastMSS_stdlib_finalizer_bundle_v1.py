"""Seal inactive CPU6/sink4 source proposal; never access live feature arrays."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'outputs/parakeet_turkish'
F = Path(r'E:\gpu_training_scratchpad\analysis\FastMSS_integration_v1')
A = F.parent / 'ASR18_memory_diagnosis_v1'


def pin(path):
    path = Path(path).resolve()
    assert path.suffix in ('.json', '.py'), 'Metadata/source sealing only, never feature/audio/payload bytes'
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    assert not path.exists(), 'Do not overwrite sealed evidence'
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return pin(path)


def main():
    failure_path = A / 'full_DATA_two_window_native_02/actual_owned_CLOSED1_v1.json'
    failure_pin = pin(failure_path)
    assert failure_pin['sha256'] == '7fd685f46de53a8528cb3794bac4b83e9942402456d09012e6568a85bbdd071a'
    failure = load(failure_path)
    assert failure['actual_exit_code'] == 1 and failure['normal_prepared_reference_children'] == failure['normal_prepared_frontend_children'] == 20
    assert failure['GPU_launched'] is False and failure['actual_Adam_updates'] == 0 and failure['final_manifest_published'] is False
    original_audio_child0s = []
    for item in failure['completed_prepared_not_committed_updates']:
        assert pin(item['path']) == item
        completed = load(item['path'])
        for stage in ('reference', 'frontend'):
            actual = completed[stage]['actual0']
            assert pin(actual['path']) == actual
            process = load(actual['path'])
            assert process['kind'] == 'ASR18_owned_CPU_process_completion_v1'
            assert process['actual_native_exit_code'] == 0 and process['actual_Popen_wait_code'] == 0
            assert process['parent_observed_wait'] is True and process['producer_dead_after_wait'] is True
            original_audio_child0s.append({'update_index': completed['update_index'], 'stage': stage, 'actual0': actual})
    assert len(original_audio_child0s) == 40
    predecessor = load(F / 'FastMSS_persistence_idle_CPU_source_review_manifest_v1.json')
    # Pin the old source files as immutable history; never edit old owner3/cache.
    predecessor_sources = predecessor['sources'] if 'sources' in predecessor else predecessor['bindings']
    for item in predecessor_sources:
        if isinstance(item, dict) and set(('path', 'bytes', 'sha256')) <= item.keys():
            assert pin(item['path']) == {key: item[key] for key in ('path', 'bytes', 'sha256')}
    old_template = load(F / 'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v5.json')
    template = deepcopy(old_template)
    template['kind'] = 'FastMSS_row_serial_CPU_preparation_contract_v6'
    replacements = {
        'common': SOURCE / 'fastmss_row_serial_cpu_v6.py',
        'sink': SOURCE / 'fastmss_phase_aware_feature_window_v4.py',
        'controller': PROJECT / 'work/research/prepare_FastMSS_phase_window_CPU_v6.py',
        'reference_worker': PROJECT / 'work/research/produce_FastMSS_independent_rows_CPU_v6.py',
        'feature_worker': PROJECT / 'work/research/produce_FastMSS_serial_features_CPU_v6.py',
        'stdlib_feature_validator': SOURCE / 'fastmss_stdlib_feature_validation_v1.py',
    }
    template['sources'].update({key: pin(path) for key, path in replacements.items()})
    template['controller_validation_runtime'] = {
        'NumPy_required': False, 'maximum_NPY_header_bytes': 1024,
        'maximum_NPY_payload_read_bytes': 65536,
        'validation_only_array_verifier': template['sources']['stdlib_feature_validator'],
        'native_numpy_reader_behavior_changed': False,
    }
    assert template['resource_review'] == old_template['resource_review']
    assert template['raw_lifetime_review'] == old_template['raw_lifetime_review']
    assert template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['production_enabled'] is False
    template_pin = write(F / 'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v6.json', template)
    report_path = F / 'stdlib_feature_validation_actual_01/report.json'
    guard_path = F / 'stdlib_feature_validation_guard_01/closure.json'
    report, guard = load(report_path), load(guard_path)
    assert report['passed'] is True and report['check_count'] == 51
    assert guard['actual_exit_code'] == 0 and guard['monitor_clean'] is True and guard['monitor_joined'] is True
    assert guard['observed']['samples'] >= 3 and guard['observed']['monitor_error'] is None
    actual_pin = write(F / 'FastMSS_stdlib_feature_validation_actual_CLOSED0_v1.json', {
        'kind': 'FastMSS_explicitly_synthetic_stdlib_NPY_actual_TOOL0_v1', 'owner': '/root/fastmss_integration',
        'actual_tool_chunk_id': 'e08c37', 'actual_tool_exit_code': 0,
        'native_pid': guard['pid'], 'native_exit_code': guard['actual_exit_code'],
        'native_private_peak_bytes': guard['observed']['peak_private_bytes'],
        'native_wall_seconds': guard['wall_seconds'], 'clean_joined_samples': guard['observed']['samples'],
        'checks': report['check_count'], 'synthetic_cases': report['synthetic_cases'],
        'report': pin(report_path), 'guard': pin(guard_path),
        'actual_original_feature_or_audio_bytes_read': False, 'retained_window_or_CACHE_changed': False,
        'actual_frontend_native_or_GPU_execution': False,
        'actual_original_controller_or_window_success_claimed': False, 'production_enabled': False,
    })
    interface_pin = write(F / 'FastMSS_stdlib_finalization_CPU6_source_interface_v1.json', {
        'kind': 'FastMSS_inactive_stdlib_finalization_CPU6_source_interface_v1',
        'production_enabled': False, 'repair_actual_original_runtime_pass_claimed': False,
        'supported_failure': failure_pin,
        'original_failure_classification': 'Actual controller1 finalization import error after all40 audio children0; not a memory stop and no GPU/Adam/draw.',
        'preserved_original_actual_audio_child0_proofs': original_audio_child0s,
        'retained_window': {'owner_source': 'Frozen sink3/CPU5, not adoptable under sink4/CPU6',
            'owner_sha256': 'c85d8604e3a184ecab3366da65efa41055e6766f91521ad601a6379b603c4c5b',
            'files': failure['retained_files'], 'bytes': failure['retained_bytes'],
            'inventory': failure['retained_feature_inventory'], 'accessed_or_changed_here': False,
            'controller_exit_reclassified': False, 'final_manifest_still_absent': True},
        'new_sources': {key: pin(path) for key, path in replacements.items()},
        'template': template_pin, 'actual_synthetic_controls': actual_pin,
        'API': {
            'validator': 'fastmss_stdlib_feature_validation_v1.validate_feature_arrays(entry, *, folder, owned_names, guard)',
            'validator_return': 'Validation-only small receipt with physical/valid/encoder frames and maximum chunk; no arrays, no child/process0/window0 or sampler commit.',
            'new_sink': 'fastmss_phase_aware_feature_window_v4; owner.source and closure.observer_source bind this NEW source. It rejects frozen owner3 because source differs.',
            'new_controller': 'work/research/prepare_FastMSS_phase_window_CPU_v6.py --bridge <bound-record-json> --output <fresh-E-folder> [--prior-boundary-verification <bound-record-json>]',
            'new_contract_kind': 'FastMSS_row_serial_CPU_preparation_contract_v6',
            'new_request_kind': 'FastMSS_row_serial_CPU_update_request_v6',
            'unchanged_bridge': template['sources']['bridge'],
            'unchanged_process': template['sources']['process_helper'],
        },
        'precise_delta': [
            'Sink4 imports one stdlib-only validator and adds _Store.validate_arrays. _validate_update and cleanup_window switch only their validation-only store.arrays calls to validate_arrays; no array is returned there.',
            'Final manifest additionally binds the exact validation-only helper source, no-NumPy requirement and64KiB read cap.',
            'Native _Store.arrays, _Store.save and _ReadCursor.take_row ASTs are unchanged, including real NumPy array loads/format/consume-once outputs.',
            'Common6 changes sink import, contract/request versions and explicit source map; adds verifier dependency. Decoder/source/PTS/resample/hash/raw-lifetime/geometry/resource algorithms unchanged.',
            'Controller6 and both worker6 sources change only their common-module import, retaining exact child ownership/exit/family guards and original numerical/frontend behavior.',
            'Template6 updates those exact source pins/kind plus no-NumPy validation metadata. All resource/raw-lifetime placeholders and caps remain unchanged; no runtime install or cap increase.',
        ],
        'validation_invariants': [
            'Exact owned resolved E path and complete file SHA; bounded NPY1/2 header1024B parsed using ast.literal_eval, unique exact header fields and tuple shape; no object/executable/Fortran or unknown-endian formats.',
            'Pinned little-endian float32[1,physical_frames,128] and bool[1,physical_frames], exact shape/dtype metadata and full payload EOF/file-byte-count/raw-array SHA. Read at most64KiB; no full array allocation.',
            'Every feature float32 exponent is finite; mask bytes are canonical0/1 with a nonempty true prefix followed only by padding. Encoder T=ceil(valid_mask_frames/8). Physical STFT frame is retained and never cropped.',
            'All masked feature values are numerically zero, accepting either signed zero exactly as NumPy.any does; nonzero subnormal padding is rejected.',
            'Existing source/context/full targets/clean+preparedPCM/policy/protected replay/cleaner/teacher/eval/closed-child/order/inventory/aggregate512MiB checks remain intact.',
        ],
        'limits_and_next_integration': [
            'Synthetic51 controls use independently struct-generated files only; they do not prove original-window finalization, CPU6 end-to-end, actual NumPy comparison, native update or recovery.',
            'Controller64MiB reserve/no-NumPy architecture, CPU2GB/global90/95critical/2GB headroom/E100GB/four threadcaps remain unchanged. Synthetic private22MB does not substitute for prospective actual whole-family admission.',
            'NEW caller/engine/runtime/assembler/config successors must import sink4 and CPU6 consistently, bind helper in complete dependency closure, and receive exact ROOT source/resource/config review before any original workload.',
            'Frozen owner3/session/CPUcontract/closures cannot be silently relabeled as sink4; no compatibility adoption API is provided. Retained artifacts stay preserved until ROOT separately reviews exact discard or supported reuse.',
            'After separately reviewed retirement, a fresh CPU6 window may be prepared only under unchanged actual admission; a source-bound comparison to the prior640feature/mask hashes can check deterministic byte identity before native, without reclassifying the old controller1.',
            'No original audio, feature arrays, shared CACHE, old sources, checkpoint, journal, lease or environment were modified by this source/fixture work.',
        ],
    })
    sources = list(replacements.values()) + [PROJECT / 'work/research/validate_FastMSS_stdlib_arrays_CPU_v1.py', Path(__file__)]
    manifest_pin = write(F / 'FastMSS_stdlib_finalization_CPU6_source_review_manifest_v1.json', {
        'kind': 'FastMSS_inactive_stdlib_finalization_CPU6_source_review_manifest_v1',
        'owner': '/root/fastmss_integration', 'production_enabled': False,
        'source': [pin(path) for path in sources], 'template': template_pin,
        'interface': interface_pin, 'actual_synthetic_controls': actual_pin,
        'supported_actual_failure': failure_pin,
        'preserved_predecessor': pin(F / 'FastMSS_persistence_idle_CPU_source_review_manifest_v1.json'),
        'actual_original_window_or_native_repair_pass_claimed': False,
    })
    print(json.dumps({'manifest': manifest_pin, 'interface': interface_pin, 'actual': actual_pin, 'template': template_pin}))


if __name__ == '__main__':
    main()
