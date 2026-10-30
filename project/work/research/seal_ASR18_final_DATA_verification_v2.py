"""Publish seven-part DATA evidence after this owner's observed verifier TOOL0.

The operator supplies the actually returned tool chunk, never a predicted exit.
This metadata sealer does not execute a DATA generator, decode or native caller.
"""
import argparse
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from fastmss_phase_aware_plan_bridge_v4 import validate_DATA, read_bound
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual

A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
B = A.parent / 'BPE_weak_then_clean_v1'
ENTRY_SHA = 'bf7c43e27a59f7a63ddd4821c4dc13413084dfcd72707b4631fd6d6b5f7bac3c'
VERIFIER_SHA = 'f8536cb791e1f2e66a5a4f7565c3b569bda067e01f77f296f908ed7d12e050ee'


def published(wrapper_path):
    wrapper = read_json(wrapper_path)
    require(wrapper['kind'] == 'ASR18_source_bound_actual_tool_exit_v1'
            and wrapper['owner'] == '/root/cpu_data_preparation'
            and type(wrapper['actual_exit_code']) is int
            and wrapper['actual_exit_code'] == 0
            and type(wrapper['actual_tool_chunk_id']) is str
            and wrapper['actual_tool_chunk_id'], 'Genuine owned DATA TOOL0 required')
    pin = wrapper['published_marker']
    require(set(pin) in ({'path', 'bytes', 'sha256'}, {'path', 'bytes', 'sha256', 'mtime_ns'}),
            'Exact three/four-field published marker required')
    require(record(pin['path']) == {k: pin[k] for k in ('path', 'bytes', 'sha256')},
            'Published metadata binding changed')
    if 'mtime_ns' in pin:
        require(type(pin['mtime_ns']) is int and Path(pin['path']).stat().st_mtime_ns == pin['mtime_ns'],
                'Published metadata signature changed')
    return pin


def main(folder, guard_folder, chunk, output):
    require(folder.resolve().is_relative_to(A.resolve())
            and guard_folder.resolve().is_relative_to(A.resolve())
            and output.resolve().parent == A.resolve() and not output.exists(),
            'New immutable E-only evidence; preserve history')
    require(type(chunk) is str and re.fullmatch(r'[0-9a-f]{6,64}', chunk),
            'Supply the actual observed successful verifier tool chunk')
    result = read_json(folder / 'report.json')
    entry = record(P / 'work/research/run_ASR18_final_DATA_verification_CPU_v1.py')
    closed = read_json(guard_folder / 'closure.json')
    require(entry['sha256'] == ENTRY_SHA
            and result['independent_verification_entrypoint'] == entry
            and result['source_verifier']['sha256'] == VERIFIER_SHA
            and result['passed'] is True and result['complete'] is True,
            'Exact accepted thin verifier result required')
    reported_pid = result['prospective_metadata_admission']['fresh_sample']['worker_private_memory']['pid']
    require(type(reported_pid) is int and type(closed['pid']) is int
            and reported_pid == closed['pid'],
            'Verifier admission PID must match this exact clean guard closure')
    raw_actual = B / 'ASR18_final_data_population_for_bounded_preparation_owned_tool_actual_CLOSED0_v1.json'
    two_actual = B / 'ASR18_final_two_pass_population_metadata_with_declared_geometry_owned_tool_actual_CLOSED0_v1.json'
    raw_pin, two_pin = published(raw_actual), published(two_actual)
    two = read_bound(two_pin)
    require(two['declared_geometry_bound'] is True,
            'The preserved denominator-only receipt cannot supply final geometry readiness')
    actual_pin = actual(folder, guard_folder, chunk)
    evidence = dict(population=raw_pin, population_actual0=record(raw_actual),
        two_pass_metadata=two_pin, two_pass_actual0=record(two_actual),
        validated=record(folder / 'report.json'), validated_actual0=actual_pin,
        verifier_source=record(S / 'bpe_final_DATA_population_v1.py'))
    checked = validate_DATA(evidence, lambda: None)
    require(checked['actual_optimizer_updates'] == 0 and checked['production_enabled'] is False,
            'DATA metadata cannot activate native work')
    write_json(output, evidence)
    print(json.dumps({'evidence': record(output), 'retained_rows': checked['retained_rows'],
        'total_target_tokens': checked['total_target_tokens'], 'population_mean_U': checked['population_mean_U'],
        'native_or_frontend_admission_claimed': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, type=Path)
    parser.add_argument('--guard-folder', required=True, type=Path)
    parser.add_argument('--actual-tool-chunk', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.folder, args.guard_folder, args.actual_tool_chunk, args.output)
