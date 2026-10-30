"""Build an unapproved CPU resource proposal from genuinely closed full scalars.

No waveform, numerical runtime, frontend, cache or native process is started.
Whole historical frontend peaks remain whole peaks, never isolated DLL costs.
"""
import argparse
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from fastmss_phase_aware_plan_bridge_v4 import actual0, read_bound
from fastmss_raw_lifetime_v1 import raw_plan, CAPACITY

A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
B = A.parent / 'BPE_weak_then_clean_v1'
F = A.parent / 'FastMSS_integration_v1'
MiB = 1024 ** 2
RESOURCE_SOURCE_SHA = '12ef7593c5d8a12768afb173c4880528167d9cd0da1796582750397acaa53ca4'
RAW_SOURCE_SHA = '8179bdcfa5d6f184fcc6dfd30a90369255f41fd4463c24068674acea53a4bf1f'


def closed_metadata(path, expected_source=None):
    proof_pin = record(path)
    proof = read_json(path)
    published = proof['published_marker']
    actual0(proof_pin, published, lambda: None, owner='/root/cpu_data_preparation')
    if expected_source is not None:
        require(Path(proof['source_path']).resolve() == expected_source.resolve()
                and proof['source_SHA256'] == record(expected_source)['sha256'] == RESOURCE_SOURCE_SHA,
                'Exact genuine scalar resource producer required')
    return read_bound(published), published, proof_pin


def main(output):
    require(output.resolve().parent == A.resolve() and not output.exists(),
            'New immutable E-only proposal; preserve prior policies')
    require(record(S / 'fastmss_raw_lifetime_v1.py')['sha256'] == RAW_SOURCE_SHA,
            'Frozen raw-lifetime formula changed')
    resource, resource_pin, resource_actual = closed_metadata(
        B / 'ASR18_final_declared_CPU_resource_envelope_owned_tool_actual_CLOSED0_v1.json',
        P / 'work/research/produce_ASR18_declared_CPU_resource_envelope_v1.py')
    envelope, envelope_pin, envelope_actual = closed_metadata(
        B / 'ASR18_final_declared_training_geometry_owned_tool_actual_CLOSED0_v1.json')
    require(resource['kind'] == 'ASR18_declared_CPU_array_payload_resource_envelope_v1'
            and resource['complete_declared_scalar_population'] is True
            and resource['actual_full_corpus_decode_or_CPU_allocation_proof_claimed'] is False
            and envelope['complete'] is True
            and resource['retained_rows'] == envelope['population']['retained_rows']
            and resource['bindings']['declared_envelope'] == envelope_pin
            and resource['bindings']['declared_envelope_actual_closed'] == {
                **envelope_actual, 'mtime_ns': Path(envelope_actual['path']).stat().st_mtime_ns},
            'Genuine full scalar resource and envelope identities must agree')
    cases = []
    for component, sr, channels in (('alania_yodas3', 48000, 8), ('kiraat', 24000, 1)):
        group = envelope['groups'][component]
        n = group['MAX_native_samples']
        require(type(n) is int and n > 0, 'Exact per-component full maximum required')
        # Each expression is monotone inside weak<=120s and protected branches;
        # include both endpoints because augmentation bypass changes the formula.
        for phase, role, native, name in (
            ('weak_label_continued_training', 'student_tr', n, 'full_component_maximum'),
            ('weak_label_continued_training', 'student_tr', min(n, 120 * sr), 'weak_eligible_endpoint'),
            ('cleaner_data_fine_tuning', 'student_tr', n, 'protected_component_maximum')):
            clean = (native * 16000 + sr - 1) // sr
            cases.append(dict(component=component, basis=name, plan=raw_plan(
                native_samples=native, native_rate=sr, clean_samples=clean, phase=phase,
                role=role, channels_bound=channels, frame_samples_bound=16384)))
        require((n * 16000 + sr - 1) // sr == group['MAX_clean16k_samples'],
                'Monotone fixed-rate component extrema disagree')
    # Every original replay metadata row has encoder T<=326. T=ceil((N//160)/8)
    # gives N<=8*T*160+159, including physical frontend frame+1.
    replay_samples = 8 * 326 * 160 + 159
    cases.append(dict(component='original_replay', basis='whole1914_metadata_T326_upper',
        plan=raw_plan(native_samples=replay_samples, native_rate=16000, clean_samples=replay_samples,
            phase='weak_label_continued_training', role='replay', channels_bound=1, frame_samples_bound=16384)))
    raw_max = max(item['plan']['raw_peak_bytes'] for item in cases)
    work = {key: max(item['plan']['workspace_incremental_envelopes'][key] for item in cases)
            for key in ('decoder', 'weak_transform_FFT', 'original_frontend_spectral')}
    reserve = 768 * MiB
    round_workspace = lambda value: ((value + 64 * MiB - 1) // (64 * MiB) + 1) * 64 * MiB
    raw_review = dict(kind='FastMSS_preallocation_raw_lifetime_review_v1', accepted=False,
        raw_capacity_bytes=CAPACITY, maximum_jobs=2, compressed_media_per_job_bytes=32 * MiB,
        maximum_runtime_channels=8, maximum_decoder_frame_samples=16384,
        decoder_workspace_allowance_bytes=round_workspace(work['decoder']),
        transform_workspace_allowance_bytes=round_workspace(work['weak_transform_FFT']),
        frontend_workspace_allowance_bytes=round_workspace(work['original_frontend_spectral']),
        numerical_runtime_reserve_bytes=reserve)
    fixture_path = F / 'feature_window_CPU_fixture_02/validation.json'
    fixture = read_json(fixture_path)
    require(fixture['passed'] is True and fixture['two_complete_full_hard_audio_frontends_byte_exact'] is True,
            'Existing whole frontend baseline proof required')
    peaks = [v['observed']['peak_private_bytes'] for v in fixture['child_resources']
             if v['actual_exit_code'] == 0 and v['pid'] in (24508, 34308)]
    require(len(peaks) == 2, 'Both actual hard-wave whole frontend controls required')
    baseline = max(peaks)
    leaf = 1_800_000_000
    controller, shim, outer = 64 * MiB, 32 * MiB, 32 * MiB
    prospective = []
    for item in cases:
        plan = item['plan']
        for stage in ('reference', 'frontend'):
            workspace = plan['workspace_incremental_envelopes']['decoder'] if stage == 'reference' else max(plan['workspace_incremental_envelopes'].values())
            increment = plan['raw_peak_bytes'] + plan['compressed_media_cap_bytes'] + workspace + reserve
            prospective.append(dict(component=item['component'], basis=item['basis'], stage=stage,
                raw_bytes=plan['raw_peak_bytes'], compressed_cap_bytes=plan['compressed_media_cap_bytes'],
                actual_formula_workspace_bytes=workspace, numerical_runtime_reserve_bytes=reserve,
                per_row_incremental_forecast_bytes=increment,
                maximum_current_private_before_this_row_bytes=leaf - increment,
                whole_historical_frontend_baseline_plus_increment_bytes=baseline + increment))
    inner = leaf + controller + shim
    review = dict(passed=False, scope='CPU_preparation_only', native_GPU_fit_claimed=False,
        maximum_simultaneous_children=1, controller_private_reserve_bytes=controller,
        shim_private_reserve_bytes=shim, outer_controller_launcher_reserve_bytes=outer,
        reference_incremental_bytes=leaf, frontend_incremental_bytes=leaf,
        reference_maximum_private_bytes=leaf, frontend_maximum_private_bytes=leaf,
        whole_owned_private_allowance_bytes=inner)
    require(inner + outer < 2_000_000_000, 'Whole reviewed CPU family must stay below unchanged2GB')
    value = dict(kind='ASR18_CPU_resource_proposal_from_genuine_full_declared_scalars_v1',
        accepted=False, requires_ROOT_source_bound_resource_review=True, production_enabled=False,
        resource_review=review, raw_lifetime_review=raw_review,
        bindings=dict(declared_resource=resource_pin, declared_resource_actual0=resource_actual,
            declared_envelope=envelope_pin, declared_envelope_actual0=envelope_actual,
            raw_formula=record(S / 'fastmss_raw_lifetime_v1.py'), proposal_source=record(__file__),
            historical_full_frontend_baselines=record(fixture_path),
            original_replay_geometry=record(A / 'whole_original_replay_labels_actual_readiness_v1.json')),
        full_declared_rows=resource['retained_rows'], scalar_cases=cases,
        conservative_raw_peak_bytes=raw_max, raw128MiB_can_cover_each_serial_case=raw_max <= CAPACITY,
        workspace_formula_maxima=work, row_admission_arithmetic=prospective,
        historical_whole_frontend_private_peaks_bytes=peaks,
        largest_baseline_plus_row_increment_bytes=max(v['whole_historical_frontend_baseline_plus_increment_bytes'] for v in prospective),
        CPU_inner_whole_private_allowance_bytes=inner,
        CPU_outer_shim_allowance_bytes=outer, CPU_total_whole_family_allowance_bytes=inner + outer,
        native_private_proposal_bytes=5_560_813_440, entire_GPU_worker_proposal_bytes=5_910_813_440,
        library_cost_isolated_from_baselines=False, runtime_current_private_must_fit_per_row=True,
        exact_Alania_original_channel_coverage_known=False,
        conditional_original_Alania_channel_cap_requires_actual_first_frame_check=8,
        positive_original_audio_frontend_or_native_execution_performed=False,
        full_corpus_CPU_private_fit_claimed=False, GPU_native_forecast_approved=False,
        sustained_current_caller_resource_success_claimed=False,
        actual_buffers_must_release_before_next_row_and_runtime_lengths_must_match=True)
    write_json(output, value)
    print(json.dumps({'proposal': record(output), 'raw_max_bytes': raw_max,
        'workspace_formula_maxima': work, 'CPU_whole_family_bytes': inner + outer,
        'baseline_plus_largest_row_increment': value['largest_baseline_plus_row_increment_bytes'],
        'approved': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
