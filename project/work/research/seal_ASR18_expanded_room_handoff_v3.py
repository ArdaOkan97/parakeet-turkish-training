"""Compact final v3 handoff and byte-exact loopback HTTP check; no audio work."""
import hashlib
import sys
import urllib.request
from pathlib import Path
from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


def main():
    import psutil
    guard = DataIOGuard('expanded_room_v3_compact_handoff', heavy=False, maximum_bytes=80_000_000)
    guard.wait()
    pins = MetadataIO(guard)
    area = ROOT / 'analysis/FastMSS_real_recorded_noise_listening_v1'
    manifest = pins.json(area / 'sample_manifest_v3.json')
    qa = pins.json(area / 'verification_v3.json')
    assert qa['passed'] and qa['audio_examples'] == 51 and qa['clipped_samples'] == 0
    assert manifest['PID'] == qa['PID'] == 28796
    assert manifest['generator_source'] == pins.binding(PROJECT / 'work/research/build_ASR18_expanded_room_listening_v3.py')
    assert manifest['generator_source']['sha256'] == '192ed6438c3768ea88674689c2185625bc11369541e05fa272a25875d074948e'
    for binding in manifest['previous_pages_preserved'].values():
        assert pins.binding(binding['path']) == binding
    assert not psutil.pid_exists(28796), 'Completed renderer PID must be absent'
    assert any('serve_ASR18_recorded_noise_listening_v1.py' in v for v in psutil.Process(39176).cmdline())
    url = 'http://127.0.0.1:8890/expanded-room-reverb-turkish-listening-v3.html'
    count, digest = 0, hashlib.sha256()
    with urllib.request.urlopen(url, timeout=20) as response:
        assert response.status == 200
        while True:
            guard.wait()
            part = response.read(1 << 20)
            if not part:
                break
            count += len(part)
            digest.update(part)
            guard.pace(len(part))
    page = pins.binding(area / 'expanded-room-reverb-turkish-listening-v3.html')
    assert page == qa['page'] and count == page['bytes'] and digest.hexdigest() == page['sha256']
    actual = dict(kind='FastMSS_expanded_room_owned_renderer_actual_CLOSED0_v3',
        owner='/root/cpu_data_preparation', at_utc=now(), actual_exit_code=0,
        native_PID=28796, owned_session=None, actual_tool_chunk_id='dea28d',
        actual_tool_wall_seconds=7.8583383, actual_tool_exit_observed=True,
        shell_explicitly_propagated_renderer_exit=True, source=manifest['generator_source'],
        published_manifest=pins.binding(area / 'sample_manifest_v3.json'),
        published_verification=pins.binding(area / 'verification_v3.json'),
        scope='Direct observed tool completion included fresh resource forecast then renderer; no fabricated inner tool handle')
    write_json(area / 'renderer_actual_CLOSED0_v3.json', actual)
    wet = [s for s in manifest['samples'] if not s['is_clean']]
    handoff = dict(kind='FastMSS_expanded_simulated_room_current_handoff_v3',
        owner='/root/cpu_data_preparation', at_utc=now(), URL=url, page=page,
        manifest=pins.binding(area / 'sample_manifest_v3.json'), verification=pins.binding(area / 'verification_v3.json'),
        renderer_actual=pins.binding(area / 'renderer_actual_CLOSED0_v3.json'),
        previous_handoff=pins.binding(area / 'current_handoff_v2.json'),
        helper=manifest['helper_source'], room_bank_registry=manifest['room_bank_bindings']['registry'],
        TRAIN_utterances=3, simulated_rooms=8, conditions_per_utterance=16,
        clean_references=3, wet_examples=48, audio_examples=51,
        supplied_source_transcripts_unchanged=True, original_input_samples_never_cropped=True,
        all51_exports_redecoded=True, all_output_lengths_exact=True, zero_clipped_samples=True,
        full_finite_response_tail_seconds=[qa['min_wet_tail_duration_s'], qa['max_tail_duration_s']],
        estimated_T60_from30dB_fit_seconds=[min(s['estimated_T60_s'] for s in wet), max(s['estimated_T60_s'] for s in wet)],
        configuration_targets_separate=True, physical_room_recording_claimed=False,
        method='experimental hybrid ISM order3 + finite ray simulation',
        alignment='Only the known40sample simulator fractional-filter delay removed; geometric propagation delay retained',
        levels='Reverberant RMS matched over original input-duration portion; one common final gain per utterance across clean and all16wet conditions',
        common_gains={u['id']: u['common_gain'] for u in manifest['utterances']},
        maximum_post_encoding_speech_window_RMS_error_dB=qa['maximum_post_encoding_speech_window_RMS_error_dB'],
        maximum_decoded_peak=qa['maximum_decoded_peak'],
        observed_private_peak_at_work_boundaries_bytes=manifest['observed_peak_private_bytes'],
        observed_commit_peak_at_work_boundaries=manifest['observed_peak_system_commit_ratio'],
        HTTP=dict(status=200, bytes=count, SHA256=digest.hexdigest(), byte_exact=True),
        server=dict(owned_session=13082, PID=39176, unchanged=True, loopback_only=True),
        v1_and_v2_immutable=True, production_enabled=False, Gaussian_or_environment_noise_added=False,
        ASR_model_or_GPU_used=False, quality_gain_claimed=False,
        browser_verification='ROOT owns playback/filter check and delivery', no_more_samples_or_jobs_requested=True)
    pins.unchanged()
    write_json(area / 'current_handoff_v3.json', handoff)
    print(dict(handoff=pins.binding(area / 'current_handoff_v3.json'), actual=pins.binding(area / 'renderer_actual_CLOSED0_v3.json'),
        HTTP=200, URL=url, actual_handoff_tool_exit_still_required=True), flush=True)


if __name__ == '__main__':
    main()
