"""Compact handoff after the observed v2 renderer tool exit; no audio rendering."""
import hashlib
import os
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
    guard = DataIOGuard('reverb_listening_v2_compact_handoff', heavy=False, maximum_bytes=40_000_000)
    guard.wait()
    pins = MetadataIO(guard)
    area = ROOT / 'analysis/FastMSS_real_recorded_noise_listening_v1'
    manifest = pins.json(area / 'sample_manifest_v2.json')
    qa = pins.json(area / 'verification_v2.json')
    assert qa['passed'] and qa['audio_examples'] == 30 and qa['clipped_samples'] == 0
    assert manifest['PID'] == qa['PID'] == 42656
    assert manifest['generator_source'] == pins.binding(PROJECT / 'work/research/build_ASR18_reverb_noise_listening_v2.py')
    assert manifest['generator_source']['sha256'] == 'bd85eaa8a4046cc3f1716604bfe4fb4707acf2ebe63e988b9ea945f7596d38cc'
    assert manifest['v1_page'] == pins.binding(area / 'recorded-noise-turkish-listening.html')
    assert not psutil.pid_exists(42656), 'The actual completed renderer PID must be absent'
    server = psutil.Process(39176)
    assert any('serve_ASR18_recorded_noise_listening_v1.py' in value for value in server.cmdline())
    url = 'http://127.0.0.1:8890/recorded-noise-reverb-turkish-listening-v2.html'
    digest, count = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=20) as response:
        assert response.status == 200
        while True:
            guard.wait()
            block = response.read(1 << 20)
            if not block:
                break
            digest.update(block)
            count += len(block)
            guard.pace(len(block))
    page = pins.binding(area / 'recorded-noise-reverb-turkish-listening-v2.html')
    assert count == page['bytes'] and digest.hexdigest() == page['sha256']
    actual = dict(kind='FastMSS_reverb_listening_owned_renderer_actual_CLOSED0_v2',
        owner='/root/cpu_data_preparation', at_utc=now(), actual_exit_code=0,
        native_PID=42656, owned_session=None, actual_tool_chunk_id='ed0628',
        actual_tool_wall_seconds=5.2045779, actual_tool_exit_observed=True,
        shell_explicitly_propagated_renderer_exit=True,
        source=manifest['generator_source'], published_manifest=pins.binding(area / 'sample_manifest_v2.json'),
        published_verification=pins.binding(area / 'verification_v2.json'),
        scope='The observed direct tool completion included fresh resource forecast then renderer; no fabricated inner-tool or peak claim')
    write_json(area / 'renderer_actual_CLOSED0_v2.json', actual)
    handoff = dict(kind='FastMSS_real_noise_and_simulated_reverb_current_handoff_v2',
        owner='/root/cpu_data_preparation', at_utc=now(), PID=os.getpid(),
        URL=url, page=page, sample_manifest=pins.binding(area / 'sample_manifest_v2.json'),
        verification=pins.binding(area / 'verification_v2.json'),
        actual_renderer=pins.binding(area / 'renderer_actual_CLOSED0_v2.json'),
        v1_handoff=pins.binding(area / 'current_handoff_v1.json'),
        real_noise_registry=manifest['registry'], RIR_registry=manifest['RIR_registry'],
        reverb_source=manifest['shared_reverb_source'], real_noise_mixer_source=manifest['real_noise_mixer_source'],
        audio_examples=30, TRAIN_utterances=3, new_reverb_only=6, new_reverb_plus_recorded_noise_10dB=3,
        old_encoded_examples_reused_byte_exact=21, v1_page_and_sources_immutable=True,
        original_supplied_transcripts_unchanged=True, common_utterance_gains_unchanged=True,
        gains={item['id']: item['common_gain'] for item in manifest['utterances']},
        simulated_room_count=1, simulated_position_count=2, measured_RT60_claimed=False,
        alignment='strongest-tap; physical direct arrival not established',
        output='original duration; post-end convolution tail omitted; clip RMS restored before common gain',
        operation_order='simulated reverb then real recorded noise; combo10dB relative to actual reverberant speech',
        SNR='wholeclip RMS including pauses, not VAD/active-speech SNR',
        HTTP_verification=dict(status=200, bytes=count, SHA256=digest.hexdigest(), matched_local_page=True),
        server=dict(owned_session=13082, PID=39176, unchanged=True, loopback_only=True),
        clipped_samples=0, maximum_SNR_error_dB=qa['maximum_post_encoding_SNR_error_dB'],
        maximum_decoded_peak=qa['maximum_decoded_peak'], production_enabled=False,
        ASR_or_model_or_GPU_used=False, quality_gain_claimed=False,
        browser_verification='ROOT owns playback/filter verification and delivery',
        no_more_samples_or_render_jobs_requested=True)
    pins.unchanged()
    write_json(area / 'current_handoff_v2.json', handoff)
    print(dict(handoff=pins.binding(area / 'current_handoff_v2.json'), actual=pins.binding(area / 'renderer_actual_CLOSED0_v2.json'),
        HTTP_status=200, bytes=count, URL=url, actual_handoff_tool_exit_still_required=True), flush=True)


if __name__ == '__main__':
    main()
