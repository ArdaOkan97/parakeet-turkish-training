"""Finite51-example full-tail room preview, with supplied TRAIN text unchanged."""
import base64
import hashlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT
from cpu_preparation_guard_v2 import memory, now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO
from ASR18_expanded_reverb_page_v3 import document

PROJECT = Path(__file__).resolve().parents[2]
SOURCES = PROJECT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(SOURCES))
from atomic_io_retry_v1 import write_json
import fastmss_room_bank_preview_v1 as rooms

AREA = ROOT / 'analysis/FastMSS_real_recorded_noise_listening_v1'
HELPER_SHA256 = '730a58a81776041f46f17adaaa4222d269b29dfa61a0558433662b6c3858b66d'


def load_assets(guard, pins):
    """Bind the completed finite bank without regenerating or changing its files."""
    path = AREA / 'expanded_room_bank_v1/room_RIR_registry_v1.json'
    bound = pins.binding(path)
    assert bound['sha256'] == '0c2b0b310ee4dcf4c6787c403431a23976ac83d012c2a5f842cd1417deec5601'
    bank = pins.json(path)
    assert bank['room_count'] == 8 and bank['condition_count'] == 16 and bank['production_enabled'] is False
    bindings = dict(registry=bound)
    for name, stage in bank['actual_generation'].items():
        assert stage['actual_exit_code'] == 0 and stage['actual_tool_chunk_id']
        for kind in ('guard_closure', 'original_registry'):
            actual = pins.binding(stage[kind]['path'])
            assert {k: actual[k] for k in ('path', 'bytes', 'sha256')} == stage[kind]
            bindings[name + '_' + kind] = actual
        footer = pins.json(stage['guard_closure']['path'])
        assert footer['actual_exit_code'] == 0 and footer['pid'] == stage['native_PID']
        assert footer['monitor_joined'] and footer['monitor_clean'] and footer['observed']['monitor_error'] is None
    original_assets = rooms.load_room_bank(path, guard=guard.wait)
    room_ids = list(dict.fromkeys(a['room_id'] for a in original_assets))
    assets = []
    for raw in original_assets:
        dims = raw['room_dimensions_m']
        label = 'Room ' + str(room_ids.index(raw['room_id']) + 1) + ' · ' + ' × '.join(f'{value:g}' for value in dims) + ' m'
        measured = raw['measured_decay']
        fit = measured['T30'] if measured['T30']['usable'] else measured['T20']
        estimate = fit['RT60_extrapolated_seconds'] if fit['usable'] else None
        assert estimate is None or math.isfinite(estimate) and estimate > 0
        assets.append(dict(id=raw['id'], room_id=raw['room_id'], room_label=label,
            position=raw['position'], position_label=raw['position'].capitalize() + ' · ' + f"{raw['source_microphone_distance_m']:.2f} m",
            mode_label='Hybrid ISM + ray simulation', helper_asset=raw,
            RT20_extrapolated_s=measured['T20']['RT60_extrapolated_seconds'],
            RT30_extrapolated_s=measured['T30']['RT60_extrapolated_seconds'],
            estimated_T60_s=estimate, decay_fit_basis='30 dB fit' if measured['T30']['usable'] else '20 dB fit',
            target_RT60_s=raw['target_RT60_seconds'],
            source_receiver_distance_m=raw['source_microphone_distance_m'],
            decay_measurement=measured, audio=raw['audio']))
    bank = dict(bank, alignment_description='Only the known simulator global fractional-filter delay of 40 samples (2.5 ms) is removed. Physical geometric source-to-microphone propagation delay is retained; this is not strongest-tap alignment.')
    return bank, assets, bindings


def main():
    import numpy as np
    import soundfile as sf
    guard = DataIOGuard('expanded_room_listening_v3', heavy=False, maximum_bytes=300_000_000)
    guard.wait()
    pins = MetadataIO(guard)
    helper = pins.binding(SOURCES / 'fastmss_room_bank_preview_v1.py')
    assert helper['sha256'] == HELPER_SHA256
    bank, assets, bank_bindings = load_assets(guard, pins)
    assert len(assets) == 16 and len({a['room_id'] for a in assets}) == 8
    old = pins.json(AREA / 'sample_manifest_v1.json')
    selection = pins.json(AREA / 'selected_TRAIN_utterances_v1.json')
    assert selection['DEV_or_TEST_used'] is False and selection['recognizer_used'] is False
    assert len(selection['clips']) == 3
    target = AREA / 'expanded-room-reverb-turkish-listening-v3.html'
    exports = AREA / 'samples_v3'
    assert not target.exists() and not exports.exists()
    exports.mkdir()
    prior = {version: pins.binding(AREA / name) for version, name in (
        ('v1', 'recorded-noise-turkish-listening.html'),
        ('v2', 'recorded-noise-reverb-turkish-listening-v2.html'))}
    samples, utterances, payloads = [], [], {}
    peak_private = peak_commit = 0

    def observe():
        nonlocal peak_private, peak_commit
        guard.wait()
        peak_private = max(peak_private, guard.process.memory_info().private)
        peak_commit = max(peak_commit, memory()['commit_ratio'])

    def rms(wave):
        values = wave.astype(np.float64)
        return math.sqrt(float(np.dot(values, values)) / len(values))

    def export(key, audio, metadata, expected_rms):
        observe()
        assert audio.dtype == np.float32 and audio.ndim == 1 and np.isfinite(audio).all()
        assert 0 < len(audio) <= 32 * 16000 and float(np.abs(audio).max()) <= .950001
        stream = io.BytesIO()
        sf.write(stream, audio, 16000, format='FLAC', subtype='PCM_16')
        encoded = stream.getvalue()
        assert len(encoded) < 3_000_000
        decoded, sr = sf.read(io.BytesIO(encoded), dtype='float32')
        assert sr == 16000 and decoded.ndim == 1 and len(decoded) == len(audio) and np.isfinite(decoded).all()
        peak = float(np.abs(decoded).max())
        assert peak < 1 and np.count_nonzero(np.abs(decoded) >= 1) == 0
        original_n = metadata['input_samples']
        error_db = 20 * math.log10(rms(decoded[:original_n]) / expected_rms)
        assert abs(error_db) < .01
        tail = decoded[original_n:]
        path = exports / (key + '.flac')
        path.write_bytes(encoded)
        guard.pace(len(encoded))
        item = dict(metadata, id=key, audio=pins.binding(path), sample_rate=16000,
            output_samples=len(audio), output_duration_s=len(audio)/16000,
            tail_samples=len(tail), tail_duration_s=len(tail)/16000,
            tail_RMS=rms(tail) if len(tail) else 0,
            tail_peak=float(np.abs(tail).max()) if len(tail) else 0,
            decoded_peak=peak, decoded_clipped_samples=0, decoded_all_finite=True,
            post_encoding_speech_window_RMS_error_dB=error_db,
            original_input_length_preserved=True, full_finite_response_tail_kept=not metadata['is_clean'],
            encoded_format='mono16k FLAC / PCM16; listening export only',
            original_supplied_transcript_unchanged=True)
        samples.append(item)
        payloads[key] = base64.b64encode(encoded).decode('ascii')
        observe()

    for clip in selection['clips']:
        observe()
        original = next(u for u in old['utterances'] if u['id'] == clip['id'])
        row = clip['source_row']
        assert row['dataset_split'] == 'train' and row['language'] == 'tr'
        source = pins.binding(row['audio_filepath'], maximum=32_000_000)
        assert source == original['source_audio'] and row['text'] == original['original_supplied_transcript']
        with sf.SoundFile(source['path']) as f:
            assert f.samplerate == 16000 and f.channels == 1 and f.frames == original['samples']
            assert 8*16000 <= f.frames <= 20*16000
            clean = f.read(dtype='float32')
            guard.pace(clean.nbytes)
        assert np.isfinite(clean).all()
        prepared = [rooms.prepare_room_reverb(clean, a['helper_asset'], guard=guard.wait) for a in assets]
        assert len(prepared) == 16
        for wet, asset in zip(prepared, assets):
            raw = asset['helper_asset']
            exact_output = len(clean) + raw['samples'] - raw['global_fractional_filter_delay_samples'] - 1
            assert len(wet['audio']) == exact_output and exact_output > len(clean)
            assert wet['receipt']['input_samples'] == len(clean) and wet['receipt']['output_samples'] == exact_output
        peaks = [float(np.abs(wet['audio']).max()) for wet in prepared]
        gain = rooms.common_gain(clean, peaks, guard=guard.wait, ceiling=.95)
        assert 0 < gain <= 1
        expected_rms = rms(clean) * gain
        meta = dict(utterance_id=clip['id'], input_samples=len(clean), input_duration_s=len(clean)/16000,
            common_utterance_gain=gain, is_clean=True, room_id=None, room_label=None,
            position=None, position_label=None, strength=None, mode_label=None,
            RT20_extrapolated_s=None, RT30_extrapolated_s=None, estimated_T60_s=None, decay_fit_basis=None)
        export(clip['id'] + '_clean', np.ascontiguousarray(clean * gain, dtype=np.float32), meta, expected_rms)
        for asset, wet in zip(assets, prepared):
            rendered = rooms.render_room_reverb(wet, gain, guard=guard.wait)
            assert len(rendered['audio']) == len(wet['audio'])
            estimate = asset['estimated_T60_s']
            strength = 'unknown' if estimate is None else 'short' if estimate <= .4 else 'medium' if estimate <= .9 else 'long'
            meta = dict(utterance_id=clip['id'], input_samples=len(clean), input_duration_s=len(clean)/16000,
                common_utterance_gain=gain, is_clean=False,
                **{key: asset[key] for key in ('room_id', 'room_label', 'position', 'position_label',
                    'mode_label', 'RT20_extrapolated_s', 'RT30_extrapolated_s', 'estimated_T60_s', 'decay_fit_basis', 'target_RT60_s',
                    'source_receiver_distance_m', 'decay_measurement')},
                strength=strength, RIR_asset_id=asset['id'], RIR_audio=asset['audio'],
                preparation_and_render_receipt=rendered['receipt'])
            export(clip['id'] + '_' + asset['id'], rendered['audio'], meta, expected_rms)
            del rendered
        utterances.append(dict(original, common_gain=gain, v1_common_gain=original['common_gain'],
            level_basis='RMS over the original input-duration portion, before one shared final gain',
            all_input_samples_retained=True, response_tail='full finite aligned RIR convolution tail'))
        del clean, prepared
    assert len(samples) == len({sample['id'] for sample in samples}) == 51
    assert sum(s['is_clean'] for s in samples) == 3
    manifest = dict(kind='FastMSS_expanded_simulated_room_listening_samples_v3', schema_version=3,
        owner='/root/cpu_data_preparation', created_at_utc=now(), PID=os.getpid(),
        generator_source=pins.binding(__file__), page_layout_source=pins.binding(PROJECT / 'work/research/ASR18_expanded_reverb_page_v3.py'),
        helper_source=helper, room_bank_bindings=bank_bindings, room_bank=bank,
        source_selection=pins.binding(AREA / 'selected_TRAIN_utterances_v1.json'),
        previous_pages_preserved=prior, original_v1_manifest=pins.binding(AREA / 'sample_manifest_v1.json'),
        utterances=utterances, samples=samples, audio_examples=51, simulated_rooms=8, positions_per_room=2,
        input_speech_cropped=False, full_finite_response_tails_kept=True,
        RMS_match_window='original input sample count only; response tail excluded from RMS target',
        one_common_headroom_gain_per_utterance=True, clipping_or_limiter=False,
        actual_recorded_room_claimed=False, target_and_extrapolated_decay_distinct=True,
        environment_noise_added=False, Gaussian_bed_added=False, production_enabled=False,
        supplied_transcripts_unchanged=True, DEV_or_TEST_used=False, ASR_model_or_GPU_used=False,
        runtime={name: importlib.metadata.version(name) for name in ('numpy', 'soundfile', 'scipy')},
        observed_peak_private_bytes=peak_private, observed_peak_system_commit_ratio=peak_commit,
        observation_scope='sampled at bounded work/export boundaries; unchanged continuous watchdog remains active')
    write_json(AREA / 'sample_manifest_v3.json', manifest)
    text = document(utterances, samples, payloads, bank)
    assert text.count('data:audio/flac;base64,') == 51 and text.count('<audio ') == 51
    target.write_text(text, encoding='utf-8')
    assert target.stat().st_size < 50_000_000
    pins.unchanged()
    qa = dict(kind='FastMSS_expanded_room_listening_verification_v3', at_utc=now(), PID=os.getpid(),
        passed=True, page=pins.binding(target), manifest=pins.binding(AREA / 'sample_manifest_v3.json'),
        audio_examples=51, all51_files_redecoded=True, clipped_samples=0, all_finite=True,
        all_output_lengths_exact=True, input_speech_samples_never_cropped=True,
        full_aligned_finite_response_tail_kept=True,
        max_tail_duration_s=max(s['tail_duration_s'] for s in samples),
        min_wet_tail_duration_s=min(s['tail_duration_s'] for s in samples if not s['is_clean']),
        maximum_decoded_peak=max(s['decoded_peak'] for s in samples),
        maximum_post_encoding_speech_window_RMS_error_dB=max(abs(s['post_encoding_speech_window_RMS_error_dB']) for s in samples),
        common_gain_per_utterance=True, previous_pages_preserved_byte_exact=True,
        simulated_rooms=8, conditions=16, supplied_transcripts_unchanged=True,
        observed_peak_private_bytes=peak_private, observed_peak_system_commit_ratio=peak_commit,
        browser_verification_by_ROOT_still_required=True, production_enabled=False)
    write_json(AREA / 'verification_v3.json', qa)
    assert not any(name in sys.modules for name in ('torch', 'transformers', 'faster_whisper', 'ctranslate2'))
    print(dict(page=qa['page'], manifest=qa['manifest'], verification=pins.binding(AREA / 'verification_v3.json'),
        PID=os.getpid(), examples=51, peak_private=peak_private, actual_tool_exit_still_required=True), flush=True)


if __name__ == '__main__':
    main()
