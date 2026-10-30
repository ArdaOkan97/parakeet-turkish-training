"""Bounded listening-only RIR hook; no training policy is enabled.

Uses the two already generated FastMSS impulses, not new assets or recorded
rooms. Shares whole-clip RMS/headroom helpers with the real-noise preview.
"""
import hashlib
import json
import math

import fastmss_real_noise_v1 as mixer

REGISTRY = mixer.ROOT / 'analysis/FastMSS_integration_v1/upstream_RIR_pilot_01/FastMSS_RIR_assets.json'
REGISTRY_SHA256 = '70fd65adcc00ffd822558d05dd317e1ef746db7d514298d79f040fe9443dcdf1'


def available_RIRs(*, guard):
    """Return exact two verified simulated-asset records, each with registry pin."""
    pin = mixer._pin(REGISTRY, guard)
    mixer.require(pin['sha256'] == REGISTRY_SHA256, 'Frozen two-position RIR registry changed')
    assets = json.loads(REGISTRY.read_text(encoding='utf-8-sig'))['assets']
    mixer.require([a['id'] for a in assets] == ['FastMSS_0_0', 'FastMSS_0_1'], 'Exactly two existing positions')
    result = []
    for asset in assets:
        provenance = asset['provenance']
        actual = mixer._pin(asset['path'], guard)
        mixer.require(actual['sha256'] == asset['sha256'] and asset['kind'] == 'rir'
            and asset['sample_rate'] == 16000 and provenance['room_index'] == 0
            and provenance['underlying_recorded_audio'] is False
            and provenance['revision'] == 'e481ceac821ef0cb31033bf43b45922e377f528d',
            'Existing simulated room/position identity required')
        result.append({**asset, 'verified_audio': actual, 'registry_binding': pin})
    return result


def prepare_reverberation(clean16k, asset, *, guard):
    """Return unscaled exact-N reverb and provenance; headroom is applied later.

    argmax(abs(RIR)) is strongest-tap alignment, NOT a demonstrated physical
    direct-arrival time. Full convolution is retained only through original N;
    the trailing post-utterance response is not appended. No input trim occurs.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve
    mixer._wave(clean16k, guard=guard)
    known = {a['id']: a for a in available_RIRs(guard=guard)}
    mixer.require(asset == known.get(asset['id']), 'Exact verified asset record required')
    with sf.SoundFile(asset['path']) as source:
        mixer.require(source.samplerate == 16000 and source.channels == 1
            and 0 < len(source) <= 24000, 'Mono16k simulated impulse at most1.5s before allocation')
        rir = source.read(dtype='float32')
    mixer.require(np.isfinite(rir).all() and np.any(rir), 'Finite nonzero RIR')
    strongest = int(np.argmax(np.abs(rir)))
    aligned = rir[strongest:]
    energy = float(np.sqrt(np.mean(aligned.astype(np.float64)**2)*len(aligned)))
    mixer.require(math.isfinite(energy) and energy > 0, 'Nonzero impulse L2 norm')
    normalized = aligned / energy
    original = mixer._stats(clean16k, guard=guard)
    mixer.require(original['RMS'] > 1e-12, 'Silent speech has undefined RMS restoration')
    guard()
    output = fftconvolve(clean16k, normalized, mode='full')[:len(clean16k)].astype(np.float32)
    measured = mixer._stats(output, guard=guard)
    mixer.require(measured['RMS'] > 1e-12, 'Convolution must retain speech energy')
    rms_gain = np.float32(original['RMS']/measured['RMS'])
    output *= rms_gain
    final = mixer._stats(output, guard=guard)
    return {'audio': output, 'receipt': {
        'kind': 'FastMSS_existing_simulated_RIR_prepared_preview_v1',
        'asset_id': asset['id'], 'asset_audio': asset['verified_audio'],
        'registry': asset['registry_binding'], 'provenance': asset['provenance'],
        'room_label': 'one locally simulated room; two source positions',
        'room_index': 0, 'source_position_index': asset['provenance']['source_position_index'],
        'measured_RT60_claimed': False,
        'configuration_RT60_range_only_seconds': asset['provenance']['config']['rt60'],
        'sample_rate': 16000, 'samples': len(clean16k), 'original_RIR_samples': len(rir),
        'strongest_tap_offset_removed_samples': strongest,
        'alignment': 'strongest_tap_not_proven_physical_direct_arrival',
        'RIR_normalization': 'unit_L2_after_strongest_tap_alignment',
        'convolution': 'scipy_fftconvolve_full_then_original_N',
        'post_utterance_tail': 'not_appended_original_N_preserved',
        'speech_input_trimmed': False, 'wholeclip_RMS_restored': True,
        'original_speech_RMS': original['RMS'], 'unscaled_output_RMS': final['RMS'],
        'RMS_restoration_gain': float(rms_gain), 'unscaled_peak': final['peak'],
        'clean_waveform_sha256': hashlib.sha256(memoryview(clean16k).cast('B')).hexdigest(),
        'unscaled_waveform_sha256': hashlib.sha256(memoryview(output).cast('B')).hexdigest(),
        'noise_or_Gaussian_used_here': False, 'transcript_policy': 'unchanged',
        'production_enabled': False}}


def render_reverberation(prepared, common_gain, *, guard):
    """Apply the same per-utterance nonamplifying gain as every other condition."""
    np = mixer._wave(prepared['audio'], guard=guard)
    mixer.require(type(common_gain) in (int, float) and math.isfinite(common_gain)
        and 0 < common_gain <= 1, 'One common nonamplifying utterance gain')
    output = (prepared['audio']*common_gain).astype(np.float32)
    stats = mixer._stats(output, guard=guard)
    mixer.require(stats['peak'] <= 1, 'Headroom insufficient; recompute all-condition gain, never clip')
    return {'audio': output, 'receipt': {**prepared['receipt'],
        'common_utterance_gain': common_gain, 'output_peak': stats['peak'],
        'output_waveform_sha256': hashlib.sha256(memoryview(output).cast('B')).hexdigest(),
        'clipping_applied': False, 'per_condition_renormalization': False}}
