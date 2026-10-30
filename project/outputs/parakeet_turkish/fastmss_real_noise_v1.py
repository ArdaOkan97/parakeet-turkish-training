"""Inactive real-recording mixer shared by listening and opt-in TRAIN prep.

No GPU, model, decoder, download, Gaussian noise or room simulation. Callers
decode a bounded provenance-bound recorded excerpt to mono16k float32 first.
No clipping/individual normalization: one utterance gain covers every condition.
"""
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(r'E:\gpu_training_scratchpad')
RATE = 16000
MAX_SAMPLES = RATE * 120
CHUNK = 65536


def require(value, message):
    if not value: raise ValueError(message)


def _pin(path, guard):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'New noise assets/evidence must stay on E')
    before = path.stat(); hashed = hashlib.sha256()
    require(0 < before.st_size <= 32*1024**2, 'One bounded compressed asset/evidence file, no fallback')
    with path.open('rb') as stream:
        while block := stream.read(1024**2): guard(); hashed.update(block)
    require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), 'Noise asset changed')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': hashed.hexdigest()}


def validate_registry(registry_path, *, guard):
    registry_pin = _pin(registry_path, guard)
    value = json.loads(Path(registry_pin['path']).read_text(encoding='utf-8-sig'))
    assets = value['assets']; ids = [asset['id'] for asset in assets]
    require(1 <= len(assets) <= 16 and len(set(ids)) == len(ids), 'Bounded unique real-noise assets')
    for asset in assets:
        audio, license = asset['audio'], asset['license']
        require(asset['real_recording'] is True and license['spdx'] == 'CC0-1.0'
                and asset['source_page_url'].startswith('https://freesound.org/')
                and license['url'].startswith('https://creativecommons.org/publicdomain/zero/1.0'),
                'Explicit actual recording and audio-specific CC0 provenance required')
        for item in (audio, license['evidence']):
            actual = _pin(item['path'], guard)
            require(actual == {key: item[key] for key in ('path', 'bytes', 'sha256')}, 'Recorded asset/evidence SHA differs')
        require(audio['asset_variant'] and audio['source_url'].startswith('https://')
                and type(audio['native_sample_rate']) is int and audio['native_sample_rate'] > 0
                and type(audio['native_channels']) is int and 1 <= audio['native_channels'] <= 8
                and type(audio['native_frames']) is int and audio['native_frames'] > 0,
                'Actual compressed variant and verified native geometry required')
    return {'registry': value, 'binding': registry_pin, 'production_enabled': False,
            'license_evidence_pinned_not_independently_adjudicated_here': True}


def _wave(value, *, guard):
    import numpy as np
    guard()
    require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.ndim == 1
            and value.flags.c_contiguous and 0 < len(value) <= MAX_SAMPLES,
            'Bounded mono16k contiguous float32 excerpt, at most120s; no crop/filter fallback')
    require(np.isfinite(value).all(), 'Finite original waveform required')
    return np


def _stats(value, *, guard):
    np = _wave(value, guard=guard); total = sum_value = peak = 0.
    for low in range(0, len(value), CHUNK):
        guard(); part = value[low:low+CHUNK].astype(np.float64)
        total += float(np.dot(part, part)); sum_value += float(part.sum())
        peak = max(peak, float(np.abs(part).max()))
    return {'RMS': math.sqrt(total/len(value)), 'mean': sum_value/len(value), 'peak': peak}


def prepare_noise_segment(noise16k, sample_count, seed, *, guard, source_offset_samples=0):
    np = _wave(noise16k, guard=guard)
    require(type(sample_count) is int and 0 < sample_count <= MAX_SAMPLES
            and type(seed) is int and 0 <= seed < 2**64
            and type(source_offset_samples) is int and source_offset_samples >= 0, 'Exact length/seed/source offset')
    rng = random.Random(seed)
    start = rng.randrange(len(noise16k)-sample_count+1) if len(noise16k) >= sample_count else rng.randrange(len(noise16k))
    result = np.empty(sample_count, dtype=np.float32); cursor = 0; position = start; seams = 0
    while cursor < sample_count:
        guard(); count = min(CHUNK, sample_count-cursor, len(noise16k)-position)
        result[cursor:cursor+count] = noise16k[position:position+count]; cursor += count; position += count
        if position == len(noise16k) and cursor < sample_count: position = 0; seams += 1
    return result, {'sample_rate': RATE, 'samples': sample_count, 'seed': seed,
                    'excerpt_start_sample': start, 'asset_start_sample': source_offset_samples+start,
                    'decoded_excerpt_samples': len(noise16k), 'wrap_seams': seams,
                    'short_recording_repeated': seams > 0, 'crossfade_or_synthetic_noise': False}


def prepare_mix(clean16k, segment16k, snr_db, *, guard):
    np = _wave(clean16k, guard=guard); _wave(segment16k, guard=guard)
    require(len(clean16k) == len(segment16k) and type(snr_db) in (int, float)
            and math.isfinite(snr_db) and -5 <= snr_db <= 40, 'Exact length and finite requested SNR')
    clean_stats, original_noise_stats = _stats(clean16k, guard=guard), _stats(segment16k, guard=guard)
    require(clean_stats['RMS'] > 1e-12, 'Silent speech has undefined SNR; keep source, do not infer rejection')
    # Remove recorded DC only; never alter the supplied speech or noise array.
    centered = np.empty_like(segment16k)
    for low in range(0, len(centered), CHUNK):
        guard(); centered[low:low+CHUNK] = segment16k[low:low+CHUNK].astype(np.float64)-original_noise_stats['mean']
    noise_stats = _stats(centered, guard=guard)
    require(noise_stats['RMS'] > 1e-12, 'Constant/silent recording has undefined noise RMS; no synthetic fallback')
    scale = clean_stats['RMS']/(noise_stats['RMS']*10**(float(snr_db)/20))
    scaled_noise, mixture = np.empty_like(centered), np.empty_like(clean16k)
    for low in range(0, len(centered), CHUNK):
        guard(); scaled_noise[low:low+CHUNK] = centered[low:low+CHUNK].astype(np.float64)*scale
        mixture[low:low+CHUNK] = clean16k[low:low+CHUNK]+scaled_noise[low:low+CHUNK]
    actual_noise = _stats(scaled_noise, guard=guard); peak = _stats(mixture, guard=guard)['peak']
    measured = 20*math.log10(clean_stats['RMS']/actual_noise['RMS'])
    require(abs(measured-float(snr_db)) < .001, 'Actual float32 whole-clip SNR differs')
    return {'clean': clean16k, 'scaled_noise': scaled_noise, 'unscaled_mixture': mixture,
            'receipt': {'kind': 'FastMSS_real_recorded_noise_prepared_mix_v1', 'sample_rate': RATE,
                'samples': len(clean16k), 'requested_snr_db': float(snr_db), 'measured_snr_db': measured,
                'SNR_measure': 'whole_clip_RMS_not_active_speech_RMS', 'clean_RMS': clean_stats['RMS'],
                'scaled_noise_RMS': actual_noise['RMS'], 'recorded_noise_DC_removed': original_noise_stats['mean'],
                'noise_gain': scale, 'unscaled_peak': peak, 'speech_modified_here': False,
                'Gaussian_or_RIR_used': False, 'transcript_policy': 'unchanged', 'production_enabled': False}}


def common_gain_for_conditions(clean16k, peak_values, *, guard, ceiling=.98):
    require(type(peak_values) is list and 1 <= len(peak_values) <= 12
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in peak_values)
            and type(ceiling) in (int, float) and 0 < ceiling < 1, 'Bounded conditions and common headroom')
    peak = max(_stats(clean16k, guard=guard)['peak'], *peak_values)
    return min(1., float(ceiling)/peak) if peak > 0 else 1.


def render_mix(prepared, common_gain, *, guard):
    np = _wave(prepared['clean'], guard=guard)
    require(type(common_gain) in (int, float) and math.isfinite(common_gain) and 0 < common_gain <= 1,
            'One common non-amplifying gain per utterance across clean and every noisy condition')
    clean = (prepared['clean']*common_gain).astype(np.float32)
    noise = (prepared['scaled_noise']*common_gain).astype(np.float32)
    audio = (clean+noise).astype(np.float32); actual = _stats(audio, guard=guard)
    require(actual['peak'] <= 1., 'Common gain insufficient; clipping is forbidden, recompute all-condition gain')
    measured = 20*math.log10(_stats(clean, guard=guard)['RMS']/_stats(noise, guard=guard)['RMS'])
    return {'audio': audio, 'clean_component': clean, 'noise_component': noise,
            'receipt': {**prepared['receipt'], 'common_utterance_gain': common_gain,
                        'output_peak': actual['peak'], 'output_measured_snr_db': measured,
                        'clipping_applied': False, 'per_condition_renormalization': False}}
