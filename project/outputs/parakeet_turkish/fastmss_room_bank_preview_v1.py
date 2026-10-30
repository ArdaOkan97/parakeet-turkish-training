"""Listening-only diverse simulated rooms, full tails, no training activation."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(r'E:\gpu_training_scratchpad')
RATE = 16000


def require(value, message):
    if not value: raise ValueError(message)


def pin(path, *, guard):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()) or
        (path.suffix == '.py' and path.is_relative_to(Path(__file__).resolve().parents[2])),
        'E-only room assets; existing project source may be pinned')
    before = path.stat(); require(0 < before.st_size <= 2*1024**2, 'Small bounded impulse/registry')
    hashed = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(65536): guard(); hashed.update(block)
    require(before.st_size == path.stat().st_size and before.st_mtime_ns == path.stat().st_mtime_ns, 'Asset changed')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': hashed.hexdigest()}


def _stats(value, *, guard, maximum_seconds=128):
    import numpy as np
    require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.ndim == 1
        and value.flags.c_contiguous and 0 < len(value) <= maximum_seconds*RATE, 'Bounded contiguous mono16k float32')
    energy = peak = 0.
    for low in range(0, len(value), 65536):
        guard(); part = value[low:low+65536].astype(np.float64)
        require(np.isfinite(part).all(), 'Finite waveform')
        energy += float(np.dot(part, part)); peak = max(peak, float(np.abs(part).max()))
    return {'RMS': math.sqrt(energy/len(value)), 'peak': peak}


def measure_decay(rir, *, expected_direct_arrival_sample, horizon_seconds, guard):
    """Unfiltered broadband Schroeder T20/T30 extrapolation, not ISO certification."""
    import numpy as np
    _stats(rir, guard=guard, maximum_seconds=7)
    square = rir.astype(np.float64)**2; energy = np.cumsum(square[::-1])[::-1]
    require(energy[0] > 0, 'Nonzero impulse')
    edc = 10*np.log10(np.maximum(energy/energy[0], 1e-300))
    times = np.arange(len(rir), dtype=np.float64)/RATE
    terminal_fraction = float(square[-min(len(square), RATE//20):].sum()/energy[0])
    near_horizon = len(rir)/RATE >= horizon_seconds-.05
    result = {}
    for name, lower in (('T20', -25.), ('T30', -35.)):
        selection = (edc <= -5) & (edc >= lower)
        x, y = times[selection], edc[selection]
        fit = {'fit_dB_range': [-5, lower], 'usable': False, 'status': 'unresolved_fit_range',
               'points': len(x), 'RT60_extrapolated_seconds': None, 'R2': None}
        if len(x) >= 64 and edc.min() <= lower and x[-1] > x[0]:
            xc = x-x.mean(); yc = y-y.mean(); slope = float(np.dot(xc, yc)/np.dot(xc, xc))
            intercept = float(y.mean()-slope*x.mean())
            residual = y-(slope*x+intercept); total = float(np.dot(yc, yc))
            r2 = 1-float(np.dot(residual, residual))/total if total > 0 else 0.
            fit.update(slope_dB_per_second=slope, R2=r2,
                       fit_time_seconds=[float(x[0]), float(x[-1])],
                       observed_fit_dB=[float(y[0]), float(y[-1])])
            if slope < 0:
                fit['RT60_extrapolated_seconds'] = -60/slope
                fit['usable'] = bool(r2 >= .95 and y[0]-y[-1] >= (-5-lower)-1)
                fit['status'] = 'usable_broadband_extrapolation' if fit['usable'] else 'poor_or_incomplete_regression'
        result[name] = fit
    split = min(len(square), max(0, int(round(expected_direct_arrival_sample)))+RATE//20)
    early, late = float(square[:split].sum()), float(square[split:].sum())
    result.update(kind='broadband_Schroeder_energy_decay_regression_v1',
        measurement_is_target=False, octave_band_or_ISO3382_certification=False,
        C50_dB=10*math.log10(early/late) if early > 0 and late > 0 else None,
        D50_fraction=early/(early+late),
        clarity_origin='geometric_direct_arrival_plus_known_fractional_filter_delay',
        terminal_last50ms_energy_fraction=terminal_fraction,
        simulated_tail_near_horizon=near_horizon,
        truncation_suspected=bool(near_horizon and terminal_fraction > 1e-7),
        raw_impulse_seconds=len(rir)/RATE,
        caveat='Finite experimental hybrid rays; unfiltered broadband estimates, not measured real-room RT60.')
    return result


def load_room_bank(registry_path, *, guard):
    bound = pin(registry_path, guard=guard)
    value = json.loads(Path(bound['path']).read_text(encoding='utf-8-sig'))
    require(value['kind'] == 'FastMSS_diverse_hybrid_simulated_room_RIR_bank_v1'
        and value['production_enabled'] is False and 1 <= len(value['assets']) <= 16, 'Explicit listening bank')
    result = []
    for asset in value['assets']:
        require(pin(asset['audio']['path'], guard=guard) == asset['audio'], 'Exact simulated impulse')
        result.append({**asset, 'registry_binding': bound})
    return result


def prepare_room_reverb(clean16k, asset, *, guard):
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve
    clean = _stats(clean16k, guard=guard, maximum_seconds=120)
    require(clean['RMS'] > 1e-12, 'Nonzero source speech required')
    require(asset in load_room_bank(asset['registry_binding']['path'], guard=guard), 'Exact bound room asset')
    with sf.SoundFile(asset['audio']['path']) as source:
        require(source.samplerate == RATE and source.channels == 1 and len(source) <= 7*RATE, 'Bounded mono16k impulse')
        rir = source.read(dtype='float32')
    offset = asset['global_fractional_filter_delay_samples']
    require(type(offset) is int and 0 <= offset < len(rir), 'Known simulator-only digital delay')
    aligned = rir[offset:]
    norm = float(np.sqrt(np.dot(aligned.astype(np.float64), aligned.astype(np.float64))))
    require(norm > 0, 'Nonzero impulse L2')
    guard(); output = fftconvolve(clean16k, aligned/norm, mode='full').astype(np.float32)
    speech_window = _stats(output[:len(clean16k)], guard=guard)
    require(speech_window['RMS'] > 1e-12, 'Nonzero original-duration reverberant speech')
    rms_gain = np.float32(clean['RMS']/speech_window['RMS']); output *= rms_gain
    full = _stats(output, guard=guard)
    return {'audio': output, 'receipt': {
        'kind': 'FastMSS_full_tail_room_reverb_listening_preparation_v1',
        'asset_id': asset['id'], 'room_id': asset['room_id'], 'position': asset['position'],
        'registry': asset['registry_binding'], 'asset_audio': asset['audio'],
        'input_samples': len(clean16k), 'output_samples': len(output), 'sample_rate': RATE,
        'tail_samples_appended': len(output)-len(clean16k), 'full_post_speech_tail_retained': True,
        'source_speech_cropped': False, 'alignment': 'only_known_simulator_global_fractional_filter_delay_removed',
        'alignment_offset_samples': offset, 'geometric_propagation_delay_retained': True,
        'RIR_normalization': 'unit_L2', 'RMS_match_window': 'first_original_input_N_samples_only',
        'RMS_gain': float(rms_gain), 'original_speech_RMS': clean['RMS'], 'unscaled_peak': full['peak'],
        'unscaled_waveform_sha256': hashlib.sha256(memoryview(output).cast('B')).hexdigest(),
        'production_enabled': False}}


def common_gain(clean16k, peaks, *, guard, ceiling=.95):
    require(type(peaks) is list and 1 <= len(peaks) <= 20 and all(math.isfinite(p) and p >= 0 for p in peaks)
        and 0 < ceiling < 1, 'Finite bounded room conditions and headroom')
    peak = max(_stats(clean16k, guard=guard, maximum_seconds=120)['peak'], *peaks)
    return min(1., ceiling/peak) if peak > 0 else 1.


def render_room_reverb(prepared, gain, *, guard):
    np = __import__('numpy')
    require(type(gain) in (int, float) and 0 < gain <= 1 and math.isfinite(gain), 'Shared nonamplifying gain')
    audio = (prepared['audio']*gain).astype(np.float32); stats = _stats(audio, guard=guard)
    require(stats['peak'] <= 1, 'No clipping permitted')
    return {'audio': audio, 'receipt': {**prepared['receipt'], 'common_utterance_gain': gain,
        'output_peak': stats['peak'], 'clipping_applied': False,
        'output_waveform_sha256': hashlib.sha256(memoryview(audio).cast('B')).hexdigest()}}
