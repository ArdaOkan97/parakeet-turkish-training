"""Small synthetic signal controls for real-recording mixer; no real-asset claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'outputs/parakeet_turkish'))


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main(output):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    site = str(PROJECT/'work/training-env/Lib/site-packages')
    sys.path.insert(0, site)
    import numpy as np
    sys.path.remove(site)
    import fastmss_real_noise_v1 as mixer
    output = Path(output).resolve()
    assert output.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve()) and not output.exists()
    from resource_watchdog_v2 import sample_memory
    def guard():
        s = sample_memory(os.getpid(), 'tiny_real_noise_mix_synthetic_control')
        assert s['paired_sampling_within_one_second'] and s['system_memory']['available'] and s['worker_private_memory']['available']
        assert s['worker_private_memory']['private_bytes'] < 2_000_000_000
        assert s['system_memory']['commit_used_bytes'] < .9*s['system_memory']['commit_limit_bytes']
        assert s['system_memory']['commit_available_bytes'] >= 2_000_000_000
    checks = []
    def check(value, name):
        assert value, name; checks.append(name)
    def rejected(function, name):
        try: function()
        except (ValueError, AssertionError): checks.append(name); return
        raise AssertionError('Unexpected acceptance: ' + name)
    clean = (.85*np.sin(np.arange(16000, dtype=np.float64)*.17)).astype(np.float32)
    source_noise = (.3*np.sin(np.arange(5000, dtype=np.float64)*.071)+.15*np.cos(np.arange(5000)*.037)+.1).astype(np.float32)
    before_clean = hashlib.sha256(memoryview(clean).cast('B')).hexdigest()
    before_noise = hashlib.sha256(memoryview(source_noise).cast('B')).hexdigest()
    python_before = random.getstate(); numpy_before = np.random.get_state()
    segment, choice = mixer.prepare_noise_segment(source_noise, len(clean), 123, guard=guard)
    same, same_choice = mixer.prepare_noise_segment(source_noise, len(clean), 123, guard=guard)
    check(np.array_equal(segment, same) and choice == same_choice, 'Same local seed reproduces exact noise segment and offsets')
    check(choice['wrap_seams'] > 0 and choice['short_recording_repeated'], 'Short recorded input repetition is explicit, never synthetic replacement')
    check(len(segment) == len(clean), 'Noise segment retains complete clean length')
    planned = [mixer.prepare_mix(clean, segment, snr, guard=guard) for snr in (20, 10, 5)]
    gain = mixer.common_gain_for_conditions(clean, [item['receipt']['unscaled_peak'] for item in planned], guard=guard)
    check(0 < gain < 1, 'One shared attenuating gain planned from all condition peaks')
    clean_reference = (clean*gain).astype(np.float32)
    for snr, item in zip((20, 10, 5), planned):
        result = mixer.render_mix(item, gain, guard=guard); receipt = result['receipt']
        check(np.array_equal(result['clean_component'], clean_reference), 'Identical speech component level at SNR' + str(snr))
        check(result['audio'].dtype == np.float32 and len(result['audio']) == len(clean), 'Original float32 length preserved at SNR' + str(snr))
        check(abs(receipt['output_measured_snr_db']-snr) < .001, 'Measured actual component RMS matches requested SNR' + str(snr))
        check(receipt['output_peak'] <= .980001 and receipt['clipping_applied'] is False
              and receipt['per_condition_renormalization'] is False, 'Common headroom/no clipping or percondition normalization at SNR' + str(snr))
        check(np.array_equal(result['audio'], result['clean_component']+result['noise_component']), 'Exact returned clean+recorded-noise component sum at SNR' + str(snr))
    check(hashlib.sha256(memoryview(clean).cast('B')).hexdigest() == before_clean
          and hashlib.sha256(memoryview(source_noise).cast('B')).hexdigest() == before_noise, 'Original clean and source-noise arrays never mutated')
    after_numpy = np.random.get_state()
    check(random.getstate() == python_before and numpy_before[0] == after_numpy[0]
          and np.array_equal(numpy_before[1], after_numpy[1]) and numpy_before[2:] == after_numpy[2:], 'Global Python/NumPy RNG unchanged')
    long_noise = np.tile(source_noise, 5)
    _, crop = mixer.prepare_noise_segment(long_noise, len(clean), 123, guard=guard)
    check(crop['wrap_seams'] == 0 and crop['excerpt_start_sample']+len(clean) <= len(long_noise), 'Long recording uses one bounded crop without wrap')
    rejected(lambda: mixer.prepare_mix(np.zeros_like(clean), segment, 10, guard=guard), 'Undefined silent-speech SNR fails without sample rejection or synthetic fallback')
    rejected(lambda: mixer.prepare_mix(clean, np.ones_like(clean), 10, guard=guard), 'DC-only recording fails without Gaussian fallback')
    rejected(lambda: mixer.prepare_mix(clean, segment, float('nan'), guard=guard), 'NaN SNR rejected')
    rejected(lambda: mixer.prepare_mix(clean.astype(np.float64), segment, 10, guard=guard), 'Wrong waveform dtype rejected')
    rejected(lambda: mixer.prepare_noise_segment(source_noise, True, 123, guard=guard), 'Boolean sample count rejected')
    rejected(lambda: mixer.prepare_noise_segment(source_noise, len(clean), True, guard=guard), 'Boolean stochastic seed rejected')
    rejected(lambda: mixer.render_mix(planned[-1], 1., guard=guard), 'Insufficient common headroom rejects instead of clipping')
    check(not any(name.split('.')[0] in {'torch', 'transformers', 'scipy', 'soundfile', 'av', 'pyarrow'} for name in sys.modules), 'No model/audio decoder/SciPy runtime imported')
    output.mkdir()
    report = {'kind': 'FastMSS_real_noise_mixer_synthetic_signal_controls_v1', 'passed': True,
              'checks': checks, 'check_count': len(checks), 'source': pin(__file__), 'mixer': pin(mixer.__file__),
              'NumPy': np.__version__, 'actual_real_recording_or_speech_read': False,
              'actual_AUDIO_files_written': False, 'GPU_or_model_used': False,
              'production_policy_enabled': False, 'synthetic_signal_samples': len(clean),
              'snr_db': [20, 10, 5], 'common_gain': gain}
    (output/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(checks), 'NumPy': np.__version__}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
