"""Inactive opt-in CPU preparation hook using the exact listening mixer.

Preserves every original clip/label; only eligible weak TRAIN/TR student audio
may change. Existing pilot, GPU worker, sampler and checkpoint sources are not
edited. New policy/registry/raw-lifetime identity must be reviewed separately.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import random

import fastmss_real_noise_v1 as mixer

ROOT = mixer.ROOT
WEAK = 'weak_label_continued_training'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def binding(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    mixer.require((path.is_relative_to(ROOT.resolve()) or path.is_relative_to(Path(__file__).resolve().parents[2]))
                  and len(raw) <= 2*1024**2, 'Small source/policy/registry only')
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def decode_noise_excerpt(asset, *, seed, maximum_seconds, guard):
    """Bounded actual compressed recording decode; no full-file waveform pool.

    The adapter's caller must reserve native/resample/workspace before this
    function. No missing-codec fallback, download, clipping or synthetic asset.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio = asset['audio']; guard()
    expected = {key: audio[key] for key in ('path', 'bytes', 'sha256')}
    mixer.require(mixer._pin(audio['path'], guard) == expected, 'Exact licensed compressed recording changed')
    with sf.SoundFile(audio['path']) as source:
        rate, channels, frames = source.samplerate, source.channels, len(source)
        mixer.require(rate == audio['native_sample_rate'] and channels == audio['native_channels']
                      and frames == audio['native_frames'] and rate <= 48000 and channels <= 2,
                      'Actual recording header differs before allocation')
        count = min(frames, int(maximum_seconds*rate))
        start = random.Random(seed).randrange(frames-count+1)
        source.seek(start); native = source.read(count, dtype='float32', always_2d=True)
    mixer.require(native.shape == (count, channels) and np.isfinite(native).all(), 'Exact finite bounded noise decode')
    mono = np.empty(count, dtype=np.float32)
    for low in range(0, count, mixer.CHUNK):
        guard(); mono[low:low+mixer.CHUNK] = native[low:low+mixer.CHUNK].mean(axis=1, dtype=np.float32)
    del native
    common = math.gcd(rate, mixer.RATE)
    decoded = resample_poly(mono, mixer.RATE//common, rate//common, window=('kaiser', 5.0)).astype(np.float32)
    mixer.require(len(decoded) == (count*mixer.RATE+rate-1)//rate and len(decoded) <= mixer.MAX_SAMPLES,
                  'Complete bounded16k excerpt after deterministic resampling')
    return decoded, {'asset_audio': expected, 'asset_variant': audio['asset_variant'],
        'native_start_frame': start, 'native_decoded_frames': count, 'native_sample_rate': rate,
        'native_channels': channels, 'sample_rate': mixer.RATE, 'output_samples': len(decoded),
        'mono_policy': 'channel_mean_float32', 'resample': 'scipy_resample_poly_kaiser5.0_constant_padding',
        'SoundFile': sf.__version__, 'NumPy': np.__version__, 'no_permanent_decoded_cache': True}


class RealRecordedNoiseCPUAdapter:
    def __init__(self, policy_path, *, ROOT_CPU_review=None, admit_extra_raw=None, admit_source=None, guard):
        self.guard = guard; self.policy_binding = binding(policy_path)
        self.policy = json.loads(Path(policy_path).read_text(encoding='utf-8-sig'))
        p = self.policy
        mixer.require(p['kind'] == 'FastMSS_real_recorded_environmental_noise_policy_v1'
            and type(p['enabled']) is bool and p['production_enabled'] is False and p['sample_rate'] == 16000
            and p['phases'] == [WEAK] and p['max_waveform_seconds'] == 120
            and p['noise_mode'] == 'real_recorded_environmental_assets_only' and p['Gaussian_or_RIR_allowed'] is False
            and p['all_rows_retained_on_clean_bypass'] is True and p['transcript_policy'] == 'unchanged_single_target',
            'Explicit separate real-noise policy; never reinterpret the frozen Gaussian/RIR pilot')
        mixer.require(type(p['seed']) is int and 0 <= p['seed'] < 2**64
            and type(p['apply_probability']) in (int, float) and 0 <= p['apply_probability'] <= 1
            and p['snr_db_choices'] == [20, 10, 5] and type(p['noise_excerpt_seconds']) is int
            and 0 < p['noise_excerpt_seconds'] <= 30
            and p['maximum_noise_native_sample_rate'] == 48000 and p['maximum_noise_native_channels'] == 2
            and p['maximum_compressed_asset_bytes'] == 32*1024**2 and p['peak_ceiling'] == .98,
            'Bounded proposed real-noise CPU recipe')
        self.policy_sha256 = hashlib.sha256(canonical(p)).hexdigest()
        self.registry = None; self.registry_sha256 = None; self.admit = admit_extra_raw
        if p['enabled']:
            mixer.require(ROOT_CPU_review is not None and callable(admit_extra_raw) and admit_source is not None,
                          'Enabled CPU policy requires explicit reviewed registry/workspace admission, not production approval')
            review = json.loads(Path(ROOT_CPU_review['path']).read_text(encoding='utf-8-sig'))
            mixer.require(binding(ROOT_CPU_review['path']) == ROOT_CPU_review and review['accepted'] is True
                and review['reviewer'] == '/root' and review['scope'] == 'real_recorded_noise_CPU_preparation_only'
                and review['production_enabled'] is False and review['policy'] == self.policy_binding
                and review['adapter'] == binding(__file__) and review['mixer'] == binding(mixer.__file__)
                and review['registry'] == p['registry_binding'] and review['raw_admitter'] == admit_source,
                'Exact new ROOT CPU policy/source/registry/raw-lifetime review required')
            mixer.require(binding(inspect.getsourcefile(admit_extra_raw)) == admit_source,
                          'Actual caller-owned raw admission source required')
            verified = mixer.validate_registry(p['registry_binding']['path'], guard=guard)
            mixer.require(verified['binding'] == p['registry_binding'], 'Exact actual licensed registry binding')
            mixer.require(all(asset['license']['audio_specific'] is True for asset in verified['registry']['assets']),
                          'Licence evidence must apply to each actual recording, not only annotations')
            self.registry = verified['registry']; self.registry_sha256 = p['registry_binding']['sha256']
        self.identity = {'kind': 'FastMSS_real_recorded_noise_CPU_preparation_identity_v1',
            'adapter': binding(__file__), 'mixer': binding(mixer.__file__), 'policy': self.policy_binding,
            'registry': p['registry_binding'], 'ROOT_CPU_review': ROOT_CPU_review, 'raw_admitter': admit_source,
            'draw_owner': 'existing immutable sampler context', 'commit_owner': 'existing actual optimizer/journal caller',
            'production_enabled': False}

    def apply(self, waveform, row, *, split, phase, role, epoch, draw_index):
        self.guard(); p = self.policy
        mixer.require(type(epoch) is int and type(draw_index) is int and epoch >= 0 and draw_index >= 0,
                      'Actual immutable durable draw context required; no new augmentation counter')
        context = {'source_id': row['source_id'], 'language': row['language'], 'split': split,
                   'phase': phase, 'role': role, 'epoch': epoch, 'draw_index': draw_index}
        base = {'policy_sha256': self.policy_sha256, 'source_id': row['source_id'], 'applied': False,
                'label_policy': 'unchanged_single_target', 'draw': context, 'identity': self.identity}
        if not p['enabled']: return waveform, {**base, 'reason': 'explicitly_disabled_policy'}
        if split != 'train' or phase != WEAK or role != 'student_tr' or row['language'] != 'tr':
            return waveform, {**base, 'reason': 'protected_replay_teacher_cleaner_or_eval'}
        if len(waveform) > mixer.MAX_SAMPLES:
            return waveform, {**base, 'reason': 'full_clip_above120s_clean_bypass_all_rows_retained'}
        seed = int.from_bytes(hashlib.sha256(canonical({'seed': p['seed'], **context})).digest()[:8], 'little')
        rng = random.Random(seed)
        if rng.random() >= p['apply_probability']:
            return waveform, {**base, 'reason': 'deterministic_probability_clean_control'}
        if mixer._stats(waveform, guard=self.guard)['RMS'] <= 1e-12:
            return waveform, {**base, 'reason': 'undefined_silent_clip_SNR_clean_bypass'}
        asset = rng.choice(self.registry['assets']); snr = rng.choice(p['snr_db_choices'])
        # These EXTRA allocations are separate from the current caller's clean
        # decode lifetime. A NEW reviewed ledger must include them before decode.
        raw = {'kind': 'FastMSS_real_noise_extra_raw_lifetime_admission_v1', 'samples': len(waveform),
            'additional_float32_bytes_upper_bound': (8*len(waveform)+30*16000)*4+2*1024**2,
            'native_noise_decode_bytes_upper_bound': (30*48000*(2+1)+30*16000)*4+2*1024**2,
            'compressed_media_domain_bytes': p['maximum_compressed_asset_bytes'],
            'decoder_resample_workspace_separately_required': True, 'existing_raw128MiB_or_two_job_cap_not_relaxed': True}
        admitted = self.admit(raw)
        mixer.require(type(admitted) is dict and admitted['passed'] is True,
                      'Actual caller raw/workspace admission rejected; no crop, row skip or synthetic fallback')
        noise, decoder = decode_noise_excerpt(asset, seed=rng.randrange(2**64),
            maximum_seconds=p['noise_excerpt_seconds'], guard=self.guard)
        segment, choice = mixer.prepare_noise_segment(noise, len(waveform), rng.randrange(2**64), guard=self.guard)
        prepared = mixer.prepare_mix(waveform, segment, snr, guard=self.guard)
        gain = mixer.common_gain_for_conditions(waveform, [prepared['receipt']['unscaled_peak']],
                                               guard=self.guard, ceiling=p['peak_ceiling'])
        output = mixer.render_mix(prepared, gain, guard=self.guard)
        return output['audio'], {**base, 'applied': True, 'reason': 'real_recorded_environmental_noise',
            'asset_id': asset['id'], 'asset_audio': decoder['asset_audio'], 'license': asset['license'],
            'decoder': decoder, 'segment': choice, 'mix': output['receipt'], 'raw_admission': admitted,
            'samples_unchanged': len(waveform), 'sample_rate': 16000, 'speed_changed': False,
            'real_background_may_contain_incidental_voices': True,
            'prepared_not_committed': True, 'sampler_journal_frontier_advanced': False}

    def audio_reader(self, original_reader, *, context_for, record):
        """Explicit opt-in CPU row hook; native GPU reader must remain clean/precomputed."""
        def read(row):
            waveform, receipt = self.apply(original_reader(row), row, **context_for(row))
            record(receipt); return waveform
        return read
