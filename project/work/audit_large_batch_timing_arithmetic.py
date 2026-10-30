"""Independent light CPU arithmetic audit, without model/tensor loading."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
checks = []

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def near(name, actual, expected):
    check(name, math.isfinite(actual) and abs(actual - expected) <= 1e-10 * max(1, abs(expected)))

matrix_path = OUT / 'large_batch_matrix.json'
matrix = json.loads(matrix_path.read_text(encoding='utf-8'))
families = {}
intervals = []
hashes = {}
for family in ('typical', 'long'):
    paths = [OUT / f'large_batch_{family}_timing_default_b{b}.json' for b in (1, 2, 4, 8, 16)]
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    baseline = reports[0]
    specifications = baseline['source_clips']
    ids = [row['source_id'] for row in specifications]
    check(family + ':16_unique_train_sources', len(ids) == len(set(ids)) == 16 and all(row['dataset_split'] == 'train' for row in specifications))
    check(family + ':eight_turkish_eight_replay', sum(row['language'] == 'tr' for row in specifications) == 8)
    pcm = []
    for row in specifications:
        info = sf.info(row['audio_filepath'])
        check('wav_format:' + row['source_id'], info.samplerate == 16000 and info.channels == 1)
        check('wav_hash:' + row['source_id'], digest(row['audio_filepath']) == row['wav_sha256'])
        seconds = info.frames / info.samplerate
        near('wav_duration:' + row['source_id'], row['duration'], seconds)
        pcm.append({'source_id': row['source_id'], 'frames': info.frames, 'sample_rate': info.samplerate,
                    'actual_pcm_seconds': seconds, 'wav_sha256': row['wav_sha256']})
    audio = math.fsum(row['actual_pcm_seconds'] for row in pcm)
    table = []
    for path, report in zip(paths, reports):
        label = path.stem
        check(label + ':completed_finite_timing_status', report['status'] in ('measured', 'measured_with_numerical_gate_failure'))
        check(label + ':same_source_order_and_wavs', report['source_ids'] == ids and report['source_clips'] == specifications)
        for key in ('identity', 'initial_trainable_sha256', 'objective_coefficients', 'optimizer', 'precision', 'backend', 'backend_flags', 'hardware', 'torch_version', 'encoder_checkpointing', 'chunk_size'):
            check(label + ':matched_' + key, report[key] == baseline[key])
        check(label + ':bf16_default_compute', report['precision'] == 'bf16' and report['backend'] == 'default')
        check(label + ':fixed_effective16', report['effective_batch'] == 16 and report['accumulation'] * report['batch_size'] == 16)
        check(label + ':fixed_clip_mean_objective', report['objective_coefficients'] == {'tdt': 1, 'token_kl': 5, 'duration_kl': .5, 'feature_distillation': 1}
              and report['turkish_preservation_terms_zero'] is True)
        check(label + ':no_adoption_claim', report['adoption_eligible'] is False)
        near(label + ':audio_seconds_per_update', report['audio_seconds_per_update'], audio)
        timing = report['timing']['end_to_end']
        updates = timing['updates']
        check(label + ':complete_update_array', len(timing['seconds']) == updates and updates == (10 if family == 'typical' else 3))
        check(label + ':matched_warmup', timing['warmup_updates'] == baseline['timing']['end_to_end']['warmup_updates'] == 2)
        check(label + ':positive_wall_times', all(math.isfinite(value) and value > 0 for value in timing['seconds']))
        wall = math.fsum(timing['seconds'])
        seen = audio * updates
        throughput = seen / wall
        near(label + ':audio_numerator', timing['audio_seconds_seen'], seen)
        near(label + ':throughput', timing['audio_min_per_wall_min'], throughput)
        near(label + ':clips_per_second', timing['clips_per_second'], 16 * updates / wall)
        near(label + ':median', timing['median_seconds_per_update'], statistics.median(timing['seconds']))
        near(label + ':p90', timing['p90_seconds_per_update'], sorted(timing['seconds'])[min(updates - 1, int(.9 * updates))])
        check(label + ':memory_peak_bounds', 0 < timing['peak_allocated_vram_gb'] <= timing['peak_reserved_vram_gb'] and timing['peak_allocated_vram_gb'] <= report['max_gpu_gb'])
        valid = math.fsum(row['encoder_frames'] * (row['target_tokens'] + 1) for row in specifications)
        padded = math.fsum(len(group) * max(specifications[i]['encoder_frames'] for i in group) *
                           (max(specifications[i]['target_tokens'] for i in group) + 1) for group in report['groups'])
        near(label + ':independent_lattice_padding', report['padding']['lattice_padding_factor'], padded / valid)
        started, ended = datetime.fromisoformat(timing['started_at_utc']), datetime.fromisoformat(timing['ended_at_utc'])
        check(label + ':timed_wall_interval', started.tzinfo is not None and ended.tzinfo is not None and (ended - started).total_seconds() + .02 >= wall)
        intervals.append((started, ended, label))
        check(label + ':solo_and_reset_recorded', report['gpu_context'] == 'solo' and timing['solo_gpu_operator_declared']
              and timing['reset_optimizer_moments_and_step'] and timing['reset_starting_trainable_sha256'] == baseline['initial_trainable_sha256'])
        entry = next(row for row in matrix['cases'] if row['case_id'] == report['case_id'])
        sha = digest(path)
        check(label + ':exact_matrix_report_hash', sha == entry['report_sha256'])
        hashes[str(path)] = sha
        table.append({'batch_size': report['batch_size'], 'accumulation': report['accumulation'],
            'updates': updates, 'audio_seconds_per_update': audio, 'audio_seconds_seen': seen,
            'measured_wall_seconds': wall, 'median_seconds_per_update': statistics.median(timing['seconds']),
            'p90_seconds_per_update': sorted(timing['seconds'])[min(updates - 1, int(.9 * updates))],
            'audio_min_per_wall_min': throughput, 'clips_per_second': 16 * updates / wall,
            'peak_allocated_vram_gb_reported': timing['peak_allocated_vram_gb'],
            'peak_reserved_vram_gb_reported': timing['peak_reserved_vram_gb'],
            'lattice_padding_factor': padded / valid,
            'same_shape_bf16_numerical_gate_passed_reported': report['parity']['passed'],
            'adoption_eligible': False})
    control = table[0]['audio_min_per_wall_min']
    for row in table:
        row['speedup_vs_matched_b1'] = row['audio_min_per_wall_min'] / control
    families[family] = {'source_ids': ids, 'actual_wav_pcm': pcm, 'table': table,
                        'fastest_measured_batch_size': max(table, key=lambda row: row['audio_min_per_wall_min'])['batch_size']}
for (_, previous_end, previous_label), (next_start, _, next_label) in zip(sorted(intervals), sorted(intervals)[1:]):
    check('nonoverlapping_timed_intervals:' + previous_label + '->' + next_label, next_start >= previous_end)
checkpoint = ROOT / 'work/lora_20261010_01_best.pt'
check('immutable_pilot_checkpoint_bytes', digest(checkpoint) == baseline['identity']['weights_sha256'])
for name, sha in baseline['source_sha256'].items():
    check('immutable_profile_source:' + name, digest(OUT / name) == sha)
result = {'schema_version': 1, 'passed': True, 'checks': len(checks),
          'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Independent CPU arithmetic from all10 raw timing arrays and actual32 WAV PCM lengths; no model/GPU/tensor artifact loading.',
          'families': families, 'report_sha256': hashes, 'matrix_sha256': digest(matrix_path),
          'weights_sha256': baseline['identity']['weights_sha256'],
          'profile_source_sha256': baseline['source_sha256'],
          'limitations': ['Allocated/reserved peaks and numerical gate outcomes are transcribed from measured reports; the separate full tensor auditor verifies their provenance and numerical integrity.',
                          'Effective16 across every physical-batch control; current active training uses effective4, so these ratios do not alone establish training-quality equivalence.',
                          'Fixed repeated TRAIN clips, warmed filesystem/frontend/kernels; long stress has only3 updates.',
                          'Every BF16 same-shape numerical gate remains explicitly recorded; no batch adoption is authorized by this arithmetic audit.']}
target = OUT / 'large_batch_timing_arithmetic_independent_audit.json'
target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'checks': len(checks), 'saved': str(target),
                  'tables': {family: value['table'] for family, value in families.items()}}, ensure_ascii=False))
