"""Independent bounded JSON arithmetic/source audit; no Torch or GPU imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
paths = {k: OUT / v for k, v in {
    'custom': 'native_training_custom_pilot_v1.json',
    'native': 'native_training_nemo_pilot_v1.json',
    'comparison': 'native_training_pilot_comparison_v1.json'}.items()}
data = {k: json.loads(p.read_text(encoding='utf-8')) for k, p in paths.items()}
checks = 0


def check(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


custom, native, comparison = (data[k] for k in ('custom', 'native', 'comparison'))
check(custom['identity'] == native['identity'], 'All starting/input/runtime/backend/environment fields identical')
check(custom['reset'] == native['reset'], 'Weights/fresh optimizer/RNG reset identical')
check(custom['initial_trainables_sha256'] == native['initial_trainables_sha256'] == custom['reset']['weights_sha256'], 'Same starting trainable tensors')
plan_paths = [p for p in OUT.glob('native_training_pilot_plan_v1*.json')
              if sha(p) == custom['identity']['plan_sha256']]
check(len(plan_paths) == 1, 'Exact immutable source plan')
plan = json.loads(plan_paths[0].read_text(encoding='utf-8'))
check(plan['weights']['selected_step'] == 1871 and plan['runtime']['batch_size'] == 8
      and plan['runtime']['accumulation'] == 2, 'Actual selected1871 B8x2 runtime')
check(plan['runtime'] == custom['identity']['runtime'], 'All hyperparameters match source plan')
summaries = {}
for arm, report in (('custom', custom), ('native', native)):
    check(report['arm'] == arm and report['status'] == 'complete', 'Arm completed successfully')
    check(len(report['updates']) == report['requested_updates'] == 20, 'Full twenty optimizer updates')
    check(len(report['warmup']) == report['requested_warmups'] == 2
          and report['cold_start']['excluded_from_training_rate'], 'Cold/JIT and two warmups excluded')
    check(report['reset']['optimizer_state_empty'] and report['reset']['warmup_updates_discarded'], 'Fresh Adam after reset')
    check(report['weights_sha256_after'] == report['identity']['source1871_weights_sha256'], 'Source weights preserved')
    check(report['frozen_sources_unchanged'] and report['code_sha256_after'] == report['identity']['code_sha256'], 'Source before/after identical')
    for name, digest in report['identity']['code_sha256'].items():
        check(sha(OUT / name) == digest, 'Current frozen source unchanged: ' + name)
    check(report['identity']['cuda_compiler_binding']['version'] == [12, 9], 'Actual coherent CUDA12.9 linker')
    check(report['timing_scope'] == custom['timing_scope'], 'Identical full optimizer-update timing scope')
    assets = {r['source_id']: r for r in report['identity']['assets']}
    check(len(assets) == len(report['identity']['assets']), 'Unique saved actual asset identities')
    for index, (update, specification) in enumerate(zip(report['updates'], plan['updates']), 1):
        rows = [r for group in specification['batches'] for r in group]
        check(update['update'] == index and update['source_ids'] == specification['source_ids'], 'Exact ordered batches/update')
        check(len(rows) == 16 and len({r['source_id'] for r in rows}) == 16
              and all(r['dataset_split'] == 'train' for r in rows), 'Sixteen distinct TRAIN clips per update')
        tr = math.fsum(r['duration'] for r in rows if r['language'] == 'tr')
        replay = math.fsum(r['duration'] for r in rows if r['language'] != 'tr')
        check(near(update['turkish_audio_seconds'], tr) and near(update['replay_audio_seconds'], replay)
              and near(update['total_audio_seconds'], tr + replay), 'Actual unpadded audio numerator')
        check(update['audio'] == specification['audio'], 'Source plan audio map')
        check(math.isfinite(update['wall_seconds']) and update['wall_seconds'] > 0, 'Positive synchronized wall time')
        check(all(math.isfinite(x) for x in update['components']) and math.isfinite(update['gradient_norm']), 'Finite loss/gradient norm')
        check(near(update['weighted_loss'], math.fsum(x * c for x, c in zip(update['components'], (1, 5, .5, 1)))), 'Objective coefficients preserved')
        check(near(update['head_lr'], 5e-5 * index / 25) and near(update['lora_lr'], 3e-4 * index / 25), 'Identical current-control warmup/LRs')
        check(update['cleanup_performed'] == (update['reserved_before_cleanup_bytes'] > 4_000_000_000), 'Same exact cache4 policy')
        for row in rows:
            asset = assets[row['source_id']]
            check(asset['wav_sha256'] == row['wav_sha256'] and near(asset['actual_pcm_seconds'], row['duration'])
                  and asset['encoder_frames'] == row['encoder_frames'] and asset['target_tokens'] == row['target_tokens'], 'Actual saved source/PCM/T/U evidence')
    wall = math.fsum(u['wall_seconds'] for u in report['updates'])
    tr = math.fsum(u['turkish_audio_seconds'] for u in report['updates'])
    replay = math.fsum(u['replay_audio_seconds'] for u in report['updates'])
    check(near(report['timing']['measured_wall_seconds'], wall), 'Whole update denominator arithmetic')
    check(near(report['timing']['combined_audio_minutes_per_wall_minute'], (tr + replay) / wall)
          and near(report['timing']['turkish_audio_minutes_per_wall_minute'], tr / wall), 'Combined and TR-only rates arithmetic')
    check(near(report['timing']['median_seconds_per_update'], statistics.median(u['wall_seconds'] for u in report['updates'])), 'Median update arithmetic')
    check(report['memory']['peak_allocated_bytes'] <= 9_000_000_000, 'Allocator cap respected')
    summaries[arm] = dict(report['timing'], peak_allocated_gb=report['memory']['peak_allocated_bytes'] / 1e9,
                          peak_reserved_gb=report['memory']['peak_reserved_bytes'] / 1e9)
check(comparison['status'] == 'complete' and comparison['common_measured_updates'] == 20, 'Truthful full matched comparison')
for key in ('custom_report', 'native_report'):
    arm = key.split('_')[0]
    check(comparison[key]['sha256'] == sha(paths[arm]), 'Comparison binds actual saved arm report')
ratio = summaries['custom']['measured_wall_seconds'] / summaries['native']['measured_wall_seconds']
check(near(comparison['native_over_custom_training_speed_ratio'], ratio), 'Real speed ratio arithmetic')
check(comparison['adoption_authorized'] is False and comparison['heldout_used'] is False, 'No automatic quality adoption/heldout decision')
check(comparison['numerical_comparison']['full_gradients_measured'] is False
      and comparison['numerical_comparison']['strict_gradient_gate_passed'] is None,
      'No fabricated full model-gradient parity')
check(comparison['numerical_comparison']['first_objective_components_measured'] is True,
      'Actual first loss components measured')
check(comparison['numerical_comparison']['component_absolute_deltas'] ==
      [abs(a-b) for a,b in zip(custom['updates'][0]['components'], native['updates'][0]['components'])],
      'Actual first loss component arithmetic')
output = OUT / 'native_training_pilot_independent_review_v1.json'
check(not output.exists(), 'Preserve prior independent review')
report = {'schema': 1, 'passed': True, 'checks': checks,
          'reports': {k: {'path': str(v), 'sha256': sha(v)} for k, v in paths.items()},
          'plan_path': str(plan_paths[0]), 'plan_sha256': sha(plan_paths[0]), 'script_sha256': sha(__file__),
          'custom': summaries['custom'], 'native': summaries['native'],
          'native_over_custom_speed_ratio': ratio, 'relative_rate_change_percent': (ratio - 1) * 100,
          'measured_wall_saving_seconds': summaries['custom']['measured_wall_seconds'] - summaries['native']['measured_wall_seconds'],
          'scope': 'Two sequential root-owned full20-update TRAIN pilots; same starting tensors/fresh Adam/RNG/B8x2/audio/settings/source hashes. Full step timing includes WAV/cache/frontend/H2D/forward/loss/backward/clip/Adam/new-row scaling/sync/cache release. Model load, cold/JIT, two warmups, telemetry/report publication excluded. No repeated-run confidence interval, native whole-model gradient capture, heldout quality validation or full NeMo trainer comparison;0.56% measured difference does not establish a useful throughput gain. Input asset hashes were compared as saved evidence, not re-read from every WAV/cache here.',
          'Torch_imported': False, 'CUDA_executed': False}
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'checks': checks, 'speed_ratio': ratio, 'custom': summaries['custom'], 'native': summaries['native']}))
