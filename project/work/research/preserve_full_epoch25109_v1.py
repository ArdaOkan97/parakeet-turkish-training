"""Archive only the actually CLOSED full epoch; no Torch/model/audio imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
RUN = 'adaptive_20261012_TR_uniform_full_epoch_01'
DIR = OUT / 'runs' / RUN
LOOP = OUT / 'runs' / (RUN + '_loop_01_window_3')
ARCHIVE = ROOT / 'work/closed-sources/TSC_full_epoch25109_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def record(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}

def atomic(path, value):
    temp = path.with_suffix('.status.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def tail_rows(path, wanted):
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 2_000_000))
        text = handle.read().decode('utf-8', 'replace')
    decoder = json.JSONDecoder()
    found = {}
    for match in re.finditer(r'(?m)^  \{', text):
        try:
            row, _ = decoder.raw_decode(text, match.start() + 2)
        except ValueError:
            continue
        if row.get('step') in wanted:
            found[row['step']] = row
    return found

status = read(LOOP / 'loop_status.json')
selection = read(DIR / 'selection.json')
fresh = Path(status['paths']['fresh_runtime'])
completion_path = fresh.with_suffix('.completion.json')
completion = read(completion_path)
assert status['status'] == 'train_window_ended' and status['child_exit_code'] == 0
assert status['completed_steps'] == 25109 and status['stop_category'] == 'configured_steps_complete'
assert not status['resource_review_required']
assert completion['status'] == 'complete' and completion['sources_unchanged'] and completion['error'] is None
assert selection['completed_steps'] == 25109 and selection['selected_step'] == 19750
uniform = selection['Turkish_domain_coverage']['uniform_pass']
assert uniform['full_pass_complete'] and uniform['unique_clips_seen'] == 253798
assert uniform['draws'] == 253800 and uniform['repeat_visits'] == 2
assert selection['consecutive_retention_failures'] == 23
# Only after terminal own-Popen and native completion admission above do we read checkpoint bytes.
ARCHIVE.mkdir(parents=True, exist_ok=False)
paths = {
    'latest_checkpoint': ROOT / 'work' / (RUN + '_latest.pt'),
    'chosen_checkpoint': ROOT / 'work' / (RUN + '_latest.pt'),
    'source_best_checkpoint': ROOT / 'work' / (RUN + '_best.pt'),
    'run_config': DIR / 'run_config.json', 'training_history': DIR / 'training_history.json',
    'selection': DIR / 'selection.json', 'controller_status': LOOP / 'loop_status.json',
    'initial_source_equivalence': DIR / 'majority_initial_source_equivalence.json',
    'original_native_ready': Path(status['paths']['native']),
    'runtime_native_ready': fresh, 'fresh_runtime': fresh, 'runtime_completion': completion_path,
    'owned_worker_identity': Path(status['paths']['identity']),
    'memory': Path(status['paths']['memory']), 'cache': Path(status['paths']['cache']), 'io': Path(status['paths']['io']),
    'original_champion': ROOT / 'work/closed-sources/TSC_headlr_BPE512_6500_6844_main_v1/best_checkpoint.pt',
    'post_epoch_protocol': OUT / 'post_epoch_experiment_protocol.json',
}
for prefix, step in [('chosen_DEV', 25109), ('source_best_DEV', 19750)]:
    result = OUT / 'results' / (RUN + '_step_' + str(step))
    paths[prefix + '_summary'] = result / 'summary.json'
    paths[prefix + '_metadata'] = result / 'metadata.json'
    paths[prefix + '_metadata_pt'] = result / 'metadata_pt.json'
    for language in ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt'):
        paths[prefix + '_' + language] = result / ('validation_' + language + '.jsonl')
original = {}
archived = {}
copies = {}
for key, source in paths.items():
    if source not in copies:
        before = source.stat()
        destination = ARCHIVE / (key + source.suffix)
        source_digest = hashlib.sha256()
        with source.open('rb') as src, destination.open('xb') as dst:
            while block := src.read(1 << 20):
                source_digest.update(block)
                assert dst.write(block) == len(block)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        saved = record(destination)
        assert saved['sha256'] == source_digest.hexdigest() and saved['bytes'] == before.st_size
        copies[source] = saved
    saved = copies[source]
    original[key] = {'path': str(source.resolve()), 'sha256': saved['sha256'], 'bytes': saved['bytes']}
    archived[key] = saved
config = read(DIR / 'run_config.json')
now = datetime.now(timezone.utc).isoformat()
final_summary = read(paths['chosen_DEV_summary'])
counts = {language: {key: final_summary['validation_' + language]['asr'][key]
                    for key in ('word_errors', 'reference_words', 'utterances')}
          for language in ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt')}
source = read(OUT / 'TSC_full_epoch9000_USER_source_v1.json')
initial = tail_rows(Path(source['artifacts']['training_history']['path']), {9000})[9000]
final = tail_rows(DIR / 'training_history.json', {25109})[25109]
dt = final['train_wall_seconds'] - initial['train_wall_seconds']
audio = {name: final[name] - initial[name] for name in
         ('audio_seen_seconds', 'turkish_audio_seen_seconds', 'replay_audio_seen_seconds')}
started = datetime.fromisoformat(read(OUT/'runs'/(RUN+'_loop_01')/'loop_status.json')['started_at_utc'])
finished = datetime.fromisoformat(status['finished_at_utc'])
wall = (finished - started).total_seconds()
throughput = {'scope': 'Current coherent source9000 to final25109, including both automatic restarts for full wall rates.',
    'updates': 16109, 'training_seconds': dt, 'owned_full_wall_seconds': wall,
    'mixed_audio_seconds': audio['audio_seen_seconds'], 'TR_audio_seconds': audio['turkish_audio_seen_seconds'],
    'RP_audio_seconds': audio['replay_audio_seen_seconds'],
    'TRAIN_only_mixed_audio_minutes_per_minute': audio['audio_seen_seconds']/dt,
    'TRAIN_only_TR_audio_minutes_per_minute': audio['turkish_audio_seen_seconds']/dt,
    'full_wall_mixed_audio_minutes_per_minute': audio['audio_seen_seconds']/wall,
    'full_wall_TR_audio_minutes_per_minute': audio['turkish_audio_seen_seconds']/wall,
    'unclassified_overhead_is_not_assigned_to_an_operation': True}
receipt = {'schema': 1, 'kind': 'TSC_full_epoch_actual_CLOSED25109_preservation_v1',
    'passed': True, 'created_at_utc': now, 'source_run_id': RUN, 'chosen_step': 25109,
    'completed_steps': 25109, 'chosen_checkpoint_role': 'latest',
    'source_actual_owned_exit': {'session_id': 67197, 'exit_code': 0, 'tool_chunk': '90d47e'},
    'native_child_exit_code': 0, 'native_sources_unchanged': True, 'full_pass_complete': True,
    'uniform_pass': uniform, 'counts': counts, 'source_retention_counter': 23,
    'selected_BEST_step': 19750, 'selected_BEST_TR_WER': 312/825,
    'final_is_all_six_qualified': False, 'final_failed_languages': ['es'],
    'source_config_identity_sha256': canonical(config['config_identity']),
    'original_artifact_records': original, 'artifacts': archived, 'archive': str(ARCHIVE),
    'throughput': throughput, 'checkpoint_tensor_deserialization_or_model_load': False,
    'TEST_used': False, 'objective_complete': False}
receipt_path = OUT / 'TSC_full_epoch_actual_CLOSED25109_preservation_v1.json'
with receipt_path.open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2)
review = {'schema': 1, 'kind': 'ROOT_USER_TSC_uniform_IO_control_review_v1',
    'passed': True, 'created_at_utc': now,
    'decision_actor': '/root/benchmark_design under user/ROOT delegated post-epoch protocol',
    'source_run_id': RUN, 'chosen_step': 25109, 'completed_steps': 25109,
    'chosen_checkpoint_role': 'latest', 'source_actual_owned_exit': receipt['source_actual_owned_exit'],
    'actual_closed_source_review': record(receipt_path),
    'source_counter_preserved': 23, 'source_selected_BEST_preserved': 19750,
    'source_unqualified_latest_used': True, 'allow_matched_IO_control': True,
    'no_hybrid_training_state_used': True, 'explicit_coverage_rewind_clips': 0,
    'renewed_Portuguese_grace': False, 'chosen_uniform_unique_clips': 253798,
    'source_uniform_draws': 253800, 'source_repeat_visits': 2,
    'source_lineage_end_step': 25109, 'technical_end_step': 25209, 'technical_updates': 100,
    'declared_change': {'parameter': 'paired_history_persistence',
        'choices': ['double_json', 'single_serialize_two_commits'], 'additional_updates': 100,
        'same_starting_all207_state_Adam_sampler_RNG_history': True,
        'same_future_Turkish_and_replay_examples': True, 'training_math_and_scoring_unchanged': True,
        'same_format_frequency_two_targets_atomic_fsync': True,
        'technical_benchmark_not_a_quality_repair_screen': True,
        'strict_all_six_winner_gates_unchanged': True, 'retention_stop_policy': 'monitor_only'},
    'post_epoch_protocol': record(paths['post_epoch_protocol']),
    'artifacts': original, 'goal_complete': False}
review_path = OUT / 'TSC_uniform_IO25109_USER_control_review_v1.json'
with review_path.open('x', encoding='utf-8') as handle:
    json.dump(review, handle, indent=2)
lease_path = ROOT / 'work/gpu_training_owner.json'
lease = read(lease_path)
lease.update(status='actual_full_epoch_CLOSED0_post_epoch_IO_control_preparing', updated_at_utc=now,
    actual_GPU_running=False, running=False, GPU_reserved=True, reserved=True,
    actual_exit_code=0, actual_closed_tool_chunk='90d47e', actual_completed_step=25109,
    actual_new_uniform_unique_TR_clips=253798, full_pass_complete=True,
    actual_full_epoch_closure_receipt=record(receipt_path), actual_last_complete_DEV_step=25109,
    actual_retention_counter=23, qualified_champion_step=19750,
    owner_task='Preserve completed full epoch and run the authorized matched original8199 persistence control; remain sole GPU owner.')
atomic(lease_path, lease)
plan_path = OUT / 'experiment_plan.json'
plan = read(plan_path)
plan.update(updated_at_utc=now, objective_complete=False,
    status_authority='Actual full epoch CLOSED0 at25109:253798 unique TR clips,2 repeats, complete final350. Final TR292/825 (35.3939%) fails only Spanish53>52; strict all-six BEST19750 remains312/825 (37.8182%). Final and champion byte archives preserved. Authorized matched persistence control is preparing; no GPU currently running.',
    actual_current_training={**receipt, 'actual_GPU_running': False, 'status': 'actual_full_epoch_CLOSED0',
                            'run_id': RUN, 'receipt': record(receipt_path)})
item = next(item for item in plan['ordered_experiments'] if item['id'] == '04')
item.update(status='actual_full_epoch_complete_reviewed_target_unmet', actual_GPU_training_running=False,
    actual_completed_steps=25109, actual_new_uniform_unique_TR_clips=253798,
    actual_new_uniform_unique_clips=253798, actual_full_pass_completed=True,
    current_remaining_uniform_updates=0, latest_actual_DEV_step=25109, latest_actual_DEV_counts=counts,
    current_failure_counter=23, current_TR_word_errors=292, current_PT_word_errors=65,
    latest_actual_TR_word_errors=292, latest_failed_retention_languages=['es'],
    latest_original_five_passed=False, latest_Portuguese_passed=True, latest_all_six_passed=False,
    actual_closure_receipt=receipt_path.name, final_checkpoints_preserved=archived['latest_checkpoint'],
    champion_checkpoint_preserved=archived['source_best_checkpoint'], actual_phase_throughput=throughput,
    pending=['Authorized post-epoch matched persistence control, then quality trials using measured evidence.'])
plan['post_epoch_protocol'] = record(paths['post_epoch_protocol'])
atomic(plan_path, plan)
note = ('Full Turkish epoch ACTUALLY completed and CLOSED0: owned session67197 / tool90d47e, '
    'native child0 / source hashes unchanged, step25109. Exactly253798 unique Turkish clips,253800 draws and2 tail repeats; complete seven-language DEV. '
    'Final TR292/825 (35.3939%); EN60, ES53>52, IT22, FR64, DE68, PT65. Final is not all-six-qualified. '
    'Saved strict all-six BEST19750 stays312/825 (37.8182%); genuine counter23 and all history preserved. '
    'Final plus champion and provenance have verified byte-identical immutable archives. No TEST used; competitive target remains unmet. '
    'Next authorized protocol item is a matched100-update native persistence A/B, identical all207 source state/examples/math/scoring, not a new quality repair screen. '
    'Evidence: ' + receipt_path.name)
for task in ('ASR-15', 'ASR-15.1'):
    subprocess.run([str(OUT/'tools/backlog.exe'), 'task', 'edit', task, '--status', 'In Progress', '--append-notes', note],
                   cwd=OUT, check=True, capture_output=True, encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'), str(OUT/'refresh_backlog.py')],
               check=True, capture_output=True, encoding='utf-8')
print(json.dumps({'receipt': record(receipt_path), 'review': record(review_path),
                  'coverage': uniform, 'counts': counts, 'throughput': throughput}))
