"""ROOT review of the actual CLOSED one-update phase2 check; JSON/bytes only."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
RUN = 'adaptive_20261011_hf_TSC_BPE512_lora_02_smoke'
folder, loop = HERE / 'runs' / RUN, HERE / 'runs' / (RUN + '_loop_01')
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def record(path): return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
status = read(loop / 'loop_status.json')
proof = read(folder / 'adaptation_actual_one_update_smoke.json')
source = read(folder / 'TSC_BPE_LoRA_initial_source_equivalence.json')
selection = read(folder / 'selection.json')
completion_path = loop / (RUN + '_native_backend.completion.json')
completion = read(completion_path)
history = read(folder / 'training_history.json')
assert status['status'] == 'train_window_ended' and status['child_exit_code'] == 0
assert status['stop_category'] == 'phase_bound' and not status['resource_review_required']
assert completion['status'] == 'complete' and completion['sources_unchanged'] and completion['run_id'] == RUN
assert source['passed'] and source['actual_GPU_decode'] and source['before_phase2_updates']
assert source['compared_fixed_DEV_utterances'] == 300 and source['all_raw_transcripts_and_emitted_paths_exact']
assert source['all_six_ASR_counts_exact'] and source['all_five_retention_gates_passed']
assert source['RNG_restored_after_added_evaluation'] and source['module_training_modes_restored_after_added_evaluation']
assert proof['passed'] and proof['actual_native_ASR_backward_and_two_Adam'] and proof['step'] == 7001
for key in ('all3_new_row_deltas_nonzero', 'LoRA192_gradients_finite_and_any_delta_nonzero',
            'protected_old_six_blank_duration_and_frozen12_rawbytes_exact', 'parked_original207_and_phase1_unchanged',
            'complete207_two_optimizer_phase_save_load_passed', 'all207_reloaded_weight_bytes_exact',
            'all195_Adam_moments_clocks_and_groups_exact', 'saved_sampler_history_config_and_actual_RNG_frontier_exact'):
    assert proof[key], key
assert proof['row_clock_K_plus_N'] == 501 and proof['lora_clock_S_plus_N'] == 6501
assert proof['gradient_evidence']['LoRA_gradients_finite'] == 192
assert len(proof['gradient_evidence']['LoRA_nonzero_gradient_names']) == len(proof['LoRA_changed_names']) == 192
assert len(history) == 7001 and history[-1]['validation']['retention_gate_passed']
assert selection['completed_steps'] == 7001 and selection['adaptation_updates_completed'] == 1
initial, last = history[6999], history[7000]
wall = last['train_wall_seconds'] - initial['train_wall_seconds']
audio = last['audio_seen_seconds'] - initial['audio_seen_seconds']
report = {'kind': 'ROOT_actual_CLOSED_TSC_BPE_encoder_LoRA_one_update_review_v1', 'passed': True,
          'run_id': RUN, 'owned_actual_exit': {'session_id': 60636, 'exit_code': 0, 'tool_chunk': '80e835', 'observed_at_utc': '2026-10-11T22:59:45Z'},
          'actual_source300_and_native_backward_two_Adam_save_load_passed': True,
          'all192_LoRA_gradients_and_deltas_nonzero': True, 'all3_new_row_gradients_and_deltas_nonzero': True,
          'protected_weights_and_immutable_sources_preserved': True, 'actual_final_gate': last['validation'],
          'phase_train_seconds': wall, 'phase_audio_seconds': audio, 'phase_audio_min_per_train_min': audio / wall,
          'only_one_update_with_native_warmup_not_longrun_throughput': True,
          'peak_allocated_vram_gb': last['peak_allocated_vram_gb'],
          'allow_separate_prepared_1000_update_main_from_same_phase1_source': True,
          'external_smoke_exact_resume_or_longrun_memory_fit_proven': False,
          'artifacts': {name: record(path) for name, path in {'source300': folder / 'TSC_BPE_LoRA_initial_source_equivalence.json',
              'actual_update_proof': folder / 'adaptation_actual_one_update_smoke.json', 'controller': loop / 'loop_status.json',
              'native_completion': completion_path, 'selection': folder / 'selection.json'}.items()}}
path = HERE / 'TSC_BPE_LoRA_actual_one_update_smoke_ROOT_review_v1.json'
assert not path.exists()
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'review': record(path), 'DEV': last['validation'], 'phase_warmup_rate': audio / wall}))
