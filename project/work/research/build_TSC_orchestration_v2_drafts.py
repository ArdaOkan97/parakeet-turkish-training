"""Create NEW v2 TSC orchestration drafts only; never modify frozen v1."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
names=('run_hf_TSC_arm','resume_TSC_same_science','launch_TSC_same_science_resume','supervise_TSC_same_science')
replacements={n+'_v1':n+'_v2' for n in names}
replacements.update({
    'TSC_runtime_chain_pins_v1':'TSC_runtime_chain_pins_v2',
    'tsc_training_inputs_v1':'tsc_training_inputs_v2','tsc_supplemental_data_policy_v1':'tsc_supplemental_data_policy_v2',
    'train_lora_batched_tsc_v1':'train_lora_batched_tsc_v2','cuda_cache_caller_tsc_v1':'cuda_cache_caller_tsc_v2',
    'launch_with_io_retry_tsc_v1':'launch_with_io_retry_tsc_v2','launch_with_cache_reclaim_tsc_v1':'launch_with_cache_reclaim_tsc_v2',
    'launch_with_memory_guard_tsc_v1':'launch_with_memory_guard_tsc_v2','launch_native_training_tsc_v1':'launch_native_training_tsc_v2',
    'launch_with_memory_guard_TSC_resume_v1':'launch_with_memory_guard_TSC_resume_v2',
    'isolated_TSC_runtime_chain_pins_v1':'isolated_TSC_runtime_chain_pins_v2',
    'TSC_supplemental_TR_frame_target_inputs_v1':'TSC_supplemental_TR_frame_target_inputs_v2',
    'isolated_HF_TSC_guarded_quality_arm_v1':'isolated_HF_TSC_guarded_quality_arm_v2',
    'TSC_same_run_exact_science_continuation_v1':'TSC_same_run_exact_science_continuation_v2',
    'TSC_same_science_native_preload_resume_v1':'TSC_same_science_native_preload_resume_v2',
    'TSC_same_science_native_preload_resume_completion_v1':'TSC_same_science_native_preload_resume_completion_v2',
    'pinned_TSC_controller_child_close_publication_source_proof_v1':'pinned_TSC_controller_child_close_publication_source_proof_v2',
    'pinned_live_TSC_controller_metadata_bootstrap_v1':'pinned_live_TSC_controller_metadata_bootstrap_v2',
    'pinned_TSC_controller_evidence_verified_window_closure_v1':'pinned_TSC_controller_evidence_verified_window_closure_v2',
    'automatic_TSC_same_science_window_supervisor_v1':'automatic_TSC_same_science_window_supervisor_v2',
    'memory_guard_training_launcher_TSC_resume_v1':'memory_guard_training_launcher_TSC_resume_v2',
    'memory_guard_training_launcher_tsc_v1':'memory_guard_training_launcher_tsc_v2',
    'cache_reclamation_training_launcher_tsc_v1':'cache_reclamation_training_launcher_tsc_v2',
    'atomic_io_retry_tsc_v1':'atomic_io_retry_tsc_v2',
})
for name in names:
    source=HERE/(name+'_v1.py');target=HERE/(name+'_v2.py')
    text=source.read_text(encoding='utf-8')
    for old,new in sorted(replacements.items(),key=lambda item:-len(item[0])):text=text.replace(old,new)
    with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print('Created four new draft sources; final new runtime pins remain pending.')
