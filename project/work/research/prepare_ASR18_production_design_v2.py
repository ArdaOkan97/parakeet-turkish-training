"""Reviewable disabled lifecycle proposal, grounded in actual finite evidence."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1'
def main():
    original=json.loads((B/'training_design_v1.json').read_text());target=B/'production_training_design_v2_prepared.json';require(not target.exists(),'Preserve pinned design history')
    value={'kind':'ASR18_production_training_design_prepared_v2','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prior_design':record(B/'training_design_v1.json'),'production_enabled':False,'production_ready':False,'ROOT_final_source_and_policy_review_pending':True,
        'scientific_scope':original['stages'],'warm_start':original['warm_start'],'vocabulary':original['vocabulary'],
        'diagnostic_checkpoints_may_seed_production':False,'diagnostic_native20_and21_are_not_completed_broad_passes':True,
        'retention':{**original['retention'],'physical_batch':1,'gradient_accumulation':'One full per-row graph/backward, averaged over actual14TR+2replay or full phase tail+2replay before one Adam',
            'physical8_accumulation2_from_original_design_replaced_explicitly':True,'old_microbatch_or_BF16_trajectory_equivalence_claimed':False,
            'proposed_full_serial_DEV_every_new_updates':250,'proposed_additional_full_DEV_at_each_complete_pass_and_stage_end':True,
            'failed_retention_marks_checkpoint_unqualified_but_does_not_truncate_requested_full_passes':True},
        'numeric_and_state_policy':{'joint':'canonicalT1_U128_FP32_v2 with full8711+5 head normalization/full native DP',
            'canonical_dense_and_tiled_reduction_order':'T increasing/U128; full FP32 projected gradients, one final cast',
            'encoder_decoder_autocast':'Existing BF16 policy; active207 model/Adam tensors remain FP32',
            'old_BF16_trajectory_claimed':False,'all207_new2D_oracle_comparison':'Unchanged tolerance pass, not byte exact',
            'source_model_optimizer_RNG':'Original BPE64250/new5122000 plus parked encoder62250; preserve all207/3Adam and source RNG',
            'read_only_frozen_bank':record(S/'stream_frozen_encoder_matrices_v1.py'),'logical_generation_device':record(S/'streamed_generation_device_v1.py'),
            'bootstrap':record(S/'bpe_broad_streamed_bootstrap_v2.py'),'official_native_import':record(S/'native_tdt_minimal_import_v1.py'),
            'blanket_model_cuda_or_dtype_transform_or_DDP_compile_or_concurrent_forward_permitted':False},
        'data_admission':{'final_broad_population_required':True,'partial_mean_U_permitted':False,'complete_passes_each_retained_ordinal_once':2,
            'source_labels_flags_retained':True,'quality_language_disagreement_recommended_filters':[],
            'declared_geometry_source_contract':record(B/'declared_training_geometry_envelope_receipt_contract_v2.json'),
            'full_upfront_PCM_decode_required':False,'exact_full_runtime_native_rate_samples_16k_samples_encoder_frames_before_consumed_draw':True,
            'unknown_or_mismatch':'Stop before consumed draw; source/catalog/labels/denominator reconciliation, no silent drop/crop/pad/skip',
            'heldout12273_exact_reference_PCM_before_draw_protection':True},
        'augmentation':{'policy':record(S/'fastmss_broad_optin_v1.json'),'policy_is_provisional_frozen_diagnostic_candidate':True,
            'broad_TR_student_only':True,'replay_teacher_cleaner_DEV_TEST_remain_clean':True,'longer_than120s_returns_complete_clean_waveform_no_exclusion':True,
            'current_noise':'Synthetic Gaussian; no natural noise dataset claim','current_RIR_registry':'Pinned two-RIR FastMSS pilot; production-policy scope review pending',
            'augmentation_quality_improvement_claimed':False,'CPU_prepared_single_consume_receipt_and_exact_resume_required':True},
        'lifecycle_proposal':{'maximum_updates_per_provisional_window':20,'feature_window_hard_bytes':536870912,'raw_PCM_hard_bytes':134217728,'raw_PCM_maximum_jobs':2,
            'window_stops_before':'Next phase/pass/DEV boundary or declared512MiB capacity; full update/tail remains intact',
            'default_process_chunk':'One serial native GPU child per complete prepared window; release child before next CPU preparation if fresh concurrent-stage admission cannot support it',
            'unjustified_one_update_process_windows_required':False,'positive_next_window_native_and_sameprocess_cleaner_support_pending':True,
            'checkpoint_proposal':'Stream named state after every completed window and before serial DEV; publish latest only after returned writer+validated full payload',
            'checkpoint_recovery':'Exact all207/3Adam/allRNG/both phase cursors/carried replay/source-bound augmentation and committed prefix; branch preserved tail',
            'provisional_failure':'Discard provisional input safely; no sampler/replay draw consumed',
            'post_Adam_or_journal_failure':'Poison live state; reload previous coherent checkpoint, never retry changed live weights with old frontier',
            'transient_input_cleanup':'Exact owned feature/PCM files only after final coherent whole-window checkpoint or explicit verified discard',
            'checkpoint_storage_retention_policy':'Must be finalized/bounded before production; preserve existing original/diagnostic/qualified checkpoints; unbounded245MB per20 storage is not admitted'},
        'measured_fixed_workload_only':{'native_updates':20,'same_GPU_PID':23568,'native_seconds':1130.515,'actual_serial_CPU_preparation_seconds':696.313,
            'parent_observed_GPU_child_seconds':1336.641,'serial_preparation_plus_child_seconds':2032.954,'repeated_fixture_audio_seconds':8121.7,
            'native_private_peak_bytes':4682813440,'parent_system_commit_peak_fraction':0.896650072248,'original_forecast_misses':[1,2],
            'full350DEV_actual_closed':record(A/'feature_window_native20_DEV_01/actual_owned_CLOSED0.json'),'DE_PT_retention_failed':True,
            'fixed_workload_not_corpus_ETA':True,'multiwindow_and_representative_population_throughput_pending':True},
        'guards':{'CUDA_allocator_cap_bytes':8_000_000_000,'CPU_private_cap_from_CUDA8GB_inferred':False,'full_joint_DP_budget_bytes':1_700_000_000,
            'packed_joint_tile_bytes':16_000_000,'native_endcache_bytes':3_000_000_000,'system90percent_three_fresh_observations':True,
            'system95critical':True,'minimum_headroom_bytes':2_000_000_000,'DATA_continues_in_parallel_without_GPU_idle_gate':True,
            'admitted_running_source_or_environment_mutation_permitted':False},
        'remaining':['Actual prospective cleaner1/state/forecast/checkpoint proof','Final DATA assets/carriers/catalog/labels/denominator/declared envelope',
            'Whole declared native allocation and conservative runtime forecast supported-processing proof','Native multiwindow/cadence/coordinator connection and measured overhead',
            'Bounded checkpoint storage lifecycle','Complete final owned source/policy review and explicit user production launch']}
    write_json(target,value);print(json.dumps({'prepared_design':record(target),'production_enabled':False}))
if __name__=='__main__':main()
