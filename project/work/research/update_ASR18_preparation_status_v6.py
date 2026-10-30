"""Current compact status for the measured feature-window preparation track."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

def main():
    value=json.loads((A/'production_preparation_status_v1.json').read_text());lease=json.loads((P/'work/gpu_training_owner.json').read_text())
    value.update(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=lease['status'],
        actual_GPU_running=lease['actual_GPU_running'],actual_owned_GPU_PID=lease['actual_owned_worker_pid'],
        current_stage=lease.get('diagnostic_phase'),diagnostic_folder=lease.get('diagnostic_folder'),production_ready=False,production_enabled=False)
    for key,relative in {'CPU_original_frontend':'CPU_frontend_actual_02/report.json',
        'full_hard_native':'full_hard_2D_native_01/full_hard_native_validation.json',
        'changed20_actual_failure':'sameprocess_stream_checkpoint20_GC_02/actual_partial_failure.json',
        'narrow_source_derivation':'narrow_training_primitives_derivation_v1.json',
        'narrow_import_CPU':'narrow_import_CPU_actual_01/report.json','full_import_CPU':'full_import_CPU_actual_01/report.json',
        'provisional20_feature_window_plan':'feature_window20_plan_01/plan.json','independent40_replay_PCM_targets':'window_replay_PCM_actual_01/report.json',
        'cheap_frontend_CUDA_CPU_comparison':'original_parent_frontend_compare_01/frontend_byte_equivalence.json'}.items():
        path=A/relative
        if path.exists():value['proofs'][key]={**record(path),'passed':json.loads(path.read_text()).get('passed',False)}
    value.update(latest_changed20_completed_updates=1,latest_changed20_passed=False,latest_changed20_checkpoint_created=False,
        sustained_sameprocess20_proven=False,full_hard_two_updates_proved=True,retained_U574_U484_current_fullU_recipe_unresolved=False,
        CPU_only_frontend_private_peak_bytes=540659712,narrow_import_private_increment_bytes=36569088,
        full_import_private_increment_bytes=189521920,narrow_import_private_increment_reduction_bytes=152952832,
        source_carried_replay1000_choices_and_final_RNG_state_cross_arm_exact=True,
        component_profile_actual_exit_code=1,component_profile_failure='Second700MBfrontendstage_admission_rejected_after_firstfullsource_forward_backward',
        component_profile_first_full_source_encoder_private_increment_bytes=1344544768,
        component_profile_first_full_source_decoder_LSTM_private_increment_bytes=292749312,
        component_profile_first_full_source_parent_collate_private_increment_bytes=70729728,
        component_profile_combined_GC_and_emptycache_private_drop_not_pure_GC_attribution=True,
        next_feature_window_maximum_updates=20,next_feature_window_maximum_bytes=512*1024**2,
        actual_production_draws_consumed=0,provisional_plan_does_not_advance_Adam_or_frontiers=True,
        full_corpus_geometry_finalized=False,
        data_finalization_contract=record(Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1\production_finalization_contract_v4.json')))
    value['production_remaining']=['BoundedCPU feature-window preparation and original-input/native-consumer equivalence',
        'Actual sustained sameGPUprocess20 native updates/checkpoint/resume under concurrentDATA',
        'Finalsourceacquisition/catalog/labels/wholemean/contiguousordinals and all729carrieraudits',
        'Final2Dnumericalpolicy/runtime/geometry and exact activeowneddependency ROOT review',
        'Production/cleaner phase coordinator and strict350DEV selection gates']
    write_json(A/'production_preparation_status_v1.json',value);write_json(A/'launch_readiness_v6.json',value)
    memory=json.loads((A/'memory_diagnosis_status.json').read_text());memory.update(updated_at_utc=value['updated_at_utc'],owned_GPU_child_running=lease['actual_GPU_running'],
        actual_owned_GPU_PID=lease['actual_owned_worker_pid'],launch_readiness_receipt=str(A/'launch_readiness_v6.json'),production_preparation_receipt=str(A/'production_preparation_status_v1.json'))
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({key:value[key] for key in ('phase','actual_owned_GPU_PID','production_ready','current_stage')}))

if __name__=='__main__':main()
