"""Current truthful preparation pointer; immutable diagnostics remain untouched."""
import json,sys
from pathlib import Path
from update_ASR18_preparation_status_v4 import main as prior
P=Path(__file__).resolve().parents[2];A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record

def main():
    prior();value=json.loads((A/'production_preparation_status_v1.json').read_text())
    for key,relative in {'full_U574_U484_CPU_decode':'hard_U_rows_decode_01/decoded/report.json',
                         'canonical2D_joint':'canonical2D_joint_GC_01/actual_joint_precision.json',
                         'canonical2D_all207_tolerance_oracle':'real_target_2D_TDT_gradients_01/real_model_equivalence.json',
                         'instance_denominator_joint_lifetime':'instance_denominator_joint_01/actual_joint_precision.json',
                         'full_hard_2D_native':'full_hard_2D_native_01/full_hard_native_validation.json'}.items():
        path=A/relative
        if path.exists():value['proofs'][key]={**record(path),'passed':json.loads(path.read_text()).get('passed',False)}
    value.update(canonical2D_all207_unchanged_tolerance_gates_passed=True,canonical2D_all207_byte_exact=False,
        canonical2D_full_audio_input_loss_frames_for_numerical_oracle=15,
        canonical2D_all207_aggregate_relative_L2=[.003390626763592747,.0026450433158553663],
        instance_denominator_captured_joint_all_gradients_byte_exact=True,
        instance_denominator_small_retained_workspace_reduction_proven=True,
        instance_denominator_training_phase_memory_improvement_claimed=False,
        sustained20_update_attempt_actual_closed_failure=True,sustained20_update_attempt_completed_updates=2,
        data_finalization_contract={**record(Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1\production_finalization_contract_v3.json'))},
        next_full_hard_row_validation_not_production=True)
    path=A/'full_hard_2D_native_01/full_hard_native_validation.json'
    if path.exists():
        report=json.loads(path.read_text());value['retained_U574_U484_current_fullU_recipe_unresolved']=not report['passed']
        value['full_hard_two_updates_proved']=report['passed']
    value['production_remaining']=[
        'final pinned v5 acquisition/catalog/labels, exact whole mean and contiguous ordinal denominator',
        'complete final geometry and all729 compressed-FLAC carrier coverage with truthful stage provenance',
        'changed same-process sustained stability and measured forecast; prior20 attempt stopped before3',
        'complete new2D full-T/U574/U484 native checkpoint validation and ROOT numerical source review',
        'production/cleaner coordinator, serial350DEV selection gates and active dependency closure review',
    ]
    write_json(A/'production_preparation_status_v1.json',value);write_json(A/'launch_readiness_v4.json',value)
    memory=json.loads((A/'memory_diagnosis_status.json').read_text());memory.update(launch_readiness_receipt=str(A/'launch_readiness_v4.json'),current_numerical_policy='canonicalT1_U128_FP32_v2',sustained20_attempt_passed=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({k:value[k] for k in ('actual_GPU_running','actual_owned_GPU_PID','production_ready','current_stage')}))

if __name__=='__main__':main()
