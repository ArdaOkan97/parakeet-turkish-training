"""Compact current production-preparation state; completed diagnostic proofs persist."""
import datetime,hashlib,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
def main():
    assert json.loads((ROOT/'storage_config.json').read_text())['storage_root']==str(ROOT)
    lease=json.loads((PROJECT/'work/gpu_training_owner.json').read_text());proofs={}
    for key,relative in {'compressed14':'broad_large_audio_actual_02/report.json','pass_state250':'broad_state_actual_01/report.json',
        'new_caller_native2':'new_broad_caller_native_03/production_caller_probe.json','target_DP_and_joint':'target_class_native_math_01/target_class_TDT_validation.json',
        'real207gradients':'real_target_TDT_gradients_01/real_model_equivalence.json','large_full_row':'real_target_TDT_large_01/full_large_row_validation.json',
        'heldout_PCM':'heldout_PCM_binding_v1/report.json'}.items():
        path=AREA/relative
        if path.exists():
            raw=path.read_bytes();value=json.loads(raw);proofs[key]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'passed':value.get('passed',value.get('complete',False))}
    value={'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':'/root/memory_diagnosis','phase':lease['status'],
        'actual_GPU_running':lease['actual_GPU_running'],'actual_owned_GPU_PID':lease['actual_owned_worker_pid'],'current_stage':lease.get('diagnostic_phase'),
        'production_ready':False,'production_enabled':False,'long_production_started':False,'proofs':proofs,
        'original_memory_remedy_completion_preserved':str(AREA/'completion_handoff_v1.json'),
        'diagnostic_checkpoints_unqualified':True,'largest_actual_dense_geometry_bytes':16_326_811_200,'all_broad_labels_flags_retained':True,
        'full_corpus_final_normalization_or_pass_denominator_admitted':False,'complete_broad_passes':0,
        'data_continues_in_parallel':True,'ROOT_final_production_source_review':'requested_pending',
        'production_remaining':['full_pinned_acquisition_catalog_labels_and_final_mean','complete_reference_PCM_binding',
            'actual_all207_and_large_source_bounded_TDT_proof','CPU_augmentation_prepare_once_and_cache_lifecycle',
            'journal_tail_branch_on_restore','production_driver_and_serial350DEV_winner_gates','final_ROOT_source_config_review']}
    write_json(AREA/'production_preparation_status_v1.json',value);write_json(AREA/'launch_readiness_v2.json',value)
    path=AREA/'memory_diagnosis_status.json';memory=json.loads(path.read_text())
    memory.update(updated_at_utc=value['updated_at_utc'],owned_GPU_child_running=lease['actual_GPU_running'],actual_owned_GPU_PID=lease['actual_owned_worker_pid'],
                  production_preparation_receipt=str(AREA/'production_preparation_status_v1.json'))
    write_json(path,memory);print(json.dumps({'phase':value['phase'],'GPU_PID':value['actual_owned_GPU_PID'],'proofs':list(proofs),'production_ready':False}))
if __name__=='__main__':main()
