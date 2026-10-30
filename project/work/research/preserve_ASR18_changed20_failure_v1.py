"""Compact actual partial receipt; failed stability stays failed."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record

def main():
    folder=A/'sameprocess_stream_checkpoint20_GC_02';report=json.loads((folder/'report.json').read_text())
    admission=json.loads((folder/'stage_admission.json').read_text());progress=json.loads((folder/'native_progress.json').read_text())
    sample=admission['fresh_sample'];workers=sample.get('concurrent_data',{}).get('workers',[])
    known=sum(v['private']['private_bytes'] for v in workers if v['private'].get('available'))
    native=json.loads((folder/'native_phase_1.json').read_text());boundaries=progress['history'][0]['graph_boundary_GC_observations']
    value={'kind':'ASR18_changed20_actual_partial_failure_v1','actual_exit_code':1,'actual_tool_chunk_id':'83cf29','owned_session':56712,'native_PID':18728,
        'passed':False,'completed_native_updates':1,'clocks':[64251,2001,62251],'checkpoint_created':False,'sustained_stability_proven':False,
        'parent_resource_stop_reason':report['stop_reason'],'rejection_stage':admission['name'],'incremental_CPU_budget_bytes':admission['incremental_budget_bytes'],
        'system_commit_bytes':sample['system_memory']['commit_used_bytes'],'owned_worker_private_bytes':sample['worker_private_memory']['private_bytes'],
        'system_minus_owned_worker_private_bytes':sample['system_memory']['commit_used_bytes']-sample['worker_private_memory']['private_bytes'],
        'dispatch_listed_DATA_private_sum_bytes':known,'dispatch_listed_DATA_workers':workers,'DATA_private_sum_is_not_all_external_commit':True,
        'native_phase':native,'GC_boundaries':{'count':len(boundaries),'private_bytes_released_at_boundaries':[v['before']['private_bytes']-v['after']['private_bytes'] for v in boundaries],
            'Python_Torch_CUDA_RNG_and207_pointer_version_invariants_passed':True,'NumPy_GC_boundary_assertion_claimed':False},
        'full159s_U574_120s_U484_two_update_proof_unchanged':record(A/'full_hard_2D_native_01/actual_owned_CLOSED0.json'),
        'original_failed20_unchanged':record(A/'sameprocess_stream_checkpoint20_01/report.json'),
        'no_unchanged_retry':True,'no_guard_or_budget_reduction':True,'DATA_paused':False,'unqualified_diagnostic_only':True,
        'bindings':[record(folder/n) for n in ('report.json','stage_admission.json','native_progress.json','native_phase_1.json')]}
    write_json(folder/'actual_partial_failure.json',value)
    status=json.loads((A/'production_preparation_status_v1.json').read_text());lease=json.loads((P/'work/gpu_training_owner.json').read_text())
    status.update(actual_GPU_running=lease['actual_GPU_running'],actual_owned_GPU_PID=lease['actual_owned_worker_pid'],phase=lease['status'],
        current_stage=lease.get('diagnostic_phase'),latest_changed20_actual_partial=record(folder/'actual_partial_failure.json'),
        latest_changed20_completed_updates=1,latest_changed20_passed=False,full_hard_two_updates_proved=True,
        retained_U574_U484_current_fullU_recipe_unresolved=False,full_corpus_geometry_finalized=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v5.json',status)
    memory=json.loads((A/'memory_diagnosis_status.json').read_text());memory.update(owned_GPU_child_running=lease['actual_GPU_running'],actual_owned_GPU_PID=lease['actual_owned_worker_pid'],
        launch_readiness_receipt=str(A/'launch_readiness_v5.json'),sustained20_attempt_passed=False,latest_changed20_partial_receipt=str(folder/'actual_partial_failure.json'))
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'partial':record(folder/'actual_partial_failure.json'),'known_DATA_private_bytes':known,'GC_private_drops':value['GC_boundaries']['private_bytes_released_at_boundaries']}))

if __name__=='__main__':main()
