"""Seal label-only/cache and metadata-planner controls; no native admission."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S));sys.path.insert(0,str(R))
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual,closure
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'

def main():
    cache=actual(A/'replay_scalar_labels_CPU_actual_01',A/'replay_scalar_labels_CPU_guard_01','20a21c')
    planner=actual(A/'DATA_window_planner_CPU_actual_03',A/'DATA_window_planner_CPU_guard_03','cb4ad9')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    roots=[S/'bpe_final_DATA_window_planner_v2.py',S/'bpe_windows_final_metadata_lock_v1.py',S/'bpe_replay_cache_labels_v1.py',
        R/'validate_ASR18_final_DATA_window_planner_v3.py',R/'validate_ASR18_replay_cache_scalar_labels_v1.py',R/'ASR18_bounded_CPU_phase_v3.py']
    manifest=A/'first_full_DATA_window_planner_source_review_manifest_v1.json';require(not manifest.exists(),'Preserve source review request')
    write_json(manifest,{'kind':'ASR18_first_full_DATA_window_planner_source_review_manifest_v1','created_at_utc':now,'owner':'/root/memory_diagnosis',
        'requested_scope':'inactive_original_source_first_window_metadata_planner_and_final_DB_readonly_lifetime',
        'direct_sources':[record(p) for p in roots],'active_owned_dependency_closure':closure(roots),
        'previous_foundation_manifest':record(A/'DATA_original_schedule_source_review_manifest_v1.json'),
        'separate_frozen_bridge_manifest':record(F/'FastMSS_phase_aware_plan_bridge_source_review_manifest_v3.json'),
        'actual_label_only_controls':cache,'actual_metadata_planner_controls':planner,
        'original40_full_replay_labels_exact_to_independent_Torch_reference':True,
        'planner_controls_use_partial_saved14_metadata_protocol_fixture':True,'genuine_final_DATA_positive_path_exercised':False,
        'no_fake_final_DATA_wrapper_created':True,'no_authoritative_or_production_schedule_created':True,
        'no_full_source_or_teacher_feature_cache_tensor_decode_or_new_waveform_read':True,
        'opaque_cache_hash_does_read_original_cache_bytes':True,
        'preserved_failures':[
            {'actual_tool_chunk_id':'81c28d','guard_closure':record(A/'DATA_window_planner_CPU_guard_01/closure.json'),
             'source':record(S/'bpe_final_DATA_window_planner_v1.py'),'cause':'Same-size fixture DB write while writer handle remains open did not immediately advance observable mtime'},
            {'actual_tool_chunk_id':'52ffd4','guard_closure':record(A/'DATA_window_planner_CPU_guard_02/closure.json'),
             'cause':'NEW read-only lifetime guard correctly rejected fixture-owned writable SQLite handle; Python connection context had committed but not closed'}],
        'supported_changes':'NEW planner2 acquires Windows share-read-only handles before full hashes, keeps them until resolver close. Fixture3 explicitly commits AND closes each owned connection.',
        'actual_optimizer_updates':0,'GPU_used':False,'next_native_forecast_approved':False,'native_multiwindow_caller_supplied':False,
        'production_ready':False,'production_enabled':False})
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=now,
        current_stage='GPU_idle_first_full_DATA_plan_metadata12_and_original40_replay_labels6_actual0_sources_review_pending',
        new_first_window_planner_source_review=record(manifest),new_first_window_planner_CPU_actual=planner,
        new_original40_label_only_CPU_actual=cache,actual_GPU_running=False,actual_owned_GPU_PID=None,
        final_DATA_positive_path_exercised=False,next_native_forecast_approved=False,production_ready=False,production_enabled=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v25.json',status)
    memory=read_json(A/'memory_diagnosis_status.json');memory.update(updated_at_utc=now,current_stage=status['current_stage'],
        launch_readiness=record(A/'launch_readiness_v25.json'),launch_readiness_receipt=str(A/'launch_readiness_v25.json'),actual_GPU_running=False,
        actual_owned_GPU_PID=None,production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'source_review':record(manifest),'planner_checks':12,'original_replay_cache_rows':40,'actual_native_updates':0,'production_ready':False}))
if __name__=='__main__':main()
