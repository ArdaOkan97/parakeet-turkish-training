"""Freeze new inactive preparation components; deliberately no native caller."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S));sys.path.insert(0,str(R))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual,closure
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1';F=A.parent/'FastMSS_integration_v1'

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    forecast=actual(A/'geometry_forecast_CPU_actual_04',A/'geometry_forecast_CPU_guard_04','8a1d10')
    roots=[S/'asr18_calibrated_geometry_forecast_v4.py',R/'validate_ASR18_geometry_forecast_v4.py',
        S/'bpe_final_DATA_window_planner_v2.py',S/'bpe_windows_final_metadata_lock_v1.py',S/'bpe_replay_cache_labels_v1.py',
        S/'fastmss_phase_aware_feature_window_v1.py',S/'fastmss_phase_aware_plan_bridge_v3.py',
        S/'bpe_final_DATA_population_v1.py',S/'bpe_production_window_coordinator_v4.py',R/'ASR18_bounded_CPU_phase_v3.py']
    metadata_wrapper=B/'ASR18_final_two_pass_population_metadata_owned_tool_actual_CLOSED0_v1.json';wrapper=read_json(metadata_wrapper)
    require(wrapper['owner']=='/root/cpu_data_preparation' and type(wrapper['actual_exit_code']) is int and wrapper['actual_exit_code']==0
        and wrapper['actual_tool_chunk_id']=='471f15','Genuine full denominator actual0 required')
    denominator=read_json(wrapper['published_marker']['path']);require(denominator['retained_rows']==2_631_115 and denominator['total_target_tokens']==101_470_757
        and denominator['population_mean_U']==101_470_757/2_631_115,'Exact genuinely closed denominator')
    target=A/'multiwindow_preparation_assembly_source_review_manifest_v1.json';require(not target.exists(),'Preserve source assembly manifest')
    value={'kind':'ASR18_inactive_multiwindow_preparation_assembly_source_review_manifest_v1','owner':'/root/memory_diagnosis','created_at_utc':now,
        'requested_scope':'source_only_preparation_components_for_future_max2window40update_diagnostic',
        'direct_sources':[record(p) for p in roots],'active_owned_dependency_closure':closure(roots),
        'foundation_review_request':record(A/'DATA_original_schedule_source_review_manifest_v1.json'),
        'first_window_planner_review_request':record(A/'first_full_DATA_window_planner_source_review_manifest_v1.json'),
        'coordinator_publication_review_request':record(A/'production_window_publication_source_review_manifest_v4.json'),
        'FastMSS_bridge_review_request':record(F/'FastMSS_phase_aware_plan_bridge_source_review_manifest_v3.json'),
        'FastMSS_sink_review_request':record(F/'FastMSS_phase_aware_feature_window_source_review_manifest_v1.json'),
        'forecast4_CPU_actual':forecast,'proposed_initial_full_worker_private_forecast_bytes':5_560_813_440,
        'forecast4_scope':'TR remains measured20 envelope; original1914 replay declared T326/U126. Explicit additional64MB padding; prospective success unproved. Stage absolute ceiling is current+required increment when higher; first-native observation must be<=3GB.',
        'forecast3_preserved':record(S/'asr18_calibrated_geometry_forecast_v3.py'),
        'full_denominator_only':{'published_marker':wrapper['published_marker'],'actual0':record(metadata_wrapper),
            'N':2_631_115,'full_target_SUM':101_470_757,'mean_U':101_470_757/2_631_115,'declared_envelope_bound':False},
        'future_full_two_pass_with_declared_geometry_must_be_separate_receipt':True,
        'single_source_lineage_for_future_diagnostic':'Original BPE64250/new5122000/parkedLoRA62250 all207 threeAdam allRNG; no diagnostic20/21 production seed',
        'required_next_actual_work':['ROOT source review','Genuine final729 carrier and declared-envelope producers/actual0 plus geometry-bound two-pass/raw DATA aggregate',
            'Actual whole-DATA validator0 and supported logical/encoder/native forecast envelope',
            'NEW bounded decoder/frozen augmentation/original frontend controller integrating row-serial sink',
            'NEW native caller integrating coordinator4 saved-state publication/streamed restore and max2windows lifecycle under fresh admission',
            'Actual multiwindow native+input+checkpoint+recovery proof, source-prefix and owned cleanup, measured throughput/overhead and required DEV barriers'],
        'all_new_source_components_ROOT_reviewed':False,'genuine_final_DATA_positive_path_exercised':False,
        'actual_public_phase_aware_sink_original_frontend_positive_path':False,'actual_native_multiwindow_loop_supplied':False,
        'actual_new_native_updates':0,'GPU_running':False,'production_enabled':False,'production_ready':False}
    write_json(target,value)
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=now,
        current_stage='GPU_idle_multiwindow_preparation_assembly_frozen_ROOT_review_and_final729_DATA_pending',
        multiwindow_preparation_source_review=record(target),next_prospective_forecast_source=record(S/'asr18_calibrated_geometry_forecast_v4.py'),
        next_forecast_metadata_controls=forecast,next_proposed_full_worker_private_forecast_bytes=5_560_813_440,
        final_denominator_only=value['full_denominator_only'],phase_aware_sink_source_manifest=value['FastMSS_sink_review_request'],
        actual_GPU_running=False,actual_owned_GPU_PID=None,actual_owned_GPU_session=None,
        final_DATA_positive_path_exercised=False,next_native_forecast_approved=False,production_ready=False,production_enabled=False,
        coordinator_safety_milestone_backlog_append_tool='ceaf52',backlog_refresh_actual0_tool='318f32')
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v26.json',status)
    memory=read_json(A/'memory_diagnosis_status.json');memory.update(updated_at_utc=now,current_stage=status['current_stage'],
        launch_readiness=record(A/'launch_readiness_v26.json'),launch_readiness_receipt=str(A/'launch_readiness_v26.json'),actual_GPU_running=False,
        actual_owned_GPU_PID=None,production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'source_assembly':record(target),'owned_dependency_sources':len(value['active_owned_dependency_closure']),
        'actual_new_native_updates':0,'GPU_running':False,'production_ready':False}))
if __name__=='__main__':main()
