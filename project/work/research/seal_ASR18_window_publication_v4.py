"""Seal the metadata guard controls and source-only review request."""
import ast,datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import read_json,write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def definitions(path):
    tree=ast.parse(path.read_text());result={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            if isinstance(node,ast.ClassDef):
                node.body=[v for v in node.body if not isinstance(v,ast.FunctionDef) or v.name!='publish_checkpoint']
            result[node.name]=ast.dump(node,include_attributes=False)
    return result
def main():
    folder=A/'production_window_CPU_actual_05';guard=A/'production_window_CPU_guard_05/closure.json';report=read_json(folder/'report.json');closed=read_json(guard)
    require(report['passed'] and report['check_count']==48 and closed['actual_exit_code']==0 and closed['monitor_joined'] and closed['monitor_clean']
        and closed['observed']['samples']>=3 and closed['observed']['monitor_error'] is None,'Actual clean48-control closure required')
    require(definitions(S/'bpe_production_window_coordinator_v2.py')==definitions(S/'bpe_production_window_coordinator_v4.py'),'Only publish_checkpoint AST may change; preserve selection/consume/frontier math')
    actual=folder/'actual_owned_CLOSED0.json';require(not actual.exists(),'Preserve actual evidence');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(actual,{'kind':'ASR18_guarded_CPU_tool_completion_v1','owner':'/root/memory_diagnosis','native_PID':closed['pid'],
        'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'834300','observed_at_utc':now,
        'report':record(folder/'report.json'),'guard_closure':record(guard),'checks':48,'private_peak_bytes':closed['observed']['peak_private_bytes'],
        'actual_native_updates':0,'actual_tensor_payload_reloaded':False,'production_ready':False})
    bindings=[record(v) for v in (S/'bpe_production_window_coordinator_v4.py',S/'bpe_streaming_checkpoint_semantics_v2.py',
        S/'bpe_training_schedule_v1.py',S/'bpe_broad_production_state_v1.py',S/'bpe_broad_audio_v1.py',S/'atomic_io_retry_v1.py',
        S/'bpe_streaming_checkpoint_v1.py',R/'ASR18_bounded_CPU_phase_v3.py',S/'resource_watchdog_v2.py',
        R/'validate_ASR18_production_window_v5.py',R/'validate_ASR18_training_schedule_v1.py')]
    value={'kind':'ASR18_inactive_window_publication_source_review_manifest_v4','created_at_utc':now,'owner':'/root/memory_diagnosis',
        'requested_scope':'inactive_window_checkpoint_publication_semantics','bindings':bindings,
        'prior_preserved_coordinator':record(S/'bpe_production_window_coordinator_v2.py'),
        'only_publish_checkpoint_AST_changed':True,'native_or_selection_or_sampler_consume_math_changed':False,
        'actual_guarded_control':record(actual),'source_bug_reproduction':record(F/'FastMSS_production_window_v2_source_findings_v1.json'),
        'failed_fixture_v3_preserved':record(A/'production_window_CPU_guard_03/closure.json'),
        'failed_fixture_v3_cause':'Independent JSON expected pointer loses two known Python RNG tuple types; only fixturev4 reconstructs/validates them',
        'actual_saved21_semantic_metadata_validated_without_tensors':report['actual21_scalar_metadata_proof'],
        'full_native_writer_coordinator_integration_yet_proved':False,'production_ready':False,'production_enabled':False,
        'semantic_rule':'Immutable pre-writer identity/update/sampler/journal/window snapshot; inspect saved typed scalar tree and trusted writer metadata; match before atomic latest publication',
        'qualification_scope_explicit':True,'writer_callback_contract':'Gets independent copies of saved sampler and wrapped window/caller identity; restore must use returned pointer identity',
        'stale_return_or_failure_policy':'Raise before atomic latest; previous latest bytes and live scientific frontiers remain unchanged',
        'full_payload_hash_verified_separately':True,'whole_tensor_reloading_required_by_semantic_guard':False}
    target=A/'production_window_publication_source_review_manifest_v4.json';require(not target.exists(),'Preserve review request');write_json(target,value)
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=now,current_stage='actual_cleaner21_closed0_inactive_coordinator4_semantics48_passed_ROOT_review_pending',
        coordinator_publication_successor_review_request=record(target),coordinator_publication_CPU_actual=record(actual),
        new_clean_feature_window_retired=True,new_clean_feature_cache_current_bytes=0,
        new_clean_feature_retirement=record(F/'FastMSS_clean_native21_actual_retirement_handoff_v1.json'),production_ready=False,production_enabled=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v23.json',status)
    memory=read_json(A/'memory_diagnosis_status.json');memory.update(updated_at_utc=now,current_stage=status['current_stage'],
        launch_readiness=record(A/'launch_readiness_v23.json'),launch_readiness_receipt=str(A/'launch_readiness_v23.json'),production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'source_review_manifest':record(target),'actual_controls':48,'GPU_running':False,'production_ready':False}))
if __name__=='__main__':main()
