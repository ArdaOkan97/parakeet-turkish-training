"""Bind observed owned CPU completions and the next preparation-only review."""
import ast,datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1';F=A.parent/'FastMSS_integration_v1'

def actual(folder,guard,chunk,*,report_name='report.json'):
    report=read_json(folder/report_name);closed=read_json(guard/'closure.json');target=folder/'actual_owned_CLOSED0.json'
    require(report['passed'] is True and type(closed['actual_exit_code']) is int and closed['actual_exit_code']==0
        and closed['monitor_clean'] is True and closed['monitor_joined'] is True and closed['observed']['monitor_error'] is None
        and closed['observed']['samples']>=3,'Observed clean CPU completion required')
    require(not target.exists(),'Preserve actual completions')
    value={'kind':'ASR18_guarded_CPU_tool_completion_v1','owner':'/root/memory_diagnosis',
        'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_PID':closed['pid'],
        'actual_native_exit_code':0,'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':chunk,
        'report':record(folder/report_name),'guard_closure':record(guard/'closure.json'),
        'sampled_private_peak_bytes':closed['observed']['peak_private_bytes'],'valid_sample_count':closed['observed']['samples'],
        'actual_optimizer_updates':0,'production_enabled':False}
    write_json(target,value);return record(target)

def closure(roots):
    found={};pending=list(roots)
    while pending:
        path=Path(pending.pop()).resolve()
        if str(path) in found:continue
        found[str(path)]=record(path);tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node,ast.Import) else []
            for name in names:
                candidate=S/(name.split('.')[0]+'.py')
                if candidate.exists():pending.append(candidate)
    return [found[k] for k in sorted(found)]

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    proofs={
        'DATA_joint_controls':actual(A/'DATA_joint_CPU_actual_01',A/'DATA_joint_CPU_guard_01','ceaff9'),
        'whole_original_replay_metadata':actual(A/'whole_replay_geometry_CPU_actual_01',A/'whole_replay_geometry_CPU_guard_01','5a92ca'),
        'original_scalar_snapshot':actual(A/'original_schedule_snapshot_CPU_actual_02',A/'original_schedule_snapshot_CPU_guard_02','b304b6'),
        'original_metadata_controls':actual(A/'original_schedule_metadata_CPU_actual_01',A/'original_schedule_metadata_CPU_guard_01','d8e3e4'),
        'final_source_signatures':actual(B/'source_identity_inventory_CPU_actual_01',B/'source_identity_inventory_CPU_guard_01','f9a26e')}
    failure=A/'original_schedule_snapshot_CPU_guard_01/resource_stop.json';require(read_json(failure)['reason']=='CPU_watchdog_exception','Preserve actual failed first extraction')
    write_json(A/'original_schedule_preparation_preserved_failures_v1.json',{'kind':'ASR18_preserved_preparation_failures_v1',
        'original_Torch_import_attempt':{'observed_exec_tool_exit_code':1,'actual_tool_chunk_id':'a0e7c5','owned_session':42386,
            'resource_stop':record(failure),'normal_native_closure_present':False,'snapshot_produced':False,
            'attribution':'Unavailable or unpaired telemetry observation; exact underlying API/GIL cause unresolved'},
        'initial_pickle_inspection':{'actual_tool_chunk_id':'13cba2','observed_exec_tool_exit_code':1,
            'guard_closure':record(A/'source_pickle_metadata_CPU_guard_01/closure.json'),
            'cause':'Metadata66,539,150 bytes exceeded deliberately small8MiB inspection bound before reading metadata'},
        'supported_successor_change':'Restricted stdlib-only scalar metadata reader; no Torch import or tensor storage entry decoding. CPU2GB/system guards unchanged.',
        'opaque_checkpoint_full_SHA_reads_raw_payload_bytes':True,'no_tensor_payload_materialization_claimed':True})
    direct=[S/'bpe_final_DATA_population_v1.py',S/'bpe_declared_joint_budget_audit_v1.py',S/'bpe_original_schedule_snapshot_v1.py',
        S/'bpe_checkpoint_scalar_metadata_v1.py',S/'bpe_replay_bucket_stdlib_v1.py',R/'finalize_ASR18_source_identity_inventory_v1.py',
        R/'extract_ASR18_original_schedule_snapshot_v2.py',R/'validate_ASR18_original_schedule_metadata_v1.py',
        R/'validate_ASR18_DATA_joint_preparation_v1.py',R/'prepare_ASR18_replay_geometry_metadata_v1.py',R/'ASR18_bounded_CPU_phase_v3.py']
    manifest=A/'DATA_original_schedule_source_review_manifest_v1.json';require(not manifest.exists(),'Preserve source review manifest')
    value={'kind':'ASR18_DATA_original_schedule_preparation_source_review_manifest_v1','owner':'/root/memory_diagnosis','created_at_utc':now,
        'requested_scope':'bounded_DATA_only_verification_source_signature_inventory_and_original_schedule_metadata',
        'direct_sources':[record(v) for v in direct],'active_owned_dependency_closure':closure(direct),
        'actual_owned_CPU_proofs':proofs,'final_source_signatures':record(B/'final_source_identity_inventory_v1.json'),
        'original_schedule_snapshot':record(A/'original_schedule_snapshot_CPU_actual_02/snapshot.json'),
        'preserved_failures':record(A/'original_schedule_preparation_preserved_failures_v1.json'),
        'separate_frozen_bridge_source_manifest':record(F/'FastMSS_phase_aware_plan_bridge_source_review_manifest_v3.json'),
        'separate_inactive_coordinator_publication_fix':record(A/'production_window_publication_source_review_manifest_v4.json'),
        'new_native_caller_supplied':False,'actual_native_updates':0,'final_positive_DATA_path_exercised':False,
        'all_source_payloads_rehashed':False,'original_all207_three_Adam_native_initialization_or_restore_newly_tested':False,
        'whole_replay_finding':{'rows':1914,'declared_MAX_T':326,'declared_MAX_U':126,'rows_above_old_replay_U75_forecast':116,
            'rows_removed':0,'genuine_cache_and_audio_before_draw_checks_still_required':True},
        'root_source_review_accepted':False,'full_native_forecast_approved':False,'production_enabled':False,'TRAIN_admitted':False}
    write_json(manifest,value)
    source_actual=B/'final_source_identity_inventory_owned_tool_actual_CLOSED0_v1.json';require(not source_actual.exists(),'Preserve source producer proof')
    # This wrapper reports the exact observed successful builder tool, not the
    # sealing tool or a predicted worker exit. The guard proof is retained too.
    source_guard=read_json(B/'source_identity_inventory_CPU_guard_01/closure.json')
    write_json(source_actual,{'kind':'ASR18_source_bound_actual_tool_exit_v1','owner':'/root/memory_diagnosis','observed_at_utc':now,
        'owned_session':38416,'native_PID':source_guard['pid'],'actual_exit_code':0,'actual_tool_chunk_id':'f9a26e',
        'source_path':str(R/'finalize_ASR18_source_identity_inventory_v1.py'),'source_SHA256':record(R/'finalize_ASR18_source_identity_inventory_v1.py')['sha256'],
        'published_marker':record(B/'final_source_identity_inventory_v1.json'),'actual_guarded_completion':proofs['final_source_signatures']})
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=now,
        current_stage='GPU_idle_original_schedule_and_full_source_identity_actual0_next_multwindow_sources_in_preparation',
        actual_GPU_running=False,actual_owned_GPU_PID=None,actual_owned_GPU_session=None,
        next_preparation_source_review=record(manifest),original_schedule_snapshot=value['original_schedule_snapshot'],
        full784_source_identity=record(B/'final_source_identity_inventory_v1.json'),full784_source_identity_actual=record(source_actual),
        final_DATA_positive_path_exercised=False,next_native_forecast_approved=False,production_ready=False,production_enabled=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v24.json',status)
    current=read_json(A/'memory_diagnosis_status.json');current.update(updated_at_utc=now,current_stage=status['current_stage'],
        launch_readiness=record(A/'launch_readiness_v24.json'),launch_readiness_receipt=str(A/'launch_readiness_v24.json'),
        actual_GPU_running=False,actual_owned_GPU_PID=None,production_ready=False)
    write_json(A/'memory_diagnosis_status.json',current)
    print(json.dumps({'source_review':record(manifest),'source_inventory_actual':record(source_actual),'owned_dependency_files':len(value['active_owned_dependency_closure']),'GPU_running':False,'production_ready':False}))
if __name__=='__main__':main()
