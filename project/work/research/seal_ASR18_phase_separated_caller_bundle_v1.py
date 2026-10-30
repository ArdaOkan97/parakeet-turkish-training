"""Seal observed metadata controls and inactive exact source closure; no launch."""
import ast, datetime, json, sys
from pathlib import Path
P=Path(__file__).resolve().parents[2]; S=P/'outputs/parakeet_turkish'; R=P/'work/research'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1'); F=A.parent/'FastMSS_integration_v1'


def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report=A/'phase_separated_parent_metadata_CPU_actual_01/report.json'
    guard=A/'phase_separated_parent_metadata_CPU_guard_01/closure.json'
    value=read_json(report); closure=read_json(guard)
    require(value['passed'] is True and value['check_count']==22 and closure['actual_exit_code']==0
        and closure['monitor_clean'] is True and closure['monitor_joined'] is True
        and closure['observed']['samples']>=3 and closure['observed']['monitor_error'] is None,
        'Actual observed metadata0 plus clean joined guard required')
    actual=report.parent/'actual_owned_CLOSED0.json'
    require(not actual.exists(),'Immutable actual proof already sealed')
    write_json(actual,{'kind':'ASR18_phase_separated_parent_metadata_controls_actual_TOOL0_v1',
        'owner':'/root/memory_diagnosis','observed_at_utc':now,'actual_exit_code':0,
        'actual_tool_chunk_id':'94e496','native_PID':closure['pid'],'report':record(report),
        'guard':record(guard),'source':value['source'],'actual_GPU_or_original_audio_or_frontend_job':False})
    prior=A/'concrete_multiwindow_caller_source_review_manifest_v8.json'; old=read_json(prior)
    pins={x['path']:x for x in old['bindings']}
    for pin in pins.values():require(record(pin['path'])==pin,'Preserved predecessor binding changed')
    direct=[R/'run_ASR18_full_DATA_multiwindow_v8.py',S/'asr18_full_DATA_native_window_v3.py',
        S/'asr18_deferred_GPU_recovery_v1.py',S/'asr18_restore_only_runtime_v1.py',
        S/'asr18_streamed_persistence_fingerprint_v1.py',S/'asr18_separated_worker_forecast_v1.py',
        S/'asr18_checkpoint_next_draw_v1.py',R/'prepare_ASR18_full_DATA_multiwindow_configuration_v6.py',
        S/'asr18_full_DATA_multiwindow.disabled.v8.json']
    cpu_path=F/'FastMSS_persistence_idle_CPU_source_review_manifest_v1.json'; cpu=read_json(cpu_path)
    direct += [Path(x['path']) for x in cpu['bindings'].values()]
    require(len(set(map(str,direct)))==len(direct),'Unique direct source list')
    # Preserve complete earlier import/source/history scope and recursively add
    # every resolvable owned Python import, including imports nested in functions.
    queue=[x for x in direct if x.suffix=='.py']; seen=set(); edges={}
    while queue:
        path=queue.pop().resolve()
        if str(path) in seen:continue
        seen.add(str(path)); pins[str(path)]=record(path); imports=[]
        tree=ast.parse(path.read_bytes(),filename=str(path))
        for node in ast.walk(tree):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                root=name.split('.')[0]
                for parent in (path.parent,S,R):
                    candidate=parent/(root+'.py')
                    if candidate.is_file() and candidate.resolve().is_relative_to(P):
                        imports.append(str(candidate.resolve())); queue.append(candidate);break
        edges[str(path)]=sorted(set(imports))
    extras=[cpu_path,Path(cpu['template']['path']),Path(cpu['interface']['path']),Path(cpu['fixture']['path']),
        Path(cpu['actual_controls']['path']),A/'phase_separated_persistence_component_source_manifest_v1.json',
        A/'phase_separated_native_engine_recovery_source_manifest_v1.json',
        A/'ROOT_phase_separated_persistence_components_and_CPU5_source_read_review_v1.json',
        A/'ROOT_phase_separated_engine3_recovery1_source_read_review_v1.json',
        F/'FastMSS_phase_separated_engine_recovery_subset_read_review_v1.json',
        R/'validate_ASR18_phase_separated_parent_metadata_CPU_v1.py',actual,report,guard,
        Path(value['actual_original_source_next_draw_packet']['path']),Path(__file__)]
    for path in direct+extras:pins[str(path.resolve())]=record(path)
    for pin in pins.values():require(record(pin['path'])==pin,'Exact final source/evidence signature changed')
    manifest=A/'concrete_multiwindow_caller_source_review_manifest_v9.json'
    require(not manifest.exists(),'New immutable full source manifest required')
    write_json(manifest,{'kind':'ASR18_concrete_resident_first_full_DATA_multiwindow_caller_source_review_manifest_v9',
        'owner':'/root/memory_diagnosis','created_at_utc':now,
        'requested_source_scope':'source_read_review_for_future_bounded_full_DATA_two_window_diagnostic',
        'later_execution_review_scope_required':'finite_full_DATA_two_window_diagnostic_v1',
        'exact_execution_configuration_binding_required':True,'production_enabled':False,
        'direct_sources':[record(x) for x in direct],
        'bindings':sorted(pins.values(),key=lambda x:x['path']),
        'owned_import_closure_roots':[record(x) for x in direct if x.suffix=='.py'],
        'owned_resolvable_AST_import_edges':edges,
        'preserved_predecessor_source_and_history_manifest':record(prior),
        'predecessor_bindings_retained_as_unchanged_history_and_dynamic_dependencies':True,
        'CPU_owner_bundle':record(cpu_path),'CPU_contract_template':cpu['template'],
        'configuration_template':record(S/'asr18_full_DATA_multiwindow.disabled.v8.json'),
        'configuration_assembler':record(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v6.py'),
        'configuration_assembler_executed':False,
        'actual_parent_metadata_and_original_nextdraw_controls':record(actual),
        'prior_component_source_reviews':[record(A/'ROOT_phase_separated_persistence_components_and_CPU5_source_read_review_v1.json'),record(A/'ROOT_phase_separated_engine3_recovery1_source_read_review_v1.json')],
        'changed_workload':{'resident_native_private_forecast_bytes':5560813440,
            'unchanged_write_and_persistence_increment_bytes':64000000,
            'resident_entire_private_forecast_bytes':5624813440,
            'resident_GPU_checkpoint_restore_performed':False,
            'startup_fallback_restore_only_before_native_within_unchanged_350MB_monitor':True,
            'final_restore_only_pre_stage_private_limit_bytes':3000000000,
            'final_restore_only_increment_bytes':350000000,
            'final_restore_only_entire_private_forecast_bytes':3350000000,
            'every_captured_checkpoint_must_actually_restore_after_final_native_process_death_before_overall_pass':True,
            'resident_first_next_CPU_preparation_policy_preserved':True},
        'preserved_limits':{'native_variation_margin_bytes':750000000,'CPU_whole_family_private_bytes':1934217728,
            'system_yield_ratio':0.9,'critical_ratio':0.95,'headroom_bytes':2000000000,
            'E_reserve_bytes':100000000000,'maximum_windows':2,'maximum_new_updates':40},
        'unchanged_science':'Original source0 all207 FP32/threeAdam/clocks/RNG; same canonical2D numerical policy, source rows, targets, draw ordering and weak augmentation. Diagnostic checkpoints never scientific seeds.',
        'actual_new_native_audio_frontend_or_GPU_work_started':False,
        'source_resource_configuration_and_fresh_admission_still_required':True})
    stage='GPU_idle_phase_separated_parent8_full_source_bundle_frozen_review_pending'
    previous=A/'launch_readiness_v49.json'; readiness=A/'launch_readiness_v50.json'
    require(not readiness.exists(),'Immutable numbered status successor required')
    prior_status=read_json(previous)
    status={'kind':'ASR18_compact_memory_and_production_preparation_readiness_v50','owner':'/root/memory_diagnosis',
        'updated_at_utc':now,'current_stage':stage,'actual_GPU_running':False,'actual_owned_GPU_PID':None,
        'current_CPU_metadata_child':None,'production_enabled':False,'production_ready':False,
        'diagnostic_checkpoints_qualified':False,'previous_history':record(previous),
        'current_source_review_manifest':record(manifest),'actual_metadata_controls':record(actual),
        'latest_failed_admission':prior_status['failed_fresh_prelaunch'],
        'new_supported_workload_admitted':False,'resident_entire_private_forecast_bytes':5624813440,
        'final_recovery_entire_private_forecast_bytes':3350000000,'whole_CPU_family_private_allowance_bytes':1934217728,
        'full_DATA':prior_status['full_DATA'],'retention':prior_status['retention'],
        'shared512MiB_feature_slot_empty':True,
        'remaining':'Exact full parent/runtime/CPU5 source review, unchanged arithmetic rebound CPU-resource review, assembler6 exact metadata config review, then fresh resident admission. Every captured checkpoint recovery in a separately admitted worker is mandatory before overall diagnostic pass.'}
    write_json(readiness,status)
    pointer=A/'current_launch_readiness.json'
    write_json(pointer,{'kind':'ASR18_current_mutable_launch_readiness_pointer_v1','owner':'/root/memory_diagnosis',
        'updated_at_utc':now,'current_readiness':record(readiness),'current_stage':stage,
        'actual_GPU_running':False,'actual_owned_GPU_PID':None,'current_CPU_metadata_child':None,
        'latest_failed_admission':status['latest_failed_admission'],'production_enabled':False})
    for name in ('memory_diagnosis_status.json','production_preparation_status_v1.json'):
        path=A/name; value=read_json(path)
        value.update(updated_at_utc=now,current_stage=stage,current_source_review_manifest=record(manifest),
            latest_launch_readiness=record(readiness),actual_GPU_running=False,current_CPU_metadata_child=None,
            current_dispatch_pointer=str(pointer),production_enabled=False,production_ready=False)
        value['source_preparation']={'parent':str(direct[0]),'assembler':str(R/'prepare_ASR18_full_DATA_multiwindow_configuration_v6.py'),
            'CPU_owner_bundle':record(cpu_path),'actual_metadata_controls':record(actual),
            'new_original_trial_performed':False,'all_new_reviews_required':True}
        write_json(path,value)
    lease=P/'work/gpu_training_owner.json'; value=read_json(lease)
    require(value['owner']=='/root/memory_diagnosis' and value['actual_GPU_running'] is False
        and value['actual_owned_worker_pid'] is None,'Only actual idle own lease updated')
    value.update(updated_at_utc=now,status='memory_diagnostic_prelaunch_blocked_source_preparation',
        diagnostic_phase=stage,current_source_preparation=record(manifest),current_readiness_pointer=str(pointer))
    write_json(lease,value)
    print(json.dumps({'passed':True,'manifest':record(manifest),'binding_count':len(pins),
        'owned_AST_source_count':len(seen),'actual_controls':record(actual),'current_readiness':record(readiness)}),flush=True)


if __name__=='__main__':main()
