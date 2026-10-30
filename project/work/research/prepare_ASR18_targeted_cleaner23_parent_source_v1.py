"""Derive NEW one-window cleaner parent; reuse frozen process/resource helpers."""
import ast,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research'
source=R/'run_ASR18_full_DATA_multiwindow_v11.py';target=R/'run_ASR18_targeted_cleaner23_v1.py'
assert not target.exists() and hashlib.sha256(source.read_bytes()).hexdigest()=='5f8ec1a953430368b15b4b094e52007bfb9edeec32f6282c8e284b9cda232d91'
base=source.read_text(encoding='utf-8-sig')
CPU=base[base.index('        CPU_review = read_bound'):base.index("        require(session['initial_sampler_sha256']")]
recover=base[base.index("        request_path=folder/'all_actual_native_checkpoints_pending_recovery.json'"):base.index("        result = {'passed': True",base.index("        request_path=folder/'all_actual_native_checkpoints_pending_recovery.json'"))]
failure=base[base.index('    except BaseException:\n        write_json(folder/\'failure.json\''):base.index("\n\nif __name__ == '__main__':")]
controller='''def controller(config_path, review_path, folder):
    storage();config=read_json(config_path);check_bindings(config)
    require(config['kind']=='ASR18_targeted_cleaner23_diagnostic_config_v1'
            and config['finite_diagnostic_execution_enabled'] is True and config['TEMPLATE_NOT_LAUNCHABLE'] is False
            and config['production_enabled'] is False and config['diagnostic_only'] is True
            and config['maximum_windows']==config['execute_window_limit']==config['maximum_updates_per_window']==1
            and config['maximum_total_native_updates']==23
            and config['worker_driver_source']==str(Path(__file__).resolve()) and config['parent_driver_source']==record(__file__),
            'Exact one NEW targeted cleaner23 diagnostic; no production phase transition')
    from asr18_separated_worker_forecast_v1 import restore_only_worker_forecast
    require(config['worker_mode']=='resident_training' and config['all_captured_checkpoints_require_separate_final_GPU_recovery'] is True
            and config['entire_worker_forecast']['native_private_forecast_bytes']==5_560_813_440
            and config['entire_worker_forecast']['checkpoint_write_increment_bytes']==64_000_000
            and config['entire_worker_forecast']['entire_worker_private_forecast_bytes']==5_624_813_440
            and digest(config['final_recovery_worker_forecast'])==digest(restore_only_worker_forecast()),
            'No new numerical policy, smaller native/write/restore margin or borrowed component credit')
    review=read_json(review_path)
    require(review['accepted'] is True and review['scope']=='finite_targeted_cleaner23_CPU6_native_and_fresh_recovery_diagnostic_v1'
            and review['production_approval_claimed'] is False and review['source_manifest']==config['source_manifest']
            and review['configuration']==record(config_path),'Exact new cleaner source/resource/configuration review')
    require(not folder.exists(),'Fresh one-window cleaner namespace');folder.mkdir()
    write_json(folder/'launch.json',{'config':record(config_path),'review':record(review_path),'production_enabled':False})
    parent=ParentGuard(folder);gpu=None;actuals=[];windows=[];selected=[];began=time.monotonic()
    from fastmss_phase_aware_plan_bridge_v5 import bridge_plan,read_bound
    from fastmss_owned_process_closure_v2 import verify_resident_idle,verify as verify_process
    from fastmss_row_serial_cpu_v6 import verify_raw_review
    try:
        from asr18_targeted_cleaner_schedule_context_v1 import CleanerContext
        context=CleanerContext(config['planning_arguments'],config['cleaner_ancestor'],guard=parent.check)
        try:
            initial=context.new_schedule();first=context.plan(initial,maximum_updates=1)
            require(initial.committed_updates()==22 and initial.cleaner.draws==0
                    and digest(initial.state_dict())==config['initial_cleaner_diagnostic_sampler_sha256'],
                    'Exact metadata-derived diagnostic state retains untouched actual22 ancestry')
        finally:context.close()
        require(digest(first)==digest(read_bound(config['provisional_cleaner_plan'],parent.check)),
                'Recomputed new cleaner plan differs; no consumed source/replay draw')
        session=read_bound(config['session'],parent.check);CPU_contract=read_bound(session['CPU_preparation_contract'],parent.check)
''' + CPU + '''        require(session['initial_optimizer_updates']==22 and session['ancestor']==config['cleaner_ancestor']
                and session['initial_sampler_sha256']==first['initial_sampler_sha256']
                and len(first['updates'])==1 and first['initial_committed_updates']==22,
                'Actual22 source lineage and one diagnostic cleaner23 update')
        conditional=read_bound(config['cleaner_conditional_CPU_resource_review'],parent.check)
        require(conditional['kind']=='ASR18_ROOT_cleaner_CPU6_conditional_resource_component_read_review_v1'
                and conditional['reviewer']=='/root' and conditional['accepted'] is True
                and conditional['source']==record(SOURCE/'fastmss_row_serial_cpu_v6.py')
                and conditional['unchanged_CPU_family_allowance_bytes']==config['CPU_owned_allowance_bytes'],
                'Exact conditional header-before-allocation CPU6 cleaner component review')
        plan_path=folder/'coordinator_plan_0.json';write_json(plan_path,first);plan_pin=record(plan_path)
        bridge=bridge_plan(plan_pin,DATA_evidence=config['planning_arguments']['DATA_evidence'],session_binding=config['session'],
            window_index=0,prior_window_boundary=None,guard=parent.check)
        bridge_path=folder/'bridge_0.json';write_json(bridge_path,bridge);bridge_arg=folder/'bridge_0_binding.json';write_json(bridge_arg,record(bridge_path))
        CPU_folder=folder/'CPU_window_0'
        CPU0=run_CPU(config,config['CPU_controller_source']['path'],['--bridge',str(bridge_arg),'--output',str(CPU_folder/'payload')],
            CPU_folder,guard=parent.check,report_relative='payload/report.json')
        actuals.append(CPU0);CPU_report=read_bound(read_bound(CPU0,parent.check)['report'],parent.check)
        require(CPU_report['passed'] is True and CPU_report['bridge']==record(bridge_path)
                and CPU_report['all_children_actually_closed0'] is True,'Actual NEW CPU6 clean reference/frontend closed0 required')
        manifest=read_bound(CPU_report['manifest'],parent.check);prepared=read_bound(manifest['updates'][0]['update'],parent.check)
        require(prepared['prepared_payload']['protection_kind']=='FastMSS_explicit_clean_phase_protection_v1'
                and len(prepared['rows'])==4
                and all(row['augmentation']['applied'] is False for row in prepared['prepared_payload']['rows']),
                'Both real cleaner rows and carried replay stay clean; no augmenter call')
        child_config={**config,'restore_parent_pointer':config['cleaner_ancestor']['checkpoint_pointer'],
            'first_feature_update':manifest['updates'][0]['update']}
        child_path=folder/'GPU_config_0.json';write_json(child_path,child_config)
        gpu=GPUService.launch(child_path,folder/'GPU_process_0',parent)
        while not (gpu.folder/'worker_ready.json').exists():gpu.check();time.sleep(.1)
        require(read_json(gpu.folder/'worker_ready.json')['optimizer_updates']==22,'Actual complete22 restore before any cleaner forward')
        response=gpu.command('run_prepared_window',window={'coordinator_plan':plan_pin,'feature_owner':CPU_report['owner'],
            'feature_manifest':CPU_report['manifest'],'session':config['session']})
        idle=gpu.seal_idle(response);complete=read_bound(response['window_completion'],parent.check);windows.append(complete)
        pointer_pin=complete['latest_pointer'];pointer=read_bound(pointer_pin,parent.check)
        require(complete['passed'] is True and complete['targeted_cleaner23'] is True and complete['actual_new_updates']==1
                and complete['global_new_updates']==pointer['committed_updates']==23
                and complete['all_checkpoint_persistence_verified'] is True and complete['exact_GPU_restores_deferred'] is True
                and complete['production_phase_advance_claimed'] is False,'Actual NEW cleaner23/PREsave/persistence; recovery pending')
        verify_resident_idle(read_bound(idle,parent.check),pointer=pointer,manifest=CPU_report['manifest'],phase=first['phase'],guard=parent.check,proof_binding=idle)
        native0=gpu.finish();actuals.append(native0);gpu=None
        cleanup_boundary={'kind':'coherent_checkpoint_boundary','owner_sha256':CPU_report['owner']['sha256'],
            'session':config['session'],'phase':first['phase'],'committed_window_updates':1,'window_manifest_sha256':CPU_report['manifest']['sha256'],
            'coherent_checkpoint_pointer':pointer_pin,'checkpoint':pointer['checkpoint'],
            'deferred_restore_required_before_overall_diagnostic_pass':True,'actual_native_closed':native0}
        cleanup_request=folder/'cleanup_request_0.json';write_json(cleanup_request,{'owner':CPU_report['owner'],'manifest':CPU_report['manifest'],
            'closures':CPU_report['closures'],'boundary':cleanup_boundary,'archive_path':str(folder/'cleaner_window_durable_feature_metadata.json'),
            'parent_source':record(__file__)})
        cleanup_folder=folder/'cleanup_0'
        cleanup0=run_CPU({**config,'maximum_CPU_phase_seconds':config['maximum_cleanup_phase_seconds']},Path(__file__).resolve(),
            ['--cleanup','--request',str(cleanup_request),'--folder',str(cleanup_folder)],cleanup_folder,guard=parent.check)
        actuals.append(cleanup0);verify_process(read_bound(native0,parent.check),kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,guard=parent.check)
        native_report=read_bound(read_bound(native0,parent.check)['report'],parent.check)
        require(native_report['windows']==[response['window_completion']] and native_report['actual_native_updates_this_process']==1,
                'One actual closed native report names the exact single diagnostic window')
        training_processes=[native0];checkpoints=complete['checkpoints']
        request={'kind':'ASR18_targeted_cleaner23_all_new_checkpoint_GPU_recovery_requests_v1','owner':OWNER,
            'production_enabled':False,'session':config['session'],'parent_source':record(__file__),
            'native_engine_source':record(SOURCE/'asr18_targeted_cleaner23_native_v1.py'),
            'native_process_completions':training_processes,'native_window_completions':native_report['windows'],
            'initial_global_updates':22,'expected_new_updates':1,'expected_global_new_updates':23,
            'ancestor':config['cleaner_ancestor'],'checkpoints':checkpoints}
''' + recover + '''        result={'kind':'ASR18_actual_targeted_cleaner23_CPU6_native_fresh_recovery_diagnostic_v1','passed':True,
            'ancestor':config['cleaner_ancestor'],'initial_inherited_global_updates':22,'actual_native_updates':1,
            'actual_final_global_updates':23,'final_mature_clocks':[64273,2023,62273],'windows':windows,
            'actual_process_completions':actuals,'actual_CPU_process0':CPU0,'actual_native_process0':native0,'actual_cleanup_process0':cleanup0,
            'actual_recovery_process0':recovery0,'all_captured_checkpoints_actual_fresh_GPU_recovery':recovery_response['recovery_report'],
            'all_deferred_recovery_obligations_discharged':True,'real_cleaner_ordinals':[1,352768],
            'actual_languages':complete['checkpoints'][-1].get('actual_languages',None),
            'cleaner_population_mean_U':11972886/352768,'production_phase_advance_claimed':False,
            'original22_fullpopulation_schedule_retained_as_immutable_ancestry':True,
            'CPU_clean_reference_frontend_header_raw_lifetime_path_actually_exercised':True,
            'zero_augmentation_protection_verified':True,'actual_next_carried_replay_preserved':True,
            'current_checkpoint_pointer':pointer_pin,'wall_seconds':time.monotonic()-began,
            'production_enabled':False,'diagnostic_only':True,'qualified_checkpoint':False,'full_broad_passes_completed':0}
        write_json(folder/'report.json',result);print(json.dumps(result),flush=True)
''' + failure
start=base.index('def controller(');end=base.index("\n\nif __name__ == '__main__':",start)
text=base[:start]+controller+base[end:]
for old,new in (
    ('asr18_full_DATA_native_window_v5','asr18_targeted_cleaner23_native_v1'),
    ('asr18_deferred_GPU_recovery_v1','asr18_targeted_cleaner23_recovery_v1'),
    ('asr18_restore_only_runtime_v1','asr18_targeted_cleaner23_restore_runtime_v1')):
    assert old in text,old;text=text.replace(old,new)
ast.parse(text,filename=str(target))
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print(str(target),len(target.read_bytes()),hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
