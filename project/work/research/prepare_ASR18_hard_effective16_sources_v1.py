"""Emit isolated hard-effective16 engine/parent/recovery source, no target import."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
def read(path):return path.read_text(encoding='utf-8-sig')
def replacements(text,items):
    for old,new in items:
        assert old in text,old;text=text.replace(old,new)
    return text
def emit(path,text):
    ast.parse(text,filename=str(path));assert not path.exists()
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    print(str(path),flush=True)

native=read(S/'asr18_full_DATA_native_window_v5.py')
native=replacements(native,[
    ('from bpe_final_DATA_schedule_context_v1 import ScheduleContext, normalize_schedule_state',
     'from asr18_hard_envelope_schedule_context_v1 import HardEnvelopeContext, normalize_diagnostic_state as normalize_schedule_state'),
    ("self.context = ScheduleContext(config['planning_arguments'], guard=guard)",
     "self.context = HardEnvelopeContext(config['planning_arguments'],config['hard_row_provenance'],guard=guard)\n        require(config['maximum_windows']==config['maximum_total_native_updates']==config['maximum_updates_per_window']==1,'One exact hard effective16 diagnostic')"),
    ('asr18_calibrated_geometry_forecast_v6','asr18_full_population_geometry_forecast_v1'),
    ('asr18_separated_worker_forecast_v1','asr18_full_population_separated_forecast_v1'),
    ('fastmss_phase_aware_feature_window_v4','fastmss_phase_aware_feature_window_v6'),
    ('from asr18_checkpoint_next_draw_v1 import next_draw_packet',
     'from asr18_hard_envelope_schedule_context_v1 import completed_diagnostic_next_draw_packet as next_draw_packet'),
    ("'kind':'ASR18_phase_separated_bounded_window_caller_identity_v2',",
     "'kind':'ASR18_hard_effective16_original_source_diagnostic_caller_identity_v1',\n            'hard_row_provenance':self.config['hard_row_provenance'],'production_source_eligible':False,"),
    ("'window_index': self.window_count-1, 'actual_new_updates': len(plan['updates']),",
     "'hard_envelope_effective16_diagnostic':True,'production_phase_advance_claimed':False,\n            'window_index': self.window_count-1, 'actual_new_updates': len(plan['updates']),")])
start=native.index('    def restore_parent(');end=native.index('    def _checkpoint(',start)
native=native[:start]+"    def restore_parent(self, pointer_binding, *, augmentation_identity):\n        raise ValueError('Hard control starts from original207/threeAdam/RNG only; no diagnostic checkpoint promotion')\n\n"+native[end:]
emit(S/'asr18_hard_effective16_native_v1.py',native)

recovery=read(S/'asr18_deferred_GPU_recovery_v1.py')
recovery=replacements(recovery,[
    ('from bpe_final_DATA_schedule_context_v1 import normalize_schedule_state','from asr18_hard_envelope_schedule_context_v1 import normalize_diagnostic_state as normalize_schedule_state'),
    ('from asr18_checkpoint_next_draw_v1 import next_draw_packet','from asr18_hard_envelope_schedule_context_v1 import completed_diagnostic_next_draw_packet as next_draw_packet'),
    ('fastmss_phase_aware_plan_bridge_v5','fastmss_phase_aware_plan_bridge_v7'),
    ('ASR18_deferred_all_checkpoint_GPU_recovery_requests_v1','ASR18_hard_effective16_all_checkpoint_GPU_recovery_requests_v1'),
    ("1 <= value['expected_global_new_updates'] <= 40","value['expected_global_new_updates'] == 1"),
    ("1 <= len(value['native_process_completions']) <= 2","len(value['native_process_completions']) == 1"),
    ("1 <= len(value['native_window_completions']) <= 2","len(value['native_window_completions']) == 1"),
    ("require(window['passed'] is True and window['exact_GPU_restores_deferred'] is True",
     "require(window['passed'] is True and window['hard_envelope_effective16_diagnostic'] is True\n                and window['production_phase_advance_claimed'] is False and window['exact_GPU_restores_deferred'] is True"),
    ('1 <= len(checkpoints) <= 6','len(checkpoints) == 1'),
    ("and caller['session'] == session and caller['engine'] == value['native_engine_source']",
     "and caller['session'] == session and caller['engine'] == value['native_engine_source']\n                and caller['hard_row_provenance']==value['hard_row_provenance'] and caller['production_source_eligible'] is False"),
    ('indices == sorted(set(indices)) and indices[-1] == actual_updates','indices == [1] and actual_updates == 1'),
    ('ASR18_actual_all_captured_checkpoints_fresh_GPU_recovery_v1','ASR18_actual_hard_effective16_checkpoint_fresh_GPU_recovery_v1')])
emit(S/'asr18_hard_effective16_recovery_v1.py',recovery)
runtime=read(S/'asr18_restore_only_runtime_v1.py')
runtime=replacements(runtime,[
    ('asr18_deferred_GPU_recovery_v1','asr18_hard_effective16_recovery_v1'),
    ('from bpe_final_DATA_schedule_context_v1 import ScheduleContext','from asr18_hard_envelope_schedule_context_v1 import HardEnvelopeContext'),
    ("context=ScheduleContext(config['planning_arguments'],guard=guard)","context=HardEnvelopeContext(config['planning_arguments'],config['hard_row_provenance'],guard=guard)")])
emit(S/'asr18_hard_effective16_restore_runtime_v1.py',runtime)

parent=read(R/'run_ASR18_targeted_cleaner23_v2.py')
prefix=parent[:parent.index('def controller(')]
prefix=replacements(prefix,[
    ('asr18_targeted_cleaner23_native_v2','asr18_hard_effective16_native_v1'),
    ('asr18_targeted_cleaner23_recovery_v2','asr18_hard_effective16_recovery_v1'),
    ('asr18_calibrated_geometry_forecast_v6','asr18_full_population_geometry_forecast_v1'),
    ('asr18_separated_worker_forecast_v1','asr18_full_population_separated_forecast_v1'),
    ('fastmss_phase_aware_feature_window_v5','fastmss_phase_aware_feature_window_v6')])
controller='''def controller(config_path,review_path,folder):
    storage();config=read_json(config_path);check_bindings(config)
    require(config['kind']=='ASR18_hard_effective16_CPU8_native_diagnostic_config_v1'
            and config['finite_diagnostic_execution_enabled'] is True and config['TEMPLATE_NOT_LAUNCHABLE'] is False
            and config['production_enabled'] is False and config['diagnostic_only'] is True
            and config['maximum_windows']==config['execute_window_limit']==config['maximum_updates_per_window']==config['maximum_total_native_updates']==1
            and config['worker_driver_source']==str(Path(__file__).resolve()) and config['parent_driver_source']==record(__file__),
            'One exact original-source hard effective16 diagnostic, never production')
    from asr18_full_population_separated_forecast_v1 import restore_only_worker_forecast
    require(config['worker_mode']=='resident_training' and config['all_captured_checkpoints_require_separate_final_GPU_recovery'] is True
            and config['entire_worker_forecast']['native_private_forecast_bytes']==6_684_646_272
            and config['entire_worker_forecast']['checkpoint_write_increment_bytes']==64_000_000
            and config['entire_worker_forecast']['entire_worker_private_forecast_bytes']==6_748_646_272
            and digest(config['final_recovery_worker_forecast'])==digest(restore_only_worker_forecast()),'Conservative full hard forecast; no component/private credit')
    review=read_json(review_path)
    require(review['accepted'] is True and review['scope']=='finite_hard_effective16_CPU8_native_fresh_recovery_diagnostic_v1'
            and review['production_approval_claimed'] is False and review['source_manifest']==config['source_manifest']
            and review['configuration']==record(config_path),'Exact ROOT source/resource/configuration review')
    require(not folder.exists(),'Fresh hard effective16 namespace');folder.mkdir()
    write_json(folder/'launch.json',{'config':record(config_path),'review':record(review_path),'production_enabled':False})
    parent=ParentGuard(folder);gpu=None;actuals=[];windows=[];began=time.monotonic()
    from fastmss_phase_aware_plan_bridge_v7 import bridge_plan,read_bound
    from fastmss_owned_process_closure_v2 import verify_resident_idle,verify as verify_process
    from fastmss_row_serial_cpu_v8 import verify_raw_review,verify_contract
    try:
        from asr18_hard_envelope_schedule_context_v1 import HardEnvelopeContext
        context=HardEnvelopeContext(config['planning_arguments'],config['hard_row_provenance'],guard=parent.check)
        try:
            initial=context.new_schedule();plan=context.plan(initial,maximum_updates=1)
            plan['hard_envelope_diagnostic_context']=config['hard_envelope_diagnostic_context']
            require(initial.committed_updates()==0 and initial.broad.draws==0 and digest(initial.state_dict())==config['initial_sampler_sha256'],
                    'Original0 schedule/RNG and honest explicit hard diagnostic cursor')
        finally:context.close()
        require(digest(plan)==digest(read_bound(config['provisional_plan'],parent.check)),'Exact pre-audio metadata-bridged plan')
        session=read_bound(config['session'],parent.check);contract=verify_contract(session['CPU_preparation_contract'],session,guard=parent.check)
        CPU_review=read_bound(config['actual_ROOT_CPU_resource_review'],parent.check)
        require(CPU_review['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and CPU_review['reviewer']=='/root'
                and CPU_review['accepted'] is True and CPU_review['scope']=='CPU_preparation_only'
                and CPU_review['outer_parent_source']==record(__file__) and CPU_review['CPU_source_manifest']==check_bindings(config)['CPU_owner_bundle']
                and CPU_review['production_enabled'] is False,'Exact new CPU8 source/resource binding')
        verify_raw_review(contract['raw_lifetime_review'])
        require(contract['actual_ROOT_CPU_resource_review']==config['actual_ROOT_CPU_resource_review']
                and digest(contract['resource_review'])==digest(CPU_review['resource_review'])
                and digest(contract['raw_lifetime_review'])==digest(CPU_review['raw_lifetime_review'])
                and config['CPU_owned_allowance_bytes']==1_934_217_728
                and config['CPU_owned_allowance_bytes']==contract['resource_review']['whole_owned_private_allowance_bytes']+contract['resource_review']['outer_controller_launcher_reserve_bytes']
                and contract['outer_parent_source']==record(__file__), 'No changed CPU/raw/margin allowance')
        require(session['mode']=='one_hard_envelope_diagnostic' and session['initial_optimizer_updates']==0
                and session['hard_envelope_diagnostic_context']==config['hard_envelope_diagnostic_context']
                and session['initial_sampler_sha256']==plan['initial_sampler_sha256'],'One unqualified hard diagnostic session')
        pp=folder/'coordinator_plan_0.json';write_json(pp,plan)
        bridge=bridge_plan(record(pp),DATA_evidence=config['planning_arguments']['DATA_evidence'],session_binding=config['session'],
            window_index=0,prior_window_boundary=None,guard=parent.check)
        bp=folder/'bridge_0.json';write_json(bp,bridge);ba=folder/'bridge_0_binding.json';write_json(ba,record(bp))
        cpu_folder=folder/'CPU_window_0'
        CPU0=run_CPU(config,config['CPU_controller_source']['path'],['--bridge',str(ba),'--output',str(cpu_folder/'payload')],
            cpu_folder,guard=parent.check,report_relative='payload/report.json');actuals.append(CPU0)
        CPU=read_json(cpu_folder/'payload/report.json');require(CPU['passed'] is True and CPU['all_children_actually_closed0'] is True
            and CPU['bridge']==record(bp),'Actual complete CPU8 independent references/frontend/finalization0')
        manifest=read_bound(CPU['manifest'],parent.check);prepared=read_bound(manifest['updates'][0]['update'],parent.check)
        hard=prepared['rows'][13]
        require(len(prepared['rows'])==16 and hard['source_id']=='kiraat:src01792-00990' and hard['encoder_frames']==1991
                and len(hard['full_target_ids'])==574 and hard['samples']==2547840
                and hard['input_features']['shape']==[1,15925,128]
                and prepared['prepared_payload']['rows'][13]['augmentation']['applied'] is False,
                'Complete hard original audio/targets/physical frames clean over120s plus13TR/exact2replay')
        child={**config,'restore_parent_pointer':None,'first_feature_update':manifest['updates'][0]['update']}
        cp=folder/'GPU_config_0.json';write_json(cp,child)
        gpu=GPUService.launch(cp,folder/'GPU_process_0',parent)
        while not(gpu.folder/'worker_ready.json').exists():gpu.check();time.sleep(.1)
        response=gpu.command('run_prepared_window',window={'coordinator_plan':record(pp),'feature_owner':CPU['owner'],
            'feature_manifest':CPU['manifest'],'session':config['session']})
        idle=gpu.seal_idle(response);complete=read_bound(response['window_completion'],parent.check);windows.append(complete)
        pointer_pin=complete['latest_pointer'];pointer=read_bound(pointer_pin,parent.check)
        require(complete['actual_new_updates']==complete['global_new_updates']==pointer['committed_updates']==1
                and complete['hard_envelope_effective16_diagnostic'] is True and complete['all_checkpoint_persistence_verified'] is True
                and complete['exact_GPU_restores_deferred'] is True,'Actual one effective16 commit/persistence, not overall recovery')
        verify_resident_idle(read_bound(idle,parent.check),pointer=pointer,manifest=CPU['manifest'],phase=plan['phase'],guard=parent.check,proof_binding=idle)
        native0=gpu.finish();actuals.append(native0);gpu=None
        boundary={'kind':'coherent_checkpoint_boundary','owner_sha256':CPU['owner']['sha256'],'session':config['session'],
            'phase':plan['phase'],'committed_window_updates':1,'window_manifest_sha256':CPU['manifest']['sha256'],
            'coherent_checkpoint_pointer':pointer_pin,'checkpoint':pointer['checkpoint'],
            'deferred_restore_required_before_overall_diagnostic_pass':True,'actual_native_closed':native0}
        cr=folder/'cleanup_request_0.json';write_json(cr,{'owner':CPU['owner'],'manifest':CPU['manifest'],'closures':CPU['closures'],
            'boundary':boundary,'archive_path':str(folder/'window_0_durable_feature_metadata.json'),'parent_source':record(__file__)})
        cleanup_folder=folder/'cleanup_0';cleanup0=run_CPU({**config,'maximum_CPU_phase_seconds':config['maximum_cleanup_phase_seconds']},
            Path(__file__).resolve(),['--cleanup','--request',str(cr),'--folder',str(cleanup_folder)],cleanup_folder,guard=parent.check)
        actuals.append(cleanup0)
        request={'kind':'ASR18_hard_effective16_all_checkpoint_GPU_recovery_requests_v1','owner':OWNER,'production_enabled':False,
            'session':config['session'],'parent_source':record(__file__),'native_engine_source':record(SOURCE/'asr18_hard_effective16_native_v1.py'),
            'expected_global_new_updates':1,'native_process_completions':[native0],'native_window_completions':[response['window_completion']],
            'checkpoints':complete['checkpoints'],'hard_row_provenance':config['hard_row_provenance']}
        rp=folder/'all_actual_native_checkpoints_pending_recovery.json';write_json(rp,request)
        recovery_config={**config,'worker_mode':'restore_only','deferred_recovery_requests':record(rp),
            'entire_worker_forecast':restore_only_worker_forecast(),'maximum_GPU_child_wall_seconds':config['maximum_final_recovery_seconds']}
        rc=folder/'GPU_final_recovery_config.json';write_json(rc,recovery_config)
        gpu=GPUService.launch(rc,folder/'GPU_recovery_process',parent)
        while not(gpu.folder/'worker_ready.json').exists():gpu.check();time.sleep(.1)
        recovered=gpu.command('restore_all_deferred_checkpoints',window={'request':record(rp)})
        all_recovery=read_bound(recovered['recovery_report'],parent.check)
        require(all_recovery['passed'] is True and all_recovery['all_captured_checkpoints_recovered'] is True
                and all_recovery['actual_native_updates_in_recovery_process']==0 and all_recovery['actual_draws_consumed_in_recovery']==0
                and all_recovery['actual_global_native_updates']==1 and len(all_recovery['recovered_checkpoints'])==1,'Every new hard capture exact fresh recovery')
        recovery0=gpu.finish();actuals.append(recovery0);gpu=None
        for proof_pin in actuals:
            proof=read_bound(proof_pin,parent.check);verify_process(proof,kind=proof['kind'],owner=OWNER,guard=parent.check)
        result={'kind':'ASR18_actual_hard_effective16_CPU8_native_fresh_recovery_diagnostic_v1','passed':True,
            'actual_native_updates':1,'actual_global_updates':1,'final_mature_clocks':[64251,2001,62251],
            'hard_original_ordinal':1051133,'hard_native_microbatch_index':14,'hard_T':1991,'hard_U':574,'hard_samples':2547840,
            'windows':windows,'actual_process_completions':actuals,'actual_CPU_process0':CPU0,'actual_native_process0':native0,
            'actual_cleanup_process0':cleanup0,'actual_recovery_process0':recovery0,'all_captured_checkpoints_actual_fresh_GPU_recovery':recovered['recovery_report'],
            'all_deferred_recovery_obligations_discharged':True,'current_checkpoint_pointer':pointer_pin,
            'actual_languages':[row['language'] for row in prepared['rows']],'hard_row_provenance':config['hard_row_provenance'],
            'full_original_source_schedule_retained_as_immutable_ancestry':True,'production_phase_advance_claimed':False,
            'production_enabled':False,'diagnostic_only':True,'qualified_checkpoint':False,'wall_seconds':time.monotonic()-began}
        write_json(folder/'report.json',result);print(json.dumps({'passed':True,'actual_updates':1,'report':record(folder/'report.json')}),flush=True)
    except BaseException:
        write_json(folder/'failure.json',{'passed':False,'error':traceback.format_exc(),'completed_windows':windows,
            'actual_process_completions':actuals,'actual_native_progress':read_json(gpu.folder/'native_progress.json')
                if gpu is not None and(gpu.folder/'native_progress.json').exists() else None,
            'production_enabled':False,'overall_diagnostic_success_claimed':False})
        raise
    finally:
        if gpu is not None:gpu.abort()
        parent.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--review');p.add_argument('--folder',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--recovery-worker',action='store_true');p.add_argument('--cleanup',action='store_true');p.add_argument('--request')
    a=p.parse_args();folder=scoped(a.folder)
    if a.worker:worker(a.config,folder)
    elif a.recovery_worker:
        from asr18_hard_effective16_restore_runtime_v1 import run
        run(a.config,folder,parent_source=record(__file__),check_bindings=check_bindings,storage=storage,stage=stage,admit_stage=admit_stage)
    elif a.cleanup:cleanup_child(a.request,folder)
    else:controller(a.config,a.review,folder)
'''
emit(R/'run_ASR18_hard_effective16_v1.py',prefix+controller)
