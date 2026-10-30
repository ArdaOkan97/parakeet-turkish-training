"""Final lightweight source-integration status, no ML/audio/runtime launch."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');E=A.parents[1]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require

def main():
    p=argparse.ArgumentParser();p.add_argument('--observed-config-tool0',required=True);p.add_argument('--owned-session',required=True,type=int);a=p.parse_args()
    require(a.observed_config_tool0=='dfc7b3' and a.owned_session==33486,'Own actually consumed metadata tool0, never inferred')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'E-only metadata environment')
    leasepath=P/'work/gpu_training_owner.json';lease=read_json(leasepath);require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Actual exclusive GPU owner idle')
    folder=A/'fullpopulation_firstwindow20_conv_configuration_CPU_actual_02';assembly=read_json(folder/'report.json');guardpath=A/'fullpopulation_firstwindow20_conv_config_guard_02/closure.json';guard=read_json(guardpath)
    require(assembly['passed'] is True and assembly['actual_native_or_CPU_frontend_jobs_launched']==0 and assembly['actual_optimizer_updates']==0 and assembly['actual_authoritative_draws_consumed']==0,'Metadata assembly only, never caller execution')
    require(guard['actual_exit_code']==0 and guard['monitor_joined'] is True and guard['monitor_clean'] is True and guard['observed']['monitor_error'] is None and guard['pid']==assembly['producer']['pid'],'Actual metadata report/guard PID-bound0')
    config=read_json(assembly['configuration']['path']);require(config['execute_window_limit']==1 and config['worker_driver_source']==str((P/'work/research/run_ASR18_full_DATA_multiwindow_v11.py').resolve()),'Exact new one-window source identity')
    import psutil
    require(not psutil.pid_exists(guard['pid']),'Owned final metadata child actually dead')
    stop=read_json(A/'fullpopulation_firstwindow20_conv_config_guard_01/resource_stop.json');require(stop['reason']=='CPU_timeout','Preserve observed60s metadata timeout failure')
    retirement=A/'CPU6_first_window_only_actual_01/retirement_after_proof_01/report.json';retired=read_json(retirement)
    require(retired['passed'] and retired['shared_feature_slot_empty'] and len(retired['removed_exact_owned_files'])==682,'Actual reviewed exact temporary proof retirement')
    cache=E/'temp/ASR18_CPU_feature_windows_v1';require(not list(cache.iterdir()),'Shared slot actually empty; no automatic cache adoption')
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=A/'memory_lifetime_focus_bounded_validation_final_summary_v2.json'
    report={'kind':'ASR18_memory_focus_caller_integration_prepared_not_executed_final_status_v3','owner':'/root/memory_diagnosis','observed_at_utc':stamp,
        'focus_deadline_utc':'2026-10-18T00:16:16Z','deadline_complete':False,'production_enabled':False,'whole_memory_fix_or_production_ready':False,
        'prior_focused_actual0_proofs_preserved':record(prior),'same_bounded_CNN_CPU6_fixed20_fullhard1_recovery_passes_remain_valid':True,
        'source_only_integration':{'parent11':record(P/'work/research/run_ASR18_full_DATA_multiwindow_v11.py'),'native_engine5':record(S/'asr18_full_DATA_native_window_v5.py'),
            'exact_validated_conv_source':config['validated_conv_only_source'],'transitive_source_manifest':config['source_manifest'],'bindings':225,
            'ROOT_source_review':record(A/'ROOT_fullpopulation_firstwindow20_conv_source_read_review_v1.json'),'ROOT_CPU_resource_rebind':record(A/'ROOT_reviewed_row_serial_CPU_resource_contract_v5.json'),
            'executed_window_limit':1,'planned_firstwindow_updates':20,'frozen_session_caps_remain2windows40':True,'no_scientific_sampler_journal_budget_or_numerical_policy_changes':True},
        'CPU6_proof_window_owned_retirement':{'owned_review':record(A/'ROOT_CPU6_successful_uncommitted_window_owned_retirement_review_v1.json'),'actual_tool_exit_code':0,'actual_tool_chunk_id':'04df49',
            'report':record(retirement),'files_removed':682,'bytes_retired':123560691,'archived_metadata':retired['preserved_metadata'],'source_audio_checkpoint_journal_untouched':True,'slot_currently_empty':True},
        'metadata_attempts':[{'owned_session':45772,'actual_tool_exit_code':1,'actual_tool_chunk_id':'6bc748','wall_ceiling_seconds':60,
            'stop':record(A/'fullpopulation_firstwindow20_conv_config_guard_01/resource_stop.json'),'reason':'CPU_timeout','private_peak_bytes':stop['observed']['peak_private_bytes'],'numerical_resource_miss':False},
            {'owned_session':a.owned_session,'actual_tool_exit_code':0,'actual_tool_chunk_id':a.observed_config_tool0,'wall_ceiling_seconds':120,'actual_wall_seconds':guard['wall_seconds'],
             'private_peak_bytes':guard['observed']['peak_private_bytes'],'guard0':record(guardpath),'report':record(folder/'report.json')}],
        'prepared_configuration':{k:assembly[k] for k in ('configuration','session','CPU_contract','provisional_original_first_plan')},
        'integration_runtime':{'actual_CPU_audio_frontend_jobs':0,'actual_GPU_processes':0,'actual_Adam_updates':0,'actual_authoritative_draws':0,'actual_fullcaller_checkpoint_or_recovery_proof':False,
            'reason_no_launch':'Metadata actual0 completed23:54:40Z after reviewed precise lateststart23:54:16Z; no late integration trial launched.',
            'full_configuration_execution_review_and_fresh_resource_admission_still_required':True,'next_wall_forecast_must_include_measured112s_preparation_validation':True},
        'next_scope':'No further jobs in this focus session. Future exact caller execution review/fresh guards then genuine full firstwindow20 native/sampler/checkpoint/freshrestore; multiwindow/phase/serialDEV/quality/retention/production remain unproved.',
        'actual_GPU_running':False,'actual_owned_GPU_PID':None,'all_owned_sessions_consumed':True,'no_further_experiments':True,'source':record(__file__)}
    target=A/'memory_lifetime_focus_final_source_integration_status_v3.json';require(not target.exists(),'Preserve immutable final successor');write_json(target,report)
    binary=S/'tools/backlog.exe';note=(f'{stamp}: Final source-integration status for the SAME memory-focus milestone: validated conv-only module is now installed before native in new fullpopulation parent11/engine5, with exact source review225pins and unchanged CPU/memory/scientific contracts. '
        'A one-window20 execution limit preserves real sampler/checkpoint/freshrestore flow; no production arm. The successful uncommitted CPU6 proof window was separately reviewed and retired archive-first (682exacttempfiles/123560691B), with640/320/20proof/source/audio/checkpoints/journals preserved and sharedslot empty; no auto-adoption. '
        'Metadata assembly60s attempt actually1/CPU_timeout at29.36MB; separately authorized120s successor actually0 in112.437s at41.35MB. Prepared exact configuration106b45c7/source225 remains NOT runtime-tested. '
        'Completion23:54:40Z missed lateststart23:54:16Z, so NOnewaudio/GPU/Adam/draw/native20 job was launched. Prior CNN/CPU6/fixed20+all3recoveries/fullhard1 passes remain valid; actualcaller runtime/multiwindow/serialDEV/retention/production readiness remain pending. '
        f'Final exact receipts: {target}.')
    edit=subprocess.run([str(binary),'task','edit','ASR-18','--append-notes',note,'--add-ref',str(target)],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    refresh=subprocess.run([str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(S/'refresh_backlog.py')],cwd=S,check=True,capture_output=True,text=True,encoding='utf-8')
    indices=[int(v.stem.rsplit('v',1)[1]) for v in A.glob('launch_readiness_v*.json') if v.stem.rsplit('v',1)[1].isdigit()];status=A/f'launch_readiness_v{max(indices)+1}.json'
    stage='GPU_idle_bounded_memory_proofs_passed_fullcaller_conv_source_config_prepared_runtime_unproved_no_late_trial'
    write_json(status,{'kind':'ASR18_memory_focus_final_idle_integration_status_v3','owner':lease['owner'],'updated_at_utc':stamp,'current_stage':stage,'actual_GPU_running':False,'actual_owned_GPU_PID':None,
        'actual_owned_session':None,'deadline_complete':False,'production_enabled':False,'whole_memory_fix_or_launch_ready':False,'final_source_integration_status':record(target),'prior_actual0_validation':record(prior)})
    pointerpath=A/'current_launch_readiness.json';pointer=read_json(pointerpath);pointer.update(updated_at_utc=stamp,current_readiness=record(status),current_stage=stage,actual_owned_session=None,
        actual_GPU_running=False,actual_owned_GPU_PID=None,current_CPU_metadata_child=None,current_run_folder=str(folder),consolidated_memory_focus_summary=record(target))
    write_json(pointerpath,pointer);lease.update(updated_at_utc=stamp,status=stage,diagnostic_phase=stage,actual_GPU_running=False,actual_owned_worker_pid=None,actual_training_optimizer_running=False,current_evidence=record(target),production_enabled=False)
    write_json(leasepath,lease)
    receipt=A/'memory_focus_final_source_integration_Backlog_refresh_v3.json';write_json(receipt,{'summary':record(target),'status':record(status),'BacklogCLI_actual_exit_code':edit.returncode,'refresh_actual_exit_code':refresh.returncode,
        'refresh_stdout':refresh.stdout.strip(),'new_milestone_created':False,'same_existing_ASR18_memory_focus_milestone_updated':True,'source':record(__file__)})
    print(json.dumps({'summary':record(target),'status':record(status),'BacklogCLI_refresh_receipt':record(receipt),'BacklogCLI_actual_exit_code':edit.returncode,'refresh_actual_exit_code':refresh.returncode}))

if __name__=='__main__':main()
