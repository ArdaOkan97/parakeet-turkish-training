"""Seal the observed finite cleaner21 tool/native exit and scoped evidence."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import read_json,write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def main():
    folder=A/'cleaner_precomputed_after20_01';report=read_json(folder/'report.json');c=report['completion'];lease=read_json(P/'work/gpu_training_owner.json')
    require(report['actual_exit_code']==0 and report['passed'] and report['stop_reason'] is None and report['PID']==36104 and report['launcher_PID']==31788
        and c['native_updates_this_child']==1 and c['coherent_chain_updates']==21 and c['final_clocks']==[64271,2021,62271] and c['sources_unchanged'],
        'Observed successful exact cleaner21 prerequisite')
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Actual GPU child closed and lease idle')
    saved=c['checkpoint'];metadata=read_json(saved['path']);require(record(saved['path'])==saved and record(metadata['payload']['path'])==metadata['payload'],'Completed streamed state/payload hashes required')
    coverage={};files=('actual_forecast_coverage_21.json','checkpoint21_write_memory.json','checkpoint21_restore_memory.json')
    for name in files:
        value=read_json(folder/name);require(value['passed'],'All actual native/write/restore forecasts must pass')
        if name==files[0]:before=value['before_private_bytes'];increment=value['forecast']['incremental_bytes'];peak=value['actual_native_peak_bytes'];phase=c['native_memory']
        else:before=value['before'];increment=value['incremental_budget_bytes'];phase=value['phase'];peak=phase['private_peak_bytes']
        require(phase['error'] is None and phase['joined'] and phase['available_sample_count']>=3,'Clean available native phase monitor closure')
        coverage[name]={'before_private_bytes':before,'incremental_forecast_bytes':increment,'total_private_forecast_bytes':before+increment,
            'actual_private_peak_bytes':peak,'remaining_margin_bytes':before+increment-peak,'valid50ms_samples':phase['available_sample_count'],'evidence':record(folder/name)}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();actual=folder/'actual_owned_CLOSED0.json';require(not actual.exists(),'Preserve owned completion')
    write_json(actual,{'kind':'ASR18_actual_owned_GPU_finite_cleaner_CLOSED0_v1','owner':'/root/memory_diagnosis','owned_session':46439,'native_PID':36104,'launcher_PID':31788,
        'actual_native_exit_code':0,'actual_tool_exit_code':0,'actual_tool_chunk_id':'d97b18','observed_at_utc':now,'report':record(folder/'report.json'),
        'actual_native_updates_this_child':1,'coherent_chain_updates':21,'clocks':[64271,2021,62271],'diagnostic_only':True,'qualified_checkpoint':False})
    samples=report['samples'];private=max(v['worker_private_memory']['private_bytes'] for v in samples if v['worker_private_memory']['available'])
    global_peak=max(v['system_memory']['commit_fraction'] for v in samples if v['system_memory']['available'])
    first=datetime.datetime.fromisoformat(samples[0]['observed_at_utc']);last=datetime.datetime.fromisoformat(samples[-1]['observed_at_utc'])
    value={'kind':'ASR18_actual_cleaner21_scoped_summary_v1','passed':True,'created_at_utc':now,'owner':'/root/memory_diagnosis','actual_owned_exit':record(actual),
        'actual_native_updates_this_child':1,'coherent_chain_updates':21,'actual_TR_rows':2,'actual_languages':c['actual_languages'],'actual_augmentation_applied':0,
        'TR_ordinals':[1,352768],'full_cleaner_population_mean_U':11972886/352768,'clocks':[64271,2021,62271],
        'native50ms_private_peak_bytes':c['native_memory']['private_peak_bytes'],'parent1s_private_peak_bytes':private,'parent1s_global_commit_peak_fraction':global_peak,
        'native_seconds':c['native_memory']['wall_seconds'],'parent_observed_first_to_last_sample_seconds':(last-first).total_seconds(),'parent_samples':len(samples),
        'forecast_phases':coverage,'all_native_write_restore_forecasts_covered':True,'forecast_production_approval_claimed':False,
        'all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_exact':c['all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_exact'],
        'original20_and_next_exact_source_draw_restored':c['original20_source_and_sampler_restored_before_next_exactdraw'],
        'actual_cleaner_replay_prefix_restored':c['actual_cleaner_cursor_carried_replay_and_journal_prefix_restored'],
        'clean_feature_payload_before_after_restore_exact':c['clean_feature_payload_exact_before_after_restore'],
        'checkpoint':saved,'tensor_payload':metadata['payload'],'parent_checkpoint':read_json(folder/'declaration.json')['parent_checkpoint'],
        'ROOT_review':record(A/'ROOT_finite_cleaner_update21_source_read_review_v1.json'),'full_forecast_prelaunch_admission':record(A/'cleaner1_prelaunch_full_forecast_01.json'),
        'original_clean_CPU_handoff':record(F/'FastMSS_actual_clean_feature_CPU_handoff_v1.json'),'actual_feature_window_retained':True,
        'parent_frontend_or_augmentation_executed':False,'parent_imported_modules':c['parent_imported_modules'],'librosa_import_does_not_mean_frontend_execution':True,
        'diagnostic_only':True,'production_ready':False,'full_broad_corpus_passes':0,'production_cleaner_phase_transition_claimed':False,
        'DEV_or_quality_improvement_measured_by_this_control':False,'prior_DE_PT_failed_retention_still_unqualified':True,
        'one_update_process_windows_not_a_production_cadence_recommendation':True}
    require(all(value[k] is True for k in ('all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_exact','original20_and_next_exact_source_draw_restored','actual_cleaner_replay_prefix_restored','clean_feature_payload_before_after_restore_exact')),'All actual state/payload checks required')
    write_json(folder/'actual_cleaner21_summary_v1.json',value)
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=now,phase='actual_cleaner21_closed0_coordinator_successor_preparation',
        current_stage='cleaner21_actual_CLOSED0_new_inactive_coordinator_publication_fix_next',actual_GPU_running=False,actual_owned_GPU_PID=None,actual_owned_GPU_session=None,
        actual_cleaner21_summary=record(folder/'actual_cleaner21_summary_v1.json'),actual_cleaner21_CLOSED0=record(actual),
        actual_cleaner21_new_native_updates=1,actual_cleaner21_all_native_write_restore_forecasts_covered=True,
        forecast_successor_prospectively_admitted=True,forecast_successor_validated_scope='ONE real2TR+FR/PT cleaner diagnostic update21 and streaming checkpoint write/restore only',
        production_ready=False,production_enabled=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v21.json',status)
    memory=read_json(A/'memory_diagnosis_status.json');memory.update(updated_at_utc=now,status='bounded_cleaner21_verified_GPU_idle',owned_GPU_child_running=False,
        actual_worker_PID=None,actual_owned_GPU_PID=None,actual_owned_GPU_session=None,current_stage=status['current_stage'],
        actual_cleaner21_summary=record(folder/'actual_cleaner21_summary_v1.json'),launch_readiness=record(A/'launch_readiness_v21.json'),
        launch_readiness_receipt=str(A/'launch_readiness_v21.json'),production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'actual_CLOSED0':record(actual),'summary':record(folder/'actual_cleaner21_summary_v1.json'),'native_private_peak':c['native_memory']['private_peak_bytes'],'global_peak':global_peak,'production_ready':False}))
if __name__=='__main__':main()
