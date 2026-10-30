"""Seal actual owned exit and measured finite-window evidence; no model import."""
import datetime, json, sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def main():
    f=A/'feature_window_native20_01';r=json.loads((f/'report.json').read_text());c=r['completion']
    require(r['passed'] and r['actual_exit_code']==0 and r['PID']==23568 and r['launcher_PID']==38856
            and c['native_updates']==20 and c['same_owned_GPU_process'] and c['sources_unchanged'], 'Actual finite20 proof')
    require(not json.loads((P/'work/gpu_training_owner.json').read_text())['actual_GPU_running'], 'Owned child must actually close')
    require(c['final_clocks']==[64270,2020,62270] and [v['updates'] for v in c['restore_proofs']]==[1,10,20]
            and all(v['passed'] and v['all207_weights_all3_Adam_and_Python_NumPy_Torch_CUDA_RNG_exact']
                    and v['actual_both_sampler_and_journal_prefix_verified'] for v in c['restore_proofs']), 'Exact coherent state proofs')
    actual={'kind':'ASR18_owned_actual_closed_GPU_child_v1','owner':'/root/memory_diagnosis','owned_session':38806,
        'native_PID':23568,'launcher_PID':38856,'actual_tool_exit_code':0,'actual_native_exit_code':0,'actual_tool_chunk_id':'b5e417',
        'report':record(f/'report.json'),'checkpoint':c['checkpoint'],'native_updates':20,'diagnostic_only':True,'source_clocks':c['final_clocks']}
    require(not (f/'actual_owned_CLOSED0.json').exists(),'Preserve sealed history')
    write_json(f/'actual_owned_CLOSED0.json',actual)
    forecasts=[]
    for n in range(1,21):
        p=json.loads((f/f'native_phase_{n}.json').read_text());b=json.loads((f/f'update_{n}_budget.json').read_text())
        require(p['error'] is None and p['joined'] and p['available_sample_count']>=3,'Clean native phase monitor')
        growth=p['private_peak_bytes']-b['current_private_bytes']
        forecasts.append({'update':n,'baseline_private_bytes':b['current_private_bytes'],'actual_native_peak_private_bytes':p['private_peak_bytes'],
            'incremental_private_budget_bytes':b['incremental_bytes'],'observed_increase_bytes':growth,'budget_margin_bytes':b['incremental_bytes']-growth,
            'native_phase':record(f/f'native_phase_{n}.json'),'budget':record(f/f'update_{n}_budget.json')})
    misses=[v for v in forecasts if v['budget_margin_bytes']<0]
    stages=[json.loads(line) for line in (f/'stages.jsonl').read_text().splitlines()]
    ss=r['samples'];native=c['native_phase_metrics']
    aug=sum(bool(v['augmentation'].get('applied')) for h in c['history'] for v in h['augmentation']['rows'])
    langs=sorted(set(lang for h in c['history'] for lang in h['languages']))
    stage_span=stages[-1]['observed_monotonic']-stages[0]['observed_monotonic']
    parent_span=ss[-1]['observed_monotonic']-ss[0]['observed_monotonic']
    cpu_prep=696.313
    summary={'kind':'ASR18_actual_sameprocess_feature_window20_summary_v1','passed':True,'actual_owned_exit':record(f/'actual_owned_CLOSED0.json'),
        'actual_native_updates':20,'same_PID':23568,'all207_FP32_three_Adam_clocks':c['final_clocks'],'checkpoint':c['checkpoint'],
        'exact_checkpoint_restore_updates':[1,10,20],'all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_both_samplers_and_prefix_exact':True,
        'actual_augmented_TR_waveforms':aug,'actual_languages':langs,'training_rows':320,'full_corpus_complete_passes':0,
        'native_phase_private_peak_50ms_bytes':max(v['private_peak_bytes'] for v in native),
        'parent_private_peak_1s_bytes':max(v['worker_private_memory']['private_bytes'] for v in ss if v['worker_private_memory']['available']),
        'parent_system_commit_peak_fraction_1s':max(v['system_memory']['commit_fraction'] for v in ss),
        'worker_stage_private_lifetime_peak_bytes':max(v.get('worker_private_peak_bytes_lifetime',0) for v in stages),
        'native_phase_seconds':c['total_native_wall_seconds'],'probe_including_checkpoint_restore_seconds':c['prepare_probe_wall_seconds'],
        'worker_stage_span_seconds':stage_span,'parent_observed_child_span_seconds':parent_span,
        'separate_CPU_window_preparation_seconds':cpu_prep,'serial_CPU_preparation_plus_parent_child_observed_seconds':cpu_prep+parent_span,
        'training_audio_seconds_repeated_diagnostic':c['total_training_audio_seconds'],
        'native_audio_seconds_per_wall_second':c['total_training_audio_seconds']/c['total_native_wall_seconds'],
        'sequential_CPU_preparation_and_GPU_observed_audio_seconds_per_wall_second':c['total_training_audio_seconds']/(cpu_prep+parent_span),
        'throughput_is_fixed_fixture_not_broad_corpus_projection':True,
        'original_private_forecast_misses':misses,'original_forecast_coverage_passed':not misses,
        'unchanged_global_headroom_CUDA_guards_stayed_active':True,'measured_stability_does_not_approve_private_forecast':True,
        'parent_frontend_or_augmentation_invoked':c['parent_frontend_or_augmentation_invoked'],'parent_imported_modules':c['parent_imported_modules'],
        'feature_cache_retained':True,'diagnostic_only':True,'qualified':False,'production_ready':False,
        'remaining':['Forecast successor prospective admission and review','Serial full350DEV of this exact checkpoint',
                     'Complete source/carrier/catalog/labels/denominator/geometry finalization','Production broad/cleaner/DEV coordinator and full review'],
        'historical_external_commit_cause_not_attributed':True,'final_forecast_audit':record(f/'report.json')}
    require(not (f/'actual_sameprocess20_summary_v1.json').exists(),'Preserve summary history')
    write_json(f/'actual_sameprocess20_summary_v1.json',summary)
    write_json(f/'forecast_audit_final_v1.json',{'kind':'ASR18_actual_closed_native20_private_forecast_audit_v1','actual_owned_exit':record(f/'actual_owned_CLOSED0.json'),
        'phases':forecasts,'missed_updates':[v['update'] for v in misses],'forecast_policy_production_approved':False})
    status=json.loads((A/'production_preparation_status_v1.json').read_text())
    status.update(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='feature_window_native20_actual_CLOSED0',
        actual_GPU_running=False,actual_owned_GPU_PID=None,current_stage='GPU_idle_after_actual20_checkpoint_restore',
        current_native20_completed_updates=20,sustained_sameprocess20_proven=True,latest_changed20_completed_updates=20,
        latest_changed20_passed=True,latest_changed20_checkpoint_created=True,production_ready=False,production_enabled=False,
        original_native20_private_forecast_coverage_passed=False,original_native20_private_forecast_missed_updates=[v['update'] for v in misses],
        actual_native20_summary=record(f/'actual_sameprocess20_summary_v1.json'),actual_native20_CLOSED0=record(f/'actual_owned_CLOSED0.json'),
        native20_forecast_audit=record(f/'forecast_audit_final_v1.json'),
        production_remaining=summary['remaining'])
    write_json(A/'launch_readiness_v13.json',status);write_json(A/'production_preparation_status_v1.json',status)
    memory=json.loads((A/'memory_diagnosis_status.json').read_text())
    memory.update(updated_at_utc=status['updated_at_utc'],status='bounded_production_preparation_GPU_idle',owned_GPU_child_running=False,
        actual_owned_GPU_PID=None,actual_worker_PID=None,current_stage=status['current_stage'],
        current_broad_window20_stability_verified=True,bounded_native_stability_verified_scope='Actual fixed14TR+2replay feature-window20, same PID23568, exact restores1/10/20; not full corpus',
        launch_readiness_receipt=str(A/'launch_readiness_v13.json'),launch_readiness=record(A/'launch_readiness_v13.json'),
        original_native20_forecast_missed_updates=[v['update'] for v in misses],production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({k:summary[k] for k in ('actual_native_updates','actual_augmented_TR_waveforms','native_phase_private_peak_50ms_bytes',
        'parent_system_commit_peak_fraction_1s','parent_observed_child_span_seconds','native_audio_seconds_per_wall_second')}))

if __name__=='__main__':main()
