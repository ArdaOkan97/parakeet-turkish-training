"""Compact live pointers after sealed broad-window20; immutable history retained."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    value=json.loads((A/'production_preparation_status_v1.json').read_text());lease=json.loads((P/'work/gpu_training_owner.json').read_text())
    f=A/'feature_window_native20_01';summary=json.loads((f/'actual_sameprocess20_summary_v1.json').read_text())
    value.update(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=lease['status'],actual_GPU_running=lease['actual_GPU_running'],
        actual_owned_GPU_PID=lease['actual_owned_worker_pid'],current_stage=lease.get('diagnostic_phase'),diagnostic_folder=lease.get('diagnostic_folder'),
        sustained_sameprocess20_proven=True,current_native20_completed_updates=20,latest_changed20_completed_updates=20,
        latest_changed20_passed=True,latest_changed20_checkpoint_created=True,actual_native20_summary=record(f/'actual_sameprocess20_summary_v1.json'),
        original_native20_private_forecast_coverage_passed=False,original_native20_private_forecast_missed_updates=[v['update'] for v in summary['original_private_forecast_misses']],
        future_forecast_retrospective20_replay=record(A/'fixed_forecast_CPU_actual_02/report.json'),forecast_successor_prospectively_admitted=False,
        full_corpus_geometry_finalized=False,actual_production_draws_consumed=0,production_ready=False,production_enabled=False)
    d=A/'feature_window_native20_DEV_01'
    value['latest_canonical2D_native20_DEV_status']='running' if lease['actual_GPU_running'] and lease.get('diagnostic_folder')==str(d) else 'not_yet_actual_closed'
    if (d/'report.json').exists():
        dev=json.loads((d/'report.json').read_text());value['latest_canonical2D_native20_DEV_status']='actual_closed_passed_execution' if dev['passed'] else 'actual_closed_failed_execution'
        value['latest_canonical2D_native20_DEV_report']=record(d/'report.json')
        if dev.get('completion'):
            value['latest_canonical2D_native20_DEV']=dev['completion'].get('DEV')
    value['production_remaining']=['Prospective conservative forecast successor and geometry policy review',
        'Strict serial350DEV quality/retention of final native20 state',
        'Final completed assets/carriers/catalog/labels/whole denominator and declared geometry with runtime match',
        'Full broad/cleaner/DEV coordinator with tail and measured multiwindow checkpoint/CPU preparation lifecycle',
        'Complete final owned source closure ROOT review; user launch decision']
    write_json(A/'launch_readiness_v14.json',value);write_json(A/'production_preparation_status_v1.json',value)
    memory=json.loads((A/'memory_diagnosis_status.json').read_text())
    memory.update(updated_at_utc=value['updated_at_utc'],status='bounded_production_preparation_GPU_running' if lease['actual_GPU_running'] else 'bounded_production_preparation_GPU_idle',
        owned_GPU_child_running=lease['actual_GPU_running'],actual_worker_PID=lease['actual_owned_worker_pid'],actual_owned_GPU_PID=lease['actual_owned_worker_pid'],
        current_stage=value['current_stage'],current_broad_window20_stability_verified=True,
        launch_readiness_receipt=str(A/'launch_readiness_v14.json'),launch_readiness=record(A/'launch_readiness_v14.json'),
        original_native20_forecast_missed_updates=value['original_native20_private_forecast_missed_updates'],production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({k:value[k] for k in ('phase','actual_owned_GPU_PID','current_stage','latest_canonical2D_native20_DEV_status','production_ready')}))
if __name__=='__main__':main()
