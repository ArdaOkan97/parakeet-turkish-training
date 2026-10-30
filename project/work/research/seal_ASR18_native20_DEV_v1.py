"""Publish actual serial DEV closure and integer-gate result without promotion."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    f=A/'feature_window_native20_DEV_01';r=json.loads((f/'report.json').read_text());c=r['completion']
    require(r['passed'] and r['actual_exit_code']==0 and r['PID']==9180 and r['launcher_PID']==32864 and c['full350_DEV_complete'], 'Actual owned serial350DEV0 proof')
    require(not json.loads((P/'work/gpu_training_owner.json').read_text())['actual_GPU_running'],'Actual idle lease')
    path=f/'actual_owned_CLOSED0.json';require(not path.exists(),'Preserve actual closure history')
    write_json(path,{'kind':'ASR18_owned_actual_closed_GPU_child_v1','owner':'/root/memory_diagnosis','owned_session':67233,
        'native_PID':9180,'launcher_PID':32864,'actual_tool_exit_code':0,'actual_native_exit_code':0,'actual_tool_chunk_id':'cec793',
        'report':record(f/'report.json'),'checkpoint':c['checkpoint'],'native_updates':0,'DEV_source_native_frontier':20,
        'diagnostic_only':True,'source_clocks':c['clocks_preserved_in_checkpoint']})
    ss=r['samples'];summary={'kind':'ASR18_actual_native20_serial350DEV_summary_v1','execution_passed':True,'actual_owned_exit':record(path),
        'full350_DEV_complete':True,'all207_weights_restored_byte_exact':c['all207_weight_restore_byte_exact'],
        'DEV':c['DEV'],'failed_retention_languages':[k for k,v in c['DEV']['retention_language_passed'].items() if not v],
        'parent_system_commit_peak_fraction':max(v['system_memory']['commit_fraction'] for v in ss),
        'parent_private_peak_bytes':max(v['worker_private_memory']['private_bytes'] for v in ss if v['worker_private_memory']['available']),
        'parent_observed_seconds':ss[-1]['observed_monotonic']-ss[0]['observed_monotonic'],
        'checkpoint':c['checkpoint'],'actual_source_native_frontier':20,'TEST_used':False,'qualified':False,
        'production_ready':False,'quality_gain_or_winner_claimed':False,'actual_native_updates_in_DEV':0}
    write_json(f/'actual_DEV_summary_v1.json',summary)
    status=json.loads((A/'production_preparation_status_v1.json').read_text())
    status.update(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='native20_and_serial350DEV_actual_CLOSED0',
        actual_GPU_running=False,actual_owned_GPU_PID=None,current_stage='GPU_idle_after_serial350DEV',
        latest_canonical2D_native20_DEV_status='actual_closed_passed_execution_failed_retention',latest_canonical2D_native20_DEV=c['DEV'],
        latest_canonical2D_native20_DEV_report=record(f/'report.json'),latest_canonical2D_native20_DEV_summary=record(f/'actual_DEV_summary_v1.json'),
        latest_retention_failed_languages=summary['failed_retention_languages'],qualified_checkpoint_created=False,production_ready=False,production_enabled=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v15.json',status)
    memory=json.loads((A/'memory_diagnosis_status.json').read_text());memory.update(updated_at_utc=status['updated_at_utc'],status='bounded_production_preparation_GPU_idle',
        owned_GPU_child_running=False,actual_owned_GPU_PID=None,actual_worker_PID=None,current_stage=status['current_stage'],
        launch_readiness_receipt=str(A/'launch_readiness_v15.json'),launch_readiness=record(A/'launch_readiness_v15.json'),
        latest_DEV_failed_retention_languages=summary['failed_retention_languages'],diagnostic_checkpoint_qualified=False,production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    g=A/'production_window_CPU_guard_01/closure.json';guard=json.loads(g.read_text())
    require(guard['actual_exit_code']==0 and guard['pid']==39028 and guard['monitor_clean'] and guard['monitor_joined'],'Actual pure-state fixture guard0')
    write_json(A/'production_window_CPU_actual_01/actual_owned_CLOSED0.json',{'kind':'ASR18_owned_actual_closed_CPU_fixture_v1','owner':'/root/memory_diagnosis',
        'native_PID':39028,'actual_tool_exit_code':0,'actual_native_exit_code':0,'actual_tool_chunk_id':'f586a5','native_guard':record(g),
        'report':record(A/'production_window_CPU_actual_01/report.json'),'actual_native_updates':0,'production_ready':False})
    print(json.dumps(summary))
if __name__=='__main__':main()
