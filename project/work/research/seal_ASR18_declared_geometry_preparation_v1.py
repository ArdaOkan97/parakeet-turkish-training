"""Seal two observed CPU controls; no final population/native claim."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1'
def main():
    completed=[]
    for actual,guard,chunk,count in (('declared_index_CPU_actual_01','declared_index_CPU_guard_01','a7603e',15),
                                   ('final_declared_metadata_CPU_actual_01','final_declared_metadata_CPU_guard_01','9a9055',18)):
        report=A/actual/'report.json';footer=A/guard/'closure.json';data=json.loads(report.read_text());closed=json.loads(footer.read_text())
        require(data['passed'] and len(data['checks'])==count and closed['actual_exit_code']==0 and closed['monitor_clean'] and closed['monitor_joined']
            and closed['observed']['samples']>=3 and closed['observed']['monitor_error'] is None,'Actual clean joined control closure required')
        receipt=A/actual/'actual_owned_CLOSED0.json';require(not receipt.exists(),'Preserve actual completion history')
        write_json(receipt,{'kind':'ASR18_guarded_CPU_tool_completion_v1','owner':'/root/memory_diagnosis','native_PID':closed['pid'],
            'actual_exec_tool_exit_code':0,'actual_native_exit_code':0,'actual_tool_chunk_id':chunk,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'report':record(report),'guard_closure':record(footer),'checks':count,'private_peak_bytes':closed['observed']['peak_private_bytes'],
            'actual_audio_decode_or_native_updates':0,'final_corpus_or_live_reader_integration_or_production_claimed':False})
        completed.append(record(receipt))
    files=[S/n for n in ('bpe_runtime_declared_geometry_v1.py','bpe_declared_geometry_index_v1.py','bpe_final_declared_geometry_v1.py','bpe_broad_final_population_v7.py')]
    files.extend([B/'declared_training_geometry_envelope_receipt_contract_v2.json',B/'declared_geometry_envelope_concrete_interface_companion_v1.json'])
    write_json(A/'declared_geometry_future_validator_preparation_v1.json',{'kind':'ASR18_inactive_declared_geometry_validator_preparation_v1',
        'owner':'/root/memory_diagnosis','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources_and_contracts':[record(v) for v in files],
        'actual_controls':completed,'final729_positive_envelope_path_exercised':False,'final_population_exists_or_armed':False,'production_enabled':False,
        'finite_cleaner_GPU_review_manifest_unchanged':record(A/'production_cleaner_forecast_coordinator_review_manifest_v3.json'),
        'upfront_full_PCM_decode_requirement':False,'runtime_full_native_and_resampled_and_encoder_geometry_must_match_before_draw':True,
        'final_source_ROOT_review_pending':True})
    memory=json.loads((A/'memory_diagnosis_status.json').read_text());memory.update(sustained20_attempt_passed=True,
        latest_completed_sameprocess_updates=20,latest_completed_sameprocess_actual=record(A/'feature_window_native20_01/actual_owned_CLOSED0.json'),
        historical_failed20_partial_receipt=memory.pop('latest_changed20_partial_receipt',None),
        declared_geometry_future_validator_preparation=record(A/'declared_geometry_future_validator_preparation_v1.json'),
        updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'sealed_CPU_controls':len(completed),'production_enabled':False,'GPU_started':False}))
if __name__=='__main__':main()
