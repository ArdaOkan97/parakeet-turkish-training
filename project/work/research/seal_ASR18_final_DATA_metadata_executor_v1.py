"""Freeze thin final-DATA metadata executor without executing its positive path."""
import datetime, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; R = P/'work/research'
sys.path[:0] = [str(S), str(R)]
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from seal_ASR18_concrete_multiwindow_caller_v1 import owned_closure
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def main():
    wrapper = R/'run_ASR18_final_DATA_verification_CPU_v1.py'; guard = R/'ASR18_bounded_CPU_phase_v3.py'
    review_path = A/'ROOT_concrete_multiwindow_caller5_source_read_review_v1.json'; review = read_json(review_path)
    require(review['accepted'] is True and review['source_manifest'] == record(A/'concrete_multiwindow_caller_source_review_manifest_v5.json'),
            'Exact accepted source-read scope only')
    target = A/'final_DATA_metadata_executor_source_review_manifest_v1.json'; require(not target.exists(), 'Preserve source review manifest')
    value = {'kind':'ASR18_final_DATA_thin_metadata_executor_source_review_manifest_v1',
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'owner':'/root/memory_diagnosis',
        'wrapper':record(wrapper), 'guard':record(guard), 'bindings':owned_closure([wrapper,guard]),
        'frozen_verifier':record(S/'bpe_final_DATA_population_v1.py'),
        'raw_publisher':record(R/'finalize_ASR18_DATA_population_for_bounded_preparation_v2.py'),
        'prospective_metadata_incremental_private_bytes':512_000_000, 'unchanged_private_hard_cap_bytes':2_000_000_000,
        'maximum_CPU_wall_seconds':3600, 'sampling':{'outer_watchdog_seconds':.2,'synchronous_callback_throttle_seconds':.1,
            'required_available_paired_samples_before_payload':3},
        'required_inputs':['genuine canonical raw aggregate and actual publisher TOOL0','exact source-bound publisher2',
            'canonical final catalog/labels/declared index read locks','fresh metadata512MB/global/headroom/E100GB admission'],
        'output':'independently validated final DATA report; successful tool/guard closure must then be separately sealed',
        'generator_not_invoked':True, 'actual_final_positive_path_executed':False,
        'audio_frontend_CACHE_or_GPU_preparation_supplied':False, 'native_or_full_CPU_frontend_envelope_approved':False,
        'production_enabled':False}
    write_json(target,value)
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=value['created_at_utc'],
        current_stage='GPU_idle_caller5_source_read_accepted_waiting_genuine_final_DATA_and_CPU_resource_review',
        current_caller5_whole_source_review_accepted=True,current_caller5_source_read_review=record(review_path),
        thin_final_DATA_metadata_executor=record(target),actual_GPU_running=False,actual_owned_GPU_PID=None,
        actual_owned_GPU_session=None,production_ready=False,production_enabled=False)
    readiness=A/'launch_readiness_v35.json';require(not readiness.exists(),'Preserve readiness history')
    write_json(A/'production_preparation_status_v1.json',status);write_json(readiness,status)
    current=read_json(A/'memory_diagnosis_status.json');current.update(updated_at_utc=value['created_at_utc'],current_stage=status['current_stage'],
        launch_readiness=record(readiness),launch_readiness_receipt=str(readiness),actual_GPU_running=False,
        actual_owned_GPU_PID=None,actual_owned_GPU_session=None,production_ready=False)
    write_json(A/'memory_diagnosis_status.json',current)
    print({'source_review':record(target),'wrapper':record(wrapper),'whole_caller_source_review':record(review_path),'GPU_running':False})


if __name__=='__main__':main()
