"""Distinguish native ceiling from entire-worker write/restore admission."""
import datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S));sys.path.insert(0,str(R))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from seal_ASR18_DATA_and_original_schedule_preparation_v1 import actual,closure
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    prior_path=A/'multiwindow_preparation_assembly_source_review_manifest_v1.json';value=read_json(prior_path)
    for pin in value['active_owned_dependency_closure']:require(record(pin['path'])==pin,'Prior frozen source changed')
    owned=actual(A/'entire_worker_forecast_CPU_actual_01',A/'entire_worker_forecast_CPU_guard_01','ae952b')
    roots=[Path(v['path']) for v in value['direct_sources']]+[S/'asr18_precomputed_worker_forecast_v1.py',R/'validate_ASR18_entire_worker_forecast_v1.py']
    value.update(kind='ASR18_inactive_multiwindow_preparation_assembly_source_review_manifest_v2',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prior_preserved_assembly=record(prior_path),direct_sources=[record(p) for p in roots],active_owned_dependency_closure=closure(roots),
        proposed_native_private_forecast_bytes=value.pop('proposed_initial_full_worker_private_forecast_bytes'),
        proposed_entire_worker_private_forecast_bytes=5_910_813_440,entire_worker_forecast_CPU_actual=owned,
        entire_worker_formula='max(startup3GB,native ceiling+write64MB,native ceiling+restore350MB); no post-native residency drop assumed',
        first_assembly_field_clarification='The5,560,813,440B field was the native ceiling, not a sufficient entire-worker launch forecast. No GPU launch used it.',
        outer_fresh_prelaunch_admission_and_every_actual_phase_must_fit=True)
    target=A/'multiwindow_preparation_assembly_source_review_manifest_v2.json';require(not target.exists(),'Preserve assembly successor');write_json(target,value)
    status=read_json(A/'production_preparation_status_v1.json');status.update(updated_at_utc=value['created_at_utc'],
        current_stage='GPU_idle_multiwindow_sources_frozen_native5p561GB_entire5p911GB_proposals_final729_and_review_pending',
        multiwindow_preparation_source_review=record(target),next_proposed_native_private_forecast_bytes=5_560_813_440,
        next_proposed_full_worker_private_forecast_bytes=5_910_813_440,next_entire_worker_forecast_controls=owned,
        next_native_forecast_approved=False,actual_GPU_running=False,actual_owned_GPU_PID=None,production_ready=False,production_enabled=False)
    write_json(A/'production_preparation_status_v1.json',status);write_json(A/'launch_readiness_v27.json',status)
    memory=read_json(A/'memory_diagnosis_status.json');memory.update(updated_at_utc=value['created_at_utc'],current_stage=status['current_stage'],
        launch_readiness=record(A/'launch_readiness_v27.json'),launch_readiness_receipt=str(A/'launch_readiness_v27.json'),
        actual_GPU_running=False,actual_owned_GPU_PID=None,production_ready=False)
    write_json(A/'memory_diagnosis_status.json',memory)
    print(json.dumps({'source_review':record(target),'owned_dependencies':len(value['active_owned_dependency_closure']),
        'native_private_proposal':5_560_813_440,'entire_worker_proposal':5_910_813_440,'GPU_running':False,'production_ready':False}))
if __name__=='__main__':main()
