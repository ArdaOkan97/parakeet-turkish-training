"""Light live/closed ownership receipts for explicitly bounded CPU successors."""
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

STORE=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STORE/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import psutil
from cpu_preparation_guard_v2 import memory


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)


def main():
    phase=int(sys.argv[1]);session=int(sys.argv[2]);assert phase in (8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)
    area=STORE/f'analysis/CPU_validation_v{phase}'
    stem={8:'new256_tiny_primary_CPU_v8',9:'bounded_real_audio_references_CPU_v9',10:'next10k_real_audio_references_CPU_v10',11:'next10k_real_audio_references_CPU_v11',12:'next10k_real_audio_references_CPU_v12',13:'next10k_real_audio_references_CPU_v13',14:'next10k_real_audio_references_CPU_v14',15:'next10k_real_audio_references_CPU_v15',16:'next10k_real_audio_references_CPU_v16',17:'next10k_real_audio_references_CPU_v17',18:'next10k_real_audio_references_CPU_v18',19:'next10k_real_audio_references_CPU_v19',20:'next10k_real_audio_references_CPU_v20',21:'next10k_real_audio_references_CPU_v21',22:'next10k_real_audio_references_CPU_v22'}[phase]
    report_name='new256_tiny_primary_report_v8.json' if phase==8 else 'accepted_audio_reference_coverage_v4.json'
    source=PROJECT/'work/research'/{8:'run_new256_tiny_primary_CPU_v8.py',9:'resume_bounded_audio_references_CPU_v9.py',10:'extend_audio_references_next10k_CPU_v10.py',11:'extend_audio_references_next10k_CPU_v11.py',12:'extend_audio_references_next10k_CPU_v12.py',13:'extend_audio_references_next10k_CPU_v13.py',14:'extend_audio_references_next10k_CPU_v14.py',15:'extend_audio_references_next10k_CPU_v15.py',16:'extend_audio_references_next10k_CPU_v16.py',17:'extend_audio_references_next10k_CPU_v17.py',18:'extend_audio_references_next10k_CPU_v18.py',19:'extend_audio_references_next10k_CPU_v19.py',20:'extend_audio_references_next10k_CPU_v20.py',21:'extend_audio_references_next10k_CPU_v21.py',22:'extend_audio_references_next10k_CPU_v22.py'}[phase]
    state=json.loads((area/(stem+'_status.json')).read_text())
    process=psutil.Process(state['pid']) if psutil.pid_exists(state['pid']) else None
    if process:assert any(source.name in value for value in process.cmdline())
    report=json.loads((area/report_name).read_text()) if (area/report_name).exists() else None
    if phase in (10,11,12,13,14,15,16,17,18,19,20,21,22) and report and report.get('phase')=='paced_next10k_actual_reference_indexing':
        report['source_group_verified_counts_are_baseline_until_phase_boundary']=True
    ast.parse(source.read_text(encoding='utf-8'))
    value={'kind':f'owned_CPU_preparation_dispatch_v{phase}','owner':'/root/cpu_data_preparation',
         'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'session':session,'PID':state['pid'],
         'process_alive':process is not None,'phase':state['phase'],'source':str(source),
         'source_SHA':hashlib.sha256(source.read_bytes()).hexdigest(),'status_path':str(area/(stem+'_status.json')),
         'report_path':str(area/report_name),'report_snapshot':report,
         'CPU_only':True,'accepted_TRAIN_hours':0,'resource_and_IO_guards_unchanged':True,
         'fresh_system_memory_at_receipt':memory(),
         'original_large_CPP_partial_preserved_bytes':(STORE/'models/whisper_cpp_full_large_v3_q5_0/ggml-large-v3-q5_0.bin.partial').stat().st_size}
    if phase==11:
        value.update(
            bounded_phase_plan_path=str(area/'bounded_next10k_reference_plan_v11.json'),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v10/actual_next10k_reference_CLOSED0_v10.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            future_sanity_review_metadata_manifest=str(area/'future_sanity8_from_tiny_v8_manifest_v11.jsonl'),
            future_sanity_review_metadata_receipt=str(area/'future_sanity8_from_tiny_v8_receipt_v11.json'),
            future_sanity_review_audio_or_inference_started=False,
        )
    if phase==12:
        value.update(
            bounded_phase_plan_path=str(area/'bounded_next10k_reference_plan_v12.json'),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v11/actual_next10k_reference_CLOSED0_v11.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            ROOT_audio_release_required=str(STORE/'analysis/tiny_language_followup8_v1/ROOT_QC_audio_release.json'),
            ROOT_audio_release_verification_receipt=str(area/'verified_ROOT_audio_release_v12.json'),
            CPU_audio_reads_started=state.get('audio_reads_started',False),
            ROOT_release_wait_reason=state.get('ROOT_release_wait_reason') if not state.get('audio_reads_started',False) else None,
            historical_ROOT_release_wait_reason=state.get('ROOT_release_wait_reason'),
        )
    if phase==13:
        value.update(
            bounded_phase_plan_path=str(area/'bounded_next10k_reference_plan_v13.json'),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v12/actual_next10k_reference_CLOSED0_v12.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>40000),
            no_new_ASR_or_review_requested=True,
        )
    if phase==14:
        plan_path=area/'bounded_next10k_reference_plan_v14.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v13/actual_next10k_reference_CLOSED0_v13.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>50000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==15:
        plan_path=area/'bounded_next10k_reference_plan_v15.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v14/actual_next10k_reference_CLOSED0_v14.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>60000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==16:
        plan_path=area/'bounded_next10k_reference_plan_v16.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v15/actual_next10k_reference_CLOSED0_v15.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>70000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==17:
        plan_path=area/'bounded_next10k_reference_plan_v17.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v16/actual_next10k_reference_CLOSED0_v16.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>80000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==18:
        plan_path=area/'bounded_next10k_reference_plan_v18.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v17/actual_next10k_reference_CLOSED0_v17.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>90000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==19:
        plan_path=area/'bounded_next10k_reference_plan_v19.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v18/actual_next10k_reference_CLOSED0_v18.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>100000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==20:
        plan_path=area/'bounded_next10k_reference_plan_v20.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v19/actual_next10k_reference_CLOSED0_v19.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>110000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==21:
        plan_path=area/'bounded_next10k_reference_plan_v21.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v20/actual_next10k_reference_CLOSED0_v20.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>120000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase==22:
        plan_path=area/'bounded_next10k_reference_plan_v22.json'
        admitted=json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else {}
        value.update(
            bounded_phase_plan_path=str(plan_path),
            prior_actual_CLOSED0_receipt=str(STORE/'analysis/CPU_validation_v21/actual_next10k_reference_CLOSED0_v21.json'),
            resumable_reference_index=str(STORE/'indexes/accepted_audio_reference_v4.sqlite'),
            prior_ROOT_followup8_release_verification=str(STORE/'analysis/CPU_validation_v12/verified_ROOT_audio_release_v12.json'),
            CPU_audio_reference_work_observed=bool(report and report.get('actual_WAV_PCM_verified_reference_clips',0)>130000),
            fresh_admission_evidence={key:admitted.get(key) for key in (
                'fresh_storage_config_SHA256','fresh_storage_free_bytes_at_admission',
                'fresh_system_memory_at_admission','fresh_GPU_state_gate_at_admission')},
            no_new_ASR_or_review_requested=True,
        )
    if phase>=10:
        closure_path=area/f'actual_next10k_reference_CLOSED0_v{phase}.json'
        if closure_path.exists():
            closed=json.loads(closure_path.read_text(encoding='utf-8'))
            assert closed['session_id']==session and closed['pid']==state['pid'] and closed['exit_code']==0
            value.update(actual_CLOSED0_receipt_path=str(closure_path),
                actual_CLOSED0_receipt_SHA256=hashlib.sha256(closure_path.read_bytes()).hexdigest(),
                actual_observed_exit_code=closed['exit_code'],actual_exit_tool_chunk=closed['tool_chunk'])
    write(area/f'owned_CPU_dispatch_v{phase}.json',value)
    write(area/f'CPU_preparation_progress_handoff_v{phase}.json',value)
    path=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json'
    receipt=json.loads(path.read_text(encoding='utf-8-sig'));receipt[f'CPU_validation_v{phase}']=value
    if phase==9 and 'CPU_validation_v4' in receipt:
        receipt['CPU_validation_v4']['operating_state']='owned_suspended_worker_retired_to_bounded_v9_successor'
        receipt['CPU_validation_v4']['retirement_receipt']=str(area/'owned_V4_retirement_v9.json')
        receipt['CPU_validation_v4']['actual_old_shell_closure_receipt']=str(area/'actual_V4_shell_retirement_receipt_v9.json')
    write(path,receipt)
    print(json.dumps({'phase':phase,'session':session,'PID':state['pid'],'alive':process is not None,
         'completed':report.get('completed') if phase==8 and report else report.get('actual_WAV_PCM_verified_reference_clips') if report else None}))


if __name__=='__main__':main()
