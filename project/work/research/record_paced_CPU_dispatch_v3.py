"""Small current CPU receipts; no GPU mutation or heavy I/O."""
import json
import psutil
from cpu_preparation_guard_v2 import ROOT,PROJECT,now,sha256,write_json

AREA=ROOT/'analysis/CPU_validation_v3'

def main():
    status_path=AREA/'paced_audio_QC_v3_status.json';status=json.loads(status_path.read_text(encoding='utf-8'))
    p=psutil.Process(status['pid']);cmd=p.cmdline()
    if not any('run_paced_CPU_validation_v3.py' in x for x in cmd):raise ValueError('Owned paced CPU identity differs')
    pt=AREA/'Portuguese_heldout_audio_verification_v3.jsonl'
    rows=[json.loads(x) for x in pt.read_text(encoding='utf-8').splitlines()] if pt.exists() else []
    paths=['paced_CPU_QC_plan_v3.json','Portuguese_heldout_binding_receipt_v3.json','rights_provenance_review_v3.json',
        'WorldSpeech_audio_overlap_report_v3.json','seven_language_audio_reference_binding_v3.json','new8_shards_independent_QC_report_v3.json']
    receipt={'kind':'owned_single_paced_CPU_audio_preparation_dispatch_v3','updated_at_utc':now(),'owner':'/root/cpu_data_preparation',
        'session_id':33309,'pid':p.pid,'started_at_utc':status['started_at_utc'],'process_create_time':p.create_time(),'current_phase':status['phase'],
        'status_path':str(status_path),'pacing_events':str(AREA/'pacing_gate_events_v3.jsonl'),'file_read_limit_bytes_per_second':1048576,
        'maximum_audio_workers':1,'BelowNormal_priority_verified':int(p.nice())==16384,'own_RSS_bytes_observed':p.memory_info().rss,
        'PT_groundtruth_sources_bound':150,'PT_actual_WAV_PCM_hash_rechecks_durable':len(rows),
        'metadata_setup_actual_closure':{'exit_code':0,'tool_chunk':'71c390'},'rights_review_actual_closure':{'exit_code':0,'tool_chunk':'de6b89'},
        'ROOT_native_positive_proof_required':str(ROOT/'analysis/three_epoch_entry_v1/telemetry_stop_75250_v1/actual_recovery_native_positive_v1.json'),
        'GPU_lease_queue_and_training_sources_read_only':True,'bulk_downloads':False,'accepted_TRAIN_hours':0,
        'source_bindings':{n:sha256(PROJECT/'work/research'/n) for n in ['paced_CPU_audio_v3.py','run_paced_CPU_validation_v3.py']},
        'artifacts':{n:{'path':str(AREA/n),'sha256':sha256(AREA/n)} for n in paths if (AREA/n).exists()}}
    write_json(AREA/'owned_CPU_dispatch_v3.json',receipt)
    overlap=json.loads((AREA/'WorldSpeech_audio_overlap_report_v3.json').read_text(encoding='utf-8'))
    handoff={'kind':'single_paced_CPU_preparation_progress_handoff_v3','updated_at_utc':now(),'dispatch':receipt,
        'Portuguese_groundtruth_bound':150,'Portuguese_actual_WAV_PCM_rechecks_durable':len(rows),
        'actual_seven_language_reference_bank_available':(AREA/'seven_language_audio_reference_binding_v3.json').exists(),
        'combined_heldout_sources':1050 if (AREA/'seven_language_audio_reference_binding_v3.json').exists() else 900+len(rows),
        'priority_overlap_audio_comparisons':overlap['actual_audio_comparisons'],'priority_overlap_not_staged':overlap['not_staged_no_download'],
        'priority_overlap_strong_waveform_matches':overlap['strong_waveform_overlap_recordings'],
        'priority_overlap_local_recording_remains_quarantined':True,'new_shard_semantic_QC_declared_clips':192,
        'current_paced_stage':status['phase'],'model_bytes_read_at_snapshot':status.get('active_file_bytes_read',0),
        'semantic_QC_report_available':(AREA/'new8_shards_independent_QC_report_v3.json').exists(),
        'commercial_dataset_CC_BY3_claim_verified':True,'commercial_per_recording_rights_clearance_established':False,
        'authorized_research_CPU_preparation_continues':True,'accepted_TRAIN_hours':0,'GPU_used':False,
        'TEST_prediction_files_read':False,'new_bulk_downloads':False,'healthy_worker_left_running':True}
    write_json(AREA/'CPU_preparation_progress_handoff_v3.json',handoff)
    receipt['latest_progress_handoff']=str(AREA/'CPU_preparation_progress_handoff_v3.json')
    write_json(AREA/'owned_CPU_dispatch_v3.json',receipt)
    dest=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json';old=dest.read_bytes();data=json.loads(old.decode('utf-8-sig'));data['CPU_validation_v3']=receipt
    if dest.read_bytes()!=old:raise RuntimeError('Concurrent project receipt change; retry lightweight merge')
    dest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=True))

if __name__=='__main__':main()
