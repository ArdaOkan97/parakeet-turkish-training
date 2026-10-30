"""Lightweight ownership/progress receipt for the single real-audio worker."""
import json
import psutil
from cpu_preparation_guard_v2 import ROOT,PROJECT,now,sha256,write_json

AREA=ROOT/'analysis/CPU_validation_v4'

def main():
    status_path=AREA/'accepted_audio_reference_v4_status.json';s=json.loads(status_path.read_text(encoding='utf-8'))
    p=psutil.Process(s['pid'])
    if not any('run_accepted_audio_references_cpu_v4.py' in x for x in p.cmdline()):raise ValueError('Owned reference worker identity differs')
    coverage=AREA/'accepted_audio_reference_coverage_v4.json';c=json.loads(coverage.read_text(encoding='utf-8'))
    paths=['accepted_reference_plan_v4.json','per_recording_source_provenance_report_v4.json']
    r={'kind':'owned_single_paced_accepted_audio_reference_worker_v4','updated_at_utc':now(),'owner':'/root/cpu_data_preparation',
        'session_id':40713,'pid':p.pid,'process_create_time':p.create_time(),'started_at_utc':s['started_at_utc'],
        'current_phase':s['phase'],'status_path':str(status_path),'coverage_path':str(coverage),
        'durable_index':str(ROOT/'indexes/accepted_audio_reference_v4.sqlite'),
        'source':str(PROJECT/'work/research/run_accepted_audio_references_cpu_v4.py'),
        'source_sha256':sha256(PROJECT/'work/research/run_accepted_audio_references_cpu_v4.py'),
        'inherited_pacing_source_sha256':sha256(PROJECT/'work/research/paced_CPU_audio_v3.py'),
        'initial_reference_batch_limit':10000,'subsequent_batch_max_clips':10000,'global_declared_reference_clips':352768,
        'one_audio_worker':True,'audio_model_loaded':False,'read_limit_bytes_per_second':1048576,'BelowNormal_verified':int(p.nice())==16384,
        'own_RSS_bytes_observed':p.memory_info().rss,'1050heldout_bank_preserved':True,'GPU_lease_queue_sources_read_only':True,
        'accepted_TRAIN_hours':0,'reference_coverage_is_partial':not c['full_accepted_audio_reference_coverage'],
        'coverage_snapshot':c,'artifacts':{name:{'path':str(AREA/name),'sha256':sha256(AREA/name)} for name in paths if (AREA/name).exists()},
        'previous_v3_actual_CLOSED0':{'path':str(ROOT/'analysis/CPU_validation_v3/actual_closure_review_v3.json'),
            'sha256':sha256(ROOT/'analysis/CPU_validation_v3/actual_closure_review_v3.json'),'session_id':33309,'exit_code':0,'tool_chunk':'402312'}}
    write_json(AREA/'owned_CPU_dispatch_v4.json',r)
    write_json(AREA/'CPU_preparation_progress_handoff_v4.json',{'kind':'current_real_audio_reference_and_provenance_handoff_v4',
        'updated_at_utc':now(),'dispatch':r,'healthy_worker_left_running':True,'accepted_TRAIN_hours':0,
        'interpretation':'Verified reference hours sum the admitted training durations of physically read and hash-verified WAV/PCM sources. This does not add TRAIN hours.',
        'provenance_gap':'6791recordings in11staged metadata shards bound; original creator/title/video URL/dated recording license fields absent in the inspected pinned schema.',
        'next_automatic_phases':'Finish initial10k real references, cache193candidate fingerprints once, compare against batch, then continue bounded next10k references under dynamic GPU gates.'})
    target=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json';raw=target.read_bytes();data=json.loads(raw.decode('utf-8-sig'));data['CPU_validation_v4']=r
    if target.read_bytes()!=raw:raise RuntimeError('Concurrent current CPU receipt mutation; retry merge')
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dispatch':str(AREA/'owned_CPU_dispatch_v4.json'),'pid':p.pid,'session_id':40713,
        'actual_verified_references':c['actual_WAV_PCM_verified_reference_clips'],'verified_reference_training_duration_h':c['actual_audio_reference_hours'],
        'spectral_usable_references':c['spectral_reference_clips_usable_under_current_min8frames'],
        'bound_staged_source_recordings':c['per_recording_staged_source_metadata_evidence_rows'],'own_RSS_bytes':p.memory_info().rss}))

if __name__=='__main__':main()
