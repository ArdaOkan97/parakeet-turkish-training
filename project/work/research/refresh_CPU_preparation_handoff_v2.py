"""Lightweight receipt refresh only; no downloads, audio decode, or new prep phase."""
import json
import pathlib
import psutil

from cpu_preparation_guard_v2 import AREA,PROJECT,ROOT,memory,now,sha256,write_json

def read(name):return json.loads((AREA/name).read_text(encoding='utf-8'))

def main():
    dispatch=read('owned_CPU_dispatch_v2.json')
    closures={
        'compressed_additional_stage':{'session_id':86358,'exit_code':0,'tool_chunk':'eda93f'},
        'source_heldout_exclusions':{'session_id':27096,'exit_code':0,'tool_chunk':'ac525a'},
        'full_accepted_pool_exclusion':{'session_id':80400,'exit_code':0,'tool_chunk':'42755b'},
        'pilot_audio_exclusion':{'session_id':85589,'exit_code':0,'tool_chunk':'74a80d'}}
    if (AREA/'independent_QC_actual_closure_v2.json').exists():
        closures['independent_QC']=read('independent_QC_actual_closure_v2.json')
    statuses={}
    for name,binding in dispatch['jobs'].items():
        status=read(binding['status_file']);statuses[name]=status
        binding.update(observed_phase=status['phase'],observed_at_utc=status['updated_at_utc'])
        if name in closures:
            if not status.get('actual_CLOSED0'):raise ValueError('Declared CPU closure lacks durable status')
            binding['actual_closure']={**closures[name],'status_completed_at_utc':status['updated_at_utc']}
    dispatch['updated_at_utc']=now();dispatch['current_running_jobs']=['independent_QC'] if not statuses['independent_QC'].get('actual_CLOSED0') else []
    stages=[json.loads(x) for x in (AREA/'additional_stage_ledger_v2.jsonl').read_text(encoding='utf-8').splitlines()]
    qc=[json.loads(x) for x in (AREA/'independent_QC_ledger_v2.jsonl').read_text(encoding='utf-8').splitlines()]
    window=[{k:r.get(k) for k in ['segment_id','completed_at_utc','elapsed_s','decode_elapsed_s','CPU_process_rss_bytes','independent_recognizer_completed']}
            for r in qc if '2026-10-16T05:07:20'<=r['completed_at_utc']<='2026-10-16T05:08:10']
    observation={'kind':'own_CPU_ledger_observations_near_telemetry_interruption_v2','recorded_at_utc':now(),
        'requested_window_utc':['2026-10-16T05:07:30','2026-10-16T05:07:50'],
        'QC_rows_in_nearby_window':window,
        'staging_interval':{'prior_shard_0002_completed_at_utc':next(x['completed_at_utc'] for x in stages if x['shard']=='0002'),
            'next_shard_0021_completed_at_utc':next(x['completed_at_utc'] for x in stages if x['shard']=='0021')},
        'exclusion_jobs_already_completed_before_requested_window':True,
        'historical_disk_queue_CPU_utilization_or_commit_counters_available':False,
        'interpretation':'One own QC decode took10.329s during concurrent0021 acquisition. This suggests a possible local delay but does not establish disk contention or the GPU telemetry failure cause.',
        'GPU_files_sources_or_sessions_inspected':False}
    write_json(AREA/'CPU_telemetry_window_observation_v2.json',observation)
    health={'at_utc':now(),'system_memory':memory(),'active_owned_CPU_jobs':[]}
    for name in dispatch['current_running_jobs']:
        status=statuses[name];p=psutil.Process(status['pid']);cmd=p.cmdline()
        if not any(dispatch['jobs'][name]['source'] in arg for arg in cmd):raise ValueError('Owned CPU process identity mismatch')
        health['active_owned_CPU_jobs'].append({'name':name,'pid':p.pid,'session_id':dispatch['jobs'][name]['session_id'],
            'process_create_time':p.create_time(),'rss_bytes':p.memory_info().rss,'priority':str(p.nice()),'cmdline':cmd})
    write_json(AREA/'actual_owned_CPU_health_snapshot_v2.json',health)
    stage=statuses['compressed_additional_stage'];plan=read('additional_stage_plan_v2.json')
    receipt={'kind':'CPU_preparation_meaningful_progress_handoff_v2','at_utc':now(),'owner':'/root/cpu_data_preparation',
        'dispatch':str(AREA/'owned_CPU_dispatch_v2.json'),'existing_healthy_CPU_worker_kept_running':bool(dispatch['current_running_jobs']),
        'new_heavy_IO_phases_started_since_telemetry_review':False,'statuses':statuses,'actual_closures':closures,
        'compressed_staging':{'new_shards_complete':len(stage['completed_shards']),'new_compressed_bytes':stage['completed_bytes'],
            'new_metadata_candidate_hours':stage['metadata_candidate_hours'],'new_metadata_candidate_clips':sum(x['candidate_clips'] for x in plan['selected_shards']),
            'existing_plus_new_shards':11,'existing_plus_new_compressed_bytes':25_245_143_040+stage['completed_bytes'],
            'existing_plus_new_metadata_candidate_hours':131.597565+stage['metadata_candidate_hours'],
            'existing_plus_new_metadata_candidate_clips':43434+sum(x['candidate_clips'] for x in plan['selected_shards'])},
        'pilot_milestone':read('original72_independent_QC_milestone_v2.json'),'audio_report':read('pilot_audio_exclusion_report_v2.json'),
        'independent_QC_report_snapshot':read('independent_QC_report_v2.json'),
        'source_heldout_exclusion_report':str(AREA/'source_heldout_exclusion_report_v2.json'),
        'full_accepted_pool_exclusion_report':str(AREA/'full_accepted_pool_exclusion_report_v2.json'),
        'telemetry_window_observation':str(AREA/'CPU_telemetry_window_observation_v2.json'),
        'long_WorldSpeech_overlap_priority':str(AREA/'long_WorldSpeech_text_overlap_priority_v2.json'),
        'current_owned_CPU_health':health,'GPU_owner_and_training_sessions_untouched':True,'accepted_TRAIN_hours':0,
        'limitations':['No admission pending rights,quality,full source/audio dedup and all-seven heldout safeguards.',
            'Existing900heldout sources cover six languages; Portuguese remains unresolved.',
            'Independent small-model flags are triage, not trusted corrections.',
            'Collision ledgers are append-only keyed events; SQLite candidate tables are authoritative resume counters.']}
    for field in ['source_heldout_exclusion_report','full_accepted_pool_exclusion_report']:
        receipt[field+'_sha256']=sha256(pathlib.Path(receipt[field]))
    write_json(AREA/'CPU_preparation_progress_handoff_v2.json',receipt)
    dispatch['latest_progress_handoff']=str(AREA/'CPU_preparation_progress_handoff_v2.json')
    dispatch['long_WorldSpeech_overlap_priority']=str(AREA/'long_WorldSpeech_text_overlap_priority_v2.json')
    write_json(AREA/'owned_CPU_dispatch_v2.json',dispatch)
    path=PROJECT/'outputs/parakeet_turkish/current_CPU_data_preparation_v1.json';old=path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
    state['CPU_validation_v2']=dispatch
    if path.read_bytes()!=old:raise RuntimeError('Concurrent project receipt update; retry lightweight merge')
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'handoff':str(AREA/'CPU_preparation_progress_handoff_v2.json'),'closures':closures,
        'staging':receipt['compressed_staging'],'QC':receipt['independent_QC_report_snapshot'],'health':health},ensure_ascii=True))

if __name__=='__main__':main()
