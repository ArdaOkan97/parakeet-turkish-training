"""Compact authoritative ASR18 data dispatch; metadata/process state only."""
from pathlib import Path
import datetime as dt,hashlib,json,os,sys
import psutil
ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/BPE_weak_then_clean_v1'
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
def read(name):
    p=AREA/name
    return read_json(p) if p.exists() else None

def write(name,value):
    write_json(AREA/name,value)

def live(pid,source):
    if not psutil.pid_exists(pid):return {'PID':pid,'process_alive':False}
    try:
        p=psutil.Process(pid)
        if not any(source in x for x in p.cmdline()):return {'PID':pid,'process_alive':False,'PID_no_longer_matches_owned_source':True}
        return {'PID':pid,'process_alive':True,'own_private_bytes':p.memory_info().private}
    except (psutil.NoSuchProcess,psutil.AccessDenied):return {'PID':pid,'process_alive':False}

def main():
    plan=read('preparation_plan_v1.json');old_done=read('acquisition_completed_files_v1.json')
    newer_done=(read('acquisition_completed_files_v2.json') or [])+(read('acquisition_completed_files_v3.json') or [])+(read('acquisition_completed_files_v4.json') or [])+(read('acquisition_completed_files_v5.json') or [])+(read('acquisition_completed_files_v6.json') or [])
    jobs={j['path']:j for j in plan['download_jobs']};completed={x['path']:x for x in old_done+newer_done}
    verified_by_repo={}
    for name,item in completed.items():
        job=jobs[name];g=verified_by_repo.setdefault(job['repository'],{'verified_files':0,'verified_bytes':0});g['verified_files']+=1;g['verified_bytes']+=job['size']
    meta=read('broad_metadata_status_v2.json');labels=read('BPE_label_preparation_status_v2.json')
    header=list({x['shard']:x for name in ('YODAS_archive_index_progress_v1.json','YODAS_archive_index_progress_v2.json','YODAS_archive_index_progress_v3.json') for x in (read(name) or [])}.values())
    metadata=live(meta['pid'],'build_BPE_broad_metadata_available_v2.py')
    metadata.update(session=28941,status=meta['status'],status_path=str(AREA/'broad_metadata_status_v2.json'),
      source=str(PROJECT/'work/research/build_BPE_broad_metadata_available_v2.py'),rows=meta['metadata_retained_rows'],
      gross_metadata_hours=meta['metadata_retained_hours'],committed_files=meta['committed_metadata_files'],
      available_first_new_rows=meta['metadata_retained_rows']-837102,paced_column_metadata_bytes=meta['paced_column_metadata_bytes'],maximum_column_read_bytes_s=262144)
    bpe=live(labels['pid'],'prepare_BPE_broad_labels_v2.py')
    bpe.update(session=54393,status=labels['status'],status_path=str(AREA/'BPE_label_preparation_status_v2.json'),
      source=str(PROJECT/'work/research/prepare_BPE_broad_labels_v2.py'),rows=labels['label_rows'],tokens=labels['total_target_tokens'],
      partial_population_mean_U=labels['population_mean_U'],vocabulary=8711,large_geometry_review_rows_retained=labels['conservative_geometry_review_rows'])
    heavy=[]
    for role,bindingname in (('acquisition','owned_paced_acquisition_dispatch_v2.json'),('archive_indexer','owned_paced_archive_indexer_dispatch_v2.json')):
        current_binding='owned_continuous_acquisition_dispatch_v3.json' if role=='acquisition' else 'owned_continuous_archive_indexer_dispatch_v3.json'
        binding=(read('owned_continuous_acquisition_dispatch_v6.json') if role=='acquisition' else None) or (read('owned_continuous_acquisition_dispatch_v5.json') if role=='acquisition' else None) or (read('owned_continuous_acquisition_dispatch_v4.json') if role=='acquisition' else None) or read(current_binding) or read(bindingname)
        version='v6' if binding and binding['source'].endswith('_v6.py') else 'v5' if binding and binding['source'].endswith('_v5.py') else 'v4' if binding and binding['source'].endswith('_v4.py') else 'v3' if binding and binding['source'].endswith('_v3.py') else 'v2'
        state=read(role+'_IO_status_'+version+'.json')
        if not binding:heavy.append({'role':role,'successor_launched':False});continue
        current=live(binding['PID'],Path(binding['source']).name)
        heavy.append({**binding,**current,'role':role,'status':state['status'],'paced_bytes':state['aggregate_paced_bytes']})
    carrier=read('KIRAAT_FLAC_carrier_current_status_v1.json')
    if carrier:
        pid=carrier.get('active_child_PID')
        supervisor_source=Path(carrier['supervisor_source']).name if carrier.get('supervisor_source') else 'run_ASR18_FLAC_carrier_batch_v6.py' if carrier['kind'].endswith('_v6') else 'run_ASR18_FLAC_carrier_continuation_v5.py'
        supervisor=live(carrier['supervisor_PID'],supervisor_source)
        heavy.append({'role':'FLAC_carrier','status':carrier['status'],
          **(live(pid,'prepare_ASR18_KIRAAT_FLAC_records') if pid else {'PID':None,'process_alive':False}),
          'supervisor_PID':carrier['supervisor_PID'],'status_path':str(AREA/'KIRAAT_FLAC_carrier_current_status_v1.json'),
          'supervisor':supervisor,'dynamic_native_PID_field':'active_child_PID',
          'current_source':carrier.get('current_source'),'child_live':carrier.get('child_live')})
    legacy_geometry=read('KIRAAT_declared_geometry_completion_current_status_v1.json')
    if legacy_geometry:
        pid=legacy_geometry.get('active_child_PID')
        supervisor=live(legacy_geometry['supervisor_PID'],Path(legacy_geometry['supervisor_source']).name)
        heavy.append({'role':'FLAC_declared_geometry_completion','status':legacy_geometry['status'],
          **(live(pid,'complete_ASR18_legacy_FLAC_declared_geometry') if pid else {'PID':None,'process_alive':False}),
          'supervisor_PID':legacy_geometry['supervisor_PID'],'supervisor':supervisor,
          'status_path':str(AREA/'KIRAAT_declared_geometry_completion_current_status_v1.json'),
          'dynamic_native_PID_field':'active_child_PID','current_source':legacy_geometry.get('current_source'),
          'child_live':legacy_geometry.get('child_live')})
    geometry=read('incoming_label_geometry_status_v1.json')
    old_exits=read('old_heavy_supported_gate_exit_v2.json')
    acquisition=read('acquisition_status_v6.json') or read('acquisition_status_v5.json') or read('acquisition_status_v4.json') or read('acquisition_status_v3.json') or read('acquisition_status_v2.json')
    active_partial=None
    if acquisition and acquisition.get('current_file') in jobs:
        active_path=Path(jobs[acquisition['current_file']]['destination']+'.part')
        active_partial={'path':str(active_path),'exists':active_path.exists(),
          'bytes':active_path.stat().st_size if active_path.exists() else 0,
          'source_file':acquisition['current_file']}
    actual_heavy_running=any((x.get('process_alive') and x.get('status') in ('paced_IO_active','paced_IO_allowed'))
      or (x.get('role') in ('FLAC_carrier','FLAC_declared_geometry_completion') and x.get('supervisor',{}).get('process_alive')
          and x.get('status') in ('prepare','audit','completion_verify','declared_geometry_completion')) for x in heavy)
    result={'kind':'ASR18_authoritative_current_data_dispatch_v2','updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'data_owner':'/root/cpu_data_preparation','GPU_owner':'/root/memory_diagnosis','task':'ASR-18','new_fixed_plan_files':len(plan['download_jobs']),
      'new_fixed_plan_bytes':plan['new_download_bytes'],'new_verified_complete_files':len(completed),
      'new_verified_complete_bytes':sum(jobs[x]['size'] for x in completed),'verified_by_repository':verified_by_repo,
      'new_plan_verified_bytes_fraction':sum(jobs[x]['size'] for x in completed)/plan['new_download_bytes'],
      'remaining_new_unverified_files':len(plan['download_jobs'])-len(completed),
      'remaining_new_unverified_bytes_including_preserved_partials':plan['new_download_bytes']-sum(jobs[x]['size'] for x in completed),
      'reused_verified_archives':len(plan['reused_archives']),'reused_verified_bytes':sum(j['size'] for j in plan['reused_archives']),
      'old_acquisition_transfer_bytes':read('acquisition_status_v1.json')['received_this_session_bytes'],
      'paced_successor_acquisition_transfer_bytes':acquisition['received_this_session_bytes'] if acquisition else 0,
      'cumulative_successor_network_bytes':sum((read('acquisition_status_'+v+'.json') or {}).get('received_this_session_bytes',0) for v in ('v2','v3','v4','v5','v6')),
      'active_successor_partial':active_partial,
      'preserved_partial':{'path':old_exits['partial_path'],'exists':Path(old_exits['partial_path']).exists(),
        'bytes':Path(old_exits['partial_path']).stat().st_size if Path(old_exits['partial_path']).exists() else 0},
      'old_heavy_native_exits':{'acquisition':old_exits['native_exit_observed'],'archive_indexer':old_exits['indexer_native_exit_observed']},
      'original_ROOT_tool_closure_bridge':read('ROOT_original_data_tool_closure_bridge_v1.json'),
      'metadata_original_native_exit':read('metadata_v1_retirement_actual_native_exit_v2.json')['actual_native_exit_observed'],
      'BPE_original_native_exit':read('BPE_labels_v1_retirement_actual_native_exit_v2.json')['actual_native_exit_observed'],
      'heavy_IO_running':actual_heavy_running,'heavy_successors':heavy,
      'continuous_parallel_data_authorized':bool(acquisition and acquisition.get('continuously_concurrent_IO')),
      'GPU_owner_IO_control_historical_only_for_v3':read('ROOT_data_IO_control_v2.json'),
      'header_archives_committed':len(header),'header_recordings_committed':sum(x['recordings'] for x in header),'metadata':metadata,'BPE_labels':bpe,
      'header_owned_actual_tool_closure':read('YODAS_header_owned_tool_actual_CLOSED0_v3.json'),
      'KIRAAT_original_FLAC_carrier_prepared_plan':read('KIRAAT_FLAC_sweep_plan_prepared_v1.json'),
      'KIRAAT_original_FLAC_carrier_current':carrier,
      'incoming_committed_label_geometry':geometry,
      'fixed_heldout_source_IDs':1050,'KIRAAT_published_heldout_source_IDs':85,'KIRAAT_published_heldout_text_keys':11648,
      'heldout_bindings_preserved_unchanged':True,'runtime_PCM_heldout_exclusion_still_required':True,
      'cached_label_alignment_audit':read('BPE_label_alignment_audit_actual_CLOSED0_v2.json'),
      'production_reader_metadata_bundle_actual_closure':read('production_reader_contract_v1/actual_CLOSED0.json'),
      'production_reader_original_source_flags':read('production_reader_contract_v1/original_source_flags_actual_CLOSED0.json'),
      'production_finalization_contract_path':str(AREA/'production_finalization_contract_v5.json'),
      'partial_geometry_envelope_actual_closure':read('partial_geometry_envelope_v1/actual_CLOSED0.json'),
      'prelaunch_permit_repair':read('launcher_UTC_cast_repair_validation_v2.json'),
      'quality_filters':[],'language_filters':[],'raw_flags_preserved':True,'TRAIN_admitted':False,'training_started':False,
      'gross_metadata_hours_are_not_unique_accepted_TRAIN_hours':True,'all_new_artifacts_on_E':True,
      'next_milestone':'Continue the fixed acquisition concurrently with GPU investigation under resource guards; commit projected metadata and existing8711 BPE labels without quality/LID cutoffs'}
    baseline=read('acquisition_rate_observation_32MiB_v6.json') or read('acquisition_rate_observation_32MiB_v5.json') or read('acquisition_rate_observation_32MiB_v4.json')
    readiness={'updated_at_utc':result['updated_at_utc'],'full_audio_corpus_ready':False,
      'full_metadata_corpus_ready':False,'TRAIN_admitted':False,
      'verified_compressed_fraction':(result['new_verified_complete_bytes']+result['reused_verified_bytes'])/(plan['new_download_bytes']+result['reused_verified_bytes']),
      'metadata_files_committed':metadata['committed_files'],'metadata_files_required':730,
      'metadata_files_remaining':730-metadata['committed_files'],
      'header_archives_committed':len(header),'header_archives_required':51,
      'source_rows_and_mean_U_remain_partial':True,
      'full_compressed_carrier_coverage_ready':False,
      'download_ETA_excludes_compressed_carrier_preparation':True,
      'known_committed_U_blockers':geometry['groups_all_committed']['count_U_gt_458'] if geometry else None,
      'geometry_observer_frontier':geometry['observed_committed_frontier'] if geometry else None,
      'finalization_prepared_receipt':str(AREA/'continuous_finalization_successors_prepared_v4.json')}
    if baseline and acquisition and acquisition['pid']==baseline['acquisition']['pid']:
        beginning=baseline['acquisition'];seconds=(dt.datetime.fromisoformat(acquisition['updated_at_utc'])-dt.datetime.fromisoformat(beginning['updated_at_utc'])).total_seconds()
        delta=acquisition['received_this_session_bytes']-beginning['received_this_session_bytes']
        if seconds>30 and delta>0:
            rate=delta/seconds
            parts=result['preserved_partial']['bytes']+(active_partial['bytes'] if active_partial and active_partial['exists'] else 0)
            remaining=max(0,result['remaining_new_unverified_bytes_including_preserved_partials']-parts)
            readiness.update(rate_interval_start=beginning['updated_at_utc'],rate_interval_end=acquisition['updated_at_utc'],
              observed_interval_seconds=seconds,actual_network_bytes_in_interval=delta,observed_MiB_per_second=rate/1048576,
              remaining_network_bytes=remaining,conditional_network_hours_remaining=remaining/rate/3600,
              conditional_network_finish_UTC=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=remaining/rate)).isoformat(),
              ETA_excludes_final_verification_and_admission=True,resource_yields_and_remote_latency_can_extend_ETA=True)
    result['readiness']=readiness
    write('current_data_readiness_v1.json',readiness)
    write('current_data_dispatch_v2.json',result);write('current_data_preparation_status_v2.json',result)
    write('DATA_ROOT_GPU_no_heavy_IO_ready_v3.json',{'kind':'ASR18_current_no_heavy_IO_boundary_v3','updated_at_utc':result['updated_at_utc'],
      'no_heavy_IO_ready':not actual_heavy_running,'old_heavy_native_exits_observed':True,'current_light_metadata_PID':metadata['PID'],
      'current_light_BPE_PID':bpe['PID'],'current_dispatch':str(AREA/'current_data_dispatch_v2.json'),'TRAIN_admitted':False})
    print(json.dumps({'dispatch':str(AREA/'current_data_dispatch_v2.json'),'metadata':metadata,'BPE_labels':bpe,'heavy_IO_running':actual_heavy_running}))
if __name__=='__main__':main()
