"""Unchanged resource primitives plus dynamic DATA-owned child observation."""
from datetime import datetime,timezone
from asr18_memory_diagnosis_v1 import (storage,record,require,now,stage,admit_stage,
    terminate_runtime_worker,STORAGE,read_json,sample_memory)


def concurrent_data_observation():
    """Read DATA's live changing carrier PID; never grant/gate its work."""
    area=STORAGE/'analysis/BPE_weak_then_clean_v1';workers=[];details={}
    try:
        dispatch=read_json(area/'current_data_dispatch_v2.json')
        require(dispatch['data_owner']=='/root/cpu_data_preparation','Known DATA owner required')
        items=[item for item in dispatch.get('heavy_successors',[]) if item.get('role')!='FLAC_carrier' and item.get('process_alive') is not False]
        items.extend(dispatch.get(key,{}) for key in ('metadata','BPE_labels','incoming_committed_label_geometry'))
        pids={item['PID'] for item in items if item.get('PID') and item.get('process_alive') is not False}
        carrier=read_json(area/'KIRAAT_FLAC_carrier_current_status_v1.json')
        details={'carrier_status_kind':carrier.get('kind'),'carrier_status_at_utc':carrier.get('at_utc'),
            'carrier_role':carrier.get('status'),'carrier_source':carrier.get('current_source'),'carrier_child_live':carrier.get('child_live')}
        published=datetime.fromisoformat(carrier['at_utc'].replace('Z','+00:00'))
        age=(datetime.now(timezone.utc)-published).total_seconds()
        details['carrier_status_age_seconds']=age
        if 0<=age<=5:
            supervisor=carrier.get('supervisor_PID')
            if supervisor:pids.add(supervisor)
            native=carrier.get('active_child_PID');launcher=carrier.get('active_launcher_PID')
            live=carrier.get('child_live') or {}
            if native and live.get('PID')==native:
                import psutil
                try:
                    process=psutil.Process(native)
                    expected_parent=supervisor if native==launcher else launcher
                    if process.ppid()==expected_parent:
                        pids.add(native);details['carrier_child_identity_checked']=True
                    else:details['carrier_child_identity_checked']=False
                except psutil.Error as exc:details['carrier_child_disappeared']=str(exc)
        else:details['carrier_sample_unavailable_reason']='fresh <=5s publication required; no stale PID substitution'
        for pid in sorted(pids):workers.append({'pid':pid,'private':sample_memory(pid,'concurrent_DATA_dynamic_observation')['worker_private_memory']})
        return {'at_utc':now(),'dispatch_updated_at_utc':dispatch.get('updated_at_utc'),'workers':workers,'dynamic_carrier':details,
            'data_IO_independently_owned':True,'pause_required':False,'all_external_commit_attribution_claimed':False}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'at_utc':now(),'workers':workers,'dynamic_carrier':details,'telemetry_error':str(exc),'pause_required':False,
            'all_external_commit_attribution_claimed':False}
