"""Metadata only, after an actually observed normal owner TOOL0; no launch."""
import argparse,json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import dead

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--output',required=True)
    p.add_argument('--observed-tool-exit-code',required=True,type=int);p.add_argument('--observed-tool-chunk',required=True)
    p.add_argument('--observed-session',required=True,type=int);a=p.parse_args()
    require(a.observed_tool_exit_code==0 and bool(a.observed_tool_chunk),'Real observed normal outer TOOL0 is required')
    folder=Path(a.folder).resolve();out=Path(a.output).resolve()
    require(folder.is_relative_to(Path(r'E:\gpu_training_scratchpad')) and out.is_relative_to(Path(r'E:\gpu_training_scratchpad'))
        and not out.exists(),'Owned E-only fresh continuation metadata')
    report_pin=record(folder/'report.json');report=read_json(report_pin['path'])
    require(report['kind']=='ASR18_full_owner_normal_boundary_pause_report_v1' and report['passed'] is True
        and dead(report['producer']),'Actual normally closed producer; abnormal owner cannot use this path')
    snapshot_pin=report['normal_pause_snapshot'];require(record(snapshot_pin['path'])==snapshot_pin,'Exact normal pause snapshot')
    snapshot=read_json(snapshot_pin['path']);config_pin=report['configuration']
    require(record(config_pin['path'])==config_pin,'Same immutable config')
    config=read_json(config_pin['path']);guard_pin=record(folder/'owner_guard_completion.json');guard=read_json(guard_pin['path'])
    require(guard['monitor_joined'] is True and guard['monitor_clean'] is True and guard['samples']>=3
        and guard['producer']==report['producer'],'Actual clean joined owner guard')
    require(snapshot['normal_coherent_pause'] is True and snapshot['configuration']==config_pin
        and snapshot['producer']==report['producer'] and snapshot['namespace']==config['run_namespace']
        and snapshot['actual_optimizer_updates']==report['actual_optimizer_updates']>0,
        'Exact normal same-genesis positive frontier, never a reset')
    out.mkdir()
    actual={'kind':'ASR18_observed_normal_owner_pause_TOOL0_v1','actual_tool_exit_code':0,'actual_exec_tool_exit_code':0,
        'actual_tool_chunk_id':a.observed_tool_chunk,'actual_owned_session':a.observed_session,
        'report':report_pin,'normal_pause_snapshot':snapshot_pin,'producer':report['producer'],
        'parent_source':report['parent_source'],'configuration':config_pin,'owner_guard':guard_pin}
    actual_path=out/'observed_normal_owner_TOOL0.json';write_json(actual_path,actual)
    request={'kind':'ASR18_exact_normal_boundary_owner_resume_request_v1','configuration':config_pin,
        'source_manifest':config['source_manifest'],'parent_source':config['parent_driver_source'],
        'controller_source':config['full_owner_controller_source'],'run_identity_sha256':digest(config['full_run_identity']),
        'normal_pause_snapshot':snapshot_pin,'prior_owner_report':report_pin,
        'prior_owner_actual_TOOL0':record(actual_path),'prior_owner_guard':guard_pin}
    request_path=out/'normal_resume_request.json';write_json(request_path,request)
    print(json.dumps({'normal_request':record(request_path),'observed_actual_TOOL0':record(actual_path),
        'same_configuration':config_pin,'actual_optimizer_updates':snapshot['actual_optimizer_updates'],
        'planned_native_updates':config['planned_native_updates'],'ROOT_execution_review_still_required':True}),flush=True)

if __name__=='__main__':main()
