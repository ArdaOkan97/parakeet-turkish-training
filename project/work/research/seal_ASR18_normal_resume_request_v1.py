"""Light immutable resume-request join after independently observed owner TOOL0."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(r'E:\gpu_training_scratchpad')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def assemble(actual_path,output):
    actual_pin=pin(actual_path);actual=read(actual_path)
    assert actual['kind']=='ASR18_observed_normal_owner_pause_TOOL0_v1' and actual['actual_tool_exit_code']==0
    report=read(actual['report']['path']);snapshot=read(actual['normal_pause_snapshot']['path']);config=read(actual['configuration']['path'])
    for key in ('report','normal_pause_snapshot','owner_guard','configuration'):assert pin(actual[key]['path'])==actual[key]
    assert report['passed'] is True and report['producer']==actual['producer']==snapshot['producer']
    assert report['configuration']==snapshot['configuration']==actual['configuration']
    assert snapshot['parent_source']==actual['parent_source']==config['parent_driver_source']
    assert output.is_relative_to(ROOT) and not output.exists()
    request={'kind':'ASR18_exact_normal_boundary_owner_resume_request_v1',
        'configuration':actual['configuration'],'source_manifest':config['source_manifest'],
        'parent_source':config['parent_driver_source'],'controller_source':config['full_owner_controller_source'],
        'run_identity_sha256':report['run_identity_sha256'],'normal_pause_snapshot':actual['normal_pause_snapshot'],
        'prior_owner_report':actual['report'],'prior_owner_actual_TOOL0':actual_pin,'prior_owner_guard':actual['owner_guard'],
        'supported_boundary':'normal_coherent_native0_CURRENT_recovery0_prefixACK_pending0_wholejournal_owner0_dead_only',
        'crash_tail_rollback_or_automatic_adoption_permitted':False,'production_enabled':False}
    output.write_text(json.dumps(request,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'resume_request':pin(output),'exact_ROOT_review_and_genuine_metadata_controls_still_required':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();assemble(a.actual,Path(a.output).resolve())
