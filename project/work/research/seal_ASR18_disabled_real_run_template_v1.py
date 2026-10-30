"""Light stdlib-only join of reviewed authorities into a DISABLED real-run template."""
import argparse,copy,hashlib,json
from pathlib import Path
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def assemble(handoff_path,output):
    inputs=read(A/'full_owner_exact_ROOT_review_inputs_v13.json');handoff=read(handoff_path)
    old_path=A/'full_owner_normal_resume_TEMPLATE_NOT_LAUNCHABLE_v1.json';old=read(old_path)
    template=copy.deepcopy(old)
    names=('durable_policy','ROOT_storage_review','ROOT_DEV_policy_review',
        'ROOT_DEV_static_execution_review','ROOT_DEV_resource_review')
    for name in names:
        value=handoff[name];assert pin(value['path'])==value
        authority=read(value['path']);assert authority['accepted'] is True and authority['reviewer']=='/root'
        if 'run_identity_sha256' in authority:assert authority['run_identity_sha256']==inputs['run_identity_sha256']
        template[name]=value
    for key in ('maximum_owner_validation_updates','maximum_validation_updates_per_window','owner_validation_pause_after_updates'):
        del template[key]
    template.update(TEMPLATE_NOT_LAUNCHABLE=True,production_enabled=False,full_run_launch_enabled=False,
        owner_validation_only=False,planned_native_updates=401072,normal_boundary_resume_enabled=True,
        maximum_owner_wall_seconds=86400,maximum_GPU_child_wall_seconds=86400,
        full_run_identity=inputs['identity'],source_manifest=inputs['source_manifest'],run_namespace=inputs['run_namespace'],
        checkpoint_storage_namespace=inputs['storage_namespace'],parent_driver_source=inputs['parent_source'],
        worker_driver_source=inputs['worker_source'],full_owner_controller_source=inputs['owner_verifier_source'])
    storage=read(template['ROOT_storage_review']['path']);template['protected_external']=storage['protected_external']
    assert digest(template['full_run_identity'])==inputs['run_identity_sha256']
    assert not Path(template['run_namespace']).exists()
    assert not output.exists();output.write_text(json.dumps(template,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    changes={key:{'old':old.get(key),'new':template.get(key)} for key in sorted(set(old)|set(template)) if old.get(key)!=template.get(key)}
    delta={'kind':'ASR18_disabled_real_run_template_delta_v1','old_template':pin(old_path),'new_template':pin(output),
        'source':pin(__file__),'ROOT_handoff':pin(handoff_path),'ROOT_review_inputs':pin(A/'full_owner_exact_ROOT_review_inputs_v13.json'),
        'exact_top_level_delta':changes,'launch_disabled':True,'actual_jobs':0,'run_namespace_created':False,
        'legacy_fresh_invocation_token':'genesis_normal_pause4',
        'legacy_token_semantics':inputs['legacy_token_semantics'],
        'time_horizon_remains_conditional_and_no_production_execution_approval':True,
        'deployment_proposal':inputs['deployment_proposal']}
    dp=output.with_name('disabled_monitored_real_run_template_delta_v1.json')
    assert not dp.exists();dp.write_text(json.dumps(delta,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'template':pin(output),'delta':pin(dp),'source':pin(__file__)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();assemble(Path(a.handoff).resolve(),Path(a.output).resolve())
