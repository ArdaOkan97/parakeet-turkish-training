"""Join immutable ROOT authorities into a disabled exact source-bound template."""
import argparse,ast,copy,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def assemble(authorities,output):
    inputs=read(A/'full_owner_exact_ROOT_review_inputs_v4.json');hard=read(inputs['base_hard_configuration']['path'])
    names=('durable_policy','ROOT_storage_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','ROOT_DEV_resource_review')
    assert set(authorities)==set(names)
    for name in names:
        value=authorities[name];assert pin(value['path'])==value
        review=read(value['path']);assert review['accepted'] is True and review['reviewer']=='/root'
        if 'run_identity_sha256' in review:assert review['run_identity_sha256']==inputs['run_identity_sha256']
    keys=('GPU_python','CPU_python','CPU_guard_source','CPU_controller_source','forecast_calibration',
          'entire_worker_forecast','final_recovery_worker_forecast','planning_arguments','augmentation_identity','validated_conv_only_source')
    template={key:copy.deepcopy(hard[key]) for key in keys}
    tree=ast.parse((R/'run_ASR18_full_run_owner_v9.py').read_text(encoding='utf-8-sig'))
    control=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='CONTROL_POLICY' for target in node.targets))
    storage_review=read(authorities['ROOT_storage_review']['path'])
    template.update(kind='ASR18_full_run_native_owner_configuration_v1',TEMPLATE_NOT_LAUNCHABLE=True,
        production_enabled=False,diagnostic_only=False,owner_validation_only=True,full_run_launch_enabled=False,
        planned_native_updates=401072,maximum_owner_validation_updates=8,maximum_validation_updates_per_window=4,
        maximum_owner_wall_seconds=3600,maximum_GPU_child_wall_seconds=3600,maximum_CPU_phase_seconds=480,
        maximum_native_window_seconds=600,maximum_archive_phase_seconds=120,maximum_checkpoint_phase_seconds=60,
        maximum_native_exit_seconds=60,maximum_cleanup_phase_seconds=60,maximum_recovery_phase_seconds=180,
        maximum_DEV_child_wall_seconds=900,control_log_policy=control,
        source_manifest=inputs['source_manifest'],full_run_identity=inputs['identity'],run_namespace=inputs['run_namespace'],
        parent_driver_source=inputs['parent_source'],worker_driver_source=inputs['worker_source'],
        full_owner_controller_source=inputs['owner_verifier_source'],checkpoint_storage_namespace=inputs['storage_namespace'],
        checkpoint_storage_policy=inputs['identity']['checkpoint_storage_policy'],protected_external=storage_review['protected_external'],
        qualified_source_winner=inputs['qualified_source_winner'],qualified_source_DEV_summary=inputs['qualified_source_DEV_summary'],
        journal_writer_source=pin(S/'bpe_bounded_production_journal_v4.py'),
        DEV_execution_sources=inputs['DEV_execution_sources'],DEV_native_boundary_sources=inputs['DEV_native_boundary_sources'],
        DEV_base_assets=inputs['DEV_base_assets'],DEV_source_bindings=inputs['DEV_source_bindings'],
        CPU_owned_allowance_bytes=1934217728,CPU_inner_owned_allowance_bytes=1900663296,
        CPU_outer_controller_launcher_reserve_bytes=33554432,**authorities)
    assert not output.exists();output.write_text(json.dumps(template,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'configuration_template':pin(output),'kind':'source_only_disabled_template_no_job'},ensure_ascii=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--authorities',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assemble(read(args.authorities),Path(args.output).resolve())
