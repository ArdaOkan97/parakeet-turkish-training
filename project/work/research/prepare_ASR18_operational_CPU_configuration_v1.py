"""Pure metadata CPU operational derivative; original science/caps unchanged.

No model, waveform, array, native process, cache or training authority is made.
The exact static template still requires a genuine separate ROOT CPU review.
"""
import argparse,copy,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility,checked_resource_stop
from fastmss_BPE9223_cpu_common_v7 import environment,resource_guard
from asr18_BPE9223_CPU_window_configuration_v6 import _template
from fastmss_BPE9223_feature_protocol_v7 import bridge
ROOT=Path(r'E:\gpu_training_scratchpad');N=ROOT/'analysis/BPE9223_training_launch_v1'
SOURCE_HANDOFF=N/'CPU_resource_preparation_v2/source_handoff_v2.json'
CONFIG=N/'full_configuration_04/configuration_ready_for_final_execution_review.json'


def read(pin):
    require(record(pin['path'])==pin and 0<pin['bytes']<=16*1024**2,'Exact bounded current metadata pin')
    resource_guard();return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))


def write(folder,name,value):
    path=folder/name
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    return record(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--compatibility',required=True);p.add_argument('--folder',required=True);a=p.parse_args()
    environment();resource_guard();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(N) and not folder.exists(),'Fresh E operational metadata output only')
    old=read(record(CONFIG));compat_pin=record(a.compatibility);compat=checked_compatibility(compat_pin,old['full_run_identity'],resource_guard)
    require(compat['original_configuration']==record(CONFIG),'Exact real production config04 lineage')
    stopped=checked_resource_stop(compat['actual_resource_stop'],identity=old['full_run_identity'],compatibility=compat_pin,guard=resource_guard)
    handoff=read(record(SOURCE_HANDOFF));src=handoff['sources'];op=compat['operational_sources']
    pairs={'bridge':'CPU_bridge','engine':'CPU_engine','protocol':'CPU_protocol','core':'CPU_core','controller':'CPU_controller',
        'reference':'CPU_reference','frontend':'CPU_frontend','source_context':'CPU_source_context','configuration_factory':'CPU_configuration_factory','CPU_resource_boundary':'CPU_resource_boundary'}
    for key,role in pairs.items():require(src[key]==op[role]==record(src[key]['path']),'Exact reviewed operational source: '+key)
    parent=op['parent'];require(parent['sha256']=='d41ac91617c96db6166a1a48488e1eaee1dae03c7ec1242b558255a74b879dae' and record(parent['path'])==parent,'Final actual parent5, never provisional parent4')
    original_session=read(old['session']);original_template=read(old['CPU_template_configuration']);original_numerical=read(original_session['CPU_preparation_contract'])
    require(bridge.frozen_augmentation(old['augmentation_identity'],resource_guard)==original_session['augmentation_identity']==handoff['augmentation_identity'], 'Original adapter3/policy/registries/scientific identity retained')
    replacements={}
    for old_name,new_name in (('fastmss_BPE9223_plan_bridge_v3.py','bridge'),('fastmss_BPE9223_feature_engine_v3.py','engine'),
        ('fastmss_BPE9223_feature_protocol_v3.py','protocol'),('fastmss_BPE9223_cpu_common_v3.py','core'),
        ('run_ASR18_BPE9223_CPU_window_v3.py','controller'),('produce_FastMSS_BPE9223_reference_CPU_v3.py','reference'),
        ('produce_FastMSS_BPE9223_frontend_CPU_v3.py','frontend'),('asr18_BPE9223_source_context_v2.py','source_context')):
        old_path=(P/'work/research' if old_name.startswith(('run_','produce_')) else S)/old_name;replacements[str(old_path)]=src[new_name]
    replacements[old['parent_driver_source']['path']]=parent
    def rebind(value):
        if type(value) is dict:
            if set(value)=={'path','bytes','sha256'} and value['path'] in replacements:
                require(record(value['path'])==value,'Original source pin remains unchanged');return copy.deepcopy(replacements[value['path']])
            return {key:rebind(item) for key,item in value.items()}
        if type(value) is list:return [rebind(item) for item in value]
        return value
    numerical=rebind(original_numerical)
    require(numerical['resource_review']==original_numerical['resource_review'] and numerical['frontend']==original_numerical['frontend']
        and numerical['python']==original_numerical['python'] and numerical['controller_validation_runtime']==original_numerical['controller_validation_runtime']
        and numerical['augmentation_identity']==old['augmentation_identity'] and numerical['feature_capacity_bytes']==536870912
        and numerical['raw_PCM_capacity_bytes']==134217728 and numerical['raw_PCM_maximum_jobs']==2,'Exact unchanged numerical/runtime/resource domains')
    folder.mkdir();numerical_pin=write(folder,'CPU_numerical_contract.json',numerical)
    session=rebind(original_session);session.update(CPU_preparation_contract=numerical_pin,configuration_assembler_source=record(__file__),
        original_session=old['session'],operational_compatibility=compat_pin,operational_metadata={'storage_owner':stopped['storage_state']['owner'] if 'owner' in stopped['storage_state'] else read(stopped['storage_state'])['owner'],
            'control_root':str(Path(old['run_namespace'])/'control'),'identity':old['full_run_identity'],
            'operational_compatibility':compat_pin,'current_native_resource_closure':None})
    # The original session's creation-time prediction remains in its immutable
    # lineage. This successor can bind the genuine already-created owner.
    require(record(original_session['durable_owner']['path'])==original_session['durable_owner'],'Genuine existing durable creator owner')
    session['durable_owner_is_predicted_not_actual_until_runtime_creation']=False
    session_pin=write(folder,'operational_CPU_session.json',session)
    template=rebind(original_template);template['session']=session_pin;template['source_contract_template']['numerical_CPU_contract']=numerical_pin
    require(template['full_run_identity_sha256']==digest(old['full_run_identity'])==original_template['full_run_identity_sha256']
        and template['source_contract_template']['resource_review']==original_template['source_contract_template']['resource_review']
        and template['checkpoint_storage_namespace']==old['checkpoint_storage_namespace'],'Same scientific identity/caps/registered original namespace')
    contract_pin=write(folder,'CPU_source_contract_template.json',template['source_contract_template'])
    template_pin=write(folder,'CPU_window_template.json',template);_template(template_pin,resource_guard)
    require(bridge.validate_session_scope(session)==bridge.FULL_MODE and session['initial_sampler_sha256']==old['initial_sampler_sha256']
        and session['label_population_contract']==original_session['label_population_contract'],'Unchanged full2broad+cleaner origin/labels')
    # Compare complete metadata trees after only declared source/file-pin changes.
    expected=copy.deepcopy(rebind(original_template));expected['session']=session_pin;expected['source_contract_template']['numerical_CPU_contract']=numerical_pin
    require(template==expected,'No undeclared static-template delta')
    updates={'session':session_pin,'CPU_template_configuration':template_pin,'CPU_controller_source':src['controller'],'CPU_bridge_source':src['bridge']}
    result={'kind':'ASR18_operational_CPU_configuration_metadata_handoff_v1','passed':True,'producer':identity(),'source':record(__file__),
        'original_configuration':record(CONFIG),'original_session':old['session'],'original_template':old['CPU_template_configuration'],
        'original_numerical_contract':original_session['CPU_preparation_contract'],'operational_compatibility':compat_pin,'CPU_source_handoff':record(SOURCE_HANDOFF),
        'full_configuration_updates':updates,'numerical_contract':numerical_pin,'CPU_source_contract_template':contract_pin,'augmentation_identity':old['augmentation_identity'],
        'pending_ROOT_CPU_execution_review_configuration':template_pin,'ROOT_CPU_execution_review_created':False,
        'resource_domains_numerical_frontend_original_policy_scientific_identity_preserved':True,'actual_audio_cache_native_GPU_jobs':0,'production_or_CPU_execution_admitted':False}
    pin=write(folder,'handoff.json',result);require(not any(k=='torch' or k.startswith('torch.') for k in sys.modules),'No Torch/model imports')
    print(json.dumps(pin),flush=True)


if __name__=='__main__':main()
