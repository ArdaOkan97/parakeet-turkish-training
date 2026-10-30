"""Pure source/configuration derivatives for ROOT; never an execution authority."""
import copy,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from run_ASR18_BPE9223_full_run_owner_v19 import storage,sample_memory,fresh_admit
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')

def main():
    storage();began=time.monotonic();calls=0;peak=0
    def guard():
        nonlocal calls,peak
        value=sample_memory(os.getpid(),'saved260_authority_candidate_metadata');calls+=1
        peak=max(peak,value['worker_private_memory']['private_bytes']);system=value['system_memory']
        require(time.monotonic()-began<180 and peak<128000000 and value['paired_sampling_within_one_second']
            and system['commit_fraction']<.95 and system['commit_available_bytes']>2000000000,'Same128M/180s/system95/headroom')
    for _ in range(3):require(fresh_admit(128000000)['passed'],'Fresh metadata admission');time.sleep(.2)
    def load(pin):
        guard();require(record(pin['path'])==pin,'Immutable metadata input');return read_json(pin['path'])
    def save(path,value):
        guard();require(not path.exists(),'New immutable metadata only');write_json(path,value);return record(path)
    previous_pin=record(N/'fresh_restart_07/configuration_ready_for_execution.json');previous=load(previous_pin)
    compatibility=record(N/'original_audio260_operational_bundle_v3/ROOT_compatibility_v2.json');current=load(compatibility)
    active=load(current['operational_source_manifest']);original=load(current['original_source_manifest'])
    original_sources={v['path'] for v in original['bindings']}
    proposed=load(record(N/'original_audio260_operational_bundle_v3/proposed_compatibility.json'))
    cohort=load(record(N/'original_Opus260_CPU_source_v1/source_handoff_v1.json'))
    pairs={v['old']['path']:v['new'] for v in cohort['source_pairs']}
    for old_role,new_role in (('full_owner_controller_source','controller'),('worker_driver_source','runtime')):
        pairs[previous[old_role]['path']]=current['operational_sources'][new_role]
    prior_review=load(previous['operational_compatibility'])
    for role in ('native_engine','prefix','durable_adapter'):
        pairs[prior_review['operational_sources'][role]['path']]=current['operational_sources'][role]
    pairs[str(P/'outputs/parakeet_turkish/fastmss_bounded_original_audio_v1.py')]=record(P/'outputs/parakeet_turkish/fastmss_bounded_original_audio_v2.py')
    def remap(value):
        if isinstance(value,dict):
            if set(value)=={'path','bytes','sha256'} and value['path'] in pairs:return copy.deepcopy(pairs[value['path']])
            return {key:remap(item) for key,item in value.items()}
        if isinstance(value,list):return [remap(item) for item in value]
        return value
    out=N/'original_audio260_authority_candidates_v1';require(not out.exists(),'New candidate folder');out.mkdir()
    old_session=load(previous['session']);old_template=load(previous['CPU_template_configuration'])
    old_numerical=load(old_template['source_contract_template']['numerical_CPU_contract'])
    numerical=remap(old_numerical);numerical_pin=save(out/'CPU_numerical_contract.json',numerical)
    session=remap(old_session);session.update(CPU_preparation_contract=numerical_pin,operational_compatibility=compatibility,
        original_audio260_prior_session=previous['session'],configuration_assembler_source=record(__file__))
    require(session['operational_metadata']==old_session['operational_metadata']==previous['process_readonly_metadata_authority']
        and session['durable_owner']==old_session['durable_owner'],'Exact stable readonly binding and actual creator retained')
    session_pin=save(out/'CPU_session.json',session)
    template=remap(old_template);template['session']=session_pin;template['source_contract_template']['numerical_CPU_contract']=numerical_pin
    template_pin=save(out/'CPU_window_template.json',template)
    CPU_review=remap(load(previous['ROOT_CPU_execution_review']))
    CPU_review.update(accepted=False,reviewer='PROPOSED_FOR_ROOT_REVIEW_ONLY',configuration=template_pin,
        numerical_CPU_contract=numerical_pin,source_context=current['operational_sources']['CPU_source_context'],
        operational_compatibility=compatibility,task_controller_source=current['operational_sources']['CPU_orchestrator'],
        prior_exact_CPU_review=previous['ROOT_CPU_execution_review'],actual_new_CPU_window_pass_claimed=False,
        GPU_launch_authorized_by_this_receipt=False)
    CPU_review_pin=save(out/'proposed_ROOT_CPU_execution_review.json',CPU_review)
    base_DEV={k:v for k,v in previous['DEV_source_bindings'].items() if v['path'] in original_sources}
    require(len(base_DEV)==416,'Exact original416 DEV sources')
    additions={Path(v['path']).name:v for v in active['bindings'] if v['path'] not in original_sources}
    require(len(additions)==32 and not set(base_DEV)&set(additions),'Exactly32 current operational sources')
    DEV_bindings={**base_DEV,**additions};DEV_execution=remap(previous['DEV_execution_sources']);DEV_boundary=remap(previous['DEV_native_boundary_sources'])
    DEV_policy=remap(load(previous['ROOT_DEV_policy_review']));DEV_resource=remap(load(previous['ROOT_DEV_resource_review']))
    for value in (DEV_policy,DEV_resource):
        value.update(accepted=False,reviewer='PROPOSED_FOR_ROOT_REVIEW_ONLY',operational_compatibility=compatibility,
            proposal_only_not_ROOT_authority=True,prior_operational_authority=previous['ROOT_DEV_policy_review'])
    DEV_policy['native_boundary_sources']=DEV_boundary;DEV_policy['execution_sources']=DEV_execution
    DEV_policy_pin=save(out/'proposed_ROOT_DEV_policy_review.json',DEV_policy)
    DEV_resource_pin=save(out/'proposed_ROOT_DEV_resource_review.json',DEV_resource)
    DEV_static=remap(load(previous['ROOT_DEV_static_execution_review']))
    DEV_static.update(accepted=False,reviewer='PROPOSED_FOR_ROOT_REVIEW_ONLY',operational_compatibility=compatibility,
        proposal_only_not_ROOT_authority=True,ROOT_policy_review=DEV_policy_pin,resource_review=DEV_resource_pin,
        source_bindings=DEV_bindings,execution_sources=DEV_execution,prior_operational_authority=previous['ROOT_DEV_static_execution_review'])
    DEV_static_pin=save(out/'proposed_ROOT_DEV_static_execution_review.json',DEV_static)
    handoff={'kind':'ASR18_saved260_CPU_DEV_source_configuration_candidates_for_ROOT_v1','prior_configuration':previous_pin,
        'genuine_compatibility':compatibility,'source':record(__file__),'numerical_CPU_contract':numerical_pin,
        'CPU_review_candidate':CPU_review_pin,'DEV_policy_candidate':DEV_policy_pin,'DEV_resource_candidate':DEV_resource_pin,
        'DEV_static_candidate':DEV_static_pin,'full_configuration_candidates':{'session':session_pin,
            'CPU_template_configuration':template_pin,'CPU_controller_source':current['operational_sources']['CPU_orchestrator'],
            'CPU_bridge_source':current['operational_sources']['CPU_bridge'],
            'CPU_leaf_execution_delegate_source':current['CPU_leaf_execution_delegate_source'],
            'DEV_source_bindings':DEV_bindings,'DEV_execution_sources':DEV_execution,'DEV_native_boundary_sources':DEV_boundary},
        'ROOT_acceptance_or_GPU_execution_claimed':False,'current_run_namespace_mutation':False,'guard_calls':calls,
        'actual_current_private_sample_peak_bytes':peak,'wall_seconds':time.monotonic()-began}
    handoff_pin=save(out/'handoff.json',handoff);print(json.dumps(handoff_pin),flush=True)

if __name__=='__main__':main()
