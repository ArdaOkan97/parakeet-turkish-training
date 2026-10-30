"""ROOT-owned metadata/empty-index preparation for explicitly requested restart."""
from pathlib import Path
import copy, hashlib, json, sys
P=Path(__file__).resolve().parents[2]; S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S));sys.path.insert(0,str(P/'work/research'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_indexed_checkpoint_storage_v1 import IndexedOwnedFutureCheckpointStore as Creator
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
D=N/'fresh_restart_07'; RUN=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_weak2_clean1_20261018_07')
oldroot=r'E:\gpu_training_scratchpad\runs\BPE9223_weak2_clean1_20261018_04'
prior=read_json(N/'fresh_restart_04_v2/configuration_ready_for_execution.json')
original=read_json(N/'full_configuration_04/configuration_pending_execution_review.json')
require(not D.exists() and not RUN.exists(),'Fresh output and fresh run only')
# Verify physical retirement as actual child0/outer1, never relabel it TOOL0.
retirement_pin=record(N/'failed_RUN06_feature_retirement_v1/actual_child0_outer_TOOL1_closed_v1.json')
require(retirement_pin['sha256']=='81e8bb588610cad099fea9b314750f6f0d3905120a52a7795f9863047de90f0f','Exact observed retirement outcome')
retirement=read_json(retirement_pin['path'])
require(retirement['actual_tool_exit_code']==1 and retirement['actual_child_guard_exit_code']==0
    and retirement['normal_owned_process0_claimed'] is False and retirement['physical_retirement_completed'] is True
    and retirement['retired_exact_owned_files']==682 and retirement['all_captured_identities_dead'] is True,
    'Preserve actual child0 and outer1; physical exact owned cleanup complete')
from fastmss_owned_process_closure_v2 import dead
require(all(dead(v) for v in retirement['all_captured_identities']),'Actual closed retirement identities')
for key in ('report','guard_closure','metadata_archive'):
    require(record(retirement[key]['path'])==retirement[key],'Actual retirement evidence binding')
for pin in retirement['protected_saved20_metadata_unchanged']:
    require(record(pin['path'])==pin,'Previous saved20/journal/storage metadata preserved')
cache=Path(r'E:\gpu_training_scratchpad\temp\BPE9223_recorded_room_CPU_windows_v1')
require(cache.exists() and not any(cache.iterdir()),'Actual shared transient feature slot empty')
D.mkdir(); mapping={}; previous_identity=prior['full_run_identity']
roles=copy.deepcopy(read_json(prior['operational_compatibility']['path'])['operational_sources'])
roles.update(controller=record(S/'asr18_BPE9223_full_run_controller_v21.py'),
    durable_adapter=record(S/'asr18_BPE9223_durable_owner_adapter_v10.py'),
    prefix=record(S/'asr18_BPE9223_persistence_prefix_v13.py'),
    indexed_storage=record(S/'asr18_BPE9223_indexed_checkpoint_storage_v2.py'),
    fresh_mutable_indexed_storage=record(S/'asr18_BPE9223_indexed_checkpoint_storage_v3.py'),
    worker_control_pool=record(S/'asr18_worker_control_pool_v3.py'))
roles.pop('cleanup_reconciliation',None)
roles['runtime']=record(P/'work/research/asr18_BPE9223_owned_GPU_runtime_v25.py')
roles['native_engine']=record(S/'asr18_BPE9223_full_run_native_v10.py')
cohort=read_json(N/'ordinary60_bridge_source_v1/source_handoff_v3.json')['cohort']
for item in cohort:
    require(item['full_source_and_AST_inverse_equal'] is True,'Only declared nonnumerical source corrections')
    mapping[item['previous']['path']]=item['source']
for role,pin in list(roles.items()):roles[role]=copy.deepcopy(mapping.get(pin['path'],pin))
controller=roles['controller']; prefix=roles['prefix']; identity=copy.deepcopy(previous_identity)
identity['permanent_namespace_scope']='human_authorized_BPE9223_two_broad_one_cleaner_restart07'
identity['persistent_prefix_sources'].update(durable_adapter=roles['durable_adapter'],prefix_verifier=prefix,
    metadata_resolver=controller,native_worker=roles['runtime'],native_parent=roles['parent'],
    recovery_worker=roles['runtime'],recovery_parent=roles['parent'])
identity_sha=digest(identity)
def remap(v):
    if v==previous_identity:return copy.deepcopy(identity)
    if isinstance(v,dict):
        if 'path'in v and 'sha256'in v and 'bytes'in v:
            return copy.deepcopy(mapping.get(v['path'],v))
        return {k:remap(x) for k,x in v.items()}
    if isinstance(v,list):return [remap(x) for x in v]
    if isinstance(v,str):
        if v==digest(previous_identity):return identity_sha
        if v.startswith(oldroot):return str(RUN)+v[len(oldroot):]
    return v
def save(name,v,old=None):
    path=D/name;write_json(path,v);pin=record(path)
    if old is not None:mapping[old['path']]=pin
    return pin
for role in ('controller','prefix','durable_adapter','runtime','native_engine'):
    old=read_json(prior['operational_compatibility']['path'])['operational_sources'][role]
    mapping[old['path']]=roles[role]
manifest=read_json(prior['source_manifest']['path']); base=read_json(identity['full_run_source_contract']['path'])
omit={'asr18_BPE9223_durable_owner_adapter_v4.py','reconcile_ASR18_BPE9223_cleanup_join_v2.py','asr18_BPE9223_current_recovery_v4.py'}
bindings={x['path']:x for x in (remap(x) for x in manifest['bindings']) if Path(x['path']).name not in omit}
for x in roles.values():bindings[x['path']]=x
x=record(S/'fastmss_BPE9223_resource_CPU_boundary_v1.py');bindings[x['path']]=x
require(len(bindings)<=len(base['bindings'])+32 and all(bindings.get(x['path'])==x for x in base['bindings']),
    'Original source432 plus unchanged32 successor cap')
require(all(record(x['path'])==x for x in bindings.values()),'All actual unchanged/current source pins')
manifest=copy.deepcopy(manifest);manifest['bindings']=list(bindings.values());manifest['fresh_restart_from_original_sources']=True
manifest_pin=save('active_source_manifest.json',manifest,prior['source_manifest'])
proofs={'verify_pending_window':controller,'verify_writer_authority':controller,'verify_durable_resume':controller,
    'verify_current_recovery_and_persistence_prefix':prefix}
old_durable=read_json(prior['durable_policy']['path'])
durable={k:copy.deepcopy(old_durable[k]) for k in ('kind','transition_source','maximum_emergency_publications',
    'maximum_index_bytes','maximum_pending_publications','maximum_pending_windows','maximum_record_bytes',
    'maximum_records','maximum_stream_bytes','normal_publication_cadence','pending_capacity_supports_complete_longest_pass')}
durable.update(accepted=True,reviewer='/root',scope='full_run_integration_validation',production_enabled=False,
    namespace=str(RUN/'durable'),run_identity_sha256=identity_sha,source=roles['durable_adapter'],
    proof_callback_sources=proofs,lifecycle_callback_sources={k:controller for k in old_durable['lifecycle_callback_sources']},
    fresh_genesis_authorized=True,human_requested_restart_from_original_seeds=True,prior_numeric_policy=prior['durable_policy'])
durable_pin=save('durable_policy.json',durable,prior['durable_policy'])
storage=remap(read_json(prior['ROOT_storage_review']['path']))
storage.update(future_namespace=str(RUN/'checkpoints'),run_identity_sha256=identity_sha,owner_verifier_source=controller,
    fresh_genesis_authorized=True,actual_audio_native_DEV_or_namespace_execution_admitted=True,
    production_approval_claimed=False,scope='human_authorized_fresh_restart_same_reviewed_storage_and_guards',
    not_approved=['Adoption or deletion of previous run artifacts','Relaxation of existing capacity/resource/retention guards'])
storage_pin=save('ROOT_storage_review.json',storage,prior['ROOT_storage_review'])
original_session=copy.deepcopy(read_json(prior['session']['path']))
for key in ('operational_metadata','operational_compatibility','original_session','prior_operational_session'):original_session.pop(key,None)
predicted_value={'kind':'ASR18_future_packed_durable_lifecycle_owner_v2','policy':durable_pin,
 'source':roles['durable_adapter'],'transition_source':durable['transition_source'],'identity':identity,
 'proof_sources':proofs,'adopt_existing_artifacts':False}
predicted_raw=(json.dumps(predicted_value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
original_session.update(full_run_identity_sha256=identity_sha,durable_owner={'path':str(RUN/'durable/owner.json'),
 'bytes':len(predicted_raw),'sha256':hashlib.sha256(predicted_raw).hexdigest()},
 durable_owner_is_predicted_not_actual_until_runtime_creation=True)
original_session_pin=save('original_fresh_CPU_session.json',original_session)
base_config=copy.deepcopy(original);base_config.update(full_run_identity=identity,durable_policy=durable_pin,
    run_namespace=str(RUN),checkpoint_storage_namespace=str(RUN/'checkpoints'),session=original_session_pin)
# This is a source/guard creator contract, not a claim that legacy parent2 is launched.
base_config_pin=save('fresh_original_creator_contract.json',base_config)
comp=remap(read_json(prior['operational_compatibility']['path']))
for key in list(comp):
    if key.startswith('prior_') or key in ('actual_resource_stop','process_readonly_metadata_authority'):del comp[key]
comp.update(original_configuration=base_config_pin,original_source_manifest=identity['full_run_source_contract'],
    operational_source_manifest=manifest_pin,operational_sources=roles,scientific_identity_sha256=identity_sha,
    legacy_proof_callback_sources=proofs,operational_proof_callback_sources=proofs,
    legacy_lifecycle_callback_sources=durable['lifecycle_callback_sources'],operational_lifecycle_callback_sources=durable['lifecycle_callback_sources'],
    legacy_storage_verifier_source=controller,fresh_storage_verifier_source=controller,
    fresh_genesis_authorized=True,fresh_genesis_namespace=str(RUN),
    human_restart_replaces_only_new_run_updates_not_original_mature_Adam=True)
ro=copy.deepcopy(comp);ro['operational_sources']['indexed_storage']=record(S/'asr18_BPE9223_indexed_checkpoint_storage_v2.py')
ro['operational_sources']['CPU_resource_boundary']=record(S/'fastmss_BPE9223_resource_CPU_boundary_v1.py')
ro_pin=save('ROOT_readonly_metadata_compatibility.json',ro)
RUN.mkdir()
created=Creator.create(RUN/'checkpoints',prior['checkpoint_storage_policy'],ROOT_review=storage_pin,run_identity=identity,
    verifier_source=controller,protected=storage['protected_external'],guard=lambda:None)
owner=created.marker_pin;created.close()
empty={'owner':owner,'state':record(RUN/'checkpoints/state.json'),'catalog':record(RUN/'checkpoints/catalog.sqlite')}
authority={'storage_owner':owner,'control_root':str(RUN/'control'),'identity':identity,
    'operational_compatibility':ro_pin,'current_native_resource_closure':None}
comp['process_readonly_metadata_authority']=authority
comp_pin=save('ROOT_fresh_operational_compatibility.json',comp,prior['operational_compatibility'])
# CPU numerical code, capacities and parent services stay exactly unchanged.
session_old=read_json(prior['session']['path']);contract_old=session_old['CPU_preparation_contract']
contract_pin=save('CPU_numerical_contract.json',remap(read_json(contract_old['path'])),contract_old)
durable_owner={'kind':'ASR18_future_packed_durable_lifecycle_owner_v2','policy':durable_pin,
    'source':roles['durable_adapter'],'transition_source':durable['transition_source'],'identity':identity,
    'proof_sources':proofs,'adopt_existing_artifacts':False}
raw=(json.dumps(durable_owner,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
expected_owner={'path':str(RUN/'durable/owner.json'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
session=remap(session_old)
for key in ('original_session','prior_operational_session'):session.pop(key,None)
session.update(original_session=original_session_pin,durable_owner=expected_owner,durable_owner_is_predicted_not_actual_until_runtime_creation=True,
    full_run_identity_sha256=identity_sha,CPU_preparation_contract=contract_pin,operational_compatibility=comp_pin,
    operational_metadata=authority,initial_optimizer_updates=0,actual_optimizer_updates=0,actual_authoritative_draws_consumed=0)
session_pin=save('CPU_session.json',session,prior['session'])
template=remap(read_json(prior['CPU_template_configuration']['path']))
template_pin=save('CPU_window_template.json',template,prior['CPU_template_configuration'])
cpu_review=remap(read_json(prior['ROOT_CPU_execution_review']['path']))
cpu_review.update(configuration=template_pin,numerical_CPU_contract=contract_pin,new_run_identity_sha256=identity_sha,
    operational_compatibility=comp_pin,human_authorized_fresh_restart=True,
    scope_limits=['Fresh RUN07 and its exact normal continuations; all original numerical/resource/CPU guards unchanged',
        'One serial leaf with real reference/frontend process0 before native; all protected splits clean'])
cpu_review_pin=save('ROOT_static_CPU_execution_resource_review.json',cpu_review,prior['ROOT_CPU_execution_review'])
# Derive existing DEV identities/source bindings without changing scorer, members or numerical budgets.
for key,name in (('ROOT_DEV_policy_review','ROOT_DEV_policy_review.json'),
                 ('ROOT_DEV_resource_review','ROOT_DEV_resource_review.json'),
                 ('ROOT_DEV_static_execution_review','ROOT_DEV_static_execution_review.json')):
    save(name,remap(read_json(prior[key]['path'])),prior[key])
c=remap(prior)
for key in list(c):
    if key.startswith('prior_') or key in ('resource_native_closure','current_cleanup_join_failure',
        'cleanup_reconciliation_process0','original_genesis_configuration','fresh_empty_storage_bindings'):
        c.pop(key,None)
c.update(full_run_identity=identity,source_manifest=manifest_pin,run_namespace=str(RUN),
    checkpoint_storage_namespace=str(RUN/'checkpoints'),operational_storage_owner=owner,
    operational_control_root=str(RUN/'control'),operational_compatibility=comp_pin,
    process_readonly_metadata_authority=authority,full_owner_controller_source=controller,
    fresh_genesis_authorized=True,fresh_genesis_namespace=str(RUN),session=session_pin,
    CPU_template_configuration=template_pin,ROOT_CPU_execution_review=cpu_review_pin,
    full_run_launch_enabled=True,configuration_assembly_source=record(__file__))
# Exercise the real frozen CPU session gate, including the real readonly store.
from contextlib import contextmanager
from fastmss_BPE9223_resource_CPU_boundary_v2 import configure_session
import asr18_BPE9223_operational_metadata_v2 as metadata
control=RUN/'control';require(not control.exists(),'Fresh temporary empty control preflight')
control.mkdir()
try:
    require(configure_session(session,lambda:None) is True,'Actual full CPU metadata boundary accepted')
    # Exact real constructor source joins, before any producer can be admitted.
    import inspect
    import run_ASR18_BPE9223_full_run_owner_v19 as parent
    import run_ASR18_BPE9223_CPU_leaf_observer_v11 as leaf
    from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility
    current=checked_compatibility(c['operational_compatibility'],identity)
    require(c['CPU_leaf_execution_delegate_source']==current['CPU_leaf_execution_delegate_source']
        ==record(inspect.getsourcefile(leaf.run_leaf)),'Exact actual leaf11 config/ROOTcompat/Popen-code source')
    require(c['CPU_execution_delegate_source']==current['CPU_execution_delegate_source']
        ==record(inspect.getsourcefile(parent.frozen_CPU_executor.run_CPU)),'Exact original cleanup executor source')
    for field,role in (('CPU_controller_source','CPU_orchestrator'),('full_owner_controller_source','controller'),
        ('worker_driver_source','runtime'),('parent_driver_source','parent'),('CPU_bridge_source','CPU_bridge')):
        require(c[field]==current['operational_sources'][role] and record(c[field]['path'])==c[field],
            'Full owner constructor actual configuration/source role join: '+field)
    require(session['source_context_source']==current['operational_sources']['CPU_source_context'],
        'Exact current numerical source context role')
    import ast
    current_controller=__import__('asr18_BPE9223_full_run_controller_v21')
    require(record(current_controller.__file__)==controller,'Actual current cleanup/owner module source')
    cleanup_source=inspect.getsource(current_controller.cleanup_child)
    require('from fastmss_BPE9223_feature_protocol_v11 import cleanup_window, read_bound' in cleanup_source
        and 'fastmss_BPE9223_feature_protocol_v9' not in cleanup_source,
        'Actual cleanup function uses same protocol11 as producer owner')
    main_source=inspect.getsource(current_controller.main)
    require('cleanup_script=Path(__file__).resolve()' in main_source,
        'Actual CLI passes its own cleanup worker21 source')
    emptylog_source=inspect.getsource(current_controller.FullOwner._seal_closed_empty_CPU_logs)
    require("add(root/'cleanup/worker.log',self.current_cleanup0,record(self.cleanup_script),self.CPU_execution_delegate_source)" in emptylog_source,
        'Exact worker21 and frozen6 cleanup observer/emptylog join')
    save('actual_cleanup_source_constructor_checks.json',{'controller':controller,'protocol':record(S/'fastmss_BPE9223_feature_protocol_v11.py'),
        'copied_cleanup_body_source_review':record(N/'ordinary60_bridge_source_v1/ROOT_controller21_cleanup_source_read_review_v1.json'),
        'actual_worker_source_is_current_controller':True,'actual_observer_source':c['CPU_execution_delegate_source'],
        'prior_real_owner11_validation_and_retirement':retirement_pin,'normal_live_cleanup_execution_claimed':False})
    from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
    dev_dir=D/'actual_DEV_callback_constructor_preflight';dev_dir.mkdir()
    SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
        execute_owned=parent.execute_owned_DEV,execute_owned_source=c['parent_driver_source'],
        native_boundary=parent.native_boundary_DEV,native_boundary_source=c['parent_driver_source'],
        native_boundary_sources=c['DEV_native_boundary_sources'],output_directory=dev_dir)

    from asr18_BPE9223_source_context_v8 import FullRunSourceContext9223
    from fastmss_BPE9223_plan_bridge_v11 import bridge_plan, same
    from asr18_BPE9223_CPU_window_configuration_v10 import derive_contract, derive_configuration
    context=FullRunSourceContext9223(c['planning_arguments'],guard=lambda:None)
    try:
        schedule=context.new_schedule();initial_sha=digest(schedule.state_dict())
        plan=context.plan(schedule,completed_boundaries=0)
        require(schedule.committed_updates()==0 and digest(schedule.state_dict())==initial_sha==c['initial_sampler_sha256'],
            'Only private first-window planning, no scientific frontier commit')
        plan_pin=save('actual_first20_private_plan.json',plan)
        bridge=bridge_plan(plan_pin,DATA_evidence=c['planning_arguments']['DATA_evidence'],
            session_binding=session_pin,window_index=0,prior_window_boundary=None,guard=lambda:None)
        bridge_pin=save('actual_first20_bridge.json',bridge)
        contract=derive_contract(template_pin,bridge_pin,guard=lambda:None)
        contract_pin=save('actual_first20_derived_CPU_contract.json',contract)
        derived=derive_configuration(template_pin,contract_pin,0,'p000000001',
            RUN/'checkpoints/transient/p000000001/CPU/payload',guard=lambda:None)
        save('actual_first20_derived_CPU_configuration.json',derived)
        require(len(bridge['updates'])==20 and derived['source_contract']==contract_pin,
            'Real new bridge/engine/protocol/factory metadata APIs and all static source hashes')
        c['prepared_metadata_plan']=plan_pin;c['actual_metadata_bridge']=bridge_pin
    finally:context.close()
    import ast
    bridge_source=read_json(N/'ordinary60_bridge_source_v1/source_handoff_v3.json')['cohort'][0]['source']
    tree=ast.parse(Path(bridge_source['path']).read_text(encoding='utf-8-sig'))
    expressions=[node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='merged_transition' for t in node.targets)]
    require(len(expressions)==1,'One actual source transition predicate')
    expression=compile(ast.Expression(body=expressions[0]),bridge_source['path'],'eval')
    cases=[]
    for wi,updates,prior_session,expected in ((3,60,session_pin,False),(4,80,session_pin,False),
        (3,60,prior['session'],True),(2,40,session_pin,False)):
        value=eval(expression,{'same':same},{'window_index':wi,'plan':{'initial_committed_updates':updates},
            'prior':{'session':prior_session},'session_binding':session_pin})
        require(value is expected,'Ordinary same-session window must not enter historical hardcoded migration')
        cases.append({'window_index':wi,'updates':updates,'same_session':same(prior_session,session_pin),
            'historical_transition_selected':value})
    save('actual_source_transition_predicate_checks.json',{'source':bridge_source,'cases':cases,
        'pure_predicate_only_no_actual_future_native_completion_claimed':True})

finally:
    if metadata._RESOLVER is not None:metadata._RESOLVER.store.close()
    require(control.is_dir() and not any(control.iterdir()),'Only explicitly created empty temporary control may retire')
    control.rmdir()
require({x.name for x in RUN.iterdir()}=={'checkpoints'},'Fresh launch root remains checkpoints only')
empty={'owner':owner,'state':record(RUN/'checkpoints/state.json'),'catalog':record(RUN/'checkpoints/catalog.sqlite')}
save('configuration_before_actual0_seal.json',c)
result={'kind':'ROOT_actual_fresh_restart_metadata_and_empty_store_preparation_v1','passed':True,
    'configuration_before_seal':record(D/'configuration_before_actual0_seal.json'),
    'empty_storage_bindings':empty,'source_manifest':manifest_pin,'operational_compatibility':comp_pin,
    'source':record(__file__),'old_run_preserved':oldroot,'new_run':str(RUN),'identity_sha256':identity_sha,
    'original_mature_sources_and_scientific_schedule_unchanged':True,'actual_optimizer_updates':0,'actual_GPU_jobs':0,'actual_frozen_CPU_configure_session_preflight':True,'actual_first_window_bridge_and_factory_metadata_preflight':True,'ordinary60_80_same_session_predicate_preflight':True,'actual_constructor_source_joins_and_DEV_callback_preflight':True,'cleanup_protocol11_actual_source_joins':True,'previous_failed_feature_retirement':retirement_pin}
save('metadata_preparation_report.json',result)
print(json.dumps(result),flush=True)
