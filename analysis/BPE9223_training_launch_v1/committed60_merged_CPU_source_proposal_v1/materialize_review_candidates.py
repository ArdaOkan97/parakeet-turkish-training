from pathlib import Path
import json,hashlib,copy
P=Path.cwd();N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,d):assert not p.exists();p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');return pin(p)
comp_pin=pin(N/'ROOT_BPE9223_resource_stop_operational_compatibility_review_v9.json');comp=read(comp_pin['path']);assert comp['accepted'] and comp['reviewer']=='/root'
old=read(N/'ROOT_BPE9223_resource_stop_operational_compatibility_review_v8.json');sources=comp['operational_sources']
remap={v['path']:sources[k]for k,v in old['operational_sources'].items()if k in sources and v!=sources[k]}
def substitute(d,m=remap):
 if isinstance(d,dict):
  if set(d)=={'path','bytes','sha256'} and d['path']in m:return copy.deepcopy(m[d['path']])
  return {k:substitute(v,m)for k,v in d.items()}
 if isinstance(d,list):return [substitute(x,m)for x in d]
 return d
A=N/'merged_CPU_configuration_candidates_v1';A.mkdir(exist_ok=False)
CPU=read(N/'operational_CPU_authorities_v2/handoff.json');new=copy.deepcopy(CPU)
# The whole stable read-only object stays byte-semantic equal to original session2.
old_session_pin=CPU['full_configuration_updates']['session'];old_session=read(old_session_pin['path']);old_num_pin=CPU['numerical_contract'];old_num=read(old_num_pin['path'])
num=substitute(old_num);num['outer_parent_source']=sources['parent'];num['resource_review'].update(controller_private_reserve_bytes=128000000,whole_owned_private_allowance_bytes=1961554432)
num_pin=write(A/'CPU_numerical_contract_candidate.json',num)
session=copy.deepcopy(old_session);session['source_context_source']=sources['CPU_source_context'];session['CPU_preparation_contract']=num_pin;session['operational_compatibility']=comp_pin;session['prior_operational_session']=old_session_pin
assert session['operational_metadata']==old_session['operational_metadata']
session_pin=write(A/'operational_CPU_session_candidate.json',session)
template=substitute(read(CPU['full_configuration_updates']['CPU_template_configuration']['path']));template['session']=session_pin;template['outer_parent_source']=sources['parent'];template['source_context_source']=sources['CPU_source_context'];sc=template['source_contract_template'];sc['numerical_CPU_contract']=num_pin
sc['resource_review']={'leaf_maximum_private_bytes':1800000000,'controller_private_reserve_bytes':128000000,'inner_shim_private_reserve_bytes':33554432,'outer_shim_private_reserve_bytes':33554432,'whole_inner_family_bytes':1961554432,'whole_family_forecast_bytes':1995108864}
sc['sources']['controller']=sources['CPU_controller'];template['sources']['parent']=sources['CPU_controller']
template_pin=write(A/'CPU_window_template_candidate.json',template)
review=substitute(read(CPU['full_configuration_updates']['ROOT_CPU_execution_review']['path']));review.update(accepted=False,reviewer='/root/memory_diagnosis',configuration=template_pin,numerical_CPU_contract=num_pin,operational_compatibility=comp_pin,outer_parent_source=sources['parent'],proposal_only_not_ROOT_authority=True)
review['resource_policy'].update(CPU_whole_family_bytes=1995108864,complete_CPU_phase_bytes=1995108864,whole_inner_family_bytes=1961554432,separate_parent_current_CPU_phase_private_bytes=128000000,separate_CPU_orchestrator_process_exists=False,child_subtree_private_allowance_bytes=1867108864)
review['scope_limits']=['Unchanged full401072/source60-to61 p4/session2-to3 metadata transition only','Merged live supervisor/orchestrator128M plus unchanged1.8GB leaf and two32MiBshims; whole1995108864<old1998217728<2GB','Both genuine numerical leaf process0/dead proofs required; typed merged task completion is not parent process0','All audio/fulltargets/augmentation3/raw128MiB/twojobs/Torch4/fourenvcaps1/System/E/OSlifetime384M unchanged']
review['source_read']='Proposed exact dependency/source pin replacements and reviewed merged role domain; ROOT acceptance still required'
review['metadata_actual0_scope']='Prior numerical source proof only; current merged task has not run'
review_pin=write(A/'CPU_static_execution_resource_review_candidate.json',review)
new.update(accepted=False,reviewer='/root/memory_diagnosis',proposal_only_not_ROOT_authority=True,operational_compatibility=comp_pin,numerical_contract=num_pin,CPU_source_contract_template=sc,original_session=old_session_pin,prior_ROOT_authority_handoff=pin(N/'operational_CPU_authorities_v2/handoff.json'))
new['full_configuration_updates']={'session':session_pin,'CPU_template_configuration':template_pin,'CPU_controller_source':sources['CPU_orchestrator'],'CPU_bridge_source':sources['CPU_bridge'],'CPU_leaf_execution_delegate_source':sources['CPU_controller'],'ROOT_CPU_execution_review':review_pin}
new['pending_ROOT_CPU_execution_review_configuration']=template_pin;new['ROOT_CPU_execution_review_created']=False
cpu_handoff=write(A/'CPU_handoff_candidate.json',new)
B=N/'merged_DEV_authority_candidates_v9';B.mkdir(exist_ok=False);DEV=read(N/'operational_DEV_authorities_v8/handoff.json');updates=substitute(DEV['full_configuration_updates']);m=dict(remap)
prior_comp=pin(N/'ROOT_BPE9223_resource_stop_operational_compatibility_review_v8.json');m[prior_comp['path']]=comp_pin
# All original416 DEV bindings retained; source additions remap to current32.
source_bindings={Path(v['path']).name:v for v in substitute(DEV['full_configuration_updates']['DEV_source_bindings']).values()};assert len(source_bindings)==448
updates['DEV_source_bindings']=source_bindings
for name in ('ROOT_DEV_policy_review','ROOT_DEV_resource_review','ROOT_DEV_static_execution_review'):
 previous=DEV['full_configuration_updates'][name];value=substitute(read(previous['path']),m)
 value.update(accepted=False,reviewer='/root/memory_diagnosis',proposal_only_not_ROOT_authority=True)
 if 'source_bindings'in value:value['source_bindings']=source_bindings
 if 'sources'in value and isinstance(value['sources'],dict) and len(value['sources'])>400:value['sources']=source_bindings
 current=write(B/(name+'_candidate.json'),value);updates[name]=current;m[previous['path']]=current
out=copy.deepcopy(DEV);out.update(accepted=False,reviewer='/root/memory_diagnosis',proposal_only_not_ROOT_authority=True,operational_compatibility=comp_pin,full_configuration_updates=updates,prior_operational_authority_handoff=pin(N/'operational_DEV_authorities_v8/handoff.json'));out['genuine_checked_policy_passed']=False
hp=write(B/'DEV_handoff_candidate.json',out)
print(json.dumps({'CPU_handoff_candidate':cpu_handoff,'DEV_handoff_candidate':hp,'CPU_template':template_pin,'session':session_pin,'numerical':num_pin,'root_acceptance_or_execution_claimed':False}))
