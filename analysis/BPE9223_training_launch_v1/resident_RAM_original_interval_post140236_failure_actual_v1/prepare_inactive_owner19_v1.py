"""Metadata/source only: freeze a narrowly typed saved140200 continuation."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).parent;N=F.parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 p=Path(p);assert not p.exists(),p;p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def replace(text,old,new,count=1):
 assert text.count(old)==count,(old,text.count(old),count);return text.replace(old,new)
def delta(path,a,b):
 Path(path).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='frozen_old',tofile='inactive_new')),encoding='utf-8')
new_checker='''def checked_original_interval_resume(pin,*,base,folder,pointer_pin):
    """Exact source18 EOF140236, saved140200, all36 positive attempts retained."""
    import psutil
    require(pin==base['prior_resident_RAM_original_interval_failure']and record(pin['path'])==pin
        and pin['sha256']=='07f2949086beb88bc113e922e4421f67c9796030a57fc56ddaa2d13e043fe078',
        'Exact ROOT-bound actual source18 EOF failure; no normal0 relabel')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_original_interval_post140236_failure_closed_v1'
        and value['owned_session']==5043 and value['actual_outer_TOOL_exit_code']==1 and value['actual_TOOL_chunk']=='80fa9e'
        and value['actual_optimizer_updates']==140236 and value['last_fully_saved_updates']==140200
        and value['unsaved_positive_tail_updates']==36 and value['CPU_failure_completed_prepared_updates']==236
        and value['native_numeric_OS_exit_code']is None and value['CPU_numeric_OS_exit_code']is None
        and value['native_Popen_wait_known_nonzero_from_wrapper_assertion']is True
        and value['no_actual_resource_admission_denial']is True,
        'Truthful actual TOOL1/unknown OS codes; saved140200 and36 historical positives')
    require(e['pointer']==pointer_pin and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
        and e['source18']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v18.py')
        and e['CPU7_source']==record(S/'fastmss_resident_feature_stream_v7.py'),
        'Exact failed source18/CPU7 and original checkpoint branch')
    for item in e.values():require(record(item['path'])==item,'Exact compact frozen failed-owner/source/checkpoint evidence')
    identities=value['all_recorded_process_identities_dead']
    expected=[('native',3588,1792670466.9104328),('native_interpreter_launcher',38980,1792670466.9041212),
        ('CPU_producer',26072,1792670575.3371205),('CPU_interpreter_launcher',26808,None),('outer_wrapper',41752,None)]
    require([(x['role'],x['pid'],x.get('create_time',x.get('historical_exact_create_time')))for x in identities]==expected,
        'Only the exact five recorded identities; historical missing timestamps remain unknown')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:continue
        require(ctime is not None and abs(now-ctime)>=.001,'Original recorded identity actually dead '+role)
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path']);CPU=read_json(e['CPU_failure']['path'])
    require('Original interval beyond source unresolved'in failure['error']
        and 'Original interval beyond source unresolved'in Path(e['CPU_stderr']['path']).read_text(encoding='utf-8')
        and CPU['completed_updates']==236 and failure['actual_optimizer_updates']==140236 and failure['last_checkpoint']==140200
        and monitor['joined']is True and monitor['clean']is True and monitor['error']is None,
        'Actual ordered CPU source coverage failure, not resource denial or normal0')
    remaining_pin=value['remaining_current_plan'];require(record(remaining_pin['path'])==remaining_pin,'Exact ordered remaining-plan metadata')
    remaining=read_json(remaining_pin['path']);plan_pin=value['current_plan'];future_pin=value['future_private_plan']
    require(record(plan_pin['path'])==plan_pin and record(future_pin['path'])==future_pin,'Exact current and future private plan bytes')
    plan=read_json(plan_pin['path']);future=read_json(future_pin['path'])
    require(plan['initial_committed_updates']==140220 and len(plan['updates'])==20
        and future['initial_committed_updates']==140240 and len(future['updates'])==20
        and remaining=={'source_plan':plan_pin,'initial_updates':140220,'first_unconsumed_update_index':16,
            'candidate_global_updates':140237,'not_assumed_native_plus_one':True,'CPU_completed_updates':236,
            'CPU_initial_frontier':140000,'future_plan':future_pin,'updates':plan['updates'][16:]},
        'Exact actual nine-key remaining schema and ordered CPU localization, never native+1 inference')
    saved=value['saved_prefix_verified_once'];whole=value['whole_journal_verified_once']
    require(saved=={'path':str(folder/'journals/commits_000012.jsonl'),'offset':284977815,
        'sha256':'6232f3f54a096e9766fd4440b0fb946af1a2e871728bf64080968872bba99e44'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved
        and whole=={'path':saved['path'],'bytes':285048587,'sha256':'96f06d001d91e4998de702da4716a7efe126d1e556d05ae01d8ea3471dda3403'}
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved140200 prefix and preserved whole failed lineage')
    tail=value['preserved_unsaved_positive_tail']
    require(record(tail['path'])==tail and whole['bytes']-saved['offset']==tail['bytes']==70772,
        'Exact retained36-event suffix; never truncate, discard or replay historical tail')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==36 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(140201,140237),events)),'All36 ordered positive attempts remain historical')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed']and persistence['optimizer_updates']==140200 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['named_tensors_verified']==854
        and persistence['journal_prefix']==saved and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],
        'Actual fully verified854/207/215 saved140200, not a new GPU restore claim')
    return {**value,'saved_optimizer_updates':140200,'saved_journal_prefix':saved,
        'whole_journal_including_preserved_tail':whole,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(140201,140237))}


'''
driver_path=R/'run_ASR18_BPE9223_resident_RAM_training_v19.py';wrapper_path=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v19.py'
old=(R/'run_ASR18_BPE9223_resident_RAM_training_v18.py').read_text(encoding='utf-8')
start=old.index('def checked_original_interval_resume(');end=old.index('def checked_DEV_admission_resume(',start)
new=old[:start]+new_checker+old[end:]
new=replace(new,'from fastmss_resident_feature_stream_v7 import','from fastmss_resident_feature_stream_v8 import',2)
new=replace(new,"a.invocation==12,'Actual original-interval TOOL1","a.invocation==15,'Actual original-interval TOOL1")
new=replace(new,'Exact actual88400 saved prefix, preserved26-tail never replayed','Exact actual140200 saved prefix, preserved36-tail never replayed')
new=replace(new,'ASR18_readonly_saved134400_prefix_after_original_interval_failure_v1','ASR18_readonly_saved140200_prefix_after_original_interval_failure_v1')
ast.parse(new);assert not driver_path.exists();driver_path.write_text(new,encoding='utf-8')
ow=(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v18.py').read_text(encoding='utf-8')
nw=replace(ow,'run_ASR18_BPE9223_resident_RAM_training_v18','run_ASR18_BPE9223_resident_RAM_training_v19',6)
nw=replace(nw,'fastmss_resident_feature_stream_v7.py','fastmss_resident_feature_stream_v8.py')
nw=replace(nw,"a.start_invocation==review['initial_owner_invocation']==12","a.start_invocation==review['initial_owner_invocation']==15")
nw=replace(nw,'actual abnormal134400 predecessor','actual abnormal140200 predecessor')
ast.parse(nw);assert not wrapper_path.exists();wrapper_path.write_text(nw,encoding='utf-8')
delta(F/'driver18_to19_typed_saved140200_and_CPU8.diff.txt',old,new)
delta(F/'wrapper18_to19_import_and_initial15.diff.txt',ow,nw)
def leaf(a,b,path=''):
 if type(a)is dict and type(b)is dict:
  out=[]
  for k in sorted(a.keys()|b.keys()):
   q=path+'/'+k
   if k not in a:out.append({'path':q,'source_absent':True,'runtime':b[k]})
   elif k not in b:out.append({'path':q,'source':a[k],'runtime_absent':True})
   else:out+=leaf(a[k],b[k],q)
  return out
 if a!=b:return [{'path':path,'source':a,'runtime':b}]
 return []
source_path=N/'resident_RAM_periodic_quiescence_plan_representation_source_v1/configuration_candidate.json'
runtime_path=B/'invocations/owner_000014/configuration.json';template_path=F/'configuration_candidate_template.json'
base=read(source_path);runtime=read(runtime_path);template=read(template_path)
source_diff=leaf(base,template);runtime_diff=leaf(base,runtime)
top=sorted(k for k in base.keys()|template.keys()if base.get(k)!=template.get(k))
assert top==['original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy']
undo=copy.deepcopy(template)
for k in top:undo[k]=base[k]
assert undo==base and template['original_clip_geometry_overlay']['prior_overlay']==base['original_clip_geometry_overlay']
resource=copy.deepcopy(template['resident_RAM_resource_scheduling_policy']);resource['CPU_protocol_source']=base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'];assert resource==base['resident_RAM_resource_scheduling_policy']
runtime_tops=sorted(k for k in base.keys()|runtime.keys()if base.get(k)!=runtime.get(k))
write(F/'source_template_and_effective_runtime_binding_delta_v1.json',{'kind':'ASR18_exact_source_template_vs_materialized_runtime_configuration_difference_v1',
 'source_base':pin(source_path),'old_effective_runtime':pin(runtime_path),'inactive_candidate_template':pin(template_path),
 'candidate_source_base_changed_top_fields':top,'candidate_source_base_leaf_differences':source_diff,
 'old_effective_runtime_changed_top_fields':runtime_tops,'old_runtime_vs_source_leaf_differences':runtime_diff,
 'runtime_materialization_original_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v18.py'),
 'runtime_materialization_lines':'c=deepcopy(base); c.update(worker_driver_source, parent_driver_source, resident_RAM_stream_configuration, resident_branch_namespace, checkpoint_interval_updates=200, source_checkpoint, source_configuration); periodic entry also binds unchanged4000 runtime policy',
 'original_runtime_checkpoint_interval_updates':runtime['checkpoint_interval_updates'],'new_driver_same_checkpoint_interval_200':"checkpoint_interval_updates=200,source_checkpoint=pointer_pin,source_configuration=record(a.configuration)"in new,
 'source_scientific_tree_exact_after_three_operational_fields':True,'prior5_policy_exact':True,'all_resource_values_exact_except_CPU_source_pin':True,
 'actual_effective_configuration_will_be_materialized_by_unchanged_driver_code_at_genuine_launch':True,'production_not_admitted':True})
oldtree=ast.parse(old);newtree=ast.parse(new)
unchanged={}
for node in oldtree.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name not in ('checked_original_interval_resume','main'):
  other=next(x for x in newtree.body if type(x)is type(node)and getattr(x,'name',None)==node.name)
  unchanged[node.name]=ast.dump(node,include_attributes=False)==ast.dump(other,include_attributes=False)
assert all(unchanged.values())
write(F/'owner19_CPU8_inactive_source_handoff_v1.json',{'kind':'ASR18_INACTIVE_sixth_terminal_interval_source19_typed_saved140200_handoff_v1',
 'sources':{'driver':pin(driver_path),'wrapper':pin(wrapper_path),'CPU':pin(S/'fastmss_resident_feature_stream_v8.py')},
 'prior_sources':{'driver':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v18.py'),'wrapper':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v18.py'),'CPU':pin(S/'fastmss_resident_feature_stream_v7.py')},
 'driver_delta':pin(F/'driver18_to19_typed_saved140200_and_CPU8.diff.txt'),'wrapper_delta':pin(F/'wrapper18_to19_import_and_initial15.diff.txt'),
 'all_nonentry_ASTs_exact':unchanged,'declared_driver_changes':['replace only typed original interval validator','two CPU7 imports to8','original interval invocation12 to15','saved prefix diagnostic text','saved prefix provenance kind134400 to140200'],
 'declared_wrapper_changes':['six current driver18 imports/path to19','current CPU7 pin to8','typed original interval initial12 to15','typed message134400 to140200'],
 'configuration_template':pin(template_path),'correct_source_base':pin(source_path),'runtime_source_distinction':pin(F/'source_template_and_effective_runtime_binding_delta_v1.json'),
 'actual_CPU0_closure':pin(F/'ingress_frontend_actual_01_closure.json'),'actual_CPU_report':pin(F/'ingress_frontend_actual_01/report.json'),
 'overlay_source':pin(S/'fastmss_original_clip_geometry_overlay_v6.py'),'certificate':pin(F/'sixth_original_terminal_interval_geometry_certificate_v1.json'),
 'actual_failure':pin(F/'actual_failure_closure_v2.json'),'checkpoint':read(F/'actual_failure_closure_v2.json')['inputs']['pointer'],
 'future_invocation':15,'future_output':str(N/'resident_RAM_training_actual_16'),'future_journal':str(B/'journals/commits_000015.jsonl'),
 'journal_filename_follows_original_invocation_not_previous_counter':True,'final_overlay_ROOT_review_must_bind_completed_actual_CPU_acceptance':True,
 'saved_prefix_copy_only_after_full_exact_restore':True,'old36_tail_historical_and_genuinely_redone':True,'full401072_adminNone':True,'production_admitted':False})
print(json.dumps({'driver':pin(driver_path),'wrapper':pin(wrapper_path),'handoff':pin(F/'owner19_CPU8_inactive_source_handoff_v1.json'),'runtime_delta':pin(F/'source_template_and_effective_runtime_binding_delta_v1.json')}),flush=True)
