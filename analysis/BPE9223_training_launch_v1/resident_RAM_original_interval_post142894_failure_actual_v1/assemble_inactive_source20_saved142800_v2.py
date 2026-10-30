import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');F=N/'resident_RAM_original_interval_post142894_failure_actual_v1';OLD=N/'resident_RAM_original_interval_post140236_failure_actual_v1'
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);assert not p.exists(),p;p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def textwrite(p,t):
 p=Path(p);assert not p.exists(),p;p.write_bytes(t.encode('utf-8'))
oldpath=R/'run_ASR18_BPE9223_resident_RAM_training_v19.py';newpath=R/'run_ASR18_BPE9223_resident_RAM_training_v20.py'
old=oldpath.read_text(encoding='utf-8');begin=old.index('def checked_original_interval_resume(');end=old.index('\ndef checked_DEV_admission_resume(',begin)
oldfunction=old[begin:end]
newfunction='''def checked_original_interval_resume(pin,*,base,folder,pointer_pin):
    """Exact source19 EOF142894, saved142800, all94 positive attempts retained."""
    import psutil
    from datetime import datetime
    require(pin==base['prior_resident_RAM_original_interval_failure']and record(pin['path'])==pin
        and pin['sha256']=='d10f4b81cb390115bd502041ffee408acfec1c8ec8b28d748a2ac1b6fc6a3418',
        'Exact ROOT-bound actual source19 EOF failure; no normal0 relabel')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_original_interval_post142894_failure_closed_v1'
        and value['owned_session']==1903 and value['actual_outer_TOOL_exit_code']==1 and value['actual_TOOL_chunk']=='1f7856'
        and value['actual_optimizer_updates']==142894 and value['last_fully_saved_updates']==142800
        and value['unsaved_positive_tail_updates']==94 and value['CPU_failure_completed_prepared_updates']==2694
        and value['native_numeric_OS_exit_code']is None and value['CPU_numeric_OS_exit_code']is None
        and value['native_Popen_wait_known_nonzero_from_wrapper_assertion']is True
        and value['all94_ordered_positive_events_verified']is True,
        'Truthful actual TOOL1/unknown OS codes; saved142800 and94 historical positives')
    require(e['pointer']==pointer_pin and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
        and e['source19']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v19.py')
        and e['wrapper19']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v19.py')
        and e['CPU8_source']==record(S/'fastmss_resident_feature_stream_v8.py'),
        'Exact failed source19/wrapper19/CPU8 and original checkpoint branch')
    for item in e.values():require(record(item['path'])==item,'Exact compact frozen failed-owner/source/checkpoint evidence')
    death_pin=value['owned_process_deaths'];require(record(death_pin['path'])==death_pin,'Exact recorded seven-identity death observation')
    observed=read_json(death_pin['path'])
    expected=[{'role':'native','pid':37316,'create_time':1792674053.7525675,'ppid':41416},
        {'role':'native_interpreter_launcher','pid':41416,'create_time':1792674053.7455387,'ppid':41820},
        {'role':'CPU_producer','pid':38236,'create_time':1792674086.2008965,'ppid':23980},
        {'role':'CPU_interpreter_launcher','pid':23980,'historical_exact_create_time':None,'ppid':None},
        {'role':'wrapper','pid':41820,'creation_UTC':'2026-10-22T13:00:53.441969Z','ppid':41796},
        {'role':'wrapper_interpreter_launcher','pid':41796,'creation_UTC':'2026-10-22T13:00:53.432518Z','ppid':35016},
        {'role':'outer_tool_shell','pid':35016,'historical_exact_create_time':None,'ppid':None}]
    require([x['original_identity']for x in observed]==expected and all(x['actual_PID_absent']is True
        and x['current_if_any']is None and x['no_historical_timestamp_or_numeric_exit_invented']is True for x in observed),
        'Only exact seven recorded identities; three numeric/two UTC/two genuinely unknown historical timestamps')
    for who in expected:
        try:now=psutil.Process(who['pid']).create_time()
        except psutil.NoSuchProcess:continue
        ctime=who.get('create_time')
        if 'creation_UTC'in who:ctime=datetime.fromisoformat(who['creation_UTC'].replace('Z','+00:00')).timestamp()
        require(ctime is not None and abs(now-ctime)>=.001,'Original recorded identity actually dead '+who['role'])
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path']);CPU=read_json(e['CPU_failure']['path'])
    require('Original interval beyond source unresolved'in failure['error']
        and 'Original interval beyond source unresolved'in Path(e['CPU_stderr']['path']).read_text(encoding='utf-8')
        and CPU['completed_updates']==2694 and failure['actual_optimizer_updates']==142894 and failure['last_checkpoint']==142800
        and monitor==value['monitor_clean_and_joined']and monitor['joined']is True and monitor['clean']is True and monitor['error']is None,
        'Actual ordered CPU source coverage failure; clean monitor, not a claimed absence of CPU allocation waits')
    require(value['captured_CPU_resource_telemetry_distinct_from_coverage_error']['prospective_commit_reserve_bytes']==2500000000,
        'Preserve source19 stronger prospective reservation without claiming freed memory or zero waits')
    remaining_pin=value['remaining_current_plan'];require(record(remaining_pin['path'])==remaining_pin,'Exact ordered remaining-plan metadata')
    remaining=read_json(remaining_pin['path']);plan_pin=value['current_plan'];future_pin=value['future_private_plan']
    require(record(plan_pin['path'])==plan_pin and record(future_pin['path'])==future_pin,'Exact current and future private plan bytes')
    plan=read_json(plan_pin['path']);future=read_json(future_pin['path'])
    require(plan['initial_committed_updates']==142880 and len(plan['updates'])==20
        and future['initial_committed_updates']==142900 and len(future['updates'])==20
        and remaining=={'source_plan':plan_pin,'initial_updates':142880,'first_unconsumed_update_index':14,
            'candidate_global_updates':142895,'not_assumed_native_plus_one':True,'CPU_completed_updates':2694,
            'CPU_initial_frontier':140200,'future_plan':future_pin,'updates':plan['updates'][14:]},
        'Exact nine-key remaining schema and actual CPU localization, never native+1 inference')
    saved=value['saved_prefix_verified_once'];whole=value['whole_journal_verified_once']
    require(saved=={'path':str(folder/'journals/commits_000015.jsonl'),'offset':290088487,
        'sha256':'e45e1adb098b5c1fdf4388116c4db6a61feb348320a1f396f0bdb73f3cbb4598'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved
        and whole=={'path':saved['path'],'bytes':290273015,'sha256':'1c69569994392387686b823c2bbec97ed069aa07f2ffc2677c5c410b5cb90f0d'}
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved142800 prefix and preserved whole failed lineage')
    tail=value['preserved_unsaved_positive_tail']
    require(record(tail['path'])==tail and whole['bytes']-saved['offset']==tail['bytes']==184528,
        'Exact retained94-event suffix; never truncate, discard or replay historical tail')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==94 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(142801,142895),events)),'All94 ordered positive attempts remain historical')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed']and persistence['optimizer_updates']==142800 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['named_tensors_verified']==854
        and persistence['journal_prefix']==saved and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],
        'Actual fully verified854/207/215 saved142800, not a new GPU restore claim')
    return {**value,'saved_optimizer_updates':142800,'saved_journal_prefix':saved,
        'whole_journal_including_preserved_tail':whole,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(142801,142895))}


'''
changes=[('from fastmss_resident_feature_stream_v8 import','from fastmss_resident_feature_stream_v9 import',2),
 ("elif interval_failure is not None:require(a.prior_process is None and a.invocation==15,","elif interval_failure is not None:require(a.prior_process is None and a.invocation==16,",1),
 ('Exact actual140200 saved prefix, preserved36-tail never replayed','Exact actual142800 saved prefix, preserved94-tail never replayed',1),
 ('ASR18_readonly_saved140200_prefix_after_original_interval_failure_v1','ASR18_readonly_saved142800_prefix_after_original_interval_failure_v1',1)]
new=old[:begin]+newfunction+old[end:]
for a,b,count in changes:assert new.count(a)==count,(a,new.count(a));new=new.replace(a,b)
restored=new
for a,b,count in reversed(changes):assert restored.count(b)==count;restored=restored.replace(b,a)
assert restored.count(newfunction)==1;restored=restored.replace(newfunction,oldfunction);assert restored==old
ast.parse(new);textwrite(newpath,new.replace('\n','\r\n'))
wrapperoldpath=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v19.py';wrappernewpath=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v20.py'
wo=wrapperoldpath.read_text(encoding='utf-8');wn=wo
wc=[('run_ASR18_BPE9223_resident_RAM_training_v19','run_ASR18_BPE9223_resident_RAM_training_v20',5),('fastmss_resident_feature_stream_v8','fastmss_resident_feature_stream_v9',1),
 ("and a.start_invocation==review['initial_owner_invocation']==15,","and a.start_invocation==review['initial_owner_invocation']==16,",1),('actual abnormal140200 predecessor','actual abnormal142800 predecessor',1)]
for a,b,c in wc:assert wn.count(a)==c,(a,wn.count(a));wn=wn.replace(a,b)
wr=wn
for a,b,c in reversed(wc):assert wr.count(b)==c;wr=wr.replace(b,a)
assert wr==wo;ast.parse(wn);textwrite(wrappernewpath,wn.replace('\n','\r\n'))
diff1=F/'driver19_to20_exact_saved142800_CPU9.diff.txt';diff2=F/'wrapper19_to20_exact_import_initial16.diff.txt'
textwrite(diff1,''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=oldpath.name,tofile=newpath.name)))
textwrite(diff2,''.join(difflib.unified_diff(wo.splitlines(True),wn.splitlines(True),fromfile=wrapperoldpath.name,tofile=wrappernewpath.name)))
oldast=ast.parse(old);newast=ast.parse(new)
nonentry={}
for node in oldast.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name not in ('main','checked_original_interval_resume'):
  other=next(x for x in newast.body if type(x)==type(node)and x.name==node.name);assert ast.dump(node,include_attributes=False)==ast.dump(other,include_attributes=False);nonentry[node.name]=True
out=F/'production_configuration_v1';assert not out.exists();out.mkdir()
base_pin=pin(OLD/'production_configuration_v1/configuration.json');base=read(base_pin['path']);config=read(F/'configuration_candidate_template.json')
review_pin=pin(N/'ROOT_seventh_original_full_clip_geometry_actual_CPU_review_v1.json');review=read(review_pin['path'])
assert review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_source_bound_seventh_original_full_clip_geometry_overlay_v1'
assert review['overlay_source']==config['original_clip_geometry_overlay']['source']and review['certificate']==config['original_clip_geometry_overlay']['certificate']
config['original_clip_geometry_overlay']['ROOT_review']=review_pin
assert config['original_clip_geometry_overlay']['prior_overlay']==base['original_clip_geometry_overlay']
changed=[k for k in base if base[k]!=config[k]];assert set(base)==set(config)and set(changed)=={'original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'}
rp=copy.deepcopy(config['resident_RAM_resource_scheduling_policy']);rp['CPU_protocol_source']=base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'];assert rp==base['resident_RAM_resource_scheduling_policy']
configpath=out/'configuration.json';write(configpath,config)
effective=read(B/'invocations/owner_000015/configuration.json');runtime_keys=[k for k in effective if effective[k]!=base.get(k)]
expected_runtime={'worker_driver_source','parent_driver_source','resident_RAM_stream_configuration','resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration'}
assert set(runtime_keys)==expected_runtime,runtime_keys
leafreport=F/'source_template_and_effective_runtime_binding_delta_v1.json'
write(leafreport,{'kind':'ASR18_source_template_operational_and_seven_actual_runtime_bindings_v1','source_base':base_pin,'old_effective':pin(B/'invocations/owner_000015/configuration.json'),
 'new_source_configuration':pin(configpath),'exact_three_source_operational_fields':changed,'old_runtime_vs_source_exact_seven_keys':{k:{'source':base.get(k),'effective':effective[k]}for k in runtime_keys},
 'new_expected_runtime_bindings':{'worker_driver_source':pin(newpath),'parent_driver_source':pin(newpath),'resident_RAM_stream_configuration':pin(configpath),'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':read(F/'actual_failure_closure.json')['inputs']['pointer'],'source_configuration':pin(configpath)},
 'all_science_and_resource_values_unchanged':True,'prior_six_overlay_policy_exact':True})
controlold=OLD/'validate_final_saved140200_metadata_v1.py';control=controlold.read_text(encoding='utf-8')
for a,b in [('training_v19','training_v20'),('stream_v8','stream_v9'),('140200','142800'),('140201,140237','142801,142895'),('all36','all94'),('tail_updates\':36','tail_updates\':94'),('ASR18_source_bound_sixth_original_terminal_interval_geometry_overlay_v1','ASR18_source_bound_seventh_original_full_clip_geometry_overlay_v1')]:
 assert a in control,a;control=control.replace(a,b)
ast.parse(control);controlpath=F/'validate_final_saved142800_metadata_v1.py';textwrite(controlpath,control)
controlargv=[str(Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')),'-X','utf8','-B',str(controlpath),'--configuration',str(configpath),'--output',str(out/'typed_metadata_actual_01')]
priorhandoff=read(OLD/'production_configuration_v1/final_source_configuration_review_handoff_v1.json')
requirements=copy.deepcopy(priorhandoff['ROOT_execution_review_required_fields'])
requirements.update(driver_source=pin(newpath),wrapper_source=pin(wrappernewpath),CPU_source=pin(S/'fastmss_resident_feature_stream_v9.py'),source_configuration=pin(configpath),
 initial_checkpoint=read(F/'actual_failure_closure.json')['inputs']['pointer'],original_interval_resume=pin(F/'actual_failure_closure.json'),original_clip_geometry_overlay=config['original_clip_geometry_overlay'],resource_scheduling_policy=config['resident_RAM_resource_scheduling_policy'],initial_owner_invocation=16)
args=priorhandoff['exact_future_wrapper_arguments']
args=args.copy();args[args.index('--configuration')+1]=str(configpath);args[args.index('--pointer')+1]=requirements['initial_checkpoint']['path'];args[args.index('--output')+1]=str(N/'resident_RAM_training_actual_17');args[args.index('--start-invocation')+1]='16';args[args.index('--original-interval-resume')+1]=str(F/'actual_failure_closure.json');assert len(args)==18
handoff={'kind':'ASR18_INACTIVE_exact_saved142800_source20_config_typed_gate_handoff_v1','production_admitted':False,'typed_control_not_executed':True,
 'sources':{'driver':pin(newpath),'wrapper':pin(wrappernewpath),'CPU':pin(S/'fastmss_resident_feature_stream_v9.py')},'source_base':base_pin,'configuration':pin(configpath),
 'driver_delta':pin(diff1),'wrapper_delta':pin(diff2),'driver_inverse_exact_after_one_typed_validator_and_four_declared_changes':True,'driver_declared_rest_changes':changes,'wrapper_declared_changes':wc,
 'all_nonentry_functions_and_classes_AST_exact':nonentry,'source_vs_runtime_leaf_report':pin(leafreport),'actual_ROOT_CPU_review':review_pin,'actual_CPU_report':pin(F/'ingress_frontend_actual_01/report.json'),'actual_CPU0_closure':pin(F/'ingress_frontend_actual_01_closure_v2.json'),
 'failure':pin(F/'actual_failure_closure.json'),'typed_metadata_control_source':pin(controlpath),'exact_metadata_control_argument_array':controlargv,'typed_control_read_scope':'Only compact frozen inputs/two plans/seven-identity death file and184528B historical tail; old whole journal stat only, no wholejournal or tensor payload read',
 'future_wrapper':pin(wrappernewpath),'exact_future_wrapper_arguments':args,'ROOT_execution_review_required_fields':requirements,
 'future_active_journal':str(B/'journals/commits_000016.jsonl'),'old_whole_and94_positive_tail_preserved':True,'full401072_adminNone':True}
handoffpath=out/'final_source_configuration_review_handoff_v1.json';write(handoffpath,handoff)
print(json.dumps({'handoff':pin(handoffpath),'driver':pin(newpath),'wrapper':pin(wrappernewpath),'configuration':pin(configpath),'typed_metadata_source':pin(controlpath),'changed_source_fields':changed},ensure_ascii=False))