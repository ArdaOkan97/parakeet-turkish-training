"""Metadata-only immutable successor assembly; never imports or runs training."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
F=Path(__file__).parent/'inactive_selector_fix_v1'
R=P/'work/research'
OLD=N/'resident_RAM_native_pressure_timeout156769_failure_actual_v1/inactive_additional_cushion225_v1/production_configuration_v1/configuration.json'
def pin(p):
 p=Path(p);d=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(d).hexdigest(),'bytes':len(d)}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);assert not p.exists();p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');return pin(p)
assert not F.exists();F.mkdir()
driver_old=R/'run_ASR18_BPE9223_resident_RAM_training_v26.py'
driver_new=R/'run_ASR18_BPE9223_resident_RAM_training_v27.py'
wrapper_old=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v26.py'
wrapper_new=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v27.py'
assert not driver_new.exists() and not wrapper_new.exists()
edits=[]
def edit(s,old,new,label,count=1):
 assert s.count(old)==count,(label,s.count(old),count)
 edits.append({'label':label,'old':old,'new':new,'count':count});return s.replace(old,new)
validator='''def checked_postrestore_selector_resume(pin,*,base,folder,pointer_pin):
    """Exact source26 metadata failure AFTER restore/prefix copy, BEFORE CPU/native."""
    import psutil
    require(pin==base['prior_resident_RAM_source26_postrestore_selector_failure'] and record(pin['path'])==pin
        and pin['sha256']=='0e36fb3035ff05bcf7e917a07d511b5a93145120c15756d175acb84f73d17942',
        'Exact actual source26 TOOL1 metadata closure, never normal0 or a225 admission failure')
    value=read_json(pin['path']);e=value['evidence']
    require(value['kind']=='ASR18_actual_source26_postrestore_predecessor_selector_startup_failure_v1'
        and value['owned_session']==24476 and value['failure_class']=='startup_metadata_wiring_not_resource_admission'
        and value['actual_optimizer_updates']==value['last_checkpoint']==156600
        and value['new_native_updates']==value['new_draws']==0 and value['CPU_or_CPU_launcher_started'] is False
        and value['native_update_or_pressure_gate_reached'] is False and value['training_stopped'] is True,
        'Genuine zero-update startup failure, not a resource or training retry')
    for item in e.values():require(record(item['path'])==item,'Exact compact source26 failure evidence')
    tool_pin=value['actual_TOOL'];require(record(tool_pin['path'])==tool_pin,'Exact consumed owned session outcome')
    tool=read_json(tool_pin['path'])
    require(tool['owned_session']==24476 and tool['actual_TOOL_exit_code']==1 and tool['actual_chunk_id']=='35b265'
        and tool['no_normal0_claim'] is True and tool['separate_native_OS_exit_code']=='UNKNOWN_NOT_PERSISTED',
        'Honest outer TOOL1 and unknown separate numeric native exit')
    old_source=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_native_pressure_timeout156769_failure_actual_v1/inactive_additional_cushion225_v1/production_configuration_v1/configuration.json'
    restored=copy.deepcopy(base);del restored['prior_resident_RAM_source26_postrestore_selector_failure']
    require(restored==read_json(old_source),'ONLY new startup-failure provenance field; approved225 and all old values exact')
    require(e['pointer']==pointer_pin and e['driver']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v26.py')
        and e['wrapper']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v26.py')
        and e['failure']==record(folder/'invocations/owner_000019/actual_failure.json')
        and e['restore']==record(folder/'invocations/owner_000019/actual_parent_restore.json'),
        'Exact historical26 invocation19 and saved156600, not successor aliases')
    failure=read_json(e['failure']['path']);restore=read_json(e['restore']['path']);monitor=read_json(e['monitor_completion']['path'])
    require(failure==value['failure'] and failure['actual_optimizer_updates']==failure['last_checkpoint']==156600
        and failure['error']=="TypeError('expected str, bytes or os.PathLike object, not NoneType')"
        and 'line 820, in main' in failure['traceback'] and "'predecessor':record(" in failure['traceback']
        and restore==value['genuine_all_state_restore'] and restore['passed'] is True
        and restore['all207_threeAdam_all4RNG_match_actual_pre_save'] is True and restore['next_draw_packet_match'] is True
        and restore['restore_occurred_only_at_startup_before_native'] is True
        and monitor==value['original_monitor'] and monitor['samples']==409 and monitor['joined'] is True
        and monitor['clean'] is True and monitor['error'] is None,
        'Exact selector TypeError after genuine restore, clean original Monitor')
    expected=[('native',34604,1792725704.227828),('native_interpreter_launcher',40624,1792725704.2208068),
        ('wrapper',37676,1792725703.8532965)]
    require([(x['role'],x['pid'],x['create_time'])for x in value['known_runtime_exact_deaths']]==expected,
        'Only the three genuinely started runtime identities, CPU never created')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time();dead=abs(now-ctime)>=.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Exact source26 identity dead '+role)
    orphan=value['orphan19_saved_prefix_STAT'];old=value['original18_STAT_unchanged']
    require(orphan=={'path':str(folder/'journals/commits_000019.jsonl'),'bytes':317194933,
        'mtime_ns':1792725912000000000,'content_read':False}
        and old=={'path':str(folder/'journals/commits_000018.jsonl'),'bytes':317525825,
        'mtime_ns':1792718256000000000,'content_read':False},'Exact inactive orphan19 and historical whole18 STAT bindings')
    for item in (orphan,old):
        current=Path(item['path']).stat()
        require(current.st_size==item['bytes'] and current.st_mtime_ns==item['mtime_ns'],'Preserve orphan19 and original18 without promotion or reread')
    require(not(folder/'invocations/owner_000019/actual_journal_branch.json').exists()
        and not(folder/'invocations/owner_000019/CPU').exists()
        and not(folder/'invocations/owner_000019/normal_completion.json').exists()
        and Path(read_json(folder/'active_journal.json')['journal_prefix']['path'])==folder/'journals/commits_000018.jsonl',
        'Absent branch receipt/CPU/normal0 and unchanged authoritative active18')
    require(read_json(pointer_pin['path'])['committed_updates']==156600
        and read_json(folder/'latest_checkpoint.json')['pointer']==pointer_pin,
        'Still genuine fully saved156600; old169 attempts must be redone')
    return value


'''
d0=driver_old.read_bytes();d=d0.decode().replace('\r\n','\n')
d=edit(d,'def checked_packet_credit_resume(',validator+'def checked_packet_credit_resume(','new exact startup failure validator')
d=edit(d,"    packet_credit_failure=None if a.packet_credit_resume is None else checked_packet_credit_resume(","    require(a.postrestore_selector_resume is None or (a.cushion225_resume is not None and a.prior_process is None),\n        'Startup metadata repair retains original approved225 abnormal lane; never normal0')\n    if a.postrestore_selector_resume is not None:\n        checked_postrestore_selector_resume(record(a.postrestore_selector_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    packet_credit_failure=None if a.packet_credit_resume is None else checked_packet_credit_resume(",'preboot exact startup validation')
d=edit(d,"(19 if a.cushion225_resume is not None else(18", "((20 if a.postrestore_selector_resume is not None else 19) if a.cushion225_resume is not None else(18",'fresh invocation20 only for exact repaired startup')
d=edit(d,"'failure':record(a.cushion225_resume if a.cushion225_resume is not None else a.packet_credit_resume", "'failure':record(a.postrestore_selector_resume if a.postrestore_selector_resume is not None else a.cushion225_resume if a.cushion225_resume is not None else a.packet_credit_resume",'restore factory metadata selects exact startup predecessor')
d=edit(d,"'predecessor':record(a.original_interval_resume if interval_failure is not None else((a.packet_credit_resume", "'predecessor':record(a.postrestore_selector_resume if a.postrestore_selector_resume is not None else a.cushion225_resume if a.cushion225_resume is not None else a.original_interval_resume if interval_failure is not None else((a.packet_credit_resume",'fix exact failed postrestore selector with new-lane precedence')
d=edit(d,"p.add_argument('--cushion225-resume');", "p.add_argument('--postrestore-selector-resume');p.add_argument('--cushion225-resume');",'parser exact startup predecessor')
driver_edits=copy.deepcopy(edits);edits=[]
w0=wrapper_old.read_bytes();w=w0.decode().replace('\r\n','\n')
w=edit(w,'run_ASR18_BPE9223_resident_RAM_training_v26','run_ASR18_BPE9223_resident_RAM_training_v27','wrapper successor imports only',8)
w=edit(w,"    if a.prior_process is not None:\n", "    require(a.postrestore_selector_resume is None or (a.cushion225_resume is not None and a.prior_process is None),\n        'Exact startup repair cannot enter a normal or unrelated predecessor lane')\n    if a.prior_process is not None:\n",'wrapper startup argument exclusivity')
w=edit(w,"and a.start_invocation==review['initial_owner_invocation']==19,", "and a.start_invocation==review['initial_owner_invocation']==(20 if a.postrestore_selector_resume is not None else 19)\n            and (a.postrestore_selector_resume is None or review['postrestore_selector_resume']==record(a.postrestore_selector_resume)),",'ROOT exact startup review and invocation20')
w=edit(w,"    if a.packet_credit_resume is not None:\n        from ","    if a.postrestore_selector_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v27 import checked_postrestore_selector_resume\n        checked_postrestore_selector_resume(record(a.postrestore_selector_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.packet_credit_resume is not None:\n        from ",'wrapper validates before control/lease/GPU')
w=edit(w,"            if a.cushion225_resume is not None:command+=['--cushion225-resume',a.cushion225_resume]", "            if a.postrestore_selector_resume is not None:command+=['--postrestore-selector-resume',a.postrestore_selector_resume]\n            if a.cushion225_resume is not None:command+=['--cushion225-resume',a.cushion225_resume]",'pass exact startup predecessor to initial native only')
w=edit(w,"p.add_argument('--cushion225-resume');", "p.add_argument('--postrestore-selector-resume');p.add_argument('--cushion225-resume');",'wrapper parser exact startup predecessor')
wrapper_edits=copy.deepcopy(edits)
def freeze(old,new,text,declared):
 ast.parse(text)
 undone=text
 for x in reversed(declared):
  assert undone.count(x['new'])==x['count'],x['label'];undone=undone.replace(x['new'],x['old'])
 original=old.read_bytes();restored=undone.replace('\n','\r\n').encode() if b'\r\n'in original else undone.encode()
 assert restored==original
 data=text.replace('\n','\r\n').encode() if b'\r\n'in original else text.encode();new.write_bytes(data)
 (F/(new.stem+'.diff')).write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),data.decode().splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8')
 return {'old':pin(old),'new':pin(new),'declared_edits':declared,'entire_raw_byte_inverse_exact':True}
driver_proof=freeze(driver_old,driver_new,d,driver_edits);wrapper_proof=freeze(wrapper_old,wrapper_new,w,wrapper_edits)
oldast=ast.parse(d0.decode());newast=ast.parse(d)
def dump(x):return ast.dump(x,include_attributes=False)
unchanged=[]
for node in oldast.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name!='main':
  nn=next(x for x in newast.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==node.name)
  assert dump(node)==dump(nn),node.name;unchanged.append(node.name)
oldmain=next(x for x in oldast.body if isinstance(x,ast.FunctionDef)and x.name=='main');newmain=next(x for x in newast.body if isinstance(x,ast.FunctionDef)and x.name=='main')
oldloops=[dump(x)for x in ast.walk(oldmain)if isinstance(x,ast.While)];newloops=[dump(x)for x in ast.walk(newmain)if isinstance(x,ast.While)];assert oldloops==newloops
oldnested={x.name:dump(x)for x in ast.walk(oldmain)if isinstance(x,ast.FunctionDef)and x is not oldmain};newnested={x.name:dump(x)for x in ast.walk(newmain)if isinstance(x,ast.FunctionDef)and x is not newmain}
assert all(newnested[k]==v for k,v in oldnested.items()if k!='journal_factory')
base=read(OLD);config=copy.deepcopy(base);config['prior_resident_RAM_source26_postrestore_selector_failure']=pin(Path(__file__).parent/'actual_failure_closure.json')
configuration=write(F/'production_configuration_v1/configuration.json',config)
inverse=copy.deepcopy(config);del inverse['prior_resident_RAM_source26_postrestore_selector_failure'];assert inverse==base
runtime={'worker_driver_source':pin(driver_new),'parent_driver_source':pin(driver_new),'resident_RAM_stream_configuration':configuration,'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':pin(B/'checkpoints/update_000156600/pointer.json'),'source_configuration':configuration}
report=write(F/'source27_full_inverse_and_configuration_report.json',{'driver':driver_proof,'wrapper':wrapper_proof,'all_old_top_nonentry_AST_exact':unchanged,'all_main_while_AST_exact':True,'all_unaffected_nested_main_AST_exact':sorted(k for k in oldnested if k!='journal_factory'),'journal_factory_only_declared_predecessor_expression_delta':True,'proper_source_base':pin(OLD),'only_added_config_field':'prior_resident_RAM_source26_postrestore_selector_failure','old_config_all_values_exact':True,'original225_policy_and_closed_CPU_review_exact':config['native_update_commit_pressure_policy'],'future_7_runtime_binding_kinds':runtime,'no_training_control_restore_copy_or_GPU_execution':True})
args=['--configuration',configuration['path'],'--pointer',str(B/'checkpoints/update_000156600/pointer.json'),'--latest-failure',str(N/'pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json'),'--branch',str(B),'--review','ROOT_FINAL_EXECUTION_REVIEW_PENDING','--output',str(N/'resident_RAM_training_actual_21'),'--queue-bytes','268435456','--start-invocation','20','--cushion225-resume',base['prior_resident_RAM_native_pressure156769_failure']['path'],'--postrestore-selector-resume',config['prior_resident_RAM_source26_postrestore_selector_failure']['path']]
handoff=write(F/'inactive_source27_handoff_v1.json',{'kind':'ASR18_inactive_source27_exact_postrestore_selector_fix_v1','production_admitted':False,'finite_metadata_control_admitted':False,'driver_source':pin(driver_new),'wrapper_source':pin(wrapper_new),'source_configuration':configuration,'raw_inverse_report':report,'driver_diff':pin(F/(driver_new.stem+'.diff')),'wrapper_diff':pin(F/(wrapper_new.stem+'.diff')),'actual_startup_failure':config['prior_resident_RAM_source26_postrestore_selector_failure'],'closed_focused225_CPU_review':config['native_update_commit_pressure_policy']['actual_CPU_control_review'],'initial_owner_invocation':20,'output':str(N/'resident_RAM_training_actual_21'),'new_journal':str(B/'journals/commits_000020.jsonl'),'orphan19_preserved_never_reused_promoted_deleted':True,'original18_saved_prefix_only_after_genuine_restore':True,'historical169_must_be_genuinely_redone_through156770':True,'full_goal':401072,'administrative_update_stop':None,'exact_future_wrapper_arguments_DRAFT':args,'expected_future_7_runtime_bindings':runtime,'next_required_proof':'ONE separately ROOT-scoped read-only typed metadata + exact postrestore predecessor expression/metadata regression; no GPU/audio/restore/prefixcopy'})
print(json.dumps({'driver':pin(driver_new),'wrapper':pin(wrapper_new),'configuration':configuration,'inverse':report,'handoff':handoff}))
