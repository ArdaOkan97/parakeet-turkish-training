"""Metadata-only immutable source32 preparation. Never imports/executes successors."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
FIX=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1'
def pin(p):
 p=Path(p).resolve();v=p.read_bytes();return {'path':str(p),'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);assert not p.exists(),p;p.write_bytes((json.dumps(v,indent=2,ensure_ascii=True)+'\n').encode())
def build(old,new,edits):
 raw=old.read_bytes();assert b'\r\n'in raw and b'\n'not in raw.replace(b'\r\n',b''),'Declared original CRLF'
 records=[];value=raw.decode('utf-8')
 for before,after,count in edits:
  before=before.replace('\n','\r\n');after=after.replace('\n','\r\n');assert value.count(before)==count,(before[:150],value.count(before),count)
  value=value.replace(before,after);records.append({'old_literal':before,'new_literal':after,'expected_count':count})
 result=value.encode();ast.parse(value);inverse=value
 for e in reversed(records):
  assert inverse.count(e['new_literal'])==e['expected_count'],('inverse',e['new_literal'][:150],inverse.count(e['new_literal']))
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode()==raw;assert not new.exists();new.write_bytes(result)
 return {'old_source':pin(old),'new_source':pin(new),'newline_normalization':'NONE; exact original CRLF retained',
  'replacements_in_forward_order':records,'full_raw_inverse_exact':True},''.join(difflib.unified_diff(raw.decode().splitlines(True),value.splitlines(True),fromfile=str(old),tofile=str(new)))

driver=R/'run_ASR18_BPE9223_resident_RAM_training_v32.py';old=R/'run_ASR18_BPE9223_resident_RAM_training_v31.py'
validator=F/'asr18_eighth_original_interval_checkpoint_resume_v1.py'
newentry="""    from asr18_eighth_original_interval_checkpoint_resume_v1 import checked_eighth_original_policy,checked_eighth_original_resume
    checked_eighth_original_policy(base)
    eighth_interval_failure=None if a.eighth_original_resume is None else checked_eighth_original_resume(
        record(a.eighth_original_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if eighth_interval_failure is not None:
        require(a.invocation==24 and all(x is None for x in (a.prior_process,a.fresh_signature_resume,a.canonical_archive_resume,
            a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,
            a.abnormal_resume,a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,
            a.periodic_quiescence_resume)), 'Only exact source31 interval TOOL1 saved170400, never normal0 or historical lane')
"""
# Import the E-local typed validator without altering any current/historical module.
entry_import="    import sys\n    if str(Path('"+F.as_posix()+"'))not in sys.path:sys.path.insert(0,str(Path('"+F.as_posix()+"')))\n"
edits=[
 ('from fastmss_resident_feature_stream_v11 import checked_resource_profile','from fastmss_resident_feature_stream_v13 import checked_resource_profile',1),
 ('    native_pressure_policy,wait_for_native_commit_pressure=checked_native_commit_pressure_policy(base)\n','    native_pressure_policy,wait_for_native_commit_pressure=checked_native_commit_pressure_policy(base)\n'+entry_import+newentry,1),
 ("    interval_failure=None if a.original_interval_resume is None else checked_original_interval_resume(\n        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin)","    interval_failure=eighth_interval_failure if eighth_interval_failure is not None else(None if a.original_interval_resume is None else checked_original_interval_resume(\n        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin))",1),
 ("elif interval_failure is not None:require(a.prior_process is None and a.invocation==16,'Actual original-interval TOOL1 is not normal0/intentional stop')","elif interval_failure is not None:require(a.prior_process is None and a.invocation==(24 if a.eighth_original_resume is not None else 16),'Actual original-interval TOOL1 is not normal0/intentional stop')",1),
 ("            if interval_failure is not None:\n                require(saved==source_journal,'Exact actual142800 saved prefix, preserved94-tail never replayed')", "            if eighth_interval_failure is not None:\n                require(saved==source_journal,'Exact actual170400 saved prefix; preserved69 COPY never replayed')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved170400_prefix_after_eighth_original_interval_failure_v1',\n                    'failure':record(a.eighth_original_resume),'original_tail_modified':False,'no_tail_replay':True}\n            if interval_failure is not None:\n                require(saved==source_journal,'Exact actual142800 saved prefix, preserved94-tail never replayed')",1),
 ("check_original_closed_journal_stat(preserved_tail['whole_journal_including_preserved_tail'])if (a.fresh_signature_resume is not None", "check_original_closed_journal_stat(preserved_tail['whole_journal_including_preserved_tail'])if (a.eighth_original_resume is not None or a.fresh_signature_resume is not None",1),
 ("'predecessor':record(a.fresh_signature_resume if a.fresh_signature_resume is not None", "'predecessor':record(a.eighth_original_resume if a.eighth_original_resume is not None else a.fresh_signature_resume if a.fresh_signature_resume is not None",1),
 ('from fastmss_resident_feature_stream_v11 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v13 import ResidentFeatureProducer',1),
 ("p.add_argument('--fresh-signature-resume');","p.add_argument('--eighth-original-resume');p.add_argument('--fresh-signature-resume');",1)]
dledger,ddiff=build(old,driver,edits)

wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v32.py';wo=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v31.py'
newwrapper="""    elif a.eighth_original_resume is not None:
        require(all(x is None for x in (a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,
            a.source_transition_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['eighth_original_resume']==record(a.eighth_original_resume)
            and review['eighth_original_transition_validator']==record(Path('VALIDATOR'))
            and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and a.start_invocation==review['initial_owner_invocation']==24,
            'Only exact reviewed actual source31 interval failure, CPU13 and completed eighth ingress proof')
""".replace('VALIDATOR',validator.as_posix())
wp="""    import sys
    if str(Path('FAILPATH'))not in sys.path:sys.path.insert(0,str(Path('FAILPATH')))
    from asr18_eighth_original_interval_checkpoint_resume_v1 import checked_eighth_original_policy,checked_eighth_original_resume
    checked_eighth_original_policy(configuration)
    if a.eighth_original_resume is not None:
        checked_eighth_original_resume(record(a.eighth_original_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
""".replace('FAILPATH',F.as_posix())
wedits=[('run_ASR18_BPE9223_resident_RAM_training_v31','run_ASR18_BPE9223_resident_RAM_training_v32',10),
 ("review['CPU_source']==record(S/'fastmss_resident_feature_stream_v11.py')","review['CPU_source']==record(S/'fastmss_resident_feature_stream_v13.py')",1),
 ("require(a.fresh_signature_resume is None and a.canonical_archive_resume is None","require(a.eighth_original_resume is None and a.fresh_signature_resume is None and a.canonical_archive_resume is None",1),
 ("    elif a.fresh_signature_resume is not None:\n",newwrapper+"    elif a.fresh_signature_resume is not None:\n",1),
 ("    checked_native_commit_pressure_policy(configuration)\n","    checked_native_commit_pressure_policy(configuration)\n"+wp,1),
 ('from fastmss_resident_feature_stream_v11 import checked_packet_credit_policy','from fastmss_resident_feature_stream_v13 import checked_packet_credit_policy',1),
 ("            if a.fresh_signature_resume is not None:command+=['--fresh-signature-resume',a.fresh_signature_resume]", "            if a.eighth_original_resume is not None:command+=['--eighth-original-resume',a.eighth_original_resume]\n            elif a.fresh_signature_resume is not None:command+=['--fresh-signature-resume',a.fresh_signature_resume]",1),
 ("p.add_argument('--fresh-signature-resume');","p.add_argument('--eighth-original-resume');p.add_argument('--fresh-signature-resume');",1)]
wledger,wdiff=build(wo,wrapper,wedits)
write(F/'source31_to32_entire_RAW_inverse_v1.json',{'kind':'ASR18_source32_exact_declared_RAW_inverse_v1','driver':dledger,'wrapper':wledger})
for name,text in [('source31_to32_driver_v1.diff',ddiff),('source31_to32_wrapper_v1.diff',wdiff)]:
 p=F/name;assert not p.exists();p.write_bytes(text.encode())
oldtree=ast.parse(old.read_bytes());newtree=ast.parse(driver.read_bytes())
oldtop={n.name:ast.dump(n,include_attributes=False)for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name!='main'}
newtop={n.name:ast.dump(n,include_attributes=False)for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name!='main'}
assert oldtop==newtop
om=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=='main');nm=next(n for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
oloops=[ast.dump(n,include_attributes=False)for n in ast.walk(om)if isinstance(n,ast.While)]
nloops=[ast.dump(n,include_attributes=False)for n in ast.walk(nm)if isinstance(n,ast.While)];assert oloops==nloops and len(oloops)==3
onested={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(om)if isinstance(n,ast.FunctionDef)and n is not om}
nnested={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(nm)if isinstance(n,ast.FunctionDef)and n is not nm}
changed=[k for k in onested if onested[k]!=nnested[k]];assert changed==['journal_factory'],changed
write(F/'source32_original_numerical_and_nonentry_AST_equivalence_v1.json',{'driver_old':pin(old),'driver_new':pin(driver),
 'all_old_nonentry_AST_exact':list(oldtop),'all_three_native_while_AST_exact':True,
 'unaffected_nested_main_AST_exact':[k for k in onested if k not in changed],'only_changed_nested_main':['journal_factory'],
 'historical_validators_CPU11_assertions_unchanged':True,'new_CPU13_imports_only_current_resource_profile_and_producer':True})

config=read(F/'configuration_source32_eighth_alias9_scoped_source_candidate_v4.json')
config['original_clip_geometry_overlay']['ROOT_review']=pin(F/'ROOT_eighth_alias9_CPU13_actual02_original_ingress_frontend_CPU_review_v1.json')
out=F/'production_configuration_source32_v1';assert not out.exists();out.mkdir();write(out/'configuration.json',config)
proper=read(FIX/'production_configuration_source31_v1/configuration.json')
changed=sorted(k for k in set(config)|set(proper)if config.get(k)!=proper.get(k));assert changed==sorted(['original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy'])
assert config['original_clip_geometry_overlay']['prior_overlay']==proper['original_clip_geometry_overlay']
resource=copy.deepcopy(config['resident_RAM_resource_scheduling_policy']);resource['CPU_protocol_source']=proper['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'];assert resource==proper['resident_RAM_resource_scheduling_policy']
credit=copy.deepcopy(config['resident_RAM_packet_credit_policy']);credit['actual_CPU_control_review']=proper['resident_RAM_packet_credit_policy']['actual_CPU_control_review'];assert credit==proper['resident_RAM_packet_credit_policy']
pointer=read(F/'actual_failure_closure.json')['saved_full_checkpoint']['pointer']
runtime=copy.deepcopy(config);runtime.update(worker_driver_source=pin(driver),parent_driver_source=pin(driver),
 resident_RAM_stream_configuration=proper['resident_RAM_stream_configuration'],resident_branch_namespace=str(B),checkpoint_interval_updates=200,
 source_checkpoint=pointer,source_configuration=pin(out/'configuration.json'))
keys=['worker_driver_source','parent_driver_source','resident_RAM_stream_configuration','resident_branch_namespace','checkpoint_interval_updates','source_checkpoint','source_configuration']
write(F/'source32_configuration_and_future7_runtime_bindings_v1.json',{'proper_source_base':pin(FIX/'production_configuration_source31_v1/configuration.json'),
 'new_source_configuration':pin(out/'configuration.json'),'exact_operational_top_fields':changed,
 'prior7_overlay_tree_exact':True,'all_old_resource_values_exact_except_CPU_source':True,'MAX2_helper_parameters_and_human_approval_exact':True,
 'native225_science_canonical_archive_fresh_signature_collect_prefetch_periodic_DEV_save200_unchanged':True,
 'future_runtime_binding_kinds':keys,'future_runtime_values':{k:runtime[k]for k in keys},
 'source_configuration_remaining_values_exact_after_only_seven_runtime_bindings':True,
 'full_training_target':401072,'admin_stop':None,'new_invocation':24,'new_output':str(F.parent/'resident_RAM_training_actual_25'),
 'new_journal':str(B/'journals/commits_000024.jsonl'),'saved170400_prefix_only':read(pointer['path'])['journal_prefix']})
print(json.dumps({'driver':pin(driver),'wrapper':pin(wrapper),'validator':pin(validator),'config':pin(out/'configuration.json'),
 'inverse':pin(F/'source31_to32_entire_RAW_inverse_v1.json'),'AST':pin(F/'source32_original_numerical_and_nonentry_AST_equivalence_v1.json'),
 'runtime7':pin(F/'source32_configuration_and_future7_runtime_bindings_v1.json')}),flush=True)
