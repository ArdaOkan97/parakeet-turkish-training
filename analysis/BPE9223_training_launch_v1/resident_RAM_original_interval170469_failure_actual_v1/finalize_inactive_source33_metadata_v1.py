"""Metadata-only successor for two pre-execution preparation wiring corrections."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research'
F=Path(__file__).resolve().parent;FIX=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
def pin(p):
 p=Path(p).resolve();v=p.read_bytes();return {'path':str(p),'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);assert not p.exists(),p;p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
records={}
for kind in ['training','autocontinue']:
 old=R/('run_ASR18_BPE9223_resident_RAM_'+kind+'_v32.py');new=R/('run_ASR18_BPE9223_resident_RAM_'+kind+'_v33.py')
 raw=old.read_bytes();value=raw.decode();edits=[{'old_literal':'    import sys\r\n    if str(Path(',
  'new_literal':'    if str(Path(','expected_count':1}]
 if kind=='autocontinue':edits.append({'old_literal':'run_ASR18_BPE9223_resident_RAM_training_v32',
  'new_literal':'run_ASR18_BPE9223_resident_RAM_training_v33','expected_count':10})
 for e in edits:assert value.count(e['old_literal'])==e['expected_count'];value=value.replace(e['old_literal'],e['new_literal'])
 inverse=value
 for e in reversed(edits):assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode()==raw;ast.parse(value);assert not new.exists();new.write_bytes(value.encode())
 # No function-local binding of sys may shadow the existing top-level import.
 main=next(n for n in ast.parse(value).body if isinstance(n,ast.FunctionDef)and n.name=='main')
 assert not any(isinstance(n,ast.Import)and any(a.name=='sys'for a in n.names)for n in ast.walk(main))
 records[kind]={'old_source':pin(old),'new_source':pin(new),'replacements_in_forward_order':edits,
  'full_raw_inverse_exact':True,'newline_normalization':'NONE; original CRLF retained'}
 diff=F/('source32_to33_'+kind+'_v1.diff');assert not diff.exists();diff.write_bytes(''.join(difflib.unified_diff(raw.decode().splitlines(True),value.splitlines(True),fromfile=str(old),tofile=str(new))).encode())
write(F/'source32_to33_entire_RAW_inverse_v1.json',records)
previous=read(F/'source31_to32_entire_RAW_inverse_v1.json')
combined={}
for newkey,oldkey in [('training','driver'),('autocontinue','wrapper')]:
 chain=[*previous[oldkey]['replacements_in_forward_order'],*records[newkey]['replacements_in_forward_order']]
 original=Path(previous[oldkey]['old_source']['path']).read_bytes();current=Path(records[newkey]['new_source']['path']).read_bytes().decode()
 value=original.decode()
 for e in chain:assert value.count(e['old_literal'])==e['expected_count'];value=value.replace(e['old_literal'],e['new_literal'])
 assert value==current
 for e in reversed(chain):assert current.count(e['new_literal'])==e['expected_count'];current=current.replace(e['new_literal'],e['old_literal'])
 assert current.encode()==original
 combined[newkey]={'old_source':previous[oldkey]['old_source'],'new_source':records[newkey]['new_source'],
  'replacements_in_forward_order':chain,'full_raw_forward_and_inverse_exact':True,'newline_normalization':'NONE'}
write(F/'source31_to33_entire_RAW_inverse_v1.json',combined)
oldtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v31.py').read_bytes());newtree=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_training_v33.py').read_bytes())
top=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name!='main'}
assert top(oldtree)==top(newtree)
om=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=='main');nm=next(n for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
loops=lambda tree:[ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.While)]
assert loops(om)==loops(nm)and len(loops(om))==3
on={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(om)if isinstance(n,ast.FunctionDef)and n is not om}
nn={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(nm)if isinstance(n,ast.FunctionDef)and n is not nm}
assert [k for k in on if on[k]!=nn[k]]==['journal_factory']
write(F/'source33_original_numerical_and_nonentry_AST_equivalence_v1.json',{'all_old_nonentry_AST_exact':list(top(oldtree)),
 'all_three_native_while_AST_exact':True,'unaffected_main_nested_AST_exact':[k for k in on if k!='journal_factory'],
 'only_changed_nested_main':['journal_factory'],'no_new_function_local_sys_import':True,
 'native_GC_DEV_checkpoint_and_commit_numerical_bodies_unchanged':True})
configpath=F/'production_configuration_source32_v1/configuration.json';configuration=read(configpath)
properpath=FIX/'production_configuration_source31_v1/configuration.json';proper=read(properpath)
oldruntime=read(B/'invocations/owner_000023/configuration.json')
driver=R/'run_ASR18_BPE9223_resident_RAM_training_v33.py';wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py'
pointer=read(F/'actual_failure_closure.json')['saved_full_checkpoint']['pointer']
bindings={'worker_driver_source':pin(driver),'parent_driver_source':pin(driver),
 'resident_RAM_stream_configuration':oldruntime['resident_RAM_stream_configuration'],
 'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':pointer,'source_configuration':pin(configpath)}
changed=sorted(k for k in set(configuration)|set(proper)if configuration.get(k)!=proper.get(k))
assert changed==sorted(['original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy'])
restored=copy.deepcopy(configuration)
for k in changed:restored[k]=proper[k]
assert restored==proper
runtime=copy.deepcopy(configuration);runtime.update(bindings)
restoredruntime=copy.deepcopy(oldruntime)
for k in bindings:
 if k in proper:restoredruntime[k]=proper[k]
 else:del restoredruntime[k]
assert restoredruntime==proper
report={'kind':'ASR18_source33_exact_source_configuration_future_runtime7_v1','source_configuration':pin(configpath),
 'proper_source31_base':pin(properpath),'exact_operational_top_fields':changed,'all_other_source_values_exact':True,
 'old_runtime_to_source_diff_is_exactly_seven_binding_kinds':True,'future_runtime_binding_kinds':list(bindings),
 'future_runtime_values':bindings,'future_runtime_JSON_not_written_or_adopted':True,
 'new_invocation':24,'new_output':str(F.parent/'resident_RAM_training_actual_25'),'new_journal':str(B/'journals/commits_000024.jsonl'),
 'full_training_target':401072,'admin_stop':None,'saved_prefix':read(pointer['path'])['journal_prefix'],
 'preserved69_require_genuine_redo':True,'source32_unexecuted_preparation_retained':True}
write(F/'source33_configuration_and_future7_runtime_bindings_v1.json',report)
write(F/'source32_preparation_only_rejection_and_source33_successor_v1.json',{'actual_TOOL_exit':1,'actual_TOOL_chunk':'e5c11b',
 'preparation_error':"KeyError('resident_RAM_stream_configuration')",'production_or_control_started':False,
 'created_immutable32_sources_config_and_inverse_preserved':True,
 'separate_static_entry_hazard':'new function-local import sys shadows the existing global sys before first use',
 'successor33_correction':pin(F/'source32_to33_entire_RAW_inverse_v1.json'),
 'runtime_report_correction':'actual original stream binding comes from prior runtime, one of seven runtime binding kinds',
 'new_configuration_numeric_or_policy_change':False})
print(json.dumps({'driver':pin(driver),'wrapper':pin(wrapper),'configuration':pin(configpath),
 'inverse':pin(F/'source31_to33_entire_RAW_inverse_v1.json'),'runtime7':pin(F/'source33_configuration_and_future7_runtime_bindings_v1.json')}),flush=True)
