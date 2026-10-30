"""Freeze inactive observer2: sampler alone owns retained query-handle closure."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).resolve().parent
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 assert not p.exists();p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
def successor(oldname,newname,edits):
 old=F/oldname;new=F/newname;raw=old.read_bytes();value=raw.decode();ledger=[]
 for before,after in edits:
  if value.count(before)==0:
   before=before.replace('\n','\r\n');after=after.replace('\n','\r\n')
  assert value.count(before)==1,(before[:100],value.count(before))
  value=value.replace(before,after);ledger.append({'old_literal':before,'new_literal':after,'expected_count':1})
 ast.parse(value);inverse=value
 for e in reversed(ledger):
  assert inverse.count(e['new_literal'])==1
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode()==raw and not new.exists();new.write_bytes(value.encode())
 return {'old_source':pin(old),'new_source':pin(new),'newline_normalization':'NONE; preserve each original byte outside declared edits',
  'replacements_in_forward_order':ledger,'entire_RAW_forward_and_inverse_exact':True}
ol=successor('observe_existing_source33_thread_CPU_and_stacks_v1.py','observe_existing_source33_thread_CPU_and_stacks_v2.py',[
 ("  sampler.close()\n  for q in lifetime_queries.values():\n   if not q.closed:q.close()\n  scalar_files.close();counter=OUT/'thread_CPU_counters.jsonl'",
  "  sampler.close()\n  scalar_files.close();counter=OUT/'thread_CPU_counters.jsonl'"),
 (" finally:\n  for q in lifetime_queries.values():\n   if not q.closed:\n    try:q.close()\n    except BaseException as e:cleanup.append({'kind':'retained_lifetime_query_close_error','type':type(e).__name__,'message':str(e)})\n  if scalar_files is not None and not scalar_files.closed:",
  " finally:\n  if scalar_files is not None and not scalar_files.closed:")])
el=successor('launch_existing_source33_thread_CPU_external_v1.py','launch_existing_source33_thread_CPU_external_v2.py',[
 ("FIXTURE=F/'observe_existing_source33_thread_CPU_and_stacks_v1.py'","FIXTURE=F/'observe_existing_source33_thread_CPU_and_stacks_v2.py'")])
tree=ast.parse(Path(ol['new_source']['path']).read_bytes())
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='ThreadQuery'for n in ast.walk(tree))
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='q'and n.func.attr=='close'for n in ast.walk(tree))
ledger=F/'current_source33_sampler_sole_owner_live_v2_entire_RAW_inverse.json'
write(ledger,{'observer':ol,'external':el,'sampler_is_only_handle_closure_owner':True,
 'existing_sampler_active_aliases_add_no_Query_or_OpenThread':'Aliases sampled after profiler; sampler.close owns all actual handle closure',
 'v1_preserved_unexecuted':True,'new_helper_or_controller_control_required':False})
oldhandoff=F/'current_source33_fixed60_existing_handle_live_inactive_handoff_v1.json'
h=json.loads(oldhandoff.read_bytes());h['kind']='ASR18_current_source33_fixed60_existing_query_handle_readonly_attribution_inactive_handoff_v2'
replacements={ol['old_source']['path']:ol['new_source'],el['old_source']['path']:el['new_source']}
h['input_pins']=[replacements.get(v['path'],v)for v in h['input_pins']]
h['input_pins'] += [pin(ledger),pin(oldhandoff)]
h['live_observer_source']=ol['new_source'];h['external_live_launcher_source']=el['new_source']
h['exact_external_arguments'][2]=el['new_source']['path']
h['whole_RAW_inverse_v1_preserved']=h.pop('whole_RAW_inverse');h['whole_RAW_inverse']=pin(ledger)
h['sampler_is_sole_query_handle_closure_owner']=True
h['lifetime_proof']='References to SAME sampler.active handles are sampled after actual profiler exit; aliases NEVER close; sampler.close is sole ownership/closure path'
h['v1_source_and_handoff_preserved_unexecuted']=pin(oldhandoff)
h['proofs_reused_without_new_control']='Closed genuine counter/role/handle8f2ee and fixed60 METHOD_STUB controller16e8 remain exact; no helper changes, no attach executed'
handoff=F/'current_source33_fixed60_existing_handle_live_inactive_handoff_v2.json';write(handoff,h)
print(json.dumps({'observer':ol['new_source'],'external':el['new_source'],'inverse':pin(ledger),'handoff':pin(handoff),
 'input_pins':len(h['input_pins']),'execution_admitted':False}),flush=True)
