"""Preserve unexecuted34; metadata/AST-only explicit static resolver repair."""
import ast,copy,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as out:json.dump(v,out,indent=2,ensure_ascii=False);out.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
prior=json.loads((F/'source33_to34_one_postDEV_refresh_entire_RAW_AST_inverse_v1.json').read_text())
all_records=[]
for stem,edits in (
 ('training',[
  ('                    old_scalar_ref=weakref.ref(prefetch)',
   '                    original_resolver_identity_for_refresh=id(prefetch.original)\n                    old_scalar_ref=weakref.ref(prefetch)'),
  ('engine.context.resolver is signature_scope.context.resolver','id(engine.context.resolver)==original_resolver_identity_for_refresh'),
  ('                    del refresh_barrier','                    del refresh_barrier,original_resolver_identity_for_refresh')]),
 ('autocontinue',[('run_ASR18_BPE9223_resident_RAM_training_v34','run_ASR18_BPE9223_resident_RAM_training_v35')])):
 old=R/('run_ASR18_BPE9223_resident_RAM_'+stem+'_v34.py');new=R/('run_ASR18_BPE9223_resident_RAM_'+stem+'_v35.py')
 assert not new.exists();t=old.read_bytes().decode();repair=[]
 for a,b in edits:
  n=10 if stem=='autocontinue'else 1
  if '\n'in b:b=b.replace('\n','\r\n')
  assert t.count(a)==n,(a,t.count(a),n)
  repair.append({'old_literal':a,'new_literal':b,'expected_count':n});t=t.replace(a,b)
 ast.parse(t);check=t
 for e in reversed(repair):
  assert check.count(e['new_literal'])==e['expected_count'];check=check.replace(e['new_literal'],e['old_literal'])
 assert check.encode()==old.read_bytes()
 base=prior['sources'][1 if stem=='training'else 2]
 direct=base['replacements']+repair;inverse=t
 for e in reversed(direct):
  assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode()==Path(base['old']['path']).read_bytes()
 new.write_bytes(t.encode())
 all_records.append({'old_unexecuted34':pin(old),'new':pin(new),'exact34_to35_repair_records':repair,
  'proper_admitted33':base['old'],'entire33_to35_replacements':direct,'full33_RAW_forward_inverse_exact':True})
old=ast.parse(Path(prior['sources'][1]['old']['path']).read_bytes());new=ast.parse(Path(all_records[0]['new']['path']).read_bytes())
main=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
loop=lambda tree:next(n for n in ast.walk(main(tree))if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)
    and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['index','entry'])
assert ast.dump(loop(old),include_attributes=False)==ast.dump(loop(new),include_attributes=False)
cp=F/'proposed_source35_configuration_pending_CPU_and_actual_stop_v1.json'
assert not cp.exists();cp.write_bytes((F/'proposed_source34_configuration_pending_CPU_and_actual_stop_v1.json').read_bytes())
repair=F/'source34_to35_resolver_static_binding_and_direct33_RAW_inverse_v1.json'
write(repair,{'kind':'ASR18_INACTIVE_source35_resolver_binding_static_repair_v1','sources':all_records,
 'prior_complete33_to34_ledger':pin(F/'source33_to34_one_postDEV_refresh_entire_RAW_AST_inverse_v1.json'),
 'source34_preserved_unexecuted':True,'static_bug':'FreshMetadataSignatureScope has no .context. Capture exact original resolver id before original worker close and use that id after join/drop.',
 'candidate_import_GC_or_production_execution_occurred':False,'full_native_update_loop_AST_equal_to_admitted33':True,
 'validator_unchanged':prior['sources'][0]['new'],'source35_configuration':pin(cp)})
alias=F/'source35_exact_barrier_dead_alias_and_intentionally_live_ledger_v1.json'
write(alias,{'kind':'ASR18_INACTIVE_source35_explicit_idle_boundary_alias_ledger_v1',
 'source':all_records[0]['new'],
 'order':['successful full original checkpoint and persistence','original20 feature drain/actualwait0/joins/pipes/both credit handle sets closed',
  'real original periodic DEV preservation and phase monitor join','record_periodic persisted','original scalar close(False)/joined worker DB+locks/errorNone',
  'prefetch=None and actual old weakref dead','drop listed aliases','original before byte fingerprint+identity manifests',
  'one unfreeze/original builtin collect/freeze','original after byte fingerprint+exact comparison','new original scalar constructor',
  'new original feature constructor/same immutable queued plan submission'],
 'actual_original_packet_arrays_already_dropped_and_no_second_release':True,
 'completed_checkpoint_aliases_dropped':['plan','plan_pin','coordinator','writer','before','nxt','caller','ptr','proof','ck'],
 'last_completed_update_and_DEV_aliases_dropped':['tx','entry','rows','metadata','native','source','binding','evidence','state',
  'CPU_quiescence','dev_pin','score','preservation','cpu_closed','phase_preserved','scalar_closed'],
 'intentionally_live':['exact original queued tuple/plan/pin (same object IDs and content)',
  'model/teacher/processor/optimizer207/threeAdam215 and owner engine/context/schedule',
  'original continuous parent three metadata locks and fresh signature scope',
  'latest durable checkpoint pins/initial pointer/provenance/augmentation binding/journal namespace and current journal',
  'bounded lifecycle/periodic result histories/winner folders/last five checkpoint paths',
  'original native and original GC/collect-observer ExitStack/Monitor/lease',
  'scalar counters/config/source metadata',
  'known before-fingerprint scalar/hash tree+serialized bytes during event; no whole tensor payload copy'],
 'per_instance_closure':'Integer generation, not retained old-worker object/id-only reuse; new generation remains eligible for final/error close.',
 'unresolved_future_cleaner_phase':'Current next candidate event is weak-label with a real original worker. ROOT interpretation pending for a later native process where no scalar worker is ever constructed; no fake join/closure claim allowed.',
 'unproved':'Opaque framework reachable cycles may later die but remain frozen until original final teardown. No fit or speed or sole-cause guarantee.'})
handoff=json.loads((F/'inactive_source34_postDEV_refresh_source_configuration_handoff_v1.json').read_text())
handoff.update(kind='ASR18_INACTIVE_source35_one_postDEV_refresh_handoff_v1',driver_source=all_records[0]['new'],
 wrapper_source=all_records[1]['new'],proposed_source_configuration=pin(cp),entire_RAW_AST_inverse=pin(repair),
 exact_barrier_alias_ledger=pin(alias),unexecuted34_preserved=all_records[0]['old_unexecuted34'],
 final_source_review_pending=True,future_cleaner_absent_scalar_handling='PENDING ROOT source-preparation interpretation; no runtime selection')
hp=F/'inactive_source35_postDEV_refresh_source_configuration_handoff_v1.json';write(hp,handoff)
print(json.dumps({'driver':all_records[0]['new'],'wrapper':all_records[1]['new'],'config':pin(cp),
 'direct33_and_repair_RAW_ledger':pin(repair),'alias_ledger':pin(alias),'handoff':pin(hp)}))
