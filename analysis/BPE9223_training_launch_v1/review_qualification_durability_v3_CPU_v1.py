"""Exact durability deltas and bounded metadata qualification reachability.

Never constructs a real adapter namespace or launches a callback/native job.
"""
import ast,difflib,importlib,json,sys,types
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_qualification_durability_v3_review_v1';O.mkdir(exist_ok=True);m.AREA=O
P=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(P))

def pin3(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}

def main():
 g=m.Guard();g.phase='qualification_durability_v3_source_and_metadata_only_review'
 delta_records=[]
 for version in (1,2,3):
  prep=N/('ROOT_9223_qualification_durability_source_preparation_v%d.json'%version)
  report=json.loads(prep.read_text())
  for key,item in report['sources'].items():
   old=Path(item['baseline']['path']);new=Path(item['source']['path']);patch=Path(item['exact_delta']['path'])
   assert m.sha(old)==item['baseline']['sha256'] and m.sha(new)==item['source']['sha256']
   actual=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=old.name,tofile=new.name))
   assert actual==patch.read_text()
   delta_records.append({'version':version,'component':key,'old':m.record(old),'new':m.record(new),'exact_delta':m.record(patch)})
 lcpath=P/'asr18_BPE9223_lifecycle_v3.py';assert m.sha(lcpath)=='fd64f67b3daec56f48e375947699965218e1437f8ad7f4eded7e0bc9daf0f9a5'
 # Only latest frozen sources are imported/exercised. Old1/2 are read-only diff baselines.
 import asr18_BPE9223_lifecycle_v3 as lc
 import asr18_BPE9223_durable_transitions_v3 as tr
 import asr18_BPE9223_durable_owner_adapter_v3 as packed
 import asr18_BPE9223_persistence_prefix_v3 as prefix
 assert not any(x in sys.modules for x in ('torch','numpy','transformers','nemo'))
 assert tr.FullRunLifecycle is packed.FullRunLifecycle is lc.FullRunLifecycle
 assert tr._LifecycleProtocol is packed._LifecycleProtocol is lc._LifecycleProtocol
 assert issubclass(packed.PackedDurableLifecycleAdapter,tr.DurableLifecycleAdapter)
 for name in ('record_pending_window','acknowledge_recovery','acknowledge_DEV_and_winner'):
  assert getattr(packed.PackedDurableLifecycleAdapter,name) is getattr(tr.DurableLifecycleAdapter,name)
 depname,depsha=prefix.FIXED['durable_adapter'];assert depname=='asr18_BPE9223_durable_owner_adapter_v3.py' and m.sha(P/depname)==depsha
 results=[]
 def rejected(name,fn,expected=None):
  try:fn()
  except (ValueError,KeyError) as error:
   if expected:assert expected in str(error),(name,error)
   results.append({'name':name,'rejected':True,'reason':str(error)})
  else:raise AssertionError('Negative qualification control accepted: '+name)
 for label,func in [('transitions',tr.diagnostic_identity),('prefix',prefix.diagnostic_identity),('packed_alias',packed.diagnostic_identity)]:
  for flag in (True,False):assert func({'owner_validation_only':flag}) is flag
  for invalid in ({},{'owner_validation_only':0},{'owner_validation_only':1},{'owner_validation_only':None},{'owner_validation_only':'True'}):
   rejected(label+'_typed_identity_'+repr(invalid),lambda f=func,v=invalid:f(v),'Typed immutable')
 # Real inherited recovery method and real prefix current-evidence method reject
 # wrong qualification flags before later evidence or any native callback.
 fixtures=O/'fixtures';fixtures.mkdir(exist_ok=True)
 for expected in (True,False):
  for observed in ((False,1) if expected else (True,0)):
   proof={'kind':'ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1','diagnostic_only':observed}
   path=fixtures/('recovery_expected%s_observed%s_%s.json'%(expected,observed,type(observed).__name__))
   path.write_text(json.dumps(proof)+'\n')
   obj=object.__new__(packed.PackedDurableLifecycleAdapter);obj.identity={'owner_validation_only':expected}
   obj.head={'pending':{'publications':1}};obj._check=lambda:None;obj.poisoned=False
   obj.lifecycle=types.SimpleNamespace(protocol=types.SimpleNamespace(poisoned=False))
   rejected('actual_inherited_recovery_bool_cross_'+path.stem,lambda o=obj,p=pin3(path):o.acknowledge_recovery(p),'Current/latest recovery')
   evidence={'kind':'ASR18_full_run_current_recovery_prefix_inputs_v1','owner':prefix.OWNER,'diagnostic_only':observed}
   rejected('actual_prefix_current_bool_cross_'+path.stem,
    lambda e=evidence,f=expected:prefix._current(e,types.SimpleNamespace(diagnostic=f),None,None,None),'Pre-ack source-bound')
 # Qualification DEV rejection occurs before schedule/evaluator/winner access.
 for cls in (tr.DurableLifecycleAdapter,packed.PackedDurableLifecycleAdapter):
  obj=object.__new__(cls);obj.identity={'owner_validation_only':True};obj._check=lambda:None;obj.poisoned=False
  obj.lifecycle=types.SimpleNamespace(protocol=types.SimpleNamespace(poisoned=False))
  rejected(cls.__name__+'_qualification_DEV_winner_ack',lambda o=obj:o.acknowledge_DEV_and_winner(report=None,winner=None),'Validation checkpoints cannot')
 p=object.__new__(lc._LifecycleProtocol);p.identity={'owner_validation_only':True}
 rejected('protocol_qualification_DEV_rejects_before_state_access',lambda:p.accept_DEV(None,None,None,None),'Qualification cannot')
 # Exercise the ACTUAL latest advance method. The early guard must run before
 # even checkpoint_storage_boundary/admit callbacks; fake callbacks never do IO.
 class CallbackReached(Exception):pass
 for flag in (True,False):
  events=[]
  def call(name,request):events.append(name);raise CallbackReached(name)
  obj=object.__new__(lc.FullRunLifecycle);obj.identity={'owner_validation_only':flag}
  obj.schedule=types.SimpleNamespace(state_dict=lambda:{'fixture':'inert_schedule'})
  obj.protocol=types.SimpleNamespace(request=lambda state:{'kind':'checkpoint_then_serial350DEV'},state={})
  obj.callbacks=types.SimpleNamespace(call=call)
  if flag:
   rejected('latest_direct_advance_qualification_rejects_before_ANY_callback',obj.advance,'Qualification cannot invoke')
   assert events==[]
  else:
   try:obj.advance()
   except CallbackReached:assert events==['checkpoint_storage_boundary']
   else:raise AssertionError('Productionfalse failed to reach unchanged source/resource callback gate')
 # Confirm no old literal diagnostic_onlyFalse gates remain in the three latest
 # durability sources; production_enabledFalse and physical/process/science
 # statements are unchanged by the exact published source deltas.
 for path in [P/'asr18_BPE9223_durable_transitions_v3.py',P/'asr18_BPE9223_durable_owner_adapter_v3.py',P/'asr18_BPE9223_persistence_prefix_v3.py']:
  for node in ast.walk(ast.parse(path.read_text())):
   if isinstance(node,ast.Compare) and "diagnostic_only" in ast.unparse(node.left):
    assert not any(isinstance(x,ast.Constant) and x.value is False for x in node.comparators)
 assert not any(x in sys.modules for x in ('torch','numpy','transformers','nemo'))
 controls={'kind':'Actual_qualification_durability_v3_metadata_controls_v1','at_utc':m.now(),'passed':True,
  'negative_controls':results,'both_true_false_typed_helpers_allowed':True,
  'all_three_components_reach_frozen_lifecycle3_classes':True,'packed_methods_are_actual_inherited_transitions3':True,
  'fixed_prefix_adapter_dependency_SHA_matches_actual_current3':True,
  'qualificationTrue_direct_advance_invokes_zero_callbacks':True,
  'productionFalse_direct_advance_reaches_existing_storage_callback_gate':True,
  'real_adapter_namespace_or_callback_native_IO_created':False,'heavy_imports_audio_GPU_payload_or_deletion':False}
 m.write('actual_metadata_controls_v1.json',controls)
 findings={'kind':'Independent_qualification_durability3_source_and_metadata_findings_v1','at_utc':m.now(),
  'reviewer':'/root/label_bpe_audit','accepted_source_and_metadata_scope':True,'execution_or_production_admission':False,
  'current_sources':{name:m.record(P/name) for name in ('asr18_BPE9223_durable_transitions_v3.py','asr18_BPE9223_durable_owner_adapter_v3.py','asr18_BPE9223_persistence_prefix_v3.py','asr18_BPE9223_lifecycle_v3.py')},
  'original_to1_to2_to3_exact_delta_chain':delta_records,'controls':m.record(O/'actual_metadata_controls_v1.json'),
  'resolved_real_gap':'Frozen lifecycle2 direct advance could call DEV/winner callbacks before late protocol rejection; new lifecycle3 rejects qualificationTrue nontraining actions before storage/admission/DEV/winner callbacks. No old2 execution was performed.',
  'findings':[
   'All former diagnostic_onlyFalse gates in exact reviewed three-source deltas now use typed immutable identity.owner_validation_only, identity comparison or typed bool comparison; missing/int/string flags and mismatching/equal-numeric bool packet flags are rejected.',
   'Current3 transitions and packed owner import the SAME frozen lifecycle3 FullRunLifecycle/_LifecycleProtocol classes; inherited normal pending-window/recovery/DEV transitions are reachable via that new type, not blocked by old isinstance checks.',
   'Both inherited durability DEV/winner acknowledgment and direct lifecycle3 advance reject validationTrue before evaluator/winner side effects. ProductionFalse still reaches the unchanged source/storage/resource gate.',
   'Prefix FIXED durable dependency points current3 packed source with exact actualSHA; immutable policy/source/identity/process/native0/recovery/persistence-prefix gates remain source-identical except explicit qualification and new module pins.',
   'production_enabledFalse, fixed science/scorer/retention/fullpasses and storage limits remain unchanged; no source gives current-artifact adoption or deletion authority.'
  ],'blocking_gaps_found_in_current3_review_scope':[],
  'remaining_actual_proofs':'Actual native/recovery/prefix/durable writer/quiescence/DEV processes, resource and final source/config admission remain separate owner tests; metadata controls do not establish them.',
  'ROOT_or_memory_owned_sources_modified':False,'old_versions_preserved':True}
 m.write('source_findings_v1.json',findings);g.phase='complete_qualification_durability3_review';g.publish(passed=True)
 print(json.dumps({'findings':m.record(O/'source_findings_v1.json'),'controls':m.record(O/'actual_metadata_controls_v1.json'),'negative_controls':len(results)}),flush=True)
if __name__=='__main__':main()
