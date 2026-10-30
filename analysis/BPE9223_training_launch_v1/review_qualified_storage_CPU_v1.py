"""Narrow qualification storage delta review; metadata-only injected IO.

No real store, file publication/retirement, model, payload or native job.
"""
import copy,difflib,importlib,json,sys,types
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_qualified_storage_review_v1';O.mkdir(exist_ok=True);m.AREA=O
P=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(P))

def tagged(value):
 if type(value) is dict:return ['dict',[[['scalar',k],tagged(v)] for k,v in value.items()]]
 if type(value) is list:return ['list',[tagged(x) for x in value]]
 return ['scalar',value]

def main():
 g=m.Guard();g.phase='narrow_storage_qualification_deltas_and_pure_metadata_controls'
 prep=N/'ROOT_BPE9223_qualified_storage_source_preparation_v1.json';assert m.sha(prep)=='d67f33f75fc57931ab16f7dc1922d52316dfe4647da36cd3e264374720aba9f0'
 spec=json.loads(prep.read_text());deltas=[]
 for component,item in spec['sources'].items():
  old,new,patch=Path(item['baseline']['path']),Path(item['source']['path']),Path(item['exact_delta']['path'])
  assert m.sha(old)==item['baseline']['sha256'] and m.sha(new)==item['source']['sha256']
  actual=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=old.name,tofile=new.name))
  assert actual==patch.read_text()
  deltas.append({'component':component,'old':m.record(old),'new':m.record(new),'exact_delta':m.record(patch)})
 import asr18_BPE9223_owned_checkpoint_storage_v1 as base
 import asr18_BPE9223_indexed_checkpoint_storage_v1 as indexed
 import asr18_BPE9223_persistence_prefix_v4 as prefix
 assert indexed.frozen is base and issubclass(indexed._IndexedProtocol,base._StorageProtocol)
 assert indexed._IndexedProtocol.protect_qualified_winner is base._StorageProtocol.protect_qualified_winner
 assert prefix.FIXED['indexed_storage']==('asr18_BPE9223_indexed_checkpoint_storage_v1.py',m.sha(P/'asr18_BPE9223_indexed_checkpoint_storage_v1.py'))
 assert prefix.FIXED['durable_adapter']==('asr18_BPE9223_durable_owner_adapter_v3.py',m.sha(P/'asr18_BPE9223_durable_owner_adapter_v3.py'))
 assert not any(x in sys.modules for x in ('torch','numpy','transformers','nemo'))
 pointer_pin={'path':str(O/'IN_MEMORY_ONLY.pointer.json'),'bytes':1,'sha256':'1'*64}
 metadata_pin={'path':str(O/'IN_MEMORY_ONLY.state.json'),'bytes':1,'sha256':'2'*64}
 payload_pin={'path':str(Path(metadata_pin['path']).with_suffix('.tensors.bin')),'bytes':1,'sha256':'3'*64}
 writer=P/'bpe_streaming_checkpoint_v1.py';writer_pin={k:v for k,v in m.record(writer).items() if k in ('path','bytes','sha256')}
 cases=[];positives=[]
 def rejected(name,func):
  try:func()
  except (ValueError,KeyError) as error:cases.append({'name':name,'rejected':True,'reason':str(error)})
  else:raise AssertionError('Negative storage qualification fixture accepted: '+name)
 def fixture(expected,pointer_flag=None,saved_flag=None):
  if pointer_flag is None:pointer_flag=expected
  if saved_flag is None:saved_flag=expected
  identity={'fixture':'scalar_semantics_only','qualification':expected};sampler={'carried_replay':{'updates':1}}
  journal={'path':str(O/'IN_MEMORY_ONLY.journal'),'offset':0,'sha256':'0'*64}
  expectation={'run_identity':{'owner_validation_only':expected},'identity':identity,'optimizer_updates':1,'sampler_state':sampler,'journal_prefix':journal}
  metadata={'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'endianness':'little',
   'dynamic_module_serialization':False,'maximum_CPU_tensor_slice_bytes':4*1024**2,'source':writer_pin,
   'tensors':[{'metadata_fixture_placeholder_not_tensor_bytes':True}],'payload':payload_pin,
   'tree':tagged({'kind':'ASR18_broad_caller_state_v1','diagnostic_only':saved_flag,'identity':identity,
    'optimizer':{'updates':1},'sampler':sampler,'augmentation':{'committed_journal':journal}})}
  pointer={'kind':'ASR18_production_window_completed_checkpoint_pointer_v2','writer_returned_and_payload_verified':True,
   'all_window_updates_committed':True,'diagnostic_only':pointer_flag,'checkpoint':metadata_pin,'identity':identity,
   'committed_updates':1,'sampler_state_sha256':base.digest(sampler),'journal_prefix':journal,'payload':payload_pin}
  # Build the semantic reference with the correct saved flag; mismatched copies
  # must be rejected by the REAL frozen scalar verifier called below.
  correct=copy.deepcopy(metadata);correct['tree']=tagged({'kind':'ASR18_broad_caller_state_v1','diagnostic_only':expected,'identity':identity,
   'optimizer':{'updates':1},'sampler':sampler,'augmentation':{'committed_journal':journal}})
  if type(expected) is bool:
   pointer['saved_semantic_state_verified']=base.verify_saved_metadata(correct,identity=identity,optimizer_updates=1,sampler_state=sampler,journal_prefix=journal,diagnostic_only=expected)
  reads=[];bounds=[]
  def read(pin):
   reads.append(pin['path'])
   if pin==pointer_pin:return copy.deepcopy(pointer)
   if pin==metadata_pin:return copy.deepcopy(metadata)
   raise AssertionError('Unexpected injected metadata read')
  def bound(pin):bounds.append(pin);assert pin==payload_pin;return pin
  # Copy only function globals; production module/read/bound remain unmodified.
  fn=types.FunctionType(base.verify_streamed_pointer.__code__,{**base.verify_streamed_pointer.__globals__,'read':read,'bound':bound},'metadata_IO_fixture')
  return fn,expectation,reads,bounds
 for flag in (True,False):
  fn,expectation,reads,bounds=fixture(flag);result=fn(pointer_pin,expectation)
  assert result['semantic_proof']['diagnostic_only'] is flag and bounds==[payload_pin] and len(reads)==2
  positives.append({'owner_validation_only':flag,'pointer_and_actual_scalar_semantic_checker_accept_matching_bool':True})
  for observed in ((False,1) if flag else (True,0)):
   fn,expectation,_,_=fixture(flag,pointer_flag=observed)
   rejected('pointer_cross_flag_'+repr(flag)+'_'+repr(observed)+'_'+type(observed).__name__,lambda f=fn,e=expectation:f(pointer_pin,e))
   fn,expectation,_,_=fixture(flag,saved_flag=observed)
   rejected('saved_semantics_cross_flag_'+repr(flag)+'_'+repr(observed)+'_'+type(observed).__name__,lambda f=fn,e=expectation:f(pointer_pin,e))
 for invalid in (0,1,None,'True'):
  fn,expectation,reads,_=fixture(True);expectation['run_identity']['owner_validation_only']=invalid
  rejected('expectation_typed_flag_'+repr(invalid),lambda f=fn,e=expectation:f(pointer_pin,e));assert reads==[]
 fn,expectation,reads,_=fixture(True);expectation['run_identity']={}
 rejected('expectation_missing_flag',lambda:fn(pointer_pin,expectation));assert reads==[]
 class LockReached(Exception):pass
 for cls in (base._StorageProtocol,indexed._IndexedProtocol):
  for flag in (True,0,1,None):
   obj=object.__new__(cls);obj.owner={'run_identity':{} if flag is None else {'owner_validation_only':flag}}
   events=[]
   def locked():events.append('lock');raise LockReached()
   obj.locked=locked
   rejected(cls.__name__+'_winner_reject_'+repr(flag),lambda o=obj:o.protect_qualified_winner('unused',None,None));assert events==[]
  obj=object.__new__(cls);obj.owner={'run_identity':{'owner_validation_only':False}};obj.locked=lambda:(_ for _ in ()).throw(LockReached())
  try:obj.protect_qualified_winner('unused',None,None)
  except LockReached:pass
  else:raise AssertionError('Explicit productionFalse failed to reach unchanged locked owner/science gate')
 assert not any(x in sys.modules for x in ('torch','numpy','transformers','nemo'))
 controls={'kind':'Actual_qualified_storage_pointer_flag_metadata_controls_v1','at_utc':m.now(),'passed':True,
  'matching_bool_positive_controls':positives,'negative_controls':cases,'actual_frozen_saved_scalar_semantics_checker_used':True,
  'pointer_read_and_payload_bound_IO_injected_without_mutating_modules':True,
  'qualification_winner_rejected_before_lock_or_owner_files':True,'explicitFalse_winner_reaches_unchanged_lock_gate':True,
  'no_real_store_namespace_payload_publication_or_retirement_created':True,
  'actual_payload_tensor_model_audio_GPU_heavy_imports_or_deletion':False}
 m.write('actual_metadata_controls_v1.json',controls)
 finding={'kind':'Independent_BPE9223_qualified_storage_exact_delta_findings_v1','at_utc':m.now(),
  'reviewer':'/root/label_bpe_audit','accepted_source_and_metadata_scope':True,'execution_or_storage_retirement_admission':False,
  'exact_source_deltas':deltas,'metadata_controls':m.record(O/'actual_metadata_controls_v1.json'),
  'findings':[
   'Indexed9223 inherits the NEW qualified base protocol and passes new verify_streamed_pointer to its public constructor; no old hardFalse verifier remains on this bound path.',
   'Pointer qualification is exact typed expectation.run_identity.owner_validation_only; both explicit bool scopes pass matching pointer/saved scalar semantic checks. Missing/integer/string scope and pointer/saved crossbool or equal-numeric flags reject.',
   'Base and inherited indexed winner-protection gate reject qualificationTrue before lock/owner/files; only explicitFalse reaches unchanged actual source/owner/scorer-qualified proof checks.',
   'Prefix4 changes only exact new indexed source pin and its creation-identity imports; fixed packed3 pin remains exact. The full original indexed accounting/cadence/archive/quiescence/retirement/protection implementation is unchanged outside import.',
   'No guard, E reserve, protected source/history/DEV/winner/current pointer rule, actual-process proof, production_enabledFalse requirement, or current-artifact deletion authority was relaxed.'
  ],'blocking_source_gaps_found_in_narrow_scope':[],
  'remaining_actual_proofs':'Real source-bound two-window/native/persistence/current-recovery/storage operations and fresh resource/admission remain owner tests; these pure metadata controls do not establish physical storage or tensors.',
  'ROOT_and_original_sources_unchanged':True}
 m.write('source_findings_v1.json',finding);g.phase='complete_qualified_storage_narrow_review';g.publish(passed=True)
 print(json.dumps({'findings':m.record(O/'source_findings_v1.json'),'controls':m.record(O/'actual_metadata_controls_v1.json'),'negative_controls':len(cases)}),flush=True)
if __name__=='__main__':main()
