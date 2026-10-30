"""Narrow sustained runner report contract/source review; synthetic metadata only."""
import ast,copy,difflib,importlib,json,sys,types
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_sustained_runner2_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';R=m.PROJECT/'work/research';sys.path.insert(0,str(S))

def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def require(value,message):
 if not value:raise ValueError(message)

def main():
 g=m.Guard();g.phase='sustained_runner2_source_and_CPU3_report_API'
 p1=R/'run_ASR18_BPE9223_sustained_v1.py';p2=R/'run_ASR18_BPE9223_sustained_v2.py'
 assert m.sha(p1)=='4c475dac472c68632965eb2aa2b322109fda878021b43dcbd4b7c7dc7290ebf9'
 assert m.sha(p2)=='231ba3c8cfa15f4237c4117d5ab1fb046d7e568b80d8d2e25c0cadc43d2f41ff'
 t1=ast.parse(p1.read_text());t2=ast.parse(p2.read_text());f1={n.name:n for n in t1.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};f2={n.name:n for n in t2.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 changed=[name for name in f1 if ast.dump(f1[name])!=ast.dump(f2[name])]
 assert changed==['gpu','run_CPU','parent'] and set(f2)-set(f1)=={'verify_aggregate_CPU_report'}
 assert ast.dump(f1['restore_rng'])==ast.dump(f2['restore_rng'])
 calls=[n for n in ast.walk(t2) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='restore_rng'];assert not calls
 source=p2.read_text();assert "proof['named_tensors_verified']==854 and proof['Adam_states_verified']==215" in source
 assert 'coordinator.index==optimizer.updates==c[\'native_updates\']' in source
 assert "c['native_updates']==20" in source and "len(plan['updates'])==20" in source
 assert "for index,entry in enumerate(plan['updates'][:c['native_updates']])" in source
 assert "all_children_actually_closed0" not in source
 cpu=R/'run_ASR18_BPE9223_CPU_window_v3.py';ct=ast.parse(cpu.read_text())
 result=next(n for n in ast.walk(ct) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='result' for x in n.targets) and isinstance(n.value,ast.Dict)).value
 keys={k.value for k in result.keys if isinstance(k,ast.Constant)}
 assert {'reference_actual_aggregate0','frontend_actual_aggregate0','real_actual_child0_count','per_update_child0_count_claimed','updates','full_rows','individual_array_raw_hashes','actual_optimizer_updates','parent_source','worker_source','owner_validation_only','checkpoint_diagnostic_only'}<=keys
 assert 'all_children_actually_closed0' not in keys
 worker_sources={'reference':rec(R/'produce_FastMSS_BPE9223_reference_CPU_v3.py'),'frontend':rec(R/'produce_FastMSS_BPE9223_frontend_CPU_v3.py')}
 controller=rec(cpu);guard_source=rec(R/'ASR18_bounded_CPU_phase_v3.py')
 import fastmss_owned_process_closure_v2 as proc
 proc_pin=rec(proc.__file__)
 pins={};objects={};checks=[]
 def add(name,obj):
  raw=json.dumps(obj,sort_keys=True).encode();pin={'path':str(O/('IN_MEMORY_ONLY_'+name+'.json')),'bytes':len(raw),'sha256':__import__('hashlib').sha256(raw).hexdigest()};pins[pin['path']]=pin;objects[pin['path']]=obj;return pin
 contract={'sources':worker_sources};contract_pin=add('contract',contract)
 config={'recorded_room_source_contract':contract_pin,'CPU_controller_source':controller}
 def read_json(path):return copy.deepcopy(objects[str(path)])
 def record(path):return copy.deepcopy(pins[str(path)]) if str(path) in pins else rec(path)
 def read(pin,guard=lambda:None):
  assert record(pin['path'])==pin;return read_json(pin['path'])
 def dead(identity):return True
 clean=types.FunctionType(proc.clean_guard.__code__,{**proc.clean_guard.__globals__,'read':read},'synthetic_clean_guard')
 verifier=types.FunctionType(proc.verify.__code__,{**proc.verify.__globals__,'read':read,'record':lambda p,guard=lambda:None:record(p),'dead':dead,'clean_guard':clean},'synthetic_process_metadata_verifier')
 facade=types.ModuleType('fastmss_owned_process_closure_v2');facade.verify=verifier
 # The helper's real body imports this facade; all physical IO/liveness are
 # explicitly injected. The real process verifier's semantic body is retained.
 sys.modules['fastmss_owned_process_closure_v2']=facade
 namespace={'read_json':read_json,'record':record,'require':require}
 exec(compile(ast.Module(body=[copy.deepcopy(f2['verify_aggregate_CPU_report'])],type_ignores=[]),str(p2),'exec'),namespace)
 helper=namespace['verify_aggregate_CPU_report']
 value={'passed':True,'real_actual_child0_count':2,'per_update_child0_count_claimed':0,'actual_optimizer_updates':0}
 for index,(key,role) in enumerate((('reference_actual_aggregate0','reference'),('frontend_actual_aggregate0','frontend'))):
  observer={'pid':9000,'ppid':8000,'create_time':1.};launcher={'pid':10000+index*10,'ppid':9000,'create_time':2.};producer={'pid':10001+index*10,'ppid':launcher['pid'],'create_time':3.}
  report=add(role+'_report',{'passed':True,'producer':producer,'worker_source':worker_sources[role],'parent_source':controller})
  closure=add(role+'_guard',{'pid':producer['pid'],'actual_exit_code':0,'monitor_joined':True,'monitor_clean':True,'observed':{'monitor_error':None,'samples':3}})
  proof={'kind':'ASR18_owned_CPU_process_completion_v1','owner':'/root/cpu_data_preparation','report':report,'producer':producer,'observer':observer,'launcher':launcher,
   'worker_source':worker_sources[role],'parent_source':controller,'guard_source':guard_source,'observer_source':proc_pin,'guard_closure':closure,
   'actual_Popen_wait_code':0,'actual_native_exit_code':0,'parent_observed_wait':True,'producer_dead_after_wait':True,
   'parent_guard_observation':{'paired_sampling_within_one_second':True,'worker_private_memory':{'pid':9000,'available':True,'private_bytes':32000000},'system_memory':{'available':True,'commit_used_bytes':10000000000,'commit_limit_bytes':40000000000,'commit_available_bytes':30000000000}}}
  value[key]={'actual0':add(role+'_proof',proof),'report':report}
 assert helper(value,config,g.check) is value
 def rejected(name,operation):
  try:operation()
  except (AssertionError,KeyError,ValueError) as e:checks.append({'name':name,'rejected':True,'reason':str(e)})
  else:raise AssertionError('Negative report fixture accepted: '+name)
 for key in ('reference_actual_aggregate0','frontend_actual_aggregate0'):
  missing=copy.deepcopy(value);del missing[key];missing['all_children_actually_closed0']=True
  rejected('legacy_flag_not_nested_proof_'+key,lambda v=missing:helper(v,config,g.check))
  altered=copy.deepcopy(value);altered[key]['actual0']['sha256']='f'*64
  rejected('wrong_actual0_binding_'+key,lambda v=altered:helper(v,config,g.check))
  path=value[key]['actual0']['path'];original=copy.deepcopy(objects[path])
  for field,bad in (('actual_Popen_wait_code',1),('actual_Popen_wait_code',False),('actual_native_exit_code',1),('owner','/root/other'),('worker_source',controller),('parent_source',worker_sources['reference'])):
   objects[path]=copy.deepcopy(original);objects[path][field]=bad
   rejected(key+'_'+field+'_'+str(bad),lambda:helper(value,config,g.check))
  objects[path]=original
 for field,bad in (('real_actual_child0_count',1),('per_update_child0_count_claimed',20),('actual_optimizer_updates',1),('passed',False)):
  altered=copy.deepcopy(value);altered[field]=bad;rejected('top_'+field,lambda v=altered:helper(v,config,g.check))
 assert len(checks)==20
 sys.modules['fastmss_owned_process_closure_v2']=proc
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'))
 g.check()
 controls={'passed':True,'positive_current_CPU3_report_contract':True,'negative_cases':checks,'negative_count':len(checks),'physical_IO_liveness_injected':True,'real_process_verifier_metadata_semantics':True,'actual_GPU_audio_model_payload_storage_jobs':0}
 (O/'actual_metadata_report_boundary_controls_v1.json').write_text(json.dumps(controls,indent=2)+'\n')
 report={'kind':'ASR18_independent_sustained_runner2_source_API_review_v1','passed':True,'reviewer':'/root/label_bpe_audit','scope':'finite original-source9223 sustained20 source/API; not execution or resource admission',
  'runner':rec(p2),'prior_runner':rec(p1),'CPU3_controller':controller,'process_proof_verifier':proc_pin,'bootstrap':rec(S/'asr18_BPE9223_bootstrap_v2.py'),'grouped_native':rec(S/'asr18_BPE9223_grouped_native_v1.py'),
  'runner1_to_2_AST_changed_functions':changed,'only_new_function':'verify_aggregate_CPU_report','CPU3_missing_legacy_field_repaired_at_all_three_sites':True,
  'CPU3_actual_two_aggregate_proofs_source_bound':True,'CPU3_report_has_no_legacy_all_children_field':True,'controls':rec(O/'actual_metadata_report_boundary_controls_v1.json'),
  'reviewed_native_contracts':['Original source BPE64250 plus parked qualifiedLoRA62250; bootstrap requires updates0/base clocks64250/2000/62250; no diagnostic seed','Original8711 leaves initialize new512; original threeAdam transfer; fresh appended clock0','Every full row/target/context/order checked; protected replay role and original teacher preserved','Exactly20 plan updates, coordinator/Adam/sampler frontier20 required; no skipped or independent reshuffled rows','Persisted854 tensors/215Adam states against PRE-save all207/threeAdam/all4RNG fingerprint; teacher unchanged','Native realwait0/all captured identities dead/clean joined guard before separate fresh restore','Fresh restore requires all-state fingerprint, full scientific samplers/prefix and next-draw equality; no new update','CPU64MB to native128MB parent uses SAME process OS lifetime highwater, all CPU family/shims charged; no baseline RNG restore invocation'],
  'observations':['Several inherited docstrings/error strings still say first4/four; operative numeric checks are20','Family/source/config admission remains ROOT responsibility; current review creates no execution authority','Mocked closure metadata is not actual Popen/physical liveness or scientific/native success'],
  'new_concrete_source_blockers':[],'actual_model_audio_GPU_payload_jobs':0,'production_enabled':False}
 (O/'source_findings_v1.json').write_text(json.dumps(report,indent=2)+'\n')
 g.publish(completed=True,findings=rec(O/'source_findings_v1.json'))
 print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json'),'controls':rec(O/'actual_metadata_report_boundary_controls_v1.json')}))

if __name__=='__main__':main()
