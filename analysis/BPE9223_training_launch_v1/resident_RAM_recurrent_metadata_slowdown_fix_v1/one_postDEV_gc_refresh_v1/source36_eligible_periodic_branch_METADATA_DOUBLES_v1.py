"""INACTIVE positive eligible source36 branch AST proof; no action at import.
Every worker/DB/GC/fingerprint owner below is an honest METHOD_STUB. Real
CPython/helper/capture controls are closed and are NOT rerun here.
"""
import ast,copy,hashlib,types,weakref
from pathlib import Path

def require(v,m):
 if not v:raise ValueError(m)
def execute(n,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(n)],type_ignores=[])),
  '<EXACT_ELIGIBLE_SOURCE36_AST_METADATA_DOUBLES>','exec'),scope)

def check_entire_eligible_driver_branch(driver_path,queued_pin):
 tree=ast.parse(Path(driver_path).read_text(encoding='utf-8-sig'))
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 event=next(n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test)=='not postDEV_refresh.opportunity_consumed')
 close_node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='close_private_metadata')
 transient_names=['plan','plan_pin','coordinator','writer','before','nxt','caller','ptr','proof','ck',
  'tx','entry','rows','metadata','native','source','binding','evidence','state','CPU_quiescence',
  'dev_pin','score','preservation','cpu_closed','phase_preserved','scalar_closed']
 actual_dels=[[x.id for x in n.targets if isinstance(x,ast.Name)] for n in ast.walk(event) if isinstance(n,ast.Delete)]
 require([x for group in actual_dels for x in group if x in transient_names]==transient_names,
  'Exact source-bound completed-alias deletion sequence; no speculative extra drops')
 order=[];receipts=[];captures=[];worker_refs=[];capture_marker=object();observer_marker=object()
 class ThreadDouble:
  def __init__(self):self.alive=True
  def is_alive(self):return self.alive
 class WorkerDouble:
  def __init__(self,original,fail=False):
   self.original=original;self.closed=False;self.thread=ThreadDouble();self.worker_status={'closed':False,'error':None}
   self.outstanding=None;self.held_result=None;self.admission={'METHOD_STUB':True};self.calls=[];self.fail=fail
  def close(self,*,discard_unconsumed):
   self.calls.append(discard_unconsumed);order.append('close_discard_'+str(discard_unconsumed))
   if self.fail:raise ValueError('METHOD_STUB new-generation close failure')
   self.closed=True;self.thread.alive=False;self.worker_status={'closed':True,'error':None}
   return {'joined':True,'worker_connections_closed_on_worker':True,'discarded_unconsumed_request':False}
 resolver=types.SimpleNamespace(metadata_locks=types.SimpleNamespace(handles=[1,2,3]))
 context=types.SimpleNamespace(resolver=resolver,closed=False)
 schedule=types.SimpleNamespace(phase='weak_label_continued_training',poisoned=False)
 engine=types.SimpleNamespace(schedule=schedule,context=context)
 optimizer=types.SimpleNamespace(updates=4000,poisoned=False)
 queued=({'METHOD_STUB_unchanged_plan':True,'draws':0},copy.deepcopy(queued_pin))
 queued_ids=tuple(map(id,(queued,queued[0],queued[1])))
 guard=lambda:None
 def admit(tag,n):order.append('original_admit_'+str(n));return {'METHOD_STUB':True,'tag':tag,'bytes':n}
 def write_json(path,value):receipts.append({'name':Path(path).name,'value':copy.deepcopy(value)})
 def capture(**kw):
  require(set(kw)=={'optimizer','torch','guard','schedule','context','journal','lifecycle','periodic','queued','latest','completed'},
   'Exact actual source capture callable keyword contract')
  require(all(kw[k] is namespace[k] for k in ('optimizer','torch','guard','journal','lifecycle','periodic','queued','latest','completed'))
   and kw['schedule'] is schedule and kw['context'] is context,'Exact scientific owners/queued objects passed through without draw')
  captures.append({k:id(v) for k,v in kw.items()});order.append('capture_'+str(len(captures)))
  return {'METHOD_STUB_full854_no_payload':True,'marker':id(capture_marker)}
 def compare(a,b):require(a==b,'Metadata capture comparison double');order.append('compare')
 class EventOwnerDouble:
  opportunity_consumed=False;expected_update=4000;completed=False
  def execute_real(self,barrier,*,capture,compare,guard,admit,original_collect_observer,emit):
   require(all(v is True for k,v in barrier.items() if k not in ('kind','updates','capture_contract')),
    'Every actual wired idle-barrier assertion is true under labelled doubles')
   require(barrier['updates']==4000 and barrier['capture_contract']=='original_bounded_live_fingerprint_and_identity_manifest_v1',
    'Actual source semantic capture contract')
   require(original_collect_observer is observer_marker,'Exact original observational owner preserved')
   require(worker_refs[0]() is None and namespace['prefetch'] is None,'Old worker weakref genuinely dead before event')
   require(all(k not in namespace for k in transient_names),'All exact completed aliases dropped before capture/event')
   order.append('event_begin');admission=admit(16777216)
   require(admission['required_commit_bytes']==4535212672 and admission['required_physical_bytes']==1785212672
    and admission['host_credit_bytes']==0,'Exact extra16MiB atop original floors/225/fullprefetch')
   before=capture();order.append('METHOD_STUB_unfreeze_collect_freeze_NO_GC_CALL');after=capture();compare(before,after)
   self.opportunity_consumed=True;self.completed=True;emit({'passed':True,'METHOD_STUB_eligible_event':True});order.append('event_complete')
 def fresh_worker(actual_context,*,guard,admit):
  require(actual_context is context and namespace['postDEV_refresh'].completed and order[-1]=='event_complete',
   'Original fresh scalar constructed only AFTER eligible event succeeds')
  require(worker_refs[0]() is None,'No old/new scalar overlap')
  order.append('new_worker_constructed');w=WorkerDouble(resolver);worker_refs.append(weakref.ref(w));return w
 namespace={'require':require,'postDEV_refresh':EventOwnerDouble(),'optimizer':optimizer,'engine':engine,
  'validation_stop':False,'budget_stop':False,'stream':None,'queued':queued,'last_checkpoint':4000,
  'latest':{'full_state_verified':True,'committed_updates':4000},'prefetch_generation':1,'prefetch_close_attempted_generation':None,
  'future_ticket':None,'prefetch_policy':{'source':queued_pin},'evaluated':{4000},
  'monitor':types.SimpleNamespace(thread=ThreadDouble(),error=None,_observe=lambda:{'system_memory':{
   'commit_available_bytes':4535212672,'physical_available_bytes':1785212672}}),
  'guard':guard,'admit':admit,'E_reserve':lambda n:order.append('E_reserve_'+str(n)),
  'record':lambda path:copy.deepcopy(queued_pin),'capture_idle_state':capture,'compare_idle_state':compare,
  'original_collect_observer':observer_marker,'PrivateScalarPrefetch':fresh_worker,'torch':None,
  'journal':object(),'lifecycle':object(),'periodic':object(),'completed':1,
  'invocation':Path('E:/gpu_training_scratchpad/EXPLICIT_METADATA_DOUBLE_NO_WRITE'),
  'write_json':write_json,'read_json':lambda p:{'actual_phase_private':{'joined':True,'error':None},
   'all207_threeAdam_all4RNG_unchanged':True,'both_cursors_journal_and_nextdraw_unchanged':True}}
 namespace['prefetch']=WorkerDouble(resolver);worker_refs.append(weakref.ref(namespace['prefetch']))
 for name in transient_names:namespace.setdefault(name,object())
 namespace['CPU_quiescence']={'CPU_actual0':{'actual_Popen_wait_code':0,'transport_threads_joined':True,'pipe_handles_closed':True,
  'packet_credit':{'all_owned_kernel_handles_closed':True},'packet_credit_owner':{'all_owned_kernel_handles_closed':True}}}
 namespace['preservation']={'path':'EXPLICIT_METADATA_DOUBLE_NO_READ'}
 execute(event,namespace)
 require(worker_refs[0]() is None and worker_refs[1]() is namespace['prefetch'],'Old dropped, exact new instance owned')
 require(namespace['prefetch_generation']==2 and namespace['prefetch_close_attempted_generation']==1,'Per-instance close generations advance independently')
 require(queued_ids==tuple(map(id,(namespace['queued'],queued[0],queued[1]))) and queued[0]['draws']==0,'SAME queued tuple/plan/pin, no regeneration/draw')
 require(len(captures)==2 and captures[0]==captures[1],'Exact same before/after capture arguments')
 require([x['name'] for x in receipts]==['first_periodic_refresh_scalar_worker_closure.json','actual_one_postDEV_gc_refresh.json',
  'private_metadata_worker_after_first_periodic_refresh_admission.json'],'Actual source closure/event/new admission write order')
 require(not any(name in namespace for name in transient_names),'Source alias deletions persist after recreated worker')
 # Compile ACTUAL nested close function inside a metadata harness so nonlocal
 # generation ownership is real Python closure behavior, not a replacement.
 harness=ast.parse('def make(prefetch,prefetch_generation,prefetch_close_attempted_generation,prefetch_policy,write_json,invocation):\n return close_private_metadata\n').body[0]
 harness.body.insert(0,copy.deepcopy(close_node));outer={};execute(harness,outer)
 closer=outer['make'](namespace['prefetch'],2,1,namespace['prefetch_policy'],write_json,namespace['invocation'])
 closer();closer();require(namespace['prefetch'].calls==[True],'Actual final close affects NEW instance exactly once')
 bad=WorkerDouble(resolver,fail=True);error_closer=outer['make'](bad,3,2,namespace['prefetch_policy'],write_json,namespace['invocation'])
 try:error_closer()
 except ValueError as error:require(str(error)=='METHOD_STUB new-generation close failure','Original close error propagation');error.__traceback__=None
 else:raise AssertionError('Actual error-close negative unexpectedly succeeded')
 error_closer();require(bad.calls==[True] and receipts[-1]['name']=='private_metadata_worker_close_failure.json',
  'Actual error closure records failure once for new generation; no old worker reclose')
 return {'passed':True,'event_AST_sha256':hashlib.sha256(ast.dump(event,include_attributes=False).encode()).hexdigest(),
  'close_AST_sha256':hashlib.sha256(ast.dump(close_node,include_attributes=False).encode()).hexdigest(),
  'exact_source_deleted_transient_aliases':transient_names,'actual_order_under_METHOD_STUB_owners':order,
  'old_weakref_dead_before_event':True,'new_generation_final_and_error_close_exactly_once':True,
  'queued_tuple_plan_pin_identity_exact':True,'capture_argument_identity_before_after_exact':True,
  'production_DB_threads_GC_GPU_or_scientific_state_exercised':False}
