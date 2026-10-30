"""INACTIVE separately scoped metadata proof; NEVER run as preparation.
Actual source36 validator/policy and complete factory/postrestore/wrapper ASTs.
Future stop/deaths/journal/file/fingerprint authority are explicit IN-MEMORY
STRUCTURAL DOUBLES. They are never written as actual acceptance or final config.
No GC operation, model/audio/tensor read, journal open/copy, sampler draw or stop.
"""
import argparse,ast,copy,hashlib,json,os,shutil,sys,time,types
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).resolve().parent;sys.path[:0]=[str(S),str(R),str(F)];BEGIN=time.monotonic()
from resource_watchdog_v2 import sample_memory

def require(value,message):
 if not value:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();raw=p.read_bytes();require(len(raw)<=8*1024**2,'Compact/source input only')
 return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,value):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:
  json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
def guard():
 sample=sample_memory(os.getpid(),'source36_inactive_selector_metadata_proof');m=sample['system_memory']
 require(sample['paired_sampling_within_one_second'] and sample['worker_private_memory']['private_bytes']<536870912
  and time.monotonic()-BEGIN<100 and m['commit_available_bytes']>=2268435456
  and m['physical_available_bytes']>=1768435456 and shutil.disk_usage('E:/').free>=100000000000,
  'Metadata100s/512MiB/FULLprefetch/originalfloors/E100GB')
 return sample
def execute(node,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),
  '<EXACT_FROZEN_SOURCE_AST_METADATA_ONLY>','exec'),scope)
def expression(node,scope):
 return eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))),
  '<EXACT_FROZEN_COMPLETE_METADATA_DICTIONARY>','eval'),scope)
def expect(kind,call):
 try:call()
 except kind as error:return {'rejected':True,'type':type(error).__name__,'message':str(error)}
 raise AssertionError('Strict negative unexpectedly passed: '+str(kind))
def assignment(nodes,name):
 matches=[n for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
 require(len(matches)==1,'Exactly one real source assignment '+name);return matches[0]

class PrefixViewDouble:
 """No original byte access, hashing, restore or draw."""
 def __init__(self,saved,guard):self.saved=copy.deepcopy(saved)
 def binding(self):return copy.deepcopy(self.saved)
class JournalBindingDouble:
 """A prospective binding only. No file/writer/native authority."""
 def __init__(self,value):self.value=copy.deepcopy(value)
 def binding(self):return copy.deepcopy(self.value)
class ForbiddenNamespace:
 def writer(self,*a,**k):raise AssertionError('No production namespace writer in metadata proof')

class VirtualPath:
 """Explicit virtual journal STAT/tail byte operations; other I/O forbidden."""
 def __init__(self,graph,value):self.graph=graph;self.value=Path(os.fspath(value))
 def __fspath__(self):return str(self.value)
 def __str__(self):return str(self.value)
 def __truediv__(self,other):return VirtualPath(self.graph,self.value/other)
 def __eq__(self,other):
  try:return self.value==Path(os.fspath(other))
  except TypeError:return False
 def resolve(self):return VirtualPath(self.graph,self.value.resolve())
 def is_relative_to(self,other):return self.value.is_relative_to(Path(os.fspath(other)))
 def stat(self):
  require(str(self.value)==self.graph.journal,'Only explicit virtual old-journal STAT is permitted')
  return types.SimpleNamespace(st_size=self.graph.stat['bytes'],st_mtime_ns=self.graph.stat['mtime_ns'])
 def read_bytes(self):
  require(str(self.value)==self.graph.tail_path,'Only explicit virtual preserved suffix is permitted')
  return self.graph.tail_bytes

class StructuralClosureGraph:
 """Future-stop METHOD_STUB. No actual closure, stop, checkpoint or death proof.
 Real existing source/proof/approval records still use original bytes and pins.
 Virtual records are held only in memory under the fresh fixture output path.
 """
 def __init__(self,review,configuration,output):
  self.review=review;self.base=copy.deepcopy(configuration);self.values={};self.raw={}
  self.allowed={str(Path(x['path']).resolve()) for x in review['input_pins']}
  self.branch=output/'EXPLICIT_IN_MEMORY_ONLY_branch';self.K=177000
  self.journal=str(self.branch/'journals/commits_000024.jsonl')
  self.tail_path=str(output/'EXPLICIT_IN_MEMORY_ONLY_preserved_tail.jsonl')
  events=[{'optimizer_updates':u,'native':{'update':u},'prepared_evidence':{'actual_positive_native_update':u}}
   for u in range(self.K+1,self.K+3)]
  self.tail_bytes=b''.join(json.dumps(x,sort_keys=True,separators=(',',':')).encode()+b'\n' for x in events)
  self.raw[self.tail_path]=self.tail_bytes
  self.saved={'path':self.journal,'offset':1000,'sha256':'1'*64}
  self.stat={'path':self.journal,'bytes':1000+len(self.tail_bytes),'mtime_ns':123456789,'whole_content_hashed':False}
  self.pointer_path=str(self.branch/'checkpoints/update_000177000/pointer.json')
  self.pointer=self.put(self.pointer_path,{'committed_updates':self.K,'journal_prefix':self.saved})
  persistence=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_persistence.json'),{'passed':True,'optimizer_updates':self.K,
   'named_tensors_verified':854,'weights_verified':207,'Adam_states_verified':215,
   'independent_pre_save_state_sha256':'2'*64,'saved_state_sha256':'2'*64,
   'saved_semantic_state_verified':{'passed':True},'journal_prefix':self.saved})
  self.put(str(self.branch/'latest_checkpoint.json'),{'pointer':self.pointer})
  self.put(str(self.branch/'active_journal.json'),{'journal_prefix':self.saved})
  runtime=copy.deepcopy(read(review['transition_intent']['path'])['expected_runtime_identities'])
  approval=pin(F.parent/'ROOT_direct_human_then_fix_authorization_v1.json')
  old_driver=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v33.py')
  old_wrapper=pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py')
  old_cpu=pin(S/'fastmss_resident_feature_stream_v13.py')
  intent=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_intent.json'),{'owned_session':65067,'old_owner_invocation':24,
   'human_approval':approval,'expected_runtime_identities':runtime})
  pre_stop=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_ROOT_pre_stop_review.json'),{'accepted':True,'reviewer':'/root',
   'scope':'ASR18_one_postDEV_refresh_checkpoint_stop_v1','transition_intent':intent,'human_approval':approval,'old_driver_source':old_driver})
  stop=self.put(str(self.branch/'invocations/owner_000024/stop.json'),{'kind':'ASR18_reviewed_one_postDEV_refresh_checkpoint_cooperative_stop_v1',
   'transition_intent':intent,'ROOT_pre_stop_review':pre_stop,'observed_saved_optimizer_updates':self.K,'observed_native_updates':self.K})
  failure=self.put(str(self.branch/'invocations/owner_000024/actual_failure.json'),{'actual_optimizer_updates':self.K+2,
   'last_checkpoint':self.K,'error':"ValueError('Actual owner cooperative stop')",'traceback':''})
  old_configuration=self.put(str(self.branch/'invocations/owner_000024/configuration.json'),{'EXPLICIT_STRUCTURAL_DOUBLE':True})
  boot=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_boot.json'),{'source':old_driver})
  ready=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_CPU_ready.json'),{'source':{'protocol':old_cpu}})
  tool=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_TOOL.json'),{'owned_session':65067,'actual_TOOL_exit_code':1,'normal0_claimed':False})
  monitor=self.put(str(output/'EXPLICIT_IN_MEMORY_ONLY_monitor.json'),{'joined':True,'clean':True,'error':None})
  self.evidence={'pointer':self.pointer,'persistence':persistence,'old_driver':old_driver,'old_wrapper':old_wrapper,
   'old_CPU':old_cpu,'old_configuration':old_configuration,'old_boot':boot,'old_CPU_ready':ready,'actual_failure':failure,
   'stop':stop,'transition_intent':intent,'ROOT_pre_stop_review':pre_stop,'actual_TOOL':tool,'monitor_completion':monitor}
  self.closure_path=str(output/'EXPLICIT_IN_MEMORY_ONLY_actual_transition.json')
  self.closure={'kind':'ASR18_actual_source33_checkpoint_one_postDEV_gc_refresh_transition_v1','owned_session':65067,
   'old_owner_invocation':24,'new_owner_invocation':25,'actual_outer_TOOL_exit_code':1,'normal0_claimed':False,
   'intentional_cooperative_stop_observed':True,'human_approval':approval,'evidence':self.evidence,
   'all_five_runtime_identities':runtime,'last_fully_saved_optimizer_updates':self.K,'actual_optimizer_updates':self.K+2,
   'published_journal_optimizer_updates':self.K+2,'saved_journal_prefix':self.saved,
   'closed_old_journal_STAT':self.stat,'preserved_unsaved_positive_tail':self.record(self.tail_path)}
  self.changed=[]
 def Path(self,value):return VirtualPath(self,value)
 def put(self,path,value):self.values[str(path)]=copy.deepcopy(value);return self.record(path)
 def read(self,path):
  key=str(path)
  if key in self.values:return copy.deepcopy(self.values[key])
  require(str(Path(key).resolve()) in self.allowed,'Only pinned real metadata read or explicit in-memory double: '+key)
  return read(key)
 def record(self,path):
  if path is None:raise TypeError('record(None) is invalid; predecessor selector must choose a real lane')
  key=str(path)
  if key in self.values:raw=json.dumps(self.values[key],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
  elif key in self.raw:raw=self.raw[key]
  else:
   require(str(Path(key).resolve()) in self.allowed,'Only pinned actual metadata/source record or virtual double: '+key)
   return pin(key)
  return {'path':key,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 def exact_dead(self,who):
  require(any(who==x for x in self.closure['all_five_runtime_identities']),'Death DOUBLE only for explicit expected old cohort')
  return True
 def finalize(self,lane):
  failure=self.values[self.evidence['actual_failure']['path']]
  self.closure['optimizer_vs_journal_frontier']=lane.validate_intentional_optimizer_journal_gap(failure,self.closure['published_journal_optimizer_updates'])
  for name in self.changed:
   self.closure['evidence'][name]=self.record(self.evidence[name]['path'])
  self.values[self.closure_path]=copy.deepcopy(self.closure)
  closure_pin=self.record(self.closure_path)
  self.base['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition']=closure_pin
  return closure_pin

def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
  and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
  and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Exact E metadata-only runtime')
 review=read(a.ROOT_review)
 require(review['accepted'] is True and review['reviewer']=='/root'
  and review['scope']=='ASR18_source36_pre_stop_entire_selector_metadata_CPU_v1'
  and review['fixture_source']==pin(__file__),'Separate exact CPU scope; preparation never authorizes execution')
 for item in review['input_pins']:require(pin(item['path'])==item,'All exact frozen source/metadata inputs')
 output=F/'source36_pre_stop_typed_selector_actual_01';require(not output.exists(),'ONE fresh proof namespace')
 initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368,'Full225+prospective512+FULLprefetch admission')
 require(all(alive(w) for w in review['protected_runtime_identities']),'Production cohort unchanged before isolated proof')
 output.mkdir();entry={'observer':identity(psutil.Process()),'launcher':identity(psutil.Process().parent()),'ROOT_scope':pin(a.ROOT_review)}
 write(output/'entry.json',entry);results={};originals=None
 try:
  configuration=read(review['source_configuration']['path'])
  require(configuration['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'] is None
   and configuration['one_postDEV_gc_refresh_policy']['completed_CPU_review']==review['actual_CPU_review'],
   'Candidate has genuine completed CPU gate but no actual stop/final production slot')
  import asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 as lane
  originals=(lane.record,lane.read_json,lane.Path,lane.exact_dead)
  # Actual source policy is invoked with REAL pins/review; no GC/capture/event call.
  lane.checked_one_postDEV_refresh_policy(configuration)
  require(lane.legacy_source33_configuration(configuration)==read(lane.SOURCE33_BASE),'Exact proper source33 inverse after only two fields')
  results['real_completed_CPU_policy_and_proper_base']={'passed':True,'CPU_review':review['actual_CPU_review'],'production_admitted':False}
  def validate(mutator=None):
   g=StructuralClosureGraph(review,configuration,output)
   if mutator:mutator(g)
   cp=g.finalize(lane)
   lane.record=g.record;lane.read_json=g.read;lane.Path=g.Path;lane.exact_dead=g.exact_dead
   try:actual=lane.checked_one_postDEV_refresh_checkpoint_resume(cp,base=g.base,folder=g.branch,pointer_pin=g.pointer)
   finally:lane.record,lane.read_json,lane.Path,lane.exact_dead=originals
   return g,cp,actual
  graph,closure,actual=validate()
  results['actual_full_validator_on_explicit_in_memory_future_stop']={'passed':True,'no_actual_stop_death_checkpoint_journal_claim':True,
   'virtual_saved':graph.K,'virtual_committed_tail_count':len(actual['preserved_tail_positive_updates'])}
  def validated_argument_negative(cp=closure,pointer=graph.pointer):
   lane.record=graph.record;lane.read_json=graph.read;lane.Path=graph.Path;lane.exact_dead=graph.exact_dead
   try:return lane.checked_one_postDEV_refresh_checkpoint_resume(cp,base=graph.base,folder=graph.branch,pointer_pin=pointer)
   finally:lane.record,lane.read_json,lane.Path,lane.exact_dead=originals
  results['stale_closure_pin']=expect(ValueError,lambda:validated_argument_negative(cp={**closure,'sha256':'0'*64}))
  results['wrong_source_checkpoint']=expect(ValueError,lambda:validated_argument_negative(pointer={**graph.pointer,'sha256':'0'*64}))
  negatives=[('wrong_owned_session',lambda g:g.closure.__setitem__('owned_session',65068)),
   ('wrong_old_invocation',lambda g:g.closure.__setitem__('old_owner_invocation',23)),
   ('normal0_alias',lambda g:g.closure.__setitem__('actual_outer_TOOL_exit_code',0)),
   ('wrong_old_cohort',lambda g:g.closure['all_five_runtime_identities'][0].__setitem__('pid',999999)),
   ('wrong_saved_prefix',lambda g:g.closure['saved_journal_prefix'].__setitem__('sha256','0'*64)),
   ('unclean_monitor',lambda g:(g.values[g.evidence['monitor_completion']['path']].__setitem__('clean',False),g.changed.append('monitor_completion'))),
   ('wrong_checkpoint_count',lambda g:(g.values[g.evidence['persistence']['path']].__setitem__('named_tensors_verified',853),g.changed.append('persistence')))]
  for name,change in negatives:results['METHOD_STUB_reject_'+name]=expect((ValueError,AssertionError),lambda change=change:validate(change))
  for name,change in [('missing_actual_stop_slot',lambda c:c.__setitem__('prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition',None)),
   ('missing_CPU_review',lambda c:c['one_postDEV_gc_refresh_policy'].__setitem__('completed_CPU_review',None)),
   ('wrong_helper',lambda c:c['one_postDEV_gc_refresh_policy']['helper_source'].__setitem__('sha256','0'*64)),
   ('wrong_proper_source33_base',lambda c:c.__setitem__('UNDECLARED_POLICY_FIELD',True))]:
   if name=='missing_actual_stop_slot':
    bad=copy.deepcopy(graph.base);change(bad)
    results[name]=expect(ValueError,lambda bad=bad:lane.checked_one_postDEV_refresh_checkpoint_resume(closure,base=bad,folder=graph.branch,pointer_pin=graph.pointer))
   else:
    bad=copy.deepcopy(configuration);change(bad)
    results[name]=expect((ValueError,TypeError,KeyError),lambda bad=bad:lane.checked_one_postDEV_refresh_policy(bad))
  source=Path(review['driver_source']['path']);tree=ast.parse(source.read_text(encoding='utf-8-sig'))
  mn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  optional=['prior_process','eighth_original_resume','fresh_signature_resume','canonical_archive_resume','lookahead_resume','cushion225_resume',
   'postrestore_selector_resume','packet_credit_resume','native_headroom_resume','abnormal_resume','source_transition_resume',
   'original_interval_resume','DEV_frontend_resume','DEV_admission_resume','periodic_quiescence_resume']
  args=types.SimpleNamespace(**{k:None for k in optional},gc_refresh_resume=closure['path'],invocation=25,start_invocation=25,
   configuration=review['source_configuration']['path'],stream_configuration=review['source_configuration']['path'],pointer=graph.pointer['path'],queue_bytes=268435456,latest_failure='EXPLICIT_NO_READ_ACK260')
  scope={'a':args,'base':graph.base,'folder':graph.branch,'pointer_pin':graph.pointer,'record':graph.record,'require':require,
   'checked_one_postDEV_refresh_checkpoint_resume':lambda *x,**kw:copy.deepcopy(actual)}
  # Assignment invokes a labelled validated-result double; full validator already ran above.
  execute(assignment(mn.body,'gc_transition'),scope)
  entry_require=next(n for n in mn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
   and isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and any(isinstance(x,ast.Constant)
    and x.value=='New GC policy enters only exact actual source33 intentional stop or genuine later normal0' for x in n.value.args))
  execute(entry_require,scope)
  entryif=next(n for n in mn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='gc_transition')
  execute(entryif,scope)
  results['driver_missing_new_lane']=expect(ValueError,lambda:execute(entry_require,{**scope,'gc_transition':None}))
  for name,attr,val in [('wrong_initial_invocation','invocation',24),('historical_lane_mix','eighth_original_resume','not_selected')]:
   wrong=copy.deepcopy(args);setattr(wrong,attr,val)
   results['entry_reject_'+name]=expect(ValueError,lambda wrong=wrong:execute(entryif,{**scope,'a':wrong}))
  scope.update(signature_transition=None,canonical_transition=None,lookahead_transition=None,interval_failure=None,abnormal=None,
   eighth_interval_failure=None,periodic_quiescence_failure=None,DEV_admission_failure=None,frontend_failure=None)
  execute(assignment(mn.body,'transition'),scope);execute(assignment(mn.body,'preserved_tail'),scope)
  require(scope['transition']==actual and scope['preserved_tail']==actual,'Both actual shared selectors choose the GC lane')
  factory=next(n for n in ast.walk(mn) if isinstance(n,ast.FunctionDef) and n.name=='journal_factory')
  fs={**scope,'source_journal':graph.saved,'SavedJournalPrefixView':PrefixViewDouble,'guard':lambda:None,'source_namespace':ForbiddenNamespace()}
  execute(factory,fs);view,before=fs['journal_factory'](graph.saved)
  require(view.binding()==graph.saved and before=={'kind':'ASR18_readonly_source_transition_saved_prefix_restore_v1',
   'actual_transition':closure,'no_tail_replay':True,'original_modified':False},'ENTIRE real pre-restore dictionary selects new predecessor')
  results['entire_real_pre_restore_factory_and_shared_selectors']={'passed':True,'metadata':before,'factory_AST_sha256':hashlib.sha256(ast.dump(factory,include_attributes=False).encode()).hexdigest()}
  results['factory_wrong_saved_prefix']=expect(ValueError,lambda:fs['journal_factory']({**graph.saved,'offset':999}))
  post=[n for n in ast.walk(mn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_json'
   and n.args and 'actual_journal_branch.json' in ast.unparse(n.args[0])]
  require(len(post)==1,'One complete actual post-restore branch dictionary')
  binding={**graph.saved,'path':str(graph.branch/'journals/commits_000025.jsonl')}
  ps={**fs,'journal':JournalBindingDouble(binding)};after=expression(post[0].args[1],ps)
  require(after=={'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1','predecessor':closure,'saved_prefix':graph.saved,
   'preserved_original':graph.stat,'new_active_journal':binding,'historical_unsaved_tail_updates':[graph.K+1,graph.K+2],
   'native_updates_or_draws_in_copy':0},'ENTIRE actual postrestore dict and STAT-only representation exact')
  require('sha256' not in after['preserved_original'],'Never fabricate old-whole SHA from STAT')
  results['entire_real_post_restore_dictionary']={'passed':True,'metadata':after,'no_actual_copy_or_journal_write':True}
  missing=copy.deepcopy(args);missing.gc_refresh_resume=None
  results['closed_record_None_selector_reproducer']=expect(TypeError,lambda:expression(post[0].args[1],{**ps,'a':missing}))
  wrong=copy.deepcopy(args);wrong.gc_refresh_resume=str(output/'UNREGISTERED_wrong_selector.json')
  results['wrong_predecessor_selector_rejected']=expect(ValueError,lambda:expression(post[0].args[1],{**ps,'a':wrong}))
  # Actual original post-copy expression, with only its STAT routine explicitly doubled.
  postcheck=next(n for n in ast.walk(mn) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
   and isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and any(isinstance(x,ast.Constant) and x.value=='Old tail byteexact after prefix copy' for x in n.value.args))
  checked=[]
  def stat_only(value):checked.append(copy.deepcopy(value));return value==graph.stat
  execute(postcheck,{**ps,'check_original_closed_journal_stat':stat_only})
  require(checked==[graph.stat],'Post-copy condition used STAT path, not whole record')
  results['actual_postcopy_condition_STAT_only_DOUBLE']={'passed':True}
  wt=ast.parse(Path(review['wrapper_source']['path']).read_text(encoding='utf-8-sig'))
  wm=next(n for n in wt.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  route=next(n for n in wm.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Attribute) and n.test.left.attr=='prior_process')
  structural={'gc_refresh_resume':closure,'gc_refresh_transition_validator':review['transition_validator'],'initial_owner_invocation':25}
  ws={'a':args,'review':structural,'configuration':graph.base,'record':graph.record,'require':require,'S':S,'Path':Path}
  execute(route,ws)
  results['wrapper_missing_lane']=expect(ValueError,lambda:execute(route,{**ws,'a':missing}))
  wrong=copy.deepcopy(args);wrong.start_invocation=24
  results['wrapper_wrong_initial_invocation']=expect(ValueError,lambda:execute(route,{**ws,'a':wrong}))
  wrong=copy.deepcopy(args);wrong.prior_process='not_a_genuine_normal0'
  results['wrapper_false_normal0']=expect(ValueError,lambda:execute(route,{**ws,'a':wrong}))
  loop=next(n for n in wm.body if isinstance(n,ast.While))
  forward=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='prior')
  fw={'a':args,'prior':None,'invocation':25,'command':[]};execute(forward,fw)
  require(fw['command']==['--gc-refresh-resume',closure['path']],'Actual entire wrapper initial forwarding')
  command_scope={'a':args,'P':P,'driver':source,'pointer':Path(graph.pointer['path']),'branch':graph.branch,'invocation':25}
  execute(assignment(loop.body,'command'),command_scope);command=command_scope['command']
  require(command[command.index('--configuration')+1]==command[command.index('--stream-configuration')+1]==review['source_configuration']['path'],
   'Actual wrapper passes SAME new config to stream and source flags')
  results['entire_wrapper_entry_forwarding_and_both_config_arguments']={'passed':True,'forwarded':fw['command'],'structural_review_is_DOUBLE':True}
  update=next(n for n in mn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
   and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='c' and n.value.func.attr=='update')
  runtime_scope={'c':copy.deepcopy(graph.base),'record':graph.record,'a':args,'folder':graph.branch,'pointer_pin':graph.pointer,'__file__':str(source)}
  execute(update,runtime_scope);c=runtime_scope['c']
  expected={'worker_driver_source':pin(source),'parent_driver_source':pin(source),'resident_RAM_stream_configuration':review['source_configuration'],
   'resident_branch_namespace':str(graph.branch),'checkpoint_interval_updates':200,'source_checkpoint':graph.pointer,'source_configuration':review['source_configuration']}
  require({k:c[k] for k in expected}==expected,'All original seven runtime binding KINDS and future values exact')
  for key in expected:
   wrong=copy.deepcopy(c);wrong[key]=None
   results['runtime7_negative_'+key]=expect(ValueError,lambda wrong=wrong:require({k:wrong[k] for k in expected}==expected,'Exact seven runtime bindings'))
  results['actual_runtime7_source_update']={'passed':True,'expected':expected,'future_source_config_is_inactive_candidate_not_final_launchable':True}
  from source36_eligible_periodic_branch_METADATA_DOUBLES_v1 import check_entire_eligible_driver_branch
  results['entire_actual_eligible_driver_event_and_per_instance_final_error_close_METHOD_STUB']=check_entire_eligible_driver_branch(
   source,pin(F/'asr18_one_postDEV_gc_refresh_v2.py'))
  for item in review['input_pins']:require(pin(item['path'])==item,'All frozen source/metadata unchanged after proof')
  require(all(alive(w) for w in review['protected_runtime_identities']),'Production family remains exact alive')
  require(not any(k in sys.modules for k in ('torch','numpy','av')),'No Torch/NumPy/AV/model imports')
  report={'kind':'ASR18_source36_pre_stop_entire_selector_metadata_actual_CPU_v1','passed':True,'results':results,
   'entry':entry,'initial_guard':initial,'final_guard':guard(),'fixture_source':pin(__file__),'input_pins':review['input_pins'],
   'genuine_completed_finite_CPU_review_reused':review['actual_CPU_review'],
   'future_stop_death_checkpoint_journal_and_structural_execution_reviews_are_EXPLICIT_IN_MEMORY_DOUBLES':True,
   'actual_candidate_gc_event_or_capture_or_restore_copy_sampler_native_audio_GPU_operation':False,
   'production_stop_or_final_config_or_authority_write':False,'actual_original_journal_or_tensor_read':False,
   'scientific_full854_and_real_DB_lifecycle_not_proved':True,'human_speed_fix_open':True}
  write(output/'report.json',report);print(json.dumps({'report':pin(output/'report.json'),'results':len(results)}),flush=True)
 except BaseException as error:
  import traceback
  write(output/'actual_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'entry':entry,'partial_results':results});raise
 finally:
  if originals is not None:lane.record,lane.read_json,lane.Path,lane.exact_dead=originals
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
