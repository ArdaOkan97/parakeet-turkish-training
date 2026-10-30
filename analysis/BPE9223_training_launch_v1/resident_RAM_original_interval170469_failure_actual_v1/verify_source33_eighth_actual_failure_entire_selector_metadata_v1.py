"""INACTIVE source33 exact metadata-only regression; separate ROOT CPU scope needed.
Executes genuine typed/policy validators and complete frozen AST metadata routes.
Prefix view, prospective journal binding and structural execution review are explicit
in-memory doubles. No restore, journal open/copy, model, audio, NumPy or GPU.
"""
import argparse,ast,copy,hashlib,json,os,shutil,sys,time,types
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).resolve().parent;sys.path[:0]=[str(S),str(R),str(F)];BEGIN=time.monotonic()
from resource_watchdog_v2 import sample_memory

def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def identity(pid=None):
 p=psutil.Process(os.getpid()if pid is None else pid);return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def guard():
 v=sample_memory(os.getpid(),'source33_actual170469_saved170400_metadata_selector');m=v['system_memory']
 require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<536870912
  and time.monotonic()-BEGIN<180 and m['commit_available_bytes']>=2268435456
  and m['physical_available_bytes']>=1768435456 and shutil.disk_usage('E:/').free>=100000000000,
  'Metadata512MiB/180s/FULLprefetch/originalfloors/E100GB')
 return v
def expect(kind,call):
 try:call()
 except kind as error:return {'type':type(error).__name__,'message':str(error)}
 raise AssertionError('Expected strict rejection '+str(kind))
def execute(node,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),
  '<EXACT_FROZEN_SOURCE_AST_METADATA_ONLY>','exec'),scope)
def expression(node,scope):
 return eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))),
  '<EXACT_FROZEN_COMPLETE_METADATA_DICTIONARY>','eval'),scope)

class PrefixViewDouble:
 """No-read metadata double, never a real restore or prefix verification."""
 def __init__(self,saved,guard):self.saved=copy.deepcopy(saved)
 def binding(self):return copy.deepcopy(self.saved)
class JournalBindingDouble:
 """Prospective new24 binding only; no writer/file/copy/native authority."""
 def __init__(self,value):self.value=copy.deepcopy(value)
 def binding(self):return copy.deepcopy(self.value)
class ForbiddenNamespace:
 def writer(self,*a,**k):raise AssertionError('Metadata proof MUST NOT create a production writer')

def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
  and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
  and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
  'Exact process-local E metadata CPU-only runtime')
 review=read(a.ROOT_review)
 require(review['accepted']is True and review['reviewer']=='/root'
  and review['scope']=='ASR18_source33_eighth_original_failure_entire_selector_metadata_CPU_v1'
  and review['fixture_source']==pin(__file__),'Separate exact finite metadata ROOT scope')
 for item in review['input_pins']:require(pin(item['path'])==item,'Frozen canonical metadata input before proof')
 output=F/'source33_typed_eighth_selector_actual_01';require(not output.exists(),'One fresh metadata output');output.mkdir()
 entry={'parent':identity(),'launcher':identity(os.getppid()),'ROOT_scope':pin(a.ROOT_review)};write(output/'entry.json',entry)
 results={};initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368,
  'Full prospective512MiB plus FULLprefetch/originalfloors')
 try:
  configuration=read(review['source_configuration']['path']);closure=review['eighth_original_resume']
  pointer=review['initial_checkpoint'];branch=Path(review['branch'])
  import run_ASR18_BPE9223_resident_RAM_training_v33 as driver
  import fastmss_resident_feature_stream_v13 as CPU
  import asr18_eighth_original_interval_checkpoint_resume_v1 as lane
  require('torch'not in sys.modules and 'numpy'not in sys.modules,'No Torch/NumPy/model imports')
  driver.checked_native_commit_pressure_policy(configuration)
  CPU.checked_resource_profile(configuration);CPU.checked_packet_credit_policy(configuration)
  lane.checked_eighth_original_policy(configuration)
  actual=lane.checked_eighth_original_resume(closure,base=configuration,folder=branch,pointer_pin=pointer)
  results['genuine_actual_failure_and_completed_policy_validators']={'passed':True,
   'actual_optimizer_updates':actual['failure']['actual_optimizer_updates'],'saved_optimizer_updates':actual['saved_optimizer_updates'],
   'actual_outer_TOOL_exit':actual['actual_outer_TOOL_exit'],'separate_OS_exits':actual['separate_native_CPU_wrapper_numeric_OS_exits'],
   'preserved69_COPY_bytes':actual['preserved_unsaved_tail']['bytes'],'original23_only_STAT_not_read':True,
   'original_cpu11_historical_provenance':read(actual['evidence']['CPU_ready']['path'])['source']['protocol']}
  source=Path(driver.__file__);tree=ast.parse(source.read_text(encoding='utf-8-sig'))
  mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
  names=['prior_process','fresh_signature_resume','canonical_archive_resume','lookahead_resume','cushion225_resume',
   'postrestore_selector_resume','packet_credit_resume','native_headroom_resume','abnormal_resume','source_transition_resume',
   'original_interval_resume','DEV_frontend_resume','DEV_admission_resume','periodic_quiescence_resume']
  args=types.SimpleNamespace(**{k:None for k in names},eighth_original_resume=closure['path'],invocation=24,start_invocation=24)
  assignment=next(n for n in mainnode.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='eighth_interval_failure'for t in n.targets))
  entryif=next(n for n in mainnode.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
   and isinstance(n.test.left,ast.Name)and n.test.left.id=='eighth_interval_failure')
  scope={'a':args,'base':configuration,'folder':branch,'pointer_pin':pointer,'record':driver.record,'require':require,
   'checked_eighth_original_resume':lane.checked_eighth_original_resume}
  execute(assignment,scope);execute(entryif,scope);require(scope['eighth_interval_failure']==actual,'Actual complete entry binding')
  intervalassignment=next(n for n in mainnode.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='interval_failure'for t in n.targets))
  scope['checked_original_interval_resume']=driver.checked_original_interval_resume
  execute(intervalassignment,scope);require(scope['interval_failure']==actual,'Actual shared interval branch selects new typed closure')
  results['actual_driver_entry_and_shared_lane_assignment']={'passed':True,
   'assignment_AST':ast.dump(assignment,include_attributes=False),'entry_AST':ast.dump(entryif,include_attributes=False),
   'shared_assignment_AST':ast.dump(intervalassignment,include_attributes=False)}
  wrong=copy.deepcopy(args);wrong.invocation=23
  results['wrong_initial_invocation_rejected']=expect(ValueError,lambda:execute(entryif,{**scope,'a':wrong}))
  wrong=copy.deepcopy(args);wrong.fresh_signature_resume='historical_unselected'
  results['mixed_historical_lane_rejected']=expect(ValueError,lambda:execute(entryif,{**scope,'a':wrong}))
  bad=copy.deepcopy(pointer);bad['sha256']='0'*64
  results['wrong_checkpoint_rejected']=expect(ValueError,lambda:lane.checked_eighth_original_resume(closure,base=configuration,folder=branch,pointer_pin=bad))
  bad=copy.deepcopy(closure);bad['sha256']='0'*64
  results['stale_closure_pin_rejected']=expect(ValueError,lambda:lane.checked_eighth_original_resume(bad,base=configuration,folder=branch,pointer_pin=pointer))
  for name,mutate in [
   ('None_completed_CPU',lambda v:v['original_clip_geometry_overlay'].__setitem__('ROOT_review',None)),
   ('failed_overlay8',lambda v:v['original_clip_geometry_overlay'].__setitem__('source',{'path':str(S/'fastmss_original_clip_geometry_overlay_v8.py')})),
   ('stale_CPU11_protocol',lambda v:v['resident_RAM_resource_scheduling_policy'].__setitem__('CPU_protocol_source',read(lane.BASE)['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'])),
   ('wrong_MAX2',lambda v:v['resident_RAM_packet_credit_policy'].__setitem__('maximum_logical_packets',1))]:
   bad=copy.deepcopy(configuration);mutate(bad)
   results['policy_reject_'+name]=expect(ValueError,lambda bad=bad:lane.checked_eighth_original_policy(bad))
  saved=read(pointer['path'])['journal_prefix']
  factory=next(n for n in ast.walk(mainnode)if isinstance(n,ast.FunctionDef)and n.name=='journal_factory')
  factoryscope={**scope,'source_journal':saved,'eighth_interval_failure':actual,'interval_failure':actual,
   'transition':None,'signature_transition':None,'canonical_transition':None,'lookahead_transition':None,
   'periodic_quiescence_failure':None,'DEV_admission_failure':None,'frontend_failure':None,'abnormal':None,
   'SavedJournalPrefixView':PrefixViewDouble,'guard':lambda:None,'source_namespace':ForbiddenNamespace()}
  execute(factory,factoryscope);view,before=factoryscope['journal_factory'](saved)
  require(view.binding()==saved and before=={'kind':'ASR18_readonly_saved170400_prefix_after_eighth_original_interval_failure_v1',
   'failure':closure,'original_tail_modified':False,'no_tail_replay':True},'ENTIRE real pre-restore dictionary binds actual closure')
  results['entire_real_pre_restore_factory_dictionary']={'metadata':before,'explicit_no_read_prefix_view_double':True,
   'factory_AST':ast.dump(factory,include_attributes=False)}
  altered={**saved,'offset':saved['offset']-1}
  results['wrong_saved_prefix_factory_rejected']=expect(ValueError,lambda:factoryscope['journal_factory'](altered))
  postcalls=[n for n in ast.walk(mainnode)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)
   and n.func.id=='write_json'and n.args and 'actual_journal_branch.json'in ast.unparse(n.args[0])]
  require(len(postcalls)==1 and isinstance(postcalls[0].args[1],ast.Dict),'One full actual postrestore metadata dictionary')
  newbinding={'path':str(branch/'journals/commits_000024.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
  postscope={**factoryscope,'preserved_tail':actual,'journal':JournalBindingDouble(newbinding)}
  after=expression(postcalls[0].args[1],postscope)
  require(after=={'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1','predecessor':closure,
   'saved_prefix':saved,'preserved_original':actual['whole_journal_including_preserved_tail'],
   'new_active_journal':newbinding,'historical_unsaved_tail_updates':list(range(170401,170470)),
   'native_updates_or_draws_in_copy':0},'ENTIRE real postrestore dict uses actual closure and STAT-only original representation')
  require('sha256'not in after['preserved_original'],'STAT-only oldwhole is not a fabricated wholehash')
  results['entire_real_post_restore_dictionary']={'metadata':after,'explicit_no_copy_journal_binding_double':True,
   'dictionary_AST':ast.dump(postcalls[0].args[1],include_attributes=False)}
  postcheck=next(n for n in ast.walk(mainnode)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
   and isinstance(n.value.func,ast.Name)and n.value.func.id=='require'
   and any(isinstance(x,ast.Constant)and x.value=='Old tail byteexact after prefix copy'for x in n.value.args))
  execute(postcheck,{**postscope,'check_original_closed_journal_stat':driver.check_original_closed_journal_stat})
  results['actual_postcopy_STAT_only_condition_executed_without_copy']={'passed':True,'AST':ast.dump(postcheck,include_attributes=False)}
  missing=copy.deepcopy(args);missing.eighth_original_resume=None
  results['missing_new_selector_complete_postdict_rejected_METHOD_STUB']=expect(TypeError,lambda:expression(postcalls[0].args[1],{**postscope,'a':missing}))
  wrong=copy.deepcopy(args);wrong.eighth_original_resume='E:/gpu_training_scratchpad/missing_unselected_closure.json'
  results['wrong_selector_complete_postdict_rejected_METHOD_STUB']=expect(FileNotFoundError,lambda:expression(postcalls[0].args[1],{**postscope,'a':wrong}))
  wrapper=Path(review['wrapper_source']['path']);wtree=ast.parse(wrapper.read_text(encoding='utf-8-sig'))
  wm=next(n for n in wtree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
  route=next(n for n in wm.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
   and isinstance(n.test.left,ast.Attribute)and n.test.left.attr=='prior_process')
  # Explicit in-memory structural review double; never serialized as ROOT approval.
  structural={'eighth_original_resume':closure,'eighth_original_transition_validator':review['eighth_original_transition_validator'],
   'original_clip_geometry_overlay':configuration['original_clip_geometry_overlay'],
   'packet_credit_policy':configuration['resident_RAM_packet_credit_policy'],'initial_owner_invocation':24}
  ws={'a':args,'review':structural,'configuration':configuration,'record':driver.record,'S':S,'Path':Path,'require':require}
  execute(route,ws)
  wrong=copy.deepcopy(args);wrong.start_invocation=23
  results['wrapper_wrong_invocation_rejected']=expect(ValueError,lambda:execute(route,{**ws,'a':wrong}))
  results['wrapper_missing_new_lane_rejected']=expect(ValueError,lambda:execute(route,{**ws,'a':missing}))
  wrong=copy.deepcopy(args);wrong.prior_process='not_a_real_normal0'
  results['wrapper_normal0_mix_rejected']=expect(ValueError,lambda:execute(route,{**ws,'a':wrong}))
  loop=next(n for n in wm.body if isinstance(n,ast.While))
  forward=next(n for n in loop.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
   and isinstance(n.test.left,ast.Name)and n.test.left.id=='prior')
  fs={'a':args,'prior':None,'invocation':24,'command':[]};execute(forward,fs)
  require(fs['command']==['--eighth-original-resume',closure['path']],'Actual wrapper complete initial forwarding binds new lane exactly once')
  results['entire_actual_wrapper_entry_and_forwarding']={'passed':True,'forwarded':fs['command'],
   'structural_review_explicit_double':True,'entry_AST':ast.dump(route,include_attributes=False),'forwarding_AST':ast.dump(forward,include_attributes=False)}
  require(not(branch/'invocations/owner_000024').exists()and not(branch/'journals/commits_000024.jsonl').exists()
   and not(F.parent/'resident_RAM_training_actual_25').exists(),'No production invocation/journal/launch output created')
  for item in review['input_pins']:require(pin(item['path'])==item,'All frozen metadata inputs unchanged after proof')
  require('torch'not in sys.modules and 'numpy'not in sys.modules,'No NumPy/Torch/model activity')
  report={'kind':'ASR18_source33_eighth_original_failure_entire_selector_metadata_CPU_actual_v1','passed':True,
   'results':results,'entry':entry,'initial_guard':initial,'final_guard':guard(),'fixture_source':pin(__file__),
   'driver_source':pin(source),'wrapper_source':pin(wrapper),'source_configuration':review['source_configuration'],
   'actual_failure_closure':closure,'input_pins':review['input_pins'],
   'real_policy_failure_and_complete_selector_dicts_wrapper_routes_executed':True,
   'prefix_view_journal_binding_structural_review_are_explicit_in_memory_doubles':True,
   'adversarial_inconsistent_postdict_scope_cases_are_METHOD_STUBS':True,
   'actual_original_journal_read_or_hash':False,'preserved_tail_COPY_only_bytes':135633,
   'actual_restore_copy_sampler_native_audio_GPU_kernel_credit_operations':False,'production_changed':False}
  write(output/'report.json',report);print(json.dumps({'report':pin(output/'report.json')}),flush=True)
 except BaseException as error:
  import traceback
  write(output/'actual_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'entry':entry,'partial_results':results});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
