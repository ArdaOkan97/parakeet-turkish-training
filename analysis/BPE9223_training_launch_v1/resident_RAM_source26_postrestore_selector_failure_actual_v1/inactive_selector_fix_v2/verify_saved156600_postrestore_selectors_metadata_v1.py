"""INACTIVE finite CPU metadata proof; requires a separate exact ROOT scope.

Uses the actual new typed validator and exact AST-extracted source expressions.
No main(), loader, restore, journal writer/copy, model, audio, transport or GPU.
The saved journal binding object is explicitly a metadata-only double.
Closed 225 CPU/Monitor/transport/169-tail controls are NOT re-executed.
"""
import argparse,ast,copy,json,os,shutil,sys,time,types
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parents[1];B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
D26=R/'run_ASR18_BPE9223_resident_RAM_training_v26.py';D27=R/'run_ASR18_BPE9223_resident_RAM_training_v27.py';W27=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v27.py'
began=time.monotonic();peak=0
def guard():
 global peak
 s=sample_memory(os.getpid(),'saved156600_exact_postrestore_selector_metadata');m=s['system_memory'];peak=max(peak,s['worker_private_memory']['private_bytes'])
 require(s['paired_sampling_within_one_second'] and peak<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original512MiB/180s/fullprefetch/headroom/E100GB bounds')
 return s
def expression(node,env):return eval(compile(ast.Expression(copy.deepcopy(node)),'<exact-production-AST-expression>','eval'),env)
def statements(nodes,env):exec(compile(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),'<exact-production-AST-statements>','exec'),env)
def main_node(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
def find(tree,predicate):
 values=[n for n in ast.walk(tree)if predicate(n)];require(len(values)==1,'One exact production AST node');return values[0]
def branch_dict(tree):return find(tree,lambda n:isinstance(n,ast.Dict)and any(isinstance(k,ast.Constant)and k.value=='predecessor'for k in n.keys))
def factory_failure(tree):
 factory=find(main_node(tree),lambda n:isinstance(n,ast.FunctionDef)and n.name=='journal_factory')
 d=find(factory,lambda n:isinstance(n,ast.Dict)and any(isinstance(k,ast.Constant)and k.value=='original_tail_modified'for k in n.keys)
  and any(isinstance(v,ast.Call)and isinstance(v.func,ast.Name)and v.func.id=='record'and 'cushion225_resume'in ast.unparse(v)for v in n.values))
 return next(v for k,v in zip(d.keys,d.values)if isinstance(k,ast.Constant)and k.value=='failure')
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only fixed thread masks')
 require(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'E storage environment applied')
 initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective512MiB admission')
 scope=read_json(a.ROOT_review);configuration=record(a.configuration)
 require(scope['accepted']is True and scope['reviewer']=='/root'
  and scope['scope']=='ASR18_saved156600_source27_exact_postrestore_selector_metadata_v1'
  and scope['typed_fixture_source']==record(__file__)and scope['source_configuration']==configuration,'Exact separately admitted ROOT metadata scope')
 for item in scope['input_pins']:require(record(item['path'])==item,'Exact input pin BEFORE finite metadata proof')
 output=Path(a.output).resolve();require(output.is_relative_to(F)and not output.exists(),'ONE fresh E selector metadata output');output.mkdir()
 me=psutil.Process();parent=me.parent();identities={'producer':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}}
 write_json(output/'actual_process_entry.json',identities)
 from run_ASR18_BPE9223_resident_RAM_training_v27 import checked_postrestore_selector_resume
 base=read_json(a.configuration);pointer=record(B/'checkpoints/update_000156600/pointer.json');newpin=base['prior_resident_RAM_source26_postrestore_selector_failure'];oldpin=base['prior_resident_RAM_native_pressure156769_failure']
 actual=checked_postrestore_selector_resume(newpin,base=base,folder=B,pointer_pin=pointer)
 require(actual['new_native_updates']==0 and actual['orphan19_saved_prefix_STAT']['bytes']==317194933,'Genuine stopped startup and preserved orphan metadata')
 old_source=read_json(scope['base_source_configuration']['path']);inverse=copy.deepcopy(base);del inverse['prior_resident_RAM_source26_postrestore_selector_failure'];require(inverse==old_source,'ONE new operational provenance field only; approved225 policy unchanged')
 trees={k:ast.parse(path.read_text())for k,path in [('old',D26),('new',D27),('wrapper',W27)]}
 saved=read_json(pointer['path'])['journal_prefix'];old_closure=read_json(oldpin['path'])
 preserved={'whole_journal_including_preserved_tail':old_closure['closed_journal_stat'],'preserved_tail_positive_updates':list(range(156601,156770))}
 class MetadataBindingOnly:
  def binding(self):return {'path':str(B/'journals/commits_000020.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
 names=('postrestore_selector_resume','cushion225_resume','packet_credit_resume','native_headroom_resume','abnormal_resume','original_interval_resume','source_transition_resume','prior_process','DEV_frontend_resume','DEV_admission_resume','periodic_quiescence_resume')
 def args(**kw):return types.SimpleNamespace(**{**dict.fromkeys(names),**kw})
 def env(aa):return {'a':aa,'record':record,'require':require,'interval_failure':None,'abnormal':old_closure,'source_journal':saved,'preserved_tail':preserved,'journal':MetadataBindingOnly()}
 cases=[]
 def passed(name,**e):cases.append({'case':name,'passed':True,**e})
 def rejects(name,call,expected=ValueError):
  try:call()
  except expected as ex:passed(name,actual_exception_type=type(ex).__name__,actual_error=str(ex));return
  raise AssertionError('Expected exact rejection '+name)
 old_args=args(cushion225_resume=oldpin['path']);new_args=args(cushion225_resume=oldpin['path'],postrestore_selector_resume=newpin['path'])
 rejects('old26_exact_full_postrestore_branch_dictionary_reproduces_record_None',lambda:expression(branch_dict(trees['old']),env(old_args)),TypeError)
 new_dict=expression(branch_dict(trees['new']),env(new_args))
 require(new_dict['predecessor']==newpin and new_dict['saved_prefix']==saved and new_dict['native_updates_or_draws_in_copy']==0
  and new_dict['historical_unsaved_tail_updates']==list(range(156601,156770)),'Exact new production branch dictionary and predecessor path')
 passed('new27_exact_full_postrestore_branch_dictionary',actual_predecessor=new_dict['predecessor'],metadata_binding_double_only=True)
 legacy_dict=expression(branch_dict(trees['new']),env(old_args));require(legacy_dict['predecessor']==oldpin,'Original225 predecessor remains correctly wired');passed('original225_lane_branch_selector')
 for label,aa,pin in [('repaired',new_args,newpin),('original225',old_args,oldpin)]:
  require(expression(factory_failure(trees['new']),env(aa))==pin,'Actual production restore-factory predecessor');passed(label+'_exact_restore_factory_failure_selector')
 guardnode=find(main_node(trees['new']),lambda n:isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(x,ast.Constant)and x.value=='Startup metadata repair retains original approved225 abnormal lane; never normal0'for x in n.value.args))
 wrapperguard=find(main_node(trees['wrapper']),lambda n:isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(x,ast.Constant)and x.value=='Exact startup repair cannot enter a normal or unrelated predecessor lane'for x in n.value.args))
 for label,node in [('driver',guardnode),('wrapper',wrapperguard)]:
  statements([node],env(new_args));passed(label+'_exact_companion_argument_guard_accepts')
  rejects(label+'_missing225_companion_rejected',lambda node=node:statements([node],env(args(postrestore_selector_resume=newpin['path']))))
  rejects(label+'_normal0_relabel_rejected',lambda node=node:statements([node],env(args(postrestore_selector_resume=newpin['path'],cushion225_resume=oldpin['path'],prior_process='not-a-real-normal0'))))
 invcmp=find(main_node(trees['new']),lambda n:isinstance(n,ast.Compare)and isinstance(n.left,ast.Attribute)and n.left.attr=='invocation'and 'postrestore_selector_resume'in ast.unparse(n))
 for invocation,good in [(19,False),(20,True),(21,False)]:
  aa=copy.copy(new_args);aa.invocation=invocation;require(expression(invcmp,env(aa))is good,'Exact initial invocation20 gate');passed('driver_exact_invocation_'+str(invocation),accepted=good)
 entry=find(main_node(trees['wrapper']),lambda n:isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(x,ast.Constant)and x.value=='Exact source24 TOOL1 saved156600 plus human-approved225 cushion, never normal0'for x in n.value.args))
 # A metadata-only review binding double; this is NOT production authorization.
 review={'cushion225_resume':oldpin,'postrestore_selector_resume':newpin,'native_update_commit_pressure_policy':base['native_update_commit_pressure_policy'],'packet_credit_policy':base['resident_RAM_packet_credit_policy'],'initial_owner_invocation':20}
 aa=copy.copy(new_args);aa.start_invocation=20;ee={**env(aa),'review':review,'configuration':base};statements([entry],ee);passed('exact_full_wrapper_initial_entry_require_accepts',review_binding_double_only=True)
 aa=copy.copy(new_args);aa.start_invocation=19
 rejects('exact_full_wrapper_wrong_initial19_rejected',lambda:statements([entry],{**env(aa),'review':review,'configuration':base}))
 wrongreview={**review,'postrestore_selector_resume':oldpin}
 aa=copy.copy(new_args);aa.start_invocation=20
 rejects('exact_full_wrapper_wrong_startup_provenance_rejected',lambda:statements([entry],{**env(aa),'review':wrongreview,'configuration':base}))
 # The actual wrapper initial command construction is evaluated without Popen.
 wm=main_node(trees['wrapper'])
 command_assign=find(wm,lambda n:isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='command'for t in n.targets))
 command_if=find(wm,lambda n:isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and ast.unparse(n.test)=='prior is not None'and any(isinstance(x,ast.Name)and x.id=='command'for x in ast.walk(n)))
 for label,invocation,prior in [('initial',20,None),('future_normal',21,Path('nonexistent-normal-report-metadata-only.json'))]:
  aa=copy.copy(new_args);aa.start_invocation=20;aa.configuration=a.configuration;aa.pointer=pointer['path'];aa.latest_failure=str(N/'pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json');aa.queue_bytes=268435456
  ee={'a':aa,'P':P,'str':str,'driver':D27,'pointer':Path(pointer['path']),'branch':B,'invocation':invocation,'prior':prior}
  statements([command_assign,command_if],ee);command=ee['command']
  require((command.count('--postrestore-selector-resume')==1 and command.count('--cushion225-resume')==1)if label=='initial'else('--postrestore-selector-resume'not in command and '--cushion225-resume'not in command and '--prior-process'in command),'Exact initial forwarding and strict future normal command')
  passed(label+'_exact_wrapper_command_AST_no_Popen',command=command)
 # Genuine new validator rejection controls; original production files stay immutable.
 badpin={**newpin,'sha256':'0'*64};rejects('typed_wrong_closure_pin',lambda:checked_postrestore_selector_resume(badpin,base=base,folder=B,pointer_pin=pointer))
 badbase=copy.deepcopy(base);badbase['native_update_commit_pressure_policy']['parameters']['additional_native_commit_pressure_buffer_bytes']=2000000000
 rejects('typed_unauthorized_policy_change',lambda:checked_postrestore_selector_resume(newpin,base=badbase,folder=B,pointer_pin=pointer))
 badpointer={**pointer,'sha256':'0'*64};rejects('typed_wrong_saved_pointer',lambda:checked_postrestore_selector_resume(newpin,base=base,folder=B,pointer_pin=badpointer))
 require(not(B/'invocations/owner_000020').exists()and not(B/'journals/commits_000020.jsonl').exists()and not(N/'resident_RAM_training_actual_21').exists(),'Fresh future names remain absent')
 require('numpy'not in sys.modules and 'torch'not in sys.modules,'No model/NumPy/Torch/GPU imports')
 for item in scope['input_pins']:require(record(item['path'])==item,'Exact input pin AFTER finite metadata proof')
 write_json(output/'report.json',{'kind':'ASR18_actual_saved156600_source27_exact_postrestore_selector_metadata_v1','passed':True,'source_configuration':configuration,'actual_source27_typed_validator_passed':True,'cases':cases,'case_count':len(cases),'exact_branch_metadata':new_dict,'old26_actual_TypeError_reproduced':True,'source_ASTs_extracted_without_rewrite':{'old':record(D26),'new':record(D27),'wrapper':record(W27)},'new_journal_binding_explicit_metadata_double_NOT_real_writer':True,'closed_225_Monitor_transport_and169tail_controls_reused_not_rerun':True,'old_original18_and_orphan19_STAT_only':True,'actual_audio_model_native_restore_copy_journal_operations':0,'actual_authoritative_sampler_draws':0,'actual_GPU_activity':False,'actual_kernel_credit_objects_created':0,'future_names_absent':True,'identities':identities,'guard_initial':initial,'guard_final':guard(),'peak_private_bytes':peak,'input_pins_before_after_exact':scope['input_pins'],'production_admitted':False})
 print(json.dumps({'report':record(output/'report.json'),'case_count':len(cases),'identities':identities}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);p.add_argument('--configuration',required=True);p.add_argument('--output',required=True);main(p.parse_args())
