"""Metadata-only final source22 assembly and declared inverse; no owner/model."""
import ast,copy,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');F=Path(__file__).parent
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def guard():
 s=sample_memory(os.getpid(),'source22_metadata_assembly_inverse');m=s['system_memory']
 require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original metadata limits/full256MiB/headroom/E guards');return s
initial=guard();require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full512MiB metadata reserve')
changes=read(F/'declared_source22_inverse_changes_v1.json')
oldD=R/'run_ASR18_BPE9223_resident_RAM_training_v20.py';newD=R/'run_ASR18_BPE9223_resident_RAM_training_v22.py'
oldW=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v20.py';newW=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v22.py'
norm=lambda p:Path(p).read_bytes().decode('utf-8').replace('\r\n','\n')
inverse=norm(newD)
for old,new in reversed(changes['driver']):
 require(inverse.count(new)==1,'One declared inverse driver replacement');inverse=inverse.replace(new,old)
require(inverse.replace('\n','\r\n').encode('utf-8')==oldD.read_bytes(),'FULL BYTE inverse source22 to frozen source20')
inverse=norm(newW).replace('run_ASR18_BPE9223_resident_RAM_training_v22','run_ASR18_BPE9223_resident_RAM_training_v21')
for item in reversed(changes['wrapper']):
 old,new=item[:2];require(inverse.count(new)>=1 if len(item)>2 and item[2]else inverse.count(new)==1,'Declared inverse wrapper replacement');inverse=inverse.replace(new,old)
require(inverse.replace('\n','\r\n').encode('utf-8')==oldW.read_bytes(),'FULL BYTE inverse wrapper22 to frozen wrapper20')
t0=ast.parse(norm(oldD));t1=ast.parse(norm(newD))
oldnodes={x.name:x for x in t0.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))and x.name!='main'}
newnodes={x.name:x for x in t1.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))and x.name!='main'}
for name,node in oldnodes.items():require(ast.dump(node,include_attributes=False)==ast.dump(newnodes[name],include_attributes=False),'Exact old nonentry AST '+name)
m0=next(x for x in t0.body if isinstance(x,ast.FunctionDef)and x.name=='main');m1=next(x for x in t1.body if isinstance(x,ast.FunctionDef)and x.name=='main')
nested0={x.name:x for x in ast.walk(m0)if isinstance(x,ast.FunctionDef)and x.name not in ('main','journal_factory')}
nested1={x.name:x for x in ast.walk(m1)if isinstance(x,ast.FunctionDef)and x.name not in ('main','journal_factory')}
for name,node in nested0.items():require(ast.dump(node,include_attributes=False)==ast.dump(nested1[name],include_attributes=False),'Exact original nested function AST '+name)
syntax={}
for path in (newD,newW,F/'asr18_native_update_commit_pressure_gate_v2.py',F/'verify_inactive_native_pressure_gate_CPU_v2.py',F/'validate_final_saved156200_native_pressure_metadata_v1.py'):
 compile(path.read_bytes(),str(path),'exec');syntax[str(path)]=True
base_path=N/'resident_RAM_original_interval_post142894_failure_actual_v1/production_configuration_v1/configuration.json'
require(pin(base_path)=={'path':str(base_path),'bytes':210999,'sha256':'68b8279422d3b294fe03d851e25270e525ced5c7586493990d50551d9ed93dce'},'Proper source20 SOURCE base, not effective invocation configuration')
base=read(base_path);new=copy.deepcopy(base)
CPU_review_path=N/'ROOT_native_preupdate_commit_pressure_actual_CPU_review_v1.json';CPU_review=read(CPU_review_path)
require(pin(CPU_review_path)['sha256']=='2e63113a9d42e0337587384ab11b685c0749f47e20c0502640e03ca83b64af1a'
 and CPU_review['accepted']is True and CPU_review['reviewer']=='/root'and CPU_review['scope']=='ASR18_native_preupdate_commit_pressure_scheduling_v1','Actual completed CPU ROOT scope')
gate_pin=pin(F/'asr18_native_update_commit_pressure_gate_v2.py')
require(CPU_review['gate_source']==gate_pin and gate_pin['sha256']=='95d89128a82595d21bab3e368dad5f391bde70aab019c2f4e7400d7578abb29c','Exact helper2 completed CPU binding')
parameters={'kind':'ASR18_native_preupdate_additional_commit_pressure_buffer_v1','native_actual_commit_headroom_bytes':2000000000,'native_actual_physical_headroom_bytes':1500000000,'full_additional_prefetch_host_reservation_bytes':268435456,'additional_native_commit_pressure_buffer_bytes':2500000000,'maximum_wait_seconds':300,'poll_interval_seconds':.5,'freed_memory_credit_claimed':False,'workload_forecast_claimed':False}
require(CPU_review['policy']==parameters,'Exact stronger scheduling parameters; no saving/forecast claim')
new['native_update_commit_pressure_policy']={'gate_source':gate_pin,'parameters':parameters,'actual_CPU_control_review':pin(CPU_review_path)}
new['prior_resident_RAM_native_headroom156300_failure']=pin(F/'actual_failure_closure.json')
removed=copy.deepcopy(new)
for name in ('native_update_commit_pressure_policy','prior_resident_RAM_native_headroom156300_failure'):del removed[name]
require(removed==base,'Only TWO operational source config fields; every original science/resource/CPU9/seven-overlay value exact')
out=F/'production_configuration_v1';require(not out.exists(),'One immutable configuration candidate');out.mkdir()
config_path=out/'configuration.json';config_path.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');config_pin=pin(config_path)
pointer_pin=pin(B/'checkpoints/update_000156200/pointer.json');pointer=read(pointer_pin['path'])
require(pointer['committed_updates']==156200 and pointer['journal_prefix']['offset']==316408290,'Exact saved156200 metadata')
runtime=copy.deepcopy(new);runtime_updates={'worker_driver_source':pin(newD),'parent_driver_source':pin(newD),'resident_RAM_stream_configuration':config_pin,'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':pointer_pin,'source_configuration':config_pin};runtime.update(runtime_updates)
prior_runtime=read(B/'invocations/owner_000016/configuration.json')
def leafdiff(a,b,path='$'):
 if type(a)is not type(b):return [{'path':path,'before':a,'after':b}]
 if isinstance(a,dict):
  result=[]
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:result.append({'path':path+'.'+k,'before':a.get(k),'after':b.get(k),'added':k not in a,'removed':k not in b})
   else:result+=leafdiff(a[k],b[k],path+'.'+k)
  return result
 if isinstance(a,list):
  if len(a)!=len(b):return [{'path':path,'before':a,'after':b}]
  return sum((leafdiff(x,y,path+'['+str(i)+']')for i,(x,y)in enumerate(zip(a,b))),[])
 return []if a==b else[{'path':path,'before':a,'after':b}]
source_diff=leafdiff(base,new)
require({x['path']for x in source_diff}=={'$.native_update_commit_pressure_policy','$.prior_resident_RAM_native_headroom156300_failure'},'Exactly2 added top fields')
runtime_diff=leafdiff(new,runtime);require(all(x['path'].split('.')[1]in runtime_updates for x in runtime_diff),'Only declared original7 runtime materializations')
leaf_report={'source_base':pin(base_path),'new_source_configuration':config_pin,'source_semantic_changes':source_diff,'source_base_vs_prior_source20_effective_runtime':leafdiff(base,prior_runtime),'new_source_vs_projected_source22_runtime':runtime_diff,'prior_source20_runtime_vs_projected_source22_runtime':leafdiff(prior_runtime,runtime),'runtime_materialization_bindings':runtime_updates,'checkpoint_cadence200_and_all_scientific_state_preserved':True}
leaf_path=out/'source_and_runtime_leaf_differences.json';leaf_path.write_text(json.dumps(leaf_report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
inverse_report={'kind':'ASR18_inactive_source22_full_BYTE_inverse_and_AST_readiness_v1','passed':True,'driver_source20':pin(oldD),'driver_source22':pin(newD),'wrapper_source20':pin(oldW),'wrapper_source22':pin(newW),'declared_changes':pin(F/'declared_source22_inverse_changes_v1.json'),'driver_full_BYTE_inverse':True,'wrapper_full_BYTE_inverse':True,'old_nonentry_functions_classes_AST_exact':list(oldnodes),'original_nested_main_functions_AST_exact_except_declared_journal_factory_metadata':list(nested0),'syntax_passed_without_execution':syntax,'only_two_operational_source_configuration_fields':True,'resource_CPU9_seven_overlays_numerical_native_DEV_checkpoint200_periodic4000_unchanged':True,'inactive_unexecuted21_history':{'driver':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v21.py'),'wrapper':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v21.py'),'correction':'Unexecuted21 else11 spelling corrected in NEW22; wrapper imports21 to22 only.'},'no_model_audio_numpy_GPU_or_authoritative_operation':True,'metadata_resource_sample':guard()}
inverse_path=out/'actual_metadata_inverse_readiness_report.json';inverse_path.write_text(json.dumps(inverse_report,indent=2)+'\n',encoding='utf-8')
original_failure=N/'pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json'
future_output=N/'resident_RAM_training_actual_18'
inputs={'source_configuration':config_pin,'driver_source':pin(newD),'wrapper_source':pin(newW),'gate_source':gate_pin,'actual_CPU_ROOT_review':pin(CPU_review_path),'actual_CPU_report':CPU_review['actual_CPU_control_report'],'actual_failure_closure':pin(F/'actual_failure_closure.json'),'pointer':pointer_pin,'persistence':pin(B/'checkpoints/update_000156200/persistence_verified.json'),'latest_failure_original_ACK260':pin(original_failure)}
args=['--review',str(N/'ROOT_native_preupdate_commit_pressure_saved156200_execution_review_v1.json'),'--configuration',str(config_path),'--pointer',pointer_pin['path'],'--latest-failure',str(original_failure),'--branch',str(B),'--output',str(future_output),'--queue-bytes','268435456','--start-invocation','17','--native-headroom-resume',str(F/'actual_failure_closure.json')]
require(len(args)==18 and all(not path.exists()for path in (future_output,B/'invocations/owner_000017',B/'journals/commits_000017.jsonl')),'Only proposed future18-string invocation; new GPU names absent')
handoff={'kind':'ASR18_INACTIVE_source22_saved156200_native_pressure_final_source_configuration_handoff_v1','production_admitted':False,'CPU_composition_only_already_accepted':True,'source_configuration':config_pin,'driver_source':pin(newD),'wrapper_source':pin(newW),'source_base':pin(base_path),'source_and_runtime_leaf_report':pin(leaf_path),'actual_metadata_inverse_readiness':pin(inverse_path),'typed_metadata_control_source':pin(F/'validate_final_saved156200_native_pressure_metadata_v1.py'),'typed_metadata_inputs':inputs,'branch':str(B),'initial_owner_invocation':17,'future_GPU_output':str(future_output),'saved_prefix':pointer['journal_prefix'],'whole_old_journal':read(F/'actual_failure_closure.json')['whole_journal_verified_once'],'native_update_commit_pressure_policy':new['native_update_commit_pressure_policy'],'typed_abnormal_resume':pin(F/'actual_failure_closure.json'),'latest_failure_original_ACK260':pin(original_failure),'exact_future_wrapper_arguments':args,'full_run_target':401072,'administrative_update_stop':None,'strict_normal_wait0_dead_all_live_saved_future_path_retained':True,'no_prevention_saving_required_allocation_or_forecast_guarantee':True,'source22_helpers_CPU9_native1_DEVhelper2_seven_overlays_prefetch_quiescence_science_cadence_unchanged':True,'typed_metadata_control_expected_scope_schema':{'accepted':True,'reviewer':'/root','scope':'ASR18_native_pressure_saved156200_typed_metadata_control_v1','source_handoff':'record of this immutable handoff','metadata_control_source':pin(F/'validate_final_saved156200_native_pressure_metadata_v1.py')}}
handoff_path=out/'final_source22_configuration_handoff_v1.json';handoff_path.write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'handoff':pin(handoff_path),'configuration':config_pin,'inverse':pin(inverse_path),'leaf_report':pin(leaf_path),'metadata_resource_sample':guard()}),flush=True)
