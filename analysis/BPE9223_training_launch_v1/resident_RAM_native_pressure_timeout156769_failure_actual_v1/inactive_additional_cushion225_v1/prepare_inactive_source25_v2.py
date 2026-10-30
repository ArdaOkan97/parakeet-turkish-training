"""Authorized inactive source25 metadata derivation only; no candidate code/control execution."""
import ast,copy,difflib,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parents[1];B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
OLD_D=R/'run_ASR18_BPE9223_resident_RAM_training_v24.py';NEW_D=R/'run_ASR18_BPE9223_resident_RAM_training_v25.py'
OLD_W=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v24.py';NEW_W=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v25.py'
OLD_GATE=N/'resident_RAM_system_headroom156300_failure_actual_v1/asr18_native_update_commit_pressure_gate_v2.py';GATE=F/'asr18_native_update_commit_pressure_gate_v3.py'
OLD_CONFIG=N/'resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1/production_configuration_v1/configuration.json'
SCOPE=N/'ROOT_INACTIVE_additional_commit_cushion2250000000_saved156600_implementation_scope_v1.json'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def guard():
 s=sample_memory(os.getpid(),'inactive_source25_cushion225_metadata_preparation');m=s['system_memory']
 require(s['worker_private_memory']['private_bytes']<512*1024**2 and s['paired_sampling_within_one_second']
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original512MiB/180s/full256MiB/headroom/E100GB')
 return s
initial=guard();m=initial['system_memory']
require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full512MiB prospective metadata allowance')
scope=read(SCOPE)
require(pin(SCOPE)['sha256']=='15722798743f196ac255206fa5170816bb844d841a7ab20f88c176089ff20ba8'
 and scope['human_authorization']['explicit_policy_override_received']is True,'Genuine exact ROOT inactive preparation scope')
for p,expected in [(OLD_D,'8104277ee24ce4329a2aeb1d970c7563c3440ba6e73648b2a29fb6b904a942ab'),
 (OLD_W,'e71af424728ecd128a0953653ff0b730a1b897c9911ff992f30282d525461c48'),
 (OLD_CONFIG,'8b5e132a8df69886a610bd42479b9d6cb10df8aa1f9b51bef9a0666246e983c6'),
 (OLD_GATE,'95d89128a82595d21bab3e368dad5f391bde70aab019c2f4e7400d7578abb29c')]:require(pin(p)['sha256']==expected,'Proper frozen source24 base')
require(GATE.read_bytes().replace(b'2_250_000_000',b'2_500_000_000')==OLD_GATE.read_bytes(),'Only helper numeric literal changed')
def text(p):return Path(p).read_bytes().decode('utf-8').replace('\r\n','\n')
def source_write(p,value,base):
 raw=base.read_bytes();encoded=value.replace('\n','\r\n').encode()if b'\r\n'in raw else value.encode()
 with p.open('xb')as f:f.write(encoded)
 return encoded
old_d=text(OLD_D);d=old_d;edits_d=[]
def change_d(before,after,count=1):
 global d
 require(d.count(before)==count,'Exact declared driver edit count: '+before[:100]);d=d.replace(before,after)
 edits_d.append({'before':before,'after':after,'count':count})
tree=ast.parse(old_d);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='checked_native_commit_pressure_policy')
old_check='\n'.join(old_d.splitlines()[node.lineno-1:node.end_lineno])
new_check=old_check.replace('resident_RAM_system_headroom156300_failure_actual_v1/asr18_native_update_commit_pressure_gate_v2.py',
 'resident_RAM_native_pressure_timeout156769_failure_actual_v1/inactive_additional_cushion225_v1/asr18_native_update_commit_pressure_gate_v3.py')
new_check=new_check.replace('95d89128a82595d21bab3e368dad5f391bde70aab019c2f4e7400d7578abb29c',pin(GATE)['sha256'])
new_check=new_check.replace('helper2, no source alias','helper3, no source alias').replace('ASR18_exact_native_commit_pressure_gate_v2','ASR18_exact_native_commit_pressure_gate_v3')
new_check=new_check.replace("review['scope']=='ASR18_native_preupdate_commit_pressure_scheduling_v1'","review['scope']=='ASR18_native_preupdate_commit_cushion2250000000_policy_v1'\n        and review['human_approved_additional_buffer_bytes']==2250000000")
new_check=new_check.replace("'Exact ROOT completed method-stub plus genuine original Monitor CPU scope, no allocation guarantee'","'Exact ROOT focused candidate proof and explicit human225 approval, no allocation guarantee'")
new_check=new_check.replace("and proof['actual_GPU_activity'] is False and proof['genuine_original_Monitor']['original_monitor_thread_joined'] is True,\n        'Honest finite CPU scope and actual original Monitor closure, never a production benchmark')",
 "and proof['actual_GPU_activity'] is False and proof['focused_policy_delta_controls_passed'] is True\n        and proof['fake_clock_and_historical_samples_explicit'] is True and proof['original_Monitor_AST_unchanged'] is True\n        and proof['old_genuine_Monitor_review_reused_without_execution'] is True\n        and proof['original_genuine_Monitor_completed_review']==record(ROOT/'analysis/BPE9223_training_launch_v1/ROOT_native_preupdate_commit_pressure_actual_CPU_review_v1.json')\n        and proof['gate_source']==policy['gate_source'] and proof['policy']==policy['parameters'],\n        'Honest focused delta proof; closed genuine Monitor evidence reused, not re-executed')")
require("genuine_original_Monitor']['original_monitor_thread_joined"not in new_check,'New proof gate schema is honest')
change_d(old_check,new_check)
fragment=text(F/'checked_cushion225_resume_body_v1.pyfrag')
change_d('def checked_packet_credit_resume(pin,*,base,folder,pointer_pin):',fragment+'def checked_packet_credit_resume(pin,*,base,folder,pointer_pin):')
change_d("require(sum(x is not None for x in (a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume))<=1",
 "require(sum(x is not None for x in (a.cushion225_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume))<=1")
change_d("    packet_credit_failure=None if a.packet_credit_resume is None else checked_packet_credit_resume(",
 "    cushion225_failure=None if a.cushion225_resume is None else checked_cushion225_resume(\n        record(a.cushion225_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    packet_credit_failure=None if a.packet_credit_resume is None else checked_packet_credit_resume(")
change_d("    if packet_credit_failure is not None:abnormal=packet_credit_failure",
 "    if packet_credit_failure is not None:abnormal=packet_credit_failure\n    if cushion225_failure is not None:abnormal=cushion225_failure")
change_d("a.invocation==(18 if a.packet_credit_resume is not None else(17 if a.native_headroom_resume is not None else 11))",
 "a.invocation==(19 if a.cushion225_resume is not None else(18 if a.packet_credit_resume is not None else(17 if a.native_headroom_resume is not None else 11)))")
change_d("check_original_closed_journal_stat(abnormal['whole_journal_including_preserved_tail'])if a.packet_credit_resume is not None else",
 "check_original_closed_journal_stat(abnormal['whole_journal_including_preserved_tail'])if (a.cushion225_resume is not None or a.packet_credit_resume is not None) else")
change_d("check_original_closed_journal_stat(preserved_tail['whole_journal_including_preserved_tail'])if a.packet_credit_resume is not None else",
 "check_original_closed_journal_stat(preserved_tail['whole_journal_including_preserved_tail'])if (a.cushion225_resume is not None or a.packet_credit_resume is not None) else")
change_d("p.add_argument('--packet-credit-resume');","p.add_argument('--cushion225-resume');p.add_argument('--packet-credit-resume');")
ast.parse(d);source_write(NEW_D,d,OLD_D)
old_w=text(OLD_W);w=old_w;edits_w=[]
def change_w(before,after,count=1):
 global w
 require(w.count(before)==count,'Exact declared wrapper edit count: '+before[:100]);w=w.replace(before,after)
 edits_w.append({'before':before,'after':after,'count':count})
change_w('run_ASR18_BPE9223_resident_RAM_training_v24','run_ASR18_BPE9223_resident_RAM_training_v25',8)
change_w("require(a.packet_credit_resume is None and a.native_headroom_resume is None",
 "require(a.cushion225_resume is None and a.packet_credit_resume is None and a.native_headroom_resume is None")
new_branch="""    elif a.cushion225_resume is not None:
        require(a.packet_credit_resume is None and a.native_headroom_resume is None and a.abnormal_resume is None
            and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None
            and a.DEV_admission_resume is None and a.periodic_quiescence_resume is None
            and review['cushion225_resume']==record(a.cushion225_resume)
            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']
            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']
            and a.start_invocation==review['initial_owner_invocation']==19,
            'Exact source24 TOOL1 saved156600 plus human-approved225 cushion, never normal0')
"""
change_w("    elif a.packet_credit_resume is not None:\n",new_branch+"    elif a.packet_credit_resume is not None:\n")
new_validation="""    if a.cushion225_resume is not None:
        from run_ASR18_BPE9223_resident_RAM_training_v25 import checked_cushion225_resume
        checked_cushion225_resume(record(a.cushion225_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
"""
change_w("    if a.packet_credit_resume is not None:\n",new_validation+"    if a.packet_credit_resume is not None:\n")
change_w("            if a.packet_credit_resume is not None:command+=['--packet-credit-resume',a.packet_credit_resume]",
 "            if a.cushion225_resume is not None:command+=['--cushion225-resume',a.cushion225_resume]\n            elif a.packet_credit_resume is not None:command+=['--packet-credit-resume',a.packet_credit_resume]")
change_w("p.add_argument('--packet-credit-resume');","p.add_argument('--cushion225-resume');p.add_argument('--packet-credit-resume');")
ast.parse(w);source_write(NEW_W,w,OLD_W)
def inverse(candidate,prior,edits):
 restored=text(candidate)
 for item in reversed(edits):
  require(restored.count(item['after'])==item['count'],'Exact inverse occurrence');restored=restored.replace(item['after'],item['before'])
 require(restored==text(prior),'Full normalized inverse')
 raw=restored.replace('\n','\r\n').encode()if b'\r\n'in prior.read_bytes()else restored.encode()
 require(raw==prior.read_bytes(),'Full original RAW BYTE inverse')
 return {'prior':pin(prior),'candidate':pin(candidate),'declared_edits':edits,'full_normalized_inverse_exact':True,'full_original_RAW_BYTE_inverse_exact':True}
inv_d=inverse(NEW_D,OLD_D,edits_d);inv_w=inverse(NEW_W,OLD_W,edits_w)
old_nodes={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(old_d).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_nodes={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(d).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
unchanged=[name for name,value in old_nodes.items()if name not in ('main','checked_native_commit_pressure_policy')and new_nodes[name]==value]
require(len(unchanged)==len(old_nodes)-2,'Every old non-entry/non-policy function/class remains AST exact')
def main_node(value):return next(n for n in ast.parse(value).body if isinstance(n,ast.FunctionDef)and n.name=='main')
old_nested={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(main_node(old_d))if isinstance(n,ast.FunctionDef)and n.name!='main'}
new_nested={n.name:ast.dump(n,include_attributes=False)for n in ast.walk(main_node(d))if isinstance(n,ast.FunctionDef)and n.name!='main'}
require(old_nested==new_nested,'All existing nested functions including DEV/record_periodic exact')
def native_loop(value):
 return next(n for n in ast.walk(main_node(value))if isinstance(n,ast.While)and ast.unparse(n.test)=="optimizer.updates < c['planned_native_updates']")
require(ast.dump(native_loop(old_d),include_attributes=False)==ast.dump(native_loop(d),include_attributes=False),'Entire existing native train/checkpoint/DEV while loop AST exact')
for before,after,name in [(old_d,d,'driver24_to25.diff'),(old_w,w,'wrapper24_to25.diff'),(text(OLD_GATE),text(GATE),'gate2_to3.diff')]:
 with (F/name).open('x',encoding='utf-8')as f:f.write(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='frozen_base',tofile='inactive_successor')))
base=read(OLD_CONFIG);candidate=copy.deepcopy(base)
candidate['native_update_commit_pressure_policy']['gate_source']=pin(GATE)
candidate['native_update_commit_pressure_policy']['parameters']['additional_native_commit_pressure_buffer_bytes']=2250000000
candidate['native_update_commit_pressure_policy']['actual_CPU_control_review']=None
candidate['prior_resident_RAM_native_pressure156769_failure']=pin(F.parent/'actual_failure_closure.json')
configuration_path=F/'proposed_configuration_v1.json';write(configuration_path,candidate)
restored=copy.deepcopy(candidate);del restored['prior_resident_RAM_native_pressure156769_failure'];restored['native_update_commit_pressure_policy']=base['native_update_commit_pressure_policy']
require(restored==base,'Only2top-level operational config fields, all old values exact')
parameters=candidate['native_update_commit_pressure_policy']['parameters']
require(parameters['native_actual_commit_headroom_bytes']+parameters['full_additional_prefetch_host_reservation_bytes']+parameters['additional_native_commit_pressure_buffer_bytes']==4518435456
 and parameters['native_actual_physical_headroom_bytes']+parameters['full_additional_prefetch_host_reservation_bytes']==1768435456,'Candidate derived thresholds')
def leaves(left,right,path=''):
 changes=[]
 if isinstance(left,dict)and isinstance(right,dict):
  for key in sorted(set(left)|set(right)):
   if key not in left:changes.append({'path':path+'/'+key,'before_absent':True,'after':right[key]})
   elif key not in right:changes.append({'path':path+'/'+key,'before':left[key],'after_absent':True})
   else:changes+=leaves(left[key],right[key],path+'/'+key)
 elif left!=right:changes.append({'path':path,'before':left,'after':right})
 return changes
leaf_delta=leaves(base,candidate)
for file in [F/'verify_focused_cushion225_policy_CPU_v2.py',GATE,NEW_D,NEW_W]:ast.parse(text(file))
report={'kind':'ASR18_INACTIVE_source25_cushion225_full_inverse_v1','candidate_functions_or_controls_executed':False,
 'implementation_scope':pin(SCOPE),'human_approved_numeric_delta_only':{'before':2500000000,'after':2250000000},
 'driver':inv_d,'wrapper':inv_w,'gate':{'prior':pin(OLD_GATE),'candidate':pin(GATE),'full_RAW_BYTE_inverse_after_one_literal_exact':True},
 'all_original_nonentry_nonpolicy_AST_exact':unchanged,'all_existing_nested_main_functions_AST_exact':sorted(old_nested),
 'entire_native_checkpoint_DEV_while_loop_AST_exact':True,'source24_proper_configuration_base':pin(OLD_CONFIG),
 'candidate_configuration':pin(configuration_path),'only_two_operational_top_fields':['native_update_commit_pressure_policy','prior_resident_RAM_native_pressure156769_failure'],
 'full_config_inverse_exact':True,'exact_leaf_delta':leaf_delta,'genuine_Monitor_transport_DEVor_numeric_proof_not_rerun':True,
 'guard_initial':initial,'guard_final':guard()}
write(F/'source25_full_inverse_and_configuration_report.json',report)
old_execution=read(N/'ROOT_owned_packet_credit_saved156400_execution_review_v1.json');pointer=B/'checkpoints/update_000156600/pointer.json'
runtime={'worker_driver_source':pin(NEW_D),'parent_driver_source':pin(NEW_D),'resident_RAM_stream_configuration':pin(configuration_path),
 'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':pin(pointer),'source_configuration':pin(configuration_path)}
require(len(runtime)==7,'Original7runtimebinding KINDS retained')
future_output=N/'resident_RAM_training_actual_20'
require(not(B/'invocations/owner_000019').exists()and not(B/'journals/commits_000019.jsonl').exists()and not future_output.exists(),'Future inv19/output20/journal19 absent; no launch')
required={k:pin(p)for k,p in {'implementation_scope':SCOPE,'driver_source':NEW_D,'wrapper_source':NEW_W,'gate_source':GATE,
 'old_gate_source':OLD_GATE,'fixture_source':F/'verify_focused_cushion225_policy_CPU_v2.py','proof_schema':F/'focused_CPU_proof_schema_v1.json',
 'source_configuration':configuration_path,'base_source_configuration':OLD_CONFIG,
 'failure':F.parent/'actual_failure_closure.json','failure_completion':F.parent/'actual_metadata_seal_completion.json',
 'ROOT_closed_failure_review':N/'ROOT_source24_native_pressure_timeout156769_saved156600_compact_review_v1.json',
 'initial_checkpoint':pointer,'inverse_report':F/'source25_full_inverse_and_configuration_report.json',
 'CPU_source':S/'fastmss_resident_feature_stream_v10.py','DEV_helper_source':S/'asr18_resident_serial350_DEV_v2.py',
 'quiescence_source':S/'asr18_resident_periodic_CPU_quiescence_v3.py','RAM_native_source':S/'asr18_BPE9223_grouped_RAM_native_v1.py',
 'private_scalar_prefetch_source':S/'asr18_BPE9223_private_scalar_prefetch_v1.py'}.items()}
handoff={'kind':'ASR18_INACTIVE_source25_cushion225_scope_handoff_v1','pins':required,'proper_source24_base':True,
 'helper_source':required['gate_source'],'focused_CPU_report_not_yet_exists':True,'actual_CPU_control_review_placeholder':None,
 'CPU_execution_requires_separate_ROOT_scope':True,'production_execution_admitted':False,'all_original_guards_reservations_science_retained':True,
 'focused_CPU_limits':{'wall_seconds':180,'external_wall_seconds':240,'private_family_bytes':512*1024**2,'prospective_full_allowance_bytes':512*1024**2,
 'full_additional_prefetch_reservation_bytes':268435456,'original_commit_floor_bytes':2000000000,'original_physical_floor_bytes':1500000000,'E_free_reserve_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all4_math_threads':1,'priority':'BelowNormal'},
 'focused_CPU_future_arguments':['--ROOT-review','ROOT_SCOPE_NOT_YET_ISSUED','--configuration',str(configuration_path),'--output',str(F/'focused_policy_actual_CPU_01')],
 'input_pins_for_exact_ROOT_CPU_scope':[required[k]for k in ('driver_source','wrapper_source','gate_source','old_gate_source','source_configuration','base_source_configuration','inverse_report','implementation_scope','proof_schema')],
 'expected_future_7_runtime_bindings_NOT_actual_boot':runtime,'future_initial_owner_invocation':19,'future_active_journal':str(B/'journals/commits_000019.jsonl'),
 'draft_future_wrapper_arguments_UNADMITTED':['--configuration',str(configuration_path),'--pointer',str(pointer),'--latest-failure',old_execution['latest_failure']['path'],'--branch',str(B),'--output',str(future_output),'--queue-bytes','268435456','--start-invocation','19','--cushion225-resume',str(F.parent/'actual_failure_closure.json'),'--review','ROOT_FINAL_EXECUTION_REVIEW_NOT_YET_ISSUED'],
 'original_latest_failure_ACK260_pin_retained':old_execution['latest_failure'],'all_existing_historical_entry_functions_unchanged':True,
 'required_completed_CPU_review_schema':read(F/'focused_CPU_proof_schema_v1.json'),
 'future_recovery_only_after_final_ROOT_review':{'saved':156600,'prefix_bytes':317194933,'prefix_sha256':'eb6b816bff1f40706a4595a045c2c3b9cafc5eaedccff6dcc010fab0d0e2a199','historical169_preserved':True,'genuine_new170':[156601,156770],'natural_next_save':156800,'full_target':401072,'administrative_stop':None,'no_finite_stop':True}}
require(len(handoff['draft_future_wrapper_arguments_UNADMITTED'])==18,'Exact18draft argumentstrings')
write(F/'inactive_source25_handoff_v1.json',handoff)
print(json.dumps({'handoff':pin(F/'inactive_source25_handoff_v1.json'),'driver':pin(NEW_D),'wrapper':pin(NEW_W),'configuration':pin(configuration_path),'inverse':pin(F/'source25_full_inverse_and_configuration_report.json'),'candidate_functions_executed':False}),flush=True)

