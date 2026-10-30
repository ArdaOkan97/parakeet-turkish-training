"""Authorized inactive source assembly only; no control/producer/model execution."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');F=Path(__file__).parent
OLD=N/'resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1/asr18_owned_packet_credit_v1.py'
APPROVAL=N/'resident_RAM_source27_speed_review_v1/ROOT_direct_human_lookahead_and_checkpoint_restart_approval_v1.json'
def pin(p):p=Path(p);d=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(d).hexdigest(),'bytes':len(d)}
def write(p,x):assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n');return pin(p)
assert pin(APPROVAL)['sha256']=='f92036ea0b494897d85cf15d69c8eba07fb1abfd97febcf0a18099a0d8caf9fb'
ledgers=[]
def freeze(old,new,changes):
 original=old.read_bytes();text=original.decode().replace('\r\n','\n');edits=[]
 for label,before,after,count in changes:
  assert text.count(before)==count,(label,text.count(before));text=text.replace(before,after);edits.append({'label':label,'old_literal':before,'new_literal':after,'expected_count':count})
 ast.parse(text);undo=text
 for item in reversed(edits):
  assert undo.count(item['new_literal'])==item['expected_count'];undo=undo.replace(item['new_literal'],item['old_literal'])
 eol='CRLF'if b'\r\n'in original else'LF';reconstructed=undo.replace('\n','\r\n').encode()if eol=='CRLF'else undo.encode();assert reconstructed==original
 assert not new.exists();new.write_bytes(text.replace('\n','\r\n').encode()if eol=='CRLF'else text.encode())
 diff=F/(new.stem+'.diff');assert not diff.exists();diff.write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),new.read_bytes().decode().splitlines(True),fromfile=str(old),tofile=str(new))))
 proof={'old_source':pin(old),'new_source':pin(new),'exact_replacements':edits,'newline_handling':'Normalize to LF solely while replacing, restore original '+eol+' on write/inverse; no other normalization','full_old_raw_byte_stream_reconstructed_exact':True,'diff':pin(diff)};ledgers.append(proof);return proof
helper=freeze(OLD,F/'asr18_owned_packet_credit_v2.py',[
 ('honest helper docstring','"""Owned Win32 max1 packet credit, separate cancellation and exact parent death.','"""Owned Win32 max2 logical credits/receiverLIVE1, cancellation and exact parent death.',1),
 ('bounded release telemetry init','super().__init__();self.posts=0;self.cancelled=False','super().__init__();self.posts=0;self.cancelled=False;self.last_release_previous_count=None;self.release_previous_count_histogram={0:0,1:0}',1),
 ('descriptor kind','ASR18_owned_single_packet_credit_v1','ASR18_owned_two_logical_packet_credit_v1',2),
 ('descriptor2',"'logical_packet_maximum':1","'logical_packet_maximum':2",1),
 ('kernel initial/max2',"CreateSemaphoreW(None,1,1,self.descriptor['semaphore_name'])","CreateSemaphoreW(None,2,2,self.descriptor['semaphore_name'])",1),
 ('strict valid previous0or1',"if previous.value!=0:raise RuntimeError('Kernel credit did not transition exactly0 to1')\n        self.posts+=1","if previous.value not in (0,1):raise RuntimeError('Kernel two-credit release previous count must be0 or1')\n        self.posts+=1;self.last_release_previous_count=previous.value;self.release_previous_count_histogram[previous.value]+=1",1),
 ('bounded proof telemetry',"'successful_release_posts':self.posts,'cancelled':self.cancelled,","'successful_release_posts':self.posts,'last_release_previous_count':self.last_release_previous_count,\n                'release_previous_count_histogram':self.release_previous_count_histogram,'cancelled':self.cancelled,",1),
 ('consumer strictmax2',"descriptor['logical_packet_maximum']!=1","descriptor['logical_packet_maximum']!=2",1)])
cpu=freeze(S/'fastmss_resident_feature_stream_v10.py',S/'fastmss_resident_feature_stream_v11.py',[
 ('new immutable helper home',"resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1","resident_RAM_two_logical_packet_lookahead_v1",1),
 ('three helper imports only','asr18_owned_packet_credit_v1','asr18_owned_packet_credit_v2',3),
 ('policy human approval field',"'original_packet_wait_seconds','actual_CPU_control_review'}","'original_packet_wait_seconds','actual_CPU_control_review','human_approval'}",1),
 ('honest policy version','ASR18_owned_end_to_end_packet_credit_policy_v1','ASR18_owned_end_to_end_two_logical_packet_credit_policy_v1',1),
 ('only logical2',"policy['maximum_logical_packets']==1","policy['maximum_logical_packets']==2",1),
 ('direct human exception gate',"    review=bound(policy['actual_CPU_control_review'])","    approval=bound(policy['human_approval'])\n    require(policy['human_approval']['sha256']=='f92036ea0b494897d85cf15d69c8eba07fb1abfd97febcf0a18099a0d8caf9fb'\n        and approval['answer']=='Approve lookahead and checkpoint restart'\n        and approval['question_call']=='call_97054c763bcd41c589a672bf0cd7c05b',\n        'Exact direct human exception for2 logical permits; every byte/floor/reserve remains unchanged')\n    review=bound(policy['actual_CPU_control_review'])",1),
 ('new completed proof scope','ASR18_owned_end_to_end_packet_credit_transport_v1','ASR18_owned_two_logical_packet_one_receiver_transport_v1',1),
 ('actual lifetime proof required',"and review['synthetic_packets_no_audio_model'] is True,","and review['synthetic_packets_no_audio_model'] is True\n        and review['human_approval']==policy['human_approval'] and review['maximum_logical_packets']==2\n        and review['maximum_live_receiver_packets']==1\n        and review['payload_lifetime_and_partial_receive_interleavings_passed'] is True,",1)])
oldtree=ast.parse((S/'fastmss_resident_feature_stream_v10.py').read_text());newtree=ast.parse((S/'fastmss_resident_feature_stream_v11.py').read_text())
def named(tree):return {node.name:ast.dump(node,include_attributes=False)for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
unchanged=[name for name,value in named(oldtree).items()if name not in('checked_packet_credit_policy','worker_main','ResidentFeatureProducer')and named(newtree)[name]==value]
assert len(unchanged)==len(named(oldtree))-3
oldclass=next(x for x in oldtree.body if isinstance(x,ast.ClassDef)and x.name=='ResidentFeatureProducer');newclass=next(x for x in newtree.body if isinstance(x,ast.ClassDef)and x.name=='ResidentFeatureProducer')
unchanged_methods=[]
for method in oldclass.body:
 if isinstance(method,ast.FunctionDef)and method.name!='__init__':
  other=next(x for x in newclass.body if isinstance(x,ast.FunctionDef)and x.name==method.name);assert ast.dump(method,include_attributes=False)==ast.dump(other,include_attributes=False);unchanged_methods.append(method.name)
basepath=N/'resident_RAM_source26_postrestore_selector_failure_actual_v1/inactive_selector_fix_v2/production_configuration_v1/configuration.json';base=json.loads(basepath.read_text());candidate=copy.deepcopy(base)
candidate['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(S/'fastmss_resident_feature_stream_v11.py')
candidate['resident_RAM_packet_credit_policy']={'kind':'ASR18_owned_end_to_end_two_logical_packet_credit_policy_v1','helper_source':pin(F/'asr18_owned_packet_credit_v2.py'),'maximum_logical_packets':2,'credit_wait_deadline_seconds':600,'original_packet_wait_seconds':300,'actual_CPU_control_review':None,'human_approval':pin(APPROVAL)}
config=write(F/'proposed_configuration_v1.json',candidate)
inverse=copy.deepcopy(candidate);inverse['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'];inverse['resident_RAM_packet_credit_policy']=base['resident_RAM_packet_credit_policy'];assert inverse==base
proof=write(F/'helper2_CPU11_exact_raw_inverse_ledger_v1.json',{'sources':ledgers,'CPU_unchanged_nonentry_ASTs':unchanged,'CPU_all_runtime_transport_methods_except_constructor_AST_exact':unchanged_methods,'worker_main_only_helper_import_delta':True,'all_original_receiver_reservation_before_partial_read_and_payload_drop_order_exact':True,'proper_source_configuration_base':pin(basepath),'proposed_configuration':config,'only_config_differences':['resident_RAM_resource_scheduling_policy.CPU_protocol_source','resident_RAM_packet_credit_policy'],'every_other_old_value_exact':True,'human_approval':pin(APPROVAL),'no_control_or_production_execution':True})
handoff=write(F/'inactive_helper2_CPU11_handoff_v1.json',{'helper_source':helper['new_source'],'CPU_source':cpu['new_source'],'exact_raw_inverse_ledger':proof,'proposed_configuration':config,'human_approval':pin(APPROVAL),'completed_CPU_proof_schema_required':{'scope':'ASR18_owned_two_logical_packet_one_receiver_transport_v1','accepted':True,'reviewer':'/root','helper_source':helper['new_source'],'CPU_source':cpu['new_source'],'human_approval':pin(APPROVAL),'maximum_logical_packets':2,'maximum_live_receiver_packets':1,'genuine_named_kernel_credit_and_pipe_flow_passed':True,'synthetic_packets_no_audio_model':True,'payload_lifetime_and_partial_receive_interleavings_passed':True},'finite_control_executed':False,'production_admitted':False})
print(json.dumps({'helper':helper['new_source'],'CPU':cpu['new_source'],'inverse':proof,'configuration':config,'handoff':handoff}))
