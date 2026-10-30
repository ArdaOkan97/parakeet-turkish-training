"""Inactive next checkpoint transition only; never writes a production stop."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).parent;OLD=F.parent/'resident_RAM_two_logical_packet_lookahead_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');A=F.parent/'resident_RAM_training_actual_22'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def successor(old,new,replacements):
 raw=old.read_bytes();t=raw.decode().replace('\r\n','\n');newline='\r\n'if b'\r\n'in raw else'\n'
 assert t.replace('\n',newline).encode()==raw
 edits=[]
 for a,b,n in replacements:
  assert t.count(a)==n,(a,t.count(a),n);t=t.replace(a,b);edits.append(dict(old_literal=a,new_literal=b,expected_count=n))
 ast.parse(t);newraw=t.replace('\n',newline).encode();assert not new.exists();new.write_bytes(newraw)
 inverse=t
 for e in reversed(edits):
  assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.replace('\n',newline).encode()==raw
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
 return dict(old=pin(old),new=pin(new),replacements=edits,newline_policy='Original '+repr(newline)+' restored byte-exact',full_old_raw_inverse=True,diff=pin(diff))

proof=F/'ROOT_canonical_prepared_archive_actual_CPU_review_v1.json';review=read(proof)
assert review['accepted']and review['reviewer']=='/root'and review['scope']=='ASR18_canonical_bytes_prepared_archive_policy_v1'
assert review['helper_source']==pin(F/'asr18_canonical_prepared_archive_v1.py')and review['all_exact_metadata_identities_dead']is True
base=F/'proposed_source30_configuration_pending_proofs_and_actual_stop_v1.json';candidate=read(base)
assert candidate['prepared_metadata_archive_representation_policy']['completed_CPU_review']is None
candidate['prepared_metadata_archive_representation_policy']['completed_CPU_review']=pin(proof)
cp=F/'candidate_source30_configuration_completed_CPU_pending_actual_stop_v1.json';write(cp,candidate)

ready=B/'invocations/owner_000021/CPU_after_periodic_DEV_000160000/actual_CPU_ready.json'
people=[('native',37836,1792733651.0335317),('native_interpreter_launcher',4344,1792733651.0267332),('wrapper',40576,1792733650.5702133),('CPU',3328,1792736431.756786),('CPU_interpreter_launcher',36296,1792736431.7388184)]
runtime=[]
for role,pid,ctime in people:
 p=psutil.Process(pid);assert abs(p.create_time()-ctime)<.001;runtime.append(dict(role=role,pid=pid,create_time=ctime,ppid=p.ppid()))
progress=read(B/'progress/latest.json');assert progress['source']==pin(R/'run_ASR18_BPE9223_resident_RAM_training_v29.py')and progress['producer']['pid']==37836 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
intent={'kind':'ASR18_INACTIVE_canonical_archive_checkpoint_transition_intent_v1','owned_session':87302,'old_owner_invocation':21,'new_owner_invocation':22,
 'branch':str(B),'old_control':str(A),'old_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v29.py'),'old_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v29.py'),
 'old_CPU_source':pin(S/'fastmss_resident_feature_stream_v11.py'),'old_CPU_ready':pin(ready),'expected_runtime_identities':runtime,
 'minimum_natural_saved_updates':max(163000,((progress['actual_optimizer_updates']//200)+1)*200),
 'human_approval':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),'next_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v30.py'),'next_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py'),
 'candidate_configuration':pin(cp),'actual_CPU_review':pin(proof),'stop_mechanism':'Only original Monitor.check stop.json guard; no kill/hotpatch/args/env mutation',
 'stop_point':'Next natural fullverified200 outside4000DEV; observed AND fresh progress.last_saved_updates==saved and emitted actual<=saved+20',
 'zero_tail_or_normal0_not_promised':True,'latest_actual_frontier_overrides_snapshot_at_closure':True,
 'optimizer_and_journal_frontiers_may_differ_by_one':True,'no_unpublished_optimizer_counter_promoted':True,'production_stop_authorized_by_this_intent':False,
 'long_validation':{'start':'actual_restored+20','end':'max(start+1600,next4000boundary+220)','all_checkpoints_and_real_DEV_included':True,'no_finite_training_stop':True}}
ip=F/'inactive_source29_canonical_archive_checkpoint_transition_intent_v1.json';write(ip,intent)
observer=successor(OLD/'observe_approved_natural_checkpoint_stop_v2.py',F/'observe_canonical_archive_natural_checkpoint_stop_v1.py',[
 ('ASR18_approved_two_logical_packet_checkpoint_stop_v1','ASR18_canonical_archive_checkpoint_stop_v1',1),
 ("proof['scope']=='ASR18_owned_two_logical_packet_one_receiver_transport_v1'\n        and proof['payload_lifetime_and_partial_receive_interleavings_passed']is True", "proof['scope']=='ASR18_canonical_bytes_prepared_archive_policy_v1'\n        and proof['helper_source']==pin(F/'asr18_canonical_prepared_archive_v1.py')\n        and proof['all_exact_metadata_identities_dead']is True",1),
 ('Completed genuine scoped transport proof BEFORE intentional stop','Completed exact codec representation proof BEFORE intentional stop',1),
 ("intent['owned_session']==56620 and intent['old_owner_invocation']==20\n        and intent['new_owner_invocation']==21", "intent['owned_session']==87302 and intent['old_owner_invocation']==21\n        and intent['new_owner_invocation']==22",1),
 ('Same exact admitted source27 owner','Same exact admitted source29 owner',1),
 ('invocations/owner_000020','invocations/owner_000021',1),
 ("F/'approved_checkpoint_stop_observer_actual_01'", "F/'canonical_archive_checkpoint_stop_observer_actual_01'",1),
 ("progress['producer']['pid']==9704 and abs(progress['producer']['create_time']-1792727133.7017913)<.001", "progress['producer']['pid']==37836 and abs(progress['producer']['create_time']-1792733651.0335317)<.001",1),
 ('ASR18_reviewed_approved_lookahead_checkpoint_cooperative_stop_v1','ASR18_reviewed_canonical_archive_checkpoint_cooperative_stop_v1',1)])
sealer=successor(OLD/'seal_actual_approved_source27_checkpoint_transition_v2.py',F/'seal_actual_source29_canonical_archive_checkpoint_transition_v1.py',[
 ('source27 stop closure','source29 stop closure',1),
 ('original20 STAT. Never reads old20 prefix/whole or old18/19/tensor/audio.','original21 STAT. Never reads any original journal prefix/whole/tensor/audio.',1),
 ('from asr18_approved_lookahead_checkpoint_transition_v3 import validate_intentional_optimizer_journal_gap','from asr18_canonical_archive_checkpoint_transition_v1 import validate_intentional_optimizer_journal_gap',1),
 ('ASR18_approved_two_logical_packet_checkpoint_stop_v1','ASR18_canonical_archive_checkpoint_stop_v1',1),
 ("tool['owned_session']==56620","tool['owned_session']==87302",1),
 ('Real consumed OWNED56620 TOOL1','Real consumed OWNED87302 TOOL1',1),
 ('invocations/owner_000020','invocations/owner_000021',1),
 ('commits_000020.jsonl','commits_000021.jsonl',1),
 ('Current source27 journal20 only','Current source29 journal21 only',1),
 ("F/'actual_approved_source27_checkpoint_transition_01'", "F/'actual_source29_canonical_archive_checkpoint_transition_01'",1),
 ('native_000020.log','native_000021.log',1),
 ("'kind':'ASR18_actual_approved_source27_checkpoint_lookahead_transition_v1','owned_session':56620,\n        'old_owner_invocation':20,'new_owner_invocation':21", "'kind':'ASR18_actual_source29_checkpoint_canonical_archive_transition_v1','owned_session':87302,\n        'old_owner_invocation':21,'new_owner_invocation':22",1)])
ledger=F/'source29_canonical_archive_stop_observer_sealer_exact_raw_inverse_v1.json';write(ledger,{'observer':observer,'sealer':sealer,'both_original_guards_and_gap_function_unchanged':True})
handoff={'kind':'ASR18_INACTIVE_source30_canonical_archive_pre_stop_handoff_v1','intent':pin(ip),'observer_source':observer['new'],'closure_sealer_source':sealer['new'],'transition_validator':pin(S/'asr18_canonical_archive_checkpoint_transition_v1.py'),'candidate_configuration':pin(cp),'actual_CPU_review':pin(proof),'helper_source':pin(F/'asr18_canonical_prepared_archive_v1.py'),'human_approval':intent['human_approval'],'old_driver_source':intent['old_driver_source'],'inverse':pin(ledger),'source_transition_handoff':pin(F/'inactive_source30_canonical_archive_transition_handoff_v1.json'),
 'pre_stop_required_scope':'ASR18_canonical_archive_checkpoint_stop_v1','all_five_expected_runtime_identities':runtime,'production_stop_admitted':False,
 'next_typed_proof':'After actual stop: exact actual validator, ENTIRE journal_factory/postrestore branch dict and wrapper forwarding; gap0/1 method-stub strict trace negatives. Original21 STAT +bounded preserved suffix COPY only; no restore/copy/model/audio/GPU/kernel.',
 'future_execution':'Only separate final ROOT exact binding; full401072/adminNone; no source29 session consumption until observer exact deaths'}
hp=F/'inactive_source30_canonical_archive_checkpoint_stop_handoff_v1.json';write(hp,handoff)
print(json.dumps({'handoff':pin(hp),'intent':pin(ip),'observer':observer['new'],'sealer':sealer['new'],'inverse':pin(ledger),'candidate':pin(cp)}))
