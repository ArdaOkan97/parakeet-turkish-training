"""Freeze inactive intent/observer/sealer only, no stop or production mutation."""
import ast,copy,difflib,hashlib,json,os
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';F=Path(__file__).parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');A=F.parent/'resident_RAM_training_actual_23'
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def successor(old,new,changes):
 raw=old.read_bytes();t=raw.decode().replace('\r\n','\n');nl='\r\n'if b'\r\n'in raw else'\n';assert t.replace('\n',nl).encode()==raw;l=[]
 for a,b,n in changes:
  assert t.count(a)==n,(a,t.count(a),n);l.append({'old_literal':a,'new_literal':b,'expected_count':n});t=t.replace(a,b)
 ast.parse(t);inv=t
 for e in reversed(l):
  assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'];inv=inv.replace(e['new_literal'],e['old_literal'])
 assert inv.replace('\n',nl).encode()==raw
 assert not new.exists();new.write_bytes(t.replace('\n',nl).encode())
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8',newline='\n')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
 return {'old':pin(old),'new':pin(new),'replacements':l,'original_newline_restored':repr(nl),'full_raw_inverse_exact':True,'diff':pin(diff)}
proof=F/'ROOT_fresh_signature_original_collect_actual_CPU_review_v1.json';review=read(proof)
assert review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_fresh_metadata_signature_and_original_collect_observation_policy_v1'
assert review['all_exact_metadata_identities_dead']is True and review['semantic_passed']is True and review['candidate_timing_eligible_in_THIS_bounded_fixed_order']is True
configuration=read(F/'proposed_source31_configuration_pending_ROOT_CPU_and_actual_stop_v1.json')
assert configuration['fresh_metadata_signature_and_collect_observation_policy']['completed_CPU_review']is None
configuration['fresh_metadata_signature_and_collect_observation_policy']['completed_CPU_review']=pin(proof)
cp=F/'candidate_source31_configuration_completed_CPU_pending_actual_stop_v1.json';write(cp,configuration)
ready=B/'invocations/owner_000022/CPU_after_periodic_DEV_000164000/actual_CPU_ready.json'
people=[('native',40012,1792743161.6231525),('native_interpreter_launcher',20284,1792743161.616002),('wrapper',39000,1792743161.2289293),('CPU',31044,1792745224.819755),('CPU_interpreter_launcher',25828,1792745224.8020935)]
runtime=[]
for role,pid,ct in people:
 p=psutil.Process(pid);assert abs(p.create_time()-ct)<.001;runtime.append({'role':role,'pid':pid,'create_time':ct,'ppid':p.ppid()})
progress=read(B/'progress/latest.json');assert progress['source']==pin(R/'run_ASR18_BPE9223_resident_RAM_training_v30.py')and progress['producer']['pid']==40012 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
intent={'kind':'ASR18_INACTIVE_fresh_signature_collect_checkpoint_transition_intent_v1','owned_session':72166,'old_owner_invocation':22,'new_owner_invocation':23,
 'branch':str(B),'old_control':str(A),'old_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v30.py'),'old_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py'),
 'old_CPU_source':pin(S/'fastmss_resident_feature_stream_v11.py'),'old_CPU_ready':pin(ready),'expected_runtime_identities':runtime,
 'minimum_natural_saved_updates':max(167400,((progress['actual_optimizer_updates']//200)+1)*200),
 'human_approval':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),'next_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v31.py'),'next_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v31.py'),
 'candidate_configuration':pin(cp),'actual_CPU_review':pin(proof),'stop_mechanism':'Only original Monitor.check stop.json guard; no kill/hotpatch/args/env mutation',
 'stop_point':'Natural fullverified200 outside4000DEV; observed AND fresh progress.last_saved_updates==saved and emitted actual<=saved+20',
 'zero_tail_or_normal0_not_promised':True,'latest_actual_frontier_overrides_snapshot_at_closure':True,
 'optimizer_and_journal_frontiers_may_differ_by_one':True,'no_unpublished_optimizer_counter_promoted':True,'production_stop_authorized_by_this_intent':False,
 'original_monitor_must_join_clean_without_error_before_final_ROOT_transition_admission':True,
 'long_validation':{'start':'actual_restored+20','end':'max(start+1600,nextscheduledDEV+1020)','all_checkpoints_and_real_DEV_and_pauses_included':True,'no_finite_training_stop':True,
 'engineering_acceptance':'Report full interval, first400/final400 and two postDEV400 blocks; near original76.57 inclusive (~75+), save-excluded first-final decline<=5%, no safeguards/science/state/reader regression. No forecast or isolated-cause guarantee.'}}
ip=F/'inactive_source30_fresh_signature_collect_checkpoint_transition_intent_v1.json';write(ip,intent)
observer=successor(F/'observe_canonical_archive_natural_checkpoint_stop_v1.py',F/'observe_fresh_signature_collect_natural_checkpoint_stop_v1.py',[
 ('ASR18_canonical_archive_checkpoint_stop_v1','ASR18_fresh_signature_collect_checkpoint_stop_v1',1),
 ("proof['scope']=='ASR18_canonical_bytes_prepared_archive_policy_v1'\n        and proof['helper_source']==pin(F/'asr18_canonical_prepared_archive_v1.py')", "proof['scope']=='ASR18_fresh_metadata_signature_and_original_collect_observation_policy_v1'\n        and proof['signature_helper_source']==pin(F/'asr18_fresh_single_open_metadata_signature_v2.py')\n        and proof['integration_helper_source']==pin(F/'asr18_fresh_metadata_signature_scope_v2.py')\n        and proof['collect_observer_source']==pin(F/'asr18_observe_original_gc_collect_v2.py')\n        and proof['semantic_passed']is True and proof['candidate_timing_eligible_in_THIS_bounded_fixed_order']is True",1),
 ('Completed exact codec representation proof BEFORE intentional stop','Completed exact signature/collect observation proof BEFORE intentional stop',1),
 ("intent['owned_session']==87302 and intent['old_owner_invocation']==21\n        and intent['new_owner_invocation']==22", "intent['owned_session']==72166 and intent['old_owner_invocation']==22\n        and intent['new_owner_invocation']==23",1),
 ('Same exact admitted source29 owner','Same exact admitted source30 owner',1),
 ('invocations/owner_000021','invocations/owner_000022',1),
 ("F/'canonical_archive_checkpoint_stop_observer_actual_01'", "F/'fresh_signature_collect_checkpoint_stop_observer_actual_01'",1),
 ("progress['producer']['pid']==37836 and abs(progress['producer']['create_time']-1792733651.0335317)<.001", "progress['producer']['pid']==40012 and abs(progress['producer']['create_time']-1792743161.6231525)<.001",1),
 ('ASR18_reviewed_canonical_archive_checkpoint_cooperative_stop_v1','ASR18_reviewed_fresh_signature_collect_checkpoint_cooperative_stop_v1',1)])
sealer=successor(F/'seal_actual_source29_canonical_archive_checkpoint_transition_v1.py',F/'seal_actual_source30_fresh_signature_collect_checkpoint_transition_v1.py',[
 ('source29 stop closure','source30 stop closure',1),
 ('original21 STAT.','original22 STAT.',1),
 ('from asr18_canonical_archive_checkpoint_transition_v1 import validate_intentional_optimizer_journal_gap','from asr18_fresh_signature_collect_checkpoint_transition_v1 import validate_intentional_optimizer_journal_gap',1),
 ('ASR18_canonical_archive_checkpoint_stop_v1','ASR18_fresh_signature_collect_checkpoint_stop_v1',1),
 ("tool['owned_session']==87302", "tool['owned_session']==72166",1),
 ('Real consumed OWNED87302 TOOL1','Real consumed OWNED72166 TOOL1',1),
 ('invocations/owner_000021','invocations/owner_000022',1),
 ('commits_000021.jsonl','commits_000022.jsonl',1),
 ('Current source29 journal21 only','Current source30 journal22 only',1),
 ("F/'actual_source29_canonical_archive_checkpoint_transition_01'", "F/'actual_source30_fresh_signature_collect_checkpoint_transition_01'",1),
 ('native_000021.log','native_000022.log',1),
 ("'kind':'ASR18_actual_source29_checkpoint_canonical_archive_transition_v1','owned_session':87302,\n        'old_owner_invocation':21,'new_owner_invocation':22", "'kind':'ASR18_actual_source30_checkpoint_fresh_signature_collect_transition_v1','owned_session':72166,\n        'old_owner_invocation':22,'new_owner_invocation':23",1)])
ledger=F/'source30_fresh_signature_collect_stop_observer_sealer_exact_raw_inverse_v1.json';write(ledger,{'observer':observer,'sealer':sealer,'both_original_guards_and_gap_function_unchanged':True})
hp=F/'inactive_source31_fresh_signature_collect_checkpoint_stop_handoff_v1.json'
write(hp,{'kind':'ASR18_INACTIVE_source31_fresh_signature_collect_pre_stop_handoff_v1','intent':pin(ip),'observer_source':observer['new'],'closure_sealer_source':sealer['new'],
 'transition_validator':pin(S/'asr18_fresh_signature_collect_checkpoint_transition_v1.py'),'candidate_configuration':pin(cp),'actual_CPU_review':pin(proof),
 'human_approval':intent['human_approval'],'old_driver_source':intent['old_driver_source'],'inverse':pin(ledger),
 'source_transition_handoff':pin(F/'inactive_source31_fresh_signature_collect_transition_handoff_v1.json'),
 'typed_fixture_handoff':pin(F/'inactive_source31_entire_selector_metadata_fixture_handoff_v1.json'),
 'pre_stop_required_scope':'ASR18_fresh_signature_collect_checkpoint_stop_v1','all_five_expected_runtime_identities':runtime,
 'production_stop_admitted':False,'production_launch_admitted':False,
 'next_typed_proof':'Actual closure/config complete real journal_factory and postrestore dictionary and wrapper entry+forwarding. Gap0/1 method-stub strict trace negatives; original22 STAT +bounded preserved suffix copy only.',
 'collect_observation_scope_note':'Each per20 scalar window is all explicit collect calls since prior emission, including intervening DEV/checkpoint calls if present. Concurrent/reentrant wall durations may overlap; not CPU or lost time.',
 'future_execution':'Only separate final ROOT exact binding, full401072/adminNone; do not consume72166 before observer confirms exact deaths; do not treat intentional TOOL1 as normal0.'})
print(json.dumps({'handoff':pin(hp),'intent':pin(ip),'candidate':pin(cp),'observer':observer['new'],'sealer':sealer['new'],'ledger':pin(ledger),'snapshot_actual':progress['actual_optimizer_updates'],'minimum_natural_saved':intent['minimum_natural_saved_updates']}))
