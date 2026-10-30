"""Inactive metadata/source materialization only. Does not import candidates or arm a stop."""
import ast, copy, difflib, hashlib, json, os
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).resolve().parent;OLD=F.parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');N=OLD.parent;A=N/'resident_RAM_training_actual_25'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def successor(old,new,changes):
 raw=old.read_bytes();text=raw.decode('utf-8');ledger=[]
 for a,b,count in changes:
  assert text.count(a)==count,(old.name,a,text.count(a),count)
  ledger.append({'old_literal':a,'new_literal':b,'expected_count':count});text=text.replace(a,b)
 ast.parse(text);inv=text
 for x in reversed(ledger):
  assert inv.count(x['new_literal'])==x['expected_count'],x['new_literal']
  inv=inv.replace(x['new_literal'],x['old_literal'])
 assert inv.encode('utf-8')==raw
 assert not new.exists();new.write_bytes(text.encode('utf-8'))
 diff=new.with_suffix('.diff')
 with diff.open('x',encoding='utf-8',newline='\n') as f:
  f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.replace('\r\n','\n').splitlines(True),fromfile=str(old),tofile=str(new))))
 return {'old':pin(old),'new':pin(new),'replacements':ledger,'newline_normalization':'NONE: raw literal replacements preserve every other byte',
  'entire_RAW_forward_and_inverse_exact':True,'diff':pin(diff)}
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
scope=F/'ROOT_source36_typed_transition_checkpoint_stop_inactive_preparation_scope_v1.json'
assert pin(scope)['sha256']=='d071e5b1f4eb97c00d23cb23fb0925b8b4213ee3218df2d8fa05725dafb838e7'
proof=F/'ROOT_one_postDEV_refresh_actual_finite_CPU_review_v1.json'
assert pin(proof)['sha256']=='2dc84d02f35210d839dc0504993638e676d67e4719e66a7c5fbd5c8a1923e532'
review=read(proof);assert review['accepted'] is True and review['reviewer']=='/root' and review['all_exact_metadata_identities_dead'] is True
proposed=F/'proposed_source36_configuration_pending_CPU_and_actual_stop_v1.json';configuration=read(proposed)
assert configuration['one_postDEV_gc_refresh_policy']['completed_CPU_review'] is None
assert configuration['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'] is None
configuration['one_postDEV_gc_refresh_policy']['completed_CPU_review']=pin(proof)
candidate=F/'candidate_source36_configuration_completed_CPU_pending_actual_stop_v1.json';write(candidate,configuration)
proper=read(N/'resident_RAM_original_interval170469_failure_actual_v1/production_configuration_source32_v1/configuration.json')
inverse=copy.deepcopy(configuration);del inverse['one_postDEV_gc_refresh_policy'];del inverse['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'];assert inverse==proper
people=[('native',3652,1792764340.2396317),('native_interpreter_launcher',31268,1792764340.232928),('wrapper',30732,1792764339.761604),
 ('CPU',31032,1792774698.5917125),('CPU_interpreter_launcher',40160,1792774698.5733004)]
runtime=[]
for role,pid,ct in people:
 process=psutil.Process(pid);assert abs(process.create_time()-ct)<.001
 runtime.append({'role':role,'pid':pid,'create_time':ct,'ppid':process.ppid()})
ready=B/'invocations/owner_000024/CPU_after_periodic_DEV_000176000/actual_CPU_ready.json'
assert read(ready)['producer_identity']['pid']==31032
progress=read(B/'progress/latest.json');old_driver=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v33.py')
assert progress['source']==old_driver and progress['producer']['pid']==3652 and not progress['optimizer_poisoned'] and not progress['schedule_poisoned']
assert progress['planned_native_updates']==401072 and progress['administrative_update_stop'] is None
intent={'kind':'ASR18_INACTIVE_one_postDEV_gc_refresh_checkpoint_transition_intent_v1','owned_session':65067,'old_owner_invocation':24,'new_owner_invocation':25,
 'branch':str(B),'old_control':str(A),'old_driver_source':old_driver,'old_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py'),
 'old_CPU_source':pin(S/'fastmss_resident_feature_stream_v13.py'),'old_CPU_ready':pin(ready),'expected_runtime_identities':runtime,
 'minimum_natural_saved_updates':max(174000,((progress['actual_optimizer_updates']//200)+1)*200),
 'human_approval':pin(OLD/'ROOT_direct_human_then_fix_authorization_v1.json'),'next_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py'),
 'next_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py'),'candidate_configuration':pin(candidate),
 'actual_CPU_review':pin(proof),'inactive_preparation_scope':pin(scope),
 'stop_mechanism':'Only original Monitor.check stop.json guard; no kill/hotpatch/args/env mutation',
 'stop_point':'Natural fullverified200 outside4000DEV; observed AND fresh progress.last_saved_updates==saved and emitted actual<=saved+20',
 'zero_tail_or_normal0_not_promised':True,'latest_actual_frontier_overrides_snapshot_at_closure':True,
 'optimizer_and_journal_frontiers_may_differ_by_one':True,'no_unpublished_optimizer_counter_promoted':True,
 'production_stop_authorized_by_this_intent':False,'original_monitor_must_join_clean_without_error_before_final_ROOT_transition_admission':True,
 'pre_stop_gates':['Actual separately scoped entire typed metadata proof accepted by ROOT','Exact current observer/intent/sealer ROOT pre-stop review'],
 'new_invocation_and_journal':25,'new_control':str(N/'resident_RAM_training_actual_26'),
 'long_validation':{'start':'actual_restored+20','end':'max(start+1600,nextscheduledDEV+1020)','all_checkpoints_and_real_DEV_and_pauses_included':True,'no_finite_training_stop':True,
 'engineering_acceptance':'Whole and ALL200 blocks, first400/final400 and two postDEV400; near original76.57 inclusive (~75+) for comparable nonDEV400; save-excluded first-final sustained decline<=5%; label DEV-containing comparison and no forecast/isolated-cause guarantee.'}}
intent_path=F/'inactive_source33_one_postDEV_refresh_checkpoint_transition_intent_v1.json';write(intent_path,intent)
observer=successor(OLD/'observe_fresh_signature_collect_natural_checkpoint_stop_v1.py',F/'observe_one_postDEV_refresh_natural_checkpoint_stop_v1.py',[
 ('ASR18_fresh_signature_collect_checkpoint_stop_v1','ASR18_one_postDEV_refresh_checkpoint_stop_v1',1),
 ("proof['scope']=='ASR18_fresh_metadata_signature_and_original_collect_observation_policy_v1'\r\n        and proof['signature_helper_source']==pin(F/'asr18_fresh_single_open_metadata_signature_v2.py')\r\n        and proof['integration_helper_source']==pin(F/'asr18_fresh_metadata_signature_scope_v2.py')\r\n        and proof['collect_observer_source']==pin(F/'asr18_observe_original_gc_collect_v2.py')\r\n        and proof['semantic_passed']is True and proof['candidate_timing_eligible_in_THIS_bounded_fixed_order']is True",
  "proof['scope']=='ASR18_one_first_periodic_DEV_quiescent_gc_refresh_policy_v1'\r\n        and proof['helper_source']==pin(F/'asr18_one_postDEV_gc_refresh_v2.py')\r\n        and proof['capture_source']==pin(F/'asr18_postDEV_refresh_state_capture_v1.py')",1),
 ('Completed exact signature/collect observation proof BEFORE intentional stop','Completed isolated CPython/METHOD_STUB proof BEFORE separately reviewed intentional stop',1),
 ("intent['owned_session']==72166 and intent['old_owner_invocation']==22\r\n        and intent['new_owner_invocation']==23","intent['owned_session']==65067 and intent['old_owner_invocation']==24\r\n        and intent['new_owner_invocation']==25",1),
 ('Same exact admitted source30 owner','Same exact admitted source33 owner',1),('invocations/owner_000022','invocations/owner_000024',1),
 ("F/'fresh_signature_collect_checkpoint_stop_observer_actual_01'","F/'one_postDEV_refresh_checkpoint_stop_observer_actual_01'",1),
 ("m['commit_available_bytes']>=2805306368","m['commit_available_bytes']>=5055306368",1),
 ('Full prospective512MiB plus256MiB and original floors','Full225 cushion plus prospective512MiB and FULLprefetch/original floors',1),
 ("progress['producer']['pid']==40012 and abs(progress['producer']['create_time']-1792743161.6231525)<.001", "progress['producer']['pid']==3652 and abs(progress['producer']['create_time']-1792764340.2396317)<.001",1),
 ('ASR18_reviewed_fresh_signature_collect_checkpoint_cooperative_stop_v1','ASR18_reviewed_one_postDEV_refresh_checkpoint_cooperative_stop_v1',1)])
sealer=successor(OLD/'seal_actual_source30_fresh_signature_collect_checkpoint_transition_v1.py',F/'seal_actual_source33_one_postDEV_refresh_checkpoint_transition_v1.py',[
 ('source30 stop closure','source33 stop closure',1),('original22 STAT.','original24 STAT.',1),
 ('from asr18_fresh_signature_collect_checkpoint_transition_v1 import validate_intentional_optimizer_journal_gap','from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import validate_intentional_optimizer_journal_gap',1),
 ('ASR18_fresh_signature_collect_checkpoint_stop_v1','ASR18_one_postDEV_refresh_checkpoint_stop_v1',1),
 ("m['commit_available_bytes']>=2805306368","m['commit_available_bytes']>=5055306368",1),
 ('Full prospective512MiB and256MiB original admission','Full225 cushion plus prospective512MiB and FULLprefetch/original admission',1),
 ("tool['owned_session']==72166","tool['owned_session']==65067",1),('Real consumed OWNED72166 TOOL1','Real consumed OWNED65067 TOOL1',1),
 ('invocations/owner_000022','invocations/owner_000024',1),('commits_000022.jsonl','commits_000024.jsonl',1),
 ('Current source30 journal22 only','Current source33 journal24 only',1),
 ("F/'actual_source30_fresh_signature_collect_checkpoint_transition_01'","F/'actual_source33_one_postDEV_refresh_checkpoint_transition_01'",1),
 ('native_000022.log','native_000024.log',1),("evidence['original_monitor_completion']","evidence['monitor_completion']",1),
 ("'kind':'ASR18_actual_source30_checkpoint_fresh_signature_collect_transition_v1','owned_session':72166,\r\n        'old_owner_invocation':22,'new_owner_invocation':23", "'kind':'ASR18_actual_source33_checkpoint_one_postDEV_gc_refresh_transition_v1','owned_session':65067,\r\n        'old_owner_invocation':24,'new_owner_invocation':25",1)])
ledger=F/'source33_to36_checkpoint_observer_sealer_exact_RAW_inverse_v1.json'
write(ledger,{'observer':observer,'sealer':sealer,'original_optimizer_journal_gap_checker_byte_unchanged':True,
 'stop_observer_admission_commit_strengthened_to_existing_full225_plus512':True,'no_actual_stop_or_candidate_import':True})
handoff=F/'inactive_source36_checkpoint_stop_source_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_source36_checkpoint_stop_handoff_v1','preparation_scope':pin(scope),'intent':pin(intent_path),
 'observer_source':observer['new'],'closure_sealer_source':sealer['new'],'transition_validator':pin(S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py'),
 'candidate_configuration':pin(candidate),'actual_CPU_review':pin(proof),'human_approval':intent['human_approval'],'old_driver_source':old_driver,
 'inverse':pin(ledger),'source_transition_handoff':pin(F/'inactive_source36_one_postDEV_refresh_source_handoff_v1.json'),
 'all_five_expected_runtime_identities':runtime,'pre_stop_required_scope':'ASR18_one_postDEV_refresh_checkpoint_stop_v1',
 'production_stop_admitted':False,'production_launch_admitted':False,'actual_stop_slot_is_None':True,
 'typed_metadata_proof_before_stop_required':True,'actual_clean_Monitor_and_supported_stop_ROOT_acceptance_before_launch_required':True,
 'all_original_science_resource_GC_perupdate_and_native_bytes_unchanged':True,
 'future_runtime7':'Exactly seven original binding KINDS; BOTH stream/source pin actual future materialized config; pointer chosen only after actual accepted stop.'})
print(json.dumps({'handoff':pin(handoff),'intent':pin(intent_path),'candidate':pin(candidate),'observer':observer['new'],'sealer':sealer['new'],
 'inverse':pin(ledger),'preparation_snapshot_updates':progress['actual_optimizer_updates'],'minimum_natural_saved':intent['minimum_natural_saved_updates']}),flush=True)
