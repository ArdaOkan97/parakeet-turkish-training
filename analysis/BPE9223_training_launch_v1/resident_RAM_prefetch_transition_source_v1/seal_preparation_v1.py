"""Lightweight inactive configuration and real recorded-metadata controls only."""
import ast, copy, hashlib, importlib, json, sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
sys.path[:0]=[str(S),str(R)];psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 assert not Path(p).exists(),str(p)
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
old_config=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_DEV_frontend_source_v1/configuration.json')
reviewed_config=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_scalar_prefetch_integration_source_v1/configuration.json')
old=read(old_config);candidate=read(reviewed_config);comparison=copy.deepcopy(candidate);comparison.pop('private_scalar_prefetch_policy');assert comparison==old
native=psutil.Process(2956);assert abs(native.create_time()-1792395250.2468245)<.001
wrapper=psutil.Process(42552);assert 'autocontinue_v4.py' in ' '.join(wrapper.cmdline())
processes=[wrapper]
parent=wrapper.parent()
if 'autocontinue_v4.py' in ' '.join(parent.cmdline()):processes.insert(0,parent)
processes.extend(wrapper.children(recursive=True))
assert {2956,20396,19160,33804}.issubset({p.pid for p in processes})
captured=[who(p)for p in processes];latest=read(B/'latest_checkpoint.json')
intent={'kind':'ASR18_authorized_private_prefetch_source_transition_intent_v1',
 'authorization':'Human explicitly requests enabling prefetch now with latest-checkpoint exact state continuation; no resets.',
 'branch':str(B),'old_owned_session':45407,'old_native':who(native),'old_wrapper':who(wrapper),
 'captured_identities':captured,'old_invocation':str(B/'invocations/owner_000005'),
 'old_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v4.py'),
 'old_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v4.py'),
 'old_configuration':pin(old_config),'old_invocation_configuration':pin(B/'invocations/owner_000005/configuration.json'),
 'old_boot':pin(B/'invocations/owner_000005/boot.json'),
 'old_native_log':'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_training_actual_04\\native_000005.log',
 'minimum_saved_updates':latest['committed_updates']+200,'checkpoint_interval_updates':200,
 'stop_mechanism':'Existing invocation stop.json guard, requested only upon a fresh fully verified checkpoint publication.',
 'no_tail_assumed_before_actual_closure':False,'normal0_relabel_permitted':False,
 'whole_original_and_any_positive_tail_preserved':True,'new_scientific_updates_reset':False}
intent_path=N/'transition_intent.json';write(intent_path,intent)
candidate['resident_source_transition_intent']=pin(intent_path);write(N/'configuration.json',candidate)
module=importlib.import_module('asr18_resident_source_transition_v1')
pointer=read(latest['pointer']['path']);saved=pointer['journal_prefix']
whole_saved={'path':saved['path'],'sha256':saved['sha256'],'bytes':saved['offset']}
assert module.check_saved_relation(saved,whole_saved,pointer['committed_updates'],pointer['committed_updates'])==0
negative=[]
for label,changed,counter in [('wrong_prefix_hash',dict(whole_saved,sha256='0'*64),pointer['committed_updates']),
 ('short_journal',dict(whole_saved,bytes=saved['offset']-1),pointer['committed_updates']),
 ('unsaved_counter_without_tail',whole_saved,pointer['committed_updates']+1)]:
 try:module.check_saved_relation(saved,changed,counter,pointer['committed_updates'])
 except ValueError:negative.append(label)
 else:raise AssertionError(label)
historical=read(old['prior_resident_RAM_headroom_failure']['path'])
actual_tail=module.check_saved_relation(historical['saved_journal_prefix'],historical['whole_journal_including_preserved_tail'],
 historical['actual_optimizer_updates_before_failure'],historical['saved_optimizer_updates'])
assert actual_tail==historical['preserved_unsaved_tail']['bytes']
names=['commit_RAM_transaction','private_successor_request','Monitor','evaluate_coherent_DEV','record_periodic','writer','stage','admit']
trees=[]
for version in(5,6):
 tree=ast.parse((R/f'run_ASR18_BPE9223_resident_RAM_training_v{version}.py').read_text(encoding='utf-8'))
 trees.append({node.name:ast.dump(node,include_attributes=False)for node in ast.walk(tree)if isinstance(node,(ast.FunctionDef,ast.ClassDef))})
for name in names:assert trees[0][name]==trees[1][name],name
for source in (R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v6.py',S/'asr18_resident_source_transition_v1.py'):
 ast.parse(source.read_text(encoding='utf-8'))
control={'kind':'ASR18_prefetch_transition_inactive_metadata_control_v1','passed':True,
 'actual_latest_checkpoint_at_preparation':latest,'actual_closed_historical_tail_relation_bytes':actual_tail,
 'current_saved_prefix_relation_exact':True,'metadata_negative_ValueErrors':negative,
 'exact_AST_unchanged_definitions':names,'same_scientific_config_after_removing_two_operational_fields':True,
 'old_family_still_live_and_unchanged':True,'GPU_audio_model_or_training_job':False,'stop_file_written':False}
write(N/'metadata_control_report.json',control)
handoff={'kind':'ASR18_prefetch_source_transition_preparation_handoff_v1','production_transition_or_launch_performed':False,
 'driver':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v6.py'),'wrapper':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v6.py'),
 'transition_validator_and_watcher':pin(S/'asr18_resident_source_transition_v1.py'),'intent':pin(intent_path),
 'configuration':pin(N/'configuration.json'),'source5_reused_review':pin('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/ROOT_integrated_private_scalar_prefetch_inactive_source_read_review_v1.json'),
 'actual_CPU_equality_review':pin('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/ROOT_private_scalar_prefetch_actual_CPU_control_review_v1.json'),
 'metadata_control':pin(N/'metadata_control_report.json'),'driver_diff':pin(N/'driver5_to6.diff.txt'),'wrapper_diff':pin(N/'wrapper5_to6.diff.txt'),
 'configuration_delta_from_source4':['private_scalar_prefetch_policy','resident_source_transition_intent'],
 'additional_prefetch_reservation_bytes':268435456,'old_guard_and_numeric_limits_unchanged':True,
 'actual_transition_closure_still_required':True,'watcher_review_required_fields':['accepted=True','reviewer=/root','cooperative_transition_authorized=True','transition_source=exact record3','transition_intent=exact record3'],
 'watcher_command':['CPU_python','-X','utf8',str(S/'asr18_resident_source_transition_v1.py'),'--intent',str(intent_path),'--review','ROOT_review_path','--output',str(N/'actual_transition_01')],
 'actual_resume_arguments_after_closure':['--source-transition-resume','actual_transition_closure.json','--start-invocation','6'],
 'timing_evidence_path_template':str(B/'invocations/owner_000006/private_metadata_window_timings.jsonl'),
 'timing_scope':'Every real20-update window: worker preparation, main adoption+submit, end wait/fallback, native seconds, audio duration, checkpoint-inclusive window wall. Startup/cache warmth separately limited; no permanent throughput claim.'}
write(N/'source_handoff.json',handoff);print(json.dumps({'handoff':pin(N/'source_handoff.json'),'intent':pin(intent_path),'configuration':pin(N/'configuration.json'),'metadata_control':pin(N/'metadata_control_report.json')}))
