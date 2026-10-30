import ast,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');F=N/'resident_RAM_two_logical_packet_lookahead_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
for name in ['observe_approved_natural_checkpoint_stop_v1.py','seal_actual_approved_source27_checkpoint_transition_v1.py']:
 ast.parse((F/name).read_text(encoding='utf-8-sig'))
latest=json.loads((B/'latest_checkpoint.json').read_text());minimum=latest['committed_updates']+200
runtime=[{'role':role,'pid':pid,'create_time':ctime,'ppid':ppid}for role,pid,ctime,ppid in [
 ('native',9704,1792727133.7017913,32860),('native_interpreter_launcher',32860,1792727133.694484,12328),
 ('wrapper',12328,1792727133.3462756,41928),('CPU',4392,1792727182.5415392,39368),
 ('CPU_interpreter_launcher',39368,1792727182.5201604,9704)]]
intent={'kind':'ASR18_INACTIVE_approved_two_logical_packet_checkpoint_transition_intent_v1',
 'owned_session':56620,'old_owner_invocation':20,'new_owner_invocation':21,'branch':str(B),'old_control':str(N/'resident_RAM_training_actual_21'),
 'old_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v27.py'),
 'old_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v27.py'),
 'old_CPU_source':pin(S/'fastmss_resident_feature_stream_v10.py'),
 'old_CPU_ready':pin(B/'invocations/owner_000020/CPU/actual_CPU_ready.json'),
 'expected_runtime_identities':runtime,'minimum_natural_saved_updates':minimum,
 'human_approval':pin(N/'resident_RAM_source27_speed_review_v1/ROOT_direct_human_lookahead_and_checkpoint_restart_approval_v1.json'),
 'next_driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v28.py'),
 'next_wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v28.py'),
 'stop_mechanism':'Only original Monitor.check stop.json guard; no native-update kill or source/args/env hotpatch',
 'stop_point':'First next natural fullyverified200 publication, outside4000 DEV, latest emitted progress<=saved+20',
 'zero_tail_or_normal0_not_promised':True,'latest_actual_frontier_overrides_snapshot_at_closure':True,
 'stage_authorization':'INACTIVE intent; observer execution still requires separate ROOT pre-stop receipt after genuine CPU acceptance'}
write(F/'inactive_approved_source27_checkpoint_transition_intent_v1.json',intent)
handoff={'kind':'ASR18_INACTIVE_approved_checkpoint_observer_closure_sources_handoff_v1',
 'transition_intent':pin(F/'inactive_approved_source27_checkpoint_transition_intent_v1.json'),
 'observer_source':pin(F/'observe_approved_natural_checkpoint_stop_v1.py'),
 'closure_sealer_source':pin(F/'seal_actual_approved_source27_checkpoint_transition_v1.py'),
 'driver_source':intent['next_driver_source'],'wrapper_source':intent['next_wrapper_source'],
 'source28_review':pin(F/'ROOT_independent_source28_lookahead_transition_source_review_v1.json'),
 'human_approval':intent['human_approval'],'actual_CPU_closure':pin(F/'finite_transport_actual_CPU_02_exact_death_closure.json'),
 'requested_ROOT_scope':'ASR18_approved_two_logical_packet_checkpoint_stop_v1',
 'observer_limits':{'internal_wall_seconds':900,'private_bytes':536870912,'full_prospective_private_bytes':536870912,
 'additional_prefetch_bytes':268435456,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,'E_free_bytes':100000000000},
 'closure_limits':{'internal_wall_seconds':180,'private_bytes':536870912,'full_prospective_private_bytes':536870912,
 'additional_prefetch_bytes':268435456,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,'E_free_bytes':100000000000},
 'process_local':'Read E storage_config + dot-source use_storage includingTORCH_EXTENSIONS_DIR; CPU-only CUDA-1/all4 caps1/UTF8-B/BelowNormal',
 'actual_observer_executed':False,'stop_written':False,'production_launch_admitted':False,
 'remaining':'ROOT completed CPU acceptance plus exact pre-stop receipt; actual owned56620 wait outcome/known deaths; actual closure+tail seal; final config and entire selector metadata control; ROOT final execution binding'}
write(F/'inactive_checkpoint_stop_observer_closure_handoff_v1.json',handoff)
print(json.dumps({'intent':pin(F/'inactive_approved_source27_checkpoint_transition_intent_v1.json'),
 'observer':handoff['observer_source'],'sealer':handoff['closure_sealer_source'],
 'handoff':pin(F/'inactive_checkpoint_stop_observer_closure_handoff_v1.json'),'minimum_natural_saved':minimum,'actual_stop_executed':False}))
