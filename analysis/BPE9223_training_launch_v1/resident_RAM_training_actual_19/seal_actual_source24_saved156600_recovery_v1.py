"""Routine compact recovery observer; ONLY fixed138 NEW journal bytes, no old17/payload/audio."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=Path(__file__).parent;I=B/'invocations/owner_000018'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def snapshot(source,destination):
 with Path(destination).open('xb')as f:f.write(Path(source).read_bytes())
 return read(destination)
def guard():
 s=sample_memory(os.getpid(),'source24_saved156600_routine_metadata_seal');m=s['system_memory']
 require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original finite metadata512MiB/180s/full256MiB/headroom/E100GB')
 return s
initial=guard();m=initial['system_memory']
require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective512MiB metadata allowance')
require(not(I/'actual_failure.json').exists(),'No actual new failure')
progress=snapshot(B/'progress/latest.json',A/'actual_post156600_positive_progress.json')
require(progress['producer']=={'pid':42304,'create_time':1792716760.1698885,'ppid':38476}
 and progress['actual_optimizer_updates']>156600 and progress['last_saved_updates']>=156600
 and progress['native_updates_this_process']==progress['actual_optimizer_updates']-156400
 and progress['optimizer_poisoned'] is False and progress['schedule_poisoned'] is False
 and progress['planned_native_updates']==401072 and progress['administrative_update_stop'] is None,'Real post156600 positive continuation/full401072, no finite stop')
delta=progress['native_updates_this_process']
require(progress['clocks']==[220650+delta,158400+delta,218650+delta],'All3 clocks advance exactly from restored156400')
restore=read(I/'actual_parent_restore.json');branch=read(I/'actual_journal_branch.json')
require(restore['passed']and restore['all207_threeAdam_all4RNG_match_actual_pre_save']and restore['next_draw_packet_match']
 and restore['restore_occurred_only_at_startup_before_native']and restore['pointer']['sha256']=='0fcaaa39a53ecbe24a5dccf3b2e7bd195a0ed9ed7c38bc631ca9612efdcf63d9','Exact genuine156400 restoration before native')
require(branch['native_updates_or_draws_in_copy']==0 and branch['historical_unsaved_tail_updates']==list(range(156401,156538))
 and branch['saved_prefix']=={'path':str(B/'journals/commits_000017.jsonl'),'offset':316802360,'sha256':'a5db7e40e8372c32c73b09b15ce87af7d3d9fed12bd8b1fede30f5fc3b461887'}
 and branch['new_active_journal']=={'path':str(B/'journals/commits_000018.jsonl'),'offset':316802360,'sha256':'a5db7e40e8372c32c73b09b15ce87af7d3d9fed12bd8b1fede30f5fc3b461887'},'ONLY saved prefix copied AFTER restore;137 historical attempts not copied')
old=branch['preserved_original'];stat=Path(old['path']).stat()
require((stat.st_size,stat.st_mtime_ns)==(old['bytes'],old['mtime_ns']),'Old17 STAT preserved; no whole17 digest invented')
root_path=N/'ROOT_source24_saved156400_direct138_redone_positive_verification_v1.json';root=read(root_path)
require(root['accepted']and root['reviewer']=='/root'and root['first_update']==156401 and root['last_update']==156538
 and root['count']==138 and root['seek_start']==316802360 and root['seek_end']==317073729
 and root['genuine_redo_of_historical137_and_one_positive_beyond_old156537']is True,'Closed independent ROOT138 proof, not repeated audit')
with Path(root['journal_path']).open('rb')as f:f.seek(root['seek_start']);raw=f.read(root['bytes'])
require(len(raw)==271369 and hashlib.sha256(raw).hexdigest()==root['sha256']=='cead29c0734bb72e42c4db0dfdea4905b1345a0efed56c4c8f43d33323eca832','Exact same bounded NEW138 snapshot bytes as independent ROOT receipt')
segment=A/'actual_new138_positive_events156401_156538.jsonl'
with segment.open('xb')as f:f.write(raw)
guard()
CK=B/'checkpoints/update_000156600';pointer=read(CK/'pointer.json');persist=read(CK/'persistence_verified.json')
require(pointer['committed_updates']==156600 and pointer['all_window_updates_committed']and not pointer['diagnostic_only']
 and persist['passed']and persist['optimizer_updates']==156600 and persist['weights_verified']==207
 and persist['Adam_states_verified']==215 and persist['named_tensors_verified']==854
 and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256']
 and persist['journal_prefix']==pointer['journal_prefix'] and persist['saved_semantic_state_verified']['passed']
 and persist['saved_semantic_state_verified']['saved_identity_frontier_sampler_and_journal_verified_before_publication'],'Natural full156600854/207/215 PRE-SAVED and semantic publication')
publication=snapshot(B/'latest_checkpoint.json',A/'actual_checkpoint_publication_at_post156600.json')
require(publication['committed_updates']>=156600 and publication['full_state_verified'],'Actual current full saved publication')
configuration=read(I/'configuration.json');execution=read(N/'ROOT_owned_packet_credit_saved156400_execution_review_v1.json')
require(all(configuration[k]==v for k,v in execution['future_7_runtime_bindings_NOT_actual_restore'].items()),'All7 effective runtime bindings actual/exact')
require(configuration['native_update_commit_pressure_policy']==execution['native_update_commit_pressure_policy']
 and configuration['resident_RAM_packet_credit_policy']==execution['packet_credit_policy']
 and configuration['resident_RAM_resource_scheduling_policy']==execution['resource_scheduling_policy'],'Full2.5GB buffer/originalguards/fullqueue+packet+prefetch retained')
ready=read(I/'CPU/actual_CPU_ready.json')
require(ready['source']['protocol']==execution['CPU_source']and ready['producer_identity']=={'pid':3868,'create_time':1792716948.434736,'ppid':40792}
 and ready['packet_credit']['logical_packet_maximum']==1 and ready['packet_credit']['credit_wait_deadline_seconds']==600
 and ready['packet_credit']['owner']['pid']==42304 and abs(ready['packet_credit']['owner']['create_time']-1792716760.1698885)<.001,'Actual CPU10 source and fresh credit bound to owned native')
family=read(A/'actual_source24_live_runtime_family_after_startup.json');observed=[]
for oldwho in family:
 p=psutil.Process(oldwho['pid']);require(abs(p.create_time()-oldwho['create_time'])<.001,'Same exact live owned identity '+oldwho['role'])
 observed.append({**oldwho,'observed_exact_alive':True})
write(A/'actual_source24_live_runtime_family_after156600.json',observed)
me=psutil.Process();parent=me.parent()
paths={'ROOT_execution':N/'ROOT_owned_packet_credit_saved156400_execution_review_v1.json','ROOT_closed_direct138':root_path,
 'restore':I/'actual_parent_restore.json','branch_copy_after_restore':I/'actual_journal_branch.json','boot':I/'boot.json',
 'effective_configuration':I/'configuration.json','CPU_ready':I/'CPU/actual_CPU_ready.json','current_child':A/'current_child.json',
 'owned_session_launch':A/'actual_owned_session96057_launch.json','postpositive_snapshot':A/'actual_post156600_positive_progress.json',
 'publication_snapshot':A/'actual_checkpoint_publication_at_post156600.json','pointer156600':CK/'pointer.json','persistence156600':CK/'persistence_verified.json',
 'new138_snapshot':segment,'current_actual_family':A/'actual_source24_live_runtime_family_after156600.json','sealer_source':Path(__file__)}
status={'kind':'ASR18_actual_source24_packet_credit_saved156400_recovered156600_status_v1','owned_session':96057,
 'actual_optimizer_updates':progress['actual_optimizer_updates'],'native_updates_this_process':delta,'last_saved_updates':progress['last_saved_updates'],
 'all3_clocks_exact':True,'optimizer_poisoned':False,'schedule_poisoned':False,'planned_native_updates':401072,'administrative_update_stop':None,
 'inputs':{k:pin(p)for k,p in paths.items()},'genuine137redo_plus_one_beyond_old_frontier_ROOT_verified':True,
 'closed_ROOT138_journal_audit_repeated':False,'same138_bytes_snapshot_sha256':root['sha256'],
 'original_old17_prefix_whole_or_checkpoint_tensor_audio_read_here':False,'only_original_active18_range_read_here':{'start':root['seek_start'],'end':root['seek_end'],'bytes':root['bytes']},
 'actual_saved156400_all_state_restore_before_native':True,'actual_prefix_copy_only_AFTER_restore':True,'no_native_or_draw_in_copy':True,
 'standard156600_full854_207_215_PRE_SAVED_and_semantic_passed':True,'fresh_GPU_restore_of156600_performed':False,
 'seven_effective_runtime_bindings_exact':True,'full_conservative_buffer_and_all_reservations_guards_unchanged':True,
 'actual_CPU10_logical_credit_owner':ready['packet_credit'],'same_five_runtime_identities_live':observed,
 'CPU10_normal0_or_kernel_handle_close_not_claimed_while_running':True,'first_actual_helper3_periodic_production_phase_pending160000':True,
 'reader_owned_by_ROOT_not_probed_or_changed_here':True,'no_quality_DEV_inference_export_benchmark_or_speed_saving_prevention_claim':True,
 'metadata_observer':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},
 'metadata_launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()},
 'guard_initial':initial,'guard_final':guard()}
write(A/'actual_source24_saved156600_recovery_status.json',status)
print(json.dumps({'status':pin(A/'actual_source24_saved156600_recovery_status.json'),'actual':progress['actual_optimizer_updates'],
 'pointer':pin(CK/'pointer.json'),'persistence':pin(CK/'persistence_verified.json'),'segment':pin(segment),
 'observer':status['metadata_observer'],'launcher':status['metadata_launcher']}),flush=True)
