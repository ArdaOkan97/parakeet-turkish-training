from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
s=(OUT/'run_experiment_loop_continuation_v7.py').read_text(encoding='utf-8')
mapping={'train_lora_batched_data_v7.py':'train_lora_batched_data_v9.py',
 'launch_with_io_retry_v5.py':'launch_with_io_retry_v7.py',
 'launch_with_cache_reclaim_v8.py':'launch_with_cache_reclaim_v10.py',
 'launch_with_memory_guard_v11.py':'launch_with_memory_guard_v13.py',
 'native_tdt_checkpointed_v1.py':'native_tdt_checkpointed_v2.py',
 'launch_native_training_v1.py':'launch_native_training_v3.py'}
for old,new in mapping.items():
 s=s.replace(sha(OUT/old),sha(OUT/new)).replace(old,new)
s=s.replace('memory_guard_training_launcher_v11','memory_guard_training_launcher_v13')
s=s.replace('cache_reclamation_training_launcher_v8','cache_reclamation_training_launcher_v10')
s=s.replace('native_tdt_training_backend_v1','native_tdt_training_backend_v2').replace('native_tdt_training_backend_completion_v1','native_tdt_training_backend_completion_v2')
s=s.replace('data_v7_actual_io_dispatch_cpu_verification.json','data_v9_actual_io_dispatch_cpu_verification.json')
s=s.replace("native['trainer_sha256']==FROZEN['train_lora_batched_data_v9.py']","native['trainer_sha256']==sha(OUTPUT/'train_lora_batched_data_v7.py')")
s=s.replace("native['native_adapter_sha256']==FROZEN['native_tdt_checkpointed_v2.py']","native['native_adapter_sha256']==sha(OUTPUT/'native_tdt_checkpointed_v1.py')")
s=s.replace("require(not (OUTPUT/'runs'/args.loop_id).exists(),'Use a fresh loop namespace')", """loop_path=OUTPUT/'runs'/args.loop_id
 require(loop_path.is_dir() and {p.name for p in loop_path.iterdir()} <= {'fork_snapshot','runtime_migration_cap6.json','runtime_migration_cap7.json','runtime_migration_cap9.json'},'Only immutable approved preseeded fork/transition files allowed')
 resource_exit=scoped(args.native_resource_exit_receipt);closed_native=read(resource_exit,{})
 require(sha(resource_exit)=='a597c2e72d0818370e3fa1a536e1cd8dff3cc526eeead3618beed1b9d8b63449'
   and closed_native['root_observed_owned_session_completed'] is True and closed_native['root_observed_exit_code']==1
   and closed_native['actual_child_exit_code']==0 and closed_native['worker_absent_confirmed'] is True
   and closed_native['completed_optimizer_updates']==63 and closed_native['selected_step']==0,'Closed native63/best0 owned receipt required')
 for value in closed_native.values():
  if isinstance(value,dict) and 'path' in value and 'sha256' in value:require(sha(scoped(value['path']))==value['sha256'],'Native predecessor artifact changed')
 fork_path=loop_path/'fork_snapshot/fork_manifest.json';fork=read(fork_path,{})
 require(sha(fork_path)=='d5bf3d509b3ca043e086df183a763e44e8ec4f6009b0609faa092bdd2a6b3527'
  and fork['passed'] is True and fork['source_step']==63 and fork['inherited_selected_step']==0
  and fork['target_run_id']==args.loop_id+'_native_resume','Actual byte-preserved fork63 identity differs')
 for value in fork['fork_checkpoint'].values():require(sha(scoped(value['path']))==value['sha256'],'Immutable fork state changed')
 require(sha(Path(fork['fork_checkpoint']['best']['mutable_work_path']))==fork['fork_checkpoint']['best']['sha256'],'New work best0 differs from immutable fork')
 migration_path=scoped(args.runtime_migration);migration=read(migration_path,{})
 require(migration['kind']=='native_runtime_state_transition_v1' and migration['passed'] is True
  and migration['target_run_id']==fork['target_run_id'] and migration['fork_manifest']['sha256']==sha(fork_path)
  and migration['declared_allocator_policy']=={'cap_gb':9,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},'Explicit native cache-only runtime transition differs')
 for old,value in migration['approved_source_changes'].items():
  require(sha(OUTPUT/old)==value['old_sha256'] and sha(OUTPUT/value['new_filename'])==value['new_sha256'],'Approved runtime source changed')
 resource_proof=scoped(args.native_resource_proof);long=read(resource_proof,{})
 require(long['passed'] is True and long['status']=='complete' and long['cap_gb']==9 and long['requested_updates']==len(long['updates'])==3
  and long['selection']['sha256']=='3ad2f08d7c6d3bcc8b4f39882cea9953054c5b461cfd10ab2d235dd1af9eee95'
  and long['source_checkpoint']['sha256']==closed_native['durable_latest_checkpoint']['sha256']
  and long['native_cleanup_evidence']['failures']==0 and long['native_cleanup_evidence']['reclamations']>0
  and long['max_system_commit_fraction']<.95 and long['minimum_owned_self_headroom_bytes']>=2e9,'Actual native2/cap9 longest16 TRAIN whole-update proof required')
 for path,digest in long['source_sha256'].items():require(sha(scoped(path))==digest,'Actual resource proof source changed')
 native_config=read(closed_native['run_config']['path'],{})""")
s=s.replace("dispatch_path,pause_path,handoff_path,native_path)}","dispatch_path,pause_path,handoff_path,native_path,resource_exit,fork_path,migration_path,resource_proof)}")
s=s.replace("return {'native_handoff_receipt':", "return {'native_resource_exit_receipt':record(resource_exit),'native_resource_proof':record(resource_proof),'fork_manifest':record(fork_path),'fork':fork,'runtime_migration':record(migration_path),'native_source_config':native_config,'native_handoff_receipt':")
s=s.replace("replay_fraction=.5,native_ready=None):", "replay_fraction=.5,native_ready=None,exact_resume=False):")
s=s.replace("argv=[]\n for key,value", "if exact_resume:\n  values.pop('resume_weights');values['resume_exact']=str(start);values['resume_runtime_migration']=str(scoped(args.runtime_migration))\n argv=[]\n for key,value")
s=s.replace("evidence['eligible_update_calls']==trial['completed_steps']", "evidence['eligible_update_calls']==trial['completed_steps']-trial['initial_step']")
s=s.replace("   'io':record(trial['io_provenance_path'])", "   'io':record(trial['io_provenance_path'])")
s=s.replace("native_ready=paths['native_provenance'])", "native_ready=paths['native_provenance'],exact_resume=stage=='native_resume')")
s=s.replace("'initial_step':0,", "'initial_step':63 if stage=='native_resume' else 0,")
s=s.replace("'fresh_optimizer_and_sampler':True,'start_weights'", "'fresh_optimizer_and_sampler':stage!='native_resume','exact_optimizer_sampler_RNG_continuation':stage=='native_resume','start_weights'")
s=s.replace("state['trials'].append(trial);state['active_run_id']=run_id;", "if stage=='native_resume':\n  directory.mkdir();source_initial=OUTPUT/'runs/adaptive_20261011_native_data_01_public_rank8_control/initial_development.json'\n  write(directory/'initial_development.json',read(source_initial,{}));trial['inherited_initial_DEV']=record(source_initial)\n state['trials'].append(trial);state['active_run_id']=run_id;")
s=s.replace("'kind':'expanded_public_native_TDT_training_window_v7'", "'kind':'native_exact_state_cache_recovery_window_v8'")
s=s.replace("loop_dir.mkdir();state=", "state=")
s=s.replace("'source_config':prepared['source_config'],", "'source_config':prepared['native_source_config'],")
s=s.replace("'native_handoff_receipt':prepared['native_handoff_receipt']", "'native_resource_exit_receipt':prepared['native_resource_exit_receipt'],'native_resource_proof':prepared['native_resource_proof'],\n  'fork_manifest':prepared['fork_manifest'],'runtime_migration':prepared['runtime_migration'],\n  'native_handoff_receipt':prepared['native_handoff_receipt']")
s=s.replace("'new_runtime_hardware_efficacy_pending':True", "'new_runtime_hardware_efficacy_pending':False,'actual_long_TRAIN_resource_pilot_passed':True,'native_inactive_cache_threshold_bytes':750000000")
s=s.replace("control=run_stage(args,state,'public_rank8_control',common,8,prepared['screen_updates'],deadline)", "control=run_stage(args,state,'native_resume',Path(prepared['fork']['fork_checkpoint']['latest']['path']),8,prepared['screen_updates'],deadline)")
s=s.replace("default='adaptive_20261011_native_data_01'", "default='adaptive_20261011_native_memory_fix_01'")
s=s.replace("OUTPUT/'continuation_v7_actual_preflight.json'", "OUTPUT/'continuation_v8_actual_preflight.json'")
s=s.replace(" p.add_argument('--source-run-id',default=SOURCE_RUN)", " p.add_argument('--source-run-id',default=SOURCE_RUN)\n p.add_argument('--native-resource-exit-receipt',type=Path,default=OUTPUT/'native_data_01_resource_exit_receipt_v1.json')\n p.add_argument('--runtime-migration',type=Path,default=OUTPUT/'runs/adaptive_20261011_native_memory_fix_01/runtime_migration_cap9.json')\n p.add_argument('--native-resource-proof',type=Path,default=OUTPUT/'native_long_memory_cap9_v3.json')")
(OUT/'run_experiment_loop_continuation_v8.py').write_text(s,encoding='utf-8',newline='\n')
