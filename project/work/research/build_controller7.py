from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'outputs/parakeet_turkish'
s=(out/'run_experiment_loop_continuation_v6.py').read_text(encoding='utf-8')
replacements={
'train_lora_batched_data_v6.py':'train_lora_batched_data_v7.py',
'2a2ef388faee61ad8b8cd1c28dc6827e0b61589c8709a42af19241bc7ded02ab':'46b5b6d5fa3e88a5aa6708747cc00f3d655bb1c4185be9a7a48e122ff8df0217',
'launch_with_cache_reclaim_v7.py':'launch_with_cache_reclaim_v8.py',
'c0003f97063a5fbe889bac764889f416729c8ca49fe549bbf40481aea4482b81':'fba6794b3689d2e763317a79025e948e7cb8e9f947ec821dee076780d8acf2f7',
'launch_with_io_retry_v4.py':'launch_with_io_retry_v5.py',
'913343ac111a99c1207761c5d73543180e9627586d7e293cd2bfa400b27becf9':'93ab95ca602835e66c037ef72199cfc7452356c96c50af0f4702a91d53ffc1ea',
'launch_with_memory_guard_v10.py':'launch_with_memory_guard_v11.py',
'f105c3ad2e25543e145bb9b1b964f8f51cc36a943423e920b8fce6196335db8d':'37cced75ea223e8ec54f04dc36a503f9a533a2fa5fc38c02298b9ade8a1abc95',
'memory_guard_training_launcher_v10':'memory_guard_training_launcher_v11',
'cache_reclamation_training_launcher_v7':'cache_reclamation_training_launcher_v8',
'data_v6_actual_io_dispatch_cpu_verification.json':'data_v7_actual_io_dispatch_cpu_verification.json',
"tdt_implementation='v2'":"tdt_implementation='nemo',native_max_joint_bytes=1700000000,native_backend_provenance=str(native_ready)",
"replay_fraction=.5):\n values=":"replay_fraction=.5,native_ready=None):\n require(native_ready is not None,'Native ready path must precede canonical argv SHA')\n values=",
"('memory_provenance','memory.json'),('worker_identity','identity.json'),('stop','stop.txt'),('log','training.log')]}":"('memory_provenance','memory.json'),('worker_identity','identity.json'),('native_provenance','native_backend.json'),('stop','stop.txt'),('log','training.log')]}",
"paths['stop'],replay_fraction=replay_fraction)":"paths['stop'],replay_fraction=replay_fraction,native_ready=paths['native_provenance'])",
"command=[str(scoped(args.python)),str(OUTPUT/WRAPPER),'--provenance'":"inner=['--provenance'",
"'--threshold-gb','4','--',*argv]\n trial=":"'--threshold-gb','4','--',*argv]\n command=[str(scoped(args.python)),str(OUTPUT/'launch_native_training_v1.py'),\n   '--native-provenance',str(paths['native_provenance']),'--memory-launcher',str(OUTPUT/WRAPPER),\n   '--memory-launcher-sha256',WRAPPER_SHA,'--trainer-sha256',state['code_sha256']['train_lora_batched_data_v7.py'],'--',*inner]\n trial=",
"'replay_fraction':replay_fraction,'TR_clips_per_update'":"'tdt_implementation':'nemo','native_max_joint_bytes':1700000000,'native_full_model_or_Lightning_trainer_equivalence_claimed':False,\n   'replay_fraction':replay_fraction,'TR_clips_per_update'",
"'kind':'expanded_public_TDT_training_window_v6'":"'kind':'expanded_public_native_TDT_training_window_v7'",
"'preparation_exit_receipt':prepared['preparation_exit_receipt'],'metadata_GPU_smoke'":"'native_handoff_receipt':prepared['native_handoff_receipt'],'native_CPU_readiness':prepared['native_CPU_readiness'],\n  'preparation_exit_receipt':prepared['preparation_exit_receipt'],'metadata_GPU_smoke'",
"'system_commit_API':'GetPerformanceInfo'":"'tdt_implementation':'nemo','native_recipe_sigma':0.,'native_recipe_omega':0.,'native_reduction':'raw per-clip NLL divided by U then clip mean',\n     'native_full_model_or_Lightning_trainer_equivalence_claimed':False,'system_commit_API':'GetPerformanceInfo'",
"default='adaptive_20261011_public_data_03'":"default='adaptive_20261011_native_data_01'",
"ROOT/'work/training-env/Scripts/python.exe'":"ROOT/'work/nemo-native-env-v2/Scripts/python.exe'",
"OUTPUT/'continuation_v6_actual_preflight.json'":"OUTPUT/'continuation_v7_actual_preflight.json'",
}
for a,b in replacements.items():
 if a not in s: raise ValueError('Missing substitution '+a)
 s=s.replace(a,b)
s=s.replace("FROZEN = {", "FROZEN = {\n 'launch_native_training_v1.py':'66a4f3bd063c2605317ee16da5b748b6655a59d6ab102e0bbd3f6828a6addae6',\n 'native_tdt_checkpointed_v1.py':'f41d19bc88b82e6a03b394e8e42d48820314aa7a3a3d189b959c5503c2e57a7c',\n 'preload_nemo_linker_v1.py':'f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba',\n 'launch_nemo_linker_v4.py':'bdd7078dff81adc961c17588ed7fd84b152da71c619c2ff114994fa68d308af0',")
s=s.replace(" require(not (OUTPUT/'runs'/args.loop_id).exists(),'Use a fresh loop namespace')", """ require(not (OUTPUT/'runs'/args.loop_id).exists(),'Use a fresh loop namespace')
 require(scoped(args.python)==ROOT/'work/nemo-native-env-v2/Scripts/python.exe','Native candidate requires the verified isolated native runtime')
 handoff_path=scoped(args.native_handoff_receipt);handoff=read(handoff_path,{})
 require(sha(handoff_path)=='d04c9113ab1036ac66ab6f9d734c320d3d1e68c3054b7cd29cfb5ed021276d50','Closed native handoff receipt changed')
 require(handoff['kind']=='root_owned_cooperative_native_loss_handoff_receipt_v1'
   and handoff['root_observed_owned_session_completed'] is True and handoff['root_observed_exit_code']==0
   and handoff['actual_child_exit_code']==0 and handoff['worker_absent_confirmed'] is True
   and handoff['completed_optimizer_updates']==18 and handoff['selected_step']==0
   and handoff['selected_checkpoint_is_original_warm_start_quality'] is True
   and handoff['original_source_step']==1871 and handoff['test_evaluated'] is False,'Actual18-update closed handoff and original1871 quality required')
 for key in ('loop_status','controller','worker_identity','selection','run_config','history','selected_checkpoint','durable_latest_checkpoint','original_source_checkpoint'):
  require(sha(scoped(handoff[key]['path']))==handoff[key]['sha256'],'Closed handoff changed: '+key)
 closed=read(handoff['loop_status']['path'],{})
 require(closed['active_run_id'] is None and closed['resource_review_required'] is False
   and closed['objective_incomplete'] is True,'Previous GPU owner must be terminal and objective incomplete')
 native_path=OUTPUT/'native_training_v7_cpu_readiness_v1.json';native=read(native_path,{})
 require(native['passed'] is True and native['trainer_sha256']==FROZEN['train_lora_batched_data_v7.py']
   and native['native_adapter_sha256']==FROZEN['native_tdt_checkpointed_v1.py']
   and native['exact_joint_bytes_per_clip_bound']==1700000000
   and all(x['rows_above1700000000']==0 for x in native['bounds'].values()),'Native all-accepted-row1.7GB bound/readiness differs')""")
s=s.replace("dispatch_path,pause_path)}", "dispatch_path,pause_path,handoff_path,native_path)}")
s=s.replace("return {'receipt':record(receipt_path)", "return {'native_handoff_receipt':record(handoff_path),'native_CPU_readiness':record(native_path),'receipt':record(receipt_path)")
s=s.replace(" 'io':record(trial['io_provenance_path']),'eligible_updates':", " 'io':record(trial['io_provenance_path']),'native_backend':record(trial['native_provenance_path']),'native_completion':record(completion_path),'eligible_updates':")
needle=" require(io['status']=='complete' and io['worker_sha256']==io['worker_sha256_after']==sources['train_lora_batched_data_v7.py'],'Actual I/O worker source changed')"
s=s.replace(needle,needle+"""
 native_path=Path(trial['native_provenance_path']);native=read(native_path,{})
 completion_path=native_path.with_suffix('.completion.json');completion=read(completion_path,{})
 require(native['kind']=='native_tdt_training_backend_v1' and native['status']=='ready'
   and native['run_id']==trial['run_id'] and native['tdt_implementation']=='nemo'
   and native['worker_arguments_sha256']==trial['canonical_worker_arguments_sha256']
   and native['trainer_sha256']==sources['train_lora_batched_data_v7.py']
   and native['adapter_sha256']==sources['native_tdt_checkpointed_v1.py']
   and native['launcher_sha256']==sources['launch_native_training_v1.py']
   and native['memory_launcher_sha256']==sources[WRAPPER],'Native ready runtime source/argv identity differs')
 require(native['preload']['actual_binding_linker']['version']==[12,9]
   and native['preload']['cuda_initialized'] is False
   and native['preload']['module_sha256']==sources['preload_nemo_linker_v1.py'],'Actual pre-Torch native linkage differs')
 require(completion['kind']=='native_tdt_training_backend_completion_v1' and completion['status']=='complete'
   and completion['run_id']==trial['run_id'] and completion['sources_unchanged'] is True
   and completion['backend_ready_sha256']==sha(native_path),'Native outer completion did not close with unchanged ready provenance')""")
s=s.replace(" p.add_argument('--source-run-id',default=SOURCE_RUN)"," p.add_argument('--source-run-id',default=SOURCE_RUN)\n p.add_argument('--native-handoff-receipt',type=Path,default=OUTPUT/'public_data_03_native_handoff_receipt_v1.json')")
s=s.replace('"""Finite expanded-data training window; fresh rank8 control, matched rank16 option.', '"""Official native NeMo TDT-loss quality window with fresh optimizer and DEV selection.')
s=s.replace('New data uses new optimizer/sampler state, retaining the selected acoustic model.', 'Changed native loss uses fresh optimizer/sampler state, retaining the selected acoustic model.\nThis uses the official NeMo loss kernel in the existing HF LoRA recipe, not the\nfull NeMo model or Lightning trainer. sigma0/omega0/per-U clip mean remain explicit.')
(out/'run_experiment_loop_continuation_v7.py').write_text(s,encoding='utf-8',newline='\n')
