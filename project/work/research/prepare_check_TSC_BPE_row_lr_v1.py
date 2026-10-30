from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'check_TSC_bpe_acquisition_controller_v1.py').read_text(encoding='utf-8')
s=s.replace('run_TSC_bpe_acquire_v1.py','run_TSC_bpe_row_lr_v1.py')
s=s.replace('native_tdt_checkpointed_v3','native_tdt_checkpointed_v4')
s=s.replace('a.acquisition_max_updates == 500','a.acquisition_max_updates == 1000')
s=s.replace('approved_default500_and_optional_separate_smoke1','default1000_DEV250_optional_smoke1')
s=s.replace("'--acquisition-lr', '.0001'","'--active-row-lr', '.001'")
s=s.replace("'--resume-acquisition'","'--resume-row-lr'").replace("'--acquisition-source-receipt'","'--row-lr-source-receipt'")
s=s.replace("native_max_joint_bytes=1850000000)","native_max_joint_bytes=1850000000, active_row_lr=1e-3)")
s=s.replace("'run_id': 'inactive_fixture_no_watched_run', 'worker_arguments_sha256': 'EXPLICIT_ARGV_MOCK',","'run_id': 'inactive_fixture_no_watched_run', 'worker_arguments_sha256': 'EXPLICIT_ARGV_MOCK',\n                'active_row_lr':1e-3, 'source_acquisition_updates':1, 'original_source_selected_step':1,")
s=s.replace("phase = SimpleNamespace(state_dict=lambda runtime_state:","phase = SimpleNamespace(source_step=1,source_acquisition_updates=1,optimizer_updates_completed=1,cumulative_step=3, state_dict=lambda runtime_state:")
s=s.replace("SimpleNamespace(state_dict=lambda: {'opaque_sampler_mock': True}),","SimpleNamespace(updates=3,state_dict=lambda: {'opaque_sampler_mock': True}),")
s=s.replace("run_id=value.run_id, step=1, config_identity={'phase': m.PHASE}, history=[],", "run_id=value.run_id, step=3, config_identity={'phase': m.PHASE}, history=[{'step':1},{'step':2},{'step':3}],")
s=s.replace('inactive_TSC_BPE_acquisition_controller_dispatch_CPU_verification_v2','inactive_TSC_BPE_continued_row_LR_dispatch_CPU_verification_v1')
s=s.replace("    m.WORKER_SHA = m.sha(worker_path); m.EXPECTED_SYNC_LINE = sync[0].lineno", "    require(m.WORKER_SHA==m.sha(worker_path) and m.EXPECTED_SYNC_LINE==sync[0].lineno,'Finalpins/frame differ'); m.finalized_core()")
s=s.replace("    publications = []",'''    old_tree=ast.parse((HERE/'train_lora_batched_tsc_bpe_acquire_v1.py').read_text(encoding='utf-8'))
    old_functions={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
    for name in ('batch_objective','_protected_snapshot','_verify_smoke_phase_reload'):
        require(ast.dump(functions[name],include_attributes=False)==ast.dump(old_functions[name],include_attributes=False),'Undeclared row-only math/preservation change:'+name)
    passed('actual_rowonly_no_grad_encoder_differentiable_decoder_objective_and207_byte_reload_checker_AST_unchanged')
    rejected('source_oldLR_is_not_active_newLR',lambda:m.check_phase_arguments(m.set_option(args,'--active-row-lr',1e-4)))
    require(m.check_phase_arguments(m.set_option(args,'--active-row-lr',3e-4))==(500,0),'Independent3e4mode')
    passed('only_two_explicit_active_row_rates_admitted')
    publications = []''')
# source namespace identity/clock fields come from the exact actual serializer AST.
s=s.replace("and 'acquisition_optimizer' in saved and 'optimizer' not in saved, 'Actual phase serializer contract')", "and 'acquisition_optimizer' in saved and 'optimizer' not in saved\n                and saved['runtime_state']['source_acquisition_updates']==1\n                and saved['runtime_state']['original_source_selected_step']==1\n                and saved['runtime_state']['acquisition_updates_completed']==1, 'Actual phase serializer contract')")
p=HERE/'check_TSC_bpe_row_lr_controller_v1.py'
if p.exists():raise ValueError('Newfixture alreadyexists')
p.write_text(s,encoding='utf-8')
print(str(p))
