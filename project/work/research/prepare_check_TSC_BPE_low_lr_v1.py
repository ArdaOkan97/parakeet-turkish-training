"""Create the narrow stdlib-only low-LR dispatch fixture from frozen tested predecessor."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
old = (HERE / "check_TSC_bpe_lora_native4_controller_v2.py").read_text(encoding="utf-8")
new = old.replace("run_TSC_bpe_lora_native4_v2.py", "run_TSC_bpe_lora_low_lr_v1.py")
new = new.replace("a.adaptation_max_updates == 1000", "a.adaptation_max_updates == 250")
new = new.replace("approved_default1000_and_optional_separate_smoke1", "default250_max1000_optional_separate_smoke1")
new = new.replace("'--lora-lr', '.0003']", "'--lora-lr', '.0003', '--adapt-lora-lr', '.00003']")
new = new.replace("native_max_joint_bytes=1850000000)", "native_max_joint_bytes=1850000000, adapt_lora_lr=3e-5, lora_lr=3e-4)")
start = new.index("    # Compare genuine new worker scientific functions")
end = new.index("    publications = []", start)
new = new[:start] + '''    # Compare the entire actual objective/backward/update loop and serializers.
    # Active LR is changed inside the utility AFTER loading original moments;
    # objective, clipping entry, RNG, native4, gates, and byte-checker stay exact.
    old_tree=ast.parse((HERE/'train_lora_batched_tsc_bpe_lora_native4_v2.py').read_text(encoding='utf-8'))
    old_functions={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
    for name in ('save_training_state','_protected_snapshot','_verify_smoke_phase_reload','native_loss_function'):
        require(ast.dump(functions[name],include_attributes=False)==ast.dump(old_functions[name],include_attributes=False),name+' changed')
    def update_loop(fn):
        return next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    require(ast.dump(update_loop(functions['run']),include_attributes=False)==ast.dump(update_loop(old_functions['run']),include_attributes=False),'Objective/update loop changed')
    def dev_loop(fn):
        return next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_current')
    require(ast.dump(dev_loop(functions['run']),include_attributes=False)==ast.dump(dev_loop(old_functions['run']),include_attributes=False),'DEV gates changed')
    passed('actual_full_objective_backward195_clip_update_DEV_and207_twoAdam_serializer_AST_unchanged')
    rejected('active3e4_cannot_masquerade_as_lowLR',lambda:m.check_phase_arguments(m.set_option(args,'--adapt-lora-lr',3e-4)))
    rejected('dormant_original_LR_cannot_be_changed',lambda:m.check_phase_arguments(m.set_option(args,'--lora-lr',3e-5)))
    rejected('screen_bound_cannot_exceed1000',lambda:m.check_phase_arguments(m.set_option(args,'--adaptation-max-updates',1001)))
    # Actual worker parser/validator AST; only inherited parser is an explicit
    # stdlib fixture, avoiding all module-level Torch imports.
    import argparse
    old_parser=argparse.ArgumentParser()
    old_parser.add_argument('--optimization-source-receipt')
    ns={'source_parser':lambda:old_parser,'Path':Path,'ROOT':ROOT,'METADATA_PATH':ROOT/'work/research/unused_metadata','REPORT_PATH':ROOT/'work/research/unused_proof','__doc__':'mock inherited parser'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[functions['parser']],type_ignores=[])),str(worker_path),'exec'),ns)
    parsed_worker=ns['parser']().parse_args(['--resume-adaptation','opaque.pt','--adaptation-source-receipt','opaque.json'])
    require(parsed_worker.adapt_lora_lr==3e-5 and parsed_worker.adaptation_max_updates==250,'Actual worker options differ')
    passed('actual_worker_parser_exposes_distinct_active_LR_default250')

''' + new[end:]
new = new.replace("inactive_TSC_BPE_LoRA_native4_controller_dispatch_CPU_verification_v2", "inactive_TSC_BPE_LoRA_low_lr_controller_dispatch_CPU_verification_v1")
target = HERE / "check_TSC_bpe_lora_low_lr_controller_v1.py"
if target.exists(): raise ValueError("Fresh fixture already exists")
target.write_text(new, encoding="utf-8")
print(str(target))
