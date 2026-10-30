"""Create only the narrowly adapted repeatable v2 route fixture, old fixture untouched."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
old=ROOT/"work/research/check_bpe512_runtime_dispatch_v1.py"
text=old.read_text(encoding="utf-8")
for stem in ("cuda_cache_caller_bpe512","launch_with_io_retry_bpe512","launch_with_cache_reclaim_bpe512",
        "launch_with_memory_guard_bpe512","launch_native_training_bpe512","train_lora_batched_bpe512"):
    text=text.replace(stem+"_v1",stem+"_v2")
text=text.replace("== 588","== 618").replace('" * 586','" * 616').replace("exact_target588","exact_target618")
text=text.replace("isolated_HF_BPE512_runtime_mocked_CPU_dispatch_verification_v1","isolated_HF_BPE512_domain_runtime_mocked_CPU_dispatch_verification_v2")
text=text.replace("    paths = {name: scratch", "    for name in ('converted.pt','source.pt','selection.json','closure.json','conversion.json'):\n        (scratch/name).write_bytes(b'opaque synthetic fixture; not checkpoint/model parsed')\n    paths = {name: scratch")
text=text.replace('"--lora-lr", "3e-4", "--eval-every-updates", "250",',
    '"--lora-lr", "3e-4", "--sampling-mode", "domain30", "--fleurs-fraction", ".3", "--replay-seed", "20261012", "--eval-every-updates", "250",')
text=text.replace('and parsed.BPE_delta_multiplier == 1 and parsed.new_token_lr_multiplier == 6)',
    'and parsed.BPE_delta_multiplier == 1 and parsed.new_token_lr_multiplier == 6\n            and parsed.sampling_mode == "domain30" and parsed.fleurs_fraction == .3 and parsed.replay_seed == 20261012)')
text=text.replace('import cuda_cache_caller_bpe512_v2 as caller','import cuda_cache_caller_bpe512_v2 as caller\nimport launch_with_memory_guard_BPE512_resume_v2 as exact_memory')
text=text.replace('    "launch_native_training_bpe512_v2.py"]','    "launch_native_training_bpe512_v2.py", "launch_with_memory_guard_BPE512_resume_v2.py"]')
insert='''    exact_flags=list(flags);exact_flags[exact_flags.index('--resume-weights')]='--resume-exact'
    resume_records=exact_memory.bpe_argument_records(exact_flags)
    check('exact_only_admission_binds_original_ready',resume_records['native_backend_provenance']['sha256']==sha(native))
    check('exact_only_admission_binds_closed_snapshot',resume_records['resume_exact']['sha256']==sha(scratch/'converted.pt'))
    for forbidden_flags,description in ((flags,'warmweights'),(exact_flags+['--resume-runtime-migration',str(scratch/'closure.json')],'migration')):
        try:exact_memory.bpe_argument_records(forbidden_flags)
        except ValueError:check('exact_route_rejects_'+description,True)
        else:raise AssertionError('invalid exact admission')
    try:memory.bpe_argument_records(exact_flags)
    except ValueError:check('initial_memory_rejects_exact_resume',True)
    else:raise AssertionError('initial accepted exact')
    check('exact_scope_cap_matches8',exact_memory.validate_capacity(16_000_000_000)['cap_bytes']==8_000_000_000)
'''
text=text.replace('    check("no_real_ML_imported_or_retained",',insert+'    check("no_real_ML_imported_or_retained",')
path=ROOT/"work/research/check_bpe512_domain_runtime_dispatch_v2.py"
compile(text,str(path),"exec")
with path.open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
print(path)
