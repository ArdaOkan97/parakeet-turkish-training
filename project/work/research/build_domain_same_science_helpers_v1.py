"""Generate only NEW domain siblings from reviewed same-science orchestration."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
pins=json.loads((HERE/'domain30_runtime_chain_pins_v1.json').read_text())['source_sha256']
text=(HERE/'launch_same_science_resume_v1.py').read_text()
start=text.index('PINS=');end=text.index('\n\n\ndef require',start)
text=text[:start]+'PINS='+repr(pins)+text[end:]
text=text.replace('train_lora_batched_data_v11.py','train_lora_batched_domain30_v1.py')
text=text.replace('launch_with_memory_guard_v15.py','launch_with_memory_guard_domain30_v1.py')
text=text.replace('HF_same_science_native_preload','domain_same_science_native_preload')
text=text.replace('SAME-run exact HF','SAME-run exact domain30')
text=text.replace('memory15/trainer11','isolated domain30')
text=text.replace('HF resume','domain30 resume')
text=text.replace("    scoped(option(worker,'--resume-exact'))", """    scoped(option(worker,'--resume-exact'))
    scoped(option(worker,'--source-closure'))
    require(option(worker,'--sampling-mode')=='domain30' and float(option(worker,'--fleurs-fraction'))==.3,
        'Only unchanged domain30/30% FLEURS sampling may use this exact route')
    int(option(worker,'--replay-seed'))
    require(old.get('launcher_sha256')==PINS['launch_native_training_domain30_v1.py']
        and old.get('memory_launcher_sha256')==PINS['launch_with_memory_guard_domain30_v1.py'],
        'Original first domain backend-ready source identity differs')""")
text=text.replace("    fresh=scoped(args.fresh_runtime_provenance);completion=fresh.with_suffix('.completion.json')", """    proof_paths=[scoped(option(memory,flag)) for flag in
        ('--provenance','--cache-provenance','--memory-provenance','--worker-identity')]
    require(len(set(proof_paths))==4 and all(not p.exists() for p in proof_paths),
        'Four distinct NEW process evidence paths required')
    fresh=scoped(args.fresh_runtime_provenance);completion=fresh.with_suffix('.completion.json')
    protected={original,scoped(option(worker,'--source-closure')),scoped(option(worker,'--resume-exact'))}
    require(not protected.intersection([fresh,completion,*proof_paths])
        and fresh not in proof_paths and completion not in proof_paths,
        'Fresh evidence must not alias immutable identity, closure, checkpoint or other evidence')""")
target=HERE/'launch_domain30_same_science_resume_v1.py'
with target.open('x',encoding='utf-8') as f:f.write(text)

text=(HERE/'resume_hf_same_science_v1.py').read_text()
text=text.replace('SAME-run exact HF','SAME-run exact domain30')
text=text.replace('import run_hf_quality_loop_v1 as quality','import run_hf_domain30_arm_v1 as domain')
text=text.replace('import launch_same_science_resume_v1 as fresh','import launch_domain30_same_science_resume_v1 as fresh')
text=text.replace('HERE,ROOT=quality.HERE,quality.ROOT\nSOURCE_PINS=quality.SOURCE_PINS',
    "HERE,ROOT=domain.HERE,domain.ROOT\nSOURCE_PINS=fresh.PINS\nDOMAIN_CONTROLLER_SHA='429daedee13ad14ee803c197eee597078940ef63525b093fe881c2af88ac772f'")
text=text.replace('return quality.runtime.previous.scoped(path)','return domain.scoped(path)')
text=text.replace('quality.runtime.previous.argument_sha','domain.guards.argument_sha')
text=text.replace('quality.runtime.previous.ProgressGuard','domain.guards.ProgressGuard')
text=text.replace('quality.runtime.previous.prior.keep_awake','domain.guards.prior.keep_awake')
text=text.replace('quality.runtime.previous.log_phase','domain.guards.log_phase')
text=text.replace('quality.runtime.previous.self_memory','domain.guards.self_memory')
text=text.replace('quality.validate_identity','domain.validate_identity')
text=text.replace('quality.runtime.replace_option','replace_option')
text=text.replace('quality.runtime.option','fresh.option')
text=text.replace('HF_same_run_exact_science_continuation','domain_same_run_exact_science_continuation')
text=text.replace('HF_same_science_native_preload','domain_same_science_native_preload')
text=text.replace('launch_same_science_resume_v1.py','launch_domain30_same_science_resume_v1.py')
text=text.replace("'runtime_source_sha256':SOURCE_PINS", "'source_sha256':SOURCE_PINS")
text=text.replace('default=4.5','default=3.')
target=HERE/'resume_domain30_same_science_v1.py'
with target.open('x',encoding='utf-8') as f:f.write(text)
