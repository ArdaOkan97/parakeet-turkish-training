"""Adapt two focused synthetic fixtures to the NEW domain-preserving TSC contract."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';WORK=ROOT/'work/research'

def one(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new)

runtime=(WORK/'check_tsc_runtime_dispatch_v2.py').read_text(encoding='utf-8')
for name in ('launch_native_training_tsc','launch_with_memory_guard_tsc','launch_with_cache_reclaim_tsc',
             'launch_with_io_retry_tsc','cuda_cache_caller_tsc','launch_with_memory_guard_TSC_resume','train_lora_batched_tsc'):
    runtime=runtime.replace(name+'_v2',name+'_v3')
runtime=one(runtime,'import tsc_training_inputs_v2 as inputs','import tsc_domain_training_identity_v3 as inputs\nimport tsc_training_inputs_v2 as data_inputs')
runtime=runtime.replace('line == 545','line == caller.caller_line(worker_path)')
runtime=one(runtime,'"--epochs", "1", "--steps", "9361",','"--epochs", "1", "--steps", str(inputs.domain_horizon(2037,74884-2037)),\n             "--sampling-mode","domain30","--fleurs-fraction",".3","--replay-seed","2026101101",')
runtime=one(runtime,'"code_sha256": inputs.DOMAIN_SOURCES,','"code_sha256": inputs.DOMAIN_SOURCES,\n         "sampling":{"initial_pool":{"domain":{"fleurs":{"pool_clips":2037},"cv":{"pool_clips":72846}}}},')
runtime=runtime.replace('root_owned_TSC_closed_source_v2','root_owned_TSC_domain30_closed_source_v3').replace('allow_source_sampler_transition_to_combined_uniform','allow_source_domain30_expansion')
start=runtime.index('    declared = inputs.source_sampling_transition');end=runtime.index('    import_proof = ',start)
runtime=runtime[:start]+'''    declared = inputs.approved_source_recipe(json.loads(source_config.read_text(encoding="utf-8")), proof)
    check("actual_domain30_independent_replay_preserved", declared == science
        and declared["sampling_mode"] == "domain30" and declared["fleurs_fraction"] == .3
        and declared["replay_seed"] == 2026101101)
    denied = dict(proof); denied["allow_source_domain30_expansion"] = False
    try: inputs.approved_source_recipe(json.loads(source_config.read_text(encoding="utf-8")), denied)
    except ValueError: check("undeclared_domain_data_expansion_rejected", True)
    else: raise AssertionError("Domain data expansion accepted without ROOT approval")
''' +runtime[end:]
runtime=runtime.replace('inputs.approve(','data_inputs.approve(')
runtime=runtime.replace('"TSC_runtime_CPU_dispatch_verification_v2"','"TSC_runtime_CPU_dispatch_verification_v3"')
ast.parse(runtime)
(WORK/'check_tsc_runtime_dispatch_v3.py').write_text(runtime,encoding='utf-8')

exact=(WORK/'TSC_same_science_tiny_check_v2.py').read_text(encoding='utf-8')
for name in ('run_hf_TSC_arm','resume_TSC_same_science','launch_TSC_same_science_resume','supervise_TSC_same_science',
             'train_lora_batched_tsc','launch_native_training_tsc','launch_with_memory_guard_tsc'):
    exact=exact.replace(name+'_v2',name+'_v3')
exact=exact.replace('isolated_HF_TSC_guarded_quality_arm_v2','isolated_HF_TSC_guarded_quality_arm_v3')
exact=exact.replace('TSC_same_science_native_preload_resume_v2','TSC_same_science_native_preload_resume_v3')
exact=exact.replace('TSC_same_science_native_preload_resume_completion_v2','TSC_same_science_native_preload_resume_completion_v3')
exact=exact.replace('memory_guard_training_launcher_TSC_resume_v2','memory_guard_training_launcher_TSC_resume_v3')
exact=exact.replace('memory_guard_training_launcher_tsc_v2','memory_guard_training_launcher_tsc_v3')
exact=exact.replace('len(chain)==23','len(chain)>=26')
exact=one(exact,"recipe['steps']=9363;recipe['eval_epoch_fraction']=.026708331664062605", "recipe['steps']=13009;recipe['eval_epoch_fraction']=.026708331664062605\n    recipe.update(sampling_mode='domain30',fleurs_fraction=.3,replay_seed=20261012)")
exact=one(exact,"'domain_sampling_identity':source_sampling},'code_sha256':DOMAIN_SOURCES})", "'domain_sampling_identity':source_sampling},'code_sha256':DOMAIN_SOURCES,\n                'sampling':{'initial_pool':{'domain':{'fleurs':{'pool_clips':2037},'cv':{'pool_clips':72846}}}}})")
exact=exact.replace('allow_source_sampler_transition_to_combined_uniform','allow_source_domain30_expansion')
exact=one(exact,"recipe.update(epochs=1.,steps=__import__('math').ceil(n/8))", "recipe.update(epochs=1.,steps=first.domain_horizon(2037,n-2037))") if "recipe.update(epochs=1.,steps=__import__('math').ceil(n/8))" in exact else exact
# Fixture's original scalar update is located by AST-independent exact text below.
exact=exact.replace("recipe.update(epochs=1.,steps=(n+7)//8)","recipe.update(epochs=1.,steps=first.domain_horizon(2037,n-2037))")
exact=exact.replace("recipe.update(epochs=1.,steps=(n+7)//8);", "recipe.update(epochs=1.,steps=first.domain_horizon(2037,n-2037));")
exact=exact.replace("'warm_start':{'mode':'new_optimizer_warm_start'},'code_sha256':chain", "'sampling':{'initial_pool':{'domain':{'fleurs':{'pool_clips':2037},'cv_tsc':{'pool_clips':n-2037}}}},\n                'warm_start':{'mode':'new_optimizer_warm_start'},'code_sha256':chain")
exact=exact.replace('math.ceil(n/8)','first.domain_horizon(2037,n-2037)')
exact=exact.replace('TSC_same_science_exact_resume_tiny_verification_v2','TSC_same_science_exact_resume_tiny_verification_v3')
ast.parse(exact)
(WORK/'TSC_same_science_tiny_check_v3.py').write_text(exact,encoding='utf-8')
print('Two NEW fixtures prepared; no fixture executed or ML imported.')
