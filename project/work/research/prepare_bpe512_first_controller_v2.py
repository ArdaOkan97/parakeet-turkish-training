"""Create an isolated CLOSED-domain BPE first controller, standard library only."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
text=(HERE/'run_hf_bpe512_arm_v1.py').read_text(encoding='utf-8')
text=text.replace('hf_bpe512_training_v1 import closed_source_identity, candidate_assets, BASELINE_SEMANTIC_SHA',
    'hf_bpe512_training_v2 import closed_source_identity, common_source_recipe, candidate_assets, BASELINE_SEMANTIC_SHA')
text=text.replace('HF_BPE512_runtime_chain_pins_v2.json','HF_BPE512_domain_runtime_chain_pins_v2.json')
text=text.replace("CHAIN_SHA = 'f8bcb1ecc09ec30cab11a260dc8cb5f839994f26fb464b29040a41522055eb45'", "CHAIN_SHA = 'PENDING_FROZEN_BPE_V2_CHAIN'")
text=text.replace('isolated_HF_BPE512_runtime_chain_pins_v2','isolated_HF_BPE512_domain_runtime_chain_pins_v2')
for name in ('launch_with_memory_guard_bpe512','launch_with_cache_reclaim_bpe512','train_lora_batched_bpe512'):
    text=text.replace(name+'_v1.py',name+'_v2.py')
text=text.replace("'max_retention_failures','target_turkish_wer')", "'max_retention_failures','target_turkish_wer','sampling_mode','fleurs_fraction','replay_seed')")
# Use its stable following assignment rather than relying on a display adjective.
text=text.replace('    result=[]\n    for key in RECIPE_KEYS:', '''    require(recipe['sampling_mode']=='domain30' and recipe['fleurs_fraction']==.3 and type(recipe['replay_seed']) is int,
        'Retain proven30%FLEURS and independent replay seed')
    result=[]
    for key in RECIPE_KEYS:''')
start=text.index('    context=SimpleNamespace(run_id=run_id,')
end=text.index("    require(config.get('run_id')",start)
text=text[:start]+'''    source_config=scoped(args.source_run_config)
    config=read(source_config,{})
    closure=read(scoped(args.source_closure),{})
    recipe=common_source_recipe(config,closure)
    context=SimpleNamespace(**recipe,run_id=run_id,bpe_source_checkpoint=scoped(args.source_checkpoint),
        bpe_source_selection=scoped(args.source_selection),bpe_source_closure=scoped(args.source_closure),
        bpe_conversion_report=scoped(args.conversion_report),resume_weights=scoped(args.converted_weights),resume_exact=None)
    identity=closed_source_identity(context)
    require(closure.get('source_run_config')==record(source_config),'ROOT closure must also bind exact source run configuration')
'''+text[end:]
text=text.replace("    recipe=config['config_identity']['training_arguments'];argv=recipe_arguments(recipe)", "    argv=recipe_arguments(recipe)")
text=text.replace("'kind':'isolated_HF_BPE512_guarded_quality_arm_v1'", "'kind':'isolated_HF_BPE512_guarded_quality_arm_v2'")
text=text.replace("'source_identity':identity,'source_run_config':record(source_config),'copied_source_recipe':recipe,", "'source_identity':identity,'source_run_config':record(source_config),'copied_source_recipe':recipe,\n        'source_sampling_policy':identity['source_sampling_transition'],'epochs_semantics':'minimum_domain_exposure',")
text=text.replace("'Fresh Adam/sampler/RNG/history/counters from CLOSED qualifying learned weights'],", "'Fresh Adam/sampler/RNG/history/counters from latest CLOSED qualifying learned domain weights',\n            'Retain30%FLEURS/restCV and independent replay RNG, source seeds/cadence/horizon; BPE target lengths may alter TR bucket ordering'],")
text=text.replace("    plan=prepare(args)\n    if args.validate_only:", "    plan=prepare(args)\n    if args.preflight_output:\n        require(not scoped(args.preflight_output).is_relative_to(HERE/'runs'/args.loop_id),'Preflight evidence must stay outside future fresh loop directory')\n    if args.validate_only:")
ast.parse(text)
with (HERE/'run_hf_bpe512_arm_v2.py').open('x',encoding='utf-8',newline='\n') as out:out.write(text)
print('NEW BPE first v2 saved; routing SHA intentionally pending actual changed-interface proof.')
