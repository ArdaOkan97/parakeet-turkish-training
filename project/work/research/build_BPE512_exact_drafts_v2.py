"""Create only NEW BPE continuation siblings from reviewed source-only templates."""
import ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def function_replace(text,name,replacement):
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])
def save(name,text):
    path=HERE/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n')

chain=json.loads((HERE/'HF_BPE512_domain_runtime_chain_pins_v2.json').read_text(encoding='utf-8'))
pins=chain['source_sha256'];first_sha=sha(HERE/'run_hf_bpe512_arm_v2.py')
pre=(HERE/'launch_TSC_same_science_resume_v3.py').read_text(encoding='utf-8')
pre=pre.replace('TSC','BPE512').replace('tsc_v3','bpe512_v2').replace('_resume_v3','_resume_v2')
pre=re.sub(r'^PINS = .*$', 'PINS = '+repr(pins),pre,flags=re.M)
pre=re.sub(r'^RUNTIME_MEMORY_SHA = .*$', 'RUNTIME_MEMORY_SHA = '+repr(pins[chain['exact_memory_launcher']]),pre,flags=re.M)
start=pre.index('    scoped(option(worker,\'--source-closure\'))')
end=pre.index('    require(old.get(\'launcher_sha256\')',start)
pre=pre[:start]+'''    assets=[scoped(option(worker,flag)) for flag in ('--bpe-source-checkpoint','--bpe-source-selection',
        '--bpe-source-closure','--bpe-conversion-report')]
    require(all(path.is_file() for path in assets),'Original BPE source/conversion assets must remain existing')
    require(option(worker,'--sampling-mode')=='domain30' and float(option(worker,'--fleurs-fraction'))==.3,
        'Retain original30%FLEURS/restCV sampling')
    int(option(worker,'--replay-seed'))
'''+pre[end:]
pre=pre.replace("scoped(option(worker,'--source-closure'))","*assets")
save('launch_BPE512_same_science_resume_v2.py',pre)

exact=(HERE/'resume_domain30_same_science_v2.py').read_text(encoding='utf-8')
exact=exact.replace('run_hf_domain30_arm_v2','run_hf_bpe512_arm_v2').replace('launch_domain30_same_science_resume_v1','launch_BPE512_same_science_resume_v2')
exact=exact.replace('DOMAIN_CONTROLLER_SHA','BPE_CONTROLLER_SHA').replace("'b14c928d5893b49b510ea8f4a6078c7954af0cd507eac46a2ffde5789b349840'",repr(first_sha))
exact=exact.replace('domain_same_run_exact_science_continuation_v2','BPE512_same_run_exact_science_continuation_v2')
exact=exact.replace('isolated_HF_domain30_guarded_quality_arm_v1','isolated_HF_BPE512_guarded_quality_arm_v2')
exact=exact.replace('domain_same_science_native_preload_resume_v1','BPE512_same_science_native_preload_resume_v2').replace('domain_same_science_native_preload_resume_completion_v1','BPE512_same_science_native_preload_resume_completion_v2')
exact=exact.replace('domain30 continuation','BPE512 domain30 continuation').replace('first domain window','first BPE512 window').replace('First domain window','First BPE512 window')
exact=exact.replace('domain first-launch','BPE512 first-launch').replace('Original domain parent','Original BPE512 parent')
exact=exact.replace("'--source-closure'","'--bpe-source-closure'")
exact=exact.replace('memory_guard_training_launcher_domain30_v1','memory_guard_training_launcher_bpe512_v1')
exact=exact.replace('cache_reclamation_training_launcher_domain30_v1','cache_reclamation_training_launcher_bpe512_v1')
exact=exact.replace('atomic_io_retry_domain30_v1','atomic_io_retry_bpe512_v1')
exact=exact.replace('launch_with_memory_guard_domain30_v1.py','launch_with_memory_guard_BPE512_resume_v2.py')
exact=exact.replace('launch_with_cache_reclaim_domain30_v1.py','launch_with_cache_reclaim_bpe512_v2.py')
exact=exact.replace('train_lora_batched_domain30_v1.py','train_lora_batched_bpe512_v2.py')
exact=function_replace(exact,'validate_config','''def validate_config(config,argv):
    identity=config['config_identity'];recipe=identity['training_arguments']
    require(config.get('run_id')==fresh.option(argv,'--run-id') and config.get('turkish_clips_used')==74883
        and config.get('replay_clips_used')==1595 and config.get('sampling_mode')==recipe.get('sampling_mode')=='domain30'
        and config.get('replay_seed')==recipe.get('replay_seed')==int(fresh.option(argv,'--replay-seed')),
        'SAME full-pool BPE domain run and independent replay seed required')
    require(set(recipe)==set(domain.RECIPE_KEYS)|{'BPE_delta_multiplier'} and recipe['BPE_delta_multiplier']==1.,
        'Retain exactly35 copied source recipe fields plus fixed1x BPE-row multiplier')
    require(config.get('epochs')==recipe.get('epochs')==1. and config.get('steps')==recipe.get('steps')==13009,
        'Retain original one minimum-domain exposure horizon')
    common={key:recipe[key] for key in domain.RECIPE_KEYS};domain.recipe_arguments(common)
    closure=read(scoped(fresh.option(argv,'--bpe-source-closure')), {})
    original_source=closure['source_run_config']
    require(record(scoped(original_source['path']))==original_source,'Original CLOSED source configuration changed')
    source_config=read(scoped(original_source['path']),{})
    source_recipe=domain.common_source_recipe(source_config,closure)
    require(common==source_recipe,'Preserve every original source recipe field including seeds,LR,KD,cadence and horizon')
    for name,value in common.items():
        flag='--'+name.replace('_','-')
        if value is None:require(flag not in argv,'Unexpected scientific argument: '+flag)
        elif type(value) is bool:require((flag in argv)==value,'Scientific boolean changed: '+flag)
        else:
            actual=fresh.option(argv,flag)
            require((actual==value if type(value) is str else type(value)(actual)==value),'Scientific argument changed: '+flag)
    require('--BPE-delta-multiplier' not in argv,'BPE multiplier is a fixed worker default, not an unvalidatedCLI option')
    for name,digest in config.get('code_sha256',{}).items():
        require(Path(name).name==name and sha(HERE/name)==digest,'Original worker scientific source changed: '+name)
    for flag in ('--native-backend-provenance','--bpe-source-closure','--bpe-source-checkpoint',
                 '--bpe-source-selection','--bpe-conversion-report'):
        path=scoped(fresh.option(argv,flag))
        require(identity['input_sha256'].get(str(path))==sha(path),'Original BPE source/ready/conversion input changed: '+flag)
    from hf_bpe512_training_v2 import GEOMETRY, established_source_equivalence, TOKENIZER_SEMANTIC_SHA
    require(identity['extended_tokenizer_sha256']==TOKENIZER_SEMANTIC_SHA
        and {key:identity['BPE512'].get(key) for key in GEOMETRY}==GEOMETRY,
        'Retain8711/blank8192/teacher8193/duration8711:8716 and six6x/BPE1x geometry')
    for key in ('tr_manifest','replay_manifest','data_v3_proof','tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit'):
        if config.get(key):
            path=scoped(config[key]);require(identity['input_sha256'].get(str(path))==sha(path),'Original data/target proof changed: '+key)
            if '--'+key.replace('_','-') in argv:require(scoped(fresh.option(argv,'--'+key.replace('_','-')))==path,'Data path changed: '+key)
    candidate=scoped(fresh.option(argv,'--bpe-candidate-dir'));domain.candidate_assets(candidate)
    for name in ('tokenizer.json','added_pieces.json','row_migration_plan.json','candidate_frozen_before_measurement.json'):
        path=candidate/name;require(identity['input_sha256'].get(str(path))==sha(path),'Original candidate file changed: '+name)
    source_identity=config['initial_domain_to_BPE_model_only_admission']
    require(source_identity['source_scientific_arguments']==source_recipe,'Original BPE admission science changed')
    established_source_equivalence(HERE/'runs'/config['run_id'],config['run_id'],source_identity,
        trainer_path=HERE/'train_lora_batched_bpe512_v2.py',runtime_path=HERE/'hf_bpe512_training_v2.py')
    sampler=identity['domain_sampling_identity']
    require(sampler.get('sampling_mode')=='domain30' and sampler.get('replay_seed')==recipe['replay_seed']
        and sampler.get('replay_rng_independent_of_turkish') is True,'Keep original independent domain/replay sampler identity')
    return identity


def validate_identity(value,plan,child_pid,started):
    # The exact-admission wrapper is a separate runtime role; original science pins stay untouched.
    runtime_plan={**plan,'source_sha256':{**plan['source_sha256'],
        'launch_with_memory_guard_bpe512_v2.py':fresh.RUNTIME_MEMORY_SHA}}
    result=domain.validate_identity(value,runtime_plan,child_pid,started)
    expected={fresh.RUNTIME_MEMORY_NAME:plan['runtime_memory_launcher']['sha256']}
    require(expected[fresh.RUNTIME_MEMORY_NAME]==fresh.RUNTIME_MEMORY_SHA,'Exact memory source differs')
    asset_records=value.get('BPE_explicit_asset_records',{})
    for flag in ('--bpe-source-checkpoint','--bpe-source-selection','--bpe-source-closure','--bpe-conversion-report',
                 '--native-backend-provenance','--resume-exact'):
        path=scoped(fresh.option(plan['worker_arguments'],flag));key=flag.removeprefix('--').replace('-','_')
        digest=(plan['snapshots']['latest']['snapshot']['sha256'] if flag=='--resume-exact'
            else plan['source_config_identity']['input_sha256'][str(path)])
        require(asset_records.get(key)=={'path':str(path),'sha256':digest},'Exact original asset declaration differs: '+flag)
    return result


def verify_pool_frontier(config,selection,history):
    counts={name:value['pool_clips'] for name,value in config['sampling']['initial_pool']['domain'].items()}
    require(counts=={'fleurs':2037,'cv':72846},'Retain original74883 domain pool counts')
    step=selection['completed_steps'];draws=step*8;nf=draws*3//10;expected={'fleurs':nf,'cv':draws-nf}
    coverage=selection['Turkish_domain_coverage']
    require(coverage.get('epoch_clock')=='minimum_domain_exposure'
        and coverage.get('turkish_draws')==draws,'Cumulative min-domain draw clock differs')
    for name,count in counts.items():
        row=coverage['domain'][name];require(row['pool_clips']==count and row['draws']==expected[name]
            and 0<=row['unique_clips_seen']<=min(count,row['draws'])
            and row['repeat_visits']==row['draws']-row['unique_clips_seen'],'Domain draw/unique/repeat accounting differs')
    epoch=min(expected[name]/count for name,count in counts.items())
    require(abs(selection['turkish_epochs_completed']-epoch)<1e-12
        and abs(coverage['minimum_domain_exposure_epochs']-epoch)<1e-12
        and coverage['turkish_unique_clips_seen']==sum(row['unique_clips_seen'] for row in coverage['domain'].values()),
        'Minimum-domain exposure/unique accounting differs')
    if history:require(history[-1]['Turkish_domain_coverage']==coverage,'Published history and durable sampler frontier differ')
    return coverage
''')
exact=exact.replace('domain.validate_identity(','validate_identity(')
# Keep original BPE equivalence proof pinned separately from live config, not re-run.
exact=exact.replace("'original_source_closure':record(scoped(fresh.option(argv,'--bpe-source-closure'))),", "'original_source_closure':record(scoped(fresh.option(argv,'--bpe-source-closure'))),\n        'source_equivalence':record(HERE/'runs'/args.run_id/'BPE512_actual_GPU_source_equivalence.json'),")
exact=exact.replace("        'memory_guard_training_launcher_bpe512_v1'", "        'memory_guard_training_launcher_bpe512_v1'")
exact=exact.replace("        require(memory.get(key)==memory.get(key+'_after')==SOURCE_PINS[name],", "        expected=fresh.RUNTIME_MEMORY_SHA if key=='memory_launcher_sha256' else SOURCE_PINS[name]\n        require(memory.get(key)==memory.get(key+'_after')==expected,")
start=exact.index("    expected=updates*config['tr_clips_per_update'] / config['turkish_clips_used']") if "    expected=updates*config['tr_clips_per_update'] / config['turkish_clips_used']" in exact else -1
if start>=0:raise RuntimeError('Unexpected uniform-frontier template')
# Old domain helper end uses the min-domain arithmetic. Replace it with explicit validated coverage.
start=exact.index("    expected=min(")
end=exact.index('    domain.chain_identity()',start)
exact=exact[:start]+"    coverage=verify_pool_frontier(config,selection,history)\n    expected=selection['turkish_epochs_completed']\n    require(record(scoped(plan['source_equivalence']['path']))==plan['source_equivalence'],\n        'Original actual300-clip source equivalence proof changed')\n"+exact[end:]
exact=exact.replace("'minimum_domain_exposure_epochs_completed':expected,", "'minimum_domain_exposure_epochs_completed':expected,'actual_pool_frontier':coverage,\n        'original_actual_source_equivalence':plan['source_equivalence'],'source_equivalence_rerun':False,")
save('resume_BPE512_same_science_v2.py',exact)

sup=(HERE/'supervise_TSC_same_science_v3.py').read_text(encoding='utf-8')
sup=sup.replace('run_hf_TSC_arm_v3','run_hf_bpe512_arm_v2').replace('resume_TSC_same_science_v3','resume_BPE512_same_science_v2').replace('launch_TSC_same_science_resume_v3','launch_BPE512_same_science_resume_v2')
sup=sup.replace('TSC','BPE512').replace('tsc_v3','bpe512_v2').replace('_v3','_v2')
sup=re.sub(r'^FIRST_SHA = .*$', 'FIRST_SHA = '+repr(first_sha),sup,flags=re.M)
sup=re.sub(r'^HELPER_SHA = .*$', 'HELPER_SHA = '+repr(sha(HERE/'resume_BPE512_same_science_v2.py')),sup,flags=re.M)
sup=re.sub(r'^PRELOADER_SHA = .*$', 'PRELOADER_SHA = '+repr(sha(HERE/'launch_BPE512_same_science_resume_v2.py')),sup,flags=re.M)
start=sup.index('    counts=domain.domain_counts(config)')
end=sup.index("    if state['kind']==EXACT_KIND:",start)
sup=sup[:start]+"    require(config.get('steps')==13009 and identity['training_arguments']['epochs']==1.,\n        'Preserve original one minimum-domain exposure horizon and fullstate frontier')\n"+sup[end:]
sup=sup.replace("    domain.validate_identity(identity,state,launcher,state['started_at_utc'])", "    if state['kind']==FIRST_KIND:domain.validate_identity(identity,state,launcher,state['started_at_utc'])\n    else:exact.validate_identity(identity,state,launcher,state['started_at_utc'])")
sup=sup.replace("    closure_path=scoped(preload.option(state['worker_arguments'],'--source-closure'))", "    closure_path=scoped(preload.option(state['worker_arguments'],'--bpe-source-closure'))")
sup=sup.replace("'configured_steps':state['configured_steps']", "'configured_steps':science['training_arguments']['steps']")
sup=sup.replace("'configured_steps':state['configured_steps']", "'configured_steps':science['training_arguments']['steps']")
sup=sup.replace("'minimum_domain_exposure_epoch_horizon':1", "'minimum_domain_exposure_epoch_horizon':1")
save('supervise_BPE512_same_science_v2.py',sup)
print(json.dumps({name:sha(HERE/name) for name in ('launch_BPE512_same_science_resume_v2.py','resume_BPE512_same_science_v2.py','supervise_BPE512_same_science_v2.py')}))
