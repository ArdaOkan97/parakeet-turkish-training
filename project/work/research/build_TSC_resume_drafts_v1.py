"""New source drafts only, no model/checkpoint/audio/data imports."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def replace_function(source,name,replacement):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.strip()+'\n'+''.join(lines[node.end_lineno:])
def new(name,source):
    path=HERE/name
    if path.exists():raise FileExistsError(path)
    ast.parse(source);path.write_text(source,encoding='utf-8',newline='\n');print(name,sha(path))
pins=json.loads((HERE/'TSC_runtime_chain_pins_v1.json').read_text(encoding='utf-8'))['source_sha256']
source=(HERE/'launch_domain30_same_science_resume_v1.py').read_text(encoding='utf-8')
tree=ast.parse(source);pinnode=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
lines=source.splitlines(keepends=True)
source=''.join(lines[:pinnode.lineno-1])+'PINS='+repr(pins)+"\nRUNTIME_MEMORY_NAME='launch_with_memory_guard_TSC_resume_v1.py'\nRUNTIME_MEMORY_SHA='PENDING_REVIEWED_EXACT_MEMORY_SOURCE'\n"+''.join(lines[pinnode.end_lineno:])
for old,newvalue in [('domain30','TSC'),('domain','TSC'),('launch_with_memory_guard_TSC_v1.py','launch_with_memory_guard_tsc_v1.py'),
 ('train_lora_batched_TSC_v1.py','train_lora_batched_tsc_v1.py'),('launch_native_training_TSC_v1.py','launch_native_training_tsc_v1.py')]:source=source.replace(old,newvalue)
# The old immutable science chain contains lowercase tsc paths. Do not replace its keys.
tree=ast.parse(source);pinnode=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets));lines=source.splitlines(keepends=True)
source=''.join(lines[:pinnode.lineno-1])+'PINS='+repr(pins)+'\n'+''.join(lines[pinnode.end_lineno:])
source=source.replace("    require(option(worker,'--sampling-mode')=='TSC' and float(option(worker,'--fleurs-fraction'))==.3,\n        'Only unchanged TSC/30% FLEURS sampling may use this exact route')\n    int(option(worker,'--replay-seed'))\n","")
source=source.replace("    original=scoped(args.original_native_provenance)","    require(sha(HERE/RUNTIME_MEMORY_NAME)==RUNTIME_MEMORY_SHA,'Reviewed exact-only runtime memory admission changed')\n    original=scoped(args.original_native_provenance)")
source=source.replace("    scoped(option(worker,'--source-closure'))", "    scoped(option(worker,'--source-closure'))\n    for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):\n        path=scoped(option(worker,flag));key=flag.removeprefix('--').replace('-','_')\n        require(old.get('TSC_explicit_asset_records',{}).get(key)=={'path':str(path),'sha256':sha(path)},'Original approved TSC asset differs: '+flag)")
source=source.replace("'source_sha256':PINS,'launcher_sha256'", "'source_sha256':PINS,'runtime_memory_launcher':{'path':str(HERE/RUNTIME_MEMORY_NAME),'sha256':RUNTIME_MEMORY_SHA},'launcher_sha256'")
source=source.replace("str(HERE/'launch_with_memory_guard_tsc_v1.py')", "str(HERE/RUNTIME_MEMORY_NAME)")
source=source.replace("and all(sha(HERE/n)==h for n,h in PINS.items()))", "and all(sha(HERE/n)==h for n,h in PINS.items()) and sha(HERE/RUNTIME_MEMORY_NAME)==RUNTIME_MEMORY_SHA)")
new('launch_TSC_same_science_resume_v1.py',source)

source=(HERE/'resume_domain30_same_science_v1.py').read_text(encoding='utf-8')
for old,newvalue in [('run_hf_domain30_arm_v1','run_hf_TSC_arm_v1'),('launch_domain30_same_science_resume_v1','launch_TSC_same_science_resume_v1'),
 ('domain_same_run_exact_science_continuation_v1','TSC_same_run_exact_science_continuation_v1'),
 ('isolated_HF_domain30_guarded_quality_arm_v1','isolated_HF_TSC_guarded_quality_arm_v1'),
 ('domain_same_science_native_preload_resume','TSC_same_science_native_preload_resume'),
 ('memory_guard_training_launcher_domain30_v1','memory_guard_training_launcher_TSC_resume_v1'),
 ('cache_reclamation_training_launcher_domain30_v1','cache_reclamation_training_launcher_tsc_v1'),
 ('atomic_io_retry_domain30_v1','atomic_io_retry_tsc_v1'),
 ('train_lora_batched_domain30_v1.py','train_lora_batched_tsc_v1.py'),
 ('launch_with_cache_reclaim_domain30_v1.py','launch_with_cache_reclaim_tsc_v1.py')]:source=source.replace(old,newvalue)
source=source.replace("DOMAIN_CONTROLLER_SHA='429daedee13ad14ee803c197eee597078940ef63525b093fe881c2af88ac772f'", "TSC_CONTROLLER_SHA='"+sha(HERE/'run_hf_TSC_arm_v1.py')+"'")
source=source.replace('DOMAIN_CONTROLLER_SHA','TSC_CONTROLLER_SHA')
source=replace_function(source,'validate_config',r'''
def validate_config(config,argv):
    identity=config['config_identity'];recipe=identity['training_arguments']
    count=config['turkish_clips_used'];horizon=math.ceil(count/8)
    require(config.get('run_id')==fresh.option(argv,'--run-id') and type(count) is int and count>74883
        and identity['public_turkish_data_policy'].get('accepted_clips')==count
        and config.get('replay_clips_used')==1595,'SAME approved whole TSC combined run required')
    require(config.get('epochs')==recipe.get('epochs')==1. and config.get('steps')==recipe.get('steps')==horizon,
        'Keep one uniform combined-pool pass and original ceil(N/8) frontier')
    for name,expected in domain.RECIPE.items():
        require(recipe.get(name,config.get(name))==expected,'Original TSC scientific recipe differs: '+name)
    require(recipe.get('eval_every_updates')==250 and recipe.get('eval_epoch_fraction')==.25
        and identity.get('extended_tokenizer_sha256')==domain.TOKENIZER_SHA,'Retain DEV250/quarter clocks and8199 tokenizer')
    for name,value in recipe.items():
        flag='--'+name.replace('_','-')
        if value is None:require(flag not in argv,'Unexpected scientific argument: '+flag)
        elif type(value) is bool:require((flag in argv)==value,'Scientific boolean changed: '+flag)
        else:
            actual=fresh.option(argv,flag)
            require((actual==value if type(value) is str else type(value)(actual)==value),'Scientific argument changed: '+flag)
    for name,digest in config.get('code_sha256',{}).items():
        require(Path(name).name==name and sha(HERE/name)==digest,'Original worker source changed: '+name)
    for flag in ('--native-backend-provenance','--source-closure','--tsc-input-proof','--tsc-training-approval',
                 '--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):
        path=scoped(fresh.option(argv,flag))
        require(identity['input_sha256'].get(str(path))==sha(path),'Immutable original TSC input changed: '+flag)
    require('--resume-runtime-migration' not in argv and '--reuse-input-audit' not in argv,
        'No source/data/runtime migration or hidden reuse in this SAME-science path')
    return identity
''')
source=source.replace("        'domain_sampling_identity':config['config_identity']['domain_sampling_identity'],\n        'replay_seed':config['replay_seed'],'sampling_mode':'domain30','configured_steps':13009,", "        'combined_TR_clips':config['turkish_clips_used'],'configured_steps':config['steps'],\n        'TSC_input_proof':record(scoped(config['tsc_input_proof'])),'TSC_training_approval':record(scoped(config['tsc_training_approval'])),\n        'runtime_memory_launcher':record(HERE/fresh.RUNTIME_MEMORY_NAME),")
source=source.replace("and 0<=len(history)<13009", "and 0<=len(history)<config['steps']")
source=source.replace("and plan['initial_step']<=len(history)<=13009", "and plan['initial_step']<=len(history)<=plan['configured_steps']")
source=source.replace("'launch_with_memory_guard_domain30_v1.py'", "fresh.RUNTIME_MEMORY_NAME")
source=source.replace("        require(memory.get(key)==memory.get(key+'_after')==SOURCE_PINS[name],", "        expected=plan['runtime_memory_launcher']['sha256'] if key=='memory_launcher_sha256' else SOURCE_PINS[name]\n        require(memory.get(key)==memory.get(key+'_after')==expected,")
start=source.index("    coverage=selection['Turkish_domain_coverage']")
end=source.index('    domain.chain_identity()',start)
source=source[:start]+"    coverage=domain.verify_pool_frontier(config,selection,history,plan)\n    expected=selection['turkish_epochs_completed']\n"+source[end:]
source=source.replace("'minimum_domain_epochs_completed':expected", "'uniform_combined_pool_epochs_completed':expected")
source=source.replace("stop_early_reason=selection['stop_early_reason'],domain_coverage=selection['Turkish_domain_coverage'],", "stop_early_reason=selection['stop_early_reason'],combined_pool_coverage=state['actual_runtime_completion']['actual_pool_frontier'],")
source=source.replace("minimum_domain_epochs_completed=selection['turkish_epochs_completed'],", "uniform_combined_pool_epochs_completed=selection['turkish_epochs_completed'],")
source=source.replace("'configured_domain_horizon'", "'configured_TSC_horizon'")
source=source.replace("'runtime_completion':record(completion_path)", "'actual_pool_frontier':coverage,'runtime_completion':record(completion_path)")
source=source.replace("        and value.get('source_sha256')==fresh.PINS", "        and value.get('source_sha256')==fresh.PINS\n        and value.get('runtime_memory_launcher')==plan['runtime_memory_launcher']")
source=source.replace("'Original domain first-launch controller changed'", "'Original TSC first-launch controller changed'")
source=source.replace('SAME-run exact domain30 continuation','SAME-run exact TSC combined-pool continuation')
source=source.replace('First domain window or a later SAME-domain exact window required','First TSC window or later SAME-TSC exact window required')
source=source.replace('Frozen domain first-launch controller changed','Frozen TSC first-launch controller changed')
source=source.replace('Original domain parent source pin differs','Original TSC parent source pin differs')
new('resume_TSC_same_science_v1.py',source)
