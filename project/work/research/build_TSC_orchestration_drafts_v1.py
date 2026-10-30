"""Workspace-only source generation; no data/model/audio/checkpoint imports."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_function(source,name,replacement):
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.strip()+'\n'+''.join(lines[node.end_lineno:])
def new(name,source):
    path=HERE/name
    if path.exists():raise FileExistsError(path)
    ast.parse(source)
    path.write_text(source,encoding='utf-8',newline='\n')
    print(name,digest(path))

source=(HERE/'run_hf_domain30_arm_v1.py').read_text(encoding='utf-8')
source=source.replace('from domain30_training_identity_v1 import closed_source_identity, TOKENIZER_SHA, RECIPE',
    'from tsc_training_inputs_v1 import closed_source_identity, approve\nTOKENIZER_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"\nRECIPE = '+repr(ast.literal_eval(next(n.value for n in ast.parse((HERE/'domain30_training_identity_v1.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RECIPE' for t in n.targets)))))
for old,target in [('domain30_runtime_chain_pins_v1.json','TSC_runtime_chain_pins_v1.json'),
 ('isolated_HF_domain30_runtime_chain_pins_v1','isolated_TSC_runtime_chain_pins_v1'),
 ('launch_with_memory_guard_domain30_v1.py','launch_with_memory_guard_tsc_v1.py'),
 ('launch_with_cache_reclaim_domain30_v1.py','launch_with_cache_reclaim_tsc_v1.py'),
 ('train_lora_batched_domain30_v1.py','train_lora_batched_tsc_v1.py')]:source=source.replace(old,target)
source=source.replace("8b88e1b442a9ac1d946d005074b14f9f87f0a20c44420ce725be6dfc89a438dd",digest(HERE/'TSC_runtime_chain_pins_v1.json'))
source=replace_function(source,'prepare',r'''
def prepare(args):
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.loop_id),'Safe NEW TSC loop ID required')
    require(math.isfinite(args.window_hours) and .25<=args.window_hours<=3
        and math.isfinite(args.poll_seconds) and 5<=args.poll_seconds<=15,'Bounded3h /5–15s polls required')
    chain=chain_identity();run_id=args.loop_id+'_TSC';loop=HERE/'runs'/args.loop_id
    require(not loop.exists() and not (HERE/'runs'/run_id).exists(),'Fresh controller/run namespace required')
    require(all(not (ROOT/'work'/(run_id+'_'+role+'.pt')).exists() for role in ('best','latest')),'Fresh Adam/checkpoint namespace required')
    closure_path=scoped(args.source_closure);closure=read(closure_path,{})
    source_config=scoped(args.source_run_config);source_selection=scoped(args.source_selection)
    require(closure.get('source_run_config')==record(source_config)
        and closure.get('source_selection')==record(source_selection),'ROOT-chosen actual CLOSED source paths differ')
    config=read(source_config,{});selection=read(source_selection,{})
    require(config.get('base_revision')=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
        and config['config_identity']['extended_tokenizer_sha256']==TOKENIZER_SHA,'Original8199 closed source required')
    input_proof=scoped(args.tsc_input_proof);proof=read(input_proof,{})
    require(proof.get('kind')=='TSC_supplemental_TR_frame_target_inputs_v1' and proof.get('passed') is True
        and proof.get('training_launch_eligible') is True and proof.get('require_full_train_attempt') is True
        and type(proof.get('clips')) is int and proof['clips']>74883
        and proof.get('original_TR_clips')==74883 and proof.get('replay_clips')==1595,
        'Actual approved full TSC input preparation required; pilot preparation is not adoption')
    approval=scoped(args.tsc_training_approval)
    approve(approval,scoped(proof['references']['import_proof']['path']),training=True)
    source_recipe=config['config_identity']['training_arguments'];recipe=dict(source_recipe)
    recipe.update(epochs=1.,steps=math.ceil(proof['clips']/8))
    require(recipe.get('eval_every_updates')==250 and recipe.get('eval_epoch_fraction')==.25,
        'Preserve source DEV250/quarter-pass clocks')
    for key,expected in RECIPE.items():
        if key in recipe:require(recipe[key]==expected,'Closed source recipe differs: '+key)
    native=config['config_identity']['native_loss'];bound=native['max_joint_bytes']
    require(native.get('sigma')==0. and native.get('omega')==0.
        and native.get('caller_reduction')=='raw per-clip NLL / each target length, then physical-batch mean'
        and proof.get('max_full_joint_bytes',math.inf)<=bound,'Preserve whole native objective/bound; no row dropping')
    context=SimpleNamespace(**recipe,run_id=run_id,resume_weights=scoped(args.source_checkpoint),resume_exact=None,
        source_closure=closure_path,resume_runtime_migration=None,supplement=None,reuse_input_audit=None,
        tsc_input_proof=input_proof,native_max_joint_bytes=bound)
    identity=closed_source_identity(context)
    identity['turkish_wer']=selection['turkish_validation_wer']
    for name,digest in config['code_sha256'].items():require(sha(HERE/name)==digest,'Chosen CLOSED source implementation changed: '+name)
    argv=recipe_arguments(recipe)
    assets={'tr_manifest':scoped(proof['files']['combined_TR']['path']),
        'replay_manifest':scoped(proof['files']['original_replay']['path']),
        'tr_frame_metadata':scoped(proof['files']['TR_metadata']['path']),
        'tr_frame_metadata_proof':input_proof,'tsc_input_proof':input_proof,
        'tsc_training_approval':approval,'data_v3_proof':scoped(config['data_v3_proof'])}
    for key in ('combined_TR','original_replay','TR_metadata'):
        item=proof['files'][key];require(record(scoped(item['path']))==item,'Approved prepared input changed: '+key)
    require(config['input_sha256'][str(assets['data_v3_proof'])]==sha(assets['data_v3_proof']),
        'Preserve original frozen CV/FLEURS preparation proof')
    for key,path in assets.items():argv.extend(('--'+key.replace('_','-'),str(path)))
    paths={key:loop/(run_id+suffix) for key,suffix in {'native':'_native_backend.json','io':'_io.json',
        'cache':'_cache.json','memory':'_memory.json','identity':'_identity.json','stop':'_stop.txt','log':'.log'}.items()}
    argv.extend(('--run-id',run_id,'--resume-weights',str(context.resume_weights),'--source-closure',str(closure_path),
        '--native-backend-provenance',str(paths['native']),'--native-max-joint-bytes',str(bound),
        '--stop-file',str(paths['stop']),'--max-wall-seconds',str(args.window_hours*3600-300)))
    pins=chain['source_sha256'];outer=scoped(chain['outer_launcher']['path']);memory_path=scoped(chain['memory_launcher']['path'])
    memory=['--provenance',str(paths['io']),'--cache-provenance',str(paths['cache']),
        '--memory-provenance',str(paths['memory']),'--worker-identity',str(paths['identity']),
        '--cap-gb','8','--threshold-gb','4','--',*argv]
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(outer),
        '--native-provenance',str(paths['native']),'--memory-launcher',str(memory_path),
        '--memory-launcher-sha256',pins[memory_path.name],'--trainer-sha256',pins['train_lora_batched_tsc_v1.py'],'--',*memory]
    return {'kind':'isolated_HF_TSC_guarded_quality_arm_v1','loop_id':args.loop_id,'run_id':run_id,
        'source_identity':identity,'source_run_config':record(source_config),'copied_source_recipe':source_recipe,
        'actual_stage_recipe':recipe,'routing_receipt':record(CHAIN_PATH),'source_sha256':pins,
        'guard_library_sha256':GUARD_SHA,'worker_arguments':argv,'command':command,
        'paths':{k:str(p) for k,p in paths.items()},'controller_sha256':sha(__file__),'window_hours':args.window_hours,
        'TSC_input_proof':record(input_proof),'TSC_training_approval':record(approval),
        'combined_TR_clips':proof['clips'],'original_TR_clips':74883,'TSC_clips':proof['TSC_clips'],
        'configured_steps':recipe['steps'],'horizon':{'configured_updates':recipe['steps'],'uniform_combined_pool_epochs':1,
            'quota_basis':'ceil(approved combined TR unique clips /8)','legacy_uniform_or_domain_cap_copied':False},
        'declared_changes':['Append approved TSC TRAIN to exact original74883 prefix; original1595 replay unchanged',
            'Fresh Adam/sampler/RNG/local counters from explicitly chosen CLOSED qualifying original8199 weights',
            'One combined-pool pass; horizon calculated from actual approved clip count',
            'Existing shared RNG retains changed-pool replay-order nuisance; unchanged replay order is not claimed'],
        'global_best_replaced_by_control_initialization':False,'source_choice_rationale':args.source_choice_rationale,
        'exact_continuation_supported_by_this_outer':False,
        'budget_stop_continuation_policy':'Use separate reviewed SAME-run full-state preloader/controller/supervisor; never reset Adam on window ends',
        'heldout_TEST_evaluated':False,'objective_complete':False,'objective_status':'incomplete'}
''')
source=replace_function(source,'verify_completion',r'''
def verify_pool_frontier(config,selection,history,plan):
    count=plan['combined_TR_clips'];horizon=math.ceil(count/8);updates=selection['completed_steps']
    require(type(updates) is int and 0<=updates<=horizon and horizon==plan['configured_steps']
        and config.get('epochs')==1. and config.get('steps')==horizon
        and config.get('turkish_clips_used')==count and config.get('replay_clips_used')==1595
        and config.get('tr_clips_per_update')==8 and config.get('replay_clips_per_update')==8,
        'Actual whole combined-pool/horizon/effective16 geometry differs')
    require(len(history)==updates and [r['step'] for r in history]==list(range(1,updates+1)),
        'Actual ordered full history frontier differs')
    require(selection['turkish_epochs_completed']==updates*8/count,'Uniform epoch means TR draws / combined pool count')
    if history:
        require(history[-1]['turkish_clips_seen']==updates*8
            and history[-1]['turkish_unique_clips_seen']==min(updates*8,count),
            'Without-replacement unique versus draw counters differ')
    if updates==horizon:require(selection['turkish_epochs_completed']>=1.,'Claiming one pass requires all combined TR sources visited')
    return {'combined_TR_clips':count,'configured_steps':horizon,'TR_draws':updates*8,
        'unique_TR_seen':min(updates*8,count),'uniform_combined_pool_epochs_completed':updates*8/count}


def verify_completion(plan,selection):
    require(selection.get('run_id')==plan['run_id'] and selection.get('heldout_tests_evaluated') is False,
        'Durable DEV-only TSC selection required')
    paths={key:Path(value) for key,value in plan['paths'].items()}
    for key in ('io','cache','memory'):require(read(paths[key],{}).get('status')=='complete','Actual runtime completion missing: '+key)
    completion=read(paths['native'].with_suffix('.completion.json'),{})
    require(completion.get('kind')=='native_tdt_training_backend_completion_v3'
        and completion.get('status')=='complete' and completion.get('sources_unchanged') is True
        and completion.get('backend_ready_sha256')==sha(paths['native']),'Native startup/completion evidence missing')
    memory=read(paths['memory'],{})
    require(memory.get('kind')=='memory_guard_training_launcher_tsc_v1' and memory.get('cap_bytes')==8000000000
        and memory.get('cleanup_threshold_bytes')==4000000000 and memory.get('allocator_cap_applied') is True
        and memory.get('all_phase_owned_memory_telemetry',{}).get('thread_join_completed') is True,
        'Actual cap8/end4/all-phase memory completion required')
    require(read(paths['cache'],{})['cache_reclamation']['eligible_update_calls']==selection['completed_steps'],
        'Scoped cache hook must observe each fresh initial update')
    run=HERE/'runs'/plan['run_id'];config=read(run/'run_config.json',{});history=read(run/'training_history.json',[])
    require(config.get('warm_start',{}).get('mode')=='new_optimizer_warm_start'
        and config['config_identity']['training_arguments']==plan['actual_stage_recipe'],
        'Initial fresh-Adam science recipe differs')
    coverage=verify_pool_frontier(config,selection,history,plan)
    require(record(Path(config['tsc_input_proof']))==plan['TSC_input_proof']
        and record(Path(config['tsc_training_approval']))==plan['TSC_training_approval'],'TSC proof/approval changed')
    chain_identity()
    return {'run_config':record(run/'run_config.json'),'initial_development':record(run/'initial_development.json'),
        'memory_provenance':record(paths['memory']),'cache_provenance':record(paths['cache']),
        'native_completion':record(paths['native'].with_suffix('.completion.json')),'actual_pool_frontier':coverage}
''')
source=source.replace("'memory_launcher_sha256':'launch_with_memory_guard_tsc_v1.py',", "'memory_launcher_sha256':'launch_with_memory_guard_tsc_v1.py',")
source=source.replace("for key,name in names.items(): require(value.get(key)==plan['source_sha256'][name],'Actual candidate source differs: '+key)",
    "for key,name in names.items():\n        expected=plan.get('runtime_memory_launcher',{}).get('sha256',plan['source_sha256'][name]) if key=='memory_launcher_sha256' else plan['source_sha256'][name]\n        require(value.get(key)==expected,'Actual candidate source differs: '+key)\n    for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):\n        index=plan['worker_arguments'].index(flag);item=record(scoped(plan['worker_arguments'][index+1]))\n        require(value.get('TSC_explicit_asset_records',{}).get(flag.removeprefix('--').replace('-','_'))==item,'Owned TSC asset identity differs: '+flag)")
source=source.replace("state['domain_coverage']=selection['Turkish_domain_coverage']", "state['combined_pool_coverage']=state['actual_runtime_completion']['actual_pool_frontier']")
source=source.replace("state['minimum_domain_epochs_completed']", "state['uniform_combined_pool_epochs_completed']")
source=source.replace("selection['completed_steps']==13009", "selection['completed_steps']==state['configured_steps']")
source=source.replace("'configured_domain_horizon'", "'configured_TSC_horizon'")
source=source.replace("state['configured_domain_epoch_completed']", "state['configured_combined_pool_pass_completed']")
source=source.replace("result.add_argument('--replay-seed',type=int,required=True)",
    "for name in ('tsc-input-proof','tsc-training-approval'):result.add_argument('--'+name,type=Path,required=True)\n    result.add_argument('--source-choice-rationale',required=True)")
source=source.replace('isolated_HF_domain30_guarded_quality_arm_v1','isolated_HF_TSC_guarded_quality_arm_v1')
source=source.replace('domain30 arm','TSC data-only arm').replace('isolated domain30 arm','isolated TSC data-only arm')
source=source.replace('exact CLOSED source recipe, fresh Adam','exact CLOSED source recipe plus approved combined data/horizon, fresh Adam')
source=source.replace('domain quality claim','TSC quality claim').replace('domain30 arm','TSC arm')
source=source.replace('actual minimum-domain coverage recorded','actual uniform combined-pool coverage recorded')
source=source.replace('Review actual DEV/retention. If continuing same domain state, prepare separate exact native-ready identity route; this first-launch outer cannot exact-resume',
    'Review actual DEV/retention; scheduled budget stops use the reviewed SAME-run exact-state supervisor route, never fresh Adam')
source=source.replace('Root-only guarded FIRST domain30 arm from a CLOSED qualifying8199 source.','Root-only guarded FIRST approved TSC data-only arm from an explicit CLOSED qualifying8199 source.')
docend=source.index('"""',3)+3
source='''"""Guarded first approved TSC TRAIN data-only arm; no TEST/export or goal completion.

Copy the ROOT-chosen CLOSED qualifying source recipe. Change only the appended
TRAIN pool and one-pass horizon ceil(actual combined unique TR clips /8).
Fresh Adam initially; scheduled windows later preserve SAME-run full state.
The original replay pool is exact, but shared-RNG replay order may differ.
"""'''+source[docend:]
new('run_hf_TSC_arm_v1.py',source)
