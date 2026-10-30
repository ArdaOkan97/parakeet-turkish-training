"""Construct only a NEW controller from the reviewed BPE lifecycle boilerplate."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
source = (OUT / "run_hf_bpe512_arm_v1.py").read_text(encoding="utf-8")
target = OUT / "run_hf_domain30_arm_v1.py"
source = source.replace('"""Root-only bounded BPE512 quality arm from a CLOSED qualifying HF source.\n\nCopies the complete source recipe, starts fresh Adam/sampler/counters, and\nretains the existing Windows resource/progress guards. The isolated trainer\nrequires actual unsuppressed six-language GPU source equivalence before Adam.\nNo TEST, export, source continuation, or quality-goal completion occurs here.\n--validate-only uses the standard library and never launches a process.\n"""',
'''"""Root-only guarded FIRST domain30 arm from a CLOSED qualifying8199 source.

Same head/LoRA/native-loss/KD10 recipe; fresh Adam and independent replay RNG.
Horizon is one minimum-domain exposure epoch:13009 updates, never the legacy
uniform28082 cap. A3h external window can end earlier with latest+best saved.
This initial outer supports only fresh weights, so a budget-stopped candidate
needs a separately reviewed exact native-ready continuation strategy later.
No TEST, export or quality-goal completion. --validate-only launches no child.
"""''')
source = source.replace('from hf_bpe512_training_v1 import closed_source_identity, candidate_assets, BASELINE_SEMANTIC_SHA',
    'from domain30_training_identity_v1 import closed_source_identity, TOKENIZER_SHA, RECIPE')
source = source.replace("CHAIN_PATH = HERE / 'HF_BPE512_runtime_chain_pins_v2.json'", "CHAIN_PATH = HERE / 'domain30_runtime_chain_pins_v1.json'")
source = source.replace("CHAIN_SHA = 'f8bcb1ecc09ec30cab11a260dc8cb5f839994f26fb464b29040a41522055eb45'", "CHAIN_SHA = '8b88e1b442a9ac1d946d005074b14f9f87f0a20c44420ce725be6dfc89a438dd'")
source = source.replace('isolated_HF_BPE512_runtime_chain_pins_v2', 'isolated_HF_domain30_runtime_chain_pins_v1')
source = source.replace('bpe512', 'domain30').replace('BPE512', 'domain30')


def function(name, replacement):
    global source
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    source = "".join(lines[:node.lineno-1]) + replacement + "\n" + "".join(lines[node.end_lineno:])


function('prepare', '''def prepare(args):
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.loop_id),'Safe NEW loop ID required')
    require(math.isfinite(args.window_hours) and .25<=args.window_hours<=3
        and math.isfinite(args.poll_seconds) and 5<=args.poll_seconds<=15,
        'Explicit bounded0.25–3h window /5–15s polls required')
    require(type(args.replay_seed) is int,'Explicit independent replay seed required')
    chain=chain_identity()
    run_id=args.loop_id+'_domain30';loop=HERE/'runs'/args.loop_id
    require(not loop.exists() and not (HERE/'runs'/run_id).exists(),'Entire first-launch controller/run namespace must be new')
    require(all(not (ROOT/'work'/(run_id+'_'+role+'.pt')).exists() for role in ('best','latest')),'Fresh Adam/checkpoint namespace required')
    context=SimpleNamespace(run_id=run_id,resume_weights=scoped(args.source_checkpoint),resume_exact=None,
        source_closure=scoped(args.source_closure))
    identity=closed_source_identity(context)
    closure=read(context.source_closure,{})
    source_config=scoped(args.source_run_config)
    require(closure['source_run_config']==record(source_config)
        and closure['source_selection']==record(scoped(args.source_selection)),
        'Controller source paths must be the actual ROOT-closed source paths')
    config=read(source_config,{})
    require(config.get('run_id')==identity['source_run_id']
        and config.get('base_revision')=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
        and config['config_identity']['extended_tokenizer_sha256']==TOKENIZER_SHA,
        'Closed original8199 source configuration required')
    native=config['config_identity']['native_loss']
    require(native.get('sigma')==0.0 and native.get('omega')==0.0
        and native.get('caller_reduction')=='raw per-clip NLL / each target length, then physical-batch mean',
        'Preserve original native kernel/objective reduction policy')
    source_recipe=config['config_identity']['training_arguments']
    recipe=dict(source_recipe)
    # Explicit science horizon, not a copied uniform-pass cap. All other
    # original recipe fields remain identical, including DEV250/quarter clocks.
    recipe.update(epochs=1.,steps=13009)
    require(recipe.get('eval_every_updates')==250 and recipe.get('eval_epoch_fraction')==.25,
        'Retain the declared250-update and quarter-exposure feedback clocks')
    for key,expected in RECIPE.items():
        if key in recipe:require(recipe[key]==expected,'Closed recipe differs: '+key)
    argv=recipe_arguments(recipe)
    for name,digest in config['code_sha256'].items():require(sha(HERE/name)==digest,'Closed source implementation changed: '+name)
    paths={key:loop/(run_id+suffix) for key,suffix in {'native':'_native_backend.json','io':'_io.json',
        'cache':'_cache.json','memory':'_memory.json','identity':'_identity.json','stop':'_stop.txt','log':'.log'}.items()}
    for key in ('tr_manifest','replay_manifest','data_v3_proof','tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit'):
        if config.get(key):
            path=scoped(config[key]);expected=config['input_sha256'].get(str(path))
            require(expected is not None and sha(path)==expected,'Original data/replay/audit input changed: '+key)
            argv.extend(('--'+key.replace('_','-'),str(path)))
    argv.extend(('--run-id',run_id,'--resume-weights',str(context.resume_weights),
        '--source-closure',str(context.source_closure),'--sampling-mode','domain30','--fleurs-fraction','0.3',
        '--replay-seed',str(args.replay_seed),'--native-backend-provenance',str(paths['native']),
        '--native-max-joint-bytes',str(config.get('native_max_joint_bytes',1700000000)),
        '--stop-file',str(paths['stop']),'--max-wall-seconds',str(args.window_hours*3600-300)))
    pins=chain['source_sha256']
    memory=['--provenance',str(paths['io']),'--cache-provenance',str(paths['cache']),
        '--memory-provenance',str(paths['memory']),'--worker-identity',str(paths['identity']),
        '--cap-gb','8','--threshold-gb','4','--',*argv]
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HERE/chain['outer_launcher']),
        '--native-provenance',str(paths['native']),'--memory-launcher',str(HERE/chain['memory_launcher']),
        '--memory-launcher-sha256',pins[chain['memory_launcher']],
        '--trainer-sha256',pins['train_lora_batched_domain30_v1.py'],'--',*memory]
    return {'kind':'isolated_HF_domain30_guarded_quality_arm_v1','loop_id':args.loop_id,'run_id':run_id,
        'source_identity':identity,'source_run_config':record(source_config),'copied_source_recipe':source_recipe,
        'actual_stage_recipe':recipe,'routing_receipt':record(CHAIN_PATH),'source_sha256':pins,
        'guard_library_sha256':GUARD_SHA,'worker_arguments':argv,'command':command,
        'paths':{k:str(p) for k,p in paths.items()},'controller_sha256':sha(__file__),'window_hours':args.window_hours,
        'horizon':{'configured_updates':13009,'minimum_domain_epochs':1,
            'legacy_uniform_cap_copied':False,'epoch_quota_basis':'min(FLEURS_draws/2037,CV_draws/72846)',
            'at_full_horizon':{'TR_draws':104072,'FLEURS_draws':31221,'CV_draws':72851}},
        'declared_changes':['TR30%FLEURS/70%CV by clips, same accepted74883 pool',
            'Independent replay seed; original1595 labels/cache/teacher unchanged; NOT legacy shared-RNG replay equality',
            'Fresh Adam/sampler/RNG/local counters from latest ROOT-chosen CLOSED qualifying learned weights',
            'Epoch clock is minimum per-domain exposure; fixed DEV250 plus quarter-domain milestones'],
        'legacy_replay_RNG_nuisance_disclosed':True,'replay_seed':args.replay_seed,
        'matched_independent_replay_control_run_requested':False,
        'exact_continuation_supported_by_this_outer':False,
        'budget_stop_continuation_policy':'Preserve latest/best; later require separately reviewed exact native-ready identity route, never silently restart fresh Adam',
        'mathematical_trajectory_equivalence_claimed':False,'heldout_TEST_evaluated':False,
        'trackio':{'project':'parakeet-turkish','publication':'Existing Trackio v3 mirror follows normal config/history; local step0 is the closed warm start'},
        'objective_complete':False,'objective_status':'incomplete'}
''')

function('verify_completion', '''def verify_completion(plan,selection):
    require(selection.get('run_id')==plan['run_id'] and selection.get('heldout_tests_evaluated') is False,
        'Durable DEV-only domain selection required')
    paths={key:Path(value) for key,value in plan['paths'].items()}
    for key in ('io','cache','memory'):
        require(read(paths[key],{}).get('status')=='complete','Actual runtime completion missing: '+key)
    completion=read(paths['native'].with_suffix('.completion.json'),{})
    require(completion.get('status')=='complete' and completion.get('sources_unchanged') is True
        and completion.get('backend_ready_sha256')==sha(paths['native']),
        'Native startup/completion evidence missing')
    memory=read(paths['memory'],{})
    require(memory.get('kind')=='memory_guard_training_launcher_domain30_v1'
        and memory.get('cap_bytes')==8000000000 and memory.get('cleanup_threshold_bytes')==4000000000
        and memory.get('allocator_cap_applied') is True
        and memory.get('all_phase_owned_memory_telemetry',{}).get('thread_join_completed') is True,
        'Actual cap8/end4/all-phase sampler completion required')
    require(read(paths['cache'],{})['cache_reclamation']['eligible_update_calls']==selection['completed_steps'],
        'Scoped cache hook did not observe every fresh update')
    run=HERE/'runs'/plan['run_id'];config=read(run/'run_config.json',{})
    require(config.get('epochs')==1. and config.get('steps')==13009
        and config.get('sampling_mode')=='domain30' and config.get('replay_seed')==plan['replay_seed']
        and config.get('warm_start',{}).get('mode')=='new_optimizer_warm_start',
        'Actual domain horizon/fresh optimizer/scientificidentity differs')
    coverage=selection['Turkish_domain_coverage'];updates=selection['completed_steps']
    require(type(updates) is int and 0<=updates<=13009
        and coverage['turkish_draws']==updates*8
        and coverage['domain']['fleurs']['draws']==updates*8*3//10
        and coverage['domain']['cv']['draws']==updates*8-updates*8*3//10
        and selection['turkish_epoch_semantics']=='minimum_domain_exposure',
        'Actual domain counters are not the declared clip quota/minimum-domain clock')
    expected=min(coverage['domain']['fleurs']['draws']/2037,coverage['domain']['cv']['draws']/72846)
    require(selection['turkish_epochs_completed']==expected,'Domain completed epoch misreported')
    if updates==13009:
        require(expected>=1. and coverage['all_Turkish_pool_sources_seen'] is True,
            'Full domain epoch requires complete source coverage')
    chain_identity()
    return {'run_config':record(run/'run_config.json'),'initial_development':record(run/'initial_development.json'),
        'memory_provenance':record(paths['memory']),'cache_provenance':record(paths['cache']),
        'native_completion':record(paths['native'].with_suffix('.completion.json'))}
''')

source = source.replace("baseline=state['source_identity']['selection']['turkish_validation_wer']", "baseline=state['source_identity']['turkish_wer']")
source = source.replace("state['actual_source_equivalence']=verify_completion(state,selection)", "state['actual_runtime_completion']=verify_completion(state,selection)")
source = source.replace("'source_recipe':", "'source_recipe':")
source = source.replace("actual unsuppressed six-language", "actual six-language")
source = source.replace("Actual pre-Adam source equivalence pending; no BPE quality claim.", "Original8199 geometry and source closure validated; no domain quality claim.")
source = source.replace("Actual300-clip pre-Adam equivalence passed; no TEST/export or goal completion.", "Original8199 model; actual minimum-domain coverage recorded; no TEST/export or goal completion.")
source = source.replace("Actual pre-Adam source equivalence pending; no domain30 quality claim.", "Original8199 geometry and source closure validated; no domain quality claim.")
source = source.replace("isolated BPE arm", "isolated domain30 arm").replace("BPE arm", "domain30 arm")
source = source.replace("status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'quality_window_ended',", "status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'quality_window_ended',")
source = source.replace("        baseline=state['source_identity']['turkish_wer']", """        state['domain_coverage']=selection['Turkish_domain_coverage']
        state['minimum_domain_epochs_completed']=selection['turkish_epochs_completed']
        state['configured_domain_epoch_completed']=selection['completed_steps']==13009
        state['objective_complete']=False;state['objective_status']='incomplete'
        state['stop_category']=('resource' if state['resource_review_required'] else
            'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures' else
            'plateau' if selection['stop_early_reason']=='development_plateau' else
            'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds' else
            'configured_domain_horizon' if selection['stop_early_reason']=='configured_steps_complete' else selection['stop_early_reason'])
        state['next_action']=('ROOT review resource stop before any further GPU job' if state['resource_review_required'] else
            'Review actual DEV/retention. If continuing same domain state, prepare separate exact native-ready identity route; this first-launch outer cannot exact-resume')
        baseline=state['source_identity']['turkish_wer']""")

function('parser', '''def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--loop-id',required=True)
    for name in ('source-checkpoint','source-selection','source-closure','source-run-config'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--replay-seed',type=int,required=True)
    result.add_argument('--window-hours',type=float,default=3.)
    result.add_argument('--poll-seconds',type=float,default=5.)
    result.add_argument('--validate-only',action='store_true')
    result.add_argument('--preflight-output',type=Path)
    result.add_argument('--gpu-window-authorized',action='store_true')
    return result
''')
ast.parse(source)
if target.exists():raise FileExistsError('NEW controller only, never overwrite')
target.write_text(source,encoding='utf-8',newline='\n')
print(target)
