"""INERT until root confirms a closed parent: SAME-run exact BPE512 domain30 continuation.

No scientific policy changes, source clones, state rewriting, or live checkpoint
copying. --prepare-only is stdlib read-only. Actual closed-source preflight
archives byte-identical full latest/BEST and metadata before a later authorized
launch. Resource/progress guards remain separate from strict DEV selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read,write_json as write
import run_hf_bpe512_arm_v2 as domain
from resource_watchdog_v2 import ResourceGuard,sample_memory
import launch_BPE512_same_science_resume_v2 as fresh

HERE,ROOT=domain.HERE,domain.ROOT
SOURCE_PINS=fresh.PINS
BPE_CONTROLLER_SHA='349bbba7dcad63f5d2a8d2bd30515e7dffaf6f9f876386018dc8db46cb195622'


def require(ok,message):
    if not ok:raise ValueError(message)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def record(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def scoped(path):return domain.scoped(path)


def replace_option(argv,flag,value):
    require(argv.count(flag)<=1,'Duplicated worker option: '+flag)
    if flag in argv:
        index=argv.index(flag);require(index+1<len(argv),'Missing worker value: '+flag)
        del argv[index:index+2]
    if value is not None:argv.extend((flag,str(value)))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def validate_config(config,argv):
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
    if step==13009:require(epoch>=1. and coverage['all_Turkish_pool_sources_seen'] is True,
        'Completed minimum-domain horizon requires actual full source coverage')
    if history:require(history[-1]['Turkish_domain_coverage']==coverage,'Published history and durable sampler frontier differ')
    return coverage


def source_plan(args,require_closed):
    require(math.isfinite(args.window_hours) and .25<=args.window_hours<=3
        and math.isfinite(args.poll_seconds) and 5<=args.poll_seconds<=15,
        'Bounded0.25-3h / fresh5-15s polling required')
    require(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id),'Safe SAME run ID required')
    source_loop=scoped(args.source_loop);folder=scoped(args.output_dir)
    require(folder!=source_loop and not folder.is_relative_to(HERE/'runs'/args.run_id),
        'NEW continuation evidence directory outside original run required')
    require(not (folder/'loop_status.json').exists(),'Never overwrite an existing continuation state')
    state=read(source_loop/'loop_status.json',{})
    require(state.get('kind') in ('isolated_HF_BPE512_guarded_quality_arm_v2',
        'BPE512_same_run_exact_science_continuation_v2')
        and state.get('run_id')==args.run_id and state.get('worker_arguments'),
        'First BPE512 window or a later SAME-domain exact window required')
    arm=state
    if require_closed:
        require(args.source_closed_confirmed,'Root must confirm the owned parent and worker are closed')
        require(state.get('status') in ('quality_window_ended','train_window_ended','resource_review_required')
            and arm.get('child_exit_code')==0,
            'Terminal parent and successfully saved closed child required')
        require(not state.get('resource_review_required') or args.resource_review_cleared,
            'Root must explicitly clear a resource review before this exact continuation')
    require(sha(HERE/'run_hf_bpe512_arm_v2.py')==BPE_CONTROLLER_SHA,
        'Frozen BPE512 first-launch controller changed')
    domain.chain_identity()
    for name,digest in SOURCE_PINS.items():require(sha(HERE/name)==digest,'Frozen same-science runtime source changed: '+name)
    config_path=HERE/'runs'/args.run_id/'run_config.json'
    config=read(config_path,{})
    original=list(arm['worker_arguments'])
    validate_config(config,original)
    original_ready=scoped(config['native_backend_provenance'])
    paths={key:folder/(args.run_id+suffix) for key,suffix in {'fresh_runtime':'_fresh_runtime.json','io':'_io.json',
        'cache':'_cache.json','memory':'_memory.json','identity':'_identity.json','stop':'_stop.txt','log':'.log'}.items()}
    require(all(not p.exists() for p in paths.values()),'New runtime/evidence paths required')
    latest=ROOT/'work'/(args.run_id+'_latest.pt');best=ROOT/'work'/(args.run_id+'_best.pt')
    archive=folder/'immutable_closed_latest.pt'
    argv=list(original)
    for flag,value in (('--resume-weights',None),('--resume-exact',archive),('--stop-file',paths['stop']),
        ('--max-wall-seconds',max(1,args.window_hours*3600-300))):replace_option(argv,flag,value)
    require('--resume-runtime-migration' not in argv,'Same-science path supports the warm-start scientific arm without migration')
    allowed=('--resume-weights','--resume-exact','--stop-file','--max-wall-seconds')
    def unchanged(values):
        values=list(values)
        for flag in allowed:replace_option(values,flag,None)
        return values
    require(unchanged(original)==unchanged(argv),'Scientific/source/data/ready identity argument changed')
    require(scoped(fresh.option(argv,'--native-backend-provenance'))==original_ready,
        'Keep original immutable ready path; fresh proof must be separate')
    memory=['--provenance',str(paths['io']),'--cache-provenance',str(paths['cache']),'--memory-provenance',str(paths['memory']),
        '--worker-identity',str(paths['identity']),'--cap-gb','8','--threshold-gb','4','--',*argv]
    original_sha=sha(original_ready)
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HERE/'launch_BPE512_same_science_resume_v2.py'),
        '--original-native-provenance',str(original_ready),'--original-native-provenance-sha256',original_sha,
        '--fresh-runtime-provenance',str(paths['fresh_runtime']),'--',*memory]
    namespace=argparse.Namespace(original_native_provenance=original_ready,original_native_provenance_sha256=original_sha,
        fresh_runtime_provenance=paths['fresh_runtime'],memory_arguments=memory,validate_only=True)
    fresh.validate(namespace)
    plan={'schema':1,'kind':'BPE512_same_run_exact_science_continuation_v2','passed':True,'run_id':args.run_id,
        'loop_id':folder.name,'source_loop':str(source_loop),'original_run_config':record(config_path),
        'original_ready':record(original_ready),'paths':{k:str(p) for k,p in paths.items()},'command':command,
        'runtime_memory_launcher':record(Path(fresh.__file__).with_name(fresh.RUNTIME_MEMORY_NAME)),
        'worker_arguments':argv,'source_sha256':SOURCE_PINS,'new_preloader':record(Path(fresh.__file__)),
        'controller_sha256':sha(__file__),'scientific_config_identity_sha256':hashlib.sha256(json.dumps(config['config_identity'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
        'source_config_identity':config['config_identity'],'scientific_arguments_unchanged':True,
        'original_source_closure':record(scoped(fresh.option(argv,'--bpe-source-closure'))),
        'source_equivalence':record(HERE/'runs'/args.run_id/'BPE512_actual_GPU_source_equivalence.json'),
        'domain_sampling_identity':config['config_identity']['domain_sampling_identity'],
        'replay_seed':config['replay_seed'],'sampling_mode':'domain30','configured_steps':13009,
        'max_retention_failures_unchanged':config['config_identity']['training_arguments']['max_retention_failures'],
        'same_run_id':True,'optimizer_sampler_RNG_counters_reset':False,
        'original_ready_is_not_current_PID_evidence':True,'fresh_actual_preload_before_Torch_required':True,
        'archive_path':str(archive),'source_latest_path':str(latest),'source_best_path':str(best),
        'source_closure_confirmed':require_closed,'resource_review_cleared':args.resource_review_cleared,
        'heldout_TEST_evaluated':False,'objective_complete':False,
        'CUDA_initialized_by_preflight':False}
    if not require_closed:return plan
    selection_path=HERE/'runs'/args.run_id/'selection.json';history_path=HERE/'runs'/args.run_id/'training_history.json'
    selection=read(selection_path,{});history=read(history_path,[])
    require(selection.get('run_id')==args.run_id and selection.get('accepted') is True
        and selection.get('completed_steps')==len(history) and 0<=len(history)<13009
        and [row['step'] for row in history]==list(range(1,len(history)+1)),
        'Closed ordered history/qualifying BEST before configured horizon required')
    require(selection.get('heldout_tests_evaluated') is False
        and selection.get('stop_early_reason') in ('soft_max_wall_seconds','controller_stop_file'),
        'Do not silently reset a quality stop or resume a finished scientific horizon')
    require(state.get('completed_steps')==selection['completed_steps']
        and state.get('selected_step')==selection['selected_step']
        and state.get('selection')==record(selection_path),
        'Closed parent durable selection frontier changed')
    if state['kind']=='isolated_HF_BPE512_guarded_quality_arm_v2':
        require(state.get('controller_sha256')==BPE_CONTROLLER_SHA
            and state.get('source_sha256')==SOURCE_PINS,'Original BPE512 parent source pin differs')
        prior_runtime_completion=domain.verify_completion(state,selection)
    else:
        require(state.get('controller_sha256')==sha(__file__)
            and state.get('source_sha256')==SOURCE_PINS,'Previous exact-window source pin differs')
        prior_paths={k:Path(v) for k,v in state['paths'].items()}
        prior_identity=validate_identity(read(prior_paths['identity'],{}),state,
            state['owned_launcher_pid'],state['started_at_utc'])
        prior_runtime_completion=verify_completion(state,selection,prior_identity)
    require(Path(selection['latest_weights']).resolve()==latest and Path(selection['weights']).resolve()==best,
        'Actual full latest and qualifying work BEST paths required')
    folder.mkdir(parents=True,exist_ok=True)
    sources={'run_config':config_path,'history':history_path,'selection':selection_path,'parent_status':source_loop/'loop_status.json',
        'latest':latest,'best':best}
    snapshots={}
    for role,path in sources.items():
        target=archive if role=='latest' else folder/('immutable_source_'+role+path.suffix)
        source_sha=sha(path)
        if target.exists():require(sha(target)==source_sha,'Prepared closed source snapshot changed: '+role)
        else:shutil.copyfile(path,target)
        require(sha(target)==source_sha==sha(path),'Closed snapshot/source byte copy differs: '+role)
        snapshots[role]={'source':{'path':str(path),'sha256':source_sha},'snapshot':record(target)}
    plan.update(initial_step=selection['completed_steps'],inherited_selected_step=selection['selected_step'],
        snapshots=snapshots,inherited_history_sha256=canonical_sha(history),
        prior_runtime_completion=prior_runtime_completion,
        original_config_identity_preserved_bytewise=True,closed_full_checkpoint_archived_without_state_rewrite=True)
    return plan


def validate_fresh_runtime(value,plan,identity):
    require(value.get('kind')=='BPE512_same_science_native_preload_resume_v2' and value.get('status')=='ready'
        and value.get('run_id')==plan['run_id'] and value.get('pid')==identity['pid']
        and value.get('worker_arguments_sha256')==domain.guards.argument_sha(plan['worker_arguments'])
        and value.get('launcher_sha256')==plan['new_preloader']['sha256']
        and value.get('source_sha256')==fresh.PINS
        and value.get('runtime_memory_launcher')==plan['runtime_memory_launcher']
        and value.get('original_first_start_ready',{}).get('sha256')==plan['original_ready']['sha256']
        and value.get('original_first_start_ready',{}).get('path')==plan['original_ready']['path']
        and value.get('scientific_config_identity_changed') is False
        and value.get('fresh_runtime_proof_separate_from_original_ready') is True,
        'Actual fresh runtime proof/PID/arguments differ; original ready PID is not fresh evidence')
    preload=value.get('preload',{})
    require(preload.get('kind')=='coherent_native_cuda12_9_preload_v1'
        and preload.get('module_sha256')==SOURCE_PINS['preload_nemo_linker_v1.py']
        and preload.get('actual_binding_linker',{}).get('version')==[12,9]
        and preload.get('cuda_initialized') is False,'Fresh actual coherent pre-Torch preload missing')
    return value


def verify_completion(plan,selection,identity):
    paths={key:Path(value) for key,value in plan['paths'].items()}
    completion_path=paths['fresh_runtime'].with_suffix('.completion.json')
    proof=validate_fresh_runtime(read(paths['fresh_runtime'],{}),plan,identity)
    completion=read(completion_path,{})
    require(completion.get('kind')=='BPE512_same_science_native_preload_resume_completion_v2'
        and completion.get('status')=='complete' and completion.get('run_id')==plan['run_id']
        and completion.get('sources_unchanged') is True and completion.get('pid')==proof['pid']
        and completion.get('launcher_sha256_after')==plan['new_preloader']['sha256']
        and completion.get('fresh_runtime_proof_sha256')==sha(paths['fresh_runtime'])
        and completion.get('original_first_start_ready_sha256_after')==plan['original_ready']['sha256'],
        'Actual fresh runtime completion missing or mismatched')
    run=HERE/'runs'/plan['run_id'];config=read(run/'run_config.json',{})
    validate_config(config,plan['worker_arguments'])
    require(config['config_identity']==plan['source_config_identity']
        and config['warm_start']['mode']=='exact_resume'
        and scoped(config['warm_start']['path'])==Path(plan['archive_path'])
        and config['warm_start']['sha256']==plan['snapshots']['latest']['snapshot']['sha256']
        and config['warm_start']['source_run_id']==plan['run_id']
        and config['warm_start']['source_step']==plan['initial_step'],
        'Exact full-state warm-start/config identity changed')
    history=read(run/'training_history.json',[])
    require(selection.get('run_id')==plan['run_id'] and selection.get('accepted') is True
        and selection.get('heldout_tests_evaluated') is False
        and selection.get('completed_steps')==len(history)
        and plan['initial_step']<=len(history)<=13009
        and [row['step'] for row in history]==list(range(1,len(history)+1))
        and canonical_sha(history[:plan['initial_step']])==plan['inherited_history_sha256'],
        'Exact inherited ordered history/qualifying BEST/durable frontier differs')
    require(scoped(selection['latest_weights'])==ROOT/'work'/(plan['run_id']+'_latest.pt')
        and scoped(selection['weights'])==ROOT/'work'/(plan['run_id']+'_best.pt'),
        'Actual full latest and qualifying BEST must retain SAME run namespace')
    for role in ('latest','best','run_config','history','selection','parent_status'):
        item=plan['snapshots'][role]['snapshot']
        require(sha(item['path'])==item['sha256'],'Immutable closed-state snapshot changed: '+role)
    require(sha(plan['original_source_closure']['path'])==plan['original_source_closure']['sha256'],
        'Original qualifying warm-start closure changed')
    memory=read(paths['memory'],{});cache=read(paths['cache'],{});io=read(paths['io'],{})
    require(memory.get('status')=='complete'
        and memory.get('kind')=='memory_guard_training_launcher_bpe512_v1'
        and memory.get('cap_bytes')==8000000000 and memory.get('cleanup_threshold_bytes')==4000000000
        and memory.get('allocator_cap_applied') is True
        and memory.get('training_math_changed') is False and memory.get('backend_flags_changed') is False
        and memory.get('all_phase_owned_memory_telemetry',{}).get('thread_join_completed') is True,
        'Unchanged cap8/end4/math/backend/all-phase telemetry completion required')
    for key,name in {'memory_launcher_sha256':'launch_with_memory_guard_BPE512_resume_v2.py',
        'cache_launcher_sha256':'launch_with_cache_reclaim_bpe512_v2.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
        'resource_helper_sha256':'resource_watchdog_v2.py','frozen_worker_sha256':'train_lora_batched_bpe512_v2.py'}.items():
        expected=fresh.RUNTIME_MEMORY_SHA if key=='memory_launcher_sha256' else SOURCE_PINS[name]
        require(memory.get(key)==memory.get(key+'_after')==expected,
            'Before/after owned runtime source differs: '+key)
    for key,name in {'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
        'data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():
        require(memory.get(key)==memory.get(name.removesuffix('.py')+'_sha256_after')==SOURCE_PINS[name],
            'Before/after telemetry or data policy differs: '+key)
    updates=selection['completed_steps'];new_updates=updates-plan['initial_step']
    require(cache.get('kind')=='cache_reclamation_training_launcher_bpe512_v1'
        and cache.get('status')=='complete'
        and cache.get('frozen_worker_sha256')==SOURCE_PINS['train_lora_batched_bpe512_v2.py']
        and cache['cache_reclamation']['eligible_update_calls']==new_updates
        and cache['cache_reclamation']['failed_reclamations']==0,
        'Scoped cache hook must observe only new updates, not inherited full history')
    require(io.get('kind')=='atomic_io_retry_bpe512_v1' and io.get('status')=='complete'
        and io.get('worker_sha256')==io.get('worker_sha256_after')==SOURCE_PINS['train_lora_batched_bpe512_v2.py'],
        'Actual I/O completion worker source differs')
    coverage=selection['Turkish_domain_coverage']
    require(selection.get('sampling_identity')==plan['domain_sampling_identity']
        and coverage['turkish_draws']==updates*8
        and coverage['domain']['fleurs']['draws']==updates*8*3//10
        and coverage['domain']['cv']['draws']==updates*8-updates*8*3//10
        and selection.get('turkish_epoch_semantics')=='minimum_domain_exposure',
        'Restored domain/replay sampler identity and cumulative domain quotas differ')
    coverage=verify_pool_frontier(config,selection,history)
    expected=selection['turkish_epochs_completed']
    require(record(scoped(plan['source_equivalence']['path']))==plan['source_equivalence'],
        'Original actual300-clip source equivalence proof changed')
    domain.chain_identity()
    require(sha(Path(fresh.__file__))==plan['new_preloader']['sha256'],'Fresh preloader changed during this window')
    return {'new_optimizer_end_calls':new_updates,'inherited_steps':plan['initial_step'],
        'cumulative_completed_steps':updates,'minimum_domain_epochs_completed':expected,'actual_pool_frontier':coverage,
        'original_actual_source_equivalence':plan['source_equivalence'],'source_equivalence_rerun':False,
        'inherited_history_prefix_unchanged':True,'scientific_config_identity_unchanged':True,
        'runtime_completion':record(completion_path),'config':record(run/'run_config.json'),
        'history':record(run/'training_history.json'),'original_native_ready':plan['original_ready'],
        'fresh_runtime':record(paths['fresh_runtime']),'memory':record(paths['memory']),
        'cache':record(paths['cache']),'io':record(paths['io']),
        'full_Adam_sampler_RNG_restore_enforced_by_frozen_worker':True,
        'independent_checkpoint_tensor_audit_performed':False}


def main(args):
    plan=source_plan(args,not args.prepare_only)
    if args.prepare_only or args.validate_only:
        if args.preflight_output:
            output=scoped(args.preflight_output)
            require(not output.exists(),'Preflight evidence must be new')
            write(output,plan)
        print(json.dumps({'passed':True,'run_id':args.run_id,'source_closed':plan['source_closure_confirmed'],
            'checkpoint_copied':not args.prepare_only,'GPU_launch':False,'Torch_imported':'torch' in sys.modules}))
        return 0
    require(args.gpu_window_authorized,'Only root may authorize exclusive GPU ownership')
    folder=scoped(args.output_dir);paths={k:Path(v) for k,v in plan['paths'].items()}
    state={**plan,'status':'starting','started_at_utc':now(),'window_hours':args.window_hours,'objective_status':'incomplete'}
    deadline=time.monotonic()+args.window_hours*3600;started=time.monotonic();child=None;stop_at=None;stop_reason=None
    resource=ResourceGuard();progress=domain.guards.ProgressGuard(started)
    state['awake_request']=domain.guards.prior.keep_awake(True)
    try:
        write(folder/'loop_status.json',state)
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(plan['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['owned_launcher_pid']=child.pid
            while child.poll() is None:
                history=read(HERE/'runs'/args.run_id/'training_history.json',[])
                identity=validate_identity(read(paths['identity'],{}),plan,child.pid,state['started_at_utc']) if paths['identity'].exists() else None
                if identity:
                    state['fresh_runtime']=validate_fresh_runtime(read(paths['fresh_runtime'],{}),plan,identity)
                phase=domain.guards.log_phase(paths['log'],history)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=domain.guards.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
                progress_reason,progress_info=progress.observe(history,time.monotonic())
                reason=reason or self_reason or progress_reason
                if not identity and time.monotonic()-started>120:reason=reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic()-started>120:reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
                if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'
                state.update(status='running',phase=phase,published_step=history[-1]['step'] if history else plan['initial_step'],
                    system_and_owned_private_memory=observation,actual_worker_self_memory=self_observation,
                    progress=progress_info,updated_at_utc=now())
                with (folder/'memory_observations.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info})+'\n')
                if reason and stop_at is None:
                    stop_at=time.monotonic();stop_reason=reason
                    paths['stop'].write_text('Cooperative save/DEV requested: '+reason+'\n',encoding='utf-8')
                    state.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
                write(folder/'loop_status.json',state)
                if stop_at and time.monotonic()-stop_at>300:raise RuntimeError('Cooperative grace exceeded; no further GPU child')
                time.sleep(args.poll_seconds)
        state['child_exit_code']=child.returncode
        require(child.returncode==0,'Exact-resume worker failed; preserve and review all evidence')
        selection=read(HERE/'runs'/args.run_id/'selection.json',{})
        identity=validate_identity(read(paths['identity'],{}),plan,child.pid,state['started_at_utc'])
        state['actual_runtime_completion']=verify_completion(plan,selection,identity)
        state.update(status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'train_window_ended',
            child_exit_code=child.returncode,completed_steps=selection['completed_steps'],selected_step=selection['selected_step'],
            selection=record(HERE/'runs'/args.run_id/'selection.json'),finished_at_utc=now(),
            stop_early_reason=selection['stop_early_reason'],domain_coverage=selection['Turkish_domain_coverage'],
            minimum_domain_epochs_completed=selection['turkish_epochs_completed'],
            resource_review_required=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary'),objective_complete=False,
            next_action='ROOT reviews DEV/retention and the active quality goal; exact state is saved, no TEST or export performed')
        state['stop_category']=('resource' if state['resource_review_required'] else
            'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures' else
            'plateau' if selection['stop_early_reason']=='development_plateau' else
            'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds' else
            'configured_domain_horizon' if selection['stop_early_reason']=='configured_steps_complete' else selection['stop_early_reason'])
        return 1 if state['resource_review_required'] else 0
    except BaseException as error:
        state.update(status='failed',error=repr(error),objective_complete=False,finished_at_utc=now())
        raise
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists():paths['stop'].write_text('Controller exit: cooperative save/DEV requested\n',encoding='utf-8')
            try:child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)
        state['awake_clear']=domain.guards.prior.keep_awake(False)
        write(folder/'loop_status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-closed-confirmed',action='store_true')
    p.add_argument('--resource-review-cleared',action='store_true')
    p.add_argument('--window-hours',type=float,default=3.)
    p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--preflight-output',type=Path)
    p.add_argument('--gpu-window-authorized',action='store_true')
    sys.exit(main(p.parse_args()))
