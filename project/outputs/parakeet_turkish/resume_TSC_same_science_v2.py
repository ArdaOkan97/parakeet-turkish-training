"""INERT until root confirms a closed parent: SAME-run exact TSC combined-pool continuation.

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
import run_hf_TSC_arm_v2 as domain
from resource_watchdog_v2 import ResourceGuard,sample_memory
import launch_TSC_same_science_resume_v2 as fresh

HERE,ROOT=domain.HERE,domain.ROOT
SOURCE_PINS=fresh.PINS
TSC_CONTROLLER_SHA = '9eafb1d474c05f64a838105209c22756fdb80c9ec0d5e17d4ae20ffcf294697a'


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
    count=config['turkish_clips_used'];horizon=math.ceil(count/8)
    require(config.get('run_id')==fresh.option(argv,'--run-id') and type(count) is int and count>74883
        and identity['public_turkish_data_policy'].get('accepted_clips')==count
        and config.get('replay_clips_used')==1595,'SAME approved whole TSC combined run required')
    require(config.get('epochs')==recipe.get('epochs')==1. and config.get('steps')==recipe.get('steps')==horizon,
        'Keep one uniform combined-pool pass and original ceil(N/8) frontier')
    for name,expected in domain.RECIPE.items():
        require(recipe.get(name,config.get(name))==expected,'Original TSC scientific recipe differs: '+name)
    closure=read(scoped(fresh.option(argv,'--source-closure')), {})
    original_source=closure['source_run_config']
    require(record(scoped(original_source['path']))==original_source,'Original closed source config changed')
    original_source_config=read(scoped(original_source['path']),{})
    source_recipe=original_source_config['config_identity']['training_arguments']
    original_common,source_sampler_transition=domain.common_source_recipe(source_recipe)
    require(domain.approved_common_source_recipe(original_source_config,closure)==original_common,
        'Original ROOT-approved source sampler transition changed')
    require(all(recipe.get(key)==original_common[key] for key in domain.RECIPE_KEYS if key not in ('epochs','steps')),
        'Keep every common original source science field; only the initial declared data/sampler/horizon transition is permitted')
    require(recipe.get('eval_every_updates')==source_recipe.get('eval_every_updates')==250
        and recipe.get('eval_epoch_fraction')==source_recipe.get('eval_epoch_fraction')
        and identity.get('extended_tokenizer_sha256')==domain.TOKENIZER_SHA,'Retain actual source DEV250/fraction clocks and8199 tokenizer')
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
    input_proof=read(scoped(fresh.option(argv,'--tsc-input-proof')), {})
    quarantine=input_proof.get('references',{}).get('quarantine_proof',{})
    require(quarantine and record(scoped(quarantine['path']))==quarantine,
        'Keep original post-import quarantine bytes and approved combined-pool identity')
    require('--resume-runtime-migration' not in argv and '--reuse-input-audit' not in argv,
        'No source/data/runtime migration or hidden reuse in this SAME-science path')
    return identity


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
    require(state.get('kind') in ('isolated_HF_TSC_guarded_quality_arm_v2',
        'TSC_same_run_exact_science_continuation_v2')
        and state.get('run_id')==args.run_id and state.get('worker_arguments'),
        'First TSC window or later SAME-TSC exact window required')
    arm=state
    if require_closed:
        require(args.source_closed_confirmed,'Root must confirm the owned parent and worker are closed')
        require(state.get('status') in ('quality_window_ended','train_window_ended','resource_review_required')
            and arm.get('child_exit_code')==0,
            'Terminal parent and successfully saved closed child required')
        require(not state.get('resource_review_required') or args.resource_review_cleared,
            'Root must explicitly clear a resource review before this exact continuation')
    require(sha(HERE/'run_hf_TSC_arm_v2.py')==TSC_CONTROLLER_SHA,
        'Frozen TSC first-launch controller changed')
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
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HERE/'launch_TSC_same_science_resume_v2.py'),
        '--original-native-provenance',str(original_ready),'--original-native-provenance-sha256',original_sha,
        '--fresh-runtime-provenance',str(paths['fresh_runtime']),'--',*memory]
    namespace=argparse.Namespace(original_native_provenance=original_ready,original_native_provenance_sha256=original_sha,
        fresh_runtime_provenance=paths['fresh_runtime'],memory_arguments=memory,validate_only=True)
    fresh.validate(namespace)
    plan={'schema':1,'kind':'TSC_same_run_exact_science_continuation_v2','passed':True,'run_id':args.run_id,
        'loop_id':folder.name,'source_loop':str(source_loop),'original_run_config':record(config_path),
        'original_ready':record(original_ready),'paths':{k:str(p) for k,p in paths.items()},'command':command,
        'worker_arguments':argv,'source_sha256':SOURCE_PINS,'new_preloader':record(Path(fresh.__file__)),
        'controller_sha256':sha(__file__),'scientific_config_identity_sha256':hashlib.sha256(json.dumps(config['config_identity'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
        'source_config_identity':config['config_identity'],'scientific_arguments_unchanged':True,
        'original_source_closure':record(scoped(fresh.option(argv,'--source-closure'))),
        'combined_TR_clips':config['turkish_clips_used'],'configured_steps':config['steps'],
        'TSC_input_proof':record(scoped(config['tsc_input_proof'])),'TSC_training_approval':record(scoped(config['tsc_training_approval'])),
        'TSC_quarantine_proof':read(scoped(config['tsc_input_proof']),{})['references']['quarantine_proof'],
        'runtime_memory_launcher':record(HERE/fresh.RUNTIME_MEMORY_NAME),
        'max_retention_failures_unchanged':2,'same_run_id':True,'optimizer_sampler_RNG_counters_reset':False,
        'original_ready_is_not_current_PID_evidence':True,'fresh_actual_preload_before_Torch_required':True,
        'archive_path':str(archive),'source_latest_path':str(latest),'source_best_path':str(best),
        'source_closure_confirmed':require_closed,'resource_review_cleared':args.resource_review_cleared,
        'heldout_TEST_evaluated':False,'objective_complete':False,
        'CUDA_initialized_by_preflight':False}
    if not require_closed:return plan
    selection_path=HERE/'runs'/args.run_id/'selection.json';history_path=HERE/'runs'/args.run_id/'training_history.json'
    selection=read(selection_path,{});history=read(history_path,[])
    require(selection.get('run_id')==args.run_id and selection.get('accepted') is True
        and selection.get('completed_steps')==len(history) and 0<=len(history)<config['steps']
        and [row['step'] for row in history]==list(range(1,len(history)+1)),
        'Closed ordered history/qualifying BEST before configured horizon required')
    require(selection.get('heldout_tests_evaluated') is False
        and selection.get('stop_early_reason') in ('soft_max_wall_seconds','controller_stop_file'),
        'Do not silently reset a quality stop or resume a finished scientific horizon')
    require(state.get('completed_steps')==selection['completed_steps']
        and state.get('selected_step')==selection['selected_step']
        and state.get('selection')==record(selection_path),
        'Closed parent durable selection frontier changed')
    if state['kind']=='isolated_HF_TSC_guarded_quality_arm_v2':
        require(state.get('controller_sha256')==TSC_CONTROLLER_SHA
            and state.get('source_sha256')==SOURCE_PINS,'Original TSC parent source pin differs')
        prior_runtime_completion=domain.verify_completion(state,selection)
    else:
        require(state.get('controller_sha256')==sha(__file__)
            and state.get('source_sha256')==SOURCE_PINS,'Previous exact-window source pin differs')
        prior_paths={k:Path(v) for k,v in state['paths'].items()}
        prior_identity=domain.validate_identity(read(prior_paths['identity'],{}),state,
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
    require(value.get('kind')=='TSC_same_science_native_preload_resume_v2' and value.get('status')=='ready'
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
    require(completion.get('kind')=='TSC_same_science_native_preload_resume_completion_v2'
        and completion.get('status')=='complete' and completion.get('run_id')==plan['run_id']
        and completion.get('sources_unchanged') is True and completion.get('pid')==proof['pid']
        and completion.get('launcher_sha256_after')==plan['new_preloader']['sha256']
        and completion.get('fresh_runtime_proof_sha256')==sha(paths['fresh_runtime'])
        and completion.get('original_first_start_ready_sha256_after')==plan['original_ready']['sha256'],
        'Actual fresh runtime completion missing or mismatched')
    run=HERE/'runs'/plan['run_id'];config=read(run/'run_config.json',{})
    validate_config(config,plan['worker_arguments'])
    require(read(scoped(config['tsc_input_proof']),{})['references']['quarantine_proof']==plan['TSC_quarantine_proof']
        and record(scoped(plan['TSC_quarantine_proof']['path']))==plan['TSC_quarantine_proof'],
        'Exact resume changed the originally approved TSC uncertainty quarantine')
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
        and plan['initial_step']<=len(history)<=plan['configured_steps']
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
        and memory.get('kind')=='memory_guard_training_launcher_TSC_resume_v2'
        and memory.get('cap_bytes')==8000000000 and memory.get('cleanup_threshold_bytes')==4000000000
        and memory.get('allocator_cap_applied') is True
        and memory.get('training_math_changed') is False and memory.get('backend_flags_changed') is False
        and memory.get('all_phase_owned_memory_telemetry',{}).get('thread_join_completed') is True,
        'Unchanged cap8/end4/math/backend/all-phase telemetry completion required')
    for key,name in {'memory_launcher_sha256':fresh.RUNTIME_MEMORY_NAME,
        'cache_launcher_sha256':'launch_with_cache_reclaim_tsc_v2.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
        'resource_helper_sha256':'resource_watchdog_v2.py','frozen_worker_sha256':'train_lora_batched_tsc_v2.py'}.items():
        expected=plan['runtime_memory_launcher']['sha256'] if key=='memory_launcher_sha256' else SOURCE_PINS[name]
        require(memory.get(key)==memory.get(key+'_after')==expected,
            'Before/after owned runtime source differs: '+key)
    for key,name in {'resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
        'data_policy_sha256':'public_turkish_data_policy_v3.py'}.items():
        require(memory.get(key)==memory.get(name.removesuffix('.py')+'_sha256_after')==SOURCE_PINS[name],
            'Before/after telemetry or data policy differs: '+key)
    updates=selection['completed_steps'];new_updates=updates-plan['initial_step']
    require(cache.get('kind')=='cache_reclamation_training_launcher_tsc_v2'
        and cache.get('status')=='complete'
        and cache.get('frozen_worker_sha256')==SOURCE_PINS['train_lora_batched_tsc_v2.py']
        and cache['cache_reclamation']['eligible_update_calls']==new_updates
        and cache['cache_reclamation']['failed_reclamations']==0,
        'Scoped cache hook must observe only new updates, not inherited full history')
    require(io.get('kind')=='atomic_io_retry_tsc_v2' and io.get('status')=='complete'
        and io.get('worker_sha256')==io.get('worker_sha256_after')==SOURCE_PINS['train_lora_batched_tsc_v2.py'],
        'Actual I/O completion worker source differs')
    coverage=domain.verify_pool_frontier(config,selection,history,plan)
    expected=selection['turkish_epochs_completed']
    domain.chain_identity()
    require(sha(Path(fresh.__file__))==plan['new_preloader']['sha256'],'Fresh preloader changed during this window')
    return {'new_optimizer_end_calls':new_updates,'inherited_steps':plan['initial_step'],
        'cumulative_completed_steps':updates,'uniform_combined_pool_epochs_completed':expected,
        'inherited_history_prefix_unchanged':True,'scientific_config_identity_unchanged':True,
        'actual_pool_frontier':coverage,'runtime_completion':record(completion_path),'config':record(run/'run_config.json'),
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
                identity=domain.validate_identity(read(paths['identity'],{}),plan,child.pid,state['started_at_utc']) if paths['identity'].exists() else None
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
        identity=domain.validate_identity(read(paths['identity'],{}),plan,child.pid,state['started_at_utc'])
        state['actual_runtime_completion']=verify_completion(plan,selection,identity)
        state.update(status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'train_window_ended',
            child_exit_code=child.returncode,completed_steps=selection['completed_steps'],selected_step=selection['selected_step'],
            selection=record(HERE/'runs'/args.run_id/'selection.json'),finished_at_utc=now(),
            stop_early_reason=selection['stop_early_reason'],combined_pool_coverage=state['actual_runtime_completion']['actual_pool_frontier'],
            uniform_combined_pool_epochs_completed=selection['turkish_epochs_completed'],
            resource_review_required=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary'),objective_complete=False,
            next_action='ROOT reviews DEV/retention and the active quality goal; exact state is saved, no TEST or export performed')
        state['stop_category']=('resource' if state['resource_review_required'] else
            'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures' else
            'plateau' if selection['stop_early_reason']=='development_plateau' else
            'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds' else
            'configured_TSC_horizon' if selection['stop_early_reason']=='configured_steps_complete' else selection['stop_early_reason'])
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
