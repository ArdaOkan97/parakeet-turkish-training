"""Guarded first approved TSC TRAIN data-only arm; no TEST/export or goal completion.

Copy the ROOT-chosen CLOSED qualifying source recipe. Change only the appended
TRAIN pool and one-pass horizon ceil(actual combined unique TR clips /8).
Fresh Adam initially; scheduled windows later preserve SAME-run full state.
The original replay pool is exact, but shared-RNG replay order may differ.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from atomic_io_retry_v1 import read_json as read, write_json as write
from tsc_training_inputs_v1 import closed_source_identity, approve
TOKENIZER_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
RECIPE = {'batch_size': 8, 'accumulation': 2, 'rank': 8, 'alpha': 16.0, 'last_n_layers': None, 'head_lr': 5e-05, 'lora_lr': 0.0003, 'new_token_lr_multiplier': 6.0, 'replay_fraction': 0.5, 'replay_kl': 10.0, 'duration_kl': 0.5, 'feature_distillation': 1.0, 'tdt_implementation': 'nemo', 'warmup_steps': 25, 'schedule': 'constant', 'seed': 20261010, 'bucket_size': 128, 'max_tokens': 180, 'chunk_size': 16, 'no_encoder_checkpoint': False, 'phonetic_sh_donors': False, 'train_limit': None, 'supplement': None, 'max_retention_failures': 2}
import run_experiment_loop_continuation_v8 as guards
from resource_watchdog_v2 import ResourceGuard, sample_memory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHAIN_PATH = HERE / 'TSC_runtime_chain_pins_v1.json'
CHAIN_SHA = '1a8294aa034e37d85f65fb66d4725ffe39432169be1dcaee9a77307f7465c220'
GUARD_SHA = 'c65e9f37eb0db4641552d35ccda52b5604223223deb88d7d9ebfcfe61f0f051d'
FIXED_GUARD_SOURCES = {
    'resource_watchdog_v2.py':'64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641',
    'paired_windows_commit_telemetry_v1.py':'2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683',
    'public_turkish_data_policy_v3.py':'08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41'}
RECIPE_KEYS = ('epochs','steps','batch_size','accumulation','replay_fraction','bucket_size','seed',
    'max_tokens','head_lr','lora_lr','new_token_lr_multiplier','replay_kl','duration_kl',
    'feature_distillation','rank','alpha','last_n_layers','chunk_size','tdt_implementation',
    'warmup_steps','schedule','min_lr_ratio','eval_per_epoch','eval_epoch_fraction',
    'eval_every_updates','min_improvement','no_encoder_checkpoint','phonetic_sh_donors',
    'train_limit','early_stop_patience','max_retention_failures','target_turkish_wer')


def require(ok, message):
    if not ok: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def record(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def scoped(path):
    result=Path(path).resolve()
    require(result.is_relative_to(ROOT),'Workspace-only source/evidence path required')
    return result


def chain_identity():
    require(sha(CHAIN_PATH)==CHAIN_SHA,'Frozen isolated routing receipt changed')
    value=read(CHAIN_PATH,{})
    require(value.get('kind')=='isolated_TSC_runtime_chain_pins_v1'
        and value.get('passed_CPU_routing_checks') is True,'Executed CPU routing proof required')
    required={'launch_with_memory_guard_tsc_v1.py','launch_with_cache_reclaim_tsc_v1.py',
        'cuda_cache_reclaim_v2.py','train_lora_batched_tsc_v1.py',*FIXED_GUARD_SOURCES}
    require(required.issubset(value.get('source_sha256',{})),'Full runtime dependency map required, not the five-source dispatch-proof map')
    for name,digest in FIXED_GUARD_SOURCES.items():
        require(value['source_sha256'][name]==digest and sha(HERE/name)==digest,'Existing guard/data policy changed: '+name)
    for name,digest in value['source_sha256'].items():
        require(Path(name).name==name and sha(HERE/name)==digest,'Isolated runtime source changed: '+name)
    proof=Path(value['CPU_dispatch_proof_path'])
    require(sha(proof)==value['CPU_dispatch_proof_sha256'],'Executed nested-dispatch evidence changed')
    require(sha(HERE/'run_experiment_loop_continuation_v8.py')==GUARD_SHA,'Existing guard library changed')
    return value


def recipe_arguments(recipe):
    require(set(recipe)==set(RECIPE_KEYS),'Copy every original scientific recipe field; unknown/missing keys rejected')
    require(recipe['tdt_implementation']=='nemo' and recipe['max_tokens']==180
        and recipe['new_token_lr_multiplier']==6.0 and recipe['train_limit'] is None
        and recipe['phonetic_sh_donors'] is False,'Whole-pool original8199 native recipe required')
    require(recipe['batch_size']==8 and recipe['accumulation']==2,'This isolated arm retains actual B8x2 geometry')
    result=[]
    for key in RECIPE_KEYS:
        value=recipe[key]
        if value is None: continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            require(key in ('no_encoder_checkpoint','phonetic_sh_donors'),'Unexpected boolean recipe option')
            if value: result.append(flag)
        else:
            require(type(value) in (str,int,float) and (not isinstance(value,float) or math.isfinite(value)),
                'Finite scalar source recipe required: '+key)
            result.extend((flag,str(value)))
    return result


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
    fraction=recipe.get('eval_epoch_fraction')
    require(recipe.get('eval_every_updates')==250 and isinstance(fraction,(int,float))
        and math.isfinite(fraction) and 0<fraction<=1,
        'Preserve actual source DEV250 and exact copied exposure-fraction clock')
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



def validate_identity(value,plan,child_pid,started):
    require(value.get('kind')=='owned_training_worker_identity_v1' and value.get('expected_run_id')==plan['run_id']
        and value.get('worker_arguments_sha256')==guards.argument_sha(plan['worker_arguments']),'Owned candidate argument identity differs')
    require(type(value.get('pid')) is int and value['pid']>0
        and (value['pid']==child_pid or value.get('ppid')==child_pid),'Actual candidate is not owned child/self')
    names={'memory_launcher_sha256':'launch_with_memory_guard_tsc_v1.py',
        'cache_launcher_sha256':'launch_with_cache_reclaim_tsc_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
        'resource_helper_sha256':'resource_watchdog_v2.py','resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
        'frozen_worker_sha256':'train_lora_batched_tsc_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}
    for key,name in names.items():
        expected=plan.get('runtime_memory_launcher',{}).get('sha256',plan['source_sha256'][name]) if key=='memory_launcher_sha256' else plan['source_sha256'][name]
        require(value.get(key)==expected,'Actual candidate source differs: '+key)
    for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):
        index=plan['worker_arguments'].index(flag);item=record(scoped(plan['worker_arguments'][index+1]))
        require(value.get('TSC_explicit_asset_records',{}).get(flag.removeprefix('--').replace('-','_'))==item,'Owned TSC asset identity differs: '+flag)
    creation=value['process_creation_identity']
    require(creation.get('available') is True and creation.get('pid')==value['pid']
        and guards.stamp(started).timestamp()-5<=creation['creation_unix_seconds']
        <=guards.stamp(value['started_at_utc']).timestamp()+1,'Owned worker creation identity differs')
    return value


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



def publish_note(state,text):
    try: guards.prior.backlog_note(now()+': '+text)
    except Exception as error:
        state.setdefault('warnings',[]).append({'kind':'backlog_update_failed','at_utc':now(),'error':repr(error)})


def run_arm(state,deadline,args):
    paths={key:Path(value) for key,value in state['paths'].items()};loop=HERE/'runs'/state['loop_id']
    child=None;stop_at=None;stop_reason=None;started=time.monotonic();started_utc=now()
    resource=ResourceGuard();progress=guards.ProgressGuard(started);last_phase=None;last_dev=-1
    try:
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(state['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',owned_launcher_pid=child.pid,started_at_utc=started_utc)
            publish_note(state,f"ACTUALLY launched isolated TSC data-only arm {state['run_id']} with owned launcher PID{child.pid}; exact CLOSED source recipe plus approved combined data/horizon, fresh Adam, guarded cap8/end4. Original8199 geometry and source closure validated; no TSC quality claim.")
            while child.poll() is None:
                history=read(HERE/'runs'/state['run_id']/'training_history.json',[])
                identity=validate_identity(read(paths['identity'],{}),state,child.pid,started_utc) if paths['identity'].exists() else None
                phase=guards.log_phase(paths['log'],history)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=guards.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
                progress_reason,progress_info=progress.observe(history,time.monotonic());reason=reason or self_reason or progress_reason
                if not identity and time.monotonic()-started>120: reason=reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic()-started>120: reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
                if not history and time.monotonic()-started>1800: reason=reason or 'bounded_preparation_timeout'
                if time.monotonic()>=deadline-300: reason=reason or 'window_budget_checkpoint_boundary'
                state.update(phase=phase,published_step=history[-1]['step'] if history else 0,system_and_owned_private_memory=observation,
                    actual_worker_self_memory=self_observation,progress=progress_info,updated_at_utc=now())
                if phase['phase']!=last_phase:
                    print(state['run_id']+' phase: '+phase['phase'],flush=True);last_phase=phase['phase']
                for row in history:
                    if row['step']>last_dev and 'validation' in row:
                        gate=row['validation'];print(f"{state['run_id']} DEV{row['step']}: TR{100*gate['turkish_wer']:.4f}%, all5 retention={gate['retention_gate_passed']}",flush=True);last_dev=row['step']
                with (loop/'memory_observations.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info})+'\n')
                if reason and stop_at is None:
                    stop_at=time.monotonic();stop_reason=reason
                    with paths['stop'].open('x',encoding='utf-8') as out: out.write('Cooperative save/DEV requested: '+reason+'\n')
                    state.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
                write(loop/'loop_status.json',state)
                if stop_at and time.monotonic()-stop_at>300: raise RuntimeError('Owned child exceeded cooperative grace; preserve artifacts and review')
                time.sleep(args.poll_seconds)
        state['child_exit_code']=child.returncode
        require(child.returncode==0,'Candidate child failed; no automatic next GPU arm')
        selection_path=HERE/'runs'/state['run_id']/'selection.json';selection=read(selection_path,{})
        state['actual_runtime_completion']=verify_completion(state,selection)
        state.update(status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'quality_window_ended',
            selected_step=selection['selected_step'],completed_steps=selection['completed_steps'],selection=record(selection_path),
            stop_early_reason=selection['stop_early_reason'],resource_review_required=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary'),finished_at_utc=now())
        state['combined_pool_coverage']=state['actual_runtime_completion']['actual_pool_frontier']
        state['uniform_combined_pool_epochs_completed']=selection['turkish_epochs_completed']
        state['configured_combined_pool_pass_completed']=selection['completed_steps']==state['configured_steps']
        state['objective_complete']=False;state['objective_status']='incomplete'
        state['stop_category']=('resource' if state['resource_review_required'] else
            'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures' else
            'plateau' if selection['stop_early_reason']=='development_plateau' else
            'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds' else
            'configured_TSC_horizon' if selection['stop_early_reason']=='configured_steps_complete' else selection['stop_early_reason'])
        state['next_action']=('ROOT review resource stop before any further GPU job' if state['resource_review_required'] else
            'Review actual DEV/retention; scheduled budget stops use the reviewed SAME-run exact-state supervisor route, never fresh Adam')
        baseline=state['source_identity']['turkish_wer']
        state['improved_qualifying_DEV_over_closed_source']=bool(selection.get('accepted') and selection['turkish_validation_wer']<baseline)
        publish_note(state,f"TSC data-only arm {state['run_id']} ended after{selection['completed_steps']} fresh updates; selected{selection['selected_step']}, qualifying DEV improvement over CLOSED source={state['improved_qualifying_DEV_over_closed_source']}. Original8199 model; actual uniform combined-pool coverage recorded; no TEST/export or goal completion.")
        return 1 if state['resource_review_required'] else 0
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists(): paths['stop'].write_text('Controller exit: cooperative save/DEV requested\n',encoding='utf-8')
            try: child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)


def main(args):
    plan=prepare(args)
    if args.validate_only:
        if args.preflight_output:
            output=scoped(args.preflight_output)
            require(not output.is_relative_to(HERE/'runs'/args.loop_id),
                'Save preflight outside the fresh loop directory, which must remain absent before launch')
            require(not output.exists(),'Preflight evidence must be new')
            write(output,{**plan,'passed':True,'Torch_imported':'torch' in sys.modules,'GPU_launched':False})
        print(json.dumps({'passed':True,'loop_id':plan['loop_id'],'run_id':plan['run_id'],'GPU_launched':False,'Torch_imported':'torch' in sys.modules}))
        return 0
    require(args.gpu_window_authorized,'Root must confirm exclusive GPU ownership before launch')
    loop=HERE/'runs'/args.loop_id;loop.mkdir(parents=True,exist_ok=True)
    state={**plan,'status':'starting','objective_complete':False,'objective_status':'incomplete',
        'absolute_deadline_utc':datetime.fromtimestamp(time.time()+args.window_hours*3600,timezone.utc).isoformat(),
        'next_action':'Review measured DEV quality/retention and continue the active goal; no TEST or model publication in this arm'}
    started=time.monotonic();state['awake_request']=guards.prior.keep_awake(True)
    try:
        write(loop/'loop_status.json',state);return run_arm(state,started+args.window_hours*3600,args)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at_utc=now());raise
    finally:
        state['awake_clear']=guards.prior.keep_awake(False);write(loop/'loop_status.json',state)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--loop-id',required=True)
    for name in ('source-checkpoint','source-selection','source-closure','source-run-config'):
        result.add_argument('--'+name,type=Path,required=True)
    for name in ('tsc-input-proof','tsc-training-approval'):result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--source-choice-rationale',required=True)
    result.add_argument('--window-hours',type=float,default=3.)
    result.add_argument('--poll-seconds',type=float,default=5.)
    result.add_argument('--validate-only',action='store_true')
    result.add_argument('--preflight-output',type=Path)
    result.add_argument('--gpu-window-authorized',action='store_true')
    return result



if __name__=='__main__': sys.exit(main(parser().parse_args()))
