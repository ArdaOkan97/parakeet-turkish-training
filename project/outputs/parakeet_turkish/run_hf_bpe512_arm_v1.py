"""Root-only bounded BPE512 quality arm from a CLOSED qualifying HF source.

Copies the complete source recipe, starts fresh Adam/sampler/counters, and
retains the existing Windows resource/progress guards. The isolated trainer
requires actual unsuppressed six-language GPU source equivalence before Adam.
No TEST, export, source continuation, or quality-goal completion occurs here.
--validate-only uses the standard library and never launches a process.
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
from hf_bpe512_training_v1 import closed_source_identity, candidate_assets, BASELINE_SEMANTIC_SHA
import run_experiment_loop_continuation_v8 as guards
from resource_watchdog_v2 import ResourceGuard, sample_memory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHAIN_PATH = HERE / 'HF_BPE512_runtime_chain_pins_v2.json'
CHAIN_SHA = 'f8bcb1ecc09ec30cab11a260dc8cb5f839994f26fb464b29040a41522055eb45'
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
    require(value.get('kind')=='isolated_HF_BPE512_runtime_chain_pins_v2'
        and value.get('passed_CPU_routing_checks') is True,'Executed CPU routing proof required')
    required={'launch_with_memory_guard_bpe512_v1.py','launch_with_cache_reclaim_bpe512_v1.py',
        'cuda_cache_reclaim_v2.py','train_lora_batched_bpe512_v1.py',*FIXED_GUARD_SOURCES}
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
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.loop_id),'Safe NEW loop ID required')
    require(.25<=args.window_hours<=6 and 5<=args.poll_seconds<=15,'Explicit bounded0.25–6h window /5–15s polls required')
    chain=chain_identity()
    run_id=args.loop_id+'_bpe512';loop=HERE/'runs'/args.loop_id
    require(not (loop/'loop_status.json').exists() and not (HERE/'runs'/run_id).exists(),'Fresh controller/run namespace required')
    require(all(not (ROOT/'work'/(run_id+'_'+role+'.pt')).exists() for role in ('best','latest')),'Fresh Adam/checkpoint namespace required')
    context=SimpleNamespace(run_id=run_id,bpe_source_checkpoint=scoped(args.source_checkpoint),
        bpe_source_selection=scoped(args.source_selection),bpe_source_closure=scoped(args.source_closure),
        bpe_conversion_report=scoped(args.conversion_report),resume_weights=scoped(args.converted_weights))
    identity=closed_source_identity(context)
    source_config=scoped(args.source_run_config);closure=identity['closed_source']
    require(closure.get('source_run_config')==record(source_config),'ROOT closure must also bind exact source run configuration')
    config=read(source_config,{})
    require(config.get('run_id')==identity['selection']['run_id']
        and config.get('base_revision')=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
        and config['config_identity']['extended_tokenizer_sha256']==BASELINE_SEMANTIC_SHA
        and config.get('turkish_clips_used')==74883 and config.get('replay_clips_used')==1595,
        'Closed original8199 whole-pool HF source configuration required')
    native=config['config_identity']['native_loss']
    require(native.get('sigma')==0.0 and native.get('omega')==0.0
        and native.get('caller_reduction')=='raw per-clip NLL / each target length, then physical-batch mean',
        'Preserve original native kernel/objective reduction policy')
    recipe=config['config_identity']['training_arguments'];argv=recipe_arguments(recipe)
    for name,digest in config['code_sha256'].items(): require(sha(HERE/name)==digest,'Closed source implementation changed: '+name)
    paths={key:loop/(run_id+suffix) for key,suffix in {'native':'_native_backend.json','io':'_io.json',
        'cache':'_cache.json','memory':'_memory.json','identity':'_identity.json','stop':'_stop.txt','log':'.log'}.items()}
    require(all(not path.exists() for path in paths.values()),'All owned BPE evidence paths must be new')
    for key in ('tr_manifest','replay_manifest','data_v3_proof','reuse_input_audit'):
        if config.get(key):
            path=scoped(config[key]);expected=config['input_sha256'].get(str(path))
            require(expected is not None and sha(path)==expected,'Original data/replay/audit input changed: '+key)
            argv.extend(('--'+key.replace('_','-'),str(path)))
    candidate=scoped(args.candidate_dir);candidate_assets(candidate)
    argv.extend(('--run-id',run_id,'--resume-weights',str(context.resume_weights),
        '--bpe-source-checkpoint',str(context.bpe_source_checkpoint),'--bpe-source-selection',str(context.bpe_source_selection),
        '--bpe-source-closure',str(context.bpe_source_closure),'--bpe-conversion-report',str(context.bpe_conversion_report),
        '--bpe-candidate-dir',str(candidate),'--native-backend-provenance',str(paths['native']),
        '--native-max-joint-bytes','1800000000','--stop-file',str(paths['stop']),
        '--max-wall-seconds',str(max(1,args.window_hours*3600-300))))
    pins=chain['source_sha256']
    memory=['--provenance',str(paths['io']),'--cache-provenance',str(paths['cache']),
        '--memory-provenance',str(paths['memory']),'--worker-identity',str(paths['identity']),
        '--cap-gb','8','--threshold-gb','4','--',*argv]
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HERE/chain['outer_launcher']),
        '--native-provenance',str(paths['native']),'--memory-launcher',str(HERE/chain['memory_launcher']),
        '--memory-launcher-sha256',pins[chain['memory_launcher']],
        '--trainer-sha256',pins['train_lora_batched_bpe512_v1.py'],'--',*memory]
    return {'kind':'isolated_HF_BPE512_guarded_quality_arm_v1','loop_id':args.loop_id,'run_id':run_id,
        'source_identity':identity,'source_run_config':record(source_config),'copied_source_recipe':recipe,
        'routing_receipt':record(CHAIN_PATH),'source_sha256':pins,'guard_library_sha256':GUARD_SHA,
        'worker_arguments':argv,'command':command,'paths':{k:str(p) for k,p in paths.items()},
        'controller_sha256':sha(__file__),'window_hours':args.window_hours,
        'declared_changes':['Append512 TRAIN-derived Turkish BPE pieces and donor rows; V8199 to8711',
            'Use candidate Turkish target metadata; original replay1595 labels/cache and teacher8193 unchanged',
            'Move five duration output rows; learned sixchars6x and newBPE1x',
            'Fresh Adam/sampler/RNG/history/counters from CLOSED qualifying learned weights'],
        'same_loss_normalization_policy_different_target_U_and_student_denominator':True,
        'mathematical_trajectory_equivalence_claimed':False,'heldout_TEST_evaluated':False,
        'trackio':{'project':'parakeet-turkish','publication':'Existing separate Trackio v3 mirror observes ordinary top-level config/history after pre-Adam equivalence passes; no Trackio import in worker/controller'},
        'objective_complete':False,'objective_status':'incomplete'}


def validate_identity(value,plan,child_pid,started):
    require(value.get('kind')=='owned_training_worker_identity_v1' and value.get('expected_run_id')==plan['run_id']
        and value.get('worker_arguments_sha256')==guards.argument_sha(plan['worker_arguments']),'Owned candidate argument identity differs')
    require(type(value.get('pid')) is int and value['pid']>0
        and (value['pid']==child_pid or value.get('ppid')==child_pid),'Actual candidate is not owned child/self')
    names={'memory_launcher_sha256':'launch_with_memory_guard_bpe512_v1.py',
        'cache_launcher_sha256':'launch_with_cache_reclaim_bpe512_v1.py','cache_helper_sha256':'cuda_cache_reclaim_v2.py',
        'resource_helper_sha256':'resource_watchdog_v2.py','resource_telemetry_sha256':'paired_windows_commit_telemetry_v1.py',
        'frozen_worker_sha256':'train_lora_batched_bpe512_v1.py','data_policy_sha256':'public_turkish_data_policy_v3.py'}
    for key,name in names.items(): require(value.get(key)==plan['source_sha256'][name],'Actual candidate source differs: '+key)
    creation=value['process_creation_identity']
    require(creation.get('available') is True and creation.get('pid')==value['pid']
        and guards.stamp(started).timestamp()-5<=creation['creation_unix_seconds']
        <=guards.stamp(value['started_at_utc']).timestamp()+1,'Owned worker creation identity differs')
    return value


def verify_completion(plan,selection):
    require(selection.get('run_id')==plan['run_id'] and selection.get('heldout_tests_evaluated') is False,'Durable DEV-only candidate selection required')
    paths={key:Path(value) for key,value in plan['paths'].items()}
    for key in ('io','cache','memory'): require(read(paths[key],{}).get('status')=='complete','Actual runtime completion missing: '+key)
    completion=read(paths['native'].with_suffix('.completion.json'),{})
    require(completion.get('status')=='complete' and completion.get('sources_unchanged') is True,'Native startup/completion evidence missing')
    equivalence=read(HERE/'runs'/plan['run_id']/'BPE512_actual_GPU_source_equivalence.json',{})
    require(equivalence.get('kind')=='HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1'
        and equivalence.get('passed') is True and equivalence.get('actual_GPU_decode') is True
        and equivalence.get('before_optimizer_creation') is True and equivalence.get('decoded_clips')==300
        and not equivalence.get('raw_transcript_changes') and not equivalence.get('token_duration_emission_changes'),
        'Complete actual unsuppressed300-clip source equivalence required')
    require(read(paths['cache'],{})['cache_reclamation']['eligible_update_calls']==selection['completed_steps'],
        'Scoped cache hook did not observe every fresh update')
    chain_identity()
    return record(HERE/'runs'/plan['run_id']/'BPE512_actual_GPU_source_equivalence.json')


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
            publish_note(state,f"ACTUALLY launched isolated BPE arm {state['run_id']} with owned launcher PID{child.pid}; exact CLOSED source recipe, fresh Adam, guarded cap8/end4. Actual pre-Adam source equivalence pending; no BPE quality claim.")
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
        state['actual_source_equivalence']=verify_completion(state,selection)
        state.update(status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'quality_window_ended',
            selected_step=selection['selected_step'],completed_steps=selection['completed_steps'],selection=record(selection_path),
            stop_early_reason=selection['stop_early_reason'],resource_review_required=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary'),finished_at_utc=now())
        baseline=state['source_identity']['selection']['turkish_validation_wer']
        state['improved_qualifying_DEV_over_closed_source']=bool(selection.get('accepted') and selection['turkish_validation_wer']<baseline)
        publish_note(state,f"BPE arm {state['run_id']} ended after{selection['completed_steps']} fresh updates; selected{selection['selected_step']}, qualifying DEV improvement over CLOSED source={state['improved_qualifying_DEV_over_closed_source']}. Actual300-clip pre-Adam equivalence passed; no TEST/export or goal completion.")
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
            output=scoped(args.preflight_output);require(not output.exists(),'Preflight evidence must be new');write(output,{**plan,'passed':True,'Torch_imported':'torch' in sys.modules,'GPU_launched':False})
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
    for name in ('source-checkpoint','source-selection','source-closure','source-run-config','conversion-report','converted-weights'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--candidate-dir',type=Path,default=ROOT/'work/hf_turkish_bpe_candidate_512_v1')
    result.add_argument('--window-hours',type=float,default=3.)
    result.add_argument('--poll-seconds',type=float,default=5.)
    result.add_argument('--validate-only',action='store_true')
    result.add_argument('--preflight-output',type=Path)
    result.add_argument('--gpu-window-authorized',action='store_true')
    return result


if __name__=='__main__': sys.exit(main(parser().parse_args()))
