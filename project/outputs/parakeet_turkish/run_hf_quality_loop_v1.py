"""Bounded DEV-only HF warm-start quality experiments using frozen B8x2 runtime.

Arm1 changes only replay token KL5->10. If it stops for consecutive retention
failures, arm2 retains KD10 and lowers heads5e-5->2e-5 / LoRA3e-4->1e-4.
Each arm explicitly starts fresh Adam/sampler/RNG/counters from the best
qualifying weights. No TEST inference, export, or quality-goal completion.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read,write_json as write
import resume_hf_training_862_v1 as runtime
from resource_watchdog_v2 import ResourceGuard,sample_memory

HERE,ROOT=runtime.HERE,runtime.ROOT
SOURCE='adaptive_20261011_hf_return_01'
SOURCE_LOOP=HERE/'runs/adaptive_20261011_hf_return_window_01'
SOURCE_PINS=runtime.SOURCE_PINS


def require(ok,message):
    if not ok:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def record(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def prepare(args):
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.loop_id),'Safe NEW loop ID required')
    require(0<args.window_hours<=3 and 5<=args.poll_seconds<=15,'Bounded3h / fresh5–15s polling required')
    loop=HERE/'runs'/args.loop_id
    require(not (loop/'loop_status.json').exists(),'Never overwrite or restart a controller state')
    state=read(SOURCE_LOOP/'loop_status.json',{})
    selection=read(HERE/'runs'/SOURCE/'selection.json',{})
    require(state.get('child_exit_code')==0 and state.get('run_id')==SOURCE
        and state.get('status')=='train_window_ended' and selection.get('completed_steps')==1000
        and selection.get('stop_early_reason')=='consecutive_retention_gate_failures'
        and selection.get('accepted') is True and selection.get('selected_step')==0,
        'Actual closed1000 source and qualifying best0 required')
    source_best=Path(selection['weights']).resolve()
    require(source_best==ROOT/'work'/(SOURCE+'_best.pt'),'Preserved qualifying BEST source required')
    for name,digest in SOURCE_PINS.items():require(sha(HERE/name)==digest,'Frozen runtime source changed: '+name)
    require(sha(HERE/'resume_hf_training_862_v1.py')=='d5ff785277e79d5fbea1ccf1c0ffdc152f4344df629cff2fe057ac239a71ff7f',
        'Reviewed exact-return controller changed')
    loop.mkdir(parents=True,exist_ok=True)
    initial=loop/'immutable_source_best0.pt'
    if initial.exists():require(sha(initial)==sha(source_best),'Prepared source BEST archive changed')
    else:
        shutil.copyfile(source_best,initial)
        require(sha(initial)==sha(source_best),'Source BEST copy changed')
    original=list(state['worker_arguments'])
    for suffix in ('stronger_kd','conservative_lr'):
        run_id=args.loop_id+'_'+suffix
        require(not (HERE/'runs'/run_id).exists(),'Fresh scientific run namespace required')
        require(not any((ROOT/'work'/(run_id+'_'+role+'.pt')).exists() for role in ('latest','best')),
            'Fresh optimizer/checkpoint namespace required')
    return {'passed':True,'kind':'HF_DEV_only_warm_start_quality_loop_v1','loop_id':args.loop_id,
        'source_run_id':SOURCE,'source_selected_step':0,'source_completed_steps':1000,
        'source_latest_nonqualifying':True,'source_best':record(source_best),'immutable_initial_best':record(initial),
        'source_selection':record(HERE/'runs'/SOURCE/'selection.json'),'source_status':record(SOURCE_LOOP/'loop_status.json'),
        'original_worker_arguments':original,'runtime_source_sha256':SOURCE_PINS,'controller_sha256':sha(__file__),
        'initial_recipe':'stronger_kd','recipes':[{'name':'stronger_kd','replay_kl':10.,'head_lr':5e-5,'lora_lr':3e-4},
            {'name':'conservative_lr','replay_kl':10.,'head_lr':2e-5,'lora_lr':1e-4}],
        'unchanged_recipe':{'batch_size':8,'accumulation':2,'replay_fraction':.5,'row_delta_multiplier':6.,
            'duration_kl':.5,'feature_distillation':1.,'eval_every_updates':250,'cap_gb':8,'end_cache_gb':4},
        'transition_policy':'Only after consecutive_retention_gate_failures with >=10min remaining; warmstart globally best qualifying DEV weights',
        'fresh_Adam_sampler_RNG_history_selection_counters_each_arm':True,
        'heldout_TEST_evaluated':False,'objective_complete':False,'CUDA_initialized_by_preflight':False}


def arm_plan(plan,recipe,best,remaining):
    loop=HERE/'runs'/plan['loop_id'];run_id=plan['loop_id']+'_'+recipe['name']
    paths={key:loop/(run_id+suffix) for key,suffix in {'native':'_native_backend.json','io':'_io.json',
        'cache':'_cache.json','memory':'_memory.json','identity':'_identity.json','stop':'_stop.txt','log':'.log'}.items()}
    require(all(not path.exists() for path in paths.values()),'New immutable arm evidence required')
    argv=list(plan['original_worker_arguments'])
    changed=(('--run-id',run_id),('--resume-exact',None),('--resume-runtime-migration',None),
        ('--resume-weights',best['path']),('--native-backend-provenance',paths['native']),('--stop-file',paths['stop']),
        ('--max-wall-seconds',max(1,remaining-300)),('--replay-kl',recipe['replay_kl']),
        ('--head-lr',recipe['head_lr']),('--lora-lr',recipe['lora_lr']))
    for flag,value in changed:runtime.replace_option(argv,flag,value)
    allowed={flag for flag,_ in changed}
    def unchanged(values):
        values=list(values)
        for flag in allowed:runtime.replace_option(values,flag,None)
        return values
    require(unchanged(argv)==unchanged(plan['original_worker_arguments']),'Undeclared scientific argument change')
    for flag,value in (('--batch-size','8'),('--accumulation','2'),('--replay-fraction','0.5'),
        ('--new-token-lr-multiplier','6.0'),('--duration-kl','0.5'),('--feature-distillation','1.0'),('--eval-every-updates','250')):
        require(float(runtime.option(argv,flag))==float(value),'Preserve declared recipe: '+flag)
    memory=['--provenance',str(paths['io']),'--cache-provenance',str(paths['cache']),'--memory-provenance',str(paths['memory']),
        '--worker-identity',str(paths['identity']),'--cap-gb','8','--threshold-gb','4','--',*argv]
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HERE/'launch_native_training_v5.py'),
        '--native-provenance',str(paths['native']),'--memory-launcher',str(HERE/'launch_with_memory_guard_v15.py'),
        '--memory-launcher-sha256',SOURCE_PINS['launch_with_memory_guard_v15.py'],
        '--trainer-sha256',SOURCE_PINS['train_lora_batched_data_v11.py'],'--',*memory]
    return {'run_id':run_id,'recipe':recipe,'warm_start':best,'worker_arguments':argv,'command':command,
        'paths':{k:str(p) for k,p in paths.items()},'mode':'new_optimizer_warm_start',
        'fresh_optimizer_sampler_RNG_history_counters':True,'exact_continuation_claimed':False}


def validate_identity(value,arm,child_pid,started):
    require(value.get('kind')=='owned_training_worker_identity_v1' and value.get('expected_run_id')==arm['run_id']
        and value.get('worker_arguments_sha256')==runtime.previous.argument_sha(arm['worker_arguments']),
        'Actual owned worker run/arguments differ')
    require(type(value.get('pid')) is int and value['pid']>0 and (value['pid']==child_pid or value.get('ppid')==child_pid),
        'Actual interpreter is not owned child/self')
    for key,name in (('memory_launcher_sha256','launch_with_memory_guard_v15.py'),('cache_launcher_sha256','launch_with_cache_reclaim_v12.py'),
        ('cache_helper_sha256','cuda_cache_reclaim_v2.py'),('resource_helper_sha256','resource_watchdog_v2.py'),
        ('resource_telemetry_sha256','paired_windows_commit_telemetry_v1.py'),('frozen_worker_sha256','train_lora_batched_data_v11.py'),
        ('data_policy_sha256','public_turkish_data_policy_v3.py')):
        require(value.get(key)==SOURCE_PINS[name],'Actual owned source differs: '+key)
    creation=value['process_creation_identity']
    require(creation['available'] and creation['pid']==value['pid']
        and runtime.previous.stamp(started).timestamp()-5<=creation['creation_unix_seconds']
        <=runtime.previous.stamp(value['started_at_utc']).timestamp()+1,'Owned worker creation identity differs')
    return value


def run_arm(state,arm,deadline,args):
    loop=HERE/'runs'/state['loop_id'];paths={k:Path(v) for k,v in arm['paths'].items()}
    child=None;stop_at=None;stop_reason=None;started=time.monotonic();started_utc=now()
    resource=ResourceGuard();progress=runtime.previous.ProgressGuard(started)
    arm.update(status='starting',started_at_utc=started_utc)
    print(f"Starting {arm['run_id']}: B8x2, fresh Adam/sampler, tokenKD{arm['recipe']['replay_kl']}, heads{arm['recipe']['head_lr']}, LoRA{arm['recipe']['lora_lr']}",flush=True)
    last_phase=None;last_dev=-1
    try:
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(arm['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            arm['owned_launcher_pid']=child.pid
            while child.poll() is None:
                history=read(HERE/'runs'/arm['run_id']/'training_history.json',[])
                identity=validate_identity(read(paths['identity'],{}),arm,child.pid,started_utc) if paths['identity'].exists() else None
                phase=runtime.previous.log_phase(paths['log'],history)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=runtime.previous.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
                progress_reason,progress_info=progress.observe(history,time.monotonic())
                reason=reason or self_reason or progress_reason
                if not identity and time.monotonic()-started>120:reason=reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic()-started>120:reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
                if not history and time.monotonic()-started>1800:reason=reason or 'bounded_preparation_timeout'
                if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'
                arm.update(status='running',phase=phase,published_step=history[-1]['step'] if history else 0,
                    system_and_owned_private_memory=observation,actual_worker_self_memory=self_observation,
                    progress=progress_info,updated_at_utc=now())
                if phase['phase']!=last_phase:
                    print(f"{arm['run_id']} phase: {phase['phase']}",flush=True);last_phase=phase['phase']
                for row in history:
                    if row['step']>last_dev and 'validation' in row:
                        gate=row['validation'];print(f"{arm['run_id']} DEV{row['step']}: TR WER{100*gate['turkish_wer']:.4f}%, all5 retention={gate['retention_gate_passed']}",flush=True);last_dev=row['step']
                with (loop/(arm['run_id']+'_memory_observations.jsonl')).open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info})+'\n')
                if reason and stop_at is None:
                    stop_at=time.monotonic();stop_reason=reason
                    paths['stop'].write_text('Cooperative save/DEV requested: '+reason+'\n',encoding='utf-8')
                    arm.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
                write(loop/'loop_status.json',state)
                if stop_at and time.monotonic()-stop_at>300:raise RuntimeError('Owned child exceeded cooperative grace; no next GPU arm')
                time.sleep(args.poll_seconds)
        arm['child_exit_code']=child.returncode
        require(child.returncode==0,'Training child failed; preserve artifacts and review before another GPU arm')
        selection_path=HERE/'runs'/arm['run_id']/'selection.json';selection=read(selection_path,{})
        require(selection.get('run_id')==arm['run_id'],'Durable final DEV selection missing')
        for key in ('io','cache','memory'):require(read(paths[key],{}).get('status')=='complete','Runtime completion missing: '+key)
        completion=read(paths['native'].with_suffix('.completion.json'),{})
        require(completion.get('status')=='complete' and completion.get('sources_unchanged') is True,'Native ready/completion source proof missing')
        for name,digest in SOURCE_PINS.items():require(sha(HERE/name)==digest,'Runtime source changed: '+name)
        require(sha(arm['warm_start']['path'])==arm['warm_start']['sha256'],'Immutable warm-start source changed')
        arm.update(status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'complete',
            completed_steps=selection['completed_steps'],selected_step=selection['selected_step'],selection=record(selection_path),
            stop_early_reason=selection['stop_early_reason'],resource_review_required=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary'),
            finished_at_utc=now())
        print(f"Completed {arm['run_id']}: {selection['completed_steps']} updates, selected{selection['selected_step']}, stop={selection['stop_early_reason']}",flush=True)
        return selection
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists():paths['stop'].write_text('Controller exit: cooperative save/DEV requested\n',encoding='utf-8')
            try:child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)


def main(args):
    plan=prepare(args)
    initial={**plan['immutable_initial_best'],'source_run_id':SOURCE,'selected_step':0,
        'turkish_wer':read(HERE/'runs'/SOURCE/'selection.json',{})['turkish_validation_wer'],'retention_gate_passed':True}
    if args.validate_only:
        arm=arm_plan(plan,plan['recipes'][0],initial,args.window_hours*3600)
        result={**plan,'first_arm':arm}
        if args.preflight_output:write(args.preflight_output,result)
        print(json.dumps({'passed':True,'loop_id':args.loop_id,'initial_recipe':'stronger_kd','CUDA_initialized':False}))
        return 0
    require(args.gpu_window_authorized,'Only root may authorize exclusive GPU ownership')
    loop=HERE/'runs'/args.loop_id
    started=time.monotonic();deadline=started+args.window_hours*3600
    state={**plan,'status':'starting','started_at_utc':now(),'window_hours':args.window_hours,
        'absolute_deadline_utc':datetime.fromtimestamp(time.time()+args.window_hours*3600,timezone.utc).isoformat(),
        'trials':[],'best_development_candidate':initial,'objective_status':'incomplete'}
    state['awake_request']=runtime.previous.prior.keep_awake(True)
    try:
        for index,recipe in enumerate(plan['recipes']):
            remaining=deadline-time.monotonic()
            if remaining<600:
                state['next_arm_skipped']='Less than10min remaining including startup/save reserve';break
            arm=arm_plan(plan,recipe,state['best_development_candidate'],remaining)
            state['trials'].append(arm);state.update(status='running',active_run_id=arm['run_id']);write(loop/'loop_status.json',state)
            selection=run_arm(state,arm,deadline,args)
            if arm['resource_review_required']:
                state['status']='resource_review_required';return 1
            if selection.get('accepted') is True and selection['turkish_validation_wer']<state['best_development_candidate']['turkish_wer']:
                selected_path=Path(selection['weights']);archive=loop/(arm['run_id']+'_selected_step_'+str(selection['selected_step'])+'.pt')
                require(not archive.exists(),'New immutable selected weights required')
                shutil.copyfile(selected_path,archive);require(sha(archive)==sha(selected_path),'Selected weights archive mismatch')
                state['best_development_candidate']={**record(archive),'source_run_id':arm['run_id'],
                    'selected_step':selection['selected_step'],'turkish_wer':selection['turkish_validation_wer'],
                    'retention_gate_passed':True,'selection':record(HERE/'runs'/arm['run_id']/'selection.json')}
            if selection['stop_early_reason']!='consecutive_retention_gate_failures':
                state['next_arm_not_needed']='No consecutive retention failure; preserve completed best and evaluate next action';break
        state.update(status='quality_window_ended',finished_at_utc=now(),objective_complete=False,
            next_action='Review DEV-only accuracy and retention; continue authorized quality work until measured target is reached')
        return 0
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at_utc=now(),objective_complete=False)
        raise
    finally:
        state['awake_clear']=runtime.previous.prior.keep_awake(False)
        write(loop/'loop_status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loop-id',default='adaptive_20261011_hf_quality_window_01')
    p.add_argument('--window-hours',type=float,default=3.)
    p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--preflight-output',type=Path)
    p.add_argument('--gpu-window-authorized',action='store_true')
    sys.exit(main(p.parse_args()))
