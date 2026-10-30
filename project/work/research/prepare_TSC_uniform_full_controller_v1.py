"""Source-only one-file full-pass controller using frozen runtime guard APIs."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
base=(HERE/'run_TSC_headlr_fork_v1.py').read_text(encoding='utf-8')
def fn(name):
 n=next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ''.join(base.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
prefix='''"""INACTIVE original8199 full253798 uniform TRAIN with Portuguese retention.

One actual owned child per bounded window. Only observed exit0/complete budget
closures continue SAME full207/Adam/sampler/RNG/history automatically. Resource,
manual, retention, plateau or unknown exits stop for review; no fresh reset.
ROOT exclusive GPU authorization and current memory admission are required.
"""
from __future__ import annotations
import argparse, ast, functools, hashlib, importlib, json, math, os, re, shutil, subprocess, sys, threading, time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from atomic_io_retry_v1 import read_json as read,write_json as write,commit_writer
from cuda_cache_reclaim_v2 import CudaCacheReclaimer as CacheOperations
import launch_with_memory_guard_TSC_resume_v3 as shared_memory
import run_hf_TSC_arm_v3 as domain
from resource_watchdog_v2 import ResourceGuard,sample_memory
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
WORKER='train_lora_batched_tsc_uniform_full_v1.py'
TRANSITION='tsc_uniform_full_transition_v1.py'
CONTROLLER_KIND='guarded_TSC_8199_uniform_full_Portuguese_controller_v1'
WORKER_SHA='PENDING'
TRANSITION_SHA='PENDING'
SAMPLER_SHA='PENDING'
EXPECTED_SYNC_LINE=0
SHARED_MEMORY_SHA='61fec6d3ef911665bdbe876dc333f3c6a640badc012596fe92ded1b80a10c7fa'
PRELOAD_SHA='f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba'
ADAPTER_SHA='24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e'
COMPACTOR_SHA='1b3f9f0e3bdd19c285f2825a53685d7cf9fd964040c9087560f610a5ed1ec96d'
'''
text=prefix+'\n\n'.join(fn(n) for n in ('require','now','sha','record','canonical','scoped','option','set_option','remove_option'))+'\n\n'
text+='''def boundary_context(argv):
    from tsc_uniform_full_transition_v1 import preflight,source_context
    receipt=read(option(argv,'--uniform-full-source-receipt'));origin=receipt['source_lineage_receipt']
    source=source_context(str(scoped(origin['path'])),origin['sha256'])
    values=dict(source['config']);values.update(source['source_recipe'])
    values.update(sampling_mode='uniform',steps=38225,run_id=option(argv,'--run-id'),resume_optimization=None,
        optimization_source_receipt=None,resume_weights=None,resume_runtime_migration=None,
        resume_exact=scoped(option(argv,'--resume-exact')) if '--resume-exact' in argv else None,
        portuguese_seed=int(option(argv,'--portuguese-seed')))
    for flag in ('--resume-uniform-full','--uniform-full-source-receipt','--uniform-full-runtime-proof',
        '--native-backend-provenance','--portuguese-preparation-proof','--portuguese-baseline-proof',
        '--portuguese-replay-manifest','--portuguese-dev-manifest','--stop-file'):
        values[flag[2:].replace('-','_')]=scoped(option(argv,flag))
    values['max_wall_seconds']=float(option(argv,'--max-wall-seconds'))
    return preflight(SimpleNamespace(**values))


def source_pins(context=None):
    require('PENDING' not in (WORKER_SHA,TRANSITION_SHA,SAMPLER_SHA) and EXPECTED_SYNC_LINE>0,'Full worker/helper/sampler pins not frozen')
    pins=dict(domain.chain_identity()['source_sha256'])
    if context:
        for name,value in context['parent_identity']['source_sha256'].items():
            require(name not in pins or pins[name]==value,'Original support science conflict:'+name);pins[name]=value
    pins.update({WORKER:WORKER_SHA,TRANSITION:TRANSITION_SHA,'tsc_uniform_full_sampler_v1.py':SAMPLER_SHA,
        'native_tdt_checkpointed_v4.py':ADAPTER_SHA,'compact_training_row_objects_v1.py':COMPACTOR_SHA,
        'launch_with_memory_guard_TSC_resume_v3.py':SHARED_MEMORY_SHA,Path(__file__).name:sha(__file__),
        'run_hf_TSC_arm_v3.py':sha(domain.__file__)})
    for name,value in pins.items():require(sha(HERE/name)==value,'Frozen full-run dependency differs:'+name)
    require(pins['preload_nemo_linker_v1.py']==PRELOAD_SHA,'Coherent preloader changed');return pins


'''+fn('caller_line')+'\n\n'
text+='''class ForkCaller(CacheOperations):
    def __init__(self,cuda,worker_path):
        self.cuda=cuda;self.worker_path=Path(worker_path).resolve();self.expected_sha=WORKER_SHA
        self.line=caller_line(self.worker_path);self.threshold_bytes=3000000000;self.max_events=128
        self.events=deque(maxlen=128);self.eligible_calls=self.ignored_calls=self.reclamations=self.failed_reclamations=0
        self.cleanup_seconds=0.;self.original=self.wrapped=None


def validate_plan(plan):
    argv=plan['worker_arguments'];context=boundary_context(argv);pins=source_pins(context)
    require(plan['kind']==CONTROLLER_KIND and plan['source_sha256']==pins and plan['controller_sha256']==sha(__file__)
        and option(argv,'--run-id')==plan['run_id'] and domain.guards.argument_sha(argv)==plan['worker_arguments_sha256'],
        'Actual full-run plan/source/argument identity required')
    require(option(argv,'--sampling-mode')=='uniform' and option(argv,'--head-lr')=='1e-05'
        and float(option(argv,'--lora-lr'))==.0003 and option(argv,'--batch-size')=='8' and option(argv,'--accumulation')=='2'
        and option(argv,'--steps')=='38225' and option(argv,'--eval-every-updates')=='250'
        and '--resume-weights' not in argv and '--resume-optimization' not in argv and '--resume-runtime-migration' not in argv,
        'Full207 originalrecipe, physicalB8x2 and complete newpass required')
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation'
    require(('--resume-exact' in argv)==exact and plan['source_closure']==context['receipt_record']
        and plan['initial_source_checkpoint']==context['source_checkpoint'],'Declared source6500 or SAME-own exact only')
    state_path=scoped(option(argv,'--resume-exact' if exact else '--resume-uniform-full'))
    require(record(state_path)==plan['source_checkpoint'],'Actual complete207/Adam frontier changed')
    paths={key:scoped(value) for key,value in plan['paths'].items()}
    runtime=paths['fresh_runtime'] if exact else paths['native']
    require(option(argv,'--uniform-full-runtime-proof')==str(runtime)
        and option(argv,'--native-backend-provenance')==str(paths['native'])
        and option(argv,'--stop-file')==str(paths['stop']) and runtime!=paths['identity'],
        'Fresh process evidence separate from immutable original ready')
    if exact:require(record(paths['native'])==plan['original_ready'],'Original first native token changed')
    caller_line(HERE/WORKER);return paths


'''
worker=fn('run_worker').replace("'--resume-optimization'","'--resume-uniform-full'")
worker=worker.replace("'--optimization-source-receipt'","'--uniform-full-source-receipt'")
worker=worker.replace("'native_tdt_checkpointed_v3.py'","'native_tdt_checkpointed_v4.py'")
worker=worker.replace("sys.modules.get('native_tdt_checkpointed_v3')","sys.modules.get('native_tdt_checkpointed_v4')")
worker=worker.replace("'end_cache_gb':4","'end_cache_gb':3").replace("'cleanup_threshold_bytes':4_000_000_000","'cleanup_threshold_bytes':3_000_000_000")
worker=worker.replace("'head LR5e-5 ->1e-5 at next update; state/frontier preserved'","'Original207 BEST6500 retained; fulluniformTR+Portuguese replay/retention explicitly added; native4 memory-only cleanup'")
worker=worker.replace("'TSC_headlr_same_new_run_fresh_preload_v1'","'TSC_uniform_full_fresh_preload_v1'")
worker=worker.replace("'TSC_headlr_fork_","'TSC_uniform_full_")
worker=worker.replace("'native_full_model_or_Lightning_trainer_equivalence_claimed':False}","'native_full_model_or_Lightning_trainer_equivalence_claimed':False,'allocator_cap_applied':False}")
worker=worker.replace("memory['allocator_cap_applied']=True;write(paths['memory'],memory)","memory['allocator_cap_applied']=True;write(paths['memory'],memory)\n        ready['allocator_cap_applied']=True;write(runtime_path,ready)")
text+=worker+'\n\n'+fn('validate_identity')+'\n\n'
text+='''def prepare(args):
    from tsc_uniform_full_transition_v1 import source_context
    require(math.isfinite(args.window_hours) and .1<=args.window_hours<=3. and 5<=args.poll_seconds<=15,'Bounded windows/freshpolls required')
    output=scoped(args.output_dir);receipt_path=scoped(args.source_receipt);receipt=read(receipt_path)
    require(output.parent==HERE/'runs' and not output.exists(),'Fresh first controller directory required')
    origin=receipt['source_lineage_receipt'];context=source_context(str(scoped(origin['path'])),origin['sha256'])
    source_status=read(context['lineage_receipt']['artifacts']['controller_status']['path'])
    argv=list(source_status['worker_arguments'])
    for flag in ('--resume-optimization','--optimization-source-receipt','--resume-exact','--resume-weights','--resume-runtime-migration'):
        argv=remove_option(argv,flag)
    paths={key:output/(args.run_id+'_'+name) for key,name in (('native','native_backend.json'),('identity','identity.json'),
        ('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    from tsc_uniform_full_transition_v1 import SOURCE_STEP
    for flag,value in (('--run-id',args.run_id),('--resume-uniform-full',context['source_checkpoint']['path']),
        ('--uniform-full-source-receipt',receipt_path),('--uniform-full-runtime-proof',paths['native']),
        ('--portuguese-preparation-proof',receipt['portuguese_preparation_proof']['path']),
        ('--portuguese-baseline-proof',receipt['portuguese_baseline_proof']['path']),
        ('--portuguese-replay-manifest',args.portuguese_replay_manifest),('--portuguese-dev-manifest',args.portuguese_dev_manifest),
        ('--portuguese-seed',args.portuguese_seed),('--sampling-mode','uniform'),('--steps',38225),
        ('--native-backend-provenance',paths['native']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    context=boundary_context(argv);pins=source_pins(context)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.run_id) and not (HERE/'runs'/args.run_id).exists(),
        'Fresh fullrun scientific namespace required')
    for suffix in ('best.pt','latest.pt'):require(not (ROOT/'work'/f'{args.run_id}_{suffix}').exists(),'Fullrun checkpoint namespace collision')
    python=ROOT/'work/nemo-native-env-v2/Scripts/python.exe'
    plan={'schema':1,'kind':CONTROLLER_KIND,'run_id':args.run_id,'controller_sha256':sha(__file__),
        'source_sha256':pins,'source_closure':context['receipt_record'],'source_checkpoint':record(context['source_checkpoint']['path']),
        'initial_source_checkpoint':context['source_checkpoint'],'source_selected_step':SOURCE_STEP,'resume_step':SOURCE_STEP,
        'source_history_prefix_sha256':context['source_history_prefix_sha256'],'window_hours':args.window_hours,
        'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':[str(python),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'],
        'paths':{key:str(value) for key,value in paths.items()},'configured_steps':38225,'full_new_uniform_updates':31725,
        'Portuguese_replay_and_retention_added':True,'competitive_champion_parked':context['source_checkpoint'],
        'objective_complete':False}
    validate_plan(plan);return plan


def verify_completion(plan,selection):
    paths={key:Path(value) for key,value in plan['paths'].items()};run=HERE/'runs'/plan['run_id']
    exact=plan.get('continuation_mode')=='same_new_run_exact_budget_continuation';runtime=paths['fresh_runtime'] if exact else paths['native']
    identity=read(paths['identity']);done=read(runtime.with_suffix('.completion.json'))
    require(done['kind']=='TSC_uniform_full_native_completion_v1' and done['status']=='complete'
        and done['sources_unchanged'] is True and done['pid']==identity['pid'] and done['run_id']==plan['run_id']
        and done['fresh_runtime_sha256']==sha(runtime) and done['actual_dispatch_source_sha256']==sha(__file__),
        'Actual owned complete uniform child/runtime required')
    for key,kind in (('memory','TSC_uniform_full_memory_runtime_v1'),('cache','TSC_uniform_full_cache_runtime_v1'),('io','TSC_uniform_full_atomic_IO_v1')):
        value=read(paths[key]);require(value['kind']==kind and value['status']=='complete' and value['sources_unchanged'] is True,'Runtimefailed:'+key)
    memory,cache=read(paths['memory']),read(paths['cache'])
    require(memory['cap_bytes']==8000000000 and memory['cleanup_threshold_bytes']==3000000000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==selection['completed_steps']-plan['resume_step'],
        'Retain cap8/end3/join/currentwindow hook scope')
    history=read(run/'training_history.json');config=read(run/'run_config.json')
    require(selection['run_id']==plan['run_id'] and selection['configured_steps']==38225
        and len(history)==selection['completed_steps'] and 6500<=len(history)<=38225
        and selection['heldout_tests_evaluated'] is False and config['trainable_parameter_tensors']==207
        and config['turkish_clips_used']==253798 and config['replay_clips_used']==1914,'Full207/253798/1914 frontier required')
    proof=read(run/'uniform_full_initial_source_equivalence.json')
    require(proof['passed'] is True and proof['actual_GPU_decode'] is True
        and proof['compared_fixed_DEV_utterances']==300 and proof['Portuguese_fixed_DEV_measured_utterances']==50
        and proof['all_six_retention_gates_passed'] is True,'Actual original300+Portuguese50 sourcegate required')
    return {'full207_frontier':selection['completed_steps'],'actual_initial_source350':record(run/'uniform_full_initial_source_equivalence.json'),
        'runtime_completion':record(runtime.with_suffix('.completion.json')),'full_unique_uniform_pass_complete':selection['full_new_uniform_pass_complete']}


def request_stop(path,reason):
    if not path.exists():
        with path.open('x',encoding='utf-8') as stream:stream.write('Cooperative whole-update save/DEV: '+reason+'\\n')


'''
window=fn('run_window').replace('last_dev=1500','last_dev=state[\'resume_step\']')
window=window.replace("f\"ACTUAL owned launch {state['run_id']} PID{child.pid}; BEST1500 full state, declared headLR1e-5\"","f\"ACTUAL owned full-pass {state['run_id']} PID{child.pid}; original6500 full207, uniform31725 updates\"")
window=window.replace("phase=domain.guards.log_phase(paths['log'],history)","suffix=[{**row,'step':i} for i,row in enumerate(history[state['resume_step']:],1)]\n                phase=domain.guards.log_phase(paths['log'],suffix)")
window=window.replace('progress.observe(history,time.monotonic())','progress.observe(suffix,time.monotonic())')
window=window.replace('if not history and time.monotonic()-started>1800','if not suffix and time.monotonic()-started>1800')
window=window.replace("if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'", "if paths['stop'].exists() and stop_at is None:reason=reason or 'ROOT_manual_cooperative_stop'\n                if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'")
window=window.replace("with paths['stop'].open('x',encoding='utf-8') as out:out.write('Cooperative save/DEV requested: '+reason+'\\n')","request_stop(paths['stop'],reason)")
window=window.replace("state['child_exit_code']=child.returncode","state['child_exit_code']=child.wait()")
window=window.replace("resource_stop=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary')","resource_stop=bool(stop_reason and stop_reason not in ('window_budget_checkpoint_boundary','ROOT_manual_cooperative_stop'))")
window=window.replace("stop_category=('resource' if resource_stop else 'retention'", "stop_category=('resource' if resource_stop else 'manual' if stop_reason=='ROOT_manual_cooperative_stop' else 'retention'")
window=window.replace("next_action='ROOT reviews actual DEV and closed full state; no automatic fresh optimizer or next GPU job'", "next_action='Only actual budget CLOSED0 may continue the same fullstate; resource/manual/scientific stops require review'")
text+=window+'\n\n'
text+='''def current_memory_admission():
    sample=sample_memory(os.getpid(),'full_uniform_before_GPU_window');memory=sample['system_memory']
    historical_peak=12176424960
    require(memory['available'] is True and memory['commit_used_bytes']+historical_peak<.9*memory['commit_limit_bytes']
        and memory['commit_limit_bytes']-memory['commit_used_bytes']-historical_peak>2000000000,
        'Current baseline plus historical owned peak fails unchanged90%/headroom guards; repair resources before dispatch')
    return {'sample':memory,'historical_peak_private_bytes':historical_peak,'projection_is_not_actual_fit_proof':True}


def exact_successor(previous,output,args):
    require(previous['status']=='train_window_ended' and previous['child_exit_code']==0
        and previous['resource_review_required'] is False and previous['stop_category']=='budget',
        'Only actual owned budget closure may auto-continue')
    run=HERE/'runs'/args.run_id;selection=read(run/'selection.json');history=read(run/'training_history.json')
    require(6500<selection['completed_steps']<38225 and history[-1].get('validation',{}).get('retention_gate_passed') is True,
        'Latest complete fixed350 DEV must pass allsix gates before automatic budget successor')
    latest=scoped(selection['latest_weights']);saved=output/(args.run_id+'_closed_latest.pt')
    require(not output.exists(),'Fresh successor namespace');output.mkdir(parents=True)
    bound=record(latest);shutil.copy2(latest,saved);require(sha(saved)==bound['sha256'] and record(latest)==bound,'Owned closed latest archive changed')
    plan=dict(previous);argv=list(previous['worker_arguments'])
    paths={key:output/(args.run_id+'_'+name) for key,name in (('native','unused_original.json'),('fresh_runtime','fresh_native.json'),
        ('identity','identity.json'),('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    paths['native']=Path(previous['paths']['native'])
    for flag,value in (('--resume-exact',saved),('--uniform-full-runtime-proof',paths['fresh_runtime']),
        ('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    plan.update(kind=CONTROLLER_KIND,continuation_mode='same_new_run_exact_budget_continuation',
        original_ready=record(paths['native']),source_checkpoint=record(saved),resume_step=selection['completed_steps'],
        worker_arguments=argv,worker_arguments_sha256=domain.guards.argument_sha(argv),paths={key:str(value) for key,value in paths.items()},
        command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'])
    for key in ('child_exit_code','resource_review_required','stop_category','cooperative_stop_reason','actual_runtime_completion','finished_at_utc'):
        plan.pop(key,None)
    validate_plan(plan);return plan


def main(args):
    if args.worker_plan:
        require(args.gpu_window_authorized,'ROOT exclusive GPU authorization required');run_worker(read(scoped(args.worker_plan)));return 0
    require(all((args.source_receipt,args.output_dir,args.run_id)),'ROOT source/newrun/output required')
    require(math.isfinite(args.max_hours) and args.window_hours<=args.max_hours<=96,'Finite overall fullrun time bound required')
    plan=prepare(args)
    if args.validate_only:
        if args.preflight_output:
            require(not scoped(args.preflight_output).exists() and not scoped(args.preflight_output).is_relative_to(scoped(args.output_dir)),'Fresh outside preflight evidence required')
            write(args.preflight_output,{**plan,'passed':True,'GPU_launched':False,'Torch_imported':'torch' in sys.modules})
        print(json.dumps({'passed':True,'GPU_launched':False,'configured_steps':38225,'new_uniform_updates':31725}));return 0
    require(args.gpu_window_authorized,'ROOT sole GPU authorized only after actualsource/PT baseline and resource repair')
    first=scoped(args.output_dir);first.mkdir(parents=True);overall=time.monotonic()+args.max_hours*3600;window=1
    while True:
        admission=current_memory_admission();loop=Path(plan['paths']['log']).parent
        state={**plan,'status':'starting','fresh_memory_admission':admission,'updated_at_utc':now()}
        deadline=min(overall,time.monotonic()+args.window_hours*3600)
        state['awake_request']=domain.guards.prior.keep_awake(True)
        try:
            write(loop/'plan.json',plan);write(loop/'loop_status.json',state)
            args.output_dir=loop;code=run_window(state,deadline,args)
        except BaseException as error:
            state.update(status='failed',error=repr(error),finished_at_utc=now());raise
        finally:
            state['awake_clear']=domain.guards.prior.keep_awake(False);write(loop/'loop_status.json',state)
        write(first/'full_run_status.json',{'status':state['status'],'current_closed_window':str(loop),
            'completed_steps':state.get('completed_steps'),'objective_complete':False,'fullcoverage_by_budget_alone':False})
        selection=read(HERE/'runs'/args.run_id/'selection.json')
        if code or state['stop_category']!='budget' or selection['completed_steps']>=38225:return code
        if overall-time.monotonic()<600:return 0
        window+=1;next_dir=first.with_name(first.name+'_window_'+str(window))
        plan=exact_successor(state,next_dir,args)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-plan',type=Path)
    p.add_argument('--source-receipt',type=Path);p.add_argument('--run-id');p.add_argument('--output-dir',type=Path)
    p.add_argument('--portuguese-replay-manifest',type=Path,default=ROOT/'work/data/portuguese_retention_v1/replay_train_pt.jsonl')
    p.add_argument('--portuguese-dev-manifest',type=Path,default=ROOT/'work/data/portuguese_retention_v1/validation_pt.jsonl')
    p.add_argument('--portuguese-seed',type=int,default=20261012)
    p.add_argument('--window-hours',type=float,default=3.);p.add_argument('--max-hours',type=float,default=48.)
    p.add_argument('--poll-seconds',type=float,default=5.);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--preflight-output',type=Path);p.add_argument('--gpu-window-authorized',action='store_true');return p


if __name__=='__main__':sys.exit(main(parser().parse_args()))
'''
ast.parse(text)
target=HERE/'run_TSC_uniform_full_v1.py'
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print(target)
