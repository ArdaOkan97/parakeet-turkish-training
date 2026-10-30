"""INERT until root confirms a closed parent: SAME-run exact HF continuation.

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
from pathlib import Path
import shutil
import subprocess
import sys
import time

from atomic_io_retry_v1 import read_json as read,write_json as write
import run_hf_quality_loop_v1 as quality
from resource_watchdog_v2 import ResourceGuard,sample_memory
import launch_same_science_resume_v1 as fresh

HERE,ROOT=quality.HERE,quality.ROOT
SOURCE_PINS=quality.SOURCE_PINS


def require(ok,message):
    if not ok:raise ValueError(message)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def record(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def scoped(path):return quality.runtime.previous.scoped(path)


def source_plan(args,require_closed):
    require(0<args.window_hours<=6 and 5<=args.poll_seconds<=15,'Bounded <=6h / fresh5–15s polling required')
    source_loop=scoped(args.source_loop);folder=scoped(args.output_dir)
    require(folder!=source_loop and not folder.is_relative_to(HERE/'runs'/args.run_id),
        'NEW continuation evidence directory outside original run required')
    require(not (folder/'loop_status.json').exists(),'Never overwrite an existing continuation state')
    state=read(source_loop/'loop_status.json',{})
    arm=next((a for a in state.get('trials',[]) if a.get('run_id')==args.run_id),None)
    if arm is None and state.get('run_id')==args.run_id and state.get('worker_arguments'):
        arm=state # A later closed same-science window can be resumed again.
    require(arm is not None,'Declared source arm arguments required')
    if require_closed:
        require(args.source_closed_confirmed,'Root must confirm the owned parent and worker are closed')
        require(state.get('status') in ('quality_window_ended','train_window_ended','resource_review_required','failed')
            and arm.get('child_exit_code')==0 and arm.get('status') in ('complete','train_window_ended','resource_review_required'),
            'Terminal parent and successfully saved closed child required')
    for name,digest in SOURCE_PINS.items():require(sha(HERE/name)==digest,'Frozen same-science runtime source changed: '+name)
    config_path=HERE/'runs'/args.run_id/'run_config.json'
    config=read(config_path,{})
    require(config.get('run_id')==args.run_id and config.get('batch_size')==8 and config.get('accumulation')==2
        and config.get('max_retention_failures')==2 and config.get('eval_every_updates')==250,
        'Preserve exact B8x2/cadence250/internal-stop2 source configuration')
    original_ready=scoped(config['native_backend_provenance'])
    paths={key:folder/(args.run_id+suffix) for key,suffix in {'fresh_runtime':'_fresh_runtime.json','io':'_io.json',
        'cache':'_cache.json','memory':'_memory.json','identity':'_identity.json','stop':'_stop.txt','log':'.log'}.items()}
    require(all(not p.exists() for p in paths.values()),'New runtime/evidence paths required')
    latest=ROOT/'work'/(args.run_id+'_latest.pt');best=ROOT/'work'/(args.run_id+'_best.pt')
    archive=folder/'immutable_closed_latest.pt'
    argv=list(arm['worker_arguments']);original=list(argv)
    for flag,value in (('--resume-weights',None),('--resume-exact',archive),('--stop-file',paths['stop']),
        ('--max-wall-seconds',max(1,args.window_hours*3600-300))):quality.runtime.replace_option(argv,flag,value)
    require('--resume-runtime-migration' not in argv,'Same-science path supports the warm-start scientific arm without migration')
    allowed=('--resume-weights','--resume-exact','--stop-file','--max-wall-seconds')
    def unchanged(values):
        values=list(values)
        for flag in allowed:quality.runtime.replace_option(values,flag,None)
        return values
    require(unchanged(original)==unchanged(argv),'Scientific/source/data/ready identity argument changed')
    require(scoped(quality.runtime.option(argv,'--native-backend-provenance'))==original_ready,
        'Keep original immutable ready path; fresh proof must be separate')
    memory=['--provenance',str(paths['io']),'--cache-provenance',str(paths['cache']),'--memory-provenance',str(paths['memory']),
        '--worker-identity',str(paths['identity']),'--cap-gb','8','--threshold-gb','4','--',*argv]
    original_sha=sha(original_ready)
    command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(HERE/'launch_same_science_resume_v1.py'),
        '--original-native-provenance',str(original_ready),'--original-native-provenance-sha256',original_sha,
        '--fresh-runtime-provenance',str(paths['fresh_runtime']),'--',*memory]
    namespace=argparse.Namespace(original_native_provenance=original_ready,original_native_provenance_sha256=original_sha,
        fresh_runtime_provenance=paths['fresh_runtime'],memory_arguments=memory,validate_only=True)
    fresh.validate(namespace)
    plan={'schema':1,'kind':'HF_same_run_exact_science_continuation_v1','passed':True,'run_id':args.run_id,
        'loop_id':folder.name,'source_loop':str(source_loop),'original_run_config':record(config_path),
        'original_ready':record(original_ready),'paths':{k:str(p) for k,p in paths.items()},'command':command,
        'worker_arguments':argv,'runtime_source_sha256':SOURCE_PINS,'new_preloader':record(Path(fresh.__file__)),
        'controller_sha256':sha(__file__),'scientific_config_identity_sha256':hashlib.sha256(json.dumps(config['config_identity'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
        'source_config_identity':config['config_identity'],'scientific_arguments_unchanged':True,
        'max_retention_failures_unchanged':2,'same_run_id':True,'optimizer_sampler_RNG_counters_reset':False,
        'original_ready_is_not_current_PID_evidence':True,'fresh_actual_preload_before_Torch_required':True,
        'archive_path':str(archive),'source_latest_path':str(latest),'source_best_path':str(best),
        'source_closure_confirmed':require_closed,'heldout_TEST_evaluated':False,'objective_complete':False,
        'CUDA_initialized_by_preflight':False}
    if not require_closed:return plan
    selection_path=HERE/'runs'/args.run_id/'selection.json';history_path=HERE/'runs'/args.run_id/'training_history.json'
    selection=read(selection_path,{});history=read(history_path,[])
    require(selection.get('run_id')==args.run_id and selection.get('accepted') is True
        and selection.get('completed_steps')==len(history)>0
        and [row['step'] for row in history]==list(range(1,len(history)+1)),
        'Closed complete ordered history/qualifying BEST selection required')
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
        require(sha(target)==source_sha,'Closed snapshot byte copy differs: '+role)
        snapshots[role]={'source':{'path':str(path),'sha256':source_sha},'snapshot':record(target)}
    plan.update(initial_step=selection['completed_steps'],inherited_selected_step=selection['selected_step'],
        snapshots=snapshots,original_config_identity_preserved_bytewise=True,closed_full_checkpoint_archived_without_state_rewrite=True)
    return plan


def validate_fresh_runtime(value,plan,identity):
    require(value.get('kind')=='HF_same_science_native_preload_resume_v1' and value.get('status')=='ready'
        and value.get('run_id')==plan['run_id'] and value.get('pid')==identity['pid']
        and value.get('worker_arguments_sha256')==quality.runtime.previous.argument_sha(plan['worker_arguments'])
        and value.get('launcher_sha256')==plan['new_preloader']['sha256']
        and value.get('source_sha256')==fresh.PINS
        and value.get('original_first_start_ready',{}).get('sha256')==plan['original_ready']['sha256'],
        'Actual fresh runtime proof/PID/arguments differ; original ready PID is not fresh evidence')
    return value


def main(args):
    plan=source_plan(args,not args.prepare_only)
    if args.prepare_only or args.validate_only:
        if args.preflight_output:write(scoped(args.preflight_output),plan)
        print(json.dumps({'passed':True,'run_id':args.run_id,'source_closed':plan['source_closure_confirmed'],
            'checkpoint_copied':not args.prepare_only,'GPU_launch':False,'Torch_imported':'torch' in sys.modules}))
        return 0
    require(args.gpu_window_authorized,'Only root may authorize exclusive GPU ownership')
    folder=scoped(args.output_dir);paths={k:Path(v) for k,v in plan['paths'].items()}
    state={**plan,'status':'starting','started_at_utc':now(),'window_hours':args.window_hours,'objective_status':'incomplete'}
    deadline=time.monotonic()+args.window_hours*3600;started=time.monotonic();child=None;stop_at=None;stop_reason=None
    resource=ResourceGuard();progress=quality.runtime.previous.ProgressGuard(started)
    state['awake_request']=quality.runtime.previous.prior.keep_awake(True)
    try:
        write(folder/'loop_status.json',state)
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(plan['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['owned_launcher_pid']=child.pid
            while child.poll() is None:
                history=read(HERE/'runs'/args.run_id/'training_history.json',[])
                identity=quality.validate_identity(read(paths['identity'],{}),plan,child.pid,state['started_at_utc']) if paths['identity'].exists() else None
                if identity:
                    state['fresh_runtime']=validate_fresh_runtime(read(paths['fresh_runtime'],{}),plan,identity)
                phase=quality.runtime.previous.log_phase(paths['log'],history)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=quality.runtime.previous.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
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
        require(child.returncode==0,'Exact-resume worker failed; preserve and review all evidence')
        completion=read(paths['fresh_runtime'].with_suffix('.completion.json'),{})
        require(completion.get('status')=='complete' and completion.get('sources_unchanged') is True
            and completion.get('pid')==state['fresh_runtime']['pid']
            and completion.get('fresh_runtime_proof_sha256')==sha(paths['fresh_runtime']),
            'Actual fresh runtime completion missing or mismatched')
        for key in ('io','cache','memory'):require(read(paths[key],{}).get('status')=='complete','Actual runtime completion missing: '+key)
        require(sha(plan['archive_path'])==plan['snapshots']['latest']['snapshot']['sha256'],
            'Immutable full resume input changed')
        selection=read(HERE/'runs'/args.run_id/'selection.json',{})
        require(selection.get('run_id')==args.run_id and selection.get('completed_steps',0)>=plan['initial_step'],
            'Actual continued durable checkpoint/selection missing')
        state.update(status='resource_review_required' if stop_reason and stop_reason!='window_budget_checkpoint_boundary' else 'train_window_ended',
            child_exit_code=child.returncode,completed_steps=selection['completed_steps'],selected_step=selection['selected_step'],
            selection=record(HERE/'runs'/args.run_id/'selection.json'),finished_at_utc=now(),
            resource_review_required=bool(stop_reason and stop_reason!='window_budget_checkpoint_boundary'),objective_complete=False)
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
        state['awake_clear']=quality.runtime.previous.prior.keep_awake(False)
        write(folder/'loop_status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-closed-confirmed',action='store_true')
    p.add_argument('--window-hours',type=float,default=4.5)
    p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--preflight-output',type=Path)
    p.add_argument('--gpu-window-authorized',action='store_true')
    sys.exit(main(p.parse_args()))
