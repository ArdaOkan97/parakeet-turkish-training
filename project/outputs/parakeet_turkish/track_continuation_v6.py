"""INACTIVE Trackio mirror v6: declared ordinary/low-LoRa phase2 rates, preserving v5.

Keeps the original cumulative metrics. Post-resume rates subtract the inherited
checkpoint's counters for warm_start.mode=exact_resume. A declared acquisition
phase separately subtracts its selected parent counters, keeps local Adam clocks,
and reads only genuinely published new-run DEV folders. Phase2 separately uses
its chosenS+K counters and rowK+N/LoRAS+N clocks. This wrapper reads
config/history once per mirror cycle and never changes the training worker.
Root owns starting/stopping this mirror; importing it does not start a mirror.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import track_experiment as original
from atomic_io_retry_v1 import read_json, write_json

OUTPUT=original.OUTPUT
ROOT=original.ROOT
trackio=original.trackio


def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def resume_context(config,history):
    """Build a cycle-local immutable baseline, never rereading files per row."""
    warm=config.get('warm_start',{})
    if warm.get('mode')!='exact_resume':
        return None
    step=warm.get('source_step')
    if not isinstance(step,int) or isinstance(step,bool) or step<0:
        raise ValueError('Exact resume needs an integer nonnegative source step')
    if step==0:
        return {'step':0,'audio_seen_seconds':0.,'train_wall_seconds':0.,'pending':False}
    baseline=next((r for r in history if r.get('step')==step),None)
    if baseline is None:
        return {'step':step,'pending':True}
    audio=baseline.get('audio_seen_seconds');wall=baseline.get('train_wall_seconds')
    if not finite_number(audio) or not finite_number(wall) or audio<0 or wall<0:
        raise ValueError('Inherited exact-resume throughput counters must be finite and nonnegative')
    return {'step':step,'audio_seen_seconds':audio,'train_wall_seconds':wall,'pending':False}


def values_for(record,context):
    values=original.metrics_for(record)
    if context is None or record['step']<=context['step']:
        return values
    if context['pending']:
        return values
    audio=record.get('audio_seen_seconds');wall=record.get('train_wall_seconds')
    if not finite_number(audio) or not finite_number(wall):
        raise ValueError('New exact-resume cumulative counters must be finite')
    audio-=context['audio_seen_seconds'];wall-=context['train_wall_seconds']
    if audio<=0 or wall<=0:
        raise ValueError('New exact-resume interval counters must increase beyond the inherited baseline')
    values.update({
        'resources/post_resume_audio_minutes_per_wall_minute':audio/wall,
        'resources/post_resume_train_wall_seconds':wall,
        'data/post_resume_audio_seen_seconds':audio,
        'data/post_resume_completed_updates':record['step']-context['step'],
    })
    return values


ACQUISITION_PHASE='HF_TSC_BPE512_new_rows_only_acquisition_v1'


def acquisition_context(config,history):
    """Cycle-local parent counters; no per-record file reads or retagged DEV."""
    if config.get('phase')!=ACQUISITION_PHASE:
        return None
    identity=config.get('config_identity',{}).get('acquisition_phase',{})
    step=config.get('source_selected_step')
    if (not isinstance(step,int) or isinstance(step,bool) or step<0
        or config.get('inherited_history_steps')!=step
        or identity.get('kind')!=ACQUISITION_PHASE or identity.get('source_selected_step')!=step
        or identity.get('active_optimizer_tables')!=3 or identity.get('acquisition_lr')!=1e-4
        or identity.get('source_Adam_retained_dormant') is not True
        or identity.get('unfreeze_allowed') is not False):
        raise ValueError('Declared row-only acquisition identity/counters required')
    baseline=next((row for row in history if row.get('step')==step),None)
    if baseline is None:
        raise ValueError('Acquisition parent throughput row is missing')
    result={'step':step,'acquisition_lr':identity['acquisition_lr']}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:
            raise ValueError('Acquisition parent counters must be finite and nonnegative')
        result[key]=value
    return result


def acquisition_values_for(record,context):
    if context is None or record['step']<context['step']:
        return {}
    updates=record['step']-context['step']
    values={'phase/acquisition_updates_completed':updates,
            'phase/acquisition_lr':context['acquisition_lr'],
            'phase/lora_active':0,'phase/active_optimizer_tables':3,
            'phase/new_BPE_update_multiplier':1}
    if updates==0:
        return values
    if (record.get('acquisition_updates_completed')!=updates
        or record.get('acquisition_lr')!=context['acquisition_lr']
        or record.get('head_lr')!=0 or record.get('lora_lr')!=0
        or record.get('new_token_update_multiplier')!=1
        or record.get('old_six_character_update_multiplier_applied') is not False):
        raise ValueError('Acquisition row must use its actual local Adam clock and active-only recipe')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=record.get(key)
        if not finite_number(value):raise ValueError('Acquisition cumulative counters must be finite')
        delta[key]=value-context[key]
        if delta[key]<=0:raise ValueError('Acquisition counters must increase beyond the parent')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_acquisition_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_acquisition_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_acquisition_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_acquisition_train_wall_seconds':wall,
        'data/post_acquisition_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_acquisition_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_acquisition_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    return values


ADAPTATION_PHASE='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_v1'
OLD_ADAPTATION_IDENTITY='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_identity_v1'
LOW_LR_ADAPTATION_IDENTITY='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_identity_v1'


def adaptation_context(config,history):
    """Chosen acquired source baseline; no inherited DEV alias or file rereads."""
    if config.get('phase')!=ADAPTATION_PHASE:
        return None
    phase=config.get('config_identity',{}).get('adaptation_phase',{})
    optimizer=config.get('config_identity',{}).get('adaptation_optimizers',{})
    step=config.get('source_phase1_step')
    original_step=config.get('original_source_selected_step')
    acquired=config.get('selected_acquisition_updates')
    optimizer_kind=optimizer.get('kind')
    if optimizer_kind==OLD_ADAPTATION_IDENTITY:
        active_lora_lr=3e-4
    elif (optimizer_kind==LOW_LR_ADAPTATION_IDENTITY
            and optimizer.get('source_original_LoRA_lr')==3e-4
            and optimizer.get('active_LoRA_lr')==3e-5
            and optimizer.get('source_LoRA_moments_reset') is False):
        active_lora_lr=3e-5
    else:
        raise ValueError('Only declared ordinary3e-4 or low-LoRa3e-5 phase2 optimizer identity is supported')
    if (not all(type(value) is int and value>0 for value in (step,original_step,acquired))
        or step!=original_step+acquired or config.get('inherited_history_steps')!=step
        or phase.get('kind')!=ADAPTATION_PHASE or phase.get('source_phase1_step')!=step
        or phase.get('original_source_selected_step')!=original_step
        or phase.get('selected_acquisition_updates')!=acquired
        or phase.get('active_parameter_tensors')!=195
        or phase.get('row_lr')!=1e-4 or phase.get('lora_lr')!=active_lora_lr
        or optimizer.get('row_group',{}).get('lr')!=1e-4
        or optimizer.get('lora_group',{}).get('lr')!=active_lora_lr
        or config.get('config_identity',{}).get('training_arguments',{}).get('lora_lr')!=3e-4
        or phase.get('old15_head_moments_active') is not False
        or phase.get('original207_Adam_artifact_retained_dormant') is not True
        or phase.get('further_unfreeze_allowed') is not False
        or optimizer.get('fixed_shared_head_tensors')!=12):
        raise ValueError('Declared phase2 S/K frontier,195 active tensors and protected12 shared heads required')
    baseline=next((row for row in history if row.get('step')==step),None)
    if baseline is None or baseline.get('acquisition_updates_completed')!=acquired:
        raise ValueError('Actual chosen phase1 history row/localK clock is missing')
    context={'step':step,'original_step':original_step,'acquired':acquired,
        'row_lr':phase['row_lr'],'lora_lr':phase['lora_lr'],'source_original_lora_lr':3e-4,
        'optimizer_identity_kind':optimizer_kind}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:
            raise ValueError('Chosen phase1 cumulative throughput counters must be finite and nonnegative')
        context[key]=value
    return context


def adaptation_values_for(record,context):
    if context is None or record['step']<context['step']:
        return {}
    updates=record['step']-context['step']
    values={'phase/adaptation_updates_completed':updates,
        'phase/source_phase1_step':context['step'],
        'phase/original_source_selected_step':context['original_step'],
        'phase/selected_acquisition_updates':context['acquired'],
        'phase/row_Adam_clock':context['acquired']+updates,
        'phase/LoRA_Adam_clock':context['original_step']+updates,
        'phase/row_lr':context['row_lr'],'phase/lora_lr':context['lora_lr'],'phase/lora_active':1,
        'phase/active_parameter_tensors':195,'phase/fixed_shared_head_tensors':12,
        'phase/new_BPE_update_multiplier':1}
    if updates==0:
        return values
    if (record.get('adaptation_updates_completed')!=updates
        or record.get('source_phase1_step')!=context['step']
        or record.get('original_source_selected_step')!=context['original_step']
        or record.get('selected_acquisition_updates')!=context['acquired']
        or record.get('row_Adam_clock')!=context['acquired']+updates
        or record.get('LoRA_Adam_clock')!=context['original_step']+updates
        or record.get('row_lr')!=1e-4 or record.get('head_lr')!=0
        or record.get('lora_lr')!=context['lora_lr'] or record.get('new_token_update_multiplier')!=1
        or record.get('old_six_character_update_multiplier_applied') is not False):
        raise ValueError('Phase2 actual localN/two Adam clocks/active recipe differ')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=record.get(key)
        if not finite_number(value):
            raise ValueError('Phase2 cumulative counters must be finite')
        delta[key]=value-context[key]
        if delta[key]<=0:
            raise ValueError('Phase2 counters must increase beyond chosen phase1 baseline')
    if not math.isclose(delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],
            delta['audio_seen_seconds'],rel_tol=1e-8,abs_tol=1e-6):
        raise ValueError('Post-phase2 mixed audio must equal actual TR plus replay deltas')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_adaptation_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_adaptation_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_adaptation_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_adaptation_train_wall_seconds':wall,
        'data/post_adaptation_completed_updates':updates,
        'data/post_adaptation_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_adaptation_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_adaptation_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    return values


def mirror_state(args):
    """Preserve own state, else ALL previous v5 entries; prior-version fallback only if absent."""
    if args.state_path.exists():
        return read_json(args.state_path,{})
    source=args.source_state_path
    if (source==ROOT/'work/trackio_continuation_mirror_state_v5.json' and not source.exists()):
        source=ROOT/'work/trackio_continuation_mirror_state_v4.json'
    if (source==ROOT/'work/trackio_continuation_mirror_state_v4.json' and not source.exists()):
        source=ROOT/'work/trackio_continuation_mirror_state_v3.json'
    return read_json(source,{})


def stop_requested(args):
    path=getattr(args,'stop_file',None)
    return path is not None and path.exists()


def scoped_control_path(path):
    if path is None:
        return None
    path=path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Mirror control path must be inside workspace')
    return path


def start_controls(args):
    """Optional fresh identity/stop paths; exclusive initial identity creation."""
    stop=scoped_control_path(getattr(args,'stop_file',None))
    identity=scoped_control_path(getattr(args,'identity_file',None))
    args.stop_file=stop;args.identity_file=identity
    paths=[args.state_path.resolve(),args.source_state_path.resolve()]
    paths.extend(path for path in (stop,identity) if path is not None)
    if len(paths)!=len(set(paths)):
        raise ValueError('Mirror controls and preserved/new state paths must be distinct')
    if stop is not None and stop.exists():
        raise ValueError('Stale existing mirror stop-file rejected before startup')
    if identity is None:
        return {'path':None,'identity':None}
    if identity.exists():
        raise ValueError('Mirror identity-file must have a fresh path')
    identity.parent.mkdir(parents=True,exist_ok=True)
    value={'schema':1,'kind':'owned_trackio_continuation_mirror_v6','status':'starting',
        'pid':os.getpid(),'ppid':os.getppid(),
        'script_path':str(Path(__file__).resolve()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_executable':sys.executable,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'started_monotonic':time.monotonic(),
        'runstate_identity':{'project':args.project,'state_path':str(args.state_path.resolve()),
            'source_state_path':str(args.source_state_path.resolve()),'active_run_id':None},
        'stop_file':str(stop) if stop is not None else None,
        'process_scope':'Actual self metadata; OS closure is observed separately by ROOT'}
    with identity.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return {'path':identity,'identity':value}


def publish_control_run(control,run_id):
    if control is None or control['path'] is None:
        return
    value=control['identity']
    value.update(status='running',updated_at_utc=datetime.now(timezone.utc).isoformat())
    value['runstate_identity']['active_run_id']=run_id
    write_json(control['path'],value)


def finish_controls(control,status,reason,error,trackio_finish_returned):
    if control is None or control['path'] is None:
        return
    value=control['identity']
    value.update(status=status,stop_reason=reason,error=error,
        trackio_finish_returned=trackio_finish_returned,
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    if trackio_finish_returned:
        value['runstate_identity']['last_run_id']=value['runstate_identity']['active_run_id']
        value['runstate_identity']['active_run_id']=None
    write_json(control['path'],value)


def development_cycle(run_id,history,config=None):
    """Read only each relevant DEV summary once, before processing rows."""
    if config and config.get('phase')==ADAPTATION_PHASE:
        steps={config['source_phase1_step'],*[r['step'] for r in history
                if 'validation' in r and 'adaptation_updates_completed' in r
                and r['step']>config['source_phase1_step']]}
    elif config and config.get('phase')==ACQUISITION_PHASE:
        # Only sourceSelected has an actual initial300 new-run summary. Other
        # inherited DEV rows retain their canonical history, never new-run aliases.
        steps={config['source_selected_step'],*[r['step'] for r in history
                if 'validation' in r and 'acquisition_updates_completed' in r]}
    else:
        steps={0,*[r['step'] for r in history if 'validation' in r]}
    result={}
    for step in steps:
        summary=read_json(OUTPUT/f'results/{run_id}_step_{step}/summary.json',{})
        if summary:
            result[step]={f'dev/wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                          for key,value in summary.items() if key.startswith('validation_')}
    return result


def cycle_rows(config,history,development):
    context=resume_context(config,history)
    phase_context=acquisition_context(config,history)
    adaptation=adaptation_context(config,history)
    for record in history:
        values=values_for(record,context)
        values.update(acquisition_values_for(record,phase_context))
        values.update(adaptation_values_for(record,adaptation))
        if adaptation is not None and record['step']<=adaptation['step']:
            values['phase/inherited_history']=int(record['step']<adaptation['step'])
            # Inherited training counters stay cumulative. DEV/retention points
            # belong to this run only when its actual new-run folder published.
            if record['step'] not in development:
                values={key:value for key,value in values.items() if not key.startswith('retention/')}
        values.update(development.get(record['step'],{}))
        yield record['step'],values


def follow(args,control=None):
    state_path=args.state_path
    state=mirror_state(args)
    active=None
    finished={k for k,v in state.items() if v.get('finished')}
    while True:
        if stop_requested(args):
            return 'cooperative_stop_file'
        config_path=OUTPUT/'run_config.json';history_path=OUTPUT/'training_history.json'
        config=read_json(config_path,{})
        history=read_json(history_path,[])
        run_id=config.get('run_id')
        coherent=bool(run_id and history and config_path.exists() and history_path.exists()
                      and history_path.stat().st_mtime>=config_path.stat().st_mtime)
        if coherent and run_id not in finished:
            development=development_cycle(run_id,history,config)
            if active!=run_id:
                if active is not None:trackio.finish()
                trackio.init(project=args.project,name=run_id,group='Turkish adaptation',
                             config=config,resume='allow',gpu_log_interval=10.)
                active=run_id;state.setdefault(run_id,{})
                publish_control_run(control,run_id)
                if 'reference_base_v3' not in state[run_id]:
                    baseline=read_json(OUTPUT/'results/base_v3/summary.json',{})
                    trackio.log({f'reference/base_v3_wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                                 for key,value in baseline.items() if key.startswith('validation_')},step=0)
                    state[run_id]['reference_base_v3']=True
                initial_step=(config['source_phase1_step'] if config.get('phase')==ADAPTATION_PHASE
                    else config['source_selected_step'] if config.get('phase')==ACQUISITION_PHASE else 0)
                initial=development.get(initial_step,{})
                old_initial=state[run_id].setdefault(str(initial_step),{})
                changed_initial={k:v for k,v in initial.items() if old_initial.get(k)!=v}
                if changed_initial:
                    trackio.log(changed_initial,step=initial_step);old_initial.update(changed_initial)
                print(f'Mirroring {run_id} with cumulative, exact-resume, acquisition and adaptation interval rates',flush=True)
            for step,values in cycle_rows(config,history,development):
                old=state[run_id].setdefault(str(step),{})
                changed={k:v for k,v in values.items() if old.get(k)!=v}
                if changed:
                    trackio.log(changed,step=step);old.update(changed)
            write_json(state_path,state)
            selected=read_json(OUTPUT/'selection.json',{})
            pointer=read_json(OUTPUT/'adapted_results_path.json',{})
            if pointer.get('tag')==run_id+'_adapted':
                final=read_json(OUTPUT/f'results/{run_id}_adapted/summary.json',{})
                if all('test_'+lang in final for lang in ['en','es','it','fr','de','tr']):
                    values={f'test/wer_{key.removeprefix("test_")}_percent':100*value['asr']['wer']
                            for key,value in final.items() if key.startswith('test_')}
                    values['selection/best_step']=selected.get('selected_step',0)
                    trackio.log(values,step=history[-1]['step']);trackio.finish()
                    state[run_id]['finished']='heldout-evaluation-complete'
                    finished.add(run_id);active=None;write_json(state_path,state)
                    print(f'Final held-out scores imported for {run_id}',flush=True)
            elif (selected.get('run_id')==run_id and selected.get('accepted') is False
                  and isinstance(selected.get('completed_steps'),int)
                  and history[-1]['step']>=selected['completed_steps']):
                trackio.log({'selection/accepted':0,
                             'selection/report':trackio.Markdown(selected.get('reason','Rejected on development gates.'))},
                            step=history[-1]['step'])
                trackio.finish();state[run_id]['finished']='rejected-on-development'
                finished.add(run_id);active=None;write_json(state_path,state)
        if not args.follow:
            if active is not None:trackio.finish()
            return 'one_cycle_complete'
        if stop_requested(args):
            return 'cooperative_stop_file'
        time.sleep(5)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='parakeet-turkish')
    p.add_argument('--follow',action='store_true')
    p.add_argument('--state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v6.json')
    p.add_argument('--source-state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v5.json',
                   help='Preserve existing flags/points on ROOT-owned mirror handoff; used only when new state does not exist')
    p.add_argument('--stop-file',type=Path,help='Optional fresh workspace path; exit between complete mirror cycles when it appears')
    p.add_argument('--identity-file',type=Path,help='Optional fresh workspace path for actual self/start/runstate and final shutdown metadata')
    return p


def main(args):
    for path in (args.state_path,args.source_state_path):
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('State path must be inside workspace')
    if args.state_path.resolve()==args.source_state_path.resolve():raise ValueError('New v6 state must be separate from preserved source state')
    control=start_controls(args)
    status='failed';reason=None;error=None;finished=False
    try:
        reason=follow(args,control)
        status='stopped' if reason=='cooperative_stop_file' else 'complete'
        return 0
    except BaseException as exception:
        error=repr(exception)
        status='interrupted' if isinstance(exception,KeyboardInterrupt) else 'failed'
        raise
    finally:
        try:
            trackio.finish();finished=True
        except BaseException as exception:
            status='failed';reason='trackio_finish_failed';error=repr(exception)
            raise
        finally:
            finish_controls(control,status,reason,error,finished)


if __name__=='__main__':
    main(parser().parse_args())
