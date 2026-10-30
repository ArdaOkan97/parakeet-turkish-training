"""INACTIVE Trackio mirror v4: acquisition-only rates plus unchanged exact-resume rates.

Keeps the original cumulative metrics. Post-resume rates subtract the inherited
checkpoint's counters for warm_start.mode=exact_resume. A declared acquisition
phase separately subtracts its selected parent counters, keeps local Adam clocks,
and reads only genuinely published new-run DEV folders. This wrapper reads
config/history once per mirror cycle and never changes the training worker.
Root owns starting/stopping this mirror; importing it does not start a mirror.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
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


def development_cycle(run_id,history,config=None):
    """Read only each relevant DEV summary once, before processing rows."""
    if config and config.get('phase')==ACQUISITION_PHASE:
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
    for record in history:
        values=values_for(record,context)
        values.update(acquisition_values_for(record,phase_context))
        values.update(development.get(record['step'],{}))
        yield record['step'],values


def follow(args):
    state_path=args.state_path
    if state_path.exists():
        state=read_json(state_path,{})
    else:
        # Keep previous dashboard backfill/terminal bookkeeping when replacing
        # the root-owned original mirror. Future state has a separate filename.
        state=read_json(args.source_state_path,{})
    active=None
    finished={k for k,v in state.items() if v.get('finished')}
    while True:
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
                if 'reference_base_v3' not in state[run_id]:
                    baseline=read_json(OUTPUT/'results/base_v3/summary.json',{})
                    trackio.log({f'reference/base_v3_wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                                 for key,value in baseline.items() if key.startswith('validation_')},step=0)
                    state[run_id]['reference_base_v3']=True
                initial_step=config['source_selected_step'] if config.get('phase')==ACQUISITION_PHASE else 0
                initial=development.get(initial_step,{})
                old_initial=state[run_id].setdefault(str(initial_step),{})
                changed_initial={k:v for k,v in initial.items() if old_initial.get(k)!=v}
                if changed_initial:
                    trackio.log(changed_initial,step=initial_step);old_initial.update(changed_initial)
                print(f'Mirroring {run_id} with cumulative, exact-resume and acquisition interval rates',flush=True)
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
            return
        time.sleep(5)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='parakeet-turkish')
    p.add_argument('--follow',action='store_true')
    p.add_argument('--state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v4.json')
    p.add_argument('--source-state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v3.json',
                   help='Preserve existing flags/points on ROOT-owned mirror handoff; used only when new state does not exist')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    for path in (args.state_path,args.source_state_path):
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('State path must be inside workspace')
    if args.state_path.resolve()==args.source_state_path.resolve():raise ValueError('New v4 state must be separate from preserved source state')
    try:follow(args)
    finally:trackio.finish()
