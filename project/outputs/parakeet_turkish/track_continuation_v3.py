"""Trackio mirror v3: model DEV and original-base reference use distinct keys.

Keeps the original cumulative metrics. Post-resume rates subtract the inherited
checkpoint's counters only for warm_start.mode=exact_resume. This wrapper reads
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


def development_cycle(run_id,history):
    """Read only each relevant DEV summary once, before processing rows."""
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
    for record in history:
        values=values_for(record,context)
        values.update(development.get(record['step'],{}))
        yield record['step'],values


def follow(args):
    state_path=args.state_path
    if state_path.exists():
        state=read_json(state_path,{})
    else:
        # Keep previous dashboard backfill/terminal bookkeeping when replacing
        # the root-owned original mirror. Future state has a separate filename.
        state=read_json(ROOT/'work/trackio_continuation_mirror_state_v2.json',{})
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
            development=development_cycle(run_id,history)
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
                initial=development.get(0,{})
                old_initial=state[run_id].setdefault('0',{})
                changed_initial={k:v for k,v in initial.items() if old_initial.get(k)!=v}
                if changed_initial:
                    trackio.log(changed_initial,step=0);old_initial.update(changed_initial)
                print(f'Mirroring {run_id} with cumulative and exact-resume interval rates',flush=True)
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
    p.add_argument('--state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v3.json')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if not args.state_path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('State path must be inside workspace')
    try:follow(args)
    finally:trackio.finish()
