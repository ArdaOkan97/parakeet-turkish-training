"""INACTIVE Trackio mirror v13: declared same-pass14TR/2replay continuation.

V8/V9/V10/V11/V12 archives are preserved. ROOT must confirm V12 CLOSED before handoff.
Only the active run map remains resident; immutable SHA shards preserve unknown
fields. No Trackio database deletion or history rewrite is performed. A raw private disk snapshot uses a1MiB copy buffer; history is read one record
at a time in two passes when changed;
unchanged history reuses bounded contexts and checks small DEV amendments.

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
import gc
import io
import tempfile
import json
import math
import os
import re
from pathlib import Path
import sys
import time

import track_experiment as original
from atomic_io_retry_v1 import read_json, write_json

OUTPUT=original.OUTPUT
ROOT=original.ROOT
trackio=original.trackio
SOURCE_V12_SHA='e34e74d504f398e541a943a5a31e6241e5ee10fc00a5fefbe04673c2cb4f5ab3'
UNIFORM_FULL_PHASE='HF_TSC_8199_uniform_fullpass_with_Portuguese_v1'
UNIFORM_MAJORITY_PHASE='HF_TSC_8199_uniform_Turkish_majority_with_Portuguese_v1'
LOWER_LORA_PHASE='HF_TSC_8199_uniform_Turkish_majority_lower_LoRA_v1'


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


CONTINUED_ACQUISITION_PHASE='HF_TSC_BPE512_new_rows_continued_acquisition_v1'
CONTINUED_ACQUISITION_IDENTITY='HF_TSC_BPE512_new_rows_continued_acquisition_identity_v1'


def continued_acquisition_context(config,history):
    """Chosen phase1 baseline with retained K distinct from new local M."""
    if config.get('phase')!=CONTINUED_ACQUISITION_PHASE:
        return None
    identity=config.get('config_identity',{})
    phase=identity.get('acquisition_phase',{})
    optimizer=identity.get('acquisition_optimizer',{})
    step=config.get('source_selected_step')
    original_step=config.get('original_source_selected_step')
    acquired=config.get('source_acquisition_updates')
    lr=phase.get('acquisition_lr')
    if (not all(type(value) is int and value>0 for value in (step,original_step,acquired))
        or step!=original_step+acquired or config.get('inherited_history_steps')!=step
        or phase.get('kind')!=CONTINUED_ACQUISITION_PHASE or phase.get('source_selected_step')!=step
        or phase.get('source_acquisition_updates')!=acquired or phase.get('source_row_lr')!=1e-4
        or phase.get('active_optimizer_tables')!=3 or lr not in (1e-3,3e-4)
        or phase.get('source_Adam_retained_dormant') is not True or phase.get('unfreeze_allowed') is not False
        or optimizer.get('kind')!=CONTINUED_ACQUISITION_IDENTITY
        or optimizer.get('source_step')!=original_step or optimizer.get('source_acquisition_updates')!=acquired
        or optimizer.get('source_row_lr')!=1e-4 or optimizer.get('active_row_lr')!=lr
        or optimizer.get('source_moments_or_clock_reset') is not False
        or optimizer.get('encoder_LoRA_active') is not False):
        raise ValueError('Declared continued row-only S/K frontier, active LR and retained moments required')
    baseline=next((row for row in history if row.get('step')==step),None)
    if baseline is None or baseline.get('acquisition_updates_completed')!=acquired:
        raise ValueError('Actual chosen phase1 parent K throughput row is missing')
    result={'step':step,'original_step':original_step,'source_acquisition_updates':acquired,'acquisition_lr':lr}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:
            raise ValueError('Continued acquisition parent counters must be finite and nonnegative')
        result[key]=value
    return result


def continued_acquisition_values_for(record,context):
    if context is None or record['step']<context['step']:
        return {}
    updates=record['step']-context['step']
    k=context['source_acquisition_updates']
    lr=context['acquisition_lr']
    values={'phase/acquisition_updates_completed':updates,
        'phase/continued_acquisition_updates_completed':updates,
        'phase/source_acquisition_updates':k,'phase/row_Adam_clock':k+updates,
        'phase/source_phase1_step':context['step'],'phase/original_source_selected_step':context['original_step'],
        'phase/acquisition_lr':lr,'phase/source_row_lr':1e-4,'phase/lora_active':0,
        'phase/active_optimizer_tables':3,'phase/frozen_parameter_tensors':204,'phase/new_BPE_update_multiplier':1}
    if updates==0:
        return values
    if (record.get('acquisition_updates_completed')!=updates
        or record.get('source_acquisition_updates')!=k or record.get('row_Adam_clock')!=k+updates
        or record.get('acquisition_lr')!=lr or record.get('active_row_lr')!=lr
        or record.get('source_row_lr')!=1e-4 or record.get('head_lr')!=0 or record.get('lora_lr')!=0
        or record.get('new_token_update_multiplier')!=1
        or record.get('old_six_character_update_multiplier_applied') is not False):
        raise ValueError('Continued acquisition local M must remain distinct from retained row Adam K+M')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=record.get(key)
        if not finite_number(value):raise ValueError('Continued acquisition counters must be finite')
        delta[key]=value-context[key]
        if delta[key]<=0:raise ValueError('Continued acquisition counters must increase beyond chosen phase1')
    if not math.isclose(delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],
            delta['audio_seen_seconds'],rel_tol=1e-8,abs_tol=1e-6):
        raise ValueError('Continued acquisition mixed audio must equal TR plus replay')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_acquisition_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_acquisition_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_acquisition_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_acquisition_train_wall_seconds':wall,
        'data/post_acquisition_completed_updates':updates,
        'data/post_acquisition_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_acquisition_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_acquisition_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    return values


SCHEDULED_ACQUISITION_PHASE='HF_TSC_BPE512_new_rows_scheduled_acquisition_v1'
SCHEDULED_ACQUISITION_IDENTITY='HF_TSC_BPE512_new_rows_scheduled_acquisition_identity_v1'
SCHEDULED_UTILITY_SHA='f72485b7d2a0c242cf3a65c77a9be8e514c281d74c1a6ed924d3ea32eb8e0875'


def scheduled_row_lr(update,peak):
    """Exact declared pure schedule; localN, never retained AdamK+N."""
    if type(update) is not int or not 0<=update<=1000 or peak not in (1e-3,3e-4):
        raise ValueError('Scheduled localN0..1000 and approved peak required')
    if update==0:return 1e-4
    if update<=50:return peak*update/50
    progress=(update-50)/950
    return peak*(.1+.9*.5*(1.+math.cos(math.pi*progress)))


def scheduled_acquisition_context(config,history):
    if config.get('phase')!=SCHEDULED_ACQUISITION_PHASE:return None
    identity=config.get('config_identity',{})
    phase=identity.get('acquisition_phase',{});optimizer=identity.get('acquisition_optimizer',{})
    step=config.get('source_selected_step');s=config.get('original_source_selected_step')
    k=config.get('source_acquisition_updates');peak=optimizer.get('peak_row_lr')
    schedule={'kind':'linear_warmup_cosine_floor_v1','warmup_updates':50,'horizon_updates':1000,
        'min_lr_ratio':.1,'peak_row_lr':peak,'clock':'local_scheduled_updates_N',
        'apply_before_Adam':True,'warmup_resets_Adam_moments_or_clock':False}
    if (not all(type(v) is int and v>0 for v in (step,s,k)) or step!=s+k
        or config.get('inherited_history_steps')!=step or peak not in (1e-3,3e-4)
        or phase.get('kind')!=SCHEDULED_ACQUISITION_PHASE or phase.get('source_selected_step')!=step
        or phase.get('source_acquisition_updates')!=k or phase.get('source_row_lr')!=1e-4
        or phase.get('active_optimizer_tables')!=3 or phase.get('source_Adam_retained_dormant') is not True
        or phase.get('unfreeze_allowed') is not False or phase.get('peak_row_lr')!=peak
        or phase.get('acquisition_lr')!=peak or phase.get('row_schedule')!=schedule
        or optimizer.get('kind')!=SCHEDULED_ACQUISITION_IDENTITY
        or optimizer.get('utility_sha256')!=SCHEDULED_UTILITY_SHA
        or optimizer.get('source_step')!=s or optimizer.get('source_acquisition_updates')!=k
        or optimizer.get('source_row_lr')!=1e-4 or optimizer.get('active_row_lr')!=peak
        or optimizer.get('base_lr')!=peak or optimizer.get('row_schedule')!=schedule
        or optimizer.get('source_moments_or_clock_reset') is not False
        or optimizer.get('encoder_LoRA_active') is not False
        or optimizer.get('dynamic_applied_lr_is_not_scientific_identity') is not True):
        raise ValueError('Pinned scheduled row-only S/K, static peak and no-reset policy required')
    baseline=next((row for row in history if row.get('step')==step),None)
    if baseline is None or baseline.get('acquisition_updates_completed')!=k:
        raise ValueError('Scheduled acquisition actual phase1K baseline is missing')
    context={'step':step,'original_step':s,'source_acquisition_updates':k,'peak_row_lr':peak,'row_schedule':schedule}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:raise ValueError('Scheduled parent counters must be finite/nonnegative')
        context[key]=value
    return context


def scheduled_acquisition_values_for(record,context):
    if context is None or record['step']<context['step']:return {}
    n=record['step']-context['step'];k=context['source_acquisition_updates'];peak=context['peak_row_lr']
    applied=scheduled_row_lr(n,peak)
    values={'phase/acquisition_updates_completed':n,'phase/scheduled_acquisition_updates_completed':n,
        'phase/scheduler_local_N':n,'phase/source_acquisition_updates':k,'phase/row_Adam_clock':k+n,
        'phase/source_phase1_step':context['step'],'phase/original_source_selected_step':context['original_step'],
        'phase/schedule_peak_row_lr':peak,'phase/acquisition_lr':applied,'phase/current_row_lr':applied,
        'phase/source_row_lr':1e-4,'phase/schedule_warmup_updates':50,'phase/schedule_horizon_updates':1000,
        'phase/schedule_min_lr_ratio':.1,'phase/lora_active':0,'phase/active_optimizer_tables':3,
        'phase/frozen_parameter_tensors':204,'phase/new_BPE_update_multiplier':1}
    if n==0:return values
    if (record.get('acquisition_updates_completed')!=n or record.get('scheduled_acquisition_updates')!=n
        or record.get('source_acquisition_updates')!=k or record.get('row_Adam_clock')!=k+n
        or record.get('peak_row_lr')!=peak or record.get('acquisition_lr')!=applied
        or record.get('active_row_lr')!=applied or record.get('current_row_lr')!=applied
        or record.get('last_applied_row_lr')!=applied or record.get('source_row_lr')!=1e-4
        or record.get('scheduled_next_update')!=(n+1 if n<1000 else None)
        or record.get('head_lr')!=0 or record.get('lora_lr')!=0 or record.get('new_token_update_multiplier')!=1
        or record.get('old_six_character_update_multiplier_applied') is not False):
        raise ValueError('Scheduled actual applied LR/localN must match deterministic policy and AdamK+N')
    values['phase/last_applied_row_lr']=applied
    values['phase/scheduler_remaining_updates']=1000-n
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=record.get(key)
        if not finite_number(value):raise ValueError('Scheduled counters must be finite')
        delta[key]=value-context[key]
        if delta[key]<=0:raise ValueError('Scheduled counters must increase beyond source7000')
    if not math.isclose(delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],
        delta['audio_seen_seconds'],rel_tol=1e-8,abs_tol=1e-6):
        raise ValueError('Scheduled audio components differ')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_acquisition_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_acquisition_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_acquisition_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_acquisition_train_wall_seconds':wall,'data/post_acquisition_completed_updates':n,
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



SHARED_HEAD_PHASE='HF_TSC_BPE512_new_rows_plus_shared_heads_v1'
SHARED_HEAD_IDENTITY='HF_TSC_BPE512_new_rows_plus_shared_heads_identity_v1'
SHARED_HEAD_UTILITY_SHA='384d01abe9b7105761491b8acda5d3de4eb8c40ae679eee2701269079cf9967c'
SOURCE_V8_SHA='9653e58db7f525007c7c2dd679f5a872592016335a3e2d6ade9be857c36684b7'


def shared_head_context(config,history):
    if config.get('phase')!=SHARED_HEAD_PHASE:
        return None
    identity=config.get('config_identity',{})
    phase=identity.get('adaptation_phase',{})
    optimizer=identity.get('adaptation_optimizers',{})
    step=config.get('source_phase1_step');original_step=config.get('original_source_selected_step')
    acquired=config.get('selected_acquisition_updates')
    row_lr=phase.get('row_lr');shared_lr=phase.get('shared_lr')
    if (not all(type(value) is int and value>0 for value in (step,original_step,acquired))
        or step!=original_step+acquired or config.get('inherited_history_steps')!=step
        or identity.get('phase')!=SHARED_HEAD_PHASE or phase.get('kind')!=SHARED_HEAD_PHASE
        or phase.get('source_phase1_step')!=step or phase.get('original_source_selected_step')!=original_step
        or phase.get('selected_acquisition_updates')!=acquired
        or phase.get('active_parameter_tensors')!=15 or phase.get('active_shared_head_tensors')!=12
        or phase.get('frozen_encoder_LoRA_tensors')!=192 or row_lr not in (1e-3,3e-4)
        or shared_lr not in (1e-5,3e-6) or phase.get('original12_shared_head_moments_active') is not True
        or phase.get('original3_resized_table_moments_active') is not False
        or phase.get('original192_encoder_LoRA_moments_active') is not False
        or phase.get('original207_Adam_artifact_retained_dormant') is not True
        or phase.get('further_unfreeze_allowed') is not False
        or optimizer.get('kind')!=SHARED_HEAD_IDENTITY or optimizer.get('phase')!=SHARED_HEAD_PHASE
        or optimizer.get('utility_sha256')!=SHARED_HEAD_UTILITY_SHA
        or optimizer.get('source_selected_step')!=original_step or optimizer.get('selected_acquisition_updates')!=acquired
        or optimizer.get('active_parameter_tensors')!=15 or optimizer.get('active_shared_head_tensors')!=12
        or optimizer.get('frozen_encoder_LoRA_tensors')!=192
        or optimizer.get('active_row_lr')!=row_lr or optimizer.get('source_row_lr')!=1e-4
        or optimizer.get('active_shared_lr')!=shared_lr or optimizer.get('source_shared_lr')!=1e-5
        or optimizer.get('row_group',{}).get('lr')!=row_lr or optimizer.get('row_group',{}).get('weight_decay')!=0
        or optimizer.get('shared_group',{}).get('lr')!=shared_lr or optimizer.get('shared_group',{}).get('weight_decay')!=.01
        or optimizer.get('source_moments_or_clocks_reset') is not False
        or optimizer.get('original12_shared_moments_active') is not True
        or optimizer.get('original3_resized_table_moments_active') is not False
        or optimizer.get('original192_encoder_LoRA_moments_active') is not False
        or optimizer.get('effective_new_rows')!=[8199,8711] or optimizer.get('protected_blank')!=8192
        or optimizer.get('protected_six_rows')!=[8193,8199] or optimizer.get('protected_duration_rows')!=[8711,8716]
        or optimizer.get('mask_before_single_global_clip') is not True or optimizer.get('clip_max_norm')!=1
        or optimizer.get('encoder_projector_must_be_outside_encoder_no_grad') is not True
        or optimizer.get('further_unfreeze_allowed') is not False
        or identity.get('training_arguments',{}).get('lora_lr')!=3e-4
        or identity.get('training_arguments',{}).get('head_lr')!=1e-5):
        raise ValueError('Shared-head phase requires the pinned15-active/192-frozen/two-Adam recipe')
    baseline=next((row for row in history if row.get('step')==step),None)
    if baseline is None or baseline.get('acquisition_updates_completed')!=acquired:
        raise ValueError('Actual chosen phase1 S+K history row is missing')
    context={'step':step,'original_step':original_step,'acquired':acquired,'row_lr':row_lr,'shared_lr':shared_lr}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:
            raise ValueError('Shared-head inherited throughput counters must be finite and nonnegative')
        context[key]=value
    return context


def shared_head_values_for(record,context):
    if context is None or record['step']<context['step']:
        return {}
    n=record['step']-context['step']
    values={'phase/shared_head_updates_completed':n,'phase/adaptation_updates_completed':n,
        'phase/source_phase1_step':context['step'],'phase/original_source_selected_step':context['original_step'],
        'phase/selected_acquisition_updates':context['acquired'],'phase/row_Adam_clock':context['acquired']+n,
        'phase/shared_Adam_clock':context['original_step']+n,'phase/row_lr':context['row_lr'],
        'phase/shared_lr':context['shared_lr'],'phase/lora_lr':0.,'phase/lora_active':0,
        'phase/active_parameter_tensors':15,'phase/active_shared_head_tensors':12,
        'phase/frozen_encoder_LoRA_tensors':192,'phase/new_BPE_update_multiplier':1}
    if n==0:return values
    if (record.get('adaptation_updates_completed')!=n or record.get('source_phase1_step')!=context['step']
        or record.get('original_source_selected_step')!=context['original_step']
        or record.get('selected_acquisition_updates')!=context['acquired']
        or record.get('row_Adam_clock')!=context['acquired']+n
        or record.get('shared_Adam_clock')!=context['original_step']+n
        or record.get('row_lr')!=context['row_lr'] or record.get('shared_lr')!=context['shared_lr']
        or record.get('head_lr')!=context['shared_lr'] or record.get('lora_lr')!=0
        or record.get('new_token_update_multiplier')!=1
        or record.get('old_six_character_update_multiplier_applied') is not False):
        raise ValueError('Shared-head actual localN/two Adam clocks/row and shared LR differ')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=record.get(key)
        if not finite_number(value):raise ValueError('Shared-head counters must be finite')
        delta[key]=value-context[key]
        if delta[key]<=0:raise ValueError('Shared-head counters must increase beyond chosen phase1')
    if not math.isclose(delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],delta['audio_seen_seconds'],rel_tol=1e-8,abs_tol=1e-6):
        raise ValueError('Shared-head mixed audio differs from TR plus replay')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_shared_head_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_shared_head_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_shared_head_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_shared_head_train_wall_seconds':wall,'data/post_shared_head_completed_updates':n,
        'data/post_shared_head_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_shared_head_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_shared_head_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    return values


def uniform_full_context(config,history):
    """Source6500 counters anchor the entire new pass across exact windows."""
    if config.get('phase')!=UNIFORM_FULL_PHASE:return None
    full=config.get('uniform_full_pass',{});identity=config.get('config_identity',{})
    recipe=identity.get('training_arguments',{});declared=identity.get('uniform_full_transition',{}).get('declared',{})
    recovery=config.get('Portuguese_recovery',{});initial=config.get('initial_source_equivalence',{})
    step=full.get('source_step');pool=full.get('expected_new_unique_TR_clips');updates=full.get('additional_update_budget')
    if (type(step) is not int or step!=6500 or config.get('source_step')!=step or config.get('inherited_history_steps')!=step
        or type(pool) is not int or pool!=253798 or type(updates) is not int or updates!=31725
        or full.get('end_step')!=step+updates or config.get('steps')!=step+updates
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or declared.get('all207_weights_and_Adam_states_retained') is not True or declared.get('fresh_optimizer_or_warmup') is not False
        or declared.get('source_selected_step')!=step or declared.get('full_TR_clips')!=pool
        or declared.get('new_uniform_pass_updates')!=updates or recipe.get('sampling_mode')!='uniform'
        or recipe.get('head_lr')!=1e-5 or recipe.get('lora_lr')!=3e-4 or recipe.get('new_token_lr_multiplier')!=6
        or recovery.get('grace_new_updates')!=1000 or recovery.get('all_six_required_for_every_eligible_BEST') is not True
        or recovery.get('source_6500_all_six_qualified') is not False
        or initial.get('passed') is not True or initial.get('actual_GPU_decode') is not True
        or initial.get('source_selected_step')!=step or initial.get('before_first_uniform_update') is not True
        or initial.get('compared_fixed_DEV_utterances')!=300 or initial.get('Portuguese_fixed_DEV_measured_utterances')!=50
        or initial.get('all_five_original_retention_gates_passed') is not True
        or initial.get('all_six_retention_gates_passed') is not False
        or set(config.get('evaluation_languages',[]))!={'tr','en','es','it','fr','de','pt'}):
        raise ValueError('Declared full253798 original207 uniform pass/PT recovery source identity required')
    baseline=next((row for row in history if row.get('step')==step),None)
    if baseline is None:raise ValueError('Actual inherited6500 throughput baseline missing')
    context={'step':step,'pool':pool,'updates':updates,'head_lr':recipe['head_lr'],'lora_lr':recipe['lora_lr'],
        'Portuguese_grace':recovery['grace_new_updates'],'initial_all_six_passed':False}
    source_errors,allowed=initial.get('Portuguese_source_word_errors'),initial.get('Portuguese_allowed_word_errors')
    if type(source_errors) is not int or type(allowed) is not int or not source_errors>allowed>=0:
        raise ValueError('Actual measured source Portuguese failure counts required')
    context.update(source_Portuguese_word_errors=source_errors,Portuguese_allowed_word_errors=allowed)
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:raise ValueError('Finite actual source6500 cumulative counters required')
        context[key]=value
    historical=baseline.get('turkish_unique_clips_seen')
    if type(historical) is not int or not 0<=historical<=pool:raise ValueError('Actual source6500 unique TR count required')
    context['historical_unique']=historical
    if not math.isclose(context['audio_seen_seconds'],context['turkish_audio_seen_seconds']+context['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Source mixed audio disagrees with TR plus replay')
    return context


def uniform_full_values_for(record,context):
    if context is None or record['step']<context['step']:return {}
    n=record['step']-context['step'];pool=context['pool']
    if not 0<=n<=context['updates']:raise ValueError('Uniform localN beyond approved fullpass')
    values={'phase/uniform_full_updates':n,'phase/source_step':context['step'],
        'phase/expected_full207_Adam_clock':context['step']+n,'phase/active_Adam_parameter_tensors':207,
        'phase/Adam_clock_derived_from_completed_worker_updates':1,'phase/lora_active':1,
        'data/uniform_full_TR_pool_clips':pool,'data/historical_TR_unique_clips_at_source':context['historical_unique'],
        'retention/source_Portuguese_word_errors':context['source_Portuguese_word_errors'],
        'retention/Portuguese_allowed_word_errors':context['Portuguese_allowed_word_errors'],
        'retention/Portuguese_recovery_grace_remaining_updates':max(0,context['Portuguese_grace']-n)}
    if n==0:
        values.update({'data/uniform_local_TR_draws':0,'data/uniform_local_unique_TR_clips':0,
            'data/uniform_local_fraction_unique_seen':0.,'data/uniform_local_epoch':0.,'data/uniform_full_pass_complete':0})
        return values
    local=record.get('Turkish_domain_coverage',{}).get('uniform_pass',{})
    unique=local.get('unique_clips_seen');cumulative=record.get('turkish_unique_clips_seen')
    if (record.get('uniform_full_updates')!=n or record.get('head_lr')!=context['head_lr']
        or record.get('lora_lr')!=context['lora_lr'] or record.get('new_token_update_multiplier')!=6
        or local.get('source_step')!=context['step'] or local.get('local_updates')!=n or local.get('pool_clips')!=pool
        or local.get('draws')!=8*n or type(unique) is not int or unique!=min(8*n,pool)
        or local.get('repeat_visits')!=8*n-unique or local.get('fraction_unique_seen')!=unique/pool
        or local.get('epoch_clock')!=8*n/pool or local.get('full_pass_complete')!=(unique==pool)
        or type(cumulative) is not int or not max(unique,context['historical_unique'])<=cumulative<=min(pool,unique+context['historical_unique'])
        or record.get('turkish_clips_seen')!=(context['step']+n)*8):
        raise ValueError('Actual local uniform coverage/LR/cumulative union counters disagree')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=record.get(key)
        if not finite_number(value):raise ValueError('Finite uniform cumulative counters required')
        delta[key]=value-context[key]
        if delta[key]<=0:raise ValueError('Uniform throughput counters must increase beyond6500')
    if not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Post-uniform mixed audio differs from TR plus six-language replay')
    wall=delta['train_wall_seconds']
    values.update({'phase/head_lr':record['head_lr'],'phase/lora_lr':record['lora_lr'],
        'data/uniform_local_TR_draws':local['draws'],'data/uniform_local_unique_TR_clips':unique,
        'data/uniform_local_fraction_unique_seen':local['fraction_unique_seen'],'data/uniform_local_epoch':local['epoch_clock'],
        'data/uniform_local_repeat_visits':local['repeat_visits'],'data/uniform_full_pass_complete':int(local['full_pass_complete']),
        'data/cumulative_TR_unique_union':cumulative,'data/cumulative_TR_draws_with_inherited_history':record['turkish_clips_seen'],
        'resources/post_uniform_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_uniform_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_uniform_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_uniform_train_wall_seconds':wall,
        'data/post_uniform_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_uniform_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_uniform_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    gate=record.get('validation')
    if gate is not None:
        if (type(gate.get('retention_gate_passed')) is not bool or type(gate.get('old_five_retention_gate_passed')) is not bool
            or gate.get('Portuguese_grace_active_for_stop_only')!=(n<context['Portuguese_grace'])
            or type(gate.get('stop_retention_gate_passed')) is not bool):
            raise ValueError('Actual strict all6/old5/grace/stop gate fields required')
        values.update({'retention/all_six_passed':int(gate['retention_gate_passed']),
            'retention/original_five_passed':int(gate['old_five_retention_gate_passed']),
            'retention/Portuguese_stop_grace_active':int(gate['Portuguese_grace_active_for_stop_only']),
            'retention/stop_guard_passed':int(gate['stop_retention_gate_passed'])})
    return values


def full_phase_baseline_steps(config):
    result=baseline_steps(config)
    if config.get('phase')==UNIFORM_FULL_PHASE:result.add(config['source_step'])
    return result


def full_phase_initial_development_step(config):
    return config['source_step'] if config.get('phase')==UNIFORM_FULL_PHASE else initial_development_step(config)


def full_phase_actual_development_row(config,record):
    if config.get('phase')==UNIFORM_FULL_PHASE:
        return ('validation' in record and record.get('uniform_full_updates')==record['step']-config['source_step']
            and record['step']>config['source_step'])
    return actual_development_row(config,record)


def full_phase_contexts_for(config,baseline_rows):
    result=contexts_for(config,baseline_rows);result['uniform_full']=uniform_full_context(config,baseline_rows)
    return result


def full_phase_value_for_record(record,contexts,development):
    values=value_for_record(record,contexts,development);context=contexts.get('uniform_full')
    if context is None:return values
    # Historical metrics are labelled inherited; their old5 DEV/gates cannot
    # masquerade as a new350/all6 evaluation, including the source6500 row.
    if record['step']<=context['step']:
        values={key:value for key,value in values.items() if not key.startswith(('retention/','dev/'))}
        values['phase/inherited_history']=int(record['step']<context['step'])
        actual=development.get(record['step'],{}) if record['step']==context['step'] else {}
        if set(actual)=={'dev/wer_'+language+'_percent' for language in ('tr','en','es','it','fr','de','pt')}:
            values.update(actual);values['retention/all_dev_gates_passed']=0;values['retention/all_six_passed']=0
            values['retention/original_five_passed']=1;values['retention/Portuguese_stop_grace_active']=1
    values.update(uniform_full_values_for(record,context));return values


def uniform_full_selection_values(config,selected,last_step):
    if config.get('phase')!=UNIFORM_FULL_PHASE or selected.get('run_id')!=config.get('run_id'):return {}
    n=selected.get('uniform_full_updates');step=selected.get('completed_steps');chosen=selected.get('selected_step')
    if (type(n) is not int or not 0<=n<=31725 or step!=6500+n or step>last_step
        or type(selected.get('accepted')) is not bool or selected.get('source_step')!=6500
        or type(selected.get('full_new_uniform_pass_complete')) is not bool
        or selected['full_new_uniform_pass_complete']!=(n==31725)):
        raise ValueError('Published fulluniform selection frontier differs')
    result={'selection/all_six_eligible_best_available':int(selected['accepted']),
        'selection/published_uniform_updates':n,'selection/published_full_pass_complete':int(selected['full_new_uniform_pass_complete'])}
    if selected['accepted']:
        if type(chosen) is not int or not 6500<chosen<=step or not finite_number(selected.get('turkish_validation_wer')):
            raise ValueError('All6 BEST must be an actual new evaluated checkpoint')
        result.update({'selection/best_step':chosen,'selection/best_qualifying_TR_wer_percent':100*selected['turkish_validation_wer']})
    elif chosen is not None or selected.get('turkish_validation_wer') is not None:
        raise ValueError('No all6 BEST must retain None; never substitute old-five6500')
    return result


def rejected_run_terminal(config,selected):
    if config.get('phase')!=UNIFORM_FULL_PHASE:return True
    # A budget-only checkpoint with no all6 winner still continues SAME-ID.
    return (selected.get('full_new_uniform_pass_complete') is True
        or selected.get('stop_early_reason') in ('consecutive_retention_gate_failures','development_plateau','development_target_met'))



def uniform_majority_context(config,history):
    if config.get('phase')!=UNIFORM_MAJORITY_PHASE:return None
    h=config.get('source_step');origin=config.get('uniform_origin_step');full=config.get('uniform_full_pass',{})
    identity=config.get('config_identity',{});recipe=identity.get('training_arguments',{})
    declared=identity.get('majority_transition',{}).get('declared',{});proof=config.get('initial_source_equivalence',{})
    prior=full.get('prior_uniform_draws');pool=full.get('expected_new_unique_TR_clips');budget=full.get('additional_update_budget')
    if (type(h) is not int or h<=6500 or origin!=6500 or pool!=253798 or prior!=8*(h-origin)
        or not 0<prior<pool or budget!=math.ceil((pool-prior)/14) or config.get('steps')!=h+budget
        or full.get('source_step')!=origin or full.get('transition_source_step')!=h or full.get('end_step')!=h+budget
        or config.get('inherited_history_steps')!=h or config.get('trainable_parameter_tensors')!=207
        or config.get('replay_clips_used')!=1914 or declared.get('source_step')!=h
        or declared.get('prior_uniform_draws')!=prior or declared.get('TR_per_update')!=14 or declared.get('replay_per_update')!=2
        or declared.get('configured_end_step')!=h+budget or declared.get('renewed_Portuguese_grace') is not False
        or declared.get('preservation_coefficients')!={'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.}
        or recipe.get('replay_fraction')!=.125 or recipe.get('replay_kl')!=40 or recipe.get('duration_kl')!=2 or recipe.get('feature_distillation')!=4
        or recipe.get('head_lr')!=1e-5 or recipe.get('lora_lr')!=3e-4 or recipe.get('new_token_lr_multiplier')!=6
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True or proof.get('source_selected_step')!=h
        or proof.get('before_first_majority_update') is not True or proof.get('compared_fixed_DEV_utterances')!=350
        or proof.get('all_seven_ASR_counts_exact') is not True):
        raise ValueError('Declared original207/sourceH same-pass14/2 and actual source350 identity required')
    base=next((row for row in history if row.get('step')==h),None)
    original=next((row for row in history if row.get('step')==origin),None)
    if base is None or original is None:raise ValueError('SourceH rates and original6500 coverage baselines required')
    local=base.get('Turkish_domain_coverage',{}).get('uniform_pass',{})
    if local.get('draws')!=prior or local.get('unique_clips_seen')!=prior or local.get('source_step')!=origin:
        raise ValueError('Actual closed source queue frontier differs')
    context={'step':h,'origin':origin,'prior':prior,'pool':pool,'budget':budget,'head_lr':recipe['head_lr'],'lora_lr':recipe['lora_lr'],
        'historical_unique':original.get('turkish_unique_clips_seen'),'source_TR_draws':base.get('turkish_clips_seen'),
        'source_RP_draws':base.get('replay_clips_seen'),'source_gates':proof['source_retention_gates'],
        'source_all_six_passed':proof['all_six_retention_gates_passed']}
    if type(context['historical_unique']) is not int or type(context['source_TR_draws']) is not int or type(context['source_RP_draws']) is not int:
        raise ValueError('Actual historical union/source draw counters required')
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        if not finite_number(base.get(key)):raise ValueError('Finite actual H counter required: '+key)
        context[key]=base[key]
    return context


def uniform_majority_values_for(record,context):
    if context is None or record['step']<context['step']:return {}
    n=record['step']-context['step'];full_n=record['step']-context['origin'];pool=context['pool']
    if not 0<=n<=context['budget']:raise ValueError('Majority localM beyond same-pass remaining bound')
    values={'phase/majority_updates':n,'phase/majority_source_step':context['step'],'phase/uniform_origin_step':context['origin'],
        'phase/uniform_full_updates':full_n,'phase/expected_full207_Adam_clock':record['step'],'phase/active_Adam_parameter_tensors':207,
        'phase/head_lr':context['head_lr'],'phase/lora_lr':context['lora_lr'],'phase/declared_TR_per_update':14,'phase/declared_RP_per_update':2,
        'phase/declared_replay_kl_coefficient':40.,'phase/declared_duration_kl_coefficient':2.,'phase/declared_feature_distillation_coefficient':4.,
        'phase/expected_RP_replay_KL_mean_density':5.,'phase/expected_RP_duration_KL_mean_density':.25,'phase/expected_RP_feature_mean_density':.5,
        'data/uniform_full_TR_pool_clips':pool,'data/historical_TR_unique_clips_at_6500':context['historical_unique'],
        'data/remaining_uniform_updates_at_H':context['budget'],
        'retention/source_H_Portuguese_word_errors':context['source_gates']['pt']['word_errors'],
        'retention/Portuguese_allowed_word_errors':context['source_gates']['pt']['allowed_word_errors'],
        'retention/source_H_all_six_qualified':int(context['source_all_six_passed']),
        'retention/Portuguese_recovery_grace_remaining_updates':max(0,1000-full_n)}
    if n==0:
        values.update({'data/uniform_local_TR_draws':context['prior'],'data/uniform_local_unique_TR_clips':context['prior'],
            'data/uniform_local_fraction_unique_seen':context['prior']/pool,'data/uniform_local_epoch':context['prior']/pool,
            'data/uniform_local_repeat_visits':0,'data/uniform_full_pass_complete':0})
        return values
    local=record.get('Turkish_domain_coverage',{}).get('uniform_pass',{})
    transition=record.get('Turkish_domain_coverage',{}).get('majority_transition',{})
    draws=context['prior']+14*n;unique=min(draws,pool);union=record.get('turkish_unique_clips_seen')
    expected_segments=[{'first_step':6501,'last_step':context['step'],'updates':context['step']-6500,'TR_per_update':8,'replay_per_update':8,
        'TR_draws':context['prior'],'replay_draws':context['prior']},
        {'first_step':context['step']+1,'last_step':record['step'],'updates':n,'TR_per_update':14,'replay_per_update':2,'TR_draws':14*n,'replay_draws':2*n}]
    if (record.get('majority_updates')!=n or record.get('majority_source_step')!=context['step'] or record.get('uniform_origin_step')!=6500
        or record.get('uniform_full_updates')!=full_n or record.get('head_lr')!=context['head_lr'] or record.get('lora_lr')!=context['lora_lr']
        or record.get('new_token_update_multiplier')!=6 or record.get('replay_kl_coefficient')!=40
        or record.get('duration_kl_coefficient')!=2 or record.get('feature_distillation_coefficient')!=4
        or local.get('source_step')!=6500 or local.get('local_updates')!=full_n or local.get('pool_clips')!=pool
        or local.get('draws')!=draws or local.get('unique_clips_seen')!=unique or local.get('repeat_visits')!=draws-unique
        or local.get('fraction_unique_seen')!=unique/pool or local.get('epoch_clock')!=draws/pool
        or local.get('full_pass_complete')!=(unique==pool) or local.get('quota_segments')!=expected_segments
        or transition.get('local_updates')!=n or transition.get('source_step')!=context['step']
        or record.get('turkish_clips_seen')!=context['source_TR_draws']+14*n or record.get('replay_clips_seen')!=context['source_RP_draws']+2*n
        or type(union) is not int or not max(unique,context['historical_unique'])<=union<=min(pool,unique+context['historical_unique'])):
        raise ValueError('Actual14/2 piecewise coverage/LR/density/global counters disagree')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        if not finite_number(record.get(key)):raise ValueError('Finite majority counter required')
        delta[key]=record[key]-context[key]
        if delta[key]<=0:raise ValueError('PostH mixed/TR/replay/complete-update counters must increase')
    if not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('PostH audio is not TR plus replay')
    wall=delta['train_wall_seconds']
    values.update({'data/uniform_local_TR_draws':draws,'data/uniform_local_unique_TR_clips':unique,
        'data/uniform_local_fraction_unique_seen':unique/pool,'data/uniform_local_epoch':draws/pool,'data/uniform_local_repeat_visits':draws-unique,
        'data/uniform_full_pass_complete':int(unique==pool),'data/cumulative_TR_unique_union':union,
        'data/majority_TR_draws':14*n,'data/majority_RP_draws':2*n,
        'resources/post_majority_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_majority_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_majority_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_majority_train_wall_seconds':wall,
        'data/post_majority_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_majority_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_majority_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    gate=record.get('validation')
    if gate is not None:
        if gate.get('Portuguese_grace_active_for_stop_only')!=(full_n<1000):raise ValueError('Portuguese grace must remain anchored6500')
        values.update({'retention/all_six_passed':int(gate['retention_gate_passed']),
            'retention/original_five_passed':int(gate['old_five_retention_gate_passed']),
            'retention/Portuguese_stop_grace_active':int(gate['Portuguese_grace_active_for_stop_only']),
            'retention/stop_guard_passed':int(gate['stop_retention_gate_passed'])})
    return values


def majority_phase_baseline_steps(config):
    if config.get('phase')==UNIFORM_MAJORITY_PHASE:return {6500,config['source_step']}
    return full_phase_baseline_steps(config)


def majority_phase_initial_development_step(config):
    return config['source_step'] if config.get('phase')==UNIFORM_MAJORITY_PHASE else full_phase_initial_development_step(config)


def majority_phase_actual_development_row(config,record):
    if config.get('phase')==UNIFORM_MAJORITY_PHASE:
        return 'validation' in record and record['step']>config['source_step'] and record.get('majority_updates')==record['step']-config['source_step']
    return full_phase_actual_development_row(config,record)


def majority_phase_contexts_for(config,baseline_rows):
    contexts=full_phase_contexts_for(config,baseline_rows)
    contexts['majority']=uniform_majority_context(config,baseline_rows)
    return contexts


def majority_phase_value_for_record(record,contexts,development):
    values=full_phase_value_for_record(record,contexts,development);context=contexts.get('majority')
    if context is None:return values
    if record['step']<=context['step']:
        values={key:value for key,value in values.items() if not key.startswith(('retention/','dev/'))}
        values['phase/inherited_history']=int(record['step']<context['step'])
        actual=development.get(record['step'],{}) if record['step']==context['step'] else {}
        if set(actual)=={'dev/wer_'+language+'_percent' for language in ('tr','en','es','it','fr','de','pt')}:
            values.update(actual);values['retention/all_dev_gates_passed']=int(context['source_all_six_passed'])
            values['retention/all_six_passed']=int(context['source_all_six_passed'])
            values['retention/original_five_passed']=int(all(context['source_gates'][language]['passed'] for language in ('en','es','it','fr','de')))
            values['retention/Portuguese_stop_grace_active']=int(context['step']-6500<1000)
    values.update(uniform_majority_values_for(record,context));return values


def majority_selection_values(config,selected,last_step):
    if config.get('phase')!=UNIFORM_MAJORITY_PHASE:return uniform_full_selection_values(config,selected,last_step)
    if selected.get('run_id')!=config.get('run_id'):return {}
    h=config['source_step'];step=selected.get('completed_steps');n=selected.get('majority_updates');chosen=selected.get('selected_step')
    if (type(n) is not int or not 0<=n<=config['remaining_uniform_pass_updates'] or step!=h+n or step>last_step
        or selected.get('uniform_origin_step')!=6500 or selected.get('uniform_full_updates')!=step-6500
        or type(selected.get('accepted')) is not bool or type(selected.get('full_new_uniform_pass_complete')) is not bool
        or selected['full_new_uniform_pass_complete']!=(n==config['remaining_uniform_pass_updates'])):
        raise ValueError('Actual majority selection frontier differs')
    result={'selection/all_six_eligible_best_available':int(selected['accepted']),'selection/published_majority_updates':n,
        'selection/published_uniform_updates':step-6500,'selection/published_full_pass_complete':int(selected['full_new_uniform_pass_complete'])}
    if selected['accepted']:
        if type(chosen) is not int or not 6500<chosen<=step or not finite_number(selected.get('turkish_validation_wer')):raise ValueError('Eligible all6 selected checkpoint required')
        result.update({'selection/best_step':chosen,'selection/best_qualifying_TR_wer_percent':100*selected['turkish_validation_wer'],
            'selection/eligible_BEST_inherited_at_or_before_H':int(chosen<=h)})
    elif chosen is not None or selected.get('turkish_validation_wer') is not None:raise ValueError('No eligible all6 BEST must remain None')
    return result


def majority_rejected_run_terminal(config,selected):
    if config.get('phase')!=UNIFORM_MAJORITY_PHASE:return rejected_run_terminal(config,selected)
    return selected.get('full_new_uniform_pass_complete') is True or selected.get('stop_early_reason') in ('consecutive_retention_gate_failures','development_plateau','development_target_met')


def _mirror_config_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _checked_parent_json(item):
    path=Path(item.get('path','')).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path.suffix!='.json':raise ValueError('Only hash-bound workspace parent JSON is admitted')
    actual=file_record(path)
    if any(actual[key]!=item.get(key) for key in ('path','sha256','bytes')):raise ValueError('Immutable lowerLR source JSON changed')
    value=read_json(path,None)
    if type(value) is not dict:raise ValueError('Parent source record must be an object')
    return value


def lower_lora_context(config,history):
    if config.get('phase')!=LOWER_LORA_PHASE:return None
    identity=config.get('config_identity',{});transition=identity.get('encoder_lr_transition',{})
    declared=transition.get('declared',{});recipe=identity.get('training_arguments',{})
    h=config.get('source_step');quota=config.get('majority_source_step');proof=config.get('initial_source_equivalence',{})
    if (h!=7750 or quota!=7621 or config.get('inherited_history_steps')!=h or config.get('uniform_origin_step')!=6500
        or transition.get('kind')!='TSC_uniform_majority_LoRA_LR_state_transition_v1'
        or declared!=config.get('encoder_learning_rate_transition') or declared.get('source_step')!=h
        or declared.get('quota_boundary_step')!=quota or declared.get('uniform_origin_step')!=6500
        or declared.get('source_LoRA_lr')!=3e-4 or declared.get('active_LoRA_lr')!=1e-4 or declared.get('head_lr')!=1e-5
        or declared.get('all207_Adam_moments_and_global_clocks_preserved') is not True
        or declared.get('fresh_optimizer_or_sampler') is not False or declared.get('renewed_Portuguese_grace') is not False
        or declared.get('parent_failure_counter_preserved') is not True or declared.get('all_six_required_for_every_BEST') is not True
        or declared.get('only_new_scientific_variable')!='encoder_LoRA_group_LR'
        or declared.get('preservation_coefficients')!=[40,2,4] or declared.get('TR_per_update')!=14 or declared.get('replay_per_update')!=2
        or recipe.get('head_lr')!=1e-5 or recipe.get('lora_lr')!=1e-4
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True or proof.get('source_selected_step')!=h
        or proof.get('compared_fixed_DEV_utterances')!=350 or proof.get('before_first_majority_update') is not True
        or proof.get('all_raw_transcripts_and_emitted_paths_exact') is not True or proof.get('all_seven_ASR_counts_exact') is not True
        or proof.get('source_receipt')!=transition.get('source_receipt')):
        raise ValueError('Declared7750 full207 lowerLR fork and actual source350 required')
    receipt=_checked_parent_json(transition['source_receipt'])
    parent=_checked_parent_json(receipt['artifacts']['run_config'])
    if (receipt.get('chosen_step')!=h or receipt.get('completed_steps')!=h
        or _mirror_config_sha(parent['config_identity'])!=transition.get('parent_config_identity_sha256')
        or parent.get('phase')!=UNIFORM_MAJORITY_PHASE or parent.get('source_step')!=quota
        or recipe!={**parent['config_identity']['training_arguments'],'lora_lr':1e-4}
        or identity.get('domain_sampling_identity')!=parent['config_identity'].get('domain_sampling_identity')
        or any(identity.get(key)!=parent['config_identity'].get(key) for key in ('tr_pool_sha256','replay_pool_sha256','extended_tokenizer_sha256'))
        or config.get('steps')!=parent.get('steps') or declared.get('configured_end_step')!=config.get('steps')
        or config.get('remaining_uniform_pass_updates')!=config['steps']-h
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or config.get('uniform_full_pass')!=parent.get('uniform_full_pass')
        or config.get('Portuguese_recovery')!=parent.get('Portuguese_recovery')
        or required_eval_languages(config)!=required_eval_languages(parent)):
        raise ValueError('Original quota/data/full horizon or only-LR source comparison changed')
    majority=uniform_majority_context(parent,history)
    baseline=next((record for record in history if record.get('step')==h),None)
    if baseline is None or baseline.get('majority_updates')!=h-quota or baseline.get('lora_lr')!=3e-4:
        raise ValueError('Actual inherited7750 source counter row required')
    uniform_majority_values_for(baseline,majority)
    context={'step':h,'quota_step':quota,'budget':config['steps']-h,'majority':majority,
        'source_gates':proof['source_retention_gates'],'source_all_six_passed':proof['all_six_retention_gates_passed']}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        value=baseline.get(key)
        if not finite_number(value) or value<0:raise ValueError('Finite actual7750 rate baseline required')
        context[key]=value
    if not math.isclose(context['audio_seen_seconds'],context['turkish_audio_seen_seconds']+context['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Actual7750 mixed audio differs from TR plus replay')
    return context


def loralr_phase_baseline_steps(config):
    return {6500,config['majority_source_step'],config['source_step']} if config.get('phase')==LOWER_LORA_PHASE else majority_phase_baseline_steps(config)


def loralr_phase_initial_development_step(config):
    return config['source_step'] if config.get('phase')==LOWER_LORA_PHASE else majority_phase_initial_development_step(config)


def loralr_phase_actual_development_row(config,record):
    if config.get('phase')==LOWER_LORA_PHASE:
        return 'validation' in record and record['step']>config['source_step'] and record.get('lora_lr_fork_updates')==record['step']-config['source_step']
    return majority_phase_actual_development_row(config,record)


def loralr_phase_contexts_for(config,baseline_rows):
    if config.get('phase')!=LOWER_LORA_PHASE:return majority_phase_contexts_for(config,baseline_rows)
    context=lower_lora_context(config,baseline_rows);contexts=full_phase_contexts_for(config,baseline_rows)
    contexts.update(majority=context['majority'],lower_lora=context)
    return contexts


def loralr_phase_value_for_record(record,contexts,development):
    context=contexts.get('lower_lora')
    if context is None:return majority_phase_value_for_record(record,contexts,development)
    h=context['step'];n=record['step']-h
    # Reuse unchanged14/2 validation/math; change only its expected LR for this
    # already independently validated new phase. No raw history is rewritten.
    checked=record
    if n>0:
        if (n>context['budget'] or record.get('lora_lr_fork_updates')!=n or record.get('lora_lr_fork_source_step')!=h
            or record.get('head_lr')!=1e-5 or record.get('lora_lr')!=1e-4):raise ValueError('Actual lowerLR local/global clocks or LR differ')
        checked={**record,'lora_lr':context['majority']['lora_lr']}
    values=majority_phase_value_for_record(checked,contexts,development)
    if record['step']<=h:
        values={key:value for key,value in values.items() if not key.startswith(('retention/','dev/'))}
        values['phase/inherited_history']=int(record['step']<h)
        actual=development.get(h,{}) if record['step']==h else {}
        if set(actual)=={'dev/wer_'+language+'_percent' for language in ('tr','en','es','it','fr','de','pt')}:
            values.update(actual);values['retention/all_dev_gates_passed']=int(context['source_all_six_passed'])
            values['retention/all_six_passed']=int(context['source_all_six_passed'])
            values['retention/original_five_passed']=int(all(context['source_gates'][lang]['passed'] for lang in ('en','es','it','fr','de')))
            values['retention/Portuguese_stop_grace_active']=0
    if n<0:return values
    values.update({'phase/lora_lr_fork_updates':n,'phase/lora_lr_fork_source_step':h,
        'phase/lora_lr_fork_source_group_lr':3e-4,'phase/lora_lr_fork_declared_group_lr':1e-4,
        'phase/expected_full207_Adam_clock':record['step'],'phase/Adam_clock_derived_from_completed_worker_updates':1})
    if n==0:return values
    values['phase/lora_lr']=record['lora_lr']
    delta={key:record[key]-context[key] for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds')}
    if any(not finite_number(value) or value<=0 for value in delta.values()) or not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Actual post7750 complete-update rate counters differ')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_lower_lora_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_lower_lora_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_lower_lora_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_lower_lora_train_wall_seconds':wall,
        'data/post_lower_lora_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_lower_lora_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_lower_lora_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    return values


def loralr_selection_values(config,selected,last_step):
    if config.get('phase')!=LOWER_LORA_PHASE:return majority_selection_values(config,selected,last_step)
    if selected.get('run_id')!=config.get('run_id'):return {}
    h=config['source_step'];quota=config['majority_source_step'];step=selected.get('completed_steps');n=selected.get('lora_lr_fork_updates')
    if (type(n) is not int or not 0<=n<=config['remaining_uniform_pass_updates'] or step!=h+n or step>last_step
        or selected.get('source_step')!=h or selected.get('majority_source_step')!=quota):raise ValueError('Published lowerLR source/local frontier differs')
    majority_config={**config,'phase':UNIFORM_MAJORITY_PHASE,'source_step':quota,'remaining_uniform_pass_updates':config['steps']-quota}
    values=majority_selection_values(majority_config,selected,last_step)
    values.update({'selection/published_lora_lr_fork_updates':n,'selection/lora_lr_fork_source_step':h})
    if selected['accepted']:
        values['selection/eligible_BEST_inherited_at_or_before_quota_H']=int(selected['selected_step']<=quota)
        values['selection/eligible_BEST_inherited_at_or_before_H']=int(selected['selected_step']<=h)
    return values


def loralr_rejected_run_terminal(config,selected):
    if config.get('phase')!=LOWER_LORA_PHASE:return majority_rejected_run_terminal(config,selected)
    return selected.get('full_new_uniform_pass_complete') is True or selected.get('stop_early_reason') in ('consecutive_retention_gate_failures','development_plateau','development_target_met')


def file_record(path):
    path=Path(path).resolve();before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('File changed while hashing')
    return {'path':str(path),'sha256':digest.hexdigest(),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}


def self_memory(phase):
    # Own PID only. No cross-job attribution or permission workaround.
    try:
        from resource_watchdog_v2 import sample_memory
        return {'phase':phase,'measurement':sample_memory(os.getpid(),phase)}
    except Exception as error:
        return {'phase':phase,'unavailable':type(error).__name__}


class TopLevelRuns:
    'Bounded-read JSON object parser; materializes only one complete run value.'
    def __init__(self,stream,chunk_size=65536):
        self.stream=stream;self.chunk_size=chunk_size;self.buffer='';self.pos=0
    def peek(self):
        if self.pos==len(self.buffer):
            self.buffer=self.stream.read(self.chunk_size);self.pos=0
        return self.buffer[self.pos:self.pos+1]
    def take(self):
        value=self.peek()
        if value:self.pos+=1
        return value
    def whitespace(self):
        while self.peek() and self.peek() in ' \t\r\n':self.take()
    def key(self):
        if self.take()!='"':raise ValueError('Expected JSON run key')
        pieces=['"'];escaped=False
        while True:
            char=self.take()
            if not char:raise ValueError('Truncated run key')
            pieces.append(char)
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':break
        value=json.loads(''.join(pieces))
        if type(value) is not str:raise ValueError('Invalid run key')
        return value
    def value(self):
        self.whitespace()
        if self.peek()!='{':raise ValueError('Every historical run must be a JSON object')
        depth=0;quoted=False;pieces=[];record_chars=0;bound=getattr(self,'max_record_chars',None)
        while True:
            part=self.buffer[self.pos:]
            if not part:raise ValueError('Truncated run object')
            # Carry a final slash to the next block, preserving escaped quotes.
            trailing_slashes=len(part)-len(part.rstrip('\\'))
            carry='\\' if trailing_slashes%2 else ''
            scan=part[:-1] if carry else part
            for match in re.finditer(r'\\.|[{}\[\]"]',scan):
                token=match.group()
                if token.startswith('\\'):continue
                if token=='"':quoted=not quoted
                elif not quoted:
                    depth+=1 if token in '{[' else -1
                    if depth<0:raise ValueError('Invalid run object depth')
                    if depth==0:
                        record_chars+=match.end()
                        if bound is not None and record_chars>bound:raise ValueError('Oversized individual history record')
                        pieces.append(scan[:match.end()]);self.pos+=match.end()
                        value=json.loads(''.join(pieces),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
                        if type(value) is not dict:raise ValueError('Historical run must be an object')
                        return value
            record_chars+=len(scan)
            if bound is not None and record_chars>bound:raise ValueError('Oversized individual history record')
            pieces.append(scan)
            self.buffer=carry+self.stream.read(self.chunk_size);self.pos=0
            if not self.buffer or self.buffer==carry:raise ValueError('Truncated run object')
    def runs(self):
        self.whitespace()
        if self.take()!='{':raise ValueError('V8 source must be a top-level run object')
        self.whitespace();seen=set()
        if self.peek()=='}':self.take()
        else:
            while True:
                self.whitespace();key=self.key()
                if key in seen:raise ValueError('Duplicate historical run key')
                seen.add(key);self.whitespace()
                if self.take()!=':':raise ValueError('Expected run separator')
                yield key,self.value()
                self.whitespace();separator=self.take()
                if separator=='}':break
                if separator!=',':raise ValueError('Truncated/invalid top-level run source')
        self.whitespace()
        if self.peek():raise ValueError('Trailing source content')


class RunShardStore:
    'Small index + immutable content shards; at most one active run map.'
    def __init__(self,index_path):
        self.path=Path(index_path).resolve();self.directory=self.path.parent/(self.path.stem+'_shards')
        self.active_id=None;self.active=None;self.dirty=False
        self.index=read_json(self.path,None) if self.path.exists() else None
        if self.path.exists() and (not isinstance(self.index,dict) or self.index.get('kind')!='Trackio_run_shard_index_v9'
                or self.index.get('schema')!=1 or type(self.index.get('runs')) is not dict
                or self.index.get('source_import',{}).get('complete') is not True):
            raise ValueError('Existing V9 index is invalid/incomplete; never silently reimport')
        if self.index is not None:
            for key,entry in self.index['runs'].items():
                if type(key) is not str or type(entry) is not dict:raise ValueError('Invalid run index')
                self.shard_path(entry['shard'])
    def shard_path(self,record):
        sha=record.get('sha256')
        if type(sha) is not str or re.fullmatch('[0-9a-f]{64}',sha) is None:
            raise ValueError('Invalid immutable shard SHA')
        path=(self.directory/(sha+'.json')).resolve()
        if Path(record.get('path','')).resolve()!=path or type(record.get('bytes')) is not int or record['bytes']<2:
            raise ValueError('Shard location/size differs from its SHA filename')
        return path
    def write_shard(self,value):
        if type(value) is not dict:raise ValueError('Run state must remain an object')
        self.directory.mkdir(parents=True,exist_ok=True)
        temporary=self.directory/f'.{os.getpid()}_{time.time_ns()}.tmp'
        digest=hashlib.sha256();count=0
        try:
            with temporary.open('xb') as stream:
                for part in json.JSONEncoder(sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).iterencode(value):
                    block=part.encode('utf-8');stream.write(block);digest.update(block);count+=len(block)
                stream.flush();os.fsync(stream.fileno())
            final=self.directory/(digest.hexdigest()+'.json')
            if final.exists():
                record=file_record(final)
                if record['sha256']!=digest.hexdigest() or record['bytes']!=count:
                    raise ValueError('Existing immutable content shard is corrupt')
                temporary.unlink()
            else:os.replace(temporary,final)
            return {'path':str(final.resolve()),'sha256':digest.hexdigest(),'bytes':count}
        finally:
            if temporary.exists():temporary.unlink()
    def migrate(self,source_path,*,source_closed_confirmed=False,source_closure_receipt=None):
        if self.index is not None:return None
        if not source_closed_confirmed:
            raise ValueError('ROOT must observe V8 CLOSED and pass --source-closed-confirmed before migration')
        source_path=Path(source_path).resolve()
        if source_path==self.path:raise ValueError('V8 source and V9 index must be distinct')
        if not source_path.exists():raise ValueError('Complete preserved V8 source is required')
        # Bind the frozen producer; its large state is read only at ROOT handoff.
        producer=Path(__file__).with_name('track_continuation_v8.py')
        if file_record(producer)['sha256']!=SOURCE_V8_SHA:raise ValueError('Frozen V8 producer source changed')
        samples=[self_memory('v9_before_streaming_migration')]
        before=file_record(source_path);entries={};largest=0
        with source_path.open('r',encoding='utf-8') as stream:
            for run_id,value in TopLevelRuns(stream).runs():
                record=self.write_shard(value)
                entries[run_id]={'shard':record,'finished_truthy':bool(value.get('finished'))}
                if record['bytes']>largest:
                    largest=record['bytes'];samples.append(self_memory('v9_largest_run_materialized'))
                del value
        after=file_record(source_path)
        if before!=after:raise ValueError('V8 source changed during migration; index not published')
        gc.collect();samples.append(self_memory('v9_after_sharding_gc'))
        imported={'kind':'preserved_V8_closed_streaming_import_v9','complete':True,
            'source_state':before,'source_v8_script':file_record(producer),
            'source_closed_confirmed_by_ROOT':True,'source_closure_receipt':file_record(source_closure_receipt) if source_closure_receipt else None,
            'runs_count':len(entries),'largest_run_shard_bytes':largest,
            'metric_or_flag_values_rewritten':False,'Trackio_database_writes':False,
            'source_state_unchanged':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        index={'schema':1,'kind':'Trackio_run_shard_index_v9','source_import':imported,'runs':entries}
        receipt={'schema':1,'kind':'Trackio_V8_to_V9_streaming_migration_v1','passed':True,
            **imported,'self_only_memory_samples':samples,'resident_scope':'one run value during import; active run only afterward',
            'source_process_OS_closure':'ROOT observation; not established by this script',
            'global_or_other_process_attribution':'not established'}
        receipt_path=self.path.with_name(self.path.stem+'_migration_receipt.json')
        write_json(receipt_path,receipt)
        index['migration_receipt']=file_record(receipt_path)
        write_json(self.path,index);self.index=index
        return receipt
    def activate(self,run_id):
        if type(run_id) is not str or not run_id:raise ValueError('Actual run ID required')
        if self.index is None:raise ValueError('Complete migration required')
        if self.active_id==run_id:return self.active
        self.flush();self.active=None;self.active_id=None
        entry=self.index['runs'].get(run_id)
        if entry is None:value={}
        else:
            path=self.shard_path(entry['shard']);record=file_record(path)
            if record['sha256']!=entry['shard']['sha256'] or record['bytes']!=entry['shard']['bytes']:
                raise ValueError('Immutable active shard checksum differs')
            value=read_json(path,None)
            if type(value) is not dict or bool(value.get('finished'))!=entry.get('finished_truthy'):
                raise ValueError('Shard flags/index differ')
        self.active_id=run_id;self.active=value;self.dirty=False
        return self.active
    def update_metrics(self,step,values):
        if self.active is None:raise ValueError('No active run')
        old=self.active.get(str(step))
        if old is None:old={}
        if type(old) is not dict:raise ValueError('Existing step is not a metric map')
        changed={key:value for key,value in values.items() if key not in old or old[key]!=value}
        if changed:
            # Caller logs changed keys first, then accepts them in memory.
            return changed
        return {}
    def accept_metrics(self,step,changed):
        if changed:
            self.active.setdefault(str(step),{}).update(changed);self.dirty=True
    def set_flag(self,key,value):
        if key not in self.active or self.active[key]!=value:self.active[key]=value;self.dirty=True
    def is_finished(self,run_id):
        if run_id==self.active_id:return bool(self.active.get('finished'))
        return bool(self.index['runs'].get(run_id,{}).get('finished_truthy'))
    def flush(self):
        if not self.dirty:return False
        record=self.write_shard(self.active)
        self.index['runs'][self.active_id]={'shard':record,'finished_truthy':bool(self.active.get('finished'))}
        write_json(self.path,self.index);self.dirty=False
        return True


class PublishedJsonChanged(ValueError):
    pass


class StatJsonCache:
    'One config/history value per path; skip unchanged full history parses.'
    def __init__(self):self.entries={}
    def read(self,path,default):
        path=Path(path)
        if not path.exists():
            self.entries.pop(str(path),None);return default
        before=path.stat();key=(before.st_mtime_ns,before.st_size)
        entry=self.entries.get(str(path))
        if entry is not None and entry[0]==key:return entry[1]
        # Drop the cached decoded list before reading the next full publication.
        # follow() also drops its own previous history reference before this call.
        self.entries.pop(str(path),None);del entry;gc.collect()
        value=read_json(path,None)
        after=path.stat()
        if key!=(after.st_mtime_ns,after.st_size):raise PublishedJsonChanged('Published JSON changed during read; wait for next complete cycle')
        if value is None:raise ValueError('Published JSON is invalid; no stale replacement accepted')
        self.entries[str(path)]=(key,value)
        return value



SOURCE_V9_SHA='076ab5f58fe6cbcd2b061af6346d98fa58aaa500700b853a8d35b356fe44a674'


def stat_token(path):
    value=Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)


def shared_read_text(path,*,binary=False):
    """Allow the worker's atomic rename while streaming a Windows file snapshot."""
    path=Path(path)
    if os.name!='nt':return path.open('rb') if binary else path.open('r',encoding='utf-8')
    import ctypes
    from ctypes import wintypes
    import msvcrt
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p]
    api.CreateFileW.restype=ctypes.c_void_p
    api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle.restype=wintypes.BOOL
    handle=api.CreateFileW(str(path),0x80000000,7,None,3,0x80,None)
    if handle in (None,ctypes.c_void_p(-1).value):raise ctypes.WinError(ctypes.get_last_error())
    try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:api.CloseHandle(handle);raise
    try:return os.fdopen(fd,'rb') if binary else os.fdopen(fd,'r',encoding='utf-8')
    except BaseException:os.close(fd);raise


class HistoryArray(TopLevelRuns):
    # One complete object record at a time; inherited nested/escape parser.
    def __init__(self,stream,chunk_size=65536,max_record_chars=4000000):
        super().__init__(stream,chunk_size);self.max_record_chars=max_record_chars
    def records(self):
        self.whitespace()
        if self.take()!='[':raise ValueError('Worker history must be a top-level array')
        self.whitespace();count=0
        if self.peek()==']':self.take()
        else:
            while True:
                record=self.value();count+=1
                if type(record.get('step')) is not int or record['step']!=count:
                    raise ValueError('Canonical worker history must have contiguous step1..N')
                yield record
                self.whitespace();separator=self.take()
                if separator==']':break
                if separator!=',':raise ValueError('Truncated/invalid history separator')
                self.whitespace()
                if self.peek()==']':raise ValueError('Trailing history comma')
        self.whitespace()
        if self.peek():raise ValueError('Trailing history content')


def baseline_steps(config):
    result=set()
    warm=config.get('warm_start',{})
    if warm.get('mode')=='exact_resume' and type(warm.get('source_step')) is int:
        result.add(warm['source_step'])
    if config.get('phase') in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):result.add(config['source_phase1_step'])
    elif config.get('phase') in (ACQUISITION_PHASE,CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):
        result.add(config['source_selected_step'])
    return result


def initial_development_step(config):
    if config.get('phase') in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):return config['source_phase1_step']
    if config.get('phase') in (ACQUISITION_PHASE,CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):return config['source_selected_step']
    return 0


def actual_development_row(config,record):
    if 'validation' not in record:return False
    phase=config.get('phase')
    if phase in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):
        return 'adaptation_updates_completed' in record and record['step']>config['source_phase1_step']
    if phase in (CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):
        return 'acquisition_updates_completed' in record and record['step']>config['source_selected_step']
    if phase==ACQUISITION_PHASE:return 'acquisition_updates_completed' in record
    return True


def development_from_steps(run_id,steps):
    result={}
    for step in steps:
        summary=read_json(OUTPUT/f'results/{run_id}_step_{step}/summary.json',{})
        if summary:
            result[step]={f'dev/wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                for key,value in summary.items() if key.startswith('validation_')}
    return result


class HistoryScanCache:
    # Bounded RAM: private raw disk snapshot + phase baselines/DEV integers.
    # The worker file is held only during a 1MiB-buffer raw copy, not parsing.
    def __init__(self,snapshot_directory=None):
        self.key=None;self.info=None;self.snapshot=None
        self.snapshot_directory=Path(snapshot_directory or ROOT/'work').resolve()
    def close(self):
        if self.snapshot is not None:self.snapshot.close();self.snapshot=None
        self.key=None;self.info=None
    def scan(self,path,config):
        path=Path(path)
        if not path.exists():self.close();return None,False
        token=stat_token(path)
        config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        key=(str(path.resolve()),token,config_sha)
        if self.key==key:return self.info,False
        self.close();self.snapshot_directory.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryFile(mode='w+b',dir=self.snapshot_directory)
        snapshot=None;digest=hashlib.sha256();copied=0;copy_start=time.monotonic()
        try:
            with shared_read_text(path,binary=True) as stream:
                fd=os.fstat(stream.fileno())
                if (fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns)!=token:raise PublishedJsonChanged('History replaced before raw snapshot')
                while True:
                    block=stream.read(1024*1024)
                    if not block:break
                    temporary.write(block);digest.update(block);copied+=len(block)
                after=os.fstat(stream.fileno())
                if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=token:
                    raise PublishedJsonChanged('History source changed during short raw copy')
            if stat_token(path)!=token or copied!=token[2]:raise PublishedJsonChanged('History path changed during raw snapshot')
            copy_seconds=time.monotonic()-copy_start
            temporary.flush();temporary.seek(0);snapshot=io.TextIOWrapper(temporary,encoding='utf-8');temporary=None
            wanted=loralr_phase_baseline_steps(config);baselines=[];steps={loralr_phase_initial_development_step(config)};last_step=0
            for record in HistoryArray(snapshot).records():
                last_step=record['step']
                if last_step in wanted:baselines.append(record)
                if loralr_phase_actual_development_row(config,record):steps.add(last_step)
            if stat_token(path)!=token:raise PublishedJsonChanged('History publication changed during baseline pass')
            info={'token':token,'contexts':loralr_phase_contexts_for(config,baselines),'baseline_rows':baselines,
                'development_steps':steps,'last_step':last_step,'records_count':last_step,
                'retained_history_record_count':len(baselines),'source_bytes':token[2],
                'raw_snapshot_sha256':digest.hexdigest(),'raw_snapshot_bytes':copied,
                'source_short_read_seconds':copy_seconds,'history_source_file_unchanged':True}
            self.snapshot=snapshot;self.key=key;self.info=info
            return info,True
        except BaseException:
            if snapshot is not None:snapshot.close()
            if temporary is not None:temporary.close()
            raise
    def rows(self,path,info):
        path=Path(path);token=info['token']
        if self.snapshot is None or self.info is not info:raise ValueError('No matching private raw snapshot')
        if stat_token(path)!=token:raise PublishedJsonChanged('History replaced before metric pass')
        self.snapshot.seek(0)
        for record in HistoryArray(self.snapshot).records():yield record
        if stat_token(path)!=token:
            # Every prior log is from the same verified byte-exact private old
            # snapshot. Flush dedup state then retry the new atomic publication.
            raise PublishedJsonChanged('History advanced during metric pass; old snapshot rows remain valid')


class StreamRunShardStore(RunShardStore):
    # All V12 and inherited source archives stay immutable; prune only own V13 shards.
    def __init__(self,index_path):
        self.path=Path(index_path).resolve();self.directory=self.path.parent/(self.path.stem+'_shards')
        self.active_id=None;self.active=None;self.dirty=False
        self.index=read_json(self.path,None) if self.path.exists() else None
        if self.path.exists() and (type(self.index) is not dict or self.index.get('schema')!=1
            or self.index.get('kind')!='Trackio_run_shard_index_v13'
            or self.index.get('source_import',{}).get('complete') is not True
            or type(self.index.get('runs')) is not dict or type(self.index.get('live_versions')) is not dict
            or type(self.index.get('obsolete_live_shards')) is not list):
            raise ValueError('Invalid/incomplete V13 index; no silent reimport')
        if self.index is not None:
            for entry in self.index['runs'].values():self.shard_path(entry['shard'])
            self.prune_obsolete()
    def shard_path(self,record):
        sha=record.get('sha256')
        if type(sha) is not str or re.fullmatch('[0-9a-f]{64}',sha) is None or type(record.get('bytes')) is not int or record['bytes']<2:
            raise ValueError('Invalid shard SHA/size')
        path=Path(record.get('path','')).resolve()
        archives=self.index['source_import'].get('source_archive_directories',[]) if self.index is not None else []
        allowed=[self.directory.resolve()]+[Path(archive).resolve() for archive in archives]
        if path.parent not in allowed or path.name!=sha+'.json':raise ValueError('Shard is outside its immutable content directories')
        return path
    def migrate(self,source_path,*,source_closed_confirmed=False,source_closure_receipt=None):
        if self.index is not None:return None
        if not source_closed_confirmed:raise ValueError('ROOT must observe V12 CLOSED before V13 handoff')
        source_path=Path(source_path).resolve()
        if source_path==self.path:raise ValueError('V11 source/V13 index must be distinct')
        producer=Path(__file__).with_name('track_continuation_v12.py')
        if file_record(producer)['sha256']!=SOURCE_V12_SHA:raise ValueError('Frozen V12 producer changed')
        before=file_record(source_path);old=read_json(source_path,None)
        if (type(old) is not dict or old.get('kind')!='Trackio_run_shard_index_v12'
            or old.get('schema')!=1 or old.get('source_import',{}).get('complete') is not True
            or type(old.get('runs')) is not dict):raise ValueError('Complete pinned V12 index required')
        if file_record(source_path)!=before:raise ValueError('V12 index changed during handoff')
        archive=source_path.parent/(source_path.stem+'_shards')
        inherited=old['source_import'].get('source_archive_directories',[])
        if not inherited and old['source_import'].get('source_archive_directory'):
            inherited=[old['source_import']['source_archive_directory']]
        archives=list(dict.fromkeys([str(archive.resolve()),*(str(Path(path).resolve()) for path in inherited)]))
        if any(not Path(path).is_relative_to(ROOT.resolve()) for path in archives):raise ValueError('Source shard archives must remain in workspace')
        imported={'complete':True,'source_index':before,'source_v12_script':file_record(producer),
            'source_archive_directory':str(archive.resolve()),'source_archive_directories':archives,'source_V12_index_payload':old,
            'source_closed_confirmed_by_ROOT':True,'source_closure_receipt':file_record(source_closure_receipt) if source_closure_receipt else None,
            'source_all_shards_retained_without_deletion':True,'source_shards_rehashed_at_handoff':False,
            'active_shard_hash_verified_on_load':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        index={'schema':1,'kind':'Trackio_run_shard_index_v13','source_import':imported,
            'runs':dict(old['runs']),'live_versions':{},'obsolete_live_shards':[],
            'retention_policy':{'latest_live_versions_per_run':2,'source_V8_V9_V10_V11_V12_archive_pruned':False,
                'delete_obsolete_own_shards_only_after_atomic_index_commit':True}}
        self.index=index
        try:
            for entry in index['runs'].values():self.shard_path(entry['shard'])
            write_json(self.path,index)
        except BaseException:self.index=None;raise
        return imported
    def prune_obsolete(self):
        if self.index is None or not self.index['obsolete_live_shards']:return False
        protected={entry['shard']['sha256'] for entry in self.index['runs'].values()}
        protected.update(record['sha256'] for versions in self.index['live_versions'].values() for record in versions)
        pending=list(self.index['obsolete_live_shards'])
        for record in pending:
            path=self.shard_path(record)
            if path.parent!=self.directory.resolve():raise ValueError('Never prune source archive shard')
            if record['sha256'] in protected:continue
            if path.exists():
                actual=file_record(path)
                if actual['sha256']!=record['sha256'] or actual['bytes']!=record['bytes']:
                    raise ValueError('Obsolete immutable shard checksum differs; refuse deletion')
                path.unlink()
        self.index['obsolete_live_shards']=[];write_json(self.path,self.index)
        return True
    def flush(self):
        if not self.dirty:self.prune_obsolete();return False
        record=self.write_shard(self.active)
        previous=self.index['live_versions'].get(self.active_id,[])
        versions=[record]+[prior for prior in previous if prior['sha256']!=record['sha256']]
        retained=versions[:2];obsolete=versions[2:]
        self.index['runs'][self.active_id]={'shard':record,'finished_truthy':bool(self.active.get('finished'))}
        self.index['live_versions'][self.active_id]=retained
        self.index['obsolete_live_shards'].extend(obsolete)
        write_json(self.path,self.index);self.dirty=False
        self.prune_obsolete()
        return True


def portuguese_reference_record(config):
    return config.get('portuguese_retention_baseline_proof') or config.get('config_identity',{}).get('portuguese_retention_baseline_proof')


def required_eval_languages(config):
    historical=['en','es','it','fr','de','tr']
    declared=config.get('evaluation_languages') or config.get('config_identity',{}).get('evaluation_languages')
    if declared is None:
        return historical+(['pt'] if portuguese_reference_record(config) else [])
    if (type(declared) not in (list,tuple) or len(declared)!=len(set(declared))
        or not set(historical).issubset(declared) or not set(declared).issubset(historical+['pt'])):
        raise ValueError('Declared evaluation languages must preserve historical6 plus optional Portuguese')
    if portuguese_reference_record(config) and 'pt' not in declared:
        raise ValueError('PT baseline reference requires Portuguese evaluation scope')
    return list(declared)


def portuguese_base_reference(config):
    reference=portuguese_reference_record(config)
    if reference is None:return None
    if type(reference) is not dict:raise ValueError('PT baseline must be an explicit immutable byte record')
    path=Path(reference.get('path','')).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path.suffix!='.json':raise ValueError('PT reference must be workspace JSON')
    actual=file_record(path)
    if any(actual[key]!=reference.get(key) for key in ('path','sha256','bytes')):raise ValueError('PT baseline proof bytes changed')
    proof=read_json(path,None)
    if (type(proof) is not dict or proof.get('kind')!='Portuguese_retention_baselines_v1' or proof.get('passed') is not True
        or proof.get('actual_GPU_decode') is not True or proof.get('fixed_PT_DEV_count')!=50
        or proof.get('base_revision')!='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
        or proof.get('original_six_language_baseline_unchanged') is not True
        or proof.get('no_TEST_prediction_or_score_used') is not True):raise ValueError('Actual separate originalv3 PT50 proof required')
    baseline=proof.get('baseline_PT',{}).get('validation_pt')
    if baseline is None or proof.get('baseline_overlay',{}).get('validation_pt')!=baseline:
        raise ValueError('Portuguese original baseline/overlay aliases differ')
    score=baseline.get('asr',{})
    if (score.get('utterances')!=50 or not finite_number(score.get('wer')) or not finite_number(score.get('cer'))
        or type(score.get('word_errors')) is not int or score['word_errors']<0
        or type(score.get('reference_words')) is not int or score['reference_words']<1
        or type(score.get('character_errors')) is not int or score['character_errors']<0
        or type(score.get('reference_characters')) is not int or score['reference_characters']<1
        or not math.isclose(score['wer'],score['word_errors']/score['reference_words'],rel_tol=1e-12,abs_tol=1e-12)
        or not math.isclose(score['cer'],score['character_errors']/score['reference_characters'],rel_tol=1e-12,abs_tol=1e-12)):
        raise ValueError('Portuguese actual original50 ASR counts must be finite/coherent')
    return {'record':{key:actual[key] for key in ('path','sha256','bytes')},
        'metrics':{'reference/base_v3_wer_pt_percent':100*score['wer']}}


def publish_cycle_memory(args,control,run_id,scan_info,history_streamed,completed=True):
    value={'schema':1,'kind':'owned_Trackio_v11_cycle_memory_v1','run_id':run_id,'pid':os.getpid(),'ppid':os.getppid(),
        'observed_at_utc':datetime.now(timezone.utc).isoformat(),'completed_metric_cycle':completed,
        'history_metric_pass_streamed':history_streamed,'history_bytes':scan_info['source_bytes'] if scan_info else None,
        'history_records':scan_info['records_count'] if scan_info else None,
        'retained_history_records':scan_info['retained_history_record_count'] if scan_info else 0,
        'history_raw_snapshot_sha256':scan_info['raw_snapshot_sha256'] if scan_info else None,
        'history_raw_snapshot_bytes':scan_info['raw_snapshot_bytes'] if scan_info else None,
        'source_short_read_seconds':scan_info['source_short_read_seconds'] if scan_info else None,
        'memory':self_memory('v11_after_metric_cycle'),
        'scope':'Own mirror PID only; no other-process attribution or GPU measurements'}
    if args.telemetry_path is not None:write_json(args.telemetry_path,value)
    if control is not None and control['path'] is not None:
        control['identity']['cycle_telemetry']=value
        control['identity']['updated_at_utc']=value['observed_at_utc']
        write_json(control['path'],control['identity'])


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
    telemetry=scoped_control_path(getattr(args,'telemetry_path',None));args.telemetry_path=telemetry
    paths.extend(path for path in (stop,identity,telemetry) if path is not None)
    if telemetry is not None and telemetry.exists():raise ValueError('Own memory telemetry must use a fresh path')
    if len(paths)!=len(set(paths)):
        raise ValueError('Mirror controls and preserved/new state paths must be distinct')
    if stop is not None and stop.exists():
        raise ValueError('Stale existing mirror stop-file rejected before startup')
    if identity is None:
        return {'path':None,'identity':None}
    if identity.exists():
        raise ValueError('Mirror identity-file must have a fresh path')
    identity.parent.mkdir(parents=True,exist_ok=True)
    value={'schema':1,'kind':'owned_trackio_continuation_mirror_v11','status':'starting',
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
    if config and config.get('phase') in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):
        steps={config['source_phase1_step'],*[r['step'] for r in history
                if 'validation' in r and 'adaptation_updates_completed' in r
                and r['step']>config['source_phase1_step']]}
    elif config and config.get('phase') in (CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):
        steps={config['source_selected_step'],*[r['step'] for r in history
            if 'validation' in r and 'acquisition_updates_completed' in r
            and r['step']>config['source_selected_step']]}
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


def contexts_for(config,baseline_rows):
    return {'context':resume_context(config,baseline_rows),
        'phase_context':acquisition_context(config,baseline_rows),
        'adaptation':adaptation_context(config,baseline_rows),
        'continued':continued_acquisition_context(config,baseline_rows),
        'scheduled':scheduled_acquisition_context(config,baseline_rows),
        'shared':shared_head_context(config,baseline_rows)}


def cycle_rows(config,history,development):
    contexts=contexts_for(config,history)
    for record in history:
        yield record['step'],value_for_record(record,contexts,development)


def value_for_record(record, contexts, development):
    context = contexts['context']
    phase_context = contexts['phase_context']
    adaptation = contexts['adaptation']
    continued = contexts['continued']
    scheduled = contexts['scheduled']
    shared = contexts['shared']
    values = values_for(record, context)
    values.update(acquisition_values_for(record, phase_context))
    values.update(adaptation_values_for(record, adaptation))
    values.update(continued_acquisition_values_for(record, continued))
    values.update(scheduled_acquisition_values_for(record, scheduled))
    values.update(shared_head_values_for(record, shared))
    if shared is not None and record['step'] <= shared['step']:
        values['phase/inherited_history'] = int(record['step'] < shared['step'])
        if record['step'] not in development:
            values = {key: value for key, value in values.items() if not key.startswith('retention/')}
    if scheduled is not None and record['step'] <= scheduled['step']:
        values['phase/inherited_history'] = int(record['step'] < scheduled['step'])
        if record['step'] not in development:
            values = {key: value for key, value in values.items() if not key.startswith('retention/')}
    if continued is not None and record['step'] <= continued['step']:
        values['phase/inherited_history'] = int(record['step'] < continued['step'])
        if record['step'] not in development:
            values = {key: value for key, value in values.items() if not key.startswith('retention/')}
    if adaptation is not None and record['step'] <= adaptation['step']:
        values['phase/inherited_history'] = int(record['step'] < adaptation['step'])
        if record['step'] not in development:
            values = {key: value for key, value in values.items() if not key.startswith('retention/')}
    values.update(development.get(record['step'], {}))
    return values


def follow(args,control=None):
    store=StreamRunShardStore(args.state_path)
    store.migrate(args.source_state_path,source_closed_confirmed=args.source_closed_confirmed,
        source_closure_receipt=args.source_closure_receipt)
    active=None;config_cache=StatJsonCache();history_cache=HistoryScanCache(args.state_path.parent);processed_key=None
    try:
        while True:
            if stop_requested(args):return 'cooperative_stop_file'
            config_path=OUTPUT/'run_config.json';history_path=OUTPUT/'training_history.json'
            try:
                config=config_cache.read(config_path,{})
                info,history_changed=history_cache.scan(history_path,config)
            except PublishedJsonChanged:
                if not args.follow:return 'publication_changed_no_metrics_logged'
                time.sleep(5);continue
            run_id=config.get('run_id')
            coherent=bool(run_id and info and info['records_count'] and config_path.exists() and history_path.exists()
                and history_path.stat().st_mtime>=config_path.stat().st_mtime)
            streamed=False
            if coherent and not store.is_finished(run_id):
                development=development_from_steps(run_id,info['development_steps'])
                switching=active!=run_id
                if switching:
                    if active is not None:trackio.finish()
                    state=store.activate(run_id)
                    trackio.init(project=args.project,name=run_id,group='Turkish adaptation',config=config,resume='allow',gpu_log_interval=10.)
                    active=run_id;publish_control_run(control,run_id)
                    if 'reference_base_v3' not in state:
                        baseline=read_json(OUTPUT/'results/base_v3/summary.json',{})
                        trackio.log({f'reference/base_v3_wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                            for key,value in baseline.items() if key.startswith('validation_')},step=0)
                        store.set_flag('reference_base_v3',True)
                    pt_reference=portuguese_base_reference(config)
                    if pt_reference is not None:
                        old_reference=state.get('reference_base_pt_verified_v1')
                        if old_reference is not None and old_reference!=pt_reference['record']:
                            raise ValueError('Original PT reference may not change within mirrored run')
                        if old_reference is None:
                            trackio.log(pt_reference['metrics'],step=0)
                            store.set_flag('reference_base_pt_verified_v1',pt_reference['record'])
                    print(f'Mirroring {run_id}; streamed history and active-run immutable shards',flush=True)
                initial_step=loralr_phase_initial_development_step(config)
                initial=store.update_metrics(initial_step,development.get(initial_step,{}))
                if initial:trackio.log(initial,step=initial_step);store.accept_metrics(initial_step,initial)
                cycle_key=(run_id,history_cache.key)
                try:
                    if processed_key!=cycle_key or switching:
                        streamed=True
                        for record in history_cache.rows(history_path,info):
                            step=record['step'];values=loralr_phase_value_for_record(record,info['contexts'],development)
                            changed=store.update_metrics(step,values)
                            if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                        processed_key=cycle_key
                    else:
                        # No history reparse: DEV amendments still log each changed
                        # key; a source-baseline row restores inherited retention
                        # fields exactly when its real initial summary appears.
                        for record in info['baseline_rows']:
                            step=record['step'];values=loralr_phase_value_for_record(record,info['contexts'],development)
                            changed=store.update_metrics(step,values)
                            if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                        for step,values in development.items():
                            changed=store.update_metrics(step,values)
                            if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                except PublishedJsonChanged:
                    store.flush();publish_cycle_memory(args,control,run_id,info,streamed,completed=False)
                    if not args.follow:return 'publication_advanced_valid_old_snapshot_rows_logged'
                    time.sleep(5);continue
                store.flush()
                selected=read_json(OUTPUT/'selection.json',{});pointer=read_json(OUTPUT/'adapted_results_path.json',{})
                uniform_selection=loralr_selection_values(config,selected,info['last_step'])
                if uniform_selection:
                    selected_step=selected['completed_steps'];changed=store.update_metrics(selected_step,uniform_selection)
                    if changed:trackio.log(changed,step=selected_step);store.accept_metrics(selected_step,changed);store.flush()
                if pointer.get('tag')==run_id+'_adapted':
                    final=read_json(OUTPUT/f'results/{run_id}_adapted/summary.json',{})
                    if all('test_'+lang in final for lang in required_eval_languages(config)):
                        values={f'test/wer_{key.removeprefix("test_")}_percent':100*value['asr']['wer'] for key,value in final.items() if key.startswith('test_')}
                        values['selection/best_step']=selected.get('selected_step',0)
                        trackio.log(values,step=info['last_step']);trackio.finish()
                        store.set_flag('finished','heldout-evaluation-complete');store.flush();active=None
                elif (selected.get('run_id')==run_id and selected.get('accepted') is False
                    and isinstance(selected.get('completed_steps'),int) and info['last_step']>=selected['completed_steps']
                    and loralr_rejected_run_terminal(config,selected)):
                    trackio.log({'selection/accepted':0,'selection/report':trackio.Markdown(selected.get('reason','Rejected on development gates.'))},step=info['last_step'])
                    trackio.finish();store.set_flag('finished','rejected-on-development');store.flush();active=None
            publish_cycle_memory(args,control,run_id,info,streamed)
            if not args.follow:
                if active is not None:trackio.finish()
                return 'one_cycle_complete'
            if stop_requested(args):return 'cooperative_stop_file'
            time.sleep(5)
    finally:
        try:store.flush()
        finally:history_cache.close()




def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='parakeet-turkish')
    p.add_argument('--follow',action='store_true')
    p.add_argument('--state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v13.json')
    p.add_argument('--source-state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v12.json',
                   help='Preserve existing flags/points on ROOT-owned mirror handoff; used only when new state does not exist')
    p.add_argument('--source-closed-confirmed',action='store_true',help='ROOT observed source V12 process CLOSED (including explicitly reviewed finalizer-only exit1); required for first small-index handoff')
    p.add_argument('--source-closure-receipt',type=Path,help='Optional immutable ROOT source closure record, checksum-bound in import receipt')
    p.add_argument('--telemetry-path',type=Path,help='Fresh workspace path for actual own-PID memory after each complete metric cycle')
    p.add_argument('--stop-file',type=Path,help='Optional fresh workspace path; exit between complete mirror cycles when it appears')
    p.add_argument('--identity-file',type=Path,help='Optional fresh workspace path for actual self/start/runstate and final shutdown metadata')
    return p


def main(args):
    for path in (args.state_path,args.source_state_path):
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('State path must be inside workspace')
    if args.state_path.resolve()==args.source_state_path.resolve():raise ValueError('New v13 index must be separate from preserved source state')
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
            if trackio.context_vars.current_run.get() is not None:trackio.finish()
            finished=True
        except BaseException as exception:
            status='failed';reason='trackio_finish_failed';error=repr(exception)
            raise
        finally:
            finish_controls(control,status,reason,error,finished)


if __name__=='__main__':
    main(parser().parse_args())
