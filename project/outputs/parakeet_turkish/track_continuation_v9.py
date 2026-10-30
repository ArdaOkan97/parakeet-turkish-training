"""INACTIVE Trackio mirror v9: shared-head metrics and bounded resident run shards.

V8 is preserved. ROOT must confirm V8 CLOSED before streaming first migration.
Only the active run map remains resident; immutable SHA shards preserve unknown
fields. No Trackio database deletion or history rewrite is performed. Growing
worker history still requires one full parse when its stat changes.

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
        depth=0;quoted=False;pieces=[]
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
                        pieces.append(scan[:match.end()]);self.pos+=match.end()
                        value=json.loads(''.join(pieces),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
                        if type(value) is not dict:raise ValueError('Historical run must be an object')
                        return value
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
    value={'schema':1,'kind':'owned_trackio_continuation_mirror_v9','status':'starting',
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


def cycle_rows(config,history,development):
    context=resume_context(config,history)
    phase_context=acquisition_context(config,history)
    adaptation=adaptation_context(config,history)
    continued=continued_acquisition_context(config,history)
    scheduled=scheduled_acquisition_context(config,history)
    shared=shared_head_context(config,history)
    for record in history:
        values=values_for(record,context)
        values.update(acquisition_values_for(record,phase_context))
        values.update(adaptation_values_for(record,adaptation))
        values.update(continued_acquisition_values_for(record,continued))
        values.update(scheduled_acquisition_values_for(record,scheduled))
        values.update(shared_head_values_for(record,shared))
        if shared is not None and record['step']<=shared['step']:
            values['phase/inherited_history']=int(record['step']<shared['step'])
            if record['step'] not in development:
                values={key:value for key,value in values.items() if not key.startswith('retention/')}
        if scheduled is not None and record['step']<=scheduled['step']:
            values['phase/inherited_history']=int(record['step']<scheduled['step'])
            if record['step'] not in development:
                values={key:value for key,value in values.items() if not key.startswith('retention/')}
        if continued is not None and record['step']<=continued['step']:
            values['phase/inherited_history']=int(record['step']<continued['step'])
            if record['step'] not in development:
                values={key:value for key,value in values.items() if not key.startswith('retention/')}
        if adaptation is not None and record['step']<=adaptation['step']:
            values['phase/inherited_history']=int(record['step']<adaptation['step'])
            # Inherited training counters stay cumulative. DEV/retention points
            # belong to this run only when its actual new-run folder published.
            if record['step'] not in development:
                values={key:value for key,value in values.items() if not key.startswith('retention/')}
        values.update(development.get(record['step'],{}))
        yield record['step'],values


def follow(args,control=None):
    store=RunShardStore(args.state_path)
    store.migrate(args.source_state_path,source_closed_confirmed=args.source_closed_confirmed,
        source_closure_receipt=args.source_closure_receipt)
    active=None;cache=StatJsonCache()
    try:
        while True:
            if stop_requested(args):return 'cooperative_stop_file'
            config_path=OUTPUT/'run_config.json';history_path=OUTPUT/'training_history.json'
            # Release follow's old list; unchanged stat still reuses the cache.
            history=None;config=None
            try:
                config=cache.read(config_path,{})
                history=cache.read(history_path,[])
            except PublishedJsonChanged:
                if not args.follow:return 'publication_changed_no_metrics_logged'
                time.sleep(5);continue
            run_id=config.get('run_id')
            coherent=bool(run_id and history and config_path.exists() and history_path.exists()
                          and history_path.stat().st_mtime>=config_path.stat().st_mtime)
            if coherent and not store.is_finished(run_id):
                # Small DEV files remain amendable even if history stat is unchanged.
                development=development_cycle(run_id,history,config)
                if active!=run_id:
                    if active is not None:trackio.finish()
                    state=store.activate(run_id)
                    trackio.init(project=args.project,name=run_id,group='Turkish adaptation',
                                 config=config,resume='allow',gpu_log_interval=10.)
                    active=run_id;publish_control_run(control,run_id)
                    if 'reference_base_v3' not in state:
                        baseline=read_json(OUTPUT/'results/base_v3/summary.json',{})
                        trackio.log({f'reference/base_v3_wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                                     for key,value in baseline.items() if key.startswith('validation_')},step=0)
                        store.set_flag('reference_base_v3',True)
                    initial_step=(config['source_phase1_step'] if config.get('phase') in (ADAPTATION_PHASE,SHARED_HEAD_PHASE)
                        else config['source_selected_step'] if config.get('phase') in (ACQUISITION_PHASE,CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE) else 0)
                    initial=store.update_metrics(initial_step,development.get(initial_step,{}))
                    if initial:trackio.log(initial,step=initial_step);store.accept_metrics(initial_step,initial)
                    print(f'Mirroring {run_id}; active-run shard and cumulative/phase interval rates',flush=True)
                for step,values in cycle_rows(config,history,development):
                    changed=store.update_metrics(step,values)
                    if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                store.flush()
                selected=read_json(OUTPUT/'selection.json',{});pointer=read_json(OUTPUT/'adapted_results_path.json',{})
                if pointer.get('tag')==run_id+'_adapted':
                    final=read_json(OUTPUT/f'results/{run_id}_adapted/summary.json',{})
                    if all('test_'+lang in final for lang in ['en','es','it','fr','de','tr']):
                        values={f'test/wer_{key.removeprefix("test_")}_percent':100*value['asr']['wer']
                                for key,value in final.items() if key.startswith('test_')}
                        values['selection/best_step']=selected.get('selected_step',0)
                        trackio.log(values,step=history[-1]['step']);trackio.finish()
                        store.set_flag('finished','heldout-evaluation-complete');store.flush();active=None
                        print(f'Final held-out scores imported for {run_id}',flush=True)
                elif (selected.get('run_id')==run_id and selected.get('accepted') is False
                      and isinstance(selected.get('completed_steps'),int)
                      and history[-1]['step']>=selected['completed_steps']):
                    trackio.log({'selection/accepted':0,'selection/report':trackio.Markdown(selected.get('reason','Rejected on development gates.'))},step=history[-1]['step'])
                    trackio.finish();store.set_flag('finished','rejected-on-development');store.flush();active=None
            if not args.follow:
                if active is not None:trackio.finish()
                return 'one_cycle_complete'
            if stop_requested(args):return 'cooperative_stop_file'
            time.sleep(5)
    finally:
        store.flush()



def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='parakeet-turkish')
    p.add_argument('--follow',action='store_true')
    p.add_argument('--state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v9.json')
    p.add_argument('--source-state-path',type=Path,default=ROOT/'work/trackio_continuation_mirror_state_v8.json',
                   help='Preserve existing flags/points on ROOT-owned mirror handoff; used only when new state does not exist')
    p.add_argument('--source-closed-confirmed',action='store_true',help='ROOT observed source V8 process CLOSED; required for first streaming import')
    p.add_argument('--source-closure-receipt',type=Path,help='Optional immutable ROOT source closure record, checksum-bound in import receipt')
    p.add_argument('--stop-file',type=Path,help='Optional fresh workspace path; exit between complete mirror cycles when it appears')
    p.add_argument('--identity-file',type=Path,help='Optional fresh workspace path for actual self/start/runstate and final shutdown metadata')
    return p


def main(args):
    for path in (args.state_path,args.source_state_path):
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('State path must be inside workspace')
    if args.state_path.resolve()==args.source_state_path.resolve():raise ValueError('New v9 index must be separate from preserved source state')
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
