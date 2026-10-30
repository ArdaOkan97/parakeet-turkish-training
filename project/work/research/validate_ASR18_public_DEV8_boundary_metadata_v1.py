"""Explicitly PRIVATE virtual-process metadata controls, plus real bounded heads.

No future actual-process proof, native checkpoint, saved predictions, model or
official DEV execution is created. Only the new pure boundary/reader API runs.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
import asr18_full_run_serial_DEV_v8 as dev
from fastmss_row_serial_cpu_v6 import environment,resource_guard,admit
from bpe_broad_production_state_v1 import canonical,digest,require
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

def guard():
    value=resource_guard();require(value['worker_private_memory']['private_bytes']<100_000_000,'Metadata fixture100MB')
    return value

def virtual(name):
    return {'path':'PRIVATE_VIRTUAL_NOT_ACTUAL:'+name,'bytes':0,'sha256':hashlib.sha256(name.encode()).hexdigest()}

def frame(stream,head,*,previous,operation,evidence):
    offset=stream.tell();value={'kind':'ASR18_packed_durable_event_and_head_record_v2','sequence':head['sequence'],
        'owner':head['owner'],'identity_sha256':head['identity_sha256'],'previous_head':previous,
        'operation':operation,'evidence':evidence,'head':head}
    for _ in range(16):
        raw=canonical(value);end=offset+40+len(raw)
        if head['total_event_bytes']==end:break
        head['total_event_bytes']=end
    else:raise ValueError('Fixture frame length convergence')
    sha=hashlib.sha256(raw).digest();stream.write(struct.pack('<Q',len(raw))+sha+raw)
    return {'kind':'ASR18_logical_immutable_durable_head_descriptor_v2','path':str(Path(stream.name).resolve()),
        'offset':offset,'framed_bytes':40+len(raw),'sha256':sha.hex(),'sequence':head['sequence']},value

def case(output):
    pins={name:virtual(name) for name in ('boundary_verifier','native_worker','native_parent','recovery_worker','recovery_parent','durable_prefix_verifier')}
    names=('native0','native_report','recovery0','recovery_report','recovered','ack','prefix','pointer','checkpoint','pre_save','next_draw')
    a={name:virtual(name) for name in names};identity={'PRIVATE_SYNTHETIC_NO_PRODUCTION_IDENTITY':True};sha=digest(identity)
    request={'native_boundary_sources':pins,'identity':identity,'optimizer_updates':21,'checkpoint_pointer':a['pointer']}
    pointer={'checkpoint':a['checkpoint'],'sampler_state_sha256':'sampler-sha','journal_prefix':{'PRIVATE_VIRTUAL':True}}
    capture={'pointer':a['pointer'],'checkpoint':a['checkpoint'],'pre_save_fingerprint':a['pre_save'],'next_draw_packet':a['next_draw'],
        'checkpoint_persistence_verified':True,'exact_GPU_restore_performed':False,'publication_role':'completed_window_end',
        'current_latest_recovery_required_before_phase_DEV_winner_advance':True}
    evidence={'kind':'ASR18_full_run_native_exit_and_CURRENT_checkpoint_recovery_boundary_for_serial_DEV_v2','owner':dev.OWNER,
        'passed':True,'diagnostic_only':False,'source':pins['boundary_verifier'],'production_recovery_scope':'CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX',
        'checkpoint_pointer':a['pointer'],'run_identity_sha256':sha,'optimizer_updates':21,'sampler_state_sha256':pointer['sampler_state_sha256'],
        'journal_prefix':pointer['journal_prefix'],'all207_threeAdam_all4RNG_exact_checkpoint_restore_verified':True,
        'current_checkpoint_capture':capture,'current_native_process0':a['native0'],'actual_recovery_process0':a['recovery0'],
        'recovery_report':a['recovery_report'],'production_recovery_boundary':a['ack'],'durable_persistence_prefix_verification':a['prefix']}
    objects={}
    def put(name,value):objects[digest(a[name])]=value
    for name,worker,parent,report_name in (('native0','native_worker','native_parent','native_report'),('recovery0','recovery_worker','recovery_parent','recovery_report')):
        put(name,{'PRIVATE_VIRTUAL_PROCESS_IO':True,'kind':'ASR18_owned_GPU_process_completion_v1','owner':dev.OWNER,
            'worker_source':pins[worker],'parent_source':pins[parent],'report':a[report_name]})
    put('native_report',{'passed':True,'diagnostic_only':False,'run_identity_sha256':sha,'current_checkpoint':deepcopy(capture),
        'final_optimizer_updates':21,'all_exact_GPU_checkpoint_restores_deferred':True})
    put('recovery_report',{'kind':'ASR18_full_run_actual_CURRENT_checkpoint_recovered_v2','passed':True,'diagnostic_only':False,
        'current_latest_checkpoint_recovered':True,'historical_fresh_GPU_restores_claimed':False,'current_native_process0':a['native0'],
        'run_identity_sha256':sha,'actual_native_updates_in_recovery_process':0,'actual_draws_consumed_in_recovery':0,'recovered_checkpoints':[a['recovered']]})
    put('recovered',{'passed':True,'pointer':a['pointer'],'checkpoint':a['checkpoint'],'pre_save_fingerprint':a['pre_save'],
        'next_draw_packet':a['next_draw'],'all207_threeAdam_all4RNG_match_actual_pre_save':True,
        'both_samplers_journal_prefix_and_next_draw_exact':True,'native_updates_performed':0,'draws_consumed':0,'separate_fresh_GPU_recovery':True})
    pending={'publications':1,'windows':1,'PRIVATE_SYNTHETIC_ONLY':True}
    lifecycle={'last_pointer':a['pointer'],'last_sampler_sha256':pointer['sampler_state_sha256']}
    common={'kind':'ASR18_packed_durable_full_run_lifecycle_head_v2','owner':{'PRIVATE_SYNTHETIC_OWNERSHIP':True},
        'identity_sha256':sha,'total_event_bytes':0,'lifecycle':lifecycle,'production_enabled':False,'TEST_complete':False,'export_complete':False}
    with (output/'PRIVATE_SYNTHETIC_records.bin').open('xb') as stream:
        pending_pin,pending_record=frame(stream,{**deepcopy(common),'sequence':1,'pending':pending,'last_recovery':None},
            previous=None,operation='PRIVATE_PENDING_NO_NATIVE_COMMIT',evidence={})
        ack={'kind':'ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1','diagnostic_only':False,
            'run_identity_sha256':sha,'pending_head':pending_pin,'pending':pending,'final_checkpoint_pointer':a['pointer'],
            'fresh_recovery_scope':'current_latest_only','all_persistence_verified_through_pending_prefix':True,
            'all_historical_checkpoints_freshly_restored':False,
            'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
            'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,'current_latest_exact_restore':a['recovered'],
            'persistence_prefix_coverage_index':a['prefix']}
        put('ack',ack)
        prefix={'kind':'ASR18_verified_durable_persistence_prefix_coverage_v1','passed':True,'source':pins['durable_prefix_verifier'],
            'run_identity_sha256':sha,'pending_head':pending_pin,'pending':pending,'final_checkpoint_pointer':a['pointer'],
            'current_checkpoint_capture_sha256':digest(capture),'all_completed_window_PRE_saved_live_persistence_verified':True,
            'historical_fresh_GPU_restores_claimed':False,'native_updates_consumed_in_verification':0}
        put('prefix',prefix)
        current_pin,current_record=frame(stream,{**deepcopy(common),'sequence':2,'pending':{'publications':0,'windows':0},'last_recovery':a['ack']},
            previous=pending_pin,operation='current_latest_recovery_and_persistence_prefix_acknowledged',evidence={'recovery':a['ack']})
        stream.flush();os.fsync(stream.fileno())
    evidence['durable_head']=current_pin
    heads={digest(pending_pin):pending_record,digest(current_pin):current_record}
    return {'evidence':evidence,'request':request,'pointer':pointer,'objects':objects,'pins':a,'heads':heads,
        'pending_pin':pending_pin,'current_pin':current_pin}

def checked(c,*,actual_head=False):
    def read_artifact(pin):return deepcopy(c['objects'][digest(pin)])
    def verify_process(value,*,kind,owner,report=None):
        require(value['PRIVATE_VIRTUAL_PROCESS_IO'] is True and value['kind']==kind and value['owner']==owner,'Only explicit virtual process adapter')
        if report is not None:require(dev.same(value['report'],report),'Exact virtual report relation')
        return True
    read_head=dev.read_logical_durable_head if actual_head else lambda pin:deepcopy(c['heads'][digest(pin)])
    return dev._checked_current_native_boundary(c['evidence'],request=c['request'],pointer=c['pointer'],
        read_artifact=read_artifact,verify_process=verify_process,read_head=read_head)

def main(output):
    environment();admission=admit(100_000_000,guard=guard);began=time.monotonic();output=Path(output).resolve()
    require(output.is_relative_to(dev.ROOT) and not output.exists(),'Fresh E-only metadata fixture');output.mkdir();checks=[];samples=[guard()]
    baseline=case(output);checked(baseline,actual_head=True);checks.append('Positive pure current-only boundary with two actual SHA-framed head reads')
    def obj(c,name):return c['objects'][digest(c['pins'][name])]
    def reject(name,change):
        c=deepcopy(baseline);change(c)
        try:checked(c)
        except (ValueError,KeyError,TypeError,OSError):checks.append(name);samples.append(guard());return
        raise ValueError('Expected rejection: '+name)
    reject('Stale current pointer',lambda c:c['evidence'].__setitem__('checkpoint_pointer',virtual('stale')))
    reject('Wrong native source',lambda c:obj(c,'native0').__setitem__('worker_source',virtual('other_source')))
    reject('Diagnostic cannot become public',lambda c:c['evidence'].__setitem__('diagnostic_only',True))
    reject('Boolean expected optimizer counter',lambda c:c['request'].__setitem__('optimizer_updates',True))
    reject('Historical GPU restoration cannot be claimed',lambda c:obj(c,'recovery_report').__setitem__('historical_fresh_GPU_restores_claimed',True))
    reject('Missing all207 proof',lambda c:obj(c,'recovered').pop('all207_threeAdam_all4RNG_match_actual_pre_save'))
    reject('Fresh GPU recovery is mandatory',lambda c:obj(c,'recovered').__setitem__('separate_fresh_GPU_recovery',False))
    reject('Duplicate recovery captures rejected',lambda c:obj(c,'recovery_report')['recovered_checkpoints'].append(c['pins']['recovered']))
    reject('Wrong current PRE-save',lambda c:obj(c,'recovered').__setitem__('pre_save_fingerprint',virtual('wrong_pre')))
    reject('Wrong current next-draw',lambda c:obj(c,'recovered').__setitem__('next_draw_packet',virtual('wrong_next')))
    reject('Wrong prefix capture SHA',lambda c:obj(c,'prefix').__setitem__('current_checkpoint_capture_sha256','f'*64))
    reject('Wrong independent prefix source',lambda c:obj(c,'prefix').__setitem__('source',virtual('wrong_prefix_source')))
    reject('Recovery consumes no draws',lambda c:obj(c,'recovered').__setitem__('draws_consumed',1))
    reject('Boolean zero-update counter rejected',lambda c:obj(c,'recovered').__setitem__('native_updates_performed',False))
    reject('Pending head owner mismatch',lambda c:c['heads'][digest(c['pending_pin'])]['head'].__setitem__('owner',{'other_owner':True}))
    reject('Post-ack previous head mismatch',lambda c:c['heads'][digest(c['current_pin'])].__setitem__('previous_head',virtual('old_head')))
    reject('Post-ack sampler mismatch',lambda c:c['heads'][digest(c['current_pin'])]['head']['lifecycle'].__setitem__('last_sampler_sha256','other'))
    reject('Pending publications must clear with exact integer',lambda c:c['heads'][digest(c['current_pin'])]['head']['pending'].__setitem__('publications',False))
    for field,value in (('path','PRIVATE_OTHER_STREAM'),('sequence',4),('offset',999)):
        def change(c,field=field,value=value):
            old=digest(c['current_pin']);pin=deepcopy(c['evidence']['durable_head']);pin[field]=value
            c['evidence']['durable_head']=pin;c['heads'][digest(pin)]=c['heads'][old]
        reject('Adjacent same-stream '+field+' required',change)
    # Real reader rejects claims encoded into fresh private records, independently
    # of the virtual-process core. These files contain no actual native evidence.
    for flag in ('production_enabled','TEST_complete','export_complete'):
        value=deepcopy(baseline['heads'][digest(baseline['current_pin'])]['head']);value[flag]=True
        with (output/('PRIVATE_'+flag+'.bin')).open('xb') as stream:
            value['sequence']=1;value['total_event_bytes']=0
            pin,_=frame(stream,value,previous=None,operation='PRIVATE_NEGATIVE',evidence={})
        try:dev.read_logical_durable_head(pin)
        except ValueError:checks.append('Actual head reader rejects '+flag);samples.append(guard())
        else:raise ValueError('Forbidden actual head flag accepted')
    bad=deepcopy(baseline['current_pin']);bad['framed_bytes']=65537
    try:dev.read_logical_durable_head(bad)
    except ValueError:checks.append('Head cap rejected before oversized read')
    else:raise ValueError('Oversized head descriptor accepted')
    require(not any(name in sys.modules for name in ('torch','numpy','librosa','transformers')),'No numerical/model/runtime import')
    allocated=sum(((p.stat().st_size+524287)//524288)*524288 for p in output.iterdir() if p.is_file())
    require(allocated<16*1024**2 and time.monotonic()-began<60,'Small metadata fixture allocation/time caps')
    result={'kind':'ASR18_PRIVATE_public_DEV8_current_boundary_metadata_controls_v1','passed':True,'source':record(__file__),
        'DEV_source':record(dev.__file__),'producer':{'pid':os.getpid(),'ppid':os.getppid()},'checks':checks,'check_count':len(checks),
        'CPU_admission':admission,'observed_private_peak_bytes':max(x['worker_private_memory']['private_bytes'] for x in samples),
        'wall_seconds':time.monotonic()-began,'fixture_allocation_rounded_bytes':allocated,
        'explicit_virtual_process_and_artifact_IO':True,'actual_native_DEV_or_recovery_proved':False,
        'actual_scorer_evaluator_model_or_payload_used':False,'public_policy_or_production_approved':False,'production_enabled':False}
    write_json(output/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'report':record(output/'report.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
