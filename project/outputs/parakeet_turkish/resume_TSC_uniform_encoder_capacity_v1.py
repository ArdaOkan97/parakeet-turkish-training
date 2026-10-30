"""One fresh-parent SAME-ID resume after actual owned capacity release.

Admits only the preserved34651 native-child0 resource boundary and a hash-bound
ROOT capacity-release review. All scientific worker/config/native tokens and
90%/95%/2GB guards remain unchanged. Fresh admission runs again immediately
before dispatch; full207 own-state validation remains in the frozen worker.
"""
from pathlib import Path
import argparse,sys,time,json
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
import run_TSC_uniform_encoder_schedule_v1 as frozen
EXPECTED='8a133bf34694412963be89b6b0dd14ca463bcf6a7d358bbd00e46de06c553770'
CLOSED_REVIEW_SHA='99a9fd290bf2391b31ce7a963e108e40064185be4fca92d5c3af8e35b4c0c6e1'

def byte_record(c,path):
    p=c.scoped(path);return {**c.record(p),'bytes':p.stat().st_size}

def validate_capacity_review(c,args,previous):
    review_path=c.scoped(args.capacity_review);review=c.read(review_path)
    c.require(review.get('kind')=='ROOT_owned_encoder_cosine34651_capacity_release_v1'
        and review.get('passed') is True and review.get('run_id')==previous['run_id']
        and review.get('source_step')==34651
        and review.get('source_actual_owned_exit')=={'session_id':99543,'exit_code':1,'tool_chunk':'dca628'}
        and review.get('capacity_release_actual_closed') is True
        and review.get('no_user_apps_closed') is True and review.get('no_guard_changes') is True,
        'Actual OWN capacity release for this source only is required')
    source=review['source_closure_review'];source_path=c.scoped(source['path'])
    c.require(source['sha256']==CLOSED_REVIEW_SHA and byte_record(c,source_path)==source,
        'Actual34651 resource closure review changed')
    closed=c.read(source_path)
    c.require(closed['status']==byte_record(c,args.previous_status)
        and closed['completed_step']==34651 and closed['actual_native_child_exit_code']==0,
        'Capacity review must bind this immutable CLOSED source')
    release=review['capacity_release_record'];release_path=c.scoped(release['path'])
    c.require(byte_record(c,release_path)==release,'Actual owned capacity-release evidence changed')
    before,after=review['baseline_before'],review['baseline_after']
    peak=previous['peak_observed_owned_private_bytes']
    c.require(before.get('available') is True and after.get('available') is True
        and type(before.get('commit_used_bytes')) is int and type(after.get('commit_used_bytes')) is int
        and after['commit_used_bytes']<before['commit_used_bytes']
        and after['commit_used_bytes']+peak<.90*after['commit_limit_bytes']
        and after['commit_limit_bytes']-after['commit_used_bytes']-peak>=2_000_000_000,
        'Actual changed baseline must fit the unchanged same-policy peak and guards')
    return c.record(review_path)


def main(args):
    c=frozen;c.require(c.sha(c.__file__)==EXPECTED,'Frozen execution source changed')
    c.require(args.gpu_window_authorized and .1<=args.window_hours<=3,'Sole authorized bounded window required')
    previous=c.read(c.scoped(args.previous_status));run=HERE/'runs'/previous['run_id'];selection=c.read(run/'selection.json')
    c.require(previous['kind']==c.CONTROLLER_KIND and previous['controller_sha256']==EXPECTED
        and previous['status']=='resource_review_required' and previous['child_exit_code']==0
        and previous['resource_review_required'] is True and previous['stop_category']=='resource'
        and previous['completed_steps']==34651
        and previous['cooperative_stop_reason']=='system_commit_above90percent_three_fresh_observations'
        and previous['completed_steps']==selection['completed_steps']
        and selection['completed_steps']<previous['configured_steps']
        and c.record(run/'selection.json')==previous['selection'],'Actual preserved34651 native-child0 resource source required')
    capacity_review=validate_capacity_review(c,args,previous)
    done=c.read(previous['actual_runtime_completion']['runtime_completion']['path'])
    c.require(done['status']=='complete' and done['sources_unchanged'] is True
        and done['run_id']==previous['run_id'],'Actual native closure required')
    archived=c.scoped(args.closed_latest);latest=c.scoped(selection['latest_weights'])
    c.require(archived.is_file() and c.sha(archived)==c.sha(latest),'Already CLOSED verified latest byte archive required')
    output=c.scoped(args.output_dir);c.require(not output.exists() and output.parent==HERE/'runs','Fresh bounded executor directory required')
    paths={key:output/(previous['run_id']+'_'+name) for key,name in (('native','unused_original.json'),('fresh_runtime','fresh_native.json'),('identity','identity.json'),('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    paths['native']=Path(previous['paths']['native']);argv=list(previous['worker_arguments'])
    for flag,value in (('--resume-exact',archived),('--uniform-full-runtime-proof',paths['fresh_runtime']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=c.set_option(argv,flag,value)
    plan=dict(previous);plan.update(continuation_mode='same_new_run_exact_budget_continuation',original_ready=c.record(paths['native']),source_checkpoint=c.record(archived),resume_step=selection['completed_steps'],worker_arguments=argv,worker_arguments_sha256=c.domain.guards.argument_sha(argv),paths={k:str(v) for k,v in paths.items()},command=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(Path(c.__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'])
    for key in ('child_exit_code','resource_review_required','stop_category','cooperative_stop_reason','cooperative_stop_requested_at_utc','actual_runtime_completion','finished_at_utc','error','owned_launcher_pid','started_at_utc'):plan.pop(key,None)
    c.validate_plan(plan)
    admission=c.current_memory_admission(previous=previous)
    output.mkdir(parents=True);state={**plan,'status':'starting','fresh_memory_admission':admission,'updated_at_utc':c.now(),'fresh_parent_execution_bootstrap':{'script':c.record(__file__),'previous_status':c.record(args.previous_status),'source_native_closed0':True,'full_JSON_history_reloaded_before_admission':False,'scientific_dispatch_body':c.record(c.__file__),'own_full207_validation_remains_in_frozen_worker':True,'actual_capacity_release_review':capacity_review,'resource_guards_unchanged':True}}
    c.write(output/'plan.json',plan);c.write(output/'loop_status.json',state);c.publish_queue(state)
    state['awake_request']=c.domain.guards.prior.keep_awake(True);args.run_id=plan['run_id']
    try:return c.run_window(state,time.monotonic()+args.window_hours*3600,args)
    except BaseException as error:state.update(status='failed',error=repr(error),finished_at_utc=c.now());raise
    finally:state['awake_clear']=c.domain.guards.prior.keep_awake(False);c.write(output/'loop_status.json',state);c.publish_queue(state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-status',type=Path,required=True);p.add_argument('--closed-latest',type=Path,required=True);p.add_argument('--capacity-review',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--window-hours',type=float,default=3);p.add_argument('--poll-seconds',type=float,default=5);p.add_argument('--gpu-window-authorized',action='store_true');sys.exit(main(p.parse_args()))
