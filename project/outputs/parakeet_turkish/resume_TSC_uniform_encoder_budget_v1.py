"""One fresh-parent SAME-ID budget window using the frozen native dispatcher.

No scientific worker/config/native token changes. The previous frozen controller
already verified the full JSON history at its actual CLOSED0 budget boundary;
this entry avoids constructing that complete history again before memory admission.
The frozen worker still validates every own-state/history/Adam/sampler boundary.
"""
from pathlib import Path
import argparse,sys,time,json
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
import run_TSC_uniform_encoder_schedule_v1 as frozen
EXPECTED='8a133bf34694412963be89b6b0dd14ca463bcf6a7d358bbd00e46de06c553770'

def main(args):
    c=frozen;c.require(c.sha(c.__file__)==EXPECTED,'Frozen execution source changed')
    c.require(args.gpu_window_authorized and .1<=args.window_hours<=3,'Sole authorized bounded window required')
    previous=c.read(c.scoped(args.previous_status));run=HERE/'runs'/previous['run_id'];selection=c.read(run/'selection.json')
    c.require(previous['kind']==c.CONTROLLER_KIND and previous['controller_sha256']==EXPECTED
        and previous['status']=='train_window_ended' and previous['child_exit_code']==0
        and previous['resource_review_required'] is False and previous['stop_category']=='budget'
        and previous['completed_steps']==selection['completed_steps']
        and selection['completed_steps']<previous['configured_steps']
        and c.record(run/'selection.json')==previous['selection'],'Actual verified CLOSED0 budget source required')
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
    output.mkdir(parents=True);state={**plan,'status':'starting','fresh_memory_admission':admission,'updated_at_utc':c.now(),'fresh_parent_execution_bootstrap':{'script':c.record(__file__),'previous_status':c.record(args.previous_status),'source_native_closed0':True,'full_JSON_history_reloaded_before_admission':False,'scientific_dispatch_body':c.record(c.__file__),'own_full207_validation_remains_in_frozen_worker':True}}
    c.write(output/'plan.json',plan);c.write(output/'loop_status.json',state);c.publish_queue(state)
    state['awake_request']=c.domain.guards.prior.keep_awake(True);args.run_id=plan['run_id']
    try:return c.run_window(state,time.monotonic()+args.window_hours*3600,args)
    except BaseException as error:state.update(status='failed',error=repr(error),finished_at_utc=c.now());raise
    finally:state['awake_clear']=c.domain.guards.prior.keep_awake(False);c.write(output/'loop_status.json',state);c.publish_queue(state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-status',type=Path,required=True);p.add_argument('--closed-latest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--window-hours',type=float,default=3);p.add_argument('--poll-seconds',type=float,default=5);p.add_argument('--gpu-window-authorized',action='store_true');sys.exit(main(p.parse_args()))
