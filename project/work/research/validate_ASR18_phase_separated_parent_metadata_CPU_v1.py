"""Source/pure-metadata finite separation controls; no numerical/audio/GPU."""
import argparse,ast,copy,importlib.util,json,os,random,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json,read_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from bpe_final_DATA_schedule_context_v1 import ScheduleContext
from asr18_checkpoint_next_draw_v1 import next_draw_packet
from asr18_deferred_GPU_recovery_v1 import verified_requests
from fastmss_row_serial_cpu_v5 import environment,resource_guard
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def function(path,cls,name):
    tree=ast.parse(path.read_bytes())
    obj=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls) if cls else tree
    return next(n for n in obj.body if isinstance(n,ast.FunctionDef) and n.name==name)


def main(folder):
    environment();require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(),'Fresh E-only controls');folder.mkdir()
    sample=resource_guard();system=sample['system_memory'];budget=128_000_000
    require(system['commit_used_bytes']+budget<.9*system['commit_limit_bytes'] and system['commit_available_bytes']-budget>2_000_000_000,'Fresh128MB metadata forecast')
    write_json(folder/'fresh128MB_admission.json',{'passed':True,'incremental_bytes':budget,'sample':sample})
    checks=[]
    def check(name,value):require(value,name);checks.append(name)
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise AssertionError('Expected fail-closed metadata rejection: '+name)
    parent=R/'run_ASR18_full_DATA_multiwindow_v8.py';prior=R/'run_ASR18_full_DATA_multiwindow_v7.py'
    for name in ('check','_run','close'):
        check('Independent ParentGuard '+name+' unchanged',ast.dump(function(parent,'ParentGuard',name))==ast.dump(function(prior,'ParentGuard',name)))
    check('Actual native-service check still returns checked parent sample unchanged',ast.dump(function(parent,'GPUService','check'))==ast.dump(function(prior,'GPUService','check')))
    n=function(parent,None,'run_CPU');n=ast.parse(ast.unparse(n).replace('fastmss_owned_process_closure_v2','fastmss_owned_process_closure_v1')).body[0]
    check('Outer CPU actual-wait/drain/resource algorithm unchanged except process2 import',ast.dump(n)==ast.dump(function(prior,None,'run_CPU')))
    spec=importlib.util.spec_from_file_location('metadata_only_parent8',parent);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    check('Parent8 imports without numerical runtime',not any(x in sys.modules for x in ('torch','numpy','scipy','transformers')))
    template=read_json(S/'asr18_full_DATA_multiwindow.disabled.v8.json')
    check('Exact worker path remains a STRING',type(template['worker_driver_source']) is str and template['worker_driver_source']==str(parent.resolve()))
    check('Exact parent source record matches actual new source',template['parent_driver_source']==record(parent))
    check('Template remains unarmed and production disabled',template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['finite_diagnostic_execution_enabled'] is False and template['production_enabled'] is False)
    check('Resident and recovery full budgets remain distinct',template['entire_worker_forecast']['entire_worker_private_forecast_bytes']==5624813440 and template['final_recovery_worker_forecast']['entire_worker_private_forecast_bytes']==3350000000)
    check('Native/write/restore allowances unchanged',template['entire_worker_forecast']['native_private_forecast_bytes']==5560813440 and template['entire_worker_forecast']['checkpoint_write_increment_bytes']==64000000 and template['final_recovery_worker_forecast']['checkpoint_restore_increment_bytes']==350000000)
    config=read_json(A/'full_DATA_two_window_configuration_CPU_actual_03/finite_two_window_configuration_pending_execution_review.json')
    before_python=random.getstate();context=ScheduleContext(config['planning_arguments'],guard=resource_guard)
    try:
        schedule=context.new_schedule();before=copy.deepcopy(schedule.state_dict())
        packet=next_draw_packet(schedule,context,optimizer_updates=0,guard=resource_guard)
        original=read_json(A/'full_DATA_two_window_configuration_CPU_actual_03/original_first_window_plan.json')['updates'][0]
        check('Actual original-source next transaction equals pinned first-plan transaction',packet['native_transaction_sha256']==original['native_transaction_sha256'])
        check('Actual full row/target identities equal source-bound first-plan rows',packet['source_resolved_full_rows_sha256']==digest(original['rows']))
        check('Actual ordered TR/replay contexts and draws equal original plan',digest(packet['contexts'])==digest(original['contexts']) and digest(packet['draws'])==digest(original['draws']))
        check('Both source cursors and replay RNG/frontier remain unchanged',digest(schedule.state_dict())==digest(before))
        check('No actual authoritative draws or replay commits',packet['draws_consumed']==0 and packet['replay_frontier_advanced'] is False)
        check('Provisional packet reads no audio and applies no augmentation',packet['audio_or_augmentation_applied'] is False)
        again=next_draw_packet(schedule,context,optimizer_updates=0,guard=resource_guard)
        check('Fresh repeated provisional packet is semantically exact',digest(again)==digest(packet))
    finally:context.close()
    check('Global Python RNG unchanged by actual source packet',random.getstate()==before_python)
    write_json(folder/'actual_original_source_next_draw_packet.json',packet)
    negative={'kind':'ASR18_deferred_all_checkpoint_GPU_recovery_requests_v1','owner':'/root/memory_diagnosis','production_enabled':False,
        'session':config['session'],'parent_source':record(parent),'expected_global_new_updates':40,
        'native_process_completions':[],'native_window_completions':[],'checkpoints':[]}
    path=folder/'explicitly_negative_missing_actual_native_closures.json';write_json(path,negative)
    reject('Recovery cannot start with missing actual native process0 proofs',lambda:verified_requests(record(path),session=config['session'],parent_source=record(parent),guard=resource_guard))
    negative['expected_global_new_updates']=True;path=folder/'explicitly_negative_boolean_frontier.json';write_json(path,negative)
    reject('Boolean recovery update count cannot pass integer frontier gate',lambda:verified_requests(record(path),session=config['session'],parent_source=record(parent),guard=resource_guard))
    check('No audio/frontend/ML runtime imported after actual metadata controls',not any(x in sys.modules for x in ('torch','numpy','scipy','transformers','av','soundfile','pyarrow')))
    result={'kind':'ASR18_phase_separated_parent_and_original_nextdraw_metadata_controls_v1','passed':True,'check_count':len(checks),'checks':checks,
        'source':record(__file__),'parent':record(parent),'template':record(S/'asr18_full_DATA_multiwindow.disabled.v8.json'),
        'actual_original_source_next_draw_packet':record(folder/'actual_original_source_next_draw_packet.json'),
        'actual_GPU_or_native_or_original_audio_frontend_job_started':False,'shared_feature_cache_changed':False,
        'actual_authoritative_draws_consumed':0,'production_enabled':False,'resource_or_execution_admission_claimed':False}
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'actual_original_nextdraw_semantics_verified':True,'no_GPU_or_native_job':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True,type=Path);args=parser.parse_args();main(args.folder)
