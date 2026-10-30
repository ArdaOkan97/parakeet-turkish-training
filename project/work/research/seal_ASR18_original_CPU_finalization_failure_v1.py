"""Read-only exact failed-run evidence; no repair, array load, or deletion."""
import datetime,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_owned_process_closure_v2 import verify
from fastmss_row_serial_cpu_v5 import environment,resource_guard
import psutil
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def main():
    environment();sample=resource_guard();memory=sample['system_memory'];budget=128_000_000
    require(memory['commit_used_bytes']+budget<.9*memory['commit_limit_bytes'] and memory['commit_available_bytes']-budget>2_000_000_000,'Fresh128MB evidence admission')
    run=A/'full_DATA_two_window_native_02';cpu=run/'CPU_window_0';stages=cpu/'payload'
    out=run/'actual_owned_CLOSED1_v1.json';require(not out.exists(),'Immutable failed trial proof')
    failure=read_json(run/'failure.json');closure=read_json(cpu/'guard/closure.json')
    require(failure['actual_completed_positive_native_updates']==0 and failure['actual_native_progress'] is None
        and closure['actual_exit_code']==1 and closure['pid']==30628 and closure['monitor_clean'] is True
        and closure['monitor_joined'] is True and closure['observed']['monitor_error'] is None,
        'Observed failed parent/clean failed controller, no native progress')
    log=(cpu/'worker.log').read_text()
    require("ModuleNotFoundError: No module named 'numpy'" in log and '_validate_update' in log and 'finalize_window' in log,
        'Exact observed finalization failure required')
    known={};results=[];peaks=[];stage_wall=0.;frontend_peak=0;reference_peak=0;global_peak=0.
    def remember(value):known[(value['pid'],value['create_time'])]=value
    dispatch=read_json(A/'current_owned_diagnostic_dispatch.json');remember(dispatch['parent'])
    family=read_json(cpu/'owned_CPU_family_live.json')
    for item in family['measured_processes']:remember(item['identity'])
    for i in range(20):
        resource_guard();path=stages/('completed_update_%02d.json'%i);value=read_json(path)
        require(value['passed'] is True and value['update_index']==i and value['actual_optimizer_updates']==0
            and value['sampler_journal_frontier_advanced'] is False,'Every exact prepared-not-committed update')
        for role in ('reference','frontend'):
            item=value[role];proof=read_json(item['actual0']['path'])
            require(record(item['actual0']['path'])==item['actual0'],'Exact actual0 child signature')
            verify(proof,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',guard=resource_guard)
            for key in ('producer','launcher','observer'):remember(proof[key])
            native=read_json(proof['guard_closure']['path']);peak=native['observed']['peak_private_bytes']
            stage_wall+=native['wall_seconds'];global_peak=max(global_peak,native['observed']['peak_commit_ratio'])
            if role=='frontend':frontend_peak=max(frontend_peak,peak)
            else:reference_peak=max(reference_peak,peak)
            peaks.append(item['CPU_stage_admission_and_family']['actual_controller_leaf_shim_private_peak_bytes'])
        results.append(record(path))
    dead=[]
    for key,value in sorted(known.items()):
        try:process=psutil.Process(value['pid']);same=process.create_time()==value['create_time'];alive=same and process.is_running()
        except psutil.NoSuchProcess:alive=False;same=False
        require(not alive,'Captured owned identity still alive; do not claim retirement')
        dead.append({'identity':value,'exact_identity_alive':False,'PID_reuse_is_not_old_identity':not same})
    owner_path=Path(r'E:\gpu_training_scratchpad\temp\ASR18_CPU_feature_windows_v1\payload\owner.json')
    owner=read_json(owner_path);folder=owner_path.parent
    files=sorted(folder.iterdir());names=owner['owned_names']
    require(all(path.is_file() and path.resolve().parent==folder.resolve() and path.name in names for path in files),
        'Preserve any unknown owner file rather than infer safe inventory')
    inventory=[]
    for path in files:
        resource_guard();pin=record(path);pin['mtime_ns']=path.stat().st_mtime_ns;inventory.append(pin)
    require(len(files)==681 and sum(x['bytes'] for x in inventory)<536870912 and not (folder/'manifest.json').exists(),
        'Exact unfinished final manifest, bounded retained original feature files')
    inv=run/'preserved_original_feature_inventory_after_failure_v1.json'
    write_json(inv,{'kind':'ASR18_readonly_failed_original_feature_inventory_v1','owner':record(owner_path),
        'actual_files':inventory,'total_bytes':sum(x['bytes'] for x in inventory),'metadata_or_array_mutation_performed':False})
    identity_path=run/'observed_captured_owned_identities_dead_v1.json'
    write_json(identity_path,{'passed':True,'exact_owned_identities':dead,'global_process_enumeration':False})
    result={'kind':'ASR18_actual_phase_separated_original_trial_failed_TOOL1_v1','owner':'/root/memory_diagnosis',
        'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owned_session':18107,
        'native_parent_PID':14292,'actual_exit_code':1,'actual_tool_chunk_id':'69282c',
        'launch':record(run/'launch.json'),'configuration':dispatch['configuration'],'source_manifest':record(A/'concrete_multiwindow_caller_source_review_manifest_v9.json'),
        'fresh_full_forecast_PASS':record(A/'full_DATA_two_window_prelaunch_full_forecast_03.json'),
        'parent_failure':record(run/'failure.json'),'CPU_log':record(cpu/'worker.log'),'CPU_controller_guard':record(cpu/'guard/closure.json'),
        'CPU_controller_actual_exit_code':1,'CPU_controller_peak_private_bytes':closure['observed']['peak_private_bytes'],
        'CPU_controller_wall_seconds':closure['wall_seconds'],'normal_prepared_reference_children':20,'normal_prepared_frontend_children':20,
        'completed_prepared_not_committed_updates':results,'maximum_reference_leaf_private_bytes':reference_peak,
        'maximum_frontend_leaf_private_bytes':frontend_peak,'maximum_inner_controller_leaf_shim_private_bytes':max(peaks),
        'inner_peak_excludes_additional_outer_shim':True,'sum_actual_child_stage_wall_seconds':stage_wall,
        'maximum_sampled_child_system_commit_ratio':global_peak,'normal_full_CPU_window_completion_claimed':False,
        'exact_error':"ModuleNotFoundError: No module named 'numpy' during sink3.finalize_window->store.arrays",
        'failure_is_not_resource_guard_stop':True,'all_captured_owned_identities_observed_dead':record(identity_path),
        'retained_feature_inventory':record(inv),'retained_files':len(files),'retained_bytes':sum(x['bytes'] for x in inventory),
        'final_manifest_published':False,'GPU_launched':False,'actual_Adam_updates':0,'actual_authoritative_draws_consumed':0,
        'unmodified_sources_rechecked':True,'retry_performed':False,'production_enabled':False,'source':record(__file__)}
    for pin in read_json(result['source_manifest']['path'])['bindings']:require(record(pin['path'])==pin,'Frozen source changed during actual failed run')
    write_json(out,result);print({'actual_failure':record(out),'completed_normal_CPU_children':40,'GPU_or_Adam':0,'retained_files':len(files),'retained_bytes':result['retained_bytes']},flush=True)


if __name__=='__main__':main()
