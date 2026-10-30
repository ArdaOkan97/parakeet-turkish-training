"""Inactive autonomous row-serial CPU window; final DATA remains mandatory.

No native training job or source-state commit. The parent native controller
owns any resident-worker combined admission, checkpoint boundary and cleanup.
"""
import argparse
from copy import deepcopy
import json,os,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import fastmss_row_serial_cpu_v1 as core
import fastmss_owned_process_closure_v1 as process

def family_guard(child,launcher,monitor,contract,guard):
    """Read only this controller and its exact owned shim/native PIDs."""
    from resource_watchdog_v2 import sample_memory
    own=guard();review=contract['resource_review']
    total=own['worker_private_memory']['private_bytes']
    core.require(total<=review['controller_private_reserve_bytes'],'Controller exceeded reviewed whole-family overhead reserve')
    pids={os.getpid()};native_pid=None
    boot=monitor/'boot.json'
    if boot.exists():
        value=json.loads(boot.read_bytes());native_pid=value['pid']
        if child.poll() is None:
            import psutil
            try:
                native=process.identity(native_pid)
                core.require(native_pid==launcher['pid'] or native['ppid']==launcher['pid'],'Actual owned native/launcher relation')
            except psutil.NoSuchProcess:
                # A Python launcher may reap its native child just before its
                # own handle signals. This is not yet an observed Popen exit.
                footer=monitor/'closure.json'
                core.require(footer.exists() and json.loads(footer.read_bytes())['pid']==native_pid,'Native disappeared without guard footer')
    for pid in (launcher['pid'],native_pid):
        if pid is None or pid in pids:continue
        value=sample_memory(pid,'row_serial_exact_owned_CPU_family')
        if not value['worker_private_memory']['available']:
            footer=monitor/'closure.json'
            reaped_native=pid==native_pid and footer.exists() and json.loads(footer.read_bytes())['pid']==pid
            core.require(child.poll() is not None or reaped_native,'Unavailable live owned CPU PID telemetry');continue
        core.require(value['paired_sampling_within_one_second'] is True,'Fresh owned family telemetry')
        private=value['worker_private_memory']['private_bytes'];pids.add(pid);total+=private
        if pid==launcher['pid'] and native_pid is not None and native_pid!=pid:
            core.require(private<=review['shim_private_reserve_bytes'],'Owned Python shim exceeded reserve')
    core.require(total<=review['whole_owned_private_allowance_bytes'],'Whole owned CPU family private allowance exceeded')
    return {'private_bytes':total,'exact_owned_PIDs':sorted(pids)}

def run_child(script,request,report,folder,contract,stage,guard):
    """The actual handle owner observes closure; no fabricated tool receipt."""
    admission=core.admit(contract['resource_review'][stage+'_incremental_bytes'],guard)
    monitor=folder/'guard';log=folder/'worker.log';folder.mkdir(parents=True,exist_ok=False)
    arguments=[contract['python']['path'],'-B',contract['sources']['guard']['path'],
        '--folder',str(monitor),'--seconds',str(contract['maximum_child_seconds']),
        '--script',str(script),'--','--request',request['path'],'--output',str(report)]
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='-1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    with log.open('xb') as stream:
        child=subprocess.Popen(arguments,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        launcher=process.identity(child.pid);began=time.monotonic();family_peak=0;family_samples=0
        try:
            while child.poll() is None:
                family=family_guard(child,launcher,monitor,contract,guard);family_peak=max(family_peak,family['private_bytes']);family_samples+=1
                if time.monotonic()-began>contract['maximum_child_seconds']+5:
                    # Stop only this exact owned child, never user apps or DATA.
                    core.write_new(monitor/'stop.json',{'reason':'owned_controller_timeout'})
                    raise TimeoutError('Bounded owned child timeout; no automatic retry')
                time.sleep(.2)
            code=child.wait(timeout=5)
        except BaseException:
            if child.poll() is None:
                monitor.mkdir(parents=True,exist_ok=True)
                stop=monitor/'stop.json'
                if not stop.exists():core.write_new(stop,{'reason':'owned_parent_guard_failed'})
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Explicit actual owned PID identity; no process-name kill.
                    import psutil
                    target=psutil.Process(launcher['pid'])
                    if abs(target.create_time()-launcher['create_time'])<=.001:
                        for descendant in target.children(recursive=True):descendant.terminate()
                        target.terminate()
                    child.wait(timeout=5)
            raise
    core.require(type(code) is int and code==0,'Actual owned CPU child failed; preserved artifacts, no retry/exclusion')
    closure=core.record(monitor/'closure.json',guard);report_binding=core.record(report,guard)
    observed=process.observe(child,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',
        report=report_binding,worker_source=core.record(script,guard),parent_source=contract['sources']['controller'],
        guard_source=contract['sources']['guard'],guard_closure=closure,launcher_identity=launcher,
        parent_guard_observation=guard(),output=folder/'actual_owned_process0.json',guard=guard)
    admission={**admission,'whole_owned_private_allowance_bytes':contract['resource_review']['whole_owned_private_allowance_bytes'],
        'actual_controller_leaf_shim_private_peak_bytes':family_peak,'actual_owned_family_samples':family_samples}
    return child,report_binding,closure,observed,admission

def main(bridge_binding,output,contract_override=None,prior_boundary_verification=None):
    core.environment();guard=core.resource_guard
    planned=core.read_bound(bridge_binding,guard);session=core.read_bound(planned['session'],guard)
    contract_pin=session['CPU_preparation_contract']
    if contract_override is not None:core.require(core.same(contract_override,contract_pin),'CLI contract cannot override session identity')
    contract=core.verify_contract(contract_pin,session,guard=guard)
    # This genuine gate is before output/CACHE creation or a child launch.
    validated=core.bridge.validate_DATA(planned['DATA_evidence'],guard)
    core.require((planned['window_index']==1)==(prior_boundary_verification is not None),'Dynamic prior-boundary evidence only/always for window2')
    def prior_boundary(boundary):
        attestation=core.read_bound(prior_boundary_verification,guard)
        core.require(attestation['kind']=='FastMSS_native_owner_prior_window_CPU_handoff_v1' and attestation['owner']=='/root/memory_diagnosis' and attestation['passed'] is True and attestation['boundary_sha256']==core.digest(boundary) and attestation['actual_all_state_prefix_checkpoint_verified'] is True and core.same(attestation['bridge'],bridge_binding) and core.same(attestation['session'],planned['session']) and core.same(attestation['parent_source'],contract['outer_parent_source']),'Source-bound dynamic exact native owner/bridge/session/coherent prior-window handoff')
        # This cannot authorize idle cleanup. That already happened under the
        # native owner's fresh nonce callback; we reverify its bound outcome.
        if 'resident_native_boundary' in boundary:
            pointer=core.read_bound(boundary['coherent_checkpoint_pointer'],guard)
            process.verify_resident_idle(core.read_bound(boundary['resident_native_boundary'],guard),pointer=pointer,proof_binding=boundary['resident_native_boundary'],guard=guard)
        return True
    rebuilt=core.bridge.bridge_plan(planned['coordinator_plan'],DATA_evidence=planned['DATA_evidence'],
        session_binding=planned['session'],window_index=planned['window_index'],prior_window_boundary=planned['prior_window_boundary'],
        verify_prior_boundary=prior_boundary if planned['window_index'] else None,guard=guard)
    core.require(core.same(rebuilt,planned),'Exact source/session/DATA/ordered plan required')
    # Recheck the actual parent's fresh mutable idle witness throughout CPU
    # preparation, rather than treating an old immutable idle marker as live.
    original_guard=guard
    if planned['prior_window_boundary'] is not None:
        boundary=core.read_bound(planned['prior_window_boundary'],original_guard)
        if 'resident_native_boundary' in boundary:
            idle_pin=boundary['resident_native_boundary'];idle=core.read_bound(idle_pin,original_guard)
            pointer=core.read_bound(boundary['coherent_checkpoint_pointer'],original_guard)
            def guard():
                v=original_guard();process.verify_resident_idle(idle,pointer=pointer,proof_binding=idle_pin,guard=original_guard);return v
    output=core.scoped(output);core.require(not output.is_relative_to(core.sink.CACHE.resolve()),'Durable reports outside transient shared CACHE')
    output.mkdir(parents=True,exist_ok=False)
    owner=core.sink.create_window(bridge_binding,window_id=output.name,frontend=contract['frontend'],guard=guard,
        verify_prior_boundary=prior_boundary if planned['window_index'] else None)
    closures=[];references=[];frontend_proofs=[]
    for index in range(len(planned['updates'])):
        base={'kind':core.REQUEST_KIND,'production_enabled':False,'bridge':bridge_binding,'contract':contract_pin,'update_index':index}
        request_path=output/f'reference_request_{index:02}.json';core.write_new(request_path,base);request=core.record(request_path,guard)
        child,report,monitor,actual,admission=run_child(contract['sources']['reference_worker']['path'],request,
            output/f'references_{index:02}.json',output/f'reference_child_{index:02}',contract,'reference',guard)
        reference_admission=admission
        input_value=core.bridge.bind_verified_update_inputs(bridge_binding,index,independent_PCM_proof=report,actual_closed0=actual,guard=guard)
        input_path=output/f'input_{index:02}.json';core.write_new(input_path,input_value);input_pin=core.record(input_path,guard)
        feature_request={**base,'owner':owner,'input_binding':input_pin,'prior_closures':deepcopy(closures)}
        feature_path=output/f'frontend_request_{index:02}.json';core.write_new(feature_path,feature_request)
        child,feature,monitor,process_proof,admission=run_child(contract['sources']['feature_worker']['path'],core.record(feature_path,guard),
            output/f'frontend_{index:02}.json',output/f'frontend_child_{index:02}',contract,'frontend',guard)
        closed=core.sink.observe_feature_child(child,owner_binding=owner,update_index=index,guard_closure=monitor,
            proof_path=output/f'frontend_closed_{index:02}.json',guard=guard)
        closures.append(closed);references.append({'report':report,'actual0':actual,'input':input_pin,'CPU_stage_admission_and_family':reference_admission})
        frontend_proofs.append({'report':feature,'actual0':process_proof,'feature_closure':closed,'CPU_stage_admission_and_family':admission})
        core.write_new(output/f'completed_update_{index:02}.json',{'passed':True,'update_index':index,'reference':references[-1],'frontend':frontend_proofs[-1],'actual_optimizer_updates':0,'sampler_journal_frontier_advanced':False})
    manifest=core.sink.finalize_window(owner,closures,guard=guard)
    result={'kind':'FastMSS_actual_row_serial_CPU_feature_window_preparation_v1','passed':True,
        'producer':core.identity(),'worker_source':core.record(__file__,guard),'parent_source':contract['outer_parent_source'],
        'bridge':bridge_binding,'contract':contract_pin,'owner':owner,'manifest':manifest,'closures':closures,
        'prior_boundary_verification':prior_boundary_verification,
        'independent_reference_proofs':references,'frontend_proofs':frontend_proofs,
        'actual_optimizer_updates':0,'sampler_journal_frontier_advanced':False,'production_enabled':False,
        'raw_PCM_files_written':0,'maximum_simultaneous_children':1,'actual_outer_exit_observed_in_child':False,
        'all_children_actually_closed0':True,'shared_feature_files_retained_for_native_owner':True,
        'cleanup_owner':'separate actual coherent native callback; intermediate checkpoints retain all future files'}
    core.write_new(output/'report.json',result)
    return result

def binding_argument(value):
    return json.loads(value) if value.lstrip().startswith('{') else json.loads(core.scoped(value).read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bridge',required=True);p.add_argument('--output',required=True);p.add_argument('--contract');p.add_argument('--prior-boundary-verification')
    a=p.parse_args();main(binding_argument(a.bridge),a.output,binding_argument(a.contract) if a.contract else None,binding_argument(a.prior_boundary_verification) if a.prior_boundary_verification else None)
