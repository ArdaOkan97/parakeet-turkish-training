"""CPU-only first-window proof; no GPUService, model or native worker path.

run_CPU is copied verbatim from reviewed caller9. Here __file__ truthfully
binds this actual entrypoint; original callable provenance is separately pinned.
"""
import argparse,copy,gc,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];SOURCE=PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(SOURCE))
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis';LEASE=PROJECT/'work/gpu_training_owner.json'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from resource_watchdog_v2 import sample_memory
spec=importlib.util.spec_from_file_location('CPU_parent_watch',PROJECT/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def fresh_admit(increment):
    checks=[]
    for _ in range(3):
        v=sample_memory(os.getpid(),'CPU_only_full_family_prospective');s=v['system_memory'];extra=increment+64_000_000
        ok=v['paired_sampling_within_one_second'] and s['available'] and v['worker_private_memory']['available']
        ok=ok and s['commit_used_bytes']+extra<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-extra>2_000_000_000
        checks.append({'passed':ok,'sample':v,'child_family_increment':increment,'additional_full_parent_cap':64_000_000});time.sleep(.05)
    return {'passed':all(v['passed'] for v in checks),'samples':checks,'whole_owned_CPU_forecast':increment+64_000_000}

def run_CPU(config, script, arguments, folder, *, guard, report_relative='report.json'):
    """One actual bounded CPU controller/cleanup child, source-bound Popen proof."""
    from fastmss_owned_process_closure_v2 import identity, observe
    folder.mkdir(exist_ok=False); monitor = folder/'guard'; log = folder/'worker.log'; phase_began=time.monotonic(); known={}; family_peak=0; family_samples=0; outer_seen=False
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = '-1'
    require(record(config['CPU_python']['path']) == config['CPU_python']
            and record(config['CPU_guard_source']['path']) == config['CPU_guard_source'], 'Pinned CPU interpreter/guard before launch')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '1'
    admission = fresh_admit(config['CPU_owned_allowance_bytes']); write_json(folder/'admission.json', admission)
    require(admission['passed'], 'Fresh CPU allowance does not fit; preserve, no DATA hold')
    with log.open('xb') as stream:
        child = subprocess.Popen([config['CPU_python']['path'], '-X', 'utf8', '-B', config['CPU_guard_source']['path'],
            '--folder', str(monitor), '--seconds', str(config['maximum_CPU_phase_seconds']), '--script', str(script), '--', *arguments],
            env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        launcher = identity(child.pid); began = time.monotonic()
        known[(launcher['pid'],launcher['create_time'])] = launcher
        exit_transition = None
        try:
            from asr18_owned_family_observation_v4 import owned_family_sample
            from asr18_outer_CPU_exit_transition_v1 import wait_actual_CPU_family_exit
            while child.poll() is None:
                guard(); require(time.monotonic()-began < config['maximum_CPU_phase_seconds']+10, 'Finite CPU controller timeout')
                try:family=owned_family_sample(child,launcher,known=known,guard=guard)
                except ProcessLookupError:
                    # A launcher syscall can race normal exit before poll sees0.
                    # Drain captured children under their own unchanged cap;
                    # the outer native parent retains its separate ParentGuard.
                    exit_transition = wait_actual_CPU_family_exit(child,launcher,known,guard=guard,
                        CPU_family_private_allowance_bytes=config['CPU_owned_allowance_bytes'])
                    break
                family_samples+=1;family_peak=max(family_peak,family['private_bytes_sum'])
                outer_seen=outer_seen or family['outer_launcher_included']
                write_json(folder/'owned_CPU_family_live.json',family)
                require(family['private_bytes_sum']<=config['CPU_owned_allowance_bytes'],'Whole CPU family private allowance exceeded, including outer launcher')
                time.sleep(.2)
            require(child.wait() == 0, 'Actual CPU phase failed; artifacts preserved, no unchanged retry')
            if exit_transition is None:
                exit_transition = wait_actual_CPU_family_exit(child,launcher,known,guard=guard,
                    CPU_family_private_allowance_bytes=config['CPU_owned_allowance_bytes'])
            family_peak=max(family_peak,exit_transition['captured_CPU_family_private_peak_bytes'])
            write_json(folder/'owned_CPU_family_exit_transition.json',exit_transition)
        except BaseException:
            from asr18_owned_family_observation_v4 import terminate_known_family
            monitor.mkdir(exist_ok=True); write_json(monitor/'stop.json', {'reason': 'owned_parent_failed'})
            if child.poll() is None:
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            # Even an exited outer launcher can leave captured descendants.
            # Observe their termination before considering this family retired.
            terminate_known_family(child,launcher,known)
            raise
    require(family_samples>=1 and outer_seen,'Actual outer launcher+CPU-family telemetry required; no guessed zero overhead')
    write_json(folder/'owned_CPU_family_completion.json',{'passed':True,'measured_family_peak_private_bytes':family_peak,
        'family_samples':family_samples,'reviewed_entire_CPU_allowance_bytes':config['CPU_owned_allowance_bytes'],
        'inner_controller_family_allowance_bytes':config['CPU_inner_owned_allowance_bytes'],
        'additional_outer_launcher_reserve_bytes':config['CPU_outer_controller_launcher_reserve_bytes'],
        'outer_launcher_included':True,'actual_parent_phase_wall_seconds':time.monotonic()-phase_began,
        'launcher_disappearance_or_normal_exit_drain':exit_transition,
        'only_actual_owned_ancestry_observed':True})
    report = record(folder/report_relative)
    return observe(child, kind='ASR18_owned_CPU_process_completion_v1', owner=OWNER, report=report,
        worker_source=record(script), parent_source=record(__file__), guard_source=config['CPU_guard_source'],
        guard_closure=record(monitor/'closure.json'), launcher_identity=launcher,
        parent_guard_observation=sample_memory(os.getpid(), 'CPU_actual_wait_observer'), output=folder/'actual_owned_process0.json', guard=guard)

def main(config_path,review_path,folder):
    H.storage();require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','No GPU access in CPU-only outer parent')
    require(Path(sys.argv[0]).resolve()==Path(__file__).resolve(),'Actual outer entrypoint source/argv')
    c=read_json(config_path);review=read_json(review_path)
    require(c['kind']=='ASR18_CPU6_first_window_only_configuration_v1' and c['GPU_execution_enabled'] is False
        and c['actual_native_updates']==0 and c['production_enabled'] is False,'CPU-only configuration, never native budget bypass')
    require(review['accepted'] is True and review['reviewer']=='/root' and review['configuration']==record(config_path)
        and review['outer_parent_source']==record(__file__) and review['resource_policy']==c['CPU_only_resource_policy'],'Exact CPU-only execution review')
    for pin in c['source_bindings'].values():require(record(pin['path'])==pin,'Frozen thin source changed')
    original=read_json(c['original_caller_configuration']['path']);require(record(c['original_caller_configuration']['path'])==c['original_caller_configuration'],'Pinned original caller config')
    manifest=read_json(original['source_manifest']['path']);require(record(original['source_manifest']['path'])==original['source_manifest'],'Original caller manifest unchanged')
    for pin in manifest['bindings']:require(record(pin['path'])==pin,'Unchanged reviewed original224 source')
    require(c['CPU_owned_allowance_bytes']==1_934_217_728 and c['CPU_only_resource_policy']['whole_CPU_job_allowance_bytes']==1_998_217_728,'Exact64MBparent+originalCPUchildfamily')
    cpu_review=read_json(c['actual_ROOT_CPU_resource_review']['path']);require(record(c['actual_ROOT_CPU_resource_review']['path'])==c['actual_ROOT_CPU_resource_review']
        and cpu_review['accepted'] is True and cpu_review['reviewer']=='/root' and cpu_review['scope']=='CPU_preparation_only'
        and cpu_review['outer_parent_source']==record(__file__) and cpu_review['CPU_source_manifest']==manifest['CPU_owner_bundle'],'Actual ROOT CPU resource rebound to true entrypoint')
    require(cpu_review['resource_review']==read_json(original['actual_ROOT_CPU_resource_review']['path'])['resource_review']
        and cpu_review['raw_lifetime_review']==original['raw_lifetime_review'],'No leaf/controller/shim/raw/workspace/runtime numerical budget changes')
    require(not folder.exists() and folder.resolve().is_relative_to(ROOT.resolve()),'New E-only CPU proof folder');folder.mkdir()
    watch=H.Watch(folder,64_000_000);code=1;began=time.monotonic()
    try:
        lease=read_json(LEASE);require(lease['owner']==OWNER and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Exclusive lease stays idle')
        admission=fresh_admit(c['CPU_owned_allowance_bytes']);write_json(folder/'outer_fresh_CPU_only_admission.json',admission);require(admission['passed'],'Full CPU-only job fresh admission')
        from fastmss_phase_aware_plan_bridge_v5 import read_bound,bridge_plan
        from fastmss_row_serial_cpu_v6 import verify_contract
        from asr18_reprepared_array_equivalence_v1 import validate_retirement,verify_repreparation
        session=read_bound(c['session'],watch.check);contract=verify_contract(session['CPU_preparation_contract'],session,guard=watch.check)
        require(contract['outer_parent_source']==record(__file__) and contract['actual_ROOT_CPU_resource_review']==c['actual_ROOT_CPU_resource_review']
            and contract['resource_review']==cpu_review['resource_review'] and contract['raw_lifetime_review']==cpu_review['raw_lifetime_review'],'Exact new contract/source/resource, not restamped old authority')
        compare=c['first_window_repreparation_comparison'];validate_retirement(compare['prior_owned_retirement_actual0'],prior_failure=compare['prior_failed_trial'],guard=watch.check)
        plan=read_bound(c['first_plan'],watch.check);require(c['first_plan']['sha256']=='6bec4cb6781805def767d0cd4b1e98363845941d6ee96ed4aa18623014e2b36b'
            and len(plan['updates'])==20 and session['initial_optimizer_updates']==0,'Unchanged exact original first20 provisional plan')
        bridge=bridge_plan(c['first_plan'],DATA_evidence=c['planning_arguments']['DATA_evidence'],session_binding=c['session'],window_index=0,prior_window_boundary=None,verify_prior_boundary=None,guard=watch.check)
        bridge_path=folder/'bridge_0.json';write_json(bridge_path,bridge);arg=folder/'bridge_0_binding.json';write_json(arg,record(bridge_path))
        actual=run_CPU(c,c['CPU_controller_source']['path'],['--bridge',str(arg),'--output',str(folder/'CPU_window_0/payload')],folder/'CPU_window_0',guard=watch.check,report_relative='payload/report.json')
        prepared=read_json(folder/'CPU_window_0/payload/report.json');require(prepared['passed'] is True and prepared['all_children_actually_closed0'] is True,'Actual CPU6 successful finalization before comparison')
        proof=verify_repreparation(prior_failure=compare['prior_failed_trial'],prior_retirement=compare['prior_owned_retirement_actual0'],new_CPU_actual=actual,
            new_owner=prepared['owner'],new_manifest=prepared['manifest'],coordinator_plan=c['first_plan'],output=folder/'actual_640_array_320_row_20_payload_equivalence.json',guard=watch.check)
        require('torch' not in sys.modules and 'numpy' not in sys.modules,'Parent remains stdlib-only, no model or numerical runtime')
        lease=read_json(LEASE);require(lease['owner']==OWNER and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'No GPU/native/Adam/draw path executed')
        write_json(folder/'report.json',{'kind':'ASR18_CPU6_first_window_only_actual_v1','passed':True,'producer':H.pid_identity(),'actual_outer_argv':list(sys.argv),
            'actual_outer_entrypoint_source':record(__file__),'copied_run_CPU_implementation_provenance':c['copied_run_CPU_provenance'],
            'CPU_actual_process0':actual,'exact_equivalence':proof,'owner':prepared['owner'],'manifest':prepared['manifest'],
            'actual_GPU_or_Adam_or_authoritative_draws':0,'GPU_admission_claimed':False,'production_enabled':False,
            'parent_cap_bytes':64_000_000,'child_family_cap_bytes':1_934_217_728,'whole_owned_CPU_job_cap_bytes':1_998_217_728,
            'wall_seconds':time.monotonic()-began,'prepared_features_retained_uncommitted':True});code=0
    finally:watch.close(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True);args=p.parse_args()
    main(args.config,args.ROOT_review,Path(args.folder))
