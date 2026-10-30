"""Sole-owned optimizer-free fixed350 DEV of an actually recovered diagnostic.

No training update, optimizer construction, sampler/replay advance or winner
promotion. Requires exact NEW ROOT policy/resource/configuration reviews.
"""
import argparse
import copy
import ctypes
from ctypes import wintypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

P=Path(__file__).resolve().parents[2]
S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad')
OWNER='/root/memory_diagnosis'
LEASE=P/'work/gpu_training_owner.json'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest
from bpe_broad_audio_v1 import record
from resource_watchdog_v2 import sample_memory,PROCESS_MEMORY_COUNTERS_EX
from fastmss_owned_process_closure_v2 import identity,observe,verify

spec=importlib.util.spec_from_file_location('DEV_component_guard',P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py')
H=importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
CONFIG=None
WATCH=None


def same(a,b):
    return a==b and digest(a)==digest(b)


def storage():
    H.storage()


def current_lifetime(cap):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    value=PROCESS_MEMORY_COUNTERS_EX();value.cb=ctypes.sizeof(value)
    require(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb), 'Actual owned lifetime counter')
    require(value.PrivateUsage<=cap and value.PeakPagefileUsage<=cap, 'Current AND Windows lifetime private cap')
    return int(value.PrivateUsage),int(value.PeakPagefileUsage)


def check_bindings(config):
    storage()
    for pin in config['source_bindings'].values():
        require(same(record(pin['path']),pin),'Exact reviewed dependency changed')
    require(same(config['source_bindings'][Path(__file__).name],record(__file__)),'Actual parent/worker entrypoint')
    require(same(config['GPU_python'],record(P/'work/nemo-native-env-v2/Scripts/python.exe')),'Exact existing GPU interpreter')
    require(config['production_enabled'] is False and config['diagnostic_only'] is True,'Diagnostic-only parent')


def checked(config_path,review_path):
    c=read_json(config_path);review=read_json(review_path)
    check_bindings(c)
    from finite_BPE9223_DEV_runtime_v2 import resource_proposal
    require(c['kind']=='ASR18_diagnostic_optimizer_free_serial350DEV_worker_configuration_v1'
            and c['worker_mode']=='serial350DEV_optimizer_free'
            and same(c['parent_driver_source'],record(__file__))
            and same(c['resource_policy'],resource_proposal())
            and c['outer_parent_private_limit_bytes']==128_000_000
            and c['all_owned_shims_private_limit_bytes']==64_000_000
            and c['whole_owned_family_private_limit_bytes']==5_642_000_000,
            'Exact separately bounded diagnostic DEV family')
    require(review['kind']=='ROOT_diagnostic_optimizer_free_serial350DEV_execution_configuration_review_v1'
            and review['accepted'] is True and review['reviewer']=='/root'
            and same(review['configuration'],record(config_path))
            and same(review['request'],c['request'])
            and same(review['source_bindings'],c['source_bindings'])
            and same(review['resource_review'],c['ROOT_resource_review'])
            and same(review['base_assets'],c['base_assets']), 'Exact ROOT execution/config review')
    from asr18_finite_BPE9223_serial_DEV_v2 import checked_request,read_bound
    request=read_bound(c['request']);checked_request(request)
    require(same(review['execution_sources'],request['execution_sources']),'Actual parent/runtime/guard source identity')
    return c


def fresh(increment,guard):
    samples=[]
    for _ in range(3):
        guard();value=sample_memory(os.getpid(),'diagnostic_DEV_fresh_prospective')
        s=value['system_memory'];w=value['worker_private_memory']
        passed=value['paired_sampling_within_one_second'] and s['available'] and w['available']
        passed=passed and s['commit_used_bytes']+increment<.90*s['commit_limit_bytes']
        passed=passed and s['commit_available_bytes']-increment>2_000_000_000
        samples.append({'sample':value,'incremental_forecast_bytes':increment,'passed':bool(passed)})
        time.sleep(.05)
    return {'passed':all(value['passed'] for value in samples),'samples':samples}


def stage(folder,name,torch,**extra):
    WATCH.check();WATCH.phase=name
    torch.cuda.synchronize()
    measured,peak=current_lifetime(5_450_000_000)
    require(torch.cuda.max_memory_allocated()<=8_000_000_000,'Unchanged decimal8GB actual CUDA allocated guard')
    value=sample_memory(os.getpid(),name)
    require(value['paired_sampling_within_one_second'] and value['worker_private_memory']['available']
            and value['system_memory']['available'],'Available paired stage telemetry')
    value.update(extra)
    value.update(pid=os.getpid(),worker_private_peak_bytes_lifetime=peak,
        actual_current_private_bytes=measured,at_utc=value['observed_at_utc'],
        cuda={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    with (Path(folder)/'stages.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value)+'\n');stream.flush()
    write_json(Path(folder)/'worker_stage.json',value)
    return value


def admit_stage(folder,name,budget):
    # Global current already includes live family usage. Reserving its entire
    # remaining192MB is conservative and never borrows a negative remainder.
    current_lifetime(5_450_000_000)
    result=fresh(budget+192_000_000,WATCH.check)
    write_json(Path(folder)/(name+'_global_admission.json'),result)
    require(result['passed'],'Fresh prospective complete phase/future-family admission failed')


def worker(config_path,folder):
    global CONFIG,WATCH
    CONFIG=read_json(config_path)
    folder=Path(folder).resolve()
    require(folder.is_relative_to(E.resolve()) and folder.is_dir(),'Actual owner-created E child namespace')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='0','Sole owned GPU0 worker')
    write_json(folder/'entrypoint_boot.json',{'producer':identity(),'entrypoint_source':record(__file__),'configuration':record(config_path),'request':CONFIG['request']})
    WATCH=H.Watch(folder,5_450_000_000,worker=True)
    code=1
    try:
        check_bindings(CONFIG)
        from finite_BPE9223_DEV_runtime_v2 import run
        run(config_path,folder,parent_source=record(__file__),check_bindings=check_bindings,
            storage=storage,stage=stage,admit_stage=admit_stage)
        current_lifetime(5_450_000_000)
        code=0
    finally:
        closure=WATCH.close(code)
        write_json(folder/'native_guard_completion.json',{'pid':os.getpid(),'actual_exit_code':code,
            'monitor_clean':True,'monitor_joined':True,'guard_source':record(__file__),
            'observed':{'samples':closure['samples'],'peak_private_bytes':closure['peak_private_bytes'],
                        'monitor_error':closure['error']},'watchdog_implementation':record(H.__file__)})


def reconcile_entrypoint(folder,child,known):
    """Even an already-closed launcher cannot hide its recorded live native."""
    if child is None:return True
    try:
        import psutil
        entry=read_json(Path(folder)/'entrypoint_boot.json');native=entry['producer']
        config_path=Path(folder).parent/'actual_worker_configuration.json'
        config=read_json(config_path)
        require(same(entry['entrypoint_source'],record(__file__))
                and same(entry['configuration'],record(config_path)) and same(entry['request'],config['request'])
                and (native['pid']==child.pid or native['ppid']==child.pid),'Exact independent early entrypoint provenance')
        known[(native['pid'],native['create_time'])]=native
        if H.alive(native):
            for process in [psutil.Process(native['pid']),*psutil.Process(native['pid']).children(recursive=True)]:
                try:value=identity(process.pid);known[(value['pid'],value['create_time'])]=value
                except psutil.NoSuchProcess:pass
        return True
    except BaseException as error:
        write_json(Path(folder).parent/'unresolved_early_owned_identity.json',
            {'error':repr(error),'source':record(__file__),'launcher_PID':child.pid,
             'captured_identities':list(known.values()),'idle_lease_release_permitted':False})
        return False


def terminate_and_wait(known,child):
    import psutil
    for value in reversed(list(known.values())):
        if H.alive(value):
            try: psutil.Process(value['pid']).terminate()
            except psutil.NoSuchProcess: pass
    if child is not None:
        if child.poll() is None: child.terminate()
        child.wait(timeout=10)
    deadline=time.monotonic()+10
    while any(H.alive(value) for value in known.values()) and time.monotonic()<deadline:
        time.sleep(.02)
    require(not any(H.alive(value) for value in known.values()),'All exact owned native/shim identities dead before lease release')


def parent(config_path,review_path,folder):
    c=checked(config_path,review_path)
    folder=Path(folder).resolve()
    require(folder.is_relative_to(E.resolve()) and not folder.exists(),'Fresh diagnostic E namespace')
    folder.mkdir();watch=H.Watch(folder,128_000_000)
    prior=read_json(LEASE)
    require(prior['owner']==OWNER and prior['actual_GPU_running'] is False
            and not prior.get('actual_owned_worker_pid'),'Actual idle sole lease before DEV')
    child=None;known={};native=None;launcher=None;runtime_verified=False;output=None;code=1
    try:
        admission=fresh(c['whole_owned_family_private_limit_bytes'],watch.check)
        write_json(folder/'whole_family_prelaunch.json',admission)
        require(admission['passed'],'Fresh5.642GB whole-family admission failed, no unchanged retry')
        output=folder/'DEV_worker';output.mkdir()
        child_config=copy.deepcopy(c)
        child_config.update(ROOT_execution_review=record(review_path),owned_worker_admission_path=str(output/'owned_worker_admission.json'))
        config=folder/'actual_worker_configuration.json';write_json(config,child_config)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
        require(not env.get('DISABLE_ADDMM_CUDA_LT') and not env.get('TORCH_CUDNN_V8_API_DISABLED'),'No failed backend variants')
        began=time.monotonic();family_peak=0;shim_peak=0
        with (output/'worker.log').open('xb') as log:
            child=subprocess.Popen([c['GPU_python']['path'],'-X','utf8','-B',str(Path(__file__).resolve()),
                '--config',str(config),'--worker','--folder',str(output)],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            launcher=identity(child.pid)
            known[(launcher['pid'],launcher['create_time'])]=launcher
            while child.poll() is None:
                watch.check();current_lifetime(128_000_000)
                require(time.monotonic()-began<c['maximum_GPU_child_wall_seconds'],'Finite serial350 wall ceiling')
                import psutil
                try:
                    for process in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                        try: value=identity(process.pid);known[(value['pid'],value['create_time'])]=value
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                if native is None and (output/'entrypoint_boot.json').exists():
                    entry=read_json(output/'entrypoint_boot.json');native=entry['producer']
                    require(same(entry['entrypoint_source'],record(__file__)) and same(entry['configuration'],record(config))
                            and same(entry['request'],c['request']) and H.alive(native)
                            and (native['pid']==child.pid or native['ppid']==child.pid)
                            and (native['pid'],native['create_time']) in known,'Captured early native entrypoint identity/source')
                    # Native startup is covered by5.45GB from first recognition,
                    # never charged to the separate ALLshim64MB ceiling.
                if native is not None and not runtime_verified and (output/'worker_boot_identity.json').exists():
                    boot=read_json(output/'worker_boot_identity.json')
                    require(same(boot['producer'],native)
                            and same(boot['source'],c['source_bindings']['finite_BPE9223_DEV_runtime_v2.py'])
                            and same(boot['request'],c['request']) and H.alive(native),
                            'Runtime boot MUST match already captured native before lease/admission')
                    runtime_verified=True
                    lease=copy.deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],
                        actual_owned_launcher_pid=child.pid,actual_training_optimizer_running=False,status='diagnostic_optimizer_free_serial350DEV')
                    write_json(LEASE,lease)
                    write_json(output/'owned_worker_admission.json',{'kind':'ASR18_sole_owned_optimizer_free_serialDEV_worker_admission_v1',
                        'owner':OWNER,'producer':native,'request':c['request'],'sole_owned_GPU_child':True,
                        'actual_training_optimizer_running':False,'optimizer_objects_constructed_or_loaded_in_DEV_worker':False,
                        'worker_source':c['source_bindings']['finite_BPE9223_DEV_runtime_v2.py'],'parent_source':record(__file__),
                        'ROOT_execution_review':record(review_path),'ROOT_resource_review':c['ROOT_resource_review'],
                        'whole_worker_fresh_three_sample_admission_passed':True,'whole_worker_forecast_bytes':5_450_000_000,
                        'parent_whole_family_admission':record(folder/'whole_family_prelaunch.json')})
                    print(json.dumps({'actual_optimizer_free_DEV_PID':native['pid'],'launcher_PID':child.pid,'folder':str(output)}),flush=True)
                total=sample_memory(os.getpid(),'diagnostic_DEV_owned_family')['worker_private_memory']['private_bytes'];shims=0
                for value in known.values():
                    if not H.alive(value):continue
                    sample=sample_memory(value['pid'],'diagnostic_DEV_actual_known_child')['worker_private_memory']
                    if not sample['available'] and not H.alive(value):continue
                    require(sample['available'],'Available exact owned family private')
                    total+=sample['private_bytes']
                    if native is None or value['pid']!=native['pid']:shims+=sample['private_bytes']
                    else:require(sample['private_bytes']<=5_450_000_000,'Actual native whole-worker private cap')
                require(shims<=64_000_000 and total<=5_642_000_000,'Whole native+parent+ALL shims cap')
                family_peak=max(family_peak,total);shim_peak=max(shim_peak,shims)
                time.sleep(.05)
            actual=child.wait();deadline=time.monotonic()+2
            while any(H.alive(value) for value in known.values()) and time.monotonic()<deadline:
                watch.check();time.sleep(.02)
            require(actual==0 and native is not None and runtime_verified and not any(H.alive(value) for value in known.values()),
                    'Real wait0 and every captured native/shim identity actually dead')
        worker_report=record(output/'serial350_worker_report.json')
        process0=observe(child,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,report=worker_report,
            worker_source=c['source_bindings']['finite_BPE9223_DEV_runtime_v2.py'],parent_source=record(__file__),guard_source=record(__file__),
            guard_closure=record(output/'native_guard_completion.json'),launcher_identity=launcher,
            parent_guard_observation=watch.check(),output=folder/'actual_owned_DEV_process0.json',guard=watch.check)
        from asr18_finite_BPE9223_serial_DEV_v2 import seal_completed_DEV
        sealed=seal_completed_DEV(c['request'],worker_report,process0,folder/'actual_diagnostic_serial350DEV.json')
        write_json(folder/'report.json',{'kind':'ASR18_actual_diagnostic_optimizer_free_serial350DEV_pipeline_v1',
            'passed':True,'actual_process0':process0,'complete350_DEV':sealed,'diagnostic_only':True,'qualified':False,
            'worker_PID':native['pid'],'whole_owned_family_private_peak_bytes':family_peak,'ALL_shims_private_peak_bytes':shim_peak,
            'wall_seconds':time.monotonic()-began,'actual_native_updates':0,'draws_consumed':0,'production_enabled':False})
        code=0
    except BaseException:
        write_json(folder/'failure.json',{'passed':False,'error':traceback.format_exc(),'known_owned_identities':list(known.values()),
            'native':native,'diagnostic_only':True,'production_enabled':False})
        raise
    finally:
        resolved=reconcile_entrypoint(output,child,known)
        terminate_and_wait(known,child)
        if resolved:
            write_json(LEASE,prior)
        else:
            unresolved=copy.deepcopy(prior)
            unresolved.update(actual_GPU_running='unknown_owned_startup_identity_unresolved',
                actual_owned_worker_pid=None,status='blocked_unresolved_owned_startup_identity',
                unresolved_startup_evidence=str(folder/'unresolved_early_owned_identity.json'))
            write_json(LEASE,unresolved)
        watch.close(code)
        require(resolved,'Early native identity unresolved; no idle lease release')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--ROOT-review')
    parser.add_argument('--folder',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.config,args.folder)
    else:
        require(args.ROOT_review is not None,'Exact ROOT execution review required')
        parent(args.config,args.ROOT_review,args.folder)
