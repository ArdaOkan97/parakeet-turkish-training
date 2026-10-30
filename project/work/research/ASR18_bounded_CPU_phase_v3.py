"""Finite CUDA-masked CPU child with a required live fail-closed watchdog."""
import argparse,json,os,runpy,sys,threading,time,traceback
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import ResourceGuard,sample_memory

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--module');p.add_argument('--script');p.add_argument('--fixture-monitor-error',action='store_true');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    assert ROOT.exists() and json.loads((ROOT/'storage_config.json').read_text())['storage_root']==str(ROOT)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ.get(key)=='1'
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
    folder=Path(a.folder).resolve();assert folder.is_relative_to(ROOT.resolve());folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'boot.json',{'pid':os.getpid(),'ppid':os.getppid()})
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.monotonic();done=threading.Event();observed={'peak_private_bytes':0,'peak_commit_ratio':0.,'samples':0,'last_sample_monotonic':None,'monitor_error':None}
    def fail(reason,**extra):
        try:write_json(folder/'resource_stop.json',{'reason':reason,'observed':dict(observed),**extra})
        finally:os._exit(75)
    def watch():
        try:
            guard=ResourceGuard()
            while not done.is_set():
                if a.fixture_monitor_error:raise RuntimeError('intentional bounded watchdog exception fixture')
                value=sample_memory(os.getpid(),'bounded_CPU_phase_v2')
                worker=value['worker_private_memory'];system=value['system_memory']
                if not(worker['available'] and system['available'] and value['paired_sampling_within_one_second']):raise ValueError('Available fresh paired CPU telemetry required')
                private=worker['private_bytes'];ratio=system['commit_used_bytes']/system['commit_limit_bytes']
                observed['peak_private_bytes']=max(private,observed['peak_private_bytes']);observed['peak_commit_ratio']=max(ratio,observed['peak_commit_ratio'])
                observed['samples']+=1;observed['last_sample_monotonic']=time.monotonic();stop,_=guard.observe(system,worker)
                if stop or private>=2_000_000_000 or time.monotonic()-began>a.seconds or (folder/'stop.json').exists():
                    fail(stop or ('private2GB' if private>=2_000_000_000 else 'cooperative_CPU_stop' if (folder/'stop.json').exists() else 'CPU_timeout'),sample=value)
                write_json(folder/'live.json',value);done.wait(.2)
        except BaseException:
            observed['monitor_error']=traceback.format_exc();fail('CPU_watchdog_exception')
    thread=threading.Thread(target=watch,daemon=True);thread.start();code=0
    # No payload starts under an unobserved/dead monitor, even a tiny fixture.
    while observed['samples']<3:
        if not thread.is_alive() or observed['monitor_error'] is not None or time.monotonic()-began>3:fail('CPU_watchdog_unavailable_before_payload')
        time.sleep(.025)
    try:
        arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
        assert bool(a.module)!=bool(a.script);sys.argv=[a.module or a.script,*arguments]
        if a.script:
            path=Path(a.script).resolve();assert path.is_relative_to(PROJECT) or path.is_relative_to(ROOT)
            sys.path.insert(0,str(path.parent));runpy.run_path(str(path),run_name='__main__')
        else:runpy.run_module(a.module,run_name='__main__',alter_sys=True)
    except SystemExit as exc:code=exc.code or 0
    except BaseException:code=1;raise
    finally:
        done.set();thread.join(timeout=2)
        if thread.is_alive() or observed['monitor_error'] is not None or observed['samples']<3 or observed['last_sample_monotonic'] is None or time.monotonic()-observed['last_sample_monotonic']>=2:
            fail('CPU_watchdog_invalid_at_closure')
        write_json(folder/'closure.json',{'pid':os.getpid(),'actual_exit_code':code,'observed':observed,'wall_seconds':time.monotonic()-began,
            'CPU_private_cap_bytes':2_000_000_000,'CUDA_masked':True,'monitor_joined':True,'monitor_clean':True,'valid_samples_before_payload_required':3,'thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}})
    raise SystemExit(code)

if __name__=='__main__':main()
