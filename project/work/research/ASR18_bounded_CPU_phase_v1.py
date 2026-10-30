"""Isolated CPU phase runner with E-only artifacts and unchanged native guards."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import ResourceGuard,sample_memory

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--module');p.add_argument('--script');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    assert ROOT.exists() and json.loads((ROOT/'storage_config.json').read_text())['storage_root']==str(ROOT)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
    folder=Path(a.folder).resolve();assert folder.is_relative_to(ROOT.resolve());folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'boot.json',{'pid':os.getpid(),'ppid':os.getppid()})
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started=time.monotonic();done=threading.Event();observed={'peak_private_bytes':0,'peak_commit_ratio':0.,'samples':0}
    def watch():
        guard=ResourceGuard()
        while not done.is_set():
            value=sample_memory(os.getpid(),'bounded_CPU_phase');private=value['worker_private_memory']['private_bytes']
            observed['peak_private_bytes']=max(private,observed['peak_private_bytes'])
            system=value['system_memory'];observed['peak_commit_ratio']=max(observed['peak_commit_ratio'],system['commit_used_bytes']/system['commit_limit_bytes'])
            observed['samples']+=1;stop,_=guard.observe(system,value['worker_private_memory'])
            if stop or private>=2_000_000_000 or time.monotonic()-started>a.seconds or (folder/'stop.json').exists():
                write_json(folder/'resource_stop.json',{'reason':stop or ('private2GB' if private>=2_000_000_000 else 'cooperative_CPU_stop' if (folder/'stop.json').exists() else 'CPU_timeout'),'sample':value})
                os._exit(75)
            write_json(folder/'live.json',value);done.wait(.2)
    thread=threading.Thread(target=watch,daemon=True);thread.start();code=0
    try:
        arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
        assert bool(a.module)!=bool(a.script)
        sys.argv=[a.module or a.script,*arguments]
        if a.script:
            path=Path(a.script).resolve();assert path.is_relative_to(PROJECT) or path.is_relative_to(ROOT)
            sys.path.insert(0,str(path.parent));runpy.run_path(str(path),run_name='__main__')
        else:runpy.run_module(a.module,run_name='__main__',alter_sys=True)
    except SystemExit as exc:code=exc.code or 0
    except BaseException:
        code=1
        raise
    finally:
        done.set();thread.join(timeout=2)
        write_json(folder/'closure.json',{'pid':os.getpid(),'actual_exit_code':code,'observed':observed,'wall_seconds':time.monotonic()-started,'CPU_private_cap_bytes':2_000_000_000,'CUDA_masked':True})
    raise SystemExit(code)

if __name__=='__main__':main()
