"""Additional512MB/180s guard around the ROOT-reviewed metadata assembler."""
import json,os,runpy,sys,threading,time,traceback
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from fastmss_BPE9223_cpu_common_v4 import environment,resource_guard
from bpe_broad_audio_v1 import record
environment()
F=Path(__file__).parent/'metadata_configuration_02';assert not F.exists();F.mkdir();script=P/'work/research/prepare_ASR18_finite_BPE9223_serial_DEV_configuration_v1.py'
assert record(script)['sha256']=='2b738aded7c4926aa9c619daf221f6e53d869978db939d3bf03d69a4e28282d2'
done=threading.Event();began=time.monotonic();state={'samples':0,'peak_private_bytes':0,'error':None}
def loop():
    try:
        while not done.is_set():
            value=resource_guard();w=value['worker_private_memory'];private=w['private_bytes']
            assert private<512_000_000 and time.monotonic()-began<180
            state['samples']+=1;state['peak_private_bytes']=max(state['peak_private_bytes'],private)
            done.wait(.1)
    except BaseException:
        state['error']=traceback.format_exc();(F/'metadata512_failure.json').write_text(json.dumps(state));os._exit(75)
thread=threading.Thread(target=loop,daemon=True);thread.start()
while state['samples']<3:
    assert thread.is_alive() and time.monotonic()-began<3;time.sleep(.02)
print(json.dumps({'actual_metadata_PID':os.getpid(),'private_cap_bytes':512_000_000,'wall_seconds':180}),flush=True)
code=1
try:
    args=sys.argv[1:]
    sys.argv=[str(script),*args];runpy.run_path(str(script),run_name='__main__');code=0
finally:
    done.set();thread.join(2);assert not thread.is_alive() and state['error'] is None
    (F/'metadata512_closure.json').write_text(json.dumps({'pid':os.getpid(),'actual_exit_code':code,
        'monitor_joined':True,'monitor_clean':True,'private_cap_bytes':512_000_000,'wall_cap_seconds':180,
        'observed':state,'metadata_source':record(script),'wall_seconds':time.monotonic()-began},indent=2))
