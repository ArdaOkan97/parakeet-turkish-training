"""ASR18 CPU data-only guards. Stdlib + psutil; never import Torch/model runtimes."""
import ctypes as c
from ctypes import wintypes as w
import datetime as dt
import io,json,os,shutil,threading,time
from pathlib import Path
import psutil
from cpu_preparation_guard_v2 import memory,scoped,write_json,now
ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/BPE_weak_then_clean_v1'
CONTROL=AREA/'ROOT_data_IO_control_v2.json'
STOP=AREA/'DATA_stop_request_v2.json'

class DataStop(RuntimeError):pass

def hard_job():
    class IO(c.Structure):
        _fields_=[(x,c.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
    class B(c.Structure):
        _fields_=[('p',c.c_longlong),('j',c.c_longlong),('flags',w.DWORD),('min',c.c_size_t),('max',c.c_size_t),('active',w.DWORD),('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class E(c.Structure):
        _fields_=[('basic',B),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
    k=c.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
    k.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];k.SetInformationJobObject.restype=w.BOOL
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.AssignProcessToJobObject.restype=w.BOOL
    k.GetCurrentProcess.restype=w.HANDLE
    handle=k.CreateJobObjectW(None,None);assert handle
    limits=E();limits.basic.flags=0x100;limits.process_memory=2_000_000_000
    if not k.SetInformationJobObject(handle,9,c.byref(limits),c.sizeof(limits)):raise c.WinError(c.get_last_error())
    if not k.AssignProcessToJobObject(handle,k.GetCurrentProcess()):raise c.WinError(c.get_last_error())
    return handle

class DataIOGuard:
    def __init__(self,role,heavy=True,maximum_bytes=1_300_000_000_000):
        cfg=json.loads((ROOT/'storage_config.json').read_text(encoding='utf-8-sig'))
        assert Path(cfg['storage_root']).resolve()==ROOT.resolve() and not cfg['allow_fallback_outside_root']
        for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR','PYTHONPYCACHEPREFIX'):scoped(os.environ[key])
        assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
        assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
        self.role=role;self.heavy=heavy;self.maximum_bytes=maximum_bytes;self.bytes=0
        self.process=psutil.Process();self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);self.job=hard_job()
        self.callback=None;self.paused=False;self.next_at=0.;self.last_status=0.;self.last_check=0.;self.last_reason=None
        self.state={'owner':'/root/cpu_data_preparation','task':'ASR-18','PID':os.getpid(),'role':role,
          'started_at_utc':now(),'GPU_used':False,'model_runtime_imported':False,'CPU_threads':1,
          'own_private_hard_limit_bytes':2_000_000_000,'yield_commit_ratio':.90,'critical_commit_ratio':.95,
          'minimum_commit_headroom_bytes':2_000_000_000,'minimum_E_free_bytes':100_000_000_000,
          'maximum_paced_bytes':maximum_bytes,'maximum_role_IO_bytes_s':1048576 if heavy else 262144,
          'TRAIN_admitted':False,'quality_filters':[],'language_filters':[]}
        self.publish('starting')
        threading.Thread(target=self.watchdog,daemon=True).start()
    def publish(self,status,**fields):
        self.state.update(fields,status=status,updated_at_utc=now(),aggregate_paced_bytes=self.bytes)
        write_json(AREA/(self.role+'_IO_status_v2.json'),self.state);self.last_status=time.monotonic()
    def watchdog(self):
        while True:
            time.sleep(2)
            try:
                m=memory();private=self.process.memory_info().private
                if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or private>=2_000_000_000:
                    self.publish('critical_resource_stop',memory=m,own_private_bytes=private);os._exit(75)
            except Exception:os._exit(76)
    def reason(self,reserve=0):
        m=memory();private=self.process.memory_info().private
        if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or private>=2_000_000_000:raise DataStop('critical_resource_bound')
        if shutil.disk_usage(ROOT).free-reserve<100_000_000_000:raise DataStop('E_100GB_reserve')
        if STOP.exists():
            request=json.loads(STOP.read_text(encoding='utf-8-sig'))
            if request.get('stop') and self.role in request.get('roles',[]):raise DataStop('explicit_owned_data_stop_request')
        if m['commit_ratio']>=.90:return 'system_commit_90percent',m,private
        if not self.heavy:return None,m,private
        try:permit=json.loads(CONTROL.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError):return 'ROOT_steady_safe_IO_permit_absent',m,private
        allowed=(permit.get('owner')=='/root/memory_diagnosis' and permit.get('phase')=='steady_safe'
          and permit.get('allow_heavy_IO') is True and self.role in permit.get('permitted_roles',[])
          and len(permit.get('permitted_roles',[]))==1)
        try:
            published=dt.datetime.fromisoformat(permit['updated_at_utc']);until=dt.datetime.fromisoformat(permit['valid_until_utc']);current=dt.datetime.now(dt.timezone.utc)
            allowed=allowed and published<=current<until and 0<(until-published).total_seconds()<=300
        except (KeyError,TypeError,ValueError):allowed=False
        return (None if allowed else 'ROOT_startup_profiling_DEV_checkpoint_or_expired_permit'),m,private
    def wait(self,reserve=0):
        while True:
            reason,m,private=self.reason(reserve)
            if reason is None:
                if self.paused:self.next_at=time.monotonic();self.paused=False;self.publish('paced_IO_allowed' if self.heavy else 'light_metadata_allowed',memory=m,own_private_bytes=private)
                return
            if not self.paused:
                if self.callback:self.callback()
                self.paused=True
            if reason!=self.last_reason or time.monotonic()-self.last_status>10:self.publish('paused_no_heavy_IO',pause_reason=reason,memory=m,own_private_bytes=private)
            self.last_reason=reason;time.sleep(.25)
    def pace(self,count):
        if self.bytes+count>self.maximum_bytes:raise DataStop('explicit_paced_IO_budget')
        self.wait()
        rate=1048576 if self.heavy else 262144
        deadline=max(self.next_at,time.monotonic())+count/rate
        while time.monotonic()<deadline:
            remaining=max(0.,deadline-time.monotonic())
            if remaining:time.sleep(min(.025,remaining))
        self.wait();self.next_at=deadline;self.bytes+=count
        if time.monotonic()-self.last_status>10:self.publish('paced_IO_active',memory=memory(),own_private_bytes=self.process.memory_info().private)
    def sha(self,path):
        import hashlib
        digest=hashlib.sha256()
        with open(path,'rb',buffering=0) as source:
            while True:
                self.pace(65536);block=source.read(65536)
                if not block:break
                digest.update(block)
        return digest.hexdigest()

class PacedInput(io.RawIOBase):
    def __init__(self,path,guard):self.raw=open(path,'rb',buffering=0);self.guard=guard
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.raw.tell()
    def seek(self,offset,whence=0):self.guard.wait();return self.raw.seek(offset,whence)
    def read(self,n=-1):
        if n<0 or n>64*1024*1024:raise ValueError('Unbounded or over64MiB single metadata read forbidden; retain source for bounded adapter')
        chunks=[]
        while n:
            count=min(n,65536);self.guard.pace(count);block=self.raw.read(count)
            if not block:break
            chunks.append(block);n-=len(block)
        return b''.join(chunks)
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
    def close(self):
        if not self.closed:self.raw.close()
        super().close()
