"""CPU preparation ownership, E-only output and unchanged resource guards."""
import ctypes
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import threading
import time
from ctypes import wintypes

ROOT = pathlib.Path(r'E:\gpu_training_scratchpad').resolve()
AREA = ROOT / 'analysis/CPU_validation_v2'
PROJECT = pathlib.Path(__file__).resolve().parents[2]
YODAS_REV = '81d802f45a1c1e1f5331260484dacf1fbea81981'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def scoped(path):
    p = pathlib.Path(path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('New storage outside designated E: root')
    return p

def write_json(path, value):
    p = scoped(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)

def append_json(path, value):
    p = scoped(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())

class Performance(ctypes.Structure):
    _fields_ = [('cb',ctypes.c_ulong),('CommitTotal',ctypes.c_size_t),('CommitLimit',ctypes.c_size_t),
        ('CommitPeak',ctypes.c_size_t),('PhysicalTotal',ctypes.c_size_t),('PhysicalAvailable',ctypes.c_size_t),
        ('SystemCache',ctypes.c_size_t),('KernelTotal',ctypes.c_size_t),('KernelPaged',ctypes.c_size_t),
        ('KernelNonpaged',ctypes.c_size_t),('PageSize',ctypes.c_size_t),('HandleCount',ctypes.c_ulong),
        ('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]

def memory():
    p=Performance(); p.cb=ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p),p.cb): raise OSError('Commit query failed')
    return {'commit_ratio':p.CommitTotal/p.CommitLimit,'commit_headroom_bytes':(p.CommitLimit-p.CommitTotal)*p.PageSize,
            'physical_available_bytes':p.PhysicalAvailable*p.PageSize}

def sha256(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

class Guard:
    def __init__(self, name):
        cfg=json.loads((ROOT/'storage_config.json').read_text(encoding='utf-8-sig'))
        if pathlib.Path(cfg['storage_root']).resolve()!=ROOT or cfg['allow_fallback_outside_root']: raise ValueError('Wrong storage policy')
        for key in ['TEMP','TMP','HF_HOME','PIP_CACHE_DIR','PYTHONPYCACHEPREFIX']:
            scoped(os.environ[key])
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1': raise ValueError('CPU-only process required')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
            if os.environ.get(key)!='1': raise ValueError('Single CPU thread required')
        self.name=name; self.path=AREA/(name+'_status.json'); self.start=now()
        self.state={'kind':name,'owner':'/root/cpu_data_preparation','pid':os.getpid(),
            'started_at_utc':self.start,'GPU_used':False,'TEST_predictions_read':False,'accepted_TRAIN_hours':0,
            'stop_commit_ratio':.95,'yield_commit_ratio':.90,'minimum_commit_headroom_bytes':2_000_000_000,
            'minimum_free_E_bytes':100_000_000_000,'permanent_WAV_or_PCM_written':False}
        ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
        ctypes.windll.kernel32.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        if not ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000):
            raise OSError('Cannot set BelowNormal CPU priority')
        self.status(phase='starting',memory=memory(),storage_free_bytes=shutil.disk_usage(ROOT).free)
        threading.Thread(target=self.watchdog,daemon=True).start()

    def status(self, **fields):
        self.state.update(fields); self.state['updated_at_utc']=now()
        write_json(self.path,self.state)
        print(json.dumps(fields,ensure_ascii=False),flush=True)

    def watchdog(self):
        while True:
            time.sleep(2)
            try:
                m=memory()
                if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000:
                    # Stop only this owned CPU worker; source archives/partial ledgers survive.
                    write_json(AREA/(self.name+'_resource_stop.json'),{'at_utc':now(),'pid':os.getpid(),'memory':m,'own_process_only':True})
                    os._exit(75)
            except Exception as e:
                print('CPU watchdog query failed: '+repr(e),flush=True); os._exit(76)

    def wait(self, reserve_bytes=0):
        while True:
            m=memory()
            if shutil.disk_usage(ROOT).free-reserve_bytes<100_000_000_000: raise RuntimeError('E:100GB free reserve reached')
            if m['commit_ratio']<.90 and m['commit_headroom_bytes']>=2_000_000_000: return m
            self.status(phase='yielding_own_CPU_work_for_memory',memory=m)
            time.sleep(10)
