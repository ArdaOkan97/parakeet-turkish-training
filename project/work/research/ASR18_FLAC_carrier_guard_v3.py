"""One CPU carrier child's resource, storage, and bounded byte-I/O guards."""
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import psutil
from ASR18_continuous_data_guard_v4 import DataIOGuard, DataStop, ROOT, AREA
from cpu_preparation_guard_v2 import memory, now, write_json

CACHE = ROOT/'cache/KIRAAT_original_FLAC_records_v1'


def signature(path):
    p=Path(path);s=p.stat()
    return {'path':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def metadata(path, limit=8_000_000):
    p=Path(path);assert p.stat().st_size<=limit
    return json.loads(p.read_text(encoding='utf-8-sig'))


class CarrierGuard(DataIOGuard):
    def __init__(self, config):
        self.config=config;self.run=Path(config['run_folder'])
        self.run.mkdir(exist_ok=True);self.flush_hook=lambda:None
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            assert os.environ.get(key)=='1'
        assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
        for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):
            assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
        super().__init__('FLAC_carrier',heavy=False,maximum_bytes=config['remaining_IO_bytes'])
        self.callback=lambda:self.flush_hook()
        self.done=threading.Event();self.samples=0;self.last_sample=None;self.peak_private=0;self.peak_commit=0
        self.monitor_error=None;self.payload_written=0;self.last_live=0;self.last_storage=0
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
        while self.samples<3:
            assert self.thread.is_alive() and self.monitor_error is None
            time.sleep(.025)
        self.wait();self.check_storage(0)

    def watch(self):
        try:
            while not self.done.is_set():
                m=memory();p=self.process.memory_info().private;t=time.monotonic()
                self.samples+=1;self.last_sample=t;self.peak_private=max(self.peak_private,p)
                self.peak_commit=max(self.peak_commit,m['commit_ratio'])
                critical=m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or p>=2_000_000_000
                if critical or t-self.last_live>1:
                    write_json(self.run/'live.json',{'at_utc':now(),'PID':os.getpid(),'private_bytes':p,
                        'memory':m,'paced_IO_bytes':self.bytes,'payload_written_bytes':self.payload_written,
                        'critical':critical,'samples':self.samples})
                    self.last_live=t
                if critical:
                    write_json(self.run/'resource_stop.json',{'kind':'ASR18_FLAC_critical_resource_stop',
                        'at_utc':now(),'private_bytes':p,'memory':m,'paced_IO_bytes':self.bytes,
                        'originals_and_payload_prefix_preserved':True})
                    os._exit(75)
                self.done.wait(.1)
        except BaseException as error:
            self.monitor_error=type(error).__name__
            try:write_json(self.run/'resource_stop.json',{'kind':'ASR18_FLAC_watchdog_failed','reason':self.monitor_error})
            finally:os._exit(75)

    def check_storage(self, additional):
        # The parent's fresh full-missing-source reservation is conservative throughout this child.
        free=psutil.disk_usage(str(ROOT)).free
        assert free>=self.config['remaining_download_bytes_reserved']+100_000_000_000+additional,'Reserve all remaining fixed downloads and E100GB'
        assert self.config['cache_bytes_before']+self.payload_written+additional+144_000_000<=400_000_000_000,'Aggregate400GB cache cap'

    def charge(self,count):
        assert self.thread.is_alive() and self.monitor_error is None and time.monotonic()-self.last_sample<2
        self.pace(count)

    def read(self,stream,count):
        result=bytearray(count);view=memoryview(result);done=0
        while done<count:
            n=min(1024**2,count-done);self.charge(n)
            got=stream.readinto(view[done:done+n])
            if not got:break
            done+=got
        return bytes(result) if done==count else bytes(view[:done])

    def write_payload(self,stream,data):
        offset=0
        while offset<len(data):
            part=memoryview(data)[offset:offset+1024**2]
            assert self.payload_written+len(part)<=self.config['remaining_new_payload_bytes'],'Declared16GB pilot new-payload bound'
            self.check_storage(len(part));self.charge(len(part))
            written=stream.write(part);assert written==len(part)
            self.payload_written+=written;offset+=written

    def binding(self,path):
        p=Path(path);s=p.stat();digest=hashlib.sha256()
        with p.open('rb',buffering=0) as f:
            while True:
                data=self.read(f,min(1024**2,s.st_size-f.tell()))
                if not data:break
                digest.update(data)
        assert p.stat().st_size==s.st_size and p.stat().st_mtime_ns==s.st_mtime_ns
        return {'path':str(p.resolve()),'bytes':s.st_size,'sha256':digest.hexdigest()}

    def close(self,code=0):
        self.done.set();self.thread.join(timeout=2)
        assert not self.thread.is_alive() and self.monitor_error is None and self.samples>=3
        write_json(self.run/'native_closure.json',{'kind':'ASR18_guarded_FLAC_native_closure_v3',
            'PID':os.getpid(),'at_utc':now(),'native_return_code':code,'watchdog_clean':True,
            'samples':self.samples,'peak_private_bytes':self.peak_private,'peak_commit_ratio':self.peak_commit,
            'actual_paced_IO_bytes':self.bytes,'actual_new_payload_bytes':self.payload_written,
            'CPU_private_cap_bytes':2_000_000_000,'CUDA_masked':True,
            'thread_caps':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}})


class PacedParquet(io.RawIOBase):
    def __init__(self,path,guard):self.raw=Path(path).open('rb',buffering=0);self.guard=guard
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.raw.tell()
    def seek(self,offset,whence=0):self.guard.wait();return self.raw.seek(offset,whence)
    def read(self,n=-1):
        if n<0 or n>512*1024**2:raise DataStop('bounded_512MiB_original_Parquet_read_unresolved')
        return self.guard.read(self.raw,n)
    def readinto(self,buffer):
        value=self.read(len(buffer));buffer[:len(value)]=value;return len(value)
    def close(self):
        if not self.closed:self.raw.close()
        super().close()
