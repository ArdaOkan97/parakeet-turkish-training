"""256MiB/s finite carrier guard; unchanged caps with fresh shared resource samples."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

import psutil
from ASR18_FLAC_carrier_guard_v3 import CarrierGuard as PreviousGuard,AREA,ROOT,memory,now
from ASR18_continuous_data_guard_v4 import DataStop,STOP,PAUSE
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


class CarrierGuard(PreviousGuard):
    RATE=256*1024**2
    CREDIT=8*1024**2

    def __init__(self,config):
        super().__init__(config)
        reserve=config.get('expected_peak_private_bytes',1_000_000_000)+250_000_000;fresh=0
        while fresh<3:
            self.wait();m=memory();limit=m['commit_headroom_bytes']/(1-m['commit_ratio'])
            projected=m['commit_ratio']+reserve/limit
            admitted=projected<.90 and m['commit_headroom_bytes']-reserve>=2_000_000_000
            fresh=fresh+1 if admitted else 0
            value={'at_utc':now(),'PID':self.process.pid,'current_memory':m,
                'reserved_Arrow_peak_plus_margin_bytes':reserve,'projected_commit_ratio':projected,
                'projected_headroom_bytes':m['commit_headroom_bytes']-reserve,'admitted':admitted,
                'consecutive_fresh_admitted_samples':fresh,'GPU_idle_permission_required':False,
                'before_PyArrow_import_or_large_source_read':True}
            write_json(self.run/'projected_memory_admission.json',value)
            if not admitted:self.publish('yielding_projected_Arrow_memory_admission',projection=value)
            time.sleep(.1 if admitted else .25)

    def watch(self):
        try:
            while not self.done.is_set():
                m=memory();private=self.process.memory_info().private;free=psutil.disk_usage(str(ROOT)).free
                request=json.loads(STOP.read_text(encoding='utf-8-sig')) if STOP.exists() else {}
                pause=json.loads(PAUSE.read_text(encoding='utf-8-sig')) if PAUSE.exists() else {}
                current=time.monotonic()
                assert 0<m['commit_ratio']<1 and m['commit_headroom_bytes']>=0
                self._paired_snapshot=(current,m,private,free,self.payload_written,request,pause)
                self.samples+=1;self.last_sample=current;self.peak_private=max(self.peak_private,private)
                self.peak_commit=max(self.peak_commit,m['commit_ratio'])
                critical=m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or private>=2_000_000_000
                if critical or current-self.last_live>1:
                    write_json(self.run/'live.json',{'at_utc':now(),'PID':os.getpid(),'private_bytes':private,
                        'memory':m,'paced_IO_bytes':self.bytes,'payload_written_bytes':self.payload_written,
                        'critical':critical,'samples':self.samples,'resource_sampling_period_seconds':.1})
                    self.last_live=current
                if critical:
                    write_json(self.run/'resource_stop.json',{'at_utc':now(),'reason':'unchanged_95percent_2GBheadroom_or_private2GB',
                        'private_bytes':private,'memory':m,'paced_IO_bytes':self.bytes,'prefix_preserved':True})
                    os._exit(75)
                self.done.wait(.1)
        except BaseException as error:
            self.monitor_error=type(error).__name__
            try:write_json(self.run/'resource_stop.json',{'reason':'fail_closed_watchdog_exception','error':self.monitor_error})
            finally:os._exit(75)

    def snapshot(self):
        value=getattr(self,'_paired_snapshot',None)
        if value is None or time.monotonic()-value[0]>.5:
            # Arrow can hold the GIL during a large buffer operation. Require a NEW sample;
            # never accept the stale snapshot or increase its allowed age.
            if not self.thread.is_alive() or self.monitor_error is not None:raise DataStop('resource_monitor_unavailable')
            m=memory();private=self.process.memory_info().private;free=psutil.disk_usage(str(ROOT)).free
            request=json.loads(STOP.read_text(encoding='utf-8-sig')) if STOP.exists() else {}
            pause=json.loads(PAUSE.read_text(encoding='utf-8-sig')) if PAUSE.exists() else {}
            if not 0<m['commit_ratio']<1:raise DataStop('fresh_synchronous_resource_sample_unavailable')
            current=time.monotonic();value=(current,m,private,free,self.payload_written,request,pause)
            self._paired_snapshot=value;self.last_sample=current;self.samples+=1
            self.peak_private=max(self.peak_private,private);self.peak_commit=max(self.peak_commit,m['commit_ratio'])
            self.synchronous_refreshes=getattr(self,'synchronous_refreshes',0)+1
        return value

    def reason(self,reserve=0):
        stamp,m,private,free,written,request,pause=self.snapshot()
        if m['commit_ratio']>=.95 or m['commit_headroom_bytes']<2_000_000_000 or private>=2_000_000_000:
            raise DataStop('critical_resource_bound')
        # Stronger than the100GB floor: account for own writes since sampling plus128MiB concurrent-write allowance.
        if free-(self.payload_written-written)-reserve<100_000_000_000+128*1024**2:raise DataStop('E_100GB_reserve')
        if request.get('stop') and self.role in request.get('roles',[]):raise DataStop('explicit_owned_data_stop_request')
        if m['commit_ratio']>=.90:return 'system_commit_90percent',m,private
        if pause.get('pause') is True and pause.get('owner') in ('/root','/root/memory_diagnosis'):
            try:
                start=dt.datetime.fromisoformat(pause['updated_at_utc']);end=dt.datetime.fromisoformat(pause['valid_until_utc'])
                if (self.role in pause.get('roles',[]) and pause.get('measurement_id') and pause.get('reason')
                    and 0<(end-start).total_seconds()<=60 and start<=dt.datetime.now(dt.timezone.utc)<end):
                    return 'bounded_named_measurement:'+pause['measurement_id'],m,private
            except (KeyError,ValueError,TypeError):pass
        return None,m,private

    def check_storage(self,additional):
        stamp,m,private,free,written,request,pause=self.snapshot()
        assert free-(self.payload_written-written)>=self.config['remaining_download_bytes_reserved']+100_000_000_000+128*1024**2+additional
        assert self.config['cache_bytes_before']+self.payload_written+additional+144_000_000<=400_000_000_000

    def publish(self,status,**fields):
        self.state.update(fields,status=status,updated_at_utc=now(),aggregate_paced_bytes=self.bytes,
            maximum_role_IO_bytes_s=self.RATE,GPU_owner_permission_required=False,
            continuously_concurrent_IO=True,pacing_credit_bound_bytes=self.CREDIT,
            resource_sample_period_seconds=.1,resource_stale_fail_closed_seconds=.5)
        write_json(AREA/'FLAC_carrier_IO_status_v6.json',self.state);self.last_status=time.monotonic()

    def pace(self,count):
        if self.bytes+count>self.maximum_bytes:raise DataStop('explicit_paced_IO_budget')
        self.wait();current=time.monotonic()
        deadline=max(self.next_at,current-self.CREDIT/self.RATE)+count/self.RATE
        while time.monotonic()<deadline:time.sleep(min(.005,max(0,deadline-time.monotonic())))
        self.wait();self.next_at=deadline;self.bytes+=count
        if time.monotonic()-self.last_status>10:
            snap=self.snapshot();self.publish('paced_IO_active',memory=snap[1],own_private_bytes=snap[2])

    def close(self,code=0):
        self.done.set();self.thread.join(timeout=2)
        assert not self.thread.is_alive() and self.monitor_error is None and self.samples>=3
        write_json(self.run/'native_closure.json',{'kind':'ASR18_guarded_FLAC_native_closure_v5',
            'PID':os.getpid(),'at_utc':now(),'native_return_code':code,'watchdog_clean':True,
            'samples':self.samples,'peak_private_bytes':self.peak_private,'peak_commit_ratio':self.peak_commit,
            'actual_paced_IO_bytes':self.bytes,'actual_new_payload_bytes':self.payload_written,
            'CPU_private_cap_bytes':2_000_000_000,'CUDA_masked':True,
            'resource_sampling_period_seconds':.1,'resource_stale_fail_closed_seconds':.5,
            'thread_caps':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}})
