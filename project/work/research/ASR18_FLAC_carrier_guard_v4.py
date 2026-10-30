"""Separately admitted64MiB/s carrier guard with projected Arrow-memory admission."""
import time
from ASR18_FLAC_carrier_guard_v3 import CarrierGuard as PreviousGuard,AREA,memory,write_json,now
from ASR18_continuous_data_guard_v4 import DataStop


class CarrierGuard(PreviousGuard):
    RATE=64*1024**2

    def __init__(self,config):
        super().__init__(config)
        reserve=config.get('expected_peak_private_bytes',1_000_000_000)+250_000_000
        fresh=0
        while fresh<3:
            self.wait();m=memory();limit=m['commit_headroom_bytes']/(1-m['commit_ratio'])
            projected=m['commit_ratio']+reserve/limit
            admitted=projected<.90 and m['commit_headroom_bytes']-reserve>=2_000_000_000
            fresh=fresh+1 if admitted else 0
            value={'at_utc':now(),'PID':self.process.pid,'current_memory':m,
                'reserved_Arrow_peak_plus_margin_bytes':reserve,'projected_commit_ratio':projected,
                'projected_headroom_bytes':m['commit_headroom_bytes']-reserve,
                'admitted':admitted,'consecutive_fresh_admitted_samples':fresh,
                'GPU_idle_permission_required':False,'before_PyArrow_import_or_large_source_read':True}
            write_json(self.run/'projected_memory_admission.json',value)
            if not admitted:self.publish('yielding_projected_Arrow_memory_admission',projection=value)
            time.sleep(.1 if admitted else .25)

    def publish(self,status,**fields):
        counters=self.process.io_counters()
        self.state.update(fields,status=status,updated_at_utc=now(),aggregate_paced_bytes=self.bytes,
            maximum_role_IO_bytes_s=self.RATE,GPU_owner_permission_required=False,
            continuously_concurrent_IO=True,process_read_bytes=counters.read_bytes,
            process_write_bytes=counters.write_bytes,pacing_credit_bound_bytes=2*1024**2)
        write_json(AREA/'FLAC_carrier_IO_status_v5.json',self.state)
        self.last_status=time.monotonic()

    def pace(self,count):
        if self.bytes+count>self.maximum_bytes:raise DataStop('explicit_paced_IO_budget')
        self.wait();current=time.monotonic()
        # Preserve limited oversleep credit so sub-timer-resolution row reads do not starve throughput.
        deadline=max(self.next_at,current-(2*1024**2)/self.RATE)+count/self.RATE
        while time.monotonic()<deadline:
            time.sleep(min(.01,max(0,deadline-time.monotonic())))
        self.wait();self.next_at=deadline;self.bytes+=count
        if time.monotonic()-self.last_status>10:self.publish('paced_IO_active',memory=memory(),own_private_bytes=self.process.memory_info().private)
