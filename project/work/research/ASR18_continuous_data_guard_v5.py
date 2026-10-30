"""Human-authorized concurrent CPU I/O, with unchanged v2 resource protections."""
import datetime as dt
import json
import time

import ASR18_data_io_guard_v2 as base
from cpu_preparation_guard_v2 import memory, now
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/parakeet_turkish"))
from atomic_io_retry_v1 import write_json

DataStop = base.DataStop
ROOT = base.ROOT
AREA = base.AREA
STOP = AREA / 'DATA_stop_request_v4.json'
PAUSE = AREA / 'DATA_named_measurement_pause_v3.json'
RATE = 32 * 1024 * 1024


class DataIOGuard(base.DataIOGuard):
    def publish(self, status, **fields):
        counters=self.process.io_counters();cpu=self.process.cpu_times()
        fields.update(process_read_bytes=counters.read_bytes,process_write_bytes=counters.write_bytes,CPU_user_seconds=cpu.user,CPU_system_seconds=cpu.system)
        self.state.update(fields, status=status, updated_at_utc=now(), aggregate_paced_bytes=self.bytes,
                          maximum_role_IO_bytes_s=RATE, GPU_owner_permission_required=False,
                          continuously_concurrent_IO=True, named_measurement_pause_maximum_seconds=60)
        write_json(AREA / (self.role + '_IO_status_v5.json'), self.state)
        self.last_status = time.monotonic()

    def reason(self, reserve=0):
        m = memory(); private = self.process.memory_info().private
        if m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000 or private >= 2_000_000_000:
            raise DataStop('critical_resource_bound')
        if base.shutil.disk_usage(ROOT).free - reserve < 100_000_000_000:
            raise DataStop('E_100GB_reserve')
        if STOP.exists():
            request = json.loads(STOP.read_text(encoding='utf-8-sig'))
            if request.get('stop') and self.role in request.get('roles', []):
                raise DataStop('explicit_owned_data_stop_request')
        if m['commit_ratio'] >= .90:
            return 'system_commit_90percent', m, private
        if PAUSE.exists():
            try:
                request = json.loads(PAUSE.read_text(encoding='utf-8-sig'))
                published = dt.datetime.fromisoformat(request['updated_at_utc'])
                until = dt.datetime.fromisoformat(request['valid_until_utc'])
                current = dt.datetime.now(dt.timezone.utc)
                if (request.get('owner') in ('/root', '/root/memory_diagnosis')
                    and request.get('pause') is True and request.get('measurement_id') and request.get('reason')
                    and self.role in request.get('roles', []) and published <= current < until
                    and 0 < (until - published).total_seconds() <= 60):
                    return 'bounded_named_measurement:' + request['measurement_id'], m, private
            except (OSError, ValueError, TypeError, KeyError):
                pass
        return None, m, private

    def pace(self, count):
        if self.bytes + count > self.maximum_bytes:
            raise DataStop('explicit_paced_IO_budget')
        self.wait()
        deadline = max(self.next_at, time.monotonic()) + count / RATE
        while time.monotonic() < deadline:
            time.sleep(min(.025, max(0, deadline - time.monotonic())))
        self.wait(); self.next_at = deadline; self.bytes += count
        if time.monotonic() - self.last_status > 10:
            self.publish('paced_IO_active', memory=memory(), own_private_bytes=self.process.memory_info().private,
                         own_RSS_bytes=self.process.memory_info().rss)
