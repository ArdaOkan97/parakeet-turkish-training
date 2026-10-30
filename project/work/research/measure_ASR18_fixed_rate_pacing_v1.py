"""Bounded timing-only diagnostic: simulated latency is never network evidence."""
import hashlib
import time
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA
from cpu_preparation_guard_v2 import now

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


def main():
    guard = DataIOGuard('fixed32_pacing_timing_diagnostic', heavy=False, maximum_bytes=64 * 1024**2)
    guard.wait()
    chunk = 1024**2
    rate = 32 * 1024**2
    outcomes = []
    for kind in ('fresh_sleep_plus_latency', 'elapsed_time_credit2MiB'):
        start = time.monotonic()
        next_at = start
        for index in range(24):
            guard.wait()
            current = time.monotonic()
            credit = 0 if kind == 'fresh_sleep_plus_latency' else 2 * 1024**2 / rate
            deadline = max(next_at, current - credit) + chunk / rate
            while time.monotonic() < deadline:
                time.sleep(min(.025, max(0, deadline - time.monotonic())))
            guard.wait()
            next_at = deadline
            time.sleep(.04)
        elapsed = time.monotonic() - start
        outcomes.append({'mode': kind, 'simulated_chunks': 24, 'simulated_chunk_bytes': chunk,
                         'injected_latency_seconds_per_chunk': .04, 'elapsed_seconds': elapsed,
                         'simulated_effective_MiB_s': 24 / elapsed})
    result = {'kind': 'ASR18_fixed32_timing_only_pacing_diagnostic_v1', 'at_utc': now(),
              'maximum_rate_bytes_s': rate, 'outcomes': outcomes,
              'actual_network_bytes': 0, 'no_dataset_or_carrier_IO': True,
              'healthy_worker_changed': False, 'rate_or_concurrency_increase': False,
              'actual_network_speedup_not_proven': True, 'TRAIN_admitted': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = AREA / 'fixed32_timing_only_pacing_diagnostic_v1.json'
    assert not path.exists()
    write_json(path, result)
    print(result)


if __name__ == '__main__':
    main()
