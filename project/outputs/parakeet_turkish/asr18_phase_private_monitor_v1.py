"""Available, fresh, clean-joined phase private samples; no fallback peak."""
import os, threading, time
from bpe_broad_production_state_v1 import require


class PhasePrivateMonitor:
    def __init__(self):
        self.done = threading.Event(); self.count = 0; self.peak = 0; self.last = None; self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        from resource_watchdog_v2 import sample_memory
        try:
            while not self.done.is_set():
                item = sample_memory(os.getpid(), 'full_DATA_native_phase')['worker_private_memory']
                require(item['available'], 'Available actual phase-private sample required')
                self.count += 1; self.peak = max(self.peak, item['private_bytes'])
                self.last = time.monotonic(); self.done.wait(.05)
        except BaseException as error: self.error = repr(error)

    def start(self):
        self.thread.start(); deadline = time.monotonic()+2
        while self.count < 3:
            require(self.thread.is_alive() and self.error is None and time.monotonic() < deadline,
                    'Three available phase samples required BEFORE work')
            time.sleep(.01)

    def finish(self):
        self.done.set(); self.thread.join(timeout=2)
        require(not self.thread.is_alive() and self.error is None and self.count >= 3
                and self.last is not None and time.monotonic()-self.last < 2,
                'Clean joined>=3 fresh phase samples required; never guess a peak')
        return {'private_peak_bytes': self.peak, 'available_sample_count': self.count,
                'last_sample_monotonic': self.last, 'error': None, 'joined': True, 'interval_seconds': .05}
