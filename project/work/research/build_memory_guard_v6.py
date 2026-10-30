"""Add bounded self telemetry during all worker phases; preserve version5."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
text=(D/'launch_with_memory_guard_v5.py').read_text(encoding='utf-8')
def once(left,right):
    global text
    if text.count(left)!=1:raise ValueError('Unique span required:'+left[:80])
    text=text.replace(left,right)
once('import time\n','import time\nimport threading\n')
text=text.replace('memory_guard_training_launcher_v5','memory_guard_training_launcher_v6')
once('class UpdateTelemetry:', '''class LiveTelemetrySampler:
    """Read-only owned-self Windows telemetry; no Torch, CUDA or RNG operations.

    A latest atomic record stays fresh through cache preparation and development.
    The thread is stopped and joined before final evidence is serialized. Every
    deque is bounded; observational failure never replaces worker exceptions.
    """
    def __init__(self, pid, sink, *, sample=sample_memory):
        self.pid = pid; self.sink = sink; self.sample = sample
        self.stop_event = threading.Event(); self.thread = None
        self.samples = deque(maxlen=128); self.successes = 0; self.failures = 0
        self.publish_failures = 0; self.join_completed = True

    def sample_once(self):
        try:
            value = self.sample(self.pid, 'owned_worker_all_phases_observational_sampler')
            self.successes += 1; self.samples.append(value)
            try:
                self.sink(value)
            except Exception as error:
                self.publish_failures += 1
                self.samples[-1]['live_publish_failure_type'] = type(error).__name__
        except Exception as error:
            self.failures += 1
            self.samples.append({'status': 'telemetry_failed', 'failure_type': type(error).__name__,
                                 'observed_at_utc': timestamp()})

    def _run(self):
        while not self.stop_event.is_set():
            self.sample_once()
            if self.stop_event.wait(5.):
                return

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Sampler may start only once')
        self.thread = threading.Thread(target=self._run, name='owned-self-memory-telemetry', daemon=True)
        self.thread.start()

    def stop_and_join(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
            self.join_completed = not self.thread.is_alive()

    def evidence(self):
        return {'scope': 'Actual owned-self process; every5sec while nestedworkeractive includingcache/DEV; stdlibWindowsAPI only',
                'successful_samples': self.successes, 'failed_samples': self.failures,
                'live_publish_failures': self.publish_failures, 'sample_capacity': 128,
                'samples_omitted': max(0, self.successes+self.failures-len(self.samples)),
                'minimum_sampling_interval_seconds': 5., 'stop_requested': self.stop_event.is_set(),
                'thread_join_completed': self.join_completed, 'samples': list(self.samples),
                'CUDA_model_RNG_operations': False}


class UpdateTelemetry:''')
once("    identity_sha = file_sha(args.worker_identity)\n    def publish_memory(value):", "    identity_sha = file_sha(args.worker_identity)\n    publish_lock = threading.Lock(); latest_published = [float('-inf')]\n    def publish_memory(value):")
once("        write_json(live_memory_path, {'schema': 1, 'kind': 'owned_worker_live_memory_v1',\n            'pid': pid, 'ppid': identity['ppid'], 'run_id': run_id, 'worker_identity_sha256': identity_sha,\n            'process_creation_identity': identity['process_creation_identity'], **sources,\n            'sample': value, 'scope': 'Self-observed telemetry; latest bounded atomic record; no cross-process memory access'},\n            retry_seconds=1.)", """        observed = value.get('observed_monotonic', time.monotonic())
        with publish_lock:
            if observed < latest_published[0]:
                return  # Concurrent sampling cannot replace a newer observation.
            write_json(live_memory_path, {'schema': 1, 'kind': 'owned_worker_live_memory_v1',
                'pid': pid, 'ppid': identity['ppid'], 'run_id': run_id, 'worker_identity_sha256': identity_sha,
                'process_creation_identity': identity['process_creation_identity'], **sources,
                'sample': value, 'scope': 'Self-observed telemetry; latest bounded atomic record; no cross-process memory access'},
                retry_seconds=1.)
            latest_published[0] = observed""")
once("    telemetry = UpdateTelemetry(pid, sink=publish_memory)","    telemetry = UpdateTelemetry(pid, sink=publish_memory)\n    live_sampler = LiveTelemetrySampler(pid, publish_memory)")
once("'live_memory_update_policy': 'Before/after Torch startup, then at most every5sec at exact optimizer-end;1sec atomic-write retry; observational failures recorded',", "'live_memory_update_policy': 'Before/after Torch startup; boundedstdlibdaemon every5sec duringcache/DEV/training plus exact optimizer-end samples;1sec atomic-write retry; stop/join finally; observational failures recorded',")
once("        cache_launcher.main(SimpleNamespace", "        live_sampler.start()\n        cache_launcher.main(SimpleNamespace")
once("    finally:\n        if original_class is not None:","    finally:\n        live_sampler.stop_and_join()\n        if original_class is not None:")
once("        evidence.update(finished_at_utc=timestamp(), owned_update_memory_telemetry=telemetry.evidence(),", "        evidence.update(finished_at_utc=timestamp(), owned_update_memory_telemetry=telemetry.evidence(),\n                        all_phase_owned_memory_telemetry=live_sampler.evidence(),")
target=D/'launch_with_memory_guard_v6.py'
if target.exists():raise FileExistsError(target)
ast.parse(text);target.write_text(text,encoding='utf-8',newline='\n')
print(hashlib.sha256(target.read_bytes()).hexdigest())
