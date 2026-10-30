"""Bounded CPU mocks and native self identity for the NEW v3 launcher."""
from pathlib import Path
import sys
import json
import os
import types
import tempfile
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/parakeet_turkish'))
import launch_with_memory_guard_v6 as launch
import launch_with_cache_reclaim_v3 as cache

checks = []
def check(name, value):
    if not value: raise AssertionError(name)
    checks.append(name)
def rejects(name, function, error=ValueError):
    try: function()
    except error: checks.append(name)
    else: raise AssertionError(name)

check('cap9-fraction', launch.validate_capacity(16_000_000_000)['per_process_allocator_fraction'] == .5625)
for n in (14_999_999_999, 18_000_000_001):
    rejects('wrong-device-capacity-' + str(n), lambda n=n: launch.validate_capacity(n))
for cap in (4, 6, 8, 13):
    rejects('wrong-cap-' + str(cap), lambda cap=cap: launch.validate_capacity(16_000_000_000, cap))
identity = launch.creation_identity(os.getpid())
check('real-owned-process-creation', identity['available'] and identity['pid'] == os.getpid() and identity['creation_filetime_100ns'] > 0)
run, sha = launch.argument_identity(['--', '--run-id', 'cpu_v3_guard'])
check('safe-argument-identity', run == 'cpu_v3_guard' and len(sha) == 64)
check('leading-separator-canonicalized', launch.argument_identity(['--run-id', 'cpu_v3_guard']) == (run, sha))
for args in ([], ['--run-id'], ['--run-id', 'one', '--run-id', 'two'], ['--run-id', '../escape']):
    rejects('invalid-run-id-' + repr(args), lambda args=args: launch.argument_identity(args))

now = [0.]
samples = []
def sample(pid, phase):
    samples.append((pid, phase)); return {'phase': phase, 'pid': pid}
telemetry = launch.UpdateTelemetry(123, sample=sample, clock=lambda: now[0])
class Base:
    def _eligible(self, frame): return frame
observed = launch.telemetry_reclaimer_class(Base, telemetry)()
check('wrong-scope-return-preserved', observed._eligible(False) is False and not samples)
check('eligible-return-preserved', observed._eligible(True) is True and len(samples) == 1)
observed._eligible(True)
check('sampling-throttled', len(samples) == 1 and telemetry.eligible_calls == 2)
now[0] = 5.
observed._eligible(True)
check('five-second-equality-sampled', len(samples) == 2)
for i in range(130):
    now[0] += 5; observed._eligible(True)
check('bounded-samples', len(telemetry.samples) == 128 and telemetry.evidence()['samples_omitted'] == 4)
broken = launch.UpdateTelemetry(123, sample=lambda *_: (_ for _ in ()).throw(OSError('fixture')), clock=lambda: 0.)
check('telemetry-failure-does-not-change-eligibility', launch.telemetry_reclaimer_class(Base, broken)()._eligible(True) is True and broken.failures == 1)
sink_fail = launch.UpdateTelemetry(123, sample=sample, clock=lambda: 0., sink=lambda _: (_ for _ in ()).throw(OSError('writer fixture')))
check('live-publish-failure-does-not-change-eligibility',launch.telemetry_reclaimer_class(Base,sink_fail)()._eligible(True) is True and sink_fail.publish_failures==1 and sink_fail.failures==0)
class Raising:
    def _eligible(self, frame): raise RuntimeError('original')
rejects('original-helper-error-preserved', lambda: launch.telemetry_reclaimer_class(Raising, telemetry)()._eligible(True), RuntimeError)

# Actual short stdlib thread lifecycle and bounded mocked telemetry; no GPU.
arrived = threading.Event(); delivered = []
sampler = launch.LiveTelemetrySampler(123, lambda value: (delivered.append(value), arrived.set()), sample=sample)
sampler.start()
check('all-phase-thread-initial-observation', arrived.wait(1.) and len(delivered)==1)
sampler.stop_and_join()
check('all-phase-thread-stopped-and-joined', sampler.evidence()['stop_requested'] and sampler.evidence()['thread_join_completed'])
check('all-phase-thread-five-second-policy', sampler.evidence()['minimum_sampling_interval_seconds']==5.)
rejects('all-phase-thread-cannot-start-twice', sampler.start, RuntimeError)
for _ in range(130):sampler.sample_once()
check('all-phase-bounded128-records',len(sampler.samples)==128 and sampler.evidence()['samples_omitted']==3)
failed_sampler=launch.LiveTelemetrySampler(123, lambda _: None, sample=lambda *_: (_ for _ in ()).throw(OSError('sample fixture')))
failed_sampler.sample_once()
check('all-phase-sample-error-recorded',failed_sampler.failures==1)
failed_sink=launch.LiveTelemetrySampler(123,lambda _: (_ for _ in ()).throw(OSError('sink fixture')),sample=sample)
failed_sink.sample_once()
check('all-phase-publish-error-recorded',failed_sink.publish_failures==1 and failed_sink.failures==0)
not_started=launch.LiveTelemetrySampler(123,lambda _: None,sample=sample)
not_started.stop_and_join()
check('all-phase-not-started-cleanup-safe',not_started.evidence()['thread_join_completed'])

with tempfile.TemporaryDirectory(dir=ROOT/'work/research', prefix='memory_v3_cpu_') as directory:
    directory = Path(directory)
    def args(prefix, **changes):
        value = types.SimpleNamespace(provenance=directory/(prefix+'_io.json'),
            cache_provenance=directory/(prefix+'_cache.json'), memory_provenance=directory/(prefix+'_memory.json'),
            worker_identity=directory/(prefix+'_identity.json'), cap_gb=9., threshold_gb=4.,
            max_events=128, retry_seconds=30., validate_only=False,
            arguments=['--', '--run-id', 'cpu_v3_guard'])
        for key, item in changes.items(): setattr(value,key,item)
        return value
    wrong = args('wrong', threshold_gb=6)
    rejects('threshold4-required', lambda: launch.main(wrong))
    duplicate = args('duplicate')
    duplicate.worker_identity = duplicate.provenance
    rejects('four-distinct-evidence-required', lambda: launch.main(duplicate))
    exists = args('exists'); exists.worker_identity.write_text('{}')
    rejects('old-identity-not-overwritten', lambda: launch.main(exists), FileExistsError)
    validate = args('validate', validate_only=True)
    before_torch = sys.modules.get('torch')
    launch.main(validate)
    check('validation-does-not-import-torch', sys.modules.get('torch') is before_torch)
    check('validation-writes-no-evidence', not any(path.exists() for path in (validate.provenance,validate.cache_provenance,validate.memory_provenance,validate.worker_identity)))
    events = []
    cuda = types.SimpleNamespace(get_device_properties=lambda *_: (events.append('properties') or types.SimpleNamespace(total_memory=16_000_000_000,name='mock16GB')),
        set_per_process_memory_fraction=lambda fraction,device=0: events.append(('cap',fraction,device)))
    fake_torch = types.SimpleNamespace(cuda=cuda)
    original_class = cache.CudaCacheReclaimer
    def mocked_worker(worker_args):
        events.append('worker')
        check('cap-before-worker', events.index(('cap',.5625,0)) < events.index('worker'))
        check('nested-cache-threshold4', worker_args.threshold_gb == 4.)
        check('nested-training-arguments-unchanged', worker_args.arguments == ['--','--run-id','cpu_v3_guard'])
        check('source-class-subclass-only', issubclass(cache.CudaCacheReclaimer, original_class))
    complete = args('complete')
    with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(cache,'main',mocked_worker):
        launch.main(complete)
    report = json.loads(complete.memory_provenance.read_text())
    ident = json.loads(complete.worker_identity.read_text())
    check('real-worker-pid-recorded', ident['pid'] == os.getpid() and ident['ppid'] == os.getppid())
    check('identity-sha-bound', report['worker_identity_sha256'] == launch.file_sha(complete.worker_identity))
    live = json.loads(Path(ident['live_memory_path']).read_text())
    check('self-live-sidecar-bound',live['pid']==os.getpid() and live['worker_identity_sha256']==launch.file_sha(complete.worker_identity))
    check('separate-global-and-self-commit-live-observations',live['sample']['system_memory']['API'].startswith('GetPerformanceInfo') and live['sample']['calling_process_commit_memory']['pid']==os.getpid())
    check('new-telemetry-dependency-bound',live['resource_telemetry_sha256']==launch.RESOURCE_TELEMETRY_SHA256)
    check('final-live-sidecar-hash',report['live_memory_sha256_after']==launch.file_sha(Path(ident['live_memory_path'])))
    check('cap-applied-and-complete', report['allocator_cap_applied'] and report['status'] == 'complete' and report['cap_bytes'] == 9_000_000_000)
    check('all-phase-completed-runtime-joined',report['all_phase_owned_memory_telemetry']['thread_join_completed'] and report['all_phase_owned_memory_telemetry']['stop_requested'])
    check('source-class-restored', cache.CudaCacheReclaimer is original_class)
    check('source-after-equality', report['memory_launcher_sha256'] == report['memory_launcher_sha256_after'])
    def failed_worker(_): raise RuntimeError('worker fixture failure')
    failed = args('failed')
    with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(cache,'main',failed_worker):
        rejects('worker-error-preserved',lambda: launch.main(failed),RuntimeError)
    failure = json.loads(failed.memory_provenance.read_text())
    check('failure-evidence-retained', failure['status'] == 'failed' and failure['failure_type'] == 'RuntimeError')
    check('all-phase-failed-runtime-joined',failure['all_phase_owned_memory_telemetry']['thread_join_completed'])
    check('class-restored-after-failure', cache.CudaCacheReclaimer is original_class)
    paired_failed = args('paired_failed')
    with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(cache,'main',failed_worker), patch.object(launch,'sample_memory',side_effect=OSError('API fixture')):
        rejects('observational-api-does-not-mask-original-worker-error', lambda: launch.main(paired_failed), RuntimeError)
    paired_report = json.loads(paired_failed.memory_provenance.read_text())
    check('failed-telemetry-retained', paired_report['telemetry_after_worker']['telemetry_failed'] and paired_report['failure_type'] == 'RuntimeError')

target = ROOT/'outputs/parakeet_turkish/memory_guard_v6_final_cpu_verification.json'
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps(dict(schema=1,passed=True,CPU_only=True,actual_GPU_policy_verification_pending=True,
    source_sha256=launch.file_sha(Path(launch.__file__)),check_count=len(checks),checks=checks),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks),source_sha256=launch.file_sha(Path(launch.__file__)))))
