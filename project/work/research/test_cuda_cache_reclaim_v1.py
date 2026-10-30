"""CPU mocked call-scope, errors, cache threshold and launcher tests."""
import json
import random
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
import cuda_cache_reclaim_v1 as helper
import launch_with_cache_reclaim_v1 as launcher
import launch_with_io_retry_v1 as original_launcher
from atomic_io_retry_v1 import write_json, read_json


class FakeCuda:
    def __init__(self):
        self.reserved = 9_000_000_000
        self.allocated = 2_850_000_000
        self.peak = 6_126_287_872
        self.calls = self.clears = self.stats_calls = 0
        self.original_error = self.cleanup_error = False
        self.last_arguments = None

    def synchronize(self, *args, **kwargs):
        self.calls += 1
        self.last_arguments = (args, kwargs)
        if self.original_error:
            raise LookupError('original synchronize error')
        return 'original sentinel'

    def empty_cache(self):
        self.clears += 1
        if self.cleanup_error:
            raise ArithmeticError('cleanup error')
        self.reserved = 3_000_000_000

    def memory_reserved(self):
        self.stats_calls += 1
        return self.reserved

    def memory_allocated(self):
        self.stats_calls += 1
        return self.allocated

    def max_memory_allocated(self):
        self.stats_calls += 1
        return self.peak

    def is_initialized(self):
        return False


def main():
    checks = []
    rng_before = random.getstate()
    def require(value, name):
        assert value, name
        checks.append(name)
    worker = OUTPUT/'train_lora_batched.py'
    originals = {path: helper.file_sha(path) for path in
                 [worker, OUTPUT/'run_experiment_loop_recovery_v2.py',
                  OUTPUT/'launch_with_io_retry_v1.py', OUTPUT/'atomic_io_retry_v1.py']}
    require(helper.validated_update_sync_line(worker) == 539, 'Exact frozen worker optimizer-end source location')
    try:
        helper.validated_update_sync_line(worker, '0'*64)
    except RuntimeError:
        checks.append('Changed frozen worker SHA rejected before any CUDA calls')
    else:
        raise AssertionError('worker SHA')
    frame = SimpleNamespace(f_code=SimpleNamespace(co_filename=str(worker.resolve()), co_name='run'), f_lineno=539)
    cuda = FakeCuda()
    hook = helper.CudaCacheReclaimer(cuda, worker, max_events=2)
    hook._caller = lambda: frame
    original = cuda.synchronize
    hook.install()
    require(cuda.synchronize(123, device='mock') == 'original sentinel', 'Original synchronize return preserved')
    require(cuda.last_arguments == ((123,), {'device': 'mock'}), 'Original positional/keyword arguments preserved')
    require(cuda.clears == 1 and cuda.reserved == 3_000_000_000, 'Threshold-triggered unused cache cleanup')
    require(cuda.allocated == 2_850_000_000 and cuda.peak == 6_126_287_872, 'Live allocation and historical peak untouched')
    require(hook.events[-1]['before']['reserved_bytes'] == 9_000_000_000
            and hook.events[-1]['after']['reserved_bytes'] == 3_000_000_000, 'Before/after cache stats retained')
    require(hook.events[-1]['cleanup_seconds'] >= 0, 'Cleanup duration captured inside wrapped measured call')
    for name, change in [('evaluation function', {'co_name': 'evaluate'}),
                         ('microbatch function', {'co_name': 'batch_objective'}),
                         ('other caller file', {'co_filename': str(ROOT/'work/other.py')}),
                         ('other line in run', {'line': 540})]:
        candidate = SimpleNamespace(f_code=SimpleNamespace(co_filename=change.get('co_filename',str(worker.resolve())),
                                    co_name=change.get('co_name','run')), f_lineno=change.get('line',539))
        hook._caller = lambda candidate=candidate: candidate
        before = (cuda.clears, cuda.stats_calls)
        require(cuda.synchronize() == 'original sentinel' and (cuda.clears, cuda.stats_calls) == before,
                'No statistics/cleanup at '+name)
    hook._caller = lambda: frame
    for value in (8_000_000_000, 3_000_000_000):
        cuda.reserved = value
        count = cuda.clears
        cuda.synchronize()
        require(cuda.clears == count, 'No cleanup at or below threshold '+str(value))
    cuda.original_error = True
    before = (cuda.stats_calls, cuda.clears, hook.eligible_calls)
    try:
        cuda.synchronize()
    except LookupError as error:
        require(str(error) == 'original synchronize error' and (cuda.stats_calls,cuda.clears,hook.eligible_calls) == before,
                'Original synchronize error propagates without side effects')
    else:
        raise AssertionError('original error')
    cuda.original_error = False
    cuda.cleanup_error = True
    cuda.reserved = 9_000_000_000
    try:
        cuda.synchronize()
    except ArithmeticError:
        require(hook.failed_reclamations == 1 and hook.events[-1]['status'] == 'failed',
                'Cleanup error propagates and leaves bounded failure evidence')
    else:
        raise AssertionError('cleanup error')
    cuda.cleanup_error = False
    for _ in range(5):
        cuda.reserved = 9_000_000_000
        cuda.synchronize()
    evidence = hook.evidence()
    require(len(evidence['events']) == 2 and evidence['events_omitted'] == 5, 'Recorded events bounded while total cleanup counts retained')
    require(all(evidence[key] is False for key in ('changes_training_math','changes_backend_flags','changes_rng_or_sampler',
                    'changes_optimizer_state','changes_cuda_allocator_policy')), 'No math/backend/RNG/optimizer/global allocator-policy change claimed')
    hook.restore()
    require(cuda.synchronize == original, 'Original synchronize restored')
    hook.restore()
    require(cuda.synchronize == original, 'Idempotent final restoration')
    for threshold, events in [(0,128), (13,128), (8,0), (8,1025)]:
        try:
            helper.CudaCacheReclaimer(cuda,worker,threshold_gb=threshold,max_events=events)
        except ValueError:
            checks.append('Conservative threshold/event bounds rejected '+str((threshold,events)))
        else:
            raise AssertionError('bounds')
    scratch = ROOT/'work/research/cache_reclaim_cpu'/uuid.uuid4().hex
    scratch.mkdir(exist_ok=True)
    args = launcher.parser().parse_args(['--provenance',str(scratch/'io.json'), '--cache-provenance',str(scratch/'cache.json'),
                                         '--','--run-id','CPU_MOCK','--resume-exact','owned_checkpoint.pt'])
    fake = FakeCuda()
    def mocked_run(original_args):
        require(original_args.arguments == args.arguments and original_args.expected_worker_sha == helper.FROZEN_TRAINER_SHA256,
                'Nested frozen I/O launcher receives exact unchanged worker argv/SHA')
        require(fake.synchronize() == 'original sentinel' and fake.clears == 0,
                'Launcher hook ignores non-trainer caller in CPU mocked lifecycle')
    with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=fake)}), patch.object(original_launcher,'main',side_effect=mocked_run):
        launcher.main(args)
    saved = read_json(args.cache_provenance)
    require(saved['status'] == 'complete' and saved['active_process_attachment'] is False, 'New-process launcher emits complete separate provenance')
    require(saved['launcher_sha256'] == saved['launcher_sha256_after']
            and saved['cache_helper_sha256'] == saved['cache_helper_sha256_after'], 'Actual launcher/helper before and after hashes match')
    require(fake.synchronize.__func__ is FakeCuda.synchronize, 'Launcher restores original CUDA call after worker exit')
    try:
        launcher.main(args)
    except FileExistsError:
        checks.append('Existing cache/I/O evidence cannot be overwritten by another launch')
    else:
        raise AssertionError('overwrite')
    failure_args = launcher.parser().parse_args(['--provenance',str(scratch/'failed_io.json'),
                           '--cache-provenance',str(scratch/'failed_cache.json'), '--','--resume-exact','owned.pt'])
    with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=fake)}), \
         patch.object(original_launcher,'main',side_effect=RuntimeError('mock worker failure')):
        try:
            launcher.main(failure_args)
        except RuntimeError as error:
            require(str(error) == 'mock worker failure', 'Original worker error is propagated')
        else:
            raise AssertionError('worker error')
    require(fake.synchronize.__func__ is FakeCuda.synchronize
            and read_json(failure_args.cache_provenance)['status'] == 'failed',
            'Exception exit restores original CUDA function and saves failed provenance')
    require(random.getstate() == rng_before, 'Python RNG state unchanged by helper and launcher guards')
    for path, digest in originals.items():
        require(helper.file_sha(path) == digest, 'Frozen original source unchanged '+path.name)
    report = {'passed':True,'checks':checks,'check_count':len(checks),
              'scope':'CPU mocked CUDA calls and nested launcher; no CUDA/GPU initialization/inference/training',
              'helper_sha256':helper.file_sha(OUTPUT/'cuda_cache_reclaim_v1.py'),
              'launcher_sha256':helper.file_sha(OUTPUT/'launch_with_cache_reclaim_v1.py'),
              'guard_sha256':helper.file_sha(__file__)}
    write_json(OUTPUT/'cuda_cache_reclaim_v1_cpu_verification.json',report)
    print(json.dumps({'passed':True,'check_count':len(checks),'helper_sha256':report['helper_sha256'],
                      'launcher_sha256':report['launcher_sha256']}))


if __name__ == '__main__':
    main()
