"""NEW-process data-v2 trainer dispatch through unchanged scoped cache/I/O helpers.

Cannot attach to or alter an already running worker. Use only after an owned
cooperative checkpoint stop and authorized exact-resume orchestration.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from atomic_io_retry_v1 import write_json
from cuda_cache_reclaim_v1 import CudaCacheReclaimer, file_sha
FROZEN_TRAINER_SHA256 = '9f158449e181768489c64dfbe9060c81c4166ec4a8999340616124bf4e8f6f5a'

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
FROZEN_IO_LAUNCHER_SHA256 = '98032d4f56e56c6441b5ab3551cbd52f0b6681d9b2f312fb307cbc9b285ef00f'
FROZEN_IO_HELPER_SHA256 = '2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0'


def main(args):
    worker_path = OUTPUT/'train_lora_batched_data_v2.py'
    for path, expected in [(OUTPUT/'launch_with_io_retry_v1.py', FROZEN_IO_LAUNCHER_SHA256),
                           (OUTPUT/'atomic_io_retry_v1.py', FROZEN_IO_HELPER_SHA256)]:
        if file_sha(path) != expected:
            raise RuntimeError('Frozen I/O adapter source differs: '+path.name)
    for path in (args.provenance, args.cache_provenance):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Both provenance paths must belong to this workspace')
        if path.exists():
            raise FileExistsError('Use a new distinct provenance path; existing evidence is immutable: '+str(path))
    if args.provenance.resolve() == args.cache_provenance.resolve():
        raise ValueError('Cache and I/O provenance require distinct files')
    import torch
    reclaimer = CudaCacheReclaimer(torch.cuda, worker_path, threshold_gb=args.threshold_gb,
                                   max_events=args.max_events, expected_sha=FROZEN_TRAINER_SHA256)
    sources = {'launcher_sha256': file_sha(__file__),
               'cache_helper_sha256': file_sha(OUTPUT/'cuda_cache_reclaim_v1.py'),
               'frozen_io_launcher_sha256': FROZEN_IO_LAUNCHER_SHA256,
               'frozen_io_helper_sha256': FROZEN_IO_HELPER_SHA256,
               'frozen_worker_sha256': FROZEN_TRAINER_SHA256}
    if args.validate_only:
        print({'passed': True, 'CUDA_initialized': torch.cuda.is_initialized(),
               'optimizer_end_line': reclaimer.line, **sources})
        return
    evidence = {'schema': 1, 'kind': 'cache_reclamation_training_launcher_v2', **sources,
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'arguments': args.arguments, 'io_provenance_path': str(args.provenance.resolve()),
                'active_process_attachment': False}
    write_json(args.cache_provenance, evidence)
    # Existing I/O evidence remains unchanged in meaning; cache evidence is a
    # separate mandatory provenance record for any successor controller/auditor.
    import launch_with_io_retry_v1 as io_launcher
    reclaimer.install()
    try:
        io_launcher.main(SimpleNamespace(worker='train_lora_batched_data_v2.py', provenance=args.provenance,
                         expected_worker_sha=FROZEN_TRAINER_SHA256,
                         retry_seconds=args.retry_seconds, arguments=args.arguments))
        evidence['status'] = 'complete'
    except BaseException as exception:
        evidence.update(status='failed', failure_type=type(exception).__name__, failure_message=str(exception))
        raise
    finally:
        try:
            reclaimer.restore()
        finally:
            evidence.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                            cache_reclamation=reclaimer.evidence(),
                            launcher_sha256_after=file_sha(__file__),
                            cache_helper_sha256_after=file_sha(OUTPUT/'cuda_cache_reclaim_v1.py'),
                            frozen_io_launcher_sha256_after=file_sha(OUTPUT/'launch_with_io_retry_v1.py'),
                            frozen_io_helper_sha256_after=file_sha(OUTPUT/'atomic_io_retry_v1.py'))
            write_json(args.cache_provenance, evidence)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--provenance', type=Path, required=True)
    result.add_argument('--cache-provenance', type=Path, required=True)
    result.add_argument('--threshold-gb', type=float, default=8.)
    result.add_argument('--max-events', type=int, default=128)
    result.add_argument('--retry-seconds', type=float, default=30.)
    result.add_argument('--validate-only', action='store_true')
    result.add_argument('arguments', nargs=argparse.REMAINDER)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
