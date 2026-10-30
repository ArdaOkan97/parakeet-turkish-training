"""Run an unchanged local worker with versioned atomic-I/O adapters.

Usage: python launch_with_io_retry_v7.py --worker train_lora_batched.py
       --provenance PATH -- [the worker's original arguments]
Only JSON persistence and training-checkpoint commit paths are adapted.
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from atomic_io_retry_v1 import commit_writer, write_json

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
ALLOWED_WORKERS = ('train_lora_batched_data_v9.py', 'finalize_lora.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    if args.worker not in ALLOWED_WORKERS:
        raise ValueError('Only the explicitly scoped frozen workers are supported')
    provenance = args.provenance.resolve()
    if not provenance.is_relative_to(ROOT.resolve()):
        raise ValueError('Provenance path must be inside this workspace')
    worker_path = OUTPUT / args.worker
    worker_sha = sha(worker_path)
    if args.expected_worker_sha and worker_sha != args.expected_worker_sha:
        raise RuntimeError('Frozen worker source differs from the controller expectation')
    events = []
    def event(value):
        events.append(value)
        print(json.dumps({'atomic_io_retry': value}), flush=True)
    atomic_write = functools.partial(write_json, retry_seconds=args.retry_seconds, event=event)
    evidence = {'schema': 1, 'kind': 'atomic_io_retry_v7', 'worker': str(worker_path),
                'worker_sha256': worker_sha, 'launcher_sha256': sha(__file__),
                'helper_sha256': sha(OUTPUT / 'atomic_io_retry_v1.py'),
                'retry_winerrors': [5, 32, 33], 'retry_seconds': args.retry_seconds,
                'unique_same_directory_temporary': True, 'fsync_before_replace': True,
                'checkpoint_policy': 'Only original DEV/start/final save points; original serializer and complete state unchanged.',
                'training_semantics_changed': False, 'arguments': args.arguments,
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    atomic_write(provenance, evidence)
    # Patch before imports, so existing from-import aliases receive the adapter.
    import benchmark
    benchmark.write_json = atomic_write
    worker = importlib.import_module(Path(args.worker).stem)
    worker.write_json = atomic_write
    if args.worker == 'train_lora_batched_data_v9.py':
        original_save = worker.save_training_state
        @functools.wraps(original_save)
        def save(path, *values, **keywords):
            return commit_writer(path, lambda target: original_save(target, *values, **keywords),
                                 retry_seconds=args.retry_seconds, event=event)
        worker.save_training_state = save
    arguments = list(args.arguments)
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    try:
        if args.worker == 'train_lora_batched_data_v9.py':
            parsed = worker.validate_arguments(worker.parser().parse_args(arguments))
            worker.run(parsed)
        else:
            worker_parser = argparse.ArgumentParser(description=worker.__doc__)
            worker_parser.add_argument('--run-id', default='lora_20261010_01')
            worker.run(worker_parser.parse_args(arguments))
        evidence['status'] = 'complete'
    except BaseException as exception:
        evidence.update(status='failed', failure_type=type(exception).__name__, failure_message=str(exception))
        raise
    finally:
        evidence.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), retry_events=events,
                        worker_sha256_after=sha(worker_path), launcher_sha256_after=sha(__file__),
                        helper_sha256_after=sha(OUTPUT / 'atomic_io_retry_v1.py'))
        atomic_write(provenance, evidence)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--worker', choices=ALLOWED_WORKERS, required=True)
    result.add_argument('--provenance', type=Path, required=True)
    result.add_argument('--retry-seconds', type=float, default=30.)
    result.add_argument('--expected-worker-sha')
    result.add_argument('arguments', nargs=argparse.REMAINDER)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
