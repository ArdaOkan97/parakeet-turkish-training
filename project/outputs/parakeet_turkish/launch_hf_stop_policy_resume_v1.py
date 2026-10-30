"""Fresh preload for DECLARED stop-policy0 exploration, using the frozen chain."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys

import launch_same_science_resume_v1 as base
import hf_retention_stop_policy_v1 as policy


def validate(args):
    values, worker, original, old, fresh, completion = base.validate(args)
    receipt = policy.verify_receipt(args.policy_receipt)
    base.require(receipt['run_id'] == old['run_id'] and base.option(worker, '--max-retention-failures') == '0',
        'Explicit same-ID stop0 transition required')
    exact = policy.scoped(base.option(worker, '--resume-exact'))
    if exact != Path(receipt['converted_checkpoint']['path']).resolve():
        base.require(args.later_closed_checkpoint_manifest is not None, 'Later stop0 closure needs its own immutable byte-copy proof')
        continuation = policy.read_json(policy.scoped(args.later_closed_checkpoint_manifest))
        base.require(continuation['policy_receipt_sha256'] == policy.sha(args.policy_receipt)
            and continuation['run_id'] == receipt['run_id'] and continuation['checkpoint']['path'] == str(exact)
            and continuation['checkpoint']['sha256'] == policy.sha(exact)
            and continuation['config_identity'] == receipt['converted_config_identity'],
            'Later closed-policy checkpoint proof differs')
    return values, worker, original, old, fresh, completion, receipt


def main(args):
    values, worker, original, old, fresh, completion, receipt = validate(args)
    if args.validate_only:
        print(policy.json.dumps({'passed': True, 'scientific_stop_policy_changed': True,
            'Torch_imported': 'torch' in sys.modules, 'GPU_launch': False}))
        return
    base.require(Path(sys.executable).resolve() == base.ROOT / 'work/nemo-native-env-v2/Scripts/python.exe', 'Use native-env-v2')
    base.require('torch' not in sys.modules, 'Actual fresh preload must precede Torch')
    evidence = {'schema': 1, 'kind': 'HF_declared_stop_policy_fresh_preload_v1', 'status': 'starting',
        'run_id': old['run_id'], 'pid': os.getpid(), 'ppid': os.getppid(), 'started_at_utc': policy.now(),
        'source_sha256': base.PINS, 'launcher_sha256': policy.sha(__file__),
        'policy_helper_sha256': policy.sha(policy.__file__), 'base_preloader_sha256': policy.sha(base.__file__),
        'worker_arguments_sha256': base.argument_sha(worker), 'policy_receipt': policy.record(args.policy_receipt),
        'scientific_stop_policy_changed': True, 'training_math_optimizer_sampler_changed': False,
        'strict_all5_BEST_gate_changed': False, 'policy_transition': receipt['policy_transition'],
        'original_first_start_ready': {'path': str(original), 'sha256': policy.sha(original), 'original_first_start_pid': old.get('pid')},
        'original_ready_scope': 'Immutable original identity only; its old PID is not this process'}
    previous = sys.argv; outcome = 'failed'; error = None; wrote = False
    try:
        from preload_nemo_linker_v1 import preload
        actual = preload()
        base.require(actual.get('kind') == 'coherent_native_cuda12_9_preload_v1'
            and actual.get('module_sha256') == base.PINS['preload_nemo_linker_v1.py']
            and actual.get('actual_binding_linker', {}).get('version') == [12, 9]
            and actual.get('cuda_initialized') is False, 'Actual coherent preload failed')
        evidence.update(status='ready', preload=actual, ready_at_utc=policy.now())
        base.write_new(fresh, evidence); wrote = True
        sys.argv = [str(base.HERE / 'launch_with_memory_guard_v15.py'), *values]
        runpy.run_path(str(base.HERE / 'launch_with_memory_guard_v15.py'), run_name='__main__')
        outcome = 'complete'
    except BaseException as exception:
        if isinstance(exception, SystemExit) and exception.code in (None, 0): outcome = 'complete'
        else: error = repr(exception)
        raise
    finally:
        sys.argv = previous
        if not wrote:
            evidence.update(status='failed', error=error); base.write_new(fresh, evidence)
        unchanged = policy.sha(__file__) == evidence['launcher_sha256'] and policy.sha(original) == args.original_native_provenance_sha256
        unchanged = unchanged and policy.sha(policy.__file__) == evidence['policy_helper_sha256']
        unchanged = unchanged and policy.sha(base.__file__) == evidence['base_preloader_sha256']
        unchanged = unchanged and policy.sha(args.policy_receipt) == evidence['policy_receipt']['sha256']
        unchanged = unchanged and all(policy.sha(base.HERE / name) == digest for name, digest in base.PINS.items())
        base.write_new(completion, {'schema': 1, 'kind': 'HF_declared_stop_policy_fresh_preload_completion_v1',
            'status': outcome, 'run_id': old['run_id'], 'pid': os.getpid(), 'sources_unchanged': unchanged,
            'fresh_runtime_proof_sha256': policy.sha(fresh), 'policy_receipt_sha256': policy.sha(args.policy_receipt),
            'error': error, 'finished_at_utc': policy.now()})
        base.require(unchanged, 'Frozen source/original ready changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-native-provenance', type=Path, required=True)
    parser.add_argument('--original-native-provenance-sha256', required=True)
    parser.add_argument('--fresh-runtime-provenance', type=Path, required=True)
    parser.add_argument('--policy-receipt', type=Path, required=True)
    parser.add_argument('--later-closed-checkpoint-manifest', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('memory_arguments', nargs=argparse.REMAINDER)
    main(parser.parse_args())
