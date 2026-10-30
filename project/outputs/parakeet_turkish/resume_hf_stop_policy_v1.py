"""INERT declared stop0 exploration: keep full state, bound FIRST uniform pass.

Never run against a live parent. Root must first observe the original parent and
worker exits, convert its closed retention-stop2 state with the separate CPU
converter, and authorize exclusive GPU ownership. All five BEST gates remain.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

import hf_retention_stop_policy_v1 as policy
import launch_hf_stop_policy_resume_v1 as fresh
from atomic_io_retry_v1 import read_json as read, write_json as write
from resource_watchdog_v2 import ResourceGuard, sample_memory

quality = policy.quality
HERE, ROOT = policy.HERE, policy.ROOT


def prepare(args):
    policy.require(300 < args.window_hours * 3600 <= 21600 and 5 <= args.poll_seconds <= 15, 'Bounded6h with5min save reserve/fresh5–15s polling required')
    info = policy.inspect_source(args.source_loop, args.run_id, require_closed=not args.prepare_only,
        source_closed_confirmed=args.source_closed_confirmed, exit_receipt=args.exit_receipt, allow_policy0=True)
    folder = policy.scoped(args.output_dir)
    policy.require(folder != policy.scoped(args.source_loop) and not folder.is_relative_to(HERE / 'runs' / args.run_id)
        and not (folder / 'loop_status.json').exists(), 'NEW controller evidence directory required')
    plan = {'schema': 1, 'kind': 'HF_declared_retention_stop_policy_exploration_v1', 'passed': True,
        'run_id': args.run_id, 'loop_id': folder.name, 'source_loop': str(policy.scoped(args.source_loop)),
        'source_closure_confirmed': info['source_closure_confirmed'], 'policy_before': info['policy_before'],
        'policy_after': 0, 'first_pass_target_step': info['first_pass_target_step'], 'same_run_id': True,
        'scientific_stop_policy_changed': True, 'strict_all5_BEST_gate_changed': False,
        'resource_progress_guards_changed': False, 'optimizer_sampler_RNG_counters_reset': False,
        'first_pass_boundary_policy': 'Request cooperative stop as soon as published history reaches first-pass target; record actual polling/save overshoot.',
        'heldout_TEST_evaluated': False, 'objective_complete': False, 'controller_sha256': policy.sha(__file__),
        'policy_helper': policy.record(Path(policy.__file__)), 'new_preloader': policy.record(Path(fresh.__file__)),
        'runtime_source_sha256': quality.SOURCE_PINS, 'CUDA_initialized_by_preflight': False}
    if args.prepare_only:
        plan.update(status='inert_preparation', checkpoint_copied=False, actual_conversion_required=True)
        return plan
    policy.require(args.policy_receipt is not None, 'Complete root CPU transition receipt required')
    receipt = policy.verify_receipt(args.policy_receipt, check_mutable_sources=info['policy_before'] == 2)
    policy.require(receipt['run_id'] == args.run_id and receipt['first_pass_target_step'] == info['first_pass_target_step'],
        'Policy receipt source/run/pool target differs')
    original_ready = policy.scoped(info['config']['native_backend_provenance'])
    paths = {key: folder / (args.run_id + suffix) for key, suffix in {
        'fresh_runtime': '_fresh_runtime.json', 'io': '_io.json', 'cache': '_cache.json',
        'memory': '_memory.json', 'identity': '_identity.json', 'stop': '_stop.txt', 'log': '.log'}.items()}
    policy.require(all(not path.exists() for path in paths.values()), 'NEW runtime evidence/stop/log paths required')
    later_manifest = None
    if info['policy_before'] == 2:
        policy.require(info['step'] == receipt['source_step']
            and info['config']['config_identity'] == receipt['source_config_identity']
            and all(item['sha256'] == receipt['source_evidence'][role]['sha256'] for role, item in info['evidence'].items()),
            'Initial policy transition must use the exact closed source cohort')
        archive = Path(receipt['converted_checkpoint']['path'])
    else:
        policy.require(info['config']['config_identity'] == receipt['converted_config_identity']
            and info['arm'].get('policy_receipt', {}).get('sha256') == policy.sha(args.policy_receipt),
            'Later continuation must inherit the already declared exact stop0 identity')
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / 'immutable_later_closed_stop0_latest.pt'
        if archive.exists():
            policy.require(policy.sha(archive) == info['evidence']['latest']['sha256'], 'Existing later archive differs; never overwrite')
        else: shutil.copyfile(info['latest_path'], archive)
        policy.require(policy.sha(archive) == info['evidence']['latest']['sha256'], 'Later full state byte copy differs')
        later_manifest = folder / 'later_closed_checkpoint_manifest.json'
        later_value = {'run_id': args.run_id, 'checkpoint': policy.record(archive),
            'config_identity': info['config']['config_identity'], 'policy_receipt_sha256': policy.sha(args.policy_receipt),
            'closed_source_evidence': info['evidence'], 'checkpoint_byte_copy_only': True}
        if later_manifest.exists(): policy.require(read(later_manifest) == later_value, 'Existing later byte-copy manifest differs')
        else: write(later_manifest, later_value)
    argv = list(info['arm']['worker_arguments']); original = list(argv)
    for flag, value in (('--resume-weights', None), ('--resume-exact', archive), ('--max-retention-failures', 0),
                        ('--stop-file', paths['stop']), ('--max-wall-seconds', args.window_hours * 3600 - 300)):
        quality.runtime.replace_option(argv, flag, value)
    policy.require('--resume-runtime-migration' not in argv, 'Only checked serialized stop-policy identity; no runtime migration flags')
    allowed = ('--resume-weights', '--resume-exact', '--max-retention-failures', '--stop-file', '--max-wall-seconds')
    def science(values):
        result = list(values)
        for flag in allowed: quality.runtime.replace_option(result, flag, None)
        return result
    policy.require(science(original) == science(argv), 'Another recipe/layout/data argument changed')
    memory = ['--provenance', str(paths['io']), '--cache-provenance', str(paths['cache']),
        '--memory-provenance', str(paths['memory']), '--worker-identity', str(paths['identity']),
        '--cap-gb', '8', '--threshold-gb', '4', '--', *argv]
    launch_args = ['--original-native-provenance', str(original_ready), '--original-native-provenance-sha256', policy.sha(original_ready),
        '--fresh-runtime-provenance', str(paths['fresh_runtime']), '--policy-receipt', str(args.policy_receipt)]
    if later_manifest: launch_args += ['--later-closed-checkpoint-manifest', str(later_manifest)]
    command = [str(ROOT / 'work/nemo-native-env-v2/Scripts/python.exe'), str(HERE / 'launch_hf_stop_policy_resume_v1.py'), *launch_args, '--', *memory]
    fresh.validate(argparse.Namespace(original_native_provenance=original_ready, original_native_provenance_sha256=policy.sha(original_ready),
        fresh_runtime_provenance=paths['fresh_runtime'], policy_receipt=args.policy_receipt,
        later_closed_checkpoint_manifest=later_manifest, memory_arguments=memory, validate_only=True))
    folder.mkdir(parents=True, exist_ok=True)
    plan.update(status='closed_CPU_preflight', initial_step=info['step'], inherited_selected_step=info['selection']['selected_step'],
        inherited_retention_failures=receipt['retention_failure_counter_carried'], policy_receipt=policy.record(args.policy_receipt),
        source_evidence=info['evidence'], original_ready=policy.record(original_ready), archive=policy.record(archive),
        worker_arguments=argv, command=command, paths={key: str(path) for key, path in paths.items()},
        original_config_identity=receipt['source_config_identity'], target_config_identity=receipt['converted_config_identity'],
        later_closed_checkpoint_manifest=policy.record(later_manifest) if later_manifest else None)
    return plan


def validate_fresh(value, plan, identity):
    policy.require(value.get('kind') == 'HF_declared_stop_policy_fresh_preload_v1' and value.get('status') == 'ready'
        and value.get('run_id') == plan['run_id'] and value.get('pid') == identity['pid']
        and value.get('launcher_sha256') == plan['new_preloader']['sha256']
        and value.get('policy_receipt') == plan['policy_receipt'] and value.get('source_sha256') == fresh.base.PINS
        and value.get('worker_arguments_sha256') == fresh.base.argument_sha(plan['worker_arguments'])
        and value.get('original_first_start_ready', {}).get('sha256') == plan['original_ready']['sha256']
        and value.get('scientific_stop_policy_changed') is True and value.get('strict_all5_BEST_gate_changed') is False,
        'Fresh actual process proof or explicit policy identity differs')
    return value


def main(args):
    plan = prepare(args)
    if args.prepare_only or args.validate_only:
        if args.preflight_output: write(policy.scoped(args.preflight_output), plan)
        print(policy.json.dumps({'passed': True, 'status': plan['status'], 'source_closed': plan['source_closure_confirmed'],
            'GPU_launch': False, 'Torch_imported': 'torch' in sys.modules}))
        return 0
    policy.require(args.gpu_window_authorized, 'Only root may authorize a sole GPU child')
    folder = policy.scoped(args.output_dir); paths = {key: Path(value) for key, value in plan['paths'].items()}
    state = {**plan, 'status': 'starting', 'started_at_utc': policy.now(), 'window_hours': args.window_hours}
    started = time.monotonic(); deadline = started + args.window_hours * 3600
    state['absolute_deadline_utc'] = datetime.fromtimestamp(time.time() + args.window_hours * 3600, timezone.utc).isoformat()
    child = None; stop_at = None; stop_reason = None
    resource = ResourceGuard(); progress = quality.runtime.previous.ProgressGuard(started)
    state['awake_request'] = quality.runtime.previous.prior.keep_awake(True)
    try:
        write(folder / 'loop_status.json', state)
        with paths['log'].open('x', encoding='utf-8') as log:
            child = subprocess.Popen(plan['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['owned_launcher_pid'] = child.pid
            while child.poll() is None:
                history = read(HERE / 'runs' / args.run_id / 'training_history.json', [])
                identity = quality.validate_identity(read(paths['identity'], {}), plan, child.pid, state['started_at_utc']) if paths['identity'].exists() else None
                if identity: state['fresh_runtime'] = validate_fresh(read(paths['fresh_runtime'], {}), plan, identity)
                phase = quality.runtime.previous.log_phase(paths['log'], history)
                sample = sample_memory(identity['pid'] if identity else child.pid, phase['phase'])
                reason, memory = resource.observe(sample['system_memory'], sample['worker_private_memory'], observed_monotonic=sample['observed_monotonic'])
                self_reason, self_memory = quality.runtime.previous.self_memory(Path(str(paths['identity']) + '.live_memory.json'), identity, paths['identity']) if identity else (None, {'available': False})
                progress_reason, progress_info = progress.observe(history, time.monotonic())
                reason = reason or self_reason or progress_reason
                if not identity and time.monotonic() - started > 120: reason = reason or 'owned_worker_identity_unavailable'
                if identity and not self_memory['available'] and time.monotonic() - started > 120: reason = reason or 'actual_worker_self_memory_unavailable_or_stale'
                published = history[-1]['step'] if history else plan['initial_step']
                if published >= plan['first_pass_target_step']: reason = reason or 'first_uniform_pass_boundary'
                if time.monotonic() >= deadline - 300: reason = reason or 'window_budget_checkpoint_boundary'
                state.update(status='running', phase=phase, published_step=published, progress=progress_info,
                    system_and_owned_private_memory=memory, actual_worker_self_memory=self_memory, updated_at_utc=policy.now())
                with (folder / 'memory_observations.jsonl').open('a', encoding='utf-8') as out:
                    out.write(policy.json.dumps({'at_utc': policy.now(), 'phase': phase, 'memory': memory, 'worker_self': self_memory, 'progress': progress_info}) + '\n')
                if reason and stop_at is None:
                    stop_at = time.monotonic(); stop_reason = reason
                    paths['stop'].write_text('Cooperative save/DEV requested: ' + reason + '\n', encoding='utf-8')
                    state.update(cooperative_stop_reason=reason, cooperative_stop_requested_at_utc=policy.now())
                write(folder / 'loop_status.json', state)
                if stop_at and time.monotonic() - stop_at > 300: raise RuntimeError('Cooperative grace exceeded; no further child')
                time.sleep(args.poll_seconds)
        policy.require(child.returncode == 0, 'Worker failed; preserve evidence and fix cause before any continuation')
        completion = read(paths['fresh_runtime'].with_suffix('.completion.json'), {})
        policy.require(completion.get('status') == 'complete' and completion.get('sources_unchanged') is True
            and completion.get('pid') == state['fresh_runtime']['pid']
            and completion.get('policy_receipt_sha256') == plan['policy_receipt']['sha256']
            and completion.get('fresh_runtime_proof_sha256') == policy.sha(paths['fresh_runtime']), 'Fresh completion mismatch')
        for role in ('io', 'cache', 'memory'): policy.require(read(paths[role], {}).get('status') == 'complete', 'Completion missing: ' + role)
        policy.require(policy.sha(plan['archive']['path']) == plan['archive']['sha256'], 'Immutable full input changed')
        selection_path = HERE / 'runs' / args.run_id / 'selection.json'; selection = read(selection_path, {})
        policy.require(selection.get('run_id') == args.run_id and selection.get('accepted') is True
            and selection['completed_steps'] >= plan['initial_step'], 'Durable continued selection/state missing')
        ordinary = stop_reason in (None, 'window_budget_checkpoint_boundary', 'first_uniform_pass_boundary')
        state.update(status='first_uniform_pass_ended' if ordinary and selection['completed_steps'] >= plan['first_pass_target_step']
            else 'train_window_ended' if ordinary else 'resource_review_required', child_exit_code=child.returncode,
            resource_review_required=not ordinary, completed_steps=selection['completed_steps'], selected_step=selection['selected_step'],
            selection=policy.record(selection_path), first_pass_completed=selection['completed_steps'] >= plan['first_pass_target_step'],
            first_pass_optimizer_overshoot=max(0, selection['completed_steps'] - plan['first_pass_target_step']),
            stop_early_reason=selection['stop_early_reason'], finished_at_utc=policy.now(), objective_complete=False)
        return 0 if ordinary else 1
    except BaseException as exception:
        state.update(status='failed', error=repr(exception), finished_at_utc=policy.now(), objective_complete=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists(): paths['stop'].write_text('Controller exit: cooperative save/DEV requested\n', encoding='utf-8')
            try: child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate(); child.wait(timeout=30)
        state['awake_clear'] = quality.runtime.previous.prior.keep_awake(False)
        write(folder / 'loop_status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-loop', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exit-receipt', type=Path)
    parser.add_argument('--policy-receipt', type=Path)
    parser.add_argument('--source-closed-confirmed', action='store_true')
    parser.add_argument('--window-hours', type=float, default=4.5)
    parser.add_argument('--poll-seconds', type=float, default=5.)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--gpu-window-authorized', action='store_true')
    sys.exit(main(parser.parse_args()))
