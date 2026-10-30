"""CPU-only audit of the first same-run checkpoint written after exact resume.

Run after progress is logged beyond the paused update and before the next DEV
checkpoint overwrites latest.pt. Every saved field/tensor must be identical to
the immutable pause state, apart from clearing its cooperative stop_reason.
This reads artifacts only; it never signals processes or invokes CUDA/model
inference. A consistency pass does not assert future numerical reproducibility.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
DEFAULT_LOOP = 'adaptive_20261010_01'
PAUSED_STEP = 1159


class AuditFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_json_with_hash(path):
    content = Path(path).read_bytes()
    return json.loads(content.decode('utf-8')), hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def local_path(value, label, under=ROOT):
    require(isinstance(value, (str, Path)), label + ': expected a local path.')
    path = Path(value).resolve()
    require(path.is_relative_to(Path(under).resolve()), label + ': path escapes the expected workspace directory.')
    require(path.is_file(), label + ': file is missing.')
    return path


def bound_file(value, expected_sha, label, under=ROOT):
    path = local_path(value, label, under)
    require(isinstance(expected_sha, str) and sha(path) == expected_sha, label + ': SHA256 differs.')
    return path


def exact_equal(expected, actual, path, stats, torch):
    """Compare logical values exactly, including tensor dtype/shape/stride/layout."""
    stats['nodes_compared'] += 1
    require(type(expected) is type(actual), path + ': Python types differ.')
    if isinstance(expected, torch.Tensor):
        require(expected.device.type == actual.device.type == 'cpu', path + ': tensor was not loaded on CPU.')
        require(expected.dtype == actual.dtype and expected.shape == actual.shape
                and expected.layout == actual.layout and expected.stride() == actual.stride(),
                path + ': tensor dtype/shape/layout/stride differs.')
        require(torch.equal(expected, actual), path + ': tensor values differ.')
        stats['tensor_leaves'] += 1
        stats['tensor_elements'] += expected.numel()
    elif isinstance(expected, dict):
        require(expected.keys() == actual.keys(), path + ': dictionary keys differ.')
        stats['dictionary_nodes'] += 1
        for key, value in expected.items():
            exact_equal(value, actual[key], path + '[' + repr(key) + ']', stats, torch)
    elif isinstance(expected, (list, tuple)):
        require(len(expected) == len(actual), path + ': sequence lengths differ.')
        stats['sequence_nodes'] += 1
        for index, value in enumerate(expected):
            exact_equal(value, actual[index], path + '[' + str(index) + ']', stats, torch)
    elif isinstance(expected, float):
        require(math.isfinite(expected) and math.isfinite(actual)
                and struct.pack('!d', expected) == struct.pack('!d', actual), path + ': floating-point scalar bits differ/nonfinite.')
        stats['scalar_leaves'] += 1
    elif expected is None or isinstance(expected, (str, int, bool)):
        require(expected == actual, path + ': scalar values differ.')
        stats['scalar_leaves'] += 1
    else:
        raise AuditFailure(path + ': unsupported checkpoint value type ' + type(expected).__name__)


def compare_checkpoints(paused, resumed, torch):
    require(isinstance(paused,dict) and isinstance(resumed,dict) and paused.keys() == resumed.keys(),
            'Top-level checkpoint keys differ.')
    require(paused.get('stop_reason') == 'controller_stop_file' and resumed.get('stop_reason') is None,
            'The sole allowed stop_reason reset is absent or differs.')
    stats = {'nodes_compared':0,'dictionary_nodes':0,'sequence_nodes':0,'tensor_leaves':0,'tensor_elements':0,'scalar_leaves':0}
    sections = {}
    for key,value in paused.items():
        if key == 'stop_reason':
            continue
        before = stats.copy()
        exact_equal(value,resumed[key],'checkpoint['+repr(key)+']',stats,torch)
        sections[key] = {name:stats[name]-before[name] for name in stats}
    require(len(paused['weights']) == len(resumed['weights']) and len(paused['optimizer']['state']) == 207,
            'Actual pause checkpoint does not contain the expected 207 Adam parameter states.')
    return stats, sections


def metadata(args):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.loop_id) is not None, 'Unsafe loop ID.')
    loop_dir = OUTPUT / 'runs' / args.loop_id
    pause_path = local_path(args.pause_manifest or loop_dir / 'gpu_benchmark_pause.json', 'Completed pause manifest', loop_dir)
    pause, pause_sha = load_json_with_hash(pause_path)
    require(pause.get('schema') == 1 and pause.get('loop_id') == args.loop_id
            and pause.get('run_id') == args.loop_id + '_screen_A' and pause.get('paused_step') == PAUSED_STEP,
            'Pause loop/run/update identity differs.')
    require(pause.get('original_controller_exited') is True and pause.get('benchmark_completed') is True,
            'Original exit and benchmark completion are not verified.')
    run_id = pause['run_id']
    run_dir = OUTPUT / 'runs' / run_id
    archived = bound_file(pause['checkpoint_path'], pause['checkpoint_sha256'], 'Archived pause checkpoint', ROOT / 'work')
    latest = local_path(ROOT / 'work' / (run_id + '_latest.pt'), 'Actual resumed latest checkpoint', ROOT / 'work')
    require(latest != archived, 'Latest checkpoint must differ from its immutable archive path.')
    paused_config_path = bound_file(pause['paused_config_path'], pause['paused_config_sha256'], 'Paused config snapshot', loop_dir)
    paused_history_path = bound_file(pause['paused_history_path'], pause['paused_history_sha256'], 'Paused history snapshot', loop_dir)
    paused_selection_path = bound_file(pause['paused_selection_path'], pause['paused_selection_sha256'], 'Paused selection snapshot', loop_dir)
    state_path = local_path(loop_dir / 'loop_status.json', 'Live continuation state', loop_dir)
    state, state_sha = load_json_with_hash(state_path)
    require(state.get('loop_id') == args.loop_id and state.get('active_run_id') == run_id,
            'The expected exact-resumed screen is not the active owned stage.')
    original_controller = bound_file(OUTPUT / 'run_experiment_loop.py', pause['original_controller_sha256'],
                                    'Original frozen controller', OUTPUT)
    require(state.get('controller_sha256') == pause['original_controller_sha256'], 'Original controller identity differs.')
    continuation = state.get('continuation', {})
    controller_path = bound_file(continuation.get('controller_path'), continuation.get('controller_sha256'),
                                 'Actual continuation controller', OUTPUT)
    require(controller_path == (OUTPUT / 'run_experiment_loop_continued.py').resolve(), 'Unexpected continuation controller path.')
    require(Path(continuation.get('pause_manifest_path', '')).resolve() == pause_path
            and continuation.get('pause_manifest_sha256') == pause_sha
            and Path(continuation.get('checkpoint_path', '')).resolve() == archived
            and continuation.get('checkpoint_sha256') == pause['checkpoint_sha256']
            and continuation.get('checkpoint_step') == PAUSED_STEP, 'Continuation does not bind the immutable exact-resume checkpoint.')
    immutable_path = bound_file(continuation.get('benchmark_checkpoint_manifest_path'),
        continuation.get('benchmark_checkpoint_manifest_sha256'), 'Immutable GPU benchmark pause manifest', loop_dir)
    immutable = load_json(immutable_path)
    require(immutable.get('checkpoint_sha256') == pause['checkpoint_sha256'] and immutable.get('paused_step') == PAUSED_STEP,
            'Benchmark and exact resume did not use the same immutable pause checkpoint.')
    require(continuation.get('benchmark_evidence') == pause.get('benchmark_evidence') and bool(pause['benchmark_evidence']),
            'Continuation benchmark-evidence list differs.')
    for record in pause['benchmark_evidence']:
        bound_file(record.get('path'), record.get('sha256'), 'Completed benchmark evidence')
    origin_path = bound_file(pause['original_status_path'], pause['original_status_sha256'], 'Original budget snapshot', loop_dir)
    origin = load_json(origin_path)
    require(state.get('started_at_utc') == origin.get('started_at_utc')
            and state.get('deadline_at_utc') == origin.get('deadline_at_utc')
            and state.get('initial_window_hours') == origin.get('initial_window_hours') == 12.,
            'Original absolute 12-hour deadline/start changed.')
    config_path = local_path(run_dir / 'run_config.json', 'Actual resumed config', run_dir)
    config, config_sha = load_json_with_hash(config_path)
    paused_config = load_json(paused_config_path)
    warm = config.get('warm_start', {})
    require(config.get('run_id') == paused_config.get('run_id') == run_id and config.get('steps') == paused_config.get('steps') == 3204,
            'Actual resumed run/update cap differs.')
    require(warm.get('mode') == 'exact_resume' and warm.get('source_run_id') == run_id
            and warm.get('source_step') == PAUSED_STEP and warm.get('sha256') == pause['checkpoint_sha256']
            and Path(warm.get('path', '')).resolve() == archived
            and Path(config.get('resume_exact', '')).resolve() == archived and config.get('resume_weights') is None,
            'Live run_config does not prove exact same-checkpoint resume.')
    require(config.get('config_identity') == paused_config.get('config_identity'), 'Saved exact optimization/data/source identity changed.')
    for name, digest in config.get('code_sha256', {}).items():
        require(Path(name).name == name, 'Unsafe frozen trainer source filename.')
        bound_file(OUTPUT / name, digest, 'Frozen source ' + name, OUTPUT)
    for value, digest in config.get('input_sha256', {}).items():
        bound_file(value, digest, 'Frozen training input')
    history_path = local_path(run_dir / 'training_history.json', 'Progress history', run_dir)
    history, history_sha = load_json_with_hash(history_path)
    require(isinstance(history, list) and len(history) > PAUSED_STEP
            and all(row.get('step') == index for index,row in enumerate(history,1)),
            'No correctly ordered training progress beyond the pause update has been recorded.')
    matches = [entry for entry in state.get('trials', []) if entry.get('run_id') == run_id]
    require(len(matches) == 1 and matches[0].get('optimizer_resume') is True
            and matches[0].get('resume_start_step') == PAUSED_STEP
            and matches[0].get('resume_checkpoint_sha256') == pause['checkpoint_sha256'],
            'Controller stage does not prove same-state optimizer resume.')
    command = matches[0].get('resume_command', [])
    require(command.count('--resume-exact') == 1 and '--resume-weights' not in command
            and Path(command[command.index('--resume-exact')+1]).resolve() == archived
            and '--steps' not in command, 'Controller exact-resume argv differs.')
    return {'pause':pause,'pause_path':pause_path,'archived':archived,'latest':latest,
        'paused_config':paused_config,'config':config,'config_path':config_path,'history':history,'history_path':history_path,
        'paused_history':load_json(paused_history_path),'paused_selection':load_json(paused_selection_path),
        'state_path':state_path,'state':state,'controller_path':controller_path,'original_controller':original_controller,
        'immutable_path':immutable_path,'loop_dir':loop_dir,'run_id':run_id,'pause_sha_at_read':pause_sha,
        'config_sha_at_read':config_sha,'history_sha_at_read':history_sha,'state_sha_at_read':state_sha}


def audit(args):
    evidence = metadata(args)
    if args.metadata_only:
        return {'passed':True,'status':'metadata_verified','cpu_only':True,'checkpoint_values_compared':False,
                'loop_id':args.loop_id,'run_id':evidence['run_id'],'paused_step':PAUSED_STEP,
                'progress_step':evidence['history'][-1]['step']}
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # A caller may already have initialized its CPU interop pool.
    before_latest_sha = sha(evidence['latest'])
    paused = torch.load(evidence['archived'], map_location='cpu', weights_only=True)
    resumed = torch.load(evidence['latest'], map_location='cpu', weights_only=True)
    require(paused.get('schema') == resumed.get('schema') == 2
            and paused.get('run_id') == resumed.get('run_id') == evidence['run_id']
            and paused.get('step') == resumed.get('step') == PAUSED_STEP,
            'Latest is no longer the initial resumed checkpoint; audit before the next DEV save.')
    stats, section_stats = compare_checkpoints(paused,resumed,torch)
    exact_equal(paused['config_identity'], evidence['config']['config_identity'], 'actual_run_config_identity', stats, torch)
    exact_equal(paused['history'], evidence['paused_history'], 'immutable_paused_history', stats, torch)
    exact_equal(paused['history'], evidence['history'][:PAUSED_STEP], 'resumed_history_preserved_prefix', stats, torch)
    for key in ('selected_step','best_score','patience','retention_failures','next_eval_epoch'):
        require(key in resumed, 'Missing restored selection/schedule counter: '+key)
    mutable_config_keys = {'resume_weights','resume_exact','warm_start','max_wall_seconds'}
    old_config = {key:value for key,value in evidence['paused_config'].items() if key not in mutable_config_keys}
    new_config = {key:value for key,value in evidence['config'].items() if key not in mutable_config_keys}
    exact_equal(old_config, new_config, 'unchanged_run_config', stats, torch)
    require(sha(evidence['archived']) == evidence['pause']['checkpoint_sha256'], 'Immutable archived checkpoint changed during audit.')
    require(sha(evidence['latest']) == before_latest_sha, 'Latest checkpoint changed during audit; discard this result and retry.')
    require(sha(evidence['config_path']) == evidence['config_sha_at_read'], 'Resumed config changed during audit.')
    require(sha(evidence['pause_path']) == evidence['pause_sha_at_read'], 'Completed pause manifest changed during audit.')
    return {'passed':True,'status':'complete','cpu_only':True,'checkpoint_values_compared':True,
        'loop_id':args.loop_id,'run_id':evidence['run_id'],'paused_step':PAUSED_STEP,
        'progress_step':evidence['history'][-1]['step'],'additional_updates_recorded':len(evidence['history'])-PAUSED_STEP,
        'checkpoint_sections_compared':len(section_stats),'adam_parameter_states_compared':len(paused['optimizer']['state']),
        'allowed_checkpoint_differences':[{'field':'stop_reason','paused':'controller_stop_file','resumed':None}],
        'all_other_checkpoint_fields_exact':True,'comparisons':stats,'section_comparisons':section_stats,
        'provenance':{'pause_manifest_path':str(evidence['pause_path']),'pause_manifest_sha256':evidence['pause_sha_at_read'],
            'archived_checkpoint_path':str(evidence['archived']),'archived_checkpoint_sha256':evidence['pause']['checkpoint_sha256'],
            'actual_resumed_latest_path':str(evidence['latest']),'actual_resumed_latest_sha256':before_latest_sha,
            'actual_run_config_path':str(evidence['config_path']),'actual_run_config_sha256':evidence['config_sha_at_read'],
            'progress_history_path':str(evidence['history_path']),'progress_history_sha256_at_read':evidence['history_sha_at_read'],
            'continuation_state_path':str(evidence['state_path']),'continuation_state_sha256_at_read':evidence['state_sha_at_read'],
            'continuation_controller_path':str(evidence['controller_path']),'continuation_controller_sha256':sha(evidence['controller_path']),
            'original_controller_sha256':sha(evidence['original_controller']),
            'immutable_benchmark_pause_manifest_sha256':sha(evidence['immutable_path']),
            'verifier_sha256':sha(Path(__file__))},
        'scope':'Exact CPU equality of the initial restored checkpoint plus real logged progress. No GPU/model inference, process signals, TEST predictions, or future bitwise-training claim.'}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--loop-id',default=DEFAULT_LOOP)
    result.add_argument('--pause-manifest',type=Path)
    result.add_argument('--output',type=Path)
    result.add_argument('--metadata-only',action='store_true',help='Check small JSON/manifest files without loading checkpoint tensors.')
    return result


def main(args):
    # Reject unsafe IDs before deriving the default output destination.
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.loop_id) is not None,'Unsafe loop ID.')
    default_name = 'exact_resume_metadata_verification.json' if args.metadata_only else 'exact_resume_verification.json'
    output = (args.output or OUTPUT/'runs'/args.loop_id/default_name).resolve()
    require(output.is_relative_to(OUTPUT.resolve()), 'Audit output must remain under user-facing outputs.')
    try:
        report = audit(args)
        report['audited_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output,report)
        print(json.dumps({'passed':report['passed'],'status':report['status'],'output':str(output),
                          'paused_step':report['paused_step'],'progress_step':report['progress_step']},ensure_ascii=False))
        return 0
    except Exception as error:
        report = {'passed':False,'status':'failed','cpu_only':True,'loop_id':args.loop_id,
            'audited_at_utc':datetime.now(timezone.utc).isoformat(),'error_type':type(error).__name__,
            'error':str(error),'verifier_sha256':sha(Path(__file__))}
        write_json(output,report)
        print(json.dumps({'passed':False,'output':str(output),'error':str(error)},ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main(parser().parse_args()))
