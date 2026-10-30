"""Bounded CPU state/forecast controls; no full-DATA, ASR or GPU claim."""
import argparse, ast, copy, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from bpe_broad_audio_v1 import require, record
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import digest
from asr18_calibrated_geometry_forecast_v6 import CalibratedGeometryForecast
from asr18_precomputed_worker_forecast_v4 import whole_worker_forecast, phase_projection
from asr18_resident_window_admission_v1 import resident_preparation_admission
from asr18_live_state_fingerprint_v1 import live_state_fingerprint, require_exact_live_restore
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1'); F = A.parent/'FastMSS_integration_v1'


def main(folder):
    require(folder.is_relative_to(A), 'E-only fixture'); folder.mkdir(exist_ok=False); checks = []
    def check(name, condition): require(condition, name); checks.append(name)
    def reject(name, action):
        try: action()
        except (ValueError, KeyError): checks.append(name); return
        raise ValueError('Expected rejection: '+name)
    args = {'summary_binding': record(A/'feature_window_native20_01/actual_sameprocess20_summary_v1.json'),
        'archive_binding': record(F/'weak_fixed14_feature_window20_durable_metadata_after_native20_v1.json'),
        'retirement_binding': record(F/'weak_fixed14_feature_window20_actual_retirement_v1.json'),
        'native_source_binding': record(S/'asr18_precomputed_native_v2.py'),
        'whole_replay_metadata': record(A/'whole_replay_geometry_CPU_actual_01/report.json'),
        'whole_replay_actual0': record(A/'whole_replay_geometry_CPU_actual_01/actual_owned_CLOSED0.json')}
    policy = CalibratedGeometryForecast(**args)
    archive = json.loads(Path(args['archive_binding']['path']).read_bytes()); rows = archive['metadata']['update_00.json']['rows']
    native = policy.budget(rows=rows, current_private_bytes=3_000_000_000, update_in_process=1)
    full = whole_worker_forecast(native)
    check('Full prospective worker5,910,813,440B includes restore and no residency drop', full['entire_worker_private_forecast_bytes'] == 5_910_813_440)
    for index in (2, 20, 21, 40):
        value = policy.budget(rows=rows, current_private_bytes=4_600_000_000, update_in_process=index)
        check('Explicit later native index'+str(index)+' preserves high-water formula', value['actual_update_index_in_this_process'] == index and value['incremental_bytes'] >= 250_000_000)
    reject('Native41 cannot run', lambda: policy.budget(rows=rows, current_private_bytes=1, update_in_process=41))
    reject('Boolean native index cannot run', lambda: policy.budget(rows=rows, current_private_bytes=1, update_in_process=True))
    reject('Initial private above3GB remains stopped', lambda: policy.budget(rows=rows, current_private_bytes=3_000_000_001, update_in_process=1))
    reject('Future350MB restore reservation cannot be consumed by native', lambda: phase_projection(full, phase='native', current_private_bytes=5_560_813_440, increment_bytes=1))
    reject('Write allowance cannot shrink', lambda: phase_projection(full, phase='checkpoint_write', current_private_bytes=1, increment_bytes=63_999_999))
    policy.observe({'error': None, 'joined': True, 'available_sample_count': 3, 'private_peak_bytes': 5_100_000_000})
    saved = policy.state_dict(); other = CalibratedGeometryForecast(**args); other.load_state_dict(saved)
    check('Actual high-water carries into a second process without decreasing', other.high_water == 5_100_000_000)
    sample = {'paired_sampling_within_one_second': True, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'observed_monotonic': time.monotonic(), 'system_memory': {'available': True, 'commit_used_bytes': 25_000_000_000,
        'commit_limit_bytes': 40_000_000_000, 'commit_available_bytes': 15_000_000_000},
        'worker_private_memory': {'available': True, 'pid': os.getpid(), 'private_bytes': 4_700_000_000}}
    admission_args = {'native_pid': os.getpid(), 'entire_GPU_worker_forecast_bytes': full['entire_worker_private_forecast_bytes'],
        'CPU_private_allowance_bytes': 900_000_000, 'CPU_contract_binding': record(__file__), 'coherent_idle_verified': True}
    resident = resident_preparation_admission(sample, **admission_args)
    check('Fresh combined admission explicitly prefers resident native', resident['resident_preparation_admitted'] and not resident['GPU_process_exit_claimed'])
    crowded = copy.deepcopy(sample); crowded['system_memory'].update(commit_used_bytes=35_500_000_000, commit_available_bytes=4_500_000_000)
    fallback = resident_preparation_admission(crowded, **admission_args)
    check('Measured combined admission failure selects actual exit/restore fallback', not fallback['resident_preparation_admitted'] and not fallback['DATA_pause_requested'])
    stale = copy.deepcopy(sample); stale['observed_monotonic'] -= 3
    reject('Stale resident telemetry rejected', lambda: resident_preparation_admission(stale, **admission_args))
    unpaired = copy.deepcopy(sample); unpaired['paired_sampling_within_one_second'] = False
    reject('Unpaired resident telemetry rejected', lambda: resident_preparation_admission(unpaired, **admission_args))
    reject('No arbitrary idle boolean or omitted actual boundary', lambda: resident_preparation_admission(sample, **{**admission_args, 'coherent_idle_verified': 1}))
    # Import NumPy explicitly from the existing project environment, then remove
    # that site before CPU Torch import. No package/env installation or mutation.
    sys.path.insert(0, str(P/'work/training-env/Lib/site-packages'))
    import numpy as np
    sys.path.pop(0)
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_initialized(), 'Actual CPU-only Torch; no CUDA init')
    random.seed(101); np.random.seed(102); torch.manual_seed(103)
    parameters = {f'p{index:03}': torch.nn.Parameter(torch.arange(6, dtype=torch.float32)+index) for index in range(207)}
    values = list(parameters.values())
    class Optimizer:
        updates = 3; poisoned = False
        def __init__(self):
            self.parameters = parameters; self.optimizers = []
            #212 Adam entries include five real storage-sharing table slices.
            for group_values in (values[:15], [torch.nn.Parameter(value.detach()[2:]) for value in values[:5]], values[15:]):
                self.optimizers.append(SimpleNamespace(param_groups=[{'params': group_values, 'lr': .01, 'betas': (.9, .999)}],
                    state={p: {'step': torch.tensor(3., dtype=torch.float32), 'exp_avg': torch.ones_like(p), 'exp_avg_sq': torch.full_like(p, 2.)} for p in group_values}))
        def state_dict(self):
            parts = []
            for adam in self.optimizers:
                group = adam.param_groups[0]; params = group['params']
                parts.append({'param_groups': [{**{k:v for k,v in group.items() if k!='params'}, 'params': list(range(len(params)))}],
                    'state': {index: adam.state[p] for index,p in enumerate(params)}})
            return {'identity': {'source_clocks': [64250,2000,62250], 'fixture': True}, 'clocks': [64253,2003,62253],
                    'updates': self.updates, 'optimizers': parts}
    optimizer = Optimizer(); before = live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)
    after = live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)
    check('207weights+212 indexed moments/slices fingerprint is stable and does not advance RNG', require_exact_live_restore(before, after))
    check('All three partitions and full207 named weights are covered', len(before['state']['weights']) == 207 and sum(len(p['state']) for p in before['state']['Adam_partitions']) == 212)
    with torch.no_grad(): values[0][0] += 1
    reject('Actual weight mutation rejected against PRE-save fingerprint', lambda: require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    with torch.no_grad(): values[0][0] -= 1
    first = optimizer.optimizers[0].state[values[0]]
    first['exp_avg'][0] += 1
    reject('Adam first-moment mutation rejected', lambda: require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    first['exp_avg'][0] -= 1; first['step'] += 1
    reject('CPU Adam clock mutation rejected', lambda: require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    first['step'] -= 1
    old_numpy = np.random.get_state(); np.random.random()
    reject('NumPy RNG mutation rejected', lambda: require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    np.random.set_state(old_numpy)
    old_python = random.getstate(); random.random()
    reject('Python RNG mutation rejected', lambda: require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    random.setstate(old_python)
    old_torch = torch.get_rng_state(); torch.rand(1)
    reject('Torch RNG mutation rejected', lambda: require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    torch.set_rng_state(old_torch)
    from bpe_streaming_checkpoint_v1 import write_checkpoint, MappedCheckpoint
    pin = write_checkpoint(folder/'tiny_named_state.json', {'weights': {k:v.detach() for k,v in parameters.items()},
        'optimizer': optimizer.state_dict()}, torch=torch, guard=lambda:None)
    mapped = MappedCheckpoint(pin, torch=torch, guard=lambda:None)
    try:
        with torch.no_grad():
            for name, parameter in parameters.items(): parameter.copy_(mapped.value['weights'][name])
            for actual, saved_part in zip(optimizer.optimizers, mapped.value['optimizer']['optimizers']):
                for index, p in enumerate(actual.param_groups[0]['params']):
                    for key in ('step', 'exp_avg', 'exp_avg_sq'): actual.state[p][key].copy_(saved_part['state'][index][key])
        del saved_part, p
    finally: mapped.close()
    check('Actual bounded named-tensor write/read matches actual PRE-save CPU state', require_exact_live_restore(before, live_state_fingerprint(optimizer, torch=torch, guard=lambda:None)))
    tampered = copy.deepcopy(before); tampered['state']['identity']['fixture'] = 1; tampered['state_sha256'] = digest(tampered['state'])
    reject('Canonical nested bool/int identity change rejected', lambda: require_exact_live_restore(before, tampered))
    owned_sources = [S/name for name in ('asr18_live_state_fingerprint_v1.py', 'asr18_resident_window_admission_v1.py',
        'asr18_phase_private_monitor_v1.py', 'asr18_precomputed_worker_forecast_v3.py', 'asr18_precomputed_worker_forecast_v4.py',
        'asr18_calibrated_geometry_forecast_v6.py', 'bpe_final_DATA_schedule_context_v1.py', 'asr18_full_DATA_native_window_v1.py')]
    owned_sources.append(P/'work/research/run_ASR18_full_DATA_multiwindow_v1.py')
    for path in owned_sources: compile(ast.parse(path.read_text(encoding='utf-8-sig')), str(path), 'exec')
    check('All9 new caller/helper sources parse without execution or bytecode writes', True)
    write_json(folder/'report.json', {'passed': True, 'checks': checks, 'check_count': len(checks), 'source': record(__file__),
        'bindings': [record(p) for p in owned_sources], 'native_proposal': native, 'whole_worker_proposal': full,
        'resident_scalar_control': resident, 'fallback_scalar_control': fallback,
        'actual_tiny_CPU_checkpoint': pin, 'CPU_Torch_version': torch.__version__, 'NumPy_version': np.__version__,
        'actual_ASR_or_GPU_updates': 0, 'actual_full_DATA_positive_path': False, 'actual_resident_native_lifecycle': False,
        'actual_CPU_frontend_or_original_audio_work': False, 'production_enabled': False})
    print(json.dumps({'passed': True, 'checks': len(checks), 'actual_GPU_updates': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--folder', required=True)
    main(Path(parser.parse_args().folder).resolve())
