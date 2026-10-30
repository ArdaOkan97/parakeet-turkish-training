"""CPU source audit plus small207-tensor exact-roundtrip fixture; no GPU launch."""
import copy
import hashlib
import json
import os
import pathlib
import sys
import threading
import time

import psutil
from cpu_preparation_guard_v2 import memory, now, write_json

ROOT = pathlib.Path(r'E:\gpu_training_scratchpad')
AREA = ROOT / 'analysis/BPE_weak_then_clean_v1'
PROJECT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))


def watchdog():
    process = psutil.Process()
    high_commit_samples = 0
    while True:
        m = memory()
        high_commit_samples = high_commit_samples + 1 if m['commit_ratio'] >= .90 else 0
        if (high_commit_samples >= 3 or m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000
                or process.memory_info().private >= 2_000_000_000):
            write_json(AREA / 'optimizer_preflight_CPU_v2_resource_stop.json',
                       {'at_utc': now(), 'pid': os.getpid(), 'memory': m,
                        'own_private_bytes': process.memory_info().private})
            os._exit(75)
        time.sleep(2)


def record(path, expected):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == expected
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert memory()['commit_ratio'] < .90
    threading.Thread(target=watchdog, daemon=True).start()
    import torch
    assert torch.version.cuda is None, 'Isolated CPU-only PyTorch runtime required after measured CUDA-runtime memory stop'
    print(json.dumps({'stage': 'CPU_runtime_loaded', 'torch': torch.__version__,
                      'private_bytes': psutil.Process().memory_info().private}), flush=True)
    from bpe_broad_full207_optimizer_v1 import BPEBroadFull207Optimizer, bits_equal, encoder_lr
    from bpe_full_decoder_optimizer_v1 import TABLES, partition_plan
    original_path = PROJECT / 'work/closed-sources/TSC_WorldSpeech_expanded_65948_closed_v1/best_checkpoint.pt'
    bpe_path = ROOT / 'runs/adaptive_20261015_TR_BPE512_full_decoder_02/latest_checkpoint.pt'
    records = {'original': record(original_path, 'b0f5d5e207caeda3c030a5f11eda4ae9004e4826a7fd050a8ac8de78333e3e38'),
               'bpe': record(bpe_path, '71ca1897d95622c5556df6a3b568d26136004e25c4bcca515f7a242df9e9b63a')}
    print(json.dumps({'stage': 'source_file_hashes_passed'}), flush=True)
    original = torch.load(original_path, map_location='cpu', weights_only=True, mmap=True)
    bpe = torch.load(bpe_path, map_location='cpu', weights_only=True, mmap=True)
    assert original['step'] == original['selected_step'] == 62250 and bpe['step'] == 64250
    names = tuple(original['weights'])
    heads, lora, mature_plan, fresh_plan = partition_plan(names, 8711)
    assert set(names) == set(bpe['weights'])
    assert bpe['optimizer']['mature_Adam_clock'] == 64250
    assert bpe['optimizer']['new_row_Adam_clock'] == 2000
    assert bpe['optimizer']['parked_encoder_Adam_clock'] == 62250
    actual_moments = 0
    for name in names:
        assert bool(torch.isfinite(bpe['weights'][name]).all())
        if name in lora:
            assert bits_equal(torch, original['weights'][name], bpe['weights'][name])
    for saved, clock in [(original['optimizer'], 62250),
                         (bpe['optimizer']['mature_optimizer'], 64250),
                         (bpe['optimizer']['fresh_optimizer'], 2000)]:
        for state in saved['state'].values():
            assert state['step'].item() == clock
            assert all(bool(torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))
            assert bool((state['exp_avg_sq'] >= 0).all())
            actual_moments += 1
    groups = copy.deepcopy(original['optimizer']['param_groups'])
    old_identity = copy.deepcopy(bpe['optimizer']['identity'])
    mature_group = copy.deepcopy(bpe['optimizer']['mature_optimizer']['param_groups'][0])
    fresh_group = copy.deepcopy(bpe['optimizer']['fresh_optimizer']['param_groups'][0])
    del original, bpe
    import gc
    gc.collect()
    torch.manual_seed(181817)
    original_weights = {}
    new_weights = {}
    for name in names:
        old_rows = 8199 + (5 if name in TABLES[1:] else 0)
        new_rows = 8711 + (5 if name in TABLES[1:] else 0)
        if name in TABLES:
            shape = (old_rows,) if name == TABLES[2] else (old_rows, 2)
            new_shape = (new_rows,) if name == TABLES[2] else (new_rows, 2)
        else:
            shape = new_shape = (2, 2)
        original_weights[name] = torch.randn(shape)
        new_weights[name] = original_weights[name].clone() if name in lora else torch.randn(new_shape)

    def moment(parameter, clock):
        return {'step': torch.tensor(float(clock)), 'exp_avg': torch.full_like(parameter, .002),
                'exp_avg_sq': torch.full_like(parameter, .004)}

    old = {'step': 62250, 'selected_step': 62250, 'weights': original_weights,
           'optimizer': {'param_groups': groups, 'state':
                         {i: moment(original_weights[n], 62250) for i, n in enumerate((*heads, *lora))}}}

    def fake_partition(plan, group, clock):
        return {'param_groups': [{**copy.deepcopy(group), 'params': list(range(len(plan)))}],
                'state': {i: moment(new_weights[item['name']] if item['target_rows'] is None else
                                   new_weights[item['name']][slice(*item['target_rows'])], clock)
                          for i, item in enumerate(plan)}}

    current = {'step': 64250, 'weights': new_weights, 'optimizer':
               {'updates': 2000, 'mature_Adam_clock': 64250, 'new_row_Adam_clock': 2000,
                'parked_encoder_Adam_clock': 62250, 'identity': old_identity,
                'mature_optimizer': fake_partition(mature_plan, mature_group, 64250),
                'fresh_optimizer': fake_partition(fresh_plan, fresh_group, 2000)}}

    class Model:
        def __init__(self):
            self.parameters = {n: torch.nn.Parameter(p.clone()) for n, p in new_weights.items()}

        def named_parameters(self):
            return self.parameters.items()

    model = Model()
    optimizer = BPEBroadFull207Optimizer(model, old, current, source_records=records, torch_module=torch)
    before = {n: p.detach().clone() for n, p in model.parameters.items()}
    for p in model.parameters.values():
        p.grad = torch.full_like(p, .01)
    optimizer.step()
    assert all(not bits_equal(torch, before[n], p) for n, p in model.parameters.items())
    assert optimizer.state_dict()['clocks'] == [64251, 2001, 62251]
    path = ROOT / 'scratch/BPE_broad_optimizer_roundtrip_v1.pt'
    torch.save({'weights': {n: p.detach().clone() for n, p in model.parameters.items()},
                'optimizer': optimizer.state_dict()}, path)
    restored = torch.load(path, weights_only=True)
    second = Model()
    second_optimizer = BPEBroadFull207Optimizer(second, old, current, source_records=records, torch_module=torch)
    with torch.no_grad():
        for n, p in second.parameters.items():
            p.copy_(restored['weights'][n])
    second_optimizer.load_state_dict(restored['optimizer'])
    corrupted = {**restored['optimizer'], 'clocks': [64251, 2000, 62251]}
    try:
        second_optimizer.load_state_dict(corrupted)
        raise AssertionError('Wrong new-row clock accepted')
    except ValueError:
        pass
    for m, opt in ((model, optimizer), (second, second_optimizer)):
        opt.zero_grad()
        for p in m.parameters.values():
            p.grad = torch.full_like(p, .015)
        opt.step()
    assert all(bits_equal(torch, model.parameters[n], second.parameters[n]) for n in names)
    a, b = optimizer.state_dict(), second_optimizer.state_dict()
    assert a['clocks'] == b['clocks'] == [64252, 2002, 62252]
    for left, right in zip(a['optimizers'], b['optimizers']):
        for i in left['state']:
            assert all(bits_equal(torch, left['state'][i][k], right['state'][i][k])
                       for k in ('step', 'exp_avg', 'exp_avg_sq'))
    assert encoder_lr(65948) == encoder_lr(500000) == 3e-5
    report = {'completed_at_utc': now(), 'passed': True, 'GPU_used': False, 'native_training_run': False,
              'pinned_sources': records, 'actual_named_weights_audited': 207,
              'actual_parked_encoder_weights_equal': 192, 'actual_Adam_partitions_audited': actual_moments,
              'source_clocks': [64250, 2000, 62250], 'fixture_changed_model_tensors': 207,
              'fixture_roundtrip_next_update_byte_exact': True, 'wrong_clock_rejected': True,
              'fixture_after_two_updates_clocks': [64252, 2002, 62252],
              'quality_gain_claimed': False, 'training_started': False,
              'runtime_torch': torch.__version__, 'runtime_CUDA': torch.version.cuda,
              'supported_workload_change': 'isolated_CPU_only_Torch_avoids_CUDA_runtime_private_commit',
              'own_private_bytes': psutil.Process().memory_info().private, 'memory': memory()}
    write_json(AREA / 'optimizer_CPU_preflight_v1.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
