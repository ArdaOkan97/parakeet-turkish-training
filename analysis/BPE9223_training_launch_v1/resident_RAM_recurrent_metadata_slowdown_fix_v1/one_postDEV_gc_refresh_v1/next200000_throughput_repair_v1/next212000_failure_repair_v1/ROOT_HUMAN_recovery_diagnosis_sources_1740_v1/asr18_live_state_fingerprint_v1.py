"""Bounded pre-save/live-restore comparison; never a whole CPU state copy.

No numerical package is imported at module scope. Hashes cover the actual
named207 live parameters, three Adam partitions, scalar policies and all four
RNG domains. A checkpoint comparing only to its own saved tensors is a separate
proof; this fingerprint additionally binds the state that existed BEFORE save.
"""
import hashlib, random
from bpe_broad_production_state_v1 import digest, require

CHUNK_BYTES = 4 * 1024**2


def tensor_digest(value, *, torch, guard):
    require(torch.is_tensor(value) and value.layout == torch.strided and value.is_contiguous(),
            'Contiguous actual tensor required; no unbounded contiguous copy')
    flat = value.detach().reshape(-1)
    per_chunk = max(1, CHUNK_BYTES // value.element_size())
    hashed = hashlib.sha256()
    with torch.no_grad():
        for offset in range(0, flat.numel(), per_chunk):
            guard()
            part = flat.narrow(0, offset, min(per_chunk, flat.numel()-offset)).to(device='cpu')
            raw = part.view(torch.uint8).numpy()
            hashed.update(memoryview(raw).cast('B'))
            del raw, part
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes': value.numel()*value.element_size(), 'sha256': hashed.hexdigest()}


def live_state_fingerprint(optimizer, *, torch, guard):
    import numpy as np
    guard()
    require(not optimizer.poisoned and type(optimizer.updates) is int and optimizer.updates >= 0
            and len(optimizer.parameters) == 207 and len(optimizer.optimizers) == 3,
            'Actual unpoisoned207/three-Adam state required')
    if torch.cuda.is_initialized(): torch.cuda.synchronize()
    require(len({id(parameter) for parameter in optimizer.parameters.values()}) == 207,
            'Distinct active named parameters required')
    weights = {name: tensor_digest(parameter, torch=torch, guard=guard)
               for name, parameter in sorted(optimizer.parameters.items())}
    # Mature/new table rows are distinct storage-sharing Adam Parameter views,
    # so their object IDs intentionally differ from the207 named parameters.
    # Bind every indexed state through the optimizer's validated named policy.
    scalar = optimizer.state_dict(); partitions = []; count = 0
    for partition in scalar['optimizers']:
        indices = [index for group in partition['param_groups'] for index in group['params']]
        require(len(indices) == len(set(indices)) and set(indices) == set(partition['state']),
                'Each actual Adam partition index must occur exactly once')
        states = {}
        for index in indices:
            state = partition['state'][index]
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                    and state['step'].device.type == 'cpu', 'All3 actual Adam tensors and cloned CPU step required')
            states[index] = {key: tensor_digest(state[key], torch=torch, guard=guard)
                             for key in ('step', 'exp_avg', 'exp_avg_sq')}
            count += 1
        partitions.append({'param_groups': partition['param_groups'], 'state': states})
    require(count >= 207, 'Complete three-Adam named/sliced partition coverage required')
    numpy = np.random.get_state()
    rng = {'python': random.getstate(), 'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
           'torch': tensor_digest(torch.get_rng_state(), torch=torch, guard=guard),
           'CUDA': [tensor_digest(value, torch=torch, guard=guard)
                    for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}
    value = {'kind': 'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1',
             'updates': optimizer.updates, 'identity': scalar['identity'], 'clocks': scalar['clocks'],
             'weights': weights, 'Adam_partitions': partitions, 'RNG': rng,
             'maximum_transfer_slice_bytes': CHUNK_BYTES}
    del scalar
    guard()
    return {'state_sha256': digest(value), 'state': value}


def require_exact_live_restore(before, after):
    require(before['state_sha256'] == digest(before['state'])
            and after['state_sha256'] == digest(after['state'])
            and before['state_sha256'] == after['state_sha256'],
            'Restored live207/threeAdam/all4RNG differs from actual PRE-save state')
    return True
