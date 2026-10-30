"""Coherent named-state checkpoint using bounded CPU tensor slices."""
from copy import deepcopy
import gc, random
from bpe_streaming_checkpoint_v1 import MappedCheckpoint, write_checkpoint, require
from bpe_broad_production_v3 import assert_model
from bpe_broad_production_state_v1 import CommitJournal
from bpe_broad_journal_branch_v1 import branch_checkpoint_prefix

def checkpoint(path, *, torch, optimizer, state, provenance, identity, augmentation, journal, diagnostic_only, guard):
    import numpy as np
    optimizer.zero_grad()
    numpy_state = np.random.get_state()
    value = {
        'kind': 'ASR18_broad_caller_state_v1', 'diagnostic_only': diagnostic_only,
        'identity': identity, 'source_provenance': provenance,
        # Named live GPU tensor views: no simultaneous full CPU weight copy.
        'weights': {name: parameter.detach() for name, parameter in optimizer.parameters.items()},
        'optimizer': optimizer.state_dict(), 'sampler': deepcopy(state),
        'augmentation': augmentation.checkpoint_binding(committed_journal=journal.binding()),
        'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(),
        'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }
    result = write_checkpoint(path, value, torch=torch, guard=guard)
    del value
    gc.collect()
    return result

def restore(binding, *, torch, model, optimizer, identity, augmentation, new_journal_path, diagnostic_only, guard):
    from bpe_broad_full207_optimizer_v1 import bits_equal
    import numpy as np
    mapped = MappedCheckpoint(binding, torch=torch, guard=guard)
    try:
        def load(saved):
            require(saved['kind'] == 'ASR18_broad_caller_state_v1' and saved['identity'] == identity
                    and saved['diagnostic_only'] == diagnostic_only, 'Exact own coherent checkpoint and qualification identity required')
            original = CommitJournal(saved['augmentation']['committed_journal']['path'])
            augmentation.verify_restore(saved['augmentation'], verify_journal_prefix=original.verify)
            journal, branch = branch_checkpoint_prefix(saved['augmentation']['committed_journal'], new_journal_path, guard=guard)
            model.load_state_dict(saved['weights'], strict=False)
            # AdamW mutates its CPU step tensor. A READ_ONLY mapped step must
            # never become active optimizer state, even when .to(cpu) is a no-op.
            for partition in saved['optimizer']['optimizers']:
                for state in partition['state'].values():
                    require(state['step'].device.type == 'cpu', 'Mapped CPU Adam step expected')
                    state['step'] = state['step'].clone()
            optimizer.load_state_dict(saved['optimizer'])
            require(all(bits_equal(torch, parameter, saved['weights'][name])
                        for name, parameter in optimizer.parameters.items()), 'All207 restored weights byte exact')
            for actual, old in zip(optimizer.optimizers, saved['optimizer']['optimizers']):
                for index, parameter in enumerate(actual.param_groups[0]['params']):
                    guard()
                    require(all(bits_equal(torch, actual.state[parameter][key], old['state'][index][key])
                                for key in ('step', 'exp_avg', 'exp_avg_sq')), 'All3 Adam moments and clocks byte exact')
                    require(actual.state[parameter]['exp_avg'].device.type == 'cuda'
                            and actual.state[parameter]['exp_avg_sq'].device.type == 'cuda', 'Active moments must be copied onto GPU')
            random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            numpy_state = saved['numpy_rng']
            np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
            require(random.getstate() == saved['python_rng'] and torch.equal(torch.get_rng_state(), saved['torch_rng'])
                    and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved['cuda_rng'])), 'Python/Torch/CUDA RNG byte exact')
            actual_numpy = np.random.get_state()
            require(actual_numpy[0] == numpy_state[0] and actual_numpy[1].tolist() == numpy_state[1]
                    and tuple(actual_numpy[2:]) == tuple(numpy_state[2:]), 'NumPy RNG byte exact')
            return deepcopy(saved['sampler']), journal, branch
        result = load(mapped.value)
    finally:
        # load() has returned, so only copied active GPU state and tiny cloned
        # CPU clocks survive. No active state may borrow the read-only mapping.
        mapped.close()
    gc.collect(); assert_model(model, optimizer, torch)
    return result
