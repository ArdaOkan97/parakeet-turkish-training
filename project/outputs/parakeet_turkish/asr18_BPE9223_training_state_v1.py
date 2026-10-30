"""Inactive exact original restore with caller-owned bounded journal resume.

Tensor/RNG restore is the frozen streaming-state recipe. Only journal creation
is delegated to an exact source-bound factory, avoiding repeated whole-prefix
copies at genuine coherent native process exits.
"""
from copy import deepcopy
import gc,inspect,random
from pathlib import Path
from bpe_streaming_checkpoint_v1 import MappedCheckpoint,require
from bpe_broad_production_v3 import assert_model
from bpe_broad_audio_v1 import record
from bpe_streaming_training_state_v1 import checkpoint


def restore(binding,*,torch,model,optimizer,identity,augmentation,journal_factory,
            journal_factory_source,diagnostic_only,guard):
    from asr18_BPE9223_model_migration_v2 import bounded_equal
    def bits_equal(torch,a,b):return bounded_equal(torch,a,b,guard=guard)
    import numpy as np
    require(type(diagnostic_only) is bool and diagnostic_only == identity['caller_identity']['full_run_lifecycle_identity']['owner_validation_only'] and callable(journal_factory)
            and record(journal_factory_source['path'])==journal_factory_source
            and str(Path(inspect.getsourcefile(journal_factory)).resolve())==journal_factory_source['path'],
            'Exact full-run source-bound journal factory, never diagnostic promotion')
    mapped=MappedCheckpoint(binding,torch=torch,guard=guard)
    try:
        def load(saved):
            require(saved['kind']=='ASR18_broad_caller_state_v1' and saved['identity']==identity
                    and saved['diagnostic_only']==diagnostic_only,
                    'Exact own coherent checkpoint and qualification identity required')
            # The factory must validate real source-native0/dead/current pointer
            # before returning either the unchanged whole journal or bounded copy.
            journal,branch=journal_factory(deepcopy(saved['augmentation']['committed_journal']))
            augmentation.verify_restore(saved['augmentation'],verify_journal_prefix=journal.verify)
            require(journal.binding()['offset']==saved['augmentation']['committed_journal']['offset']
                    and journal.binding()['sha256']==saved['augmentation']['committed_journal']['sha256'],
                    'Exact authoritative journal prefix before any weight mutation')
            model.load_state_dict(saved['weights'],strict=False)
            # Active mutable Adam CPU clocks may never borrow read-only mappings.
            for partition in saved['optimizer']['optimizers']:
                for state in partition['state'].values():
                    require(state['step'].device.type=='cpu','Mapped CPU Adam step expected')
                    state['step']=state['step'].clone()
            optimizer.load_state_dict(saved['optimizer'])
            require(all(bits_equal(torch,parameter,saved['weights'][name])
                        for name,parameter in optimizer.parameters.items()),'All207 restored weights byte exact')
            for actual,old in zip(optimizer.optimizers,saved['optimizer']['optimizers']):
                for index,parameter in enumerate(actual.param_groups[0]['params']):
                    guard()
                    require(all(bits_equal(torch,actual.state[parameter][key],old['state'][index][key])
                                for key in ('step','exp_avg','exp_avg_sq')),
                            'All3 Adam moments and clocks byte exact')
                    require(actual.state[parameter]['exp_avg'].device.type=='cuda'
                            and actual.state[parameter]['exp_avg_sq'].device.type=='cuda',
                            'Active moments must be copied onto GPU')
            random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            numpy_state=saved['numpy_rng']
            np.random.set_state((numpy_state[0],np.array(numpy_state[1],dtype=np.uint32),*numpy_state[2:]))
            require(random.getstate()==saved['python_rng'] and torch.equal(torch.get_rng_state(),saved['torch_rng'])
                    and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng'])),
                    'Python/Torch/CUDA RNG byte exact')
            actual_numpy=np.random.get_state()
            require(actual_numpy[0]==numpy_state[0] and actual_numpy[1].tolist()==numpy_state[1]
                    and tuple(actual_numpy[2:])==tuple(numpy_state[2:]),'NumPy RNG byte exact')
            return deepcopy(saved['sampler']),journal,branch
        result=load(mapped.value)
    finally:
        mapped.close()
    gc.collect();assert_model(model,optimizer,torch)
    return result
