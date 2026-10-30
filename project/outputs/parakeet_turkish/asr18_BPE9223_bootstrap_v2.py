"""Original production sources -> mature8711 -> deterministic appended9223.

No diagnostic checkpoint is accepted. Existing three Adam objects/states move
by ownership to new storage-sharing table views; old objects cannot be reused.
"""
import gc
from bpe_broad_audio_v1 import require


def bootstrap(model,processor,*,torch,stage,admit,guard,migration_plan):
    from bpe_broad_streamed_bootstrap_v2 import bootstrap as mature_bootstrap
    from asr18_BPE9223_model_migration_v2 import expand_model
    from asr18_BPE9223_optimizer_v2 import BPE9223Full207Optimizer
    original,provenance,snapshot=mature_bootstrap(model,processor,torch=torch,stage=stage,admit=admit)
    require(original.updates==0 and original.base_clocks==(64250,2000,62250), 'Actual original production source clocks only')
    migration=expand_model(model,processor,migration_plan,torch=torch,guard=guard,admit=admit)
    admit('BPE9223_new512_zero_moments',2*migration['added_parameter_bytes']+4*1024**2+4096)
    optimizer=BPE9223Full207Optimizer(model,original,migration=migration,torch_module=torch,guard=guard)
    require(original.poisoned and optimizer.optimizers is original.optimizers, 'Exact three source Adam objects transferred once')
    del original;gc.collect();torch.cuda.empty_cache()
    require(sum(p.requires_grad for p in model.parameters())==207 and all(p.device.type=='cuda' and p.dtype==torch.float32
        for p in model.parameters() if p.requires_grad),'All207 active FP32GPU parameters after append migration')
    provenance.update(BPE9223_append_migration=migration,optimizer_identity=optimizer.identity(),
        preserved_existing_Adam_clocks=[64250,2000,62250],appended512_Adam_clock=0,
        exactly_three_original_Adam_objects_preserved=True,diagnostic_checkpoint_ancestry=False)
    stage('after_BPE9223_source_migration_full207_training_ready',migration=migration,
        initial_Adam_state_counts=[17,6,192],new512_clock=0,expected_checkpoint_tensor_entries=854)
    return optimizer,provenance,snapshot
