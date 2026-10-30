"""Serial optimizer-free evaluation of an immutable native diagnostic close."""
import gc
import json
import random
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from asr18_diagnostic_bootstrap_v1 import record,require

def evaluate_checkpoint(model,processor,checkpoint,folder,*,stage,admit):
    from bpe_broad_full207_optimizer_v1 import bits_equal
    from train_lora_batched_tsc_bpe_full_decoder_v1 import evaluate,gate
    import benchmark
    checkpoint=Path(checkpoint)
    training=json.loads((checkpoint.parent/'native_training_validation.json').read_text())
    require(training['passed'] and training['completed_updates']==20 and training['native_checkpoint_restore_byte_exact']
            and training['checkpoint']==record(checkpoint),'Actual closed20-update native training proof required')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    require(saved['diagnostic_updates']==20 and saved['optimizer']['clocks']==[64270,2020,62270],'Actual diagnostic clocks differ')
    model.load_state_dict(saved['weights'],strict=False)
    actual=dict(model.named_parameters())
    require(len(saved['weights'])==207 and all(bits_equal(torch,actual[n],tensor) for n,tensor in saved['weights'].items()),
            'All207 final diagnostic weights must restore byte exactly')
    random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del saved;gc.collect()
    model.requires_grad_(False);model.eval();torch.cuda.empty_cache()
    stage('optimizer_free_final_DEV_ready')
    admit('optimizer_free_full350_DEV',1_600_000_000)
    previous=benchmark.inputs_for
    def checked_inputs(processor,row):
        require(not (folder/'stop.json').exists(),'Parent stop before next final DEV clip')
        return previous(processor,row)
    benchmark.inputs_for=checked_inputs
    try:scores=evaluate(model,processor,'ASR18_memory_'+folder.name+'_after20')
    finally:benchmark.inputs_for=previous
    value=gate(scores);stage('optimizer_free_full350_DEV_complete',DEV=value)
    report={'passed':True,'full350_DEV_complete':True,'diagnostic_updates':20,'training_validation':record(checkpoint.parent/'native_training_validation.json'),
        'checkpoint':record(checkpoint),'all207_weight_restore_byte_exact':True,'optimizer_histories_preserved_in_checkpoint_not_loaded_for_DEV':True,
        'clocks_preserved':[64270,2020,62270],'DEV':value,'quality_gain_or_qualified_winner_claimed':False,
        'large_geometry_fit_proved':False,'final_DEV_optimizer_free_serial_child':True}
    write_json(folder/'final_DEV_validation.json',report)
    return {'passed':True,'positive_native_update':False,'diagnostic_training_completed_updates':20,
            'final_DEV_complete':True,'bounded_memory_validation_complete':True,'full350_DEV_complete':True,
            'quality_gain_claimed':False,'large_geometry_fit_proved':False}
