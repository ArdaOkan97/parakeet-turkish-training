"""Actual checkpointed-encoder output and all192 LoRA gradient comparison."""
import gc
import json
import torch
from asr18_diagnostic_bootstrap_v1 import AREA,record,require
from atomic_io_retry_v1 import write_json


def validate(model,processor,folder,*,candidate,stage,admit):
    from runtime import inputs_for
    from asr18_cached_teacher_control_v1 import bits
    def equal(left,right):
        return left.shape==right.shape and left.dtype==right.dtype and bits(left.reshape(-1),right.reshape(-1))
    manifest=AREA/'control_manifest_v3/manifest.json'
    rows=json.loads(manifest.read_text())['rows'][:3]
    names={name:parameter for name,parameter in model.named_parameters() if name.endswith(('lora_A','lora_B'))}
    require(len(names)==192 and all(p.device.type=='cuda' and p.dtype==torch.float32 for p in names.values()),'All192 active GPU FP32 LoRA required')
    model.encoder.eval();results=[];comparisons=[]
    prior=None
    if candidate:prior=torch.load(AREA/'streamed_encoder_reference_01/encoder_reference.pt',map_location='cpu',weights_only=True,mmap=True)
    admit('first_checkpointed_encoder_forward_backward',1_300_000_000)
    for index,row in enumerate(rows):
        for parameter in model.parameters():parameter.grad=None
        inputs=inputs_for(processor,row)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model.encoder(**inputs)
            loss=encoded.last_hidden_state.float().square().mean()
        loss.backward();torch.cuda.synchronize()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in names.values()),'Finite gradients on all192 required')
        actual={'source_id':row['source_id'],'features':encoded.last_hidden_state.detach().cpu(),
                'mask':encoded.attention_mask.detach().cpu(),'loss':loss.detach().cpu(),
                'LoRA_gradients':{name:parameter.grad.cpu() for name,parameter in names.items()}}
        if candidate:
            reference=prior[index]
            exact=actual['source_id']==reference['source_id'] and all(equal(actual[key],reference[key]) for key in ('features','mask','loss'))
            exact=exact and all(equal(gradient,reference['LoRA_gradients'][name]) for name,gradient in actual['LoRA_gradients'].items())
            require(exact,'Streamed frozen matrices changed checkpointed encoder output or LoRA gradient')
            comparisons.append({'source_id':row['source_id'],'features_mask_loss_all192_gradients_byte_exact':True})
        else:results.append(actual)
        del inputs,encoded,loss,actual
        for parameter in model.parameters():parameter.grad=None
        gc.collect();torch.cuda.empty_cache();stage('checkpointed_encoder_equivalence_case_complete',clip=index+1)
    if candidate:
        write_json(folder/'streaming_equivalence.json',{'passed':True,'comparisons':comparisons,'checkpoint_recomputation_active':True,
            'reference':record(AREA/'streamed_encoder_reference_01/encoder_reference.pt'),'loader':record(__import__('pathlib').Path(__file__).with_name('stream_frozen_encoder_matrices_v1.py')),
            'metrics':model._diagnostic_frozen_metrics,'model_or_optimizer_updates':0})
    else:torch.save(results,folder/'encoder_reference.pt')
    return {'passed':True,'positive_native_update':False,'encoder_checkpointed_gradient_equivalence':candidate,'cases':3}
