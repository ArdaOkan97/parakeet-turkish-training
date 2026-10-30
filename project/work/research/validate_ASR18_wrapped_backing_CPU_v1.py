"""Regression: resolve the actual hooked Linear after existing LoRA wrapping."""
import argparse,sys,json
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
def main():
    import torch
    from encoder_lora import LoRALinear
    from bpe_broad_production_v1 import frozen_parameter
    assert torch.version.cuda is None;torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    model=torch.nn.Module();model.encoder=torch.nn.Module();base=torch.nn.Linear(4,4,dtype=torch.bfloat16).requires_grad_(False)
    model.encoder.proj=base;backing=base.weight
    assert frozen_parameter(model,'encoder.proj.weight') is backing
    model.encoder.proj=LoRALinear(base,rank=2,alpha=4)
    assert frozen_parameter(model,'encoder.proj.weight') is backing and frozen_parameter(model,'encoder.proj.weight').data_ptr()==backing.data_ptr()
    assert not frozen_parameter(model,'encoder.proj.weight').requires_grad
    # A changed underlying parameter is detected by the caller's pointer check.
    model.encoder.proj.base_layer.weight=torch.nn.Parameter(backing.clone(),requires_grad=False)
    assert frozen_parameter(model,'encoder.proj.weight').data_ptr()!=backing.data_ptr()
    write_json(folder/'report.json',{'passed':True,'actual_existing_LoRALinear_used':True,'unwrapped_and_wrapped_original_identity':True,'changed_backing_detected':True,'GPU_used':False})
    print(json.dumps({'passed':True}),flush=True)
if __name__=='__main__':main()
