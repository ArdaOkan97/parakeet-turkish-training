"""No GPU: full-vocabulary probabilities and arbitrary-edge gradients oracle."""
import argparse,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from tdt_target_class_compression_v1 import compress_lexical
from atomic_io_retry_v1 import write_json

def main():
    import torch
    assert torch.version.cuda is None;torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    observations=[]
    for dtype in (torch.float64,torch.float32):
        for seed,vocab,blank in ((17,7,6),(21,11,3),(24,8711,8192)):
            generator=torch.Generator().manual_seed(seed)
            logits=torch.randn((1,5,8,vocab),dtype=dtype,generator=generator,requires_grad=True)
            ids=torch.randint(vocab-1,(8,),generator=generator);ids=ids+(ids>=blank).long();ids[1]=ids[0]
            edge=torch.rand((1,5,8,2),dtype=dtype,generator=generator)
            full=torch.log_softmax(logits,dim=-1);selected=full.gather(-1,ids[None,None,:,None].expand(1,5,8,1)).squeeze(-1)
            dense=(selected*edge[...,0]+full[...,blank]*edge[...,1]).sum()
            expected=torch.autograd.grad(dense,logits)[0]
            compact=compress_lexical(logits,ids,blank,torch_module=torch);small=torch.log_softmax(compact,dim=-1)
            result=(small[...,0]*edge[...,0]+small[...,2]*edge[...,1]).sum();actual=torch.autograd.grad(result,logits)[0]
            probabilities_error=max(float((small[...,0]-selected).abs().max()),float((small[...,2]-full[...,blank]).abs().max()))
            gradient_error=float((expected-actual).abs().max());relative=float(torch.linalg.vector_norm(expected-actual)/torch.linalg.vector_norm(expected))
            tolerance=1e-11 if dtype==torch.float64 else 5e-6
            assert probabilities_error<tolerance and gradient_error<tolerance and relative<tolerance
            observations.append({'dtype':str(dtype),'vocabulary':vocab,'blank':blank,'probability_log_max_abs':probabilities_error,'gradient_max_abs':gradient_error,'gradient_relative_L2':relative})
    write_json(folder/'report.json',{'passed':True,'observations':observations,'GPU_used':False,'official_DP_tested':False,
        'model_or_native_gradient_tested':False,'production_admitted':False,'scope':'Arbitrary nonnegative blank/target edge weights prove per-cell denominator and chain-rule sufficient statistic; native DP integration pending'})
    print(json.dumps({'passed':True,'cases':len(observations)}),flush=True)

if __name__=='__main__':main()
