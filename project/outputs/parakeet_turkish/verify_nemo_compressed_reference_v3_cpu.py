"""Small full-vocabulary vs OTHER-contracted official-reference CPU fixtures."""
import json
from pathlib import Path
import torch
import verify_nemo_native_loss_v3 as worker


def main():
    torch.set_num_threads(2)
    torch.manual_seed(126771)
    require=worker.require
    require(not torch.cuda.is_initialized(),'CPU fixtures initialized CUDA')
    output=Path(__file__).with_name('nemo_compressed_reference_v3_cpu_verification.json')
    require(not output.exists(),'NEW fixture report required')
    tdt,_=worker.reference_class('TDTLossPytorch')
    rnnt,_=worker.reference_class('RNNTLossPytorch')
    cases=[]
    for batch,frames,states,vocab,blank in ((1,4,3,17,9),(2,5,4,8199,8192),(1,3,3,3,2)):
        labels=torch.randint(0,vocab,(batch,states-1))
        labels[labels==blank]=(blank+1)%vocab
        if vocab==8199:labels[:,0]=8198
        if vocab==3:labels[0]=torch.tensor([0,1])
        lengths=torch.tensor([frames-index for index in range(batch)])
        targets=torch.tensor([states-1-index for index in range(batch)])
        original=torch.randn(batch,frames,states,vocab+5)
        upstream=torch.arange(1,batch+1).float()/3
        for sigma,omega in ((0.,0.),(.02,0.),(.02,1.)):
            left=original.clone().requires_grad_();right=original.clone().requires_grad_()
            compressed,mapped,cblank,cwidth,meta=worker.compress_reference_logits(right,labels,blank,vocab)
            if omega==1:
                expected=rnnt(blank=blank,reduction=None)(left[...,:vocab],labels,lengths,targets)
                actual=rnnt(blank=cblank,reduction=None)(compressed[...,:cwidth],mapped,lengths,targets)
            else:
                expected=tdt(blank=blank,durations=[0,1,2,3,4],reduction=None,sigma=sigma)(left,labels,lengths,targets)
                actual=tdt(blank=cblank,durations=[0,1,2,3,4],reduction=None,sigma=sigma)(compressed,mapped,lengths,targets)
            (expected*upstream).sum().backward();(actual*upstream).sum().backward()
            delta=float((expected-actual).abs().max());stats=worker.grad_stats(left.grad,right.grad)
            require(delta<1e-5 and stats['finite'] and stats['relative_l2']<1e-5 and stats['cosine']>.999999999,
                    'Full vs contracted official loss/gradient changed')
            if omega==1:require(float(right.grad[...,vocab:].abs().max())==0,'RNNT duration gradient')
            require(meta['other_original_token_class_count']==vocab-len(meta['kept_original_token_ids']),'Compression class count')
            require(meta['kept_original_token_ids'][cblank]==blank,'Original blank map')
            cases.append({'batch':batch,'T':frames,'U':states-1,'vocab':vocab,'blank':blank,'sigma':sigma,'omega':omega,
                'maximum_raw_loss_delta':delta,'gradient':stats,'compression':meta})
    rejected=[]
    for bad in (torch.tensor([[17]]),torch.tensor([[-1]])):
        try:worker.compress_reference_logits(torch.zeros(1,2,2,22),bad,9,17)
        except RuntimeError:rejected.append(bad.tolist())
        else:raise RuntimeError('Out-of-vocab compression accepted')
    require(not torch.cuda.is_initialized(),'Fixture initialized CUDA')
    report={'schema':1,'passed':True,'checks':sum(5+(case['omega']==1) for case in cases)+len(rejected)+2,
        'cases':cases,'rejected_corruptions':rejected,'worker_sha256':worker.sha(worker.__file__),
        'fixture_sha256':worker.sha(__file__),'cuda_initialized':False,
        'scope':'Synthetic CPU official-reference full-width vs exactOTHER contraction loss and full-logit derivative. Includes actual8199 token-width/paddedB2/nonlastblank/newIDs and all-classes-kept edge; no native CUDA or actual-audio claim.'}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'checks':report['checks'],'max_loss_delta':max(x['maximum_raw_loss_delta'] for x in cases),
        'max_grad_relative_l2':max(x['gradient']['relative_l2'] for x in cases),'worker_sha256':report['worker_sha256']}))


if __name__=='__main__':main()
