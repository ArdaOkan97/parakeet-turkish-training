"""Cheap actual trained-joint oracle. No encoder/model load or Adam update."""
import json,random
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from tdt_target_class_compression_v1 import compress_lexical
from tdt_bounded_target_classes_v1 import bounded_tdt_loss as candidate1
from tdt_bounded_target_classes_v2 import bounded_tdt_loss as candidate2
from tdt_tiled_full_native_v2 import bounded_tdt_loss as candidate3
from asr18_target_class_TDT_validation_v1 import metric
ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'

def validate(folder,*,stage,admit,stop_check):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    from native_tdt_checkpointed_v4 import chunked_tdt_loss,_native_loss
    admit('actual_joint_precision_oracle',1_000_000_000);observations=[]
    class Joint(torch.nn.Module):
        def __init__(self,hidden):super().__init__();self.head=torch.nn.Linear(hidden,8716)
        def forward(self,e,d):return self.head(torch.relu(e+d))
    for index in (0,1):
        stop_check();path=AREA/f'real_target_TDT_gradients_02/actual_joint_fixture_{index}.pt';saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        require(saved['diagnostic_only'] and saved['vocabulary']==8711,'Pinned actual trained joint fixture')
        joint=Joint(saved['ep'].shape[-1]).to('cuda');joint.load_state_dict(saved['joint']);parameters=tuple(joint.parameters())
        e0=saved['ep'].to('cuda');d0=saved['dp'].to('cuda');labels=saved['labels'].to('cuda')
        e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=chunked_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',max_joint_bytes=1_700_000_000)
        expected_loss=float(loss);expected=torch.autograd.grad(loss.sum(),(e,d,*parameters));variants={}
        for name,fn in (('v1_backward_inside_autocast',candidate1),('v2_backward_outside_autocast',candidate2),('v3_exact_full_native_tile',candidate3)):
            e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
            with torch.autocast('cuda',dtype=torch.bfloat16):value=fn(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',guard=stop_check)
            actual=torch.autograd.grad(value.sum(),(e,d,*parameters))
            variants[name]={'loss':float(value),'loss_abs_difference':abs(float(value)-expected_loss),
                'gradients':{key:{**metric(a,b),'byte_equal':torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))}
                    for key,a,b in zip(('ep','dp','joint_weight','joint_bias'),expected,actual)}}
            del actual,value
        if index==0:
            require(all(v['byte_equal'] for v in variants['v3_exact_full_native_tile']['gradients'].values()),'One-tile exact native-order candidate must match actual trained joint gradients BYTE EXACT')
        # Split native-DP compression arithmetic from joint/backward dtype.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):packed=joint(e0[:,:,None],d0[:,None])
        lex0=packed[...,:8711].float();dur0=packed[...,8711:].float();del packed
        lex=lex0.detach().requires_grad_();dur=dur0.detach().requires_grad_();t,u=saved['T'],saved['U']
        lens=(torch.tensor([t],device='cuda'),torch.tensor([u],device='cuda'))
        dense=_native_loss(8192,(0,1,2,3,4),0.);small=_native_loss(2,(0,1,2,3,4),0.)
        before=random.getstate()
        try:value=dense.loss(lex,torch.log_softmax(dur,-1).contiguous(),labels,*lens,8192,dense.durations,'none',0.,0.,0.,0.)
        finally:random.setstate(before)
        raw_expected=torch.autograd.grad(value.sum(),(lex,dur))
        lex=lex0.detach().requires_grad_();dur=dur0.detach().requires_grad_();by_u=torch.cat((labels[0],labels.new_tensor([0])))
        c=compress_lexical(lex,by_u,8192,torch_module=torch).contiguous()
        before=random.getstate()
        try:value=small.loss(c,torch.log_softmax(dur,-1).contiguous(),torch.zeros_like(labels),*lens,2,small.durations,'none',0.,0.,0.,0.)
        finally:random.setstate(before)
        raw_actual=torch.autograd.grad(value.sum(),(lex,dur))
        raw_metrics={key:{**metric(a,b),'BF16_mismatching_elements':int((a.bfloat16()!=b.bfloat16()).sum()),'elements':a.numel()}
            for key,a,b in zip(('lexical','duration'),raw_expected,raw_actual)}
        item={'fixture':record(path),'T':t,'U':u,'actual_projected_dtype':str(e0.dtype),'dense_raw_loss':expected_loss,'variants':variants,
            'actual_lexical_range':[float(lex0.min()),float(lex0.max())],'raw_DP_compression_gradients':raw_metrics}
        observations.append(item);write_json(folder/'actual_joint_precision_progress.json',{'cases':observations});stage('actual_joint_precision_case',case=index+1)
        del saved,joint,parameters,e0,d0,e,d,loss,expected,lex0,dur0,lex,dur,c,value,raw_expected,raw_actual;torch.cuda.empty_cache()
    report={'passed':True,'kind':'actual_trained_joint_precision_diagnosis','observations':observations,'full_ASR_model_loaded':False,'trained_joint_head_loaded':True,
        'optimizer_updates':0,'allow_bf16_reduced_precision_reduction':False,'candidate_production_admitted':False,'all207_gradient_gate_still_required':True,
        'v3_change':'exact original8711reduction and originalgradient math with tileaddressing; fullUeachframe; no recompressionJacobian'}
    write_json(folder/'actual_joint_precision.json',report);return report
