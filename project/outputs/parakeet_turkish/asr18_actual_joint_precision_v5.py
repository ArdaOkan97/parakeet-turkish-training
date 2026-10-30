"""Actual trained-joint FP32 numerical-policy oracle, no ASR/Adam load."""
import json
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from tdt_tiled_full_native_v2 import bounded_tdt_loss
from tdt_joint_fp32_v1 import apply,POLICY
from asr18_target_class_TDT_validation_v1 import metric
AREA=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def validate(folder,*,stage,admit,stop_check):
    from native_tdt_checkpointed_v4 import chunked_tdt_loss
    require(not torch.backends.cuda.matmul.allow_tf32,'Declared FP32 head has no TF32')
    admit('actual_FP32_joint_oracle',1_000_000_000);observations=[]
    class Joint(torch.nn.Module):
        def __init__(self,hidden):super().__init__();self.head=torch.nn.Linear(hidden,8716)
        def forward(self,e,d):return self.head(torch.relu(e+d))
    for index in (0,1):
        stop_check();path=AREA/f'real_target_TDT_gradients_02/actual_joint_fixture_{index}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        joint=Joint(saved['ep'].shape[-1]).to('cuda');joint.load_state_dict(saved['joint']);parameters=tuple(joint.parameters())
        e0=saved['ep'].to('cuda');d0=saved['dp'].to('cuda');labels=saved['labels'].to('cuda')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            old=chunked_tdt_loss(joint,e0,d0,labels,8711,8192,(0,1,2,3,4),reduction='none',max_joint_bytes=1_700_000_000)
        old_value=float(old);del old;policy=apply(joint)
        e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            dense=chunked_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',max_joint_bytes=1_700_000_000)
        reference_loss=float(dense);reference=torch.autograd.grad(dense.sum(),(e,d,*parameters))
        e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tiled=bounded_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',guard=stop_check)
        actual=torch.autograd.grad(tiled.sum(),(e,d,*parameters))
        metrics={key:{**metric(a,b),'byte_equal':torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))}
            for key,a,b in zip(('ep','dp','joint_weight','joint_bias'),reference,actual)}
        passed=abs(float(tiled)-reference_loss)<1e-5*max(1,abs(reference_loss)) and all(v['relative_L2']<.0001 for v in metrics.values())
        item={'fixture':record(path),'T':saved['T'],'U':saved['U'],'old_BF16_raw_loss':old_value,'FP32_dense_loss':reference_loss,
            'FP32_tiled_loss':float(tiled),'gradients':metrics,'passed':passed,'policy':policy}
        observations.append(item);write_json(folder/'actual_joint_precision_progress.json',{'cases':observations});stage('actual_FP32_joint_case',case=index+1)
        del saved,joint,parameters,e0,d0,e,d,dense,tiled,reference,actual;torch.cuda.empty_cache()
    report={'passed':all(v['passed'] for v in observations),'kind':'actual_trained_FP32_joint_policy_equivalence','observations':observations,
        'optimizer_updates':0,'full_ASR_model_loaded':False,'policy':POLICY,'original_BF16_trajectory_equivalent':False,
        'all207_gradient_gate_still_required':True,'candidate_production_admitted':False}
    write_json(folder/'actual_joint_precision.json',report);return report
