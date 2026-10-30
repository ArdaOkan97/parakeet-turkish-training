"""Actual trained-joint FP32 numerical-policy oracle, no ASR/Adam load."""
import gc,json,os
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from tdt_tiled_full_native_v5 import bounded_tdt_loss
from tdt_joint_canonical_2d_fp32_v2 import apply,POLICY
from asr18_target_class_TDT_validation_v1 import metric
AREA=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def validate(folder,*,stage,admit,stop_check):
    from native_tdt_checkpointed_v4 import chunked_tdt_loss
    require(not torch.backends.cuda.matmul.allow_tf32,'Declared FP32 head has no TF32')
    admit('actual_FP32_joint_oracle',1_000_000_000);observations=[]
    class Joint(torch.nn.Module):
        def __init__(self,hidden):super().__init__();self.head=torch.nn.Linear(hidden,8716)
        def forward(self,e,d):return self.head(torch.relu(e+d))
    for case_index,(index,expanded_U) in enumerate(((0,None),(1,None),(1,484),(1,574))):
        stop_check();path=AREA/f'real_target_TDT_gradients_02/actual_joint_fixture_{index}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        joint=Joint(saved['ep'].shape[-1]).to('cuda');joint.load_state_dict(saved['joint']);parameters=tuple(joint.parameters())
        e0=saved['ep'].to('cuda');d0=saved['dp'].to('cuda');labels=saved['labels'].to('cuda')
        if expanded_U is not None:
            e0=e0[:,:8].contiguous();d0=d0.repeat(1,(expanded_U+1+d0.shape[1]-1)//d0.shape[1],1)[:,:expanded_U+1].contiguous()
            labels=labels.repeat(1,(expanded_U+labels.shape[1]-1)//labels.shape[1])[:,:expanded_U].contiguous()
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
        passed=float(tiled)==reference_loss and all(v['byte_equal'] for v in metrics.values())
        item={'fixture':record(path),'T':e0.shape[1],'U':labels.shape[1],'synthetic_shape_expansion_of_bound_actual_values':expanded_U is not None,'old_BF16_raw_loss':old_value,'FP32_dense_loss':reference_loss,
            'FP32_tiled_loss':float(tiled),'gradients':metrics,'passed':passed,'policy':policy}
        observations.append(item);write_json(folder/'actual_joint_precision_progress.json',{'cases':observations});stage('actual_FP32_joint_case',case=case_index+1)
        del saved,joint,parameters,e0,d0,e,d,dense,tiled,reference,actual;torch.cuda.synchronize();torch.cuda.empty_cache()
        from resource_watchdog_v2 import sample_memory
        before_gc={'CUDA_allocated':torch.cuda.memory_allocated(),'CUDA_reserved':torch.cuda.memory_reserved(),'private':sample_memory(os.getpid(),'joint_after_graphdelete_beforeGC')['worker_private_memory']['private_bytes']}
        collected=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        after_gc={'CUDA_allocated':torch.cuda.memory_allocated(),'CUDA_reserved':torch.cuda.memory_reserved(),'private':sample_memory(os.getpid(),'joint_after_forcedGC')['worker_private_memory']['private_bytes']}
        item['post_graphdelete_GC_control']={'before':before_gc,'collected_objects':collected,'after':after_gc,'protected_ASR_Adam_state_loaded':False}
    report={'passed':all(v['passed'] for v in observations),'kind':'ASR18_canonical_T1_U128_actual_and_expanded_joint_oracle_v1','observations':observations,
        'optimizer_updates':0,'full_ASR_model_loaded':False,'policy':POLICY,'original_BF16_or_v1_trajectory_equivalent':False,
        'all207_gradient_gate_still_required':True,'candidate_production_admitted':False}
    write_json(folder/'actual_joint_precision.json',report);return report
