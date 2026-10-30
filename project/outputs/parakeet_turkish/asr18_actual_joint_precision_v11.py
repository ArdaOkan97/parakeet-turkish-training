"""Captured real U574/U484 joint oracle and retained-workspace lifetime control."""
import gc,json,os
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from tdt_tiled_full_native_v6 import bounded_tdt_loss
from tdt_tiled_full_native_v5 import bounded_tdt_loss as old_loss
from tdt_joint_canonical_2d_fp32_v2 import apply,POLICY
from asr18_target_class_TDT_validation_v1 import metric
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def validate(folder,*,stage,admit,stop_check):
    from native_tdt_checkpointed_v4 import chunked_tdt_loss
    require(not torch.backends.cuda.matmul.allow_tf32,'No TF32 policy change')
    admit('actual_instance_denominator_oracle',1_000_000_000);observations=[]
    class Joint(torch.nn.Module):
        def __init__(self,hidden):super().__init__();self.head=torch.nn.Linear(hidden,8716)
    for index in range(2):
        path=A/f'real_target_2D_TDT_gradients_01/actual_joint_fixture_{index}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        joint=Joint(saved['ep'].shape[-1]).to('cuda');joint.load_state_dict(saved['joint']);apply(joint)
        parameters=tuple(joint.parameters());e0=saved['ep'][:,:15].to('cuda');d0=saved['dp'].to('cuda');labels=saved['labels'].to('cuda')
        e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            dense=chunked_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',max_joint_bytes=1_700_000_000)
        loss=float(dense);reference=torch.autograd.grad(dense.sum(),(e,d,*parameters))
        e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tiled=bounded_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',guard=stop_check)
        actual=torch.autograd.grad(tiled.sum(),(e,d,*parameters))
        metrics={name:{**metric(a,b),'byte_equal':torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))}
            for name,a,b in zip(('ep','dp','weight','bias'),reference,actual)}
        item={'fixture':record(path),'T':15,'U':labels.shape[1],'dense_loss':loss,'tiled_loss':float(tiled),'gradients':metrics,
            'passed':float(tiled)==loss and all(v['byte_equal'] for v in metrics.values())}
        del dense,tiled,e,d,actual,reference;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        controls=[]
        for arm,function in (('old_per_call_class',old_loss),('new_instance_bindings',bounded_tdt_loss)):
            stop_check();before=torch.cuda.memory_allocated()
            e=e0.detach().requires_grad_();d=d0.detach().requires_grad_()
            with torch.autocast('cuda',dtype=torch.bfloat16):value=function(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',guard=stop_check)
            grads=torch.autograd.grad(value.sum(),(e,d,*parameters));del grads,value,e,d
            torch.cuda.synchronize();torch.cuda.empty_cache();after_delete=torch.cuda.memory_allocated()
            collected=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();after_gc=torch.cuda.memory_allocated()
            controls.append({'arm':arm,'before_allocated_bytes':before,'after_graph_delete_allocated_bytes':after_delete,
                'after_forced_GC_allocated_bytes':after_gc,'GC_reclaimed_CUDA_bytes':after_delete-after_gc,'collected_objects':collected,
                'same_joint_head_and_ep_dp_inputs_kept_alive':True,'ASR_model_or_Adam_loaded':False})
        item['lifetime_controls']=controls;observations.append(item)
        write_json(folder/'actual_joint_progress.json',{'observations':observations});stage('actual_instance_denominator_case',case=index+1)
        del saved,joint,parameters,e0,d0,labels;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    report={'passed':all(v['passed'] for v in observations),'kind':'ASR18_real_U574_U484_instance_denominator_joint_oracle_v1',
        'observations':observations,'policy':POLICY,'optimizer_updates':0,'full_ASR_model_loaded':False,'full_T_objective_fit_claimed':False,
        'training_phase_memory_improvement_claimed':False,'candidate_production_admitted':False}
    write_json(folder/'actual_joint_precision.json',report);return report
