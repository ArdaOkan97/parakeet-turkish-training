"""Finite dense-official versus compact-official and tiled-joint gradient oracles."""
import hashlib,json,random,time
import torch
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from tdt_target_class_compression_v1 import compress_lexical
from tdt_bounded_target_classes_v1 import bounded_tdt_loss

def metric(expected,actual):
    d=expected.detach().float();a=actual.detach().float()
    return {'max_abs':float((d-a).abs().max()),'relative_L2':float(torch.linalg.vector_norm(d-a)/torch.linalg.vector_norm(d).clamp_min(1e-20))}

def validate(folder,*,stage,admit,stop_check):
    from native_tdt_checkpointed_v4 import _native_loss,chunked_tdt_loss
    dense=_native_loss(8192,(0,1,2,3,4),0.);small=_native_loss(2,(0,1,2,3,4),0.)
    declaration={'kind':'bounded_full_vocabulary_target_class_TDT_oracle','trained_model_loaded':False,'optimizer_updates':0,
        'dense_cases':[[6,2],[12,5],[24,13],[64,37]],'vocabulary':8711,'sigma':0.,'omega':0.,'clamp':0.,
        'compact_classes':['next_target','other_mass_unreachable_sink','blank'],'full_T_U_DP':True,
        'native_DP_loss_abs_tolerance':.001,'native_gradient_max_abs_tolerance':.0001,'native_gradient_relative_L2_tolerance':.001,
        'joint_gradient_FP32_relative_L2_tolerance':.001,'joint_gradient_BF16_relative_L2_tolerance':.03,
        'BF16_rounding_groups_not_byte_equality':True,'max_tile_joint_bytes':16_000_000,'unchanged_actual_operand_bound_bytes':1_700_000_000,
        'production_admitted':False,'long_real_model_fit_claimed':False}
    write_json(folder/'declaration.json',declaration);admit('official_dense_compact_DP_oracle',1_000_000_000)
    observations=[]
    for t,u in declaration['dense_cases']:
        stop_check();width=8711
        original=(torch.arange(t*(u+1)*width,device='cuda',dtype=torch.float32).remainder(997)/997).reshape(1,t,u+1,width)
        duration=(torch.arange(t*(u+1)*5,device='cuda',dtype=torch.float32).remainder(19)/19).reshape(1,t,u+1,5)
        labels=torch.tensor([[1,8199,5,200,8195][i%5] for i in range(u)],device='cuda',dtype=torch.long)[None]
        lengths=(torch.tensor([t],device='cuda'),torch.tensor([u],device='cuda'))
        lex=original.detach().requires_grad_();raw=duration.detach().requires_grad_()
        before=random.getstate()
        try:value=dense.loss(lex,torch.log_softmax(raw,-1).contiguous(),labels,*lengths,8192,dense.durations,'none',0.,0.,0.,0.)
        finally:random.setstate(before)
        dg=torch.autograd.grad(value.sum(),(lex,raw));expected_loss=value.detach();expected=[v.detach() for v in dg]
        lex=original.detach().requires_grad_();raw=duration.detach().requires_grad_();by_u=torch.cat((labels[0],labels.new_tensor([0])))
        compact=compress_lexical(lex,by_u,8192,torch_module=torch).contiguous()
        before=random.getstate()
        try:value=small.loss(compact,torch.log_softmax(raw,-1).contiguous(),torch.zeros_like(labels),*lengths,2,small.durations,'none',0.,0.,0.,0.)
        finally:random.setstate(before)
        actual=torch.autograd.grad(value.sum(),(lex,raw));loss_error=float((value-expected_loss).abs().max())
        comparisons=[metric(a,b) for a,b in zip(expected,actual)]
        require(loss_error<.001 and all(v['max_abs']<.0001 and v['relative_L2']<.001 for v in comparisons),'Official compact DP loss/lexical/duration gradient differs beyond declared oracle tolerances')
        observations.append({'T':t,'U':u,'loss_max_abs':loss_error,'lexical_gradient':comparisons[0],'duration_gradient':comparisons[1]})
        del original,duration,lex,raw,value,compact,dg,expected,actual;torch.cuda.empty_cache();stage('official_dense_compact_DP_case',T=t,U=u,comparison=observations[-1])
    write_json(folder/'native_DP_equivalence.json',{'passed':True,'observations':observations,'source':record(__file__),'compression':record(__import__('tdt_target_class_compression_v1').__file__),
        'full_vocabulary_denominator_retained':True,'full_T_U_DP_unchanged':True,'long_real_model_fit_claimed':False})
    class Joint(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(16,8716)
        def forward(self,e,d):return self.head(torch.relu(e+d))
    joint_cases=[]
    for autocast in (False,True):
        stop_check();admit('tiled_joint_'+str(autocast),300_000_000)
        torch.manual_seed(180317);joint=Joint().to('cuda');t,u=64,37
        original_e=(torch.arange(t*16,device='cuda',dtype=torch.float32).remainder(41)/41-.5).reshape(1,t,16)
        original_d=(torch.arange((u+1)*16,device='cuda',dtype=torch.float32).remainder(37)/37-.5).reshape(1,u+1,16)
        labels=torch.tensor([[1,8199,5,200,8195][i%5] for i in range(u)],device='cuda',dtype=torch.long)[None]
        parameters=tuple(joint.parameters());e=original_e.detach().requires_grad_();d=original_d.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast):
            value=chunked_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',max_joint_bytes=1_700_000_000)
        expected_loss=value.detach();expected=torch.autograd.grad(value.sum(),(e,d,*parameters));e=original_e.detach().requires_grad_();d=original_d.detach().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast):
            value=bounded_tdt_loss(joint,e,d,labels,8711,8192,(0,1,2,3,4),reduction='none',guard=stop_check)
        actual=torch.autograd.grad(value.sum(),(e,d,*parameters));loss_error=float((value-expected_loss).abs().max());comparisons=[metric(a,b) for a,b in zip(expected,actual)]
        tolerance=.03 if autocast else .001
        require(loss_error<.01 and all(v['relative_L2']<tolerance for v in comparisons),'Bounded recomputation ep/dp/joint gradients differ beyond predeclared numerical scope')
        item={'BF16_autocast':autocast,'T':t,'U':u,'loss_max_abs':loss_error,'gradient_metrics':dict(zip(('ep','dp','joint_weight','joint_bias'),comparisons)),
              'full_dense_joint_bytes':t*(u+1)*8716*4,'tile_joint_bytes':16_000_000}
        joint_cases.append(item);stage('dense_tiled_joint_case',comparison=item)
        del joint,parameters,original_e,original_d,e,d,value,expected,actual;torch.cuda.empty_cache()
    result={'passed':True,'official_compact_DP_cases':observations,'dense_tiled_joint_cases':joint_cases,'optimizer_updates':0,'production_admitted':False,
        'model_loaded':False,'real_encoder_all207_gradient_test_pending':True,'real_large_row_resource_test_pending':True,'declaration':declaration}
    write_json(folder/'target_class_TDT_validation.json',result);return result
