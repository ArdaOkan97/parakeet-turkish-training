"""Actual 96 adapter tensors and pinned frozen projections; CPU-only."""
from pathlib import Path
import copy
import json
import sys
import torch
from torch.nn import functional as F
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import upgrade_lora_rank_v1 as u
torch.set_num_threads(1)
torch.backends.mkldnn.enabled=True
source=ROOT/'work/adaptive_20261010_continuation_01_resume_B8_best.pt'
base=ROOT/'work/hf-cache/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/model.safetensors'
before=u.sha(source)
saved=torch.load(source,map_location='cpu',weights_only=True)
up=u.upgrade(saved)
checks=[];rejections=[]
def check(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def reject(call,label):
    try:call()
    except (ValueError,KeyError):rejections.append(label)
    else:raise AssertionError(label)
def error(a,b):
    a,b=a.double(),b.double();delta=b-a
    return {'relative_l2':float(delta.norm()/a.norm().clamp_min(1e-300)),
            'max_absolute_delta':float(delta.abs().max()),'bitwise_equal':bool(torch.equal(a,b))}
check(saved['step']==1773 and saved['selected_step']==1773,'Frozen qualifying parent1773')
check(len(up['weights'])==len(saved['weights'])==207,'All207 trainable keys retained')
check(up['lora_metadata']['adapter_parameter_count']==3145728,'Rank16 encoder adapters3,145,728 parameters')
check(up['lora_metadata']['non_adapter_encoder_trainable_parameter_count']==0,'Frozen encoder remains frozen')
check(not any(k in up for k in ('optimizer','sampler','history','config_identity','cuda_rng')),'Warmstart cannot masquerade as exact optimizer resume')
reject(lambda:u.upgrade(saved,alpha=16.),'Scaling-changing rank16/alpha16 rejected')
reject(lambda:u.upgrade(saved,rank=4,alpha=8.),'Rank decrease rejected')
bad=copy.deepcopy(saved['lora_metadata']);bad['targets'][0]['dropout']=.1
reject(lambda:u.upgrade({**saved,'lora_metadata':bad}),'Stochastic dropout layout rejected')
records=[];generator=torch.Generator().manual_seed(20261010)
with safe_open(base,framework='pt',device='cpu') as handle:
    keys=set(handle.keys())
    for target in saved['lora_metadata']['targets']:
        name=target['module'];ak=name+'.lora_A';bk=name+'.lora_B'
        a,b=saved['weights'][ak],saved['weights'][bk];na,nb=up['weights'][ak],up['weights'][bk]
        check(torch.equal(a,na[:8]) and torch.equal(b,nb[:,:8]),'Old factors copied exactly: '+name)
        check(bool((nb[:,8:]==0).all()) and bool((na[8:]!=0).any()),'Live randomA/zeroB new coordinates: '+name)
        x=torch.randn(2,5,target['in_features'],generator=generator)
        weight=handle.get_tensor(name+'.weight').float()
        bias=handle.get_tensor(name+'.bias').float() if name+'.bias' in keys else None
        old=F.linear(x,weight,bias)+2*F.linear(F.linear(x,a),b)
        new=F.linear(x,weight,bias)+2*F.linear(F.linear(x,na),nb)
        fp32=error(old,new)
        check(fp32['relative_l2']<1e-6 and fp32['max_absolute_delta']<1e-5,'FP32 actual projection function tolerance: '+name)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            old_bf=F.linear(x,weight,bias)+2*F.linear(F.linear(x,a),b)
            new_bf=F.linear(x,weight,bias)+2*F.linear(F.linear(x,na),nb)
        bf16=error(old_bf,new_bf)
        check(bf16['relative_l2']<1e-4 and bf16['max_absolute_delta']<.02,'BF16 CPU actual projection function tolerance: '+name)
        # Additional A starts gradient-free at B=0; additional B must receive
        # learning signal immediately, avoiding a dead both-zero expansion.
        grad_a=na.clone().requires_grad_(True);grad_b=nb.clone().requires_grad_(True)
        update=2*F.linear(F.linear(x,grad_a),grad_b)
        update.square().mean().backward()
        check(float(grad_b.grad[:,8:].norm())>0.,'NewB learns on first step: '+name)
        check(bool((grad_a.grad[8:]==0).all()),'NewA initially zero gradient at zeroB as expected: '+name)
        records.append({'module':name,'input_shape':list(x.shape),'base_weight_shape':list(weight.shape),
                        'fp32_output':fp32,'cpu_bf16_autocast_output':bf16,
                        'new_B_gradient_norm':float(grad_b.grad[:,8:].norm())})
for key,value in saved['weights'].items():
    if not key.endswith(('.lora_A','.lora_B')):
        check(torch.equal(value,up['weights'][key]),'Non-adapter tensor untouched: '+key)
check(u.sha(source)==before,'Immutable parent checkpoint unchanged')
check(torch.cuda.is_initialized() is False,'No CUDA initialized')
report={'schema':1,'kind':'rank8_to16_actual_tensor_CPU_function_verification_v1','passed':True,
        'source_path':str(source),'source_sha256':before,'parent_run_id':saved['run_id'],'parent_step':saved['step'],
        'base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','base_path':str(base),'base_sha256':u.sha(base),
        'implementation_path':str(Path(u.__file__)),'implementation_sha256':u.sha(u.__file__),
        'test_path':str(Path(__file__)),'test_sha256':u.sha(__file__),'torch_version':str(torch.__version__),
        'cpu_threads':1,'mkldnn_enabled':torch.backends.mkldnn.enabled,'GPU_initialized':False,
        'rank8_alpha16_scaling2_to_rank16_alpha32_scaling2':True,'adapter_targets':len(records),
        'non_adapter_tensor_count':sum(not k.endswith(('.lora_A','.lora_B')) for k in saved['weights']),
        'positive_assertions':len(checks),'rejected_corruptions':len(rejections),
        'fp32_limits':{'relative_l2':1e-6,'max_absolute_delta':1e-5},
        'cpu_bf16_limits':{'relative_l2':1e-4,'max_absolute_delta':.02},
        'fp32_max_relative_l2':max(r['fp32_output']['relative_l2'] for r in records),
        'fp32_max_absolute_delta':max(r['fp32_output']['max_absolute_delta'] for r in records),
        'cpu_bf16_max_relative_l2':max(r['cpu_bf16_autocast_output']['relative_l2'] for r in records),
        'cpu_bf16_max_absolute_delta':max(r['cpu_bf16_autocast_output']['max_absolute_delta'] for r in records),
        'cpu_bf16_bitwise_equal_modules':sum(r['cpu_bf16_autocast_output']['bitwise_equal'] for r in records),
        'limitation':'Actual CPU per-projection checks, not full encoder or CUDA/BF16 equivalence; a matched GPU correctness and DEV experiment is still required.',
        'records':records,'checks':checks,'rejections':rejections}
path=OUTPUT/'rank16_upgrade_actual_cpu_verification_v1.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k]for k in ['passed','adapter_targets','positive_assertions','rejected_corruptions','fp32_max_relative_l2','fp32_max_absolute_delta','cpu_bf16_max_relative_l2','cpu_bf16_max_absolute_delta','cpu_bf16_bitwise_equal_modules']}))
