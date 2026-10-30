"""Actual tiny CPU autograd check of extracted per-language wrapper; no model."""
from pathlib import Path
from types import SimpleNamespace
import ast,argparse,hashlib,json
import torch
ROOT=Path(__file__).resolve().parents[2];source=ROOT/'outputs/parakeet_turkish/train_lora_batched_tsc_uniform_constraint_v1.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('preservation_row_scales','pt_focused_batch_objective')]
from dataclasses import dataclass
@dataclass
class Objective:
    total:object
    per_clip:dict
rows=[{'language':'tr'} for _ in range(6)]+[{'language':'en'},{'language':'pt'}]
values={key:torch.linspace(.1,.8,8,requires_grad=True) for key in ('tdt','token_kl','duration_kl','feature_distillation')}
calls=[]
def frozen(*args,**kwargs):
    calls.append(kwargs)
    return Objective(values['tdt'].mean(),{**values,'total':values['tdt']})
ns={'torch':torch,'batch_objective':frozen}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
result=ns['pt_focused_batch_objective'](None,None,SimpleNamespace(rows=rows),replay_kl=40.,duration_kl=2.,feature_distillation=20.,old_size=8193)
loss=result.total/2
loss.backward()
expected_scale=torch.tensor([0.]*6+[5/3,1.])
checks={
    'one_frozen_native_forward_with_zero_scalar_preservation':len(calls)==1 and all(calls[0][key]==0 for key in ('replay_kl','duration_kl','feature_distillation')),
    'TR_preservation_literal_zero':all(torch.equal(values[key].grad[:6],torch.zeros(6)) for key in ('token_kl','duration_kl','feature_distillation')),
    'TDT_mean16_coefficient_unchanged':torch.allclose(values['tdt'].grad,torch.full((8,),1/16)),
    'token_KL_differentiable_PT40_oldfive66_6667':torch.allclose(values['token_kl'].grad,40*expected_scale/16),
    'duration_KL_differentiable_PT2_oldfive3_3333':torch.allclose(values['duration_kl'].grad,2*expected_scale/16),
    'features_differentiable_PT20_oldfive33_3333':torch.allclose(values['feature_distillation'].grad,20*expected_scale/16),
    'no_GPU_initialized':torch.cuda.is_initialized() is False,
    'all_old_five_use_identical_compensation':ns['preservation_row_scales']([{'language':language} for language in ('en','es','it','fr','de')])==[5/3]*5,
}
expected=(values['tdt'].detach()+expected_scale*(40*values['token_kl'].detach()+2*values['duration_kl'].detach()+20*values['feature_distillation'].detach())).mean()/2
checks['exact_mean16_weighted_forward']=torch.allclose(loss.detach(),expected)
try:ns['preservation_row_scales']([{'language':'unknown'}])
except ValueError:checks['reject_unknown_language']=True
if not all(checks.values()):raise AssertionError(checks)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
report={'kind':'feature5x_preservation_actual_tiny_CPU_autograd_v1','passed':True,'checks':checks,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'scope':'Actual Torch CPU autograd on8 explicit synthetic per-clip scalar inputs; no ASR/native kernel/model/audio/checkpoint read or CUDA initialization',
    'new_PT_expected_token_KL_mean16_density':40/16,'prior_PT_uniform_six_density':(2/6)*40/16,
    'each_old_five_expected_density_preserved':(1/5)*(40*5/3)/16,'prior_each_old_five_density':(2/6)*40/16,
    'equal_gradients_or_actual_retention_claimed':False}
with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'GPU_initialized':torch.cuda.is_initialized(),'output':str(args.output)}))
