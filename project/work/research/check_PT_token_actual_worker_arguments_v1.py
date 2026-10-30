"""Actual parser/validation and real archived metadata; native PID proof deferred."""
from pathlib import Path
import argparse,ast,hashlib,json,math,sys,copy
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True);cli=p.parse_args()
import tsc_uniform_pttoken_constraint_transition_v1 as boundary
source=HERE/'train_lora_batched_tsc_uniform_pttoken_v1.py';tree=ast.parse(source.read_text(encoding='utf-8'))
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('parser','validate_arguments')]
native_calls=[]
ns={'argparse':argparse,'Path':Path,'DATA':ROOT/'work/data','OUTPUT':HERE,'math':math,'__doc__':'actual extracted worker CLI',
    'native_loss_identity':lambda args:native_calls.append('future same-PID runtime proof is tested at actual dispatch'),
    'majority_preflight':boundary.preflight}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
plan=json.loads(cli.preflight.read_text(encoding='utf-8'));args=ns['parser']().parse_args(plan['worker_arguments']);ns['validate_arguments'](args)
assert args.portuguese_token_kl_multiplier==5 and args.feature_distillation==20 and args.replay_kl==40 and args.duration_kl==2
rejections=[]
for key,value in [('portuguese_token_kl_multiplier',1.),('portuguese_token_kl_multiplier',math.inf),('feature_distillation',4.),('replay_kl',200.),('lora_lr',3e-4)]:
    altered=copy.copy(args);setattr(altered,key,value)
    try:ns['validate_arguments'](altered)
    except ValueError:rejections.append([key,str(value)])
    else:raise AssertionError(('Accepted undeclared change',key,value))
report={'kind':'PT_token5x_actual_worker_parser_metadata_validation_v1','passed':True,
    'actual_parser_and_validate_arguments_extracted_from_source':True,'accepted_actual_plan':True,'rejections':rejections,
    'native_runtime_PID_only_deferred_to_actual_dispatch':True,'real_source_metadata_preflight':True,
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preflight_sha256':hashlib.sha256(cli.preflight.read_bytes()).hexdigest(),
    'Torch_imported':'torch' in sys.modules,'GPU_launched':False,'checkpoint_tensor_bytes_read':False}
assert 'torch' not in sys.modules
with cli.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({'passed':True,'accepted_real_plan':True,'rejections':len(rejections),'Torch_imported':False}))
